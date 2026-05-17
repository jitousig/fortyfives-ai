import os
import random
import sys

import rlcard
from rlcard.envs.registration import register, registry

from fortyfives.games.fortyfives.game import (
    FortyfivesGame,
    PHASE_AUCTION, PHASE_DECLARATION, PHASE_DISCARD, PHASE_GAMEPLAY,
    BID_VALUES, BID_SUCCESS_VALUES,
    BID_PASS, BID_20, BID_25, BID_30, BID_HOLD, DISCARD_DONE,
)

# ── SOTA opponent (composite, wired exactly like play_eval._run_hand) ──
# Phases 1/2/3 (bid/declare/discard) -> RuleBasedAgent
# Phase  4     (card play)           -> PIMCAgent  (v3: constrained +
#                                       rule-based rollout, defaults)
# Both agents are use_raw=True, consume the rlcard env's extracted state
# and return ENV action ids fed straight to env.step(). The agents and
# this branch's fortyfives engine come from agent-sota (commit 8345df7).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXAMPLES = os.path.join(_REPO_ROOT, "examples")
for _p in (_REPO_ROOT, _EXAMPLES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fortyfives_rule_based import RuleBasedAgent  # noqa: E402
from fortyfives_pimc import PIMCAgent  # noqa: E402

if 'fortyfives' not in registry.env_specs:
    register(
        env_id='fortyfives',
        entry_point='fortyfives.envs.fortyfives_env:FortyfivesEnv',
    )

PLAYER_NAMES = ["You (South)", "West", "North", "East"]
SUIT_SYMBOLS = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}
PHASE_NAMES = {
    PHASE_AUCTION: 'Bidding',
    PHASE_DECLARATION: 'Trump Declaration',
    PHASE_DISCARD: 'Discard',
    PHASE_GAMEPLAY: 'Play',
}


def card_to_str(card):
    if card is None:
        return None
    return f"{card.rank}{card.suit}"


def format_card(card):
    if card is None:
        return "?"
    rank = '10' if card.rank == 'T' else card.rank
    return f"{rank}{SUIT_SYMBOLS.get(card.suit, card.suit)}"


def _bids_to_dict(bids, num_players):
    """Normalize bids (dict or list) to a str-keyed dict."""
    result = {}
    if bids is None:
        return {str(i): None for i in range(num_players)}
    if isinstance(bids, dict):
        for k, v in bids.items():
            result[str(k)] = v
    else:
        for i, v in enumerate(bids):
            result[str(i)] = v
    return result


class GameSession:
    def __init__(self, human_player=0):
        self.env = rlcard.make('fortyfives')
        self._state, self._pid = self.env.reset()
        # serialize_state and server.py read the raw game off .game; the
        # env wraps the same FortyfivesGame instance.
        self.game = self.env.game
        self.human_player = human_player
        self.log = []
        self.game_over = False
        self.hand_over = False
        self.last_hand_summary = None
        # Rich per-hand breakdown for the hand-summary popup. tricks /
        # high-trump / bid_made are reset by FortyfivesGame.end_hand(),
        # so capture them at that boundary (instance wrapper — keeps the
        # rlcard env's own game instance; main's pause/dismiss flow is
        # untouched, only the popup payload is enriched).
        self._last_hand_result = None
        _orig_end_hand = self.game.end_hand

        def _capturing_end_hand():
            g = self.game
            pre_ns = g.points.get(0, 0) if isinstance(g.points, dict) else 0
            pre_ew = g.points.get(1, 0) if isinstance(g.points, dict) else 0
            tricks = list(g.tricks_won) if g.tricks_won else [0, 0, 0, 0]
            ht_player = g.highest_trump_player
            ht_card = (str(g.highest_trump_played)
                       if g.highest_trump_played else None)
            _orig_end_hand()  # scores, updates points, resets trump fields
            bid_team_idx = (g.highest_bidder % 2
                            if g.highest_bidder is not None else None)
            self._last_hand_result = {
                "bid_team": ("ns" if bid_team_idx == 0
                             else "ew" if bid_team_idx == 1 else None),
                "bid_value": BID_VALUES.get(g.highest_bid),
                "bid_made": bool(getattr(g, "bid_made", False)),
                "bid_success_val": BID_SUCCESS_VALUES.get(g.highest_bid),
                "ns_tricks": tricks[0] + tricks[2],
                "ew_tricks": tricks[1] + tricks[3],
                "ns_raw": int(g.hand_points[0]) if g.hand_points else 0,
                "ew_raw": int(g.hand_points[1]) if g.hand_points else 0,
                "high_trump_team": ("ns" if (ht_player is not None
                                             and ht_player % 2 == 0)
                                    else "ew" if ht_player is not None
                                    else None),
                "high_trump_card": ht_card,
                "trump_suit": g.trump_suit,
            }

        self.game.end_hand = _capturing_end_hand

        # Composite SOTA opponent. PIMC n_worlds=10: documented as
        # statistically indistinguishable from 20 and ~2x faster — picked
        # for interactive latency. PIMCAgent() defaults = v3 (constrained,
        # rule-based rollout); do not override.
        self._rule_agent = RuleBasedAgent(num_actions=18)
        self._pimc_agent = PIMCAgent(num_actions=18, n_worlds=10)

        self.log.append("Game started! You are South (partner: North).")
        self._log_phase()

    def _agent_for_phase(self, phase):
        return self._pimc_agent if phase == PHASE_GAMEPLAY else self._rule_agent

    def _log_phase(self):
        game = self.game
        phase = game.phase
        if phase == PHASE_AUCTION:
            dealer = PLAYER_NAMES[game.dealer_id]
            self.log.append(f"New hand — {dealer} deals. Bidding begins.")
        elif phase == PHASE_DECLARATION:
            bidder = PLAYER_NAMES[game.highest_bidder]
            val = BID_VALUES.get(game.highest_bid, '?')
            self.log.append(f"{bidder} won bid ({val}). Select trump suit.")
        elif phase == PHASE_DISCARD:
            self.log.append("Discard — remove unwanted cards, then click Done.")
        elif phase == PHASE_GAMEPLAY:
            trump = SUIT_SYMBOLS.get(game.trump_suit, '?')
            self.log.append(f"Play begins. Trump: {trump}")

    def serialize_state(self):
        game = self.game

        def _hand(i):
            h = game.hands
            if isinstance(h, dict):
                return h.get(i, [])
            return h[i] if i < len(h) else []

        hand = [card_to_str(c) for c in _hand(self.human_player)]

        current_trick = [None] * game.num_players
        if game.current_trick:
            for i in range(game.num_players):
                c = game.current_trick[i]
                if c is not None:
                    current_trick[i] = card_to_str(c)

        trick_history = []
        for trick in (game.trick_history or []):
            trick_history.append([card_to_str(c) for c in trick])

        pts = game.points or {}
        ns_points = pts.get(0, 0) if isinstance(pts, dict) else 0
        ew_points = pts.get(1, 0) if isinstance(pts, dict) else 0

        legal_actions = []
        if not self.game_over and game.current_player_id == self.human_player:
            legal_actions = game.get_legal_actions()

        bids = _bids_to_dict(game.bids, game.num_players)
        hand_counts = {str(i): len(_hand(i)) for i in range(game.num_players)}

        return {
            "type": "state",
            "phase": game.phase,
            "phase_name": PHASE_NAMES.get(game.phase, "Unknown"),
            "human_player": self.human_player,
            "current_player": game.current_player_id,
            "is_human_turn": not self.game_over and game.current_player_id == self.human_player,
            "hand": hand,
            "legal_actions": legal_actions,
            "current_trick": current_trick,
            "trick_lead_suit": game.trick_lead_suit,
            "trump_suit": game.trump_suit,
            "trump_display": SUIT_SYMBOLS.get(game.trump_suit) if game.trump_suit else None,
            "bids": bids,
            "passed": (game.passed or [False] * game.num_players),
            "highest_bid": game.highest_bid,
            "highest_bid_value": BID_VALUES.get(game.highest_bid) if game.highest_bid is not None else None,
            "highest_bidder": game.highest_bidder,
            "dealer": game.dealer_id,
            "tricks_won": (game.tricks_won or [0, 0, 0, 0]),
            "trick_count": (game.trick_count or 0),
            "trick_history": trick_history,
            "trick_winners": (game.trick_winners or []),
            "points": {"ns": ns_points, "ew": ew_points},
            "hand_counts": hand_counts,
            "game_over": self.game_over,
            "hand_over": self.hand_over and not self.game_over,
            "hand_summary": self.last_hand_summary,
            "log": self.log[-40:],
        }

    def run_ai_turn(self):
        """Run one AI turn. Returns True if more AI turns are needed."""
        game = self.game
        if game.current_player_id == self.human_player:
            return False
        if self.game_over:
            return False

        pre_phase = game.phase
        pre_points = dict(game.points) if isinstance(game.points, dict) else {}

        legal = game.get_legal_actions()
        if not legal:
            return False

        agent = self._agent_for_phase(pre_phase)
        env_action = agent.step(self._state)
        raw_action = self.env._decode_action(env_action)

        # Defensive: a correctly-wired agent always returns a legal env id.
        # If decode lands outside the legal set, surface it and fall back
        # rather than crash the live game.
        if raw_action not in legal:
            print(
                f"[game_session] agent returned illegal action "
                f"env={env_action} raw={raw_action} phase={pre_phase} "
                f"legal={legal}; falling back to random",
                file=sys.stderr,
            )
            raw_action = random.choice(legal)
            env_action = self.env._game_to_env_action(raw_action, pre_phase)

        player_id = game.current_player_id
        action_desc = self._describe_action(raw_action, game.phase, player_id)
        name = PLAYER_NAMES[player_id]

        self._state, self._pid = self.env.step(env_action)
        self.log.append(f"{name}: {action_desc}")

        post_phase = game.phase
        if pre_phase != post_phase:
            self._log_phase()

        if pre_phase == PHASE_GAMEPLAY and post_phase == PHASE_AUCTION:
            ns = game.points.get(0, 0)
            ew = game.points.get(1, 0)
            d_ns = ns - pre_points.get(0, 0)
            d_ew = ew - pre_points.get(1, 0)
            self.log.append(f"Hand over — NS: {ns} ({d_ns:+d}), EW: {ew} ({d_ew:+d})")
            self.hand_over = True
            self.last_hand_summary = self._compose_hand_summary(
                ns, ew, d_ns, d_ew)

        if game.check_game_over():
            ns = game.points.get(0, 0)
            ew = game.points.get(1, 0)
            winner = "NS (You & North)" if ns >= 125 else "EW (West & East)"
            self.log.append(f"=== GAME OVER === {winner} wins! NS: {ns}, EW: {ew}")
            self.game_over = True
            self.hand_over = False  # game-over overlay takes precedence
            return False

        if self.hand_over:
            return False  # pause for the hand-summary popup

        return game.current_player_id != self.human_player

    def _compose_hand_summary(self, ns, ew, d_ns, d_ew):
        """Main's score line ({ns,ew,d_ns,d_ew}) plus the rich
        per-hand breakdown captured at the end_hand() boundary. The
        legacy keys are kept so any older client still works; the
        frontend renders the detailed view when the rich keys exist."""
        summary = {"ns": ns, "ew": ew, "d_ns": d_ns, "d_ew": d_ew}
        if self._last_hand_result:
            summary.update(self._last_hand_result)
        self._last_hand_result = None
        return summary

    def continue_after_hand(self):
        """Dismiss the hand-summary popup and resume play."""
        self.hand_over = False
        self.last_hand_summary = None
        self._last_hand_result = None

    def take_human_action(self, action):
        """Process human action. Returns error dict or None on success."""
        game = self.game
        if self.game_over:
            return {"error": "Game is over"}
        if game.current_player_id != self.human_player:
            return {"error": "Not your turn"}

        legal = game.get_legal_actions()
        if action not in legal:
            return {"error": f"Illegal action {action}. Legal: {legal}"}

        pre_phase = game.phase
        pre_points = dict(game.points) if isinstance(game.points, dict) else {}
        action_desc = self._describe_action(action, game.phase, self.human_player)

        # Frontend speaks raw game ids; the env expects env ids.
        env_action = self.env._game_to_env_action(action, pre_phase)
        self._state, self._pid = self.env.step(env_action)
        self.log.append(f"You: {action_desc}")

        post_phase = game.phase
        if pre_phase != post_phase:
            self._log_phase()

        if pre_phase == PHASE_GAMEPLAY and post_phase == PHASE_AUCTION:
            ns = game.points.get(0, 0)
            ew = game.points.get(1, 0)
            d_ns = ns - pre_points.get(0, 0)
            d_ew = ew - pre_points.get(1, 0)
            self.log.append(f"Hand over — NS: {ns} ({d_ns:+d}), EW: {ew} ({d_ew:+d})")
            self.hand_over = True
            self.last_hand_summary = self._compose_hand_summary(
                ns, ew, d_ns, d_ew)

        if game.check_game_over():
            ns = game.points.get(0, 0)
            ew = game.points.get(1, 0)
            winner = "NS (You & North)" if ns >= 125 else "EW (West & East)"
            self.log.append(f"=== GAME OVER === {winner} wins! NS: {ns}, EW: {ew}")
            self.game_over = True
            self.hand_over = False  # game-over overlay takes precedence

        return None

    def _describe_action(self, action, phase, player_id):
        if phase == PHASE_AUCTION:
            labels = {BID_PASS: "passes", BID_20: "bids 20", BID_25: "bids 25",
                      BID_30: "bids 30", BID_HOLD: "holds"}
            return labels.get(action, f"bid {action}")

        if phase == PHASE_DECLARATION:
            suits = {0: "♠ Spades", 1: "♥ Hearts", 2: "♦ Diamonds", 3: "♣ Clubs"}
            return f"declares {suits.get(action, action)} as trump"

        if phase == PHASE_DISCARD:
            if action == DISCARD_DONE:
                return "done discarding"
            hand = self._get_hand(player_id)
            if 0 <= action < len(hand):
                return f"discards {format_card(hand[action])}"
            return f"discards card {action}"

        if phase == PHASE_GAMEPLAY:
            hand = self._get_hand(player_id)
            if 0 <= action < len(hand):
                return f"plays {format_card(hand[action])}"
            return f"plays card {action}"

        return f"action {action}"

    def _get_hand(self, player_id):
        h = self.game.hands
        if isinstance(h, dict):
            return h.get(player_id, [])
        return h[player_id] if player_id < len(h) else []

    def reset(self):
        self.__init__(human_player=self.human_player)
