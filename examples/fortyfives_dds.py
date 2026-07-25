#!/usr/bin/env python3
"""
Exact double-dummy solver (DDS) for the Fortyfives play phase.

Solves a deal with all four hands face up: exact alpha-beta minimax over
the remaining play-out, replicating THIS engine's rules bit-for-bit
(legality incl. the trump renege rule, trick winner, running
highest-trump bonus). See RESEARCH.md § Active work and memory
`project-dds-handoff` for why this exists and how it is validated.

Three consumers:
  1. Play oracle (OracleAgent): upper bound for the play_eval yardstick.
     Two opponent models:
       - 'rulebased' (the DECISION-GATE ruler): EW nodes are FORCED to
         the move the real RuleBasedAgent._play_strategy would make.
         This is the true ceiling of avg_diff vs the fixed rule-based
         eval table. (Paranoid minimax is NOT an upper bound there —
         it may forgo exploiting rule-based mistakes.)
       - 'minimax': EW plays adversarially (classic double-dummy).
  2. PIMC-DDS agent: per-world exact solve inside PIMC determinization
     (opponent='minimax' — the world's holder is known, the policy isn't).
  3. (later) bidding oracle.

Payoff at the leaves is the ACTUAL yardstick quantity by default
(payoff='delta'): the NS game-point delta after `game.end_hand`'s bid
adjustments — failed bid => flat -bid_value, made 30-bid => flat 60,
otherwise raw points (5/trick + 5 for playing the high trump). The 100+
pegging restriction is unreachable in eval (every hand starts 0-0) and
is asserted against, not modelled. payoff='raw' (our-opp raw points)
kept for parity with PIMCAgent._simulate ablations.

Faithfulness invariants replicated from the engine (verified in
tests/test_dds_vs_engine.py against the REAL engine — do not edit one
side without re-running that gate):
  - A-hearts is ALWAYS trump (game.py process_play / wins_over).
  - Legality = game.get_legal_plays (game.py L432, non-test-mode
    branch): must follow lead if able but trump is always legal; when
    trump is led, must follow with trump EXCEPT that a top-3 trump
    (rank >= 1001: 5/J/A-hearts) outranking the LED trump may be
    withheld; if every trump in hand is withholdable, any card goes.
    "Trump led" means the led CARD's suit == trump suit (an A-hearts
    lead under a non-hearts trump is a HEARTS lead, per the engine).
  - Trick winner: highest-rank trump played, else highest-rank card of
    the lead suit (unique ranks within each pool; game.wins_over).
  - High-trump bonus goes to the team that PLAYED the highest-ranked
    trump this hand, only if some trump was played (game.py L742-746).
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

from fortyfives.games.fortyfives.card import (
    SUITS, RANKS, FortyfivesCard, get_card_rank,
)

sys.path.insert(0, os.path.dirname(__file__))
from fortyfives_rule_based import RuleBasedAgent

# ---------------------------------------------------------------------------
# Precomputed tables, card-id space (id = suit_idx*13 + rank_idx, the same
# encoding as FortyfivesCard.id). Built by calling the REAL get_card_rank so
# ranking can never drift from the engine.
# ---------------------------------------------------------------------------

_CARD = [FortyfivesCard(i) for i in range(52)]
_SUIT_OF = [i // 13 for i in range(52)]
_AH = 13 * SUITS.index('H') + RANKS.index('A')  # A-hearts card id (25)

# _RANK[trump_idx][card_id], _ISTRUMP[trump_idx][card_id]
_RANK = [[get_card_rank(_CARD[c], SUITS[t]) for c in range(52)]
         for t in range(4)]
_ISTRUMP = [[(_SUIT_OF[c] == t or c == _AH) for c in range(52)]
            for t in range(4)]

_EMPTY_TRICK = (-1, -1, -1, -1)
_BID_VALUE = {1: 20, 2: 25, 3: 30}   # BID_20/25/30 action ids -> value

_INF = float('inf')


def card_id(card):
    """FortyfivesCard -> int id (identity on .id, kept for clarity)."""
    return card.id


def legal_plays(hand, lead_card, trump):
    """Exact replica of game.get_legal_plays (normal mode) in id space.

    hand: ordered tuple of card ids (engine hand order — order matters
    downstream for the forced rule-based policy). lead_card: id of the
    led card, or None when leading. Returns a tuple of HAND INDICES,
    ascending (engine returns ascending too; consumers are
    order-insensitive).
    """
    n = len(hand)
    if lead_card is None or n == 0:
        return tuple(range(n))

    rank_t = _RANK[trump]
    istr_t = _ISTRUMP[trump]
    lead_suit = _SUIT_OF[lead_card]
    trump_idx = tuple(i for i in range(n) if istr_t[hand[i]])

    if lead_suit == trump:
        # Trump led: must follow with trump unless every trump in hand
        # is a withholdable top-3 trump outranking the led card.
        if not trump_idx:
            return tuple(range(n))
        led_rank = rank_t[lead_card]
        obligated = [i for i in trump_idx
                     if not (rank_t[hand[i]] >= 1001 and
                             rank_t[hand[i]] > led_rank)]
        if obligated:
            return trump_idx
        return tuple(range(n))

    lead_idx = [i for i in range(n) if _SUIT_OF[hand[i]] == lead_suit]
    if lead_idx:
        return tuple(sorted(set(lead_idx) | set(trump_idx)))
    return tuple(range(n))


def trick_winner(trick, leader, trump):
    """Winning seat of a completed 4-card trick (id-space replica of
    game.get_trick_winner/wins_over). trick: 4-tuple of card ids."""
    rank_t = _RANK[trump]
    istr_t = _ISTRUMP[trump]
    best_seat, best_rank, any_trump = leader, -1, False
    lead_suit = _SUIT_OF[trick[leader]]
    for s in range(4):
        c = trick[s]
        if c < 0:
            continue
        if istr_t[c]:
            if not any_trump:
                any_trump, best_seat, best_rank = True, s, rank_t[c]
            elif rank_t[c] > best_rank:
                best_seat, best_rank = s, rank_t[c]
        elif not any_trump and _SUIT_OF[c] == lead_suit:
            if rank_t[c] > best_rank:
                best_seat, best_rank = s, rank_t[c]
    return best_seat


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class DDSolver:
    """Exact solver for one hand-context (trump + bid). NS seats (0/2)
    maximize the payoff, EW seats (1/3) minimize it — or, with
    opponent='rulebased', EW moves are forced to the real
    RuleBasedAgent._play_strategy choice (deterministic best response).

    Payoff (payoff='delta', default): final NS game-point delta per
    end_hand — the play_eval yardstick. payoff='raw': ns_raw - ew_raw.
    """

    def __init__(self, trump, bid_team, bid_kind,
                 opponent='minimax', payoff='delta'):
        assert trump in (0, 1, 2, 3)
        assert bid_team in (0, 1)
        assert bid_kind in _BID_VALUE, (
            'highest_bid must be a real level (holds resolve to one)')
        assert opponent in ('minimax', 'rulebased')
        assert payoff in ('delta', 'raw')
        self._trump = trump
        self._bid_team = bid_team
        self._bid_kind = bid_kind
        self._bid_value = _BID_VALUE[bid_kind]
        self._payoff = payoff
        self._forced = (opponent == 'rulebased')
        self._rb = RuleBasedAgent(18) if self._forced else None
        self._trump_str = SUITS[trump]
        self._tt = {}
        self._total_tricks = None
        self.rb_fallbacks = 0  # defensive-fallback count; must stay 0

    # -- public API ---------------------------------------------------------

    def solve(self, hands, leader, trick=_EMPTY_TRICK,
              ns_tricks=0, ew_tricks=0, best_rank=-1, best_par=-1):
        """Exact value of the position for the payoff/opponent model.

        hands: 4-tuple of ordered tuples of card ids (ENGINE hand order).
        leader: seat that led the current trick (= seat to move if the
        trick is empty). trick: 4-tuple of card ids, -1 = not yet played.
        ns_tricks/ew_tricks: tricks already resolved this hand.
        best_rank/best_par: rank and team parity (0=NS, 1=EW) of the
        highest trump played so far, -1/-1 if none.
        """
        self._prepare(hands, trick, ns_tricks, ew_tricks)
        return self._search(hands, trick, leader, ns_tricks,
                            best_rank, best_par, -_INF, _INF)

    def root_values(self, hands, leader, trick=_EMPTY_TRICK,
                    ns_tricks=0, ew_tricks=0, best_rank=-1, best_par=-1):
        """{hand_index: exact value} for every legal move of the seat to
        move (full-window child solves — needed for PIMC averaging and
        for deterministic argmax with margins)."""
        self._prepare(hands, trick, ns_tricks, ew_tricks)
        played = sum(1 for c in trick if c >= 0)
        seat = (leader + played) % 4
        lead_card = trick[leader] if played else None
        legal = legal_plays(hands[seat], lead_card, self._trump)
        if self._forced and seat % 2 == 1:
            # Forced seat: the position has exactly one continuation.
            legal = (self._rb_choice(hands[seat], trick, legal),)
        out = {}
        for i in legal:
            out[i] = self._child_value(hands, trick, leader, seat, i,
                                       ns_tricks, best_rank, best_par,
                                       -_INF, _INF)
        return out

    def best_move(self, *args, **kwargs):
        """Argmax (NS) / argmin (EW) hand index, lowest index on ties
        (deterministic — the paired-eval benchmark requirement)."""
        vals = self.root_values(*args, **kwargs)
        hands, leader = args[0], args[1]
        trick = kwargs.get('trick', args[2] if len(args) > 2 else _EMPTY_TRICK)
        played = sum(1 for c in trick if c >= 0)
        seat = (leader + played) % 4
        sign = 1 if seat % 2 == 0 else -1
        return max(sorted(vals), key=lambda i: sign * vals[i])

    # -- internals ----------------------------------------------------------

    def _prepare(self, hands, trick, ns_tricks, ew_tricks):
        played = sum(1 for c in trick if c >= 0)
        remaining = (sum(len(h) for h in hands) + played) // 4
        total = ns_tricks + ew_tricks + remaining
        if self._total_tricks is None:
            self._total_tricks = total
        elif self._total_tricks != total:
            # Different hand context — stale TT would be wrong.
            self._tt = {}
            self._total_tricks = total

    def _leaf(self, ns_tricks, best_par):
        ns_raw = 5 * ns_tricks + (5 if best_par == 0 else 0)
        if self._payoff == 'raw':
            ew_raw = (5 * (self._total_tricks - ns_tricks)
                      + (5 if best_par == 1 else 0))
            return ns_raw - ew_raw
        if self._bid_team == 0:
            if ns_raw >= self._bid_value:
                return 60 if self._bid_kind == 3 else ns_raw
            return -self._bid_value
        # EW bid: NS banks its raw points either way (no 100+ rule here).
        return ns_raw

    def _rb_choice(self, hand, trick, legal):
        """The move the REAL RuleBasedAgent makes from this state.
        Inputs reconstructed exactly as the env presents them: ordered
        hand of Card objects, seat-indexed current_trick with Nones,
        trump as a suit string, legal actions as game hand indices."""
        raw_obs = {
            'hand': [_CARD[c] for c in hand],
            'current_trick': [(_CARD[trick[s]] if trick[s] >= 0 else None)
                              for s in range(4)],
            'trump_suit': self._trump_str,
        }
        choice = self._rb._play_strategy(raw_obs, list(legal))
        if choice not in legal:      # never expected; counted, not hidden
            self.rb_fallbacks += 1
            return min(legal)
        return choice

    def _child_value(self, hands, trick, leader, seat, i,
                     ns_tricks, best_rank, best_par, alpha, beta):
        """Value after seat plays hand index i. Shared by search and
        root_values so move semantics can never diverge."""
        hand = hands[seat]
        card = hand[i]
        nh = list(hands)
        nh[seat] = hand[:i] + hand[i + 1:]
        nh = tuple(nh)
        ntrick = trick[:seat] + (card,) + trick[seat + 1:]
        nbr, nbp = best_rank, best_par
        if _ISTRUMP[self._trump][card]:
            r = _RANK[self._trump][card]
            if r > nbr:
                nbr, nbp = r, seat % 2
        if sum(1 for c in ntrick if c >= 0) == 4:
            w = trick_winner(ntrick, leader, self._trump)
            nns = ns_tricks + (1 if w % 2 == 0 else 0)
            return self._search(nh, _EMPTY_TRICK, w, nns,
                                nbr, nbp, alpha, beta)
        return self._search(nh, ntrick, leader, ns_tricks,
                            nbr, nbp, alpha, beta)

    def _search(self, hands, trick, leader, ns_tricks,
                best_rank, best_par, alpha, beta):
        if not any(hands):
            return self._leaf(ns_tricks, best_par)

        key = (hands, trick, leader, ns_tricks, best_rank, best_par)
        entry = self._tt.get(key)
        if entry is not None:
            val, flag = entry
            if flag == 0:                       # exact
                return val
            if flag == 1:                       # lower bound
                if val >= beta:
                    return val
                if val > alpha:
                    alpha = val
            else:                               # upper bound
                if val <= alpha:
                    return val
                if val < beta:
                    beta = val

        played = sum(1 for c in trick if c >= 0)
        seat = (leader + played) % 4
        lead_card = trick[leader] if played else None
        legal = legal_plays(hands[seat], lead_card, self._trump)

        if self._forced and seat % 2 == 1:
            order = (self._rb_choice(hands[seat], trick, legal),)
        else:
            rank_t = _RANK[self._trump]
            hand = hands[seat]
            order = sorted(legal, key=lambda i: -rank_t[hand[i]])

        maximizing = (seat % 2 == 0)
        a0, b0 = alpha, beta
        best = -_INF if maximizing else _INF
        for i in order:
            v = self._child_value(hands, trick, leader, seat, i,
                                  ns_tricks, best_rank, best_par,
                                  alpha, beta)
            if maximizing:
                if v > best:
                    best = v
                if best > alpha:
                    alpha = best
            else:
                if v < best:
                    best = v
                if best < beta:
                    beta = best
            if alpha >= beta:
                break

        if best <= a0:
            self._tt[key] = (best, 2)           # upper bound
        elif best >= b0:
            self._tt[key] = (best, 1)           # lower bound
        else:
            self._tt[key] = (best, 0)           # exact
        return best


# ---------------------------------------------------------------------------
# Oracle agent — drop-in play agent for play_eval, reads TRUE hands from
# env.game (it cheats by charter: it is the ruler, not a fair agent).
# ---------------------------------------------------------------------------

class OracleAgent:
    """Perfect-information play oracle for play_eval.

    opponent='rulebased' (default): exact best response to the real
    rule-based EW — the true ceiling of the yardstick. 'minimax':
    classic double-dummy (paranoid) oracle.

    Must be constructed with the SAME env object play_eval drives, since
    step(state) ignores the (hidden-information) observation and reads
    env.game directly.
    """

    def __init__(self, env, opponent='rulebased', payoff='delta',
                 num_actions=18):
        self.env = env
        self.opponent = opponent
        self.payoff = payoff
        self.num_actions = num_actions
        self.use_raw = True
        self.rb_fallbacks = 0

    def step(self, state):
        g = self.env.game
        raw_legal = list(state.get('raw_legal_actions') or [])
        if g.phase != 4 or not raw_legal:
            env_legal = list(state['legal_actions'].keys())
            return min(env_legal) if env_legal else 0

        # Eval hands always start 0-0, so the 100+ pegging restriction
        # (which the leaf payoff does not model) can never apply.
        assert g.points.get(0, 0) < 100 and g.points.get(1, 0) < 100, (
            'OracleAgent payoff model assumes scores below 100')

        trump = SUITS.index(g.trump_suit)
        hands = tuple(tuple(c.id for c in g.hands[s]) for s in range(4))
        trick = tuple((g.current_trick[s].id if g.current_trick[s] is not None
                       else -1) for s in range(4))
        leader = g.trick_starter
        ns_tr = g.tricks_won[0] + g.tricks_won[2]
        ew_tr = g.tricks_won[1] + g.tricks_won[3]
        if g.highest_trump_played is not None:
            best_rank = get_card_rank(g.highest_trump_played, g.trump_suit)
            best_par = g.highest_trump_player % 2
        else:
            best_rank, best_par = -1, -1

        solver = DDSolver(trump, g.highest_bidder % 2, g.highest_bid,
                          opponent=self.opponent, payoff=self.payoff)
        vals = solver.root_values(hands, leader, trick,
                                  ns_tr, ew_tr, best_rank, best_par)
        self.rb_fallbacks += solver.rb_fallbacks

        seat = g.current_player_id
        sign = 1 if seat % 2 == 0 else -1
        best = max(sorted(vals), key=lambda i: sign * vals[i])
        assert best in raw_legal
        return best + 9   # game hand index -> env play action id

    def eval_step(self, state):
        return self.step(state), {}
