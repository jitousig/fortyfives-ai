#!/usr/bin/env python3
"""
Perfect-Information Monte Carlo (PIMC) play agent for Fortyfives.

Categorically different from the DQN line (no training): at each phase-4
decision it samples N consistent worlds for the hidden hands, plays each
legal card to the end of the hand under a fast heuristic playout, and
picks the card with the best mean (our-team minus opp-team) points under
this engine's scoring (5/trick + 5/high). This is the standard strong
approach for trick-taking card games and sidesteps the rule-based
plateau entirely.

v1 scope / known approximations (measured empirically, not assumed):
  - Determinization is unconstrained (does not yet infer opponent voids
    from earlier "couldn't follow suit" information).
  - The deep playout uses an approximate legality model (must-follow +
    trump-always-legal); the AGENT's own top-level choice uses the
    engine's true legal_actions, so real renege rules are respected
    where it matters most.
v2 lever 1 (constrained determinization) and v3 lever 3 (rule-based
playout) are now the CANONICAL defaults — v3 robustly beats the
competent rule-based on independent seeds. Lever 2 (faithful
renege/follow legality in rollouts) is still pending. constrained /
rollout flags remain for ablation.

Usage: it's a drop-in play agent for play_eval. PIMCAgent() == v3.
    from fortyfives_pimc import PIMCAgent
    from play_eval import evaluate_paired
    evaluate_paired(PIMCAgent(n_worlds=20), num_hands=2000)  # v3
    PIMCAgent(rollout='cheap')                # v2 ablation
    PIMCAgent(rollout='cheap', constrained=False)  # v1 ablation
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from fortyfives.games.fortyfives.card import (
    SUITS, RANKS, FortyfivesCard, get_card_rank,
)

sys.path.insert(0, os.path.dirname(__file__))
from fortyfives_rule_based import RuleBasedAgent  # rollout policy (lever 3)

# All 52 cards, indexable by (rank, suit). FortyfivesCard(card_id) with
# card_id = suit_index*13 + rank_index.
_DECK = [FortyfivesCard(i) for i in range(52)]
_BY_RS = {(c.rank, c.suit): c for c in _DECK}


def _is_trump(card, trump):
    """A♥ is always trump in this game, regardless of declared suit."""
    return card.suit == trump or (card.rank == 'A' and card.suit == 'H')


def _trick_winner(plays, lead_suit, trump):
    """plays: {seat: card} for the 4 cards of a completed trick.
    Trump (incl. A♥) beats all; else only lead-suit cards can win;
    ties impossible (unique cards). Returns the winning seat."""
    trumps = {s: c for s, c in plays.items() if _is_trump(c, trump)}
    pool = trumps if trumps else {
        s: c for s, c in plays.items() if c.suit == lead_suit
    }
    return max(pool, key=lambda s: get_card_rank(pool[s], trump))


def _rollout_legal(hand, lead_suit, trump):
    """Approx legality for the deep playout: must follow the lead suit if
    able, but trump is always allowed (this game permits reneging). When
    leading, anything goes."""
    if lead_suit is None:
        return list(hand)
    follow = [c for c in hand if c.suit == lead_suit]
    if not follow:
        return list(hand)
    trumps = [c for c in hand if _is_trump(c, trump)]
    # follow ∪ trump, de-duplicated, stable order
    seen, legal = set(), []
    for c in follow + trumps:
        key = (c.rank, c.suit)
        if key not in seen:
            seen.add(key)
            legal.append(c)
    return legal


def _cheap_pick(hand, trick, order_cards, lead_suit, trump):
    """v1/v2 fast policy: cheapest card that currently wins the trick;
    else dump the lowest. (order_cards unused; uniform signature.)"""
    legal = _rollout_legal(hand, lead_suit, trump)
    legal.sort(key=lambda c: get_card_rank(c, trump))  # low -> high
    if not trick:
        return legal[0]  # leading: lead the lowest (simple v1)
    best_seat = _trick_winner(trick, lead_suit, trump) if len(trick) else None
    best_rank = get_card_rank(trick[best_seat], trump) if best_seat is not None else -1
    for c in legal:  # cheapest first
        beats = _is_trump(c, trump) or c.suit == lead_suit
        if beats and get_card_rank(c, trump) > best_rank:
            return c
    return legal[0]  # cannot win -> dump lowest


def _rb_pick_factory(rb):
    """Lever 3: use the (fixed, competent) rule-based play strategy as
    the playout policy. _play_strategy infers led suit from the FIRST
    entry of current_trick, so pass cards in PLAY ORDER (leader first),
    NOT seat-indexed — then empty => leading, else first = leader's
    card => correct led suit. Returns a closure with the uniform
    pick signature."""
    def pick(hand, trick, order_cards, lead_suit, trump):
        legal = _rollout_legal(hand, lead_suit, trump)
        legal_keys = {(c.rank, c.suit) for c in legal}
        legal_idx = {i for i, c in enumerate(hand)
                     if (c.rank, c.suit) in legal_keys}
        raw_obs = {'hand': hand,
                   'current_trick': list(order_cards),
                   'trump_suit': trump}
        try:
            choice = rb._play_strategy(raw_obs, legal_idx)
        except Exception:
            choice = None
        if choice is None or not (0 <= choice < len(hand)) or choice not in legal_idx:
            legal.sort(key=lambda c: get_card_rank(c, trump))
            return legal[0]
        return hand[choice]
    return pick


def _simulate(hands, leader, trump, our_parity, pick,
              partial=None, partial_lead=None):
    """Play out the rest of the hand with playout policy `pick`
    (signature: hand, trick_by_seat, order_cards, lead_suit, trump ->
    card). hands: {seat:[cards]}. partial: {seat:card} already in the
    current trick; partial_lead its lead suit. Returns our_team_points
    - opp_team_points (5/trick + 5/high)."""
    tricks = [0, 0, 0, 0]
    best_trump_seat, best_trump_val = None, -1

    cur_lead = leader
    while any(hands.values()):
        trick = dict(partial) if partial else {}
        lead_suit = partial_lead
        order = [(cur_lead + i) % 4 for i in range(4)]
        # cards already in the carried partial, in play order (leader first)
        order_cards = [partial[s] for s in order if partial and s in partial]
        for seat in order:
            if seat in trick:
                continue
            if not hands[seat]:
                continue
            if lead_suit is None and not trick:
                card = pick(hands[seat], {}, [], None, trump)
                lead_suit = card.suit
            else:
                card = pick(hands[seat], trick, order_cards, lead_suit, trump)
            hands[seat].remove(card)
            trick[seat] = card
            order_cards.append(card)
            tv = get_card_rank(card, trump)
            if _is_trump(card, trump) and tv > best_trump_val:
                best_trump_val, best_trump_seat = tv, seat
        partial, partial_lead = None, None  # consumed
        w = _trick_winner(trick, lead_suit, trump)
        tricks[w] += 1
        cur_lead = w

    our = sum(tricks[s] for s in range(4) if s % 2 == our_parity)
    opp = sum(tricks[s] for s in range(4) if s % 2 != our_parity)
    our_pts = 5 * our + (5 if (best_trump_seat is not None and
                               best_trump_seat % 2 == our_parity) else 0)
    opp_pts = 5 * opp + (5 if (best_trump_seat is not None and
                               best_trump_seat % 2 != our_parity) else 0)
    return our_pts - opp_pts


class PIMCAgent:
    def __init__(self, num_actions=18, n_worlds=20, seed=0,
                 constrained=True, rollout='rulebased'):
        # Canonical config = v3: constrained determinization +
        # rule-based playout. It robustly beats the competent
        # rule-based on independent seeds (memory: project-play-phase-
        # ceiling). constrained/rollout remain overridable for ablation
        # (rollout='cheap' = v1/v2; constrained=False = unconstrained).
        self.num_actions = num_actions
        self.n_worlds = n_worlds
        # v2 lever 1: constrain determinization by inferred opponent
        # voids (a seat that played a non-lead, non-trump card had no
        # card of the lead suit — reneging is only legal via trump in
        # this game, so an off-suit sluff is a hard void). constrained=
        # False reproduces v1 exactly for a single-variable A/B.
        self.constrained = constrained
        # v2 lever 3: playout policy. 'cheap' = v1/v2 win-or-dump
        # heuristic; 'rulebased' = the fixed competent rule-based play
        # strategy as the rollout policy (single-variable vs v2).
        self.rollout = rollout
        if rollout == 'rulebased':
            self._pick = _rb_pick_factory(RuleBasedAgent(num_actions))
        else:
            self._pick = _cheap_pick
        self.use_raw = True
        self._rng = np.random.RandomState(seed)

    # --- helpers ---------------------------------------------------------
    def _seen(self, hand, current_trick, trick_history):
        seen = {(c.rank, c.suit) for c in hand}
        for c in current_trick:
            if c is not None:
                seen.add((c.rank, c.suit))
        for tr in (trick_history or []):
            for c in tr:
                if c is not None:
                    seen.add((c.rank, c.suit))
        return seen

    def _voids(self, raw, trump, cur_leader):
        """Infer hard suit voids. A seat that played a card that is
        neither the lead suit nor trump (A♥ counts as trump) had no
        lead-suit card — a real void from that trick onward. Trump plays
        are inconclusive (always legal). Reconstruct each completed
        trick's leader from highest_bidder (trick 0) then trick_winners.
        Returns {seat: set(void_suits)}."""
        voids = {s: set() for s in range(4)}

        def scan(trick, leader):
            if leader is None or leader >= len(trick) or trick[leader] is None:
                return
            lead = trick[leader].suit
            for s, c in enumerate(trick):
                if c is None or s == leader:
                    continue
                if c.suit != lead and not _is_trump(c, trump):
                    voids[s].add(lead)

        hb = raw.get('highest_bidder')
        winners = list(raw.get('trick_winners') or [])
        history = list(raw.get('trick_history') or [])
        lead0 = (hb + 1) % 4 if hb is not None else None
        for ti, tr in enumerate(history):
            leader = lead0 if ti == 0 else (
                winners[ti - 1] if ti - 1 < len(winners) else None)
            scan(tr, leader)
        # in-progress trick (leader reconstructed locally in step())
        cur = raw.get('current_trick')
        if cur and any(c is not None for c in cur):
            scan(cur, cur_leader)
        return voids

    def _determinize(self, seen, sizes, voids=None):
        """sizes: {seat: n}. Deal unseen cards to those seats. If voids
        given, never deal a seat a card of a suit it is void in. Greedy
        with reshuffled retries; falls back to unconstrained if the
        sample is over-constrained (rare)."""
        unseen = [_BY_RS[k] for k in _BY_RS if k not in seen]
        if voids:
            for _ in range(8):
                self._rng.shuffle(unseen)
                pool, out, ok = list(unseen), {}, True
                for seat, n in sizes.items():
                    vs = voids.get(seat, ())
                    picked, rest = [], []
                    for c in pool:
                        if len(picked) < n and c.suit not in vs:
                            picked.append(c)
                        else:
                            rest.append(c)
                    if len(picked) < n:
                        ok = False
                        break
                    out[seat], pool = picked, rest
                if ok:
                    return out
        self._rng.shuffle(unseen)
        out, i = {}, 0
        for seat, n in sizes.items():
            out[seat] = unseen[i:i + n]
            i += n
        return out

    # --- agent API -------------------------------------------------------
    def step(self, state):
        raw = state['raw_obs']
        # Work in GAME id / hand-index space; env.step expects ENV ids
        # (play card k -> k+9). Using state['legal_actions'] (ENV ids)
        # here was the same bug that made rule-based fall back to
        # min-legal. See memory project-action-space-bug.
        raw_legal = list(state.get('raw_legal_actions') or [])
        if raw.get('phase') != 4 or not raw_legal:
            env_legal = list(state['legal_actions'].keys())
            return min(env_legal) if env_legal else 0

        hand = raw['hand']
        trump = raw['trump_suit']
        our = raw['current_player']
        our_parity = our % 2
        ct = raw['current_trick']
        played = {s: c for s, c in enumerate(ct) if c is not None}
        k = len(played)
        leader = (our - k) % 4
        lead_suit = ct[leader].suit if k > 0 else None
        t = len(raw.get('trick_history') or [])

        sizes = {}
        for s in range(4):
            if s == our:
                continue
            sizes[s] = (5 - t) - (1 if ct[s] is not None else 0)

        voids = self._voids(raw, trump, leader) if self.constrained else None
        seen = self._seen(hand, ct, raw.get('trick_history'))

        # raw_legal entries are game card indices into the hand. Evaluate
        # each by PIMC; return the ENV id (game index + 9).
        best_action, best_score = sorted(raw_legal)[0], -1e9
        for a in sorted(raw_legal):
            if not (0 <= a < len(hand)):
                continue
            cand = hand[a]
            total = 0.0
            for _ in range(self.n_worlds):
                opp = self._determinize(seen, sizes, voids)
                hands = {our: [c for c in hand if (c.rank, c.suit) != (cand.rank, cand.suit)]}
                hands.update({s: list(cs) for s, cs in opp.items()})
                # carry the in-progress trick + our just-played card
                partial = dict(played)
                partial[our] = cand
                p_lead = lead_suit if lead_suit is not None else cand.suit
                # after we play, the trick continues from our+1; once it
                # completes _simulate computes the winner and continues.
                total += _simulate(hands, leader, trump, our_parity,
                                   self._pick,
                                   partial=partial, partial_lead=p_lead)
            score = total / self.n_worlds
            if score > best_score:
                best_score, best_action = score, a
        return best_action + 9  # game card index -> env play id

    def eval_step(self, state):
        return self.step(state), {}
