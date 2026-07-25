#!/usr/bin/env python3
"""
PIMC-DDS play agent: PIMC determinization with an EXACT double-dummy
solve per sampled world (the textbook Bridge/Skat recipe), replacing
PIMC v3's rule-based heuristic playout.

Two deliberate upgrades over PIMC v3, both downstream of the solver
(see RESEARCH.md § Active work):
  - Exact per-world play-out (fixes rollout-legality infidelity,
    "lever 2", and removes playout-policy noise entirely).
  - Bid-aware values: each world is scored as the final NS game-point
    delta (make/fail bid, 30-for-60), and the cross-world AVERAGE is
    taken in that space. E[delta] != f(E[raw]) exactly at
    make-the-bid vs risk-getting-set decisions — value PIMC v3
    structurally cannot see.

Determinization (seen cards, hard-void inference, world sizes) is
INHERITED from PIMCAgent unchanged, so PIMC v3 -> PIMC-DDS is a
single-variable change: the per-world evaluator.

opponent='minimax' (default): worlds are solved double-dummy — hidden
hands are known per world, policies are not, so paranoid play is the
principled choice. opponent='rulebased' is an ablation that models the
eval table's fixed EW policy inside each world (note: the forced policy
is hand-order-sensitive and sampled worlds have arbitrary order, so
this mode is approximate there by construction).

Usage:
    from fortyfives_pimc_dds import PIMCDDSAgent
    from play_eval import evaluate_paired
    evaluate_paired(PIMCDDSAgent(n_worlds=20), num_hands=2000)
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

from fortyfives.games.fortyfives.card import SUITS, get_card_rank

sys.path.insert(0, os.path.dirname(__file__))
from fortyfives_pimc import PIMCAgent, _is_trump, _BY_RS
from fortyfives_dds import DDSolver


class PIMCDDSAgent(PIMCAgent):

    def __init__(self, num_actions=18, n_worlds=20, seed=0,
                 constrained=True, opponent='minimax', payoff='delta',
                 discard_counts=False):
        # rollout is irrelevant here (no heuristic playout); pass
        # 'cheap' so the parent doesn't build a rule-based picker.
        super().__init__(num_actions=num_actions, n_worlds=n_worlds,
                         seed=seed, constrained=constrained,
                         rollout='cheap')
        self.opponent = opponent
        self.payoff = payoff
        # Estimator lever 1: constrain sampled worlds by each seat's
        # post-discard draw count (public at a real table). Rule-based
        # seats keep ONLY trump at discard, so kept = 5 - drawn is that
        # seat's trump count at replenish; minus trumps it has since
        # publicly played, it lower-bounds trumps still in hand.
        # Near-exact vs rule-based discarders; a heuristic prior vs
        # humans.
        self.discard_counts = discard_counts
        self._min_trumps = None   # per-step context for _determinize
        self._trump_str = None

    def _played_trumps(self, raw, trump_str):
        counts = {s: 0 for s in range(4)}
        for tr in (raw.get('trick_history') or []):
            for s, c in enumerate(tr):
                if c is not None and _is_trump(c, trump_str):
                    counts[s] += 1
        for s, c in enumerate(raw.get('current_trick') or []):
            if c is not None and _is_trump(c, trump_str):
                counts[s] += 1
        return counts

    def _determinize(self, seen, sizes, voids=None):
        """With discard-count constraints active, deal each seat at
        least its inferred minimum trump count, then fill the remaining
        slots from the full unseen pool (replenished cards are random,
        so trump stays in the fill pool). Greedy with reshuffled
        retries; falls back to the parent's sampler if over-constrained
        (e.g. a void conflict — rare/impossible vs rule-based)."""
        mt = self._min_trumps
        if not mt or not any(mt.get(s, 0) for s in sizes):
            return super()._determinize(seen, sizes, voids)
        trump = self._trump_str
        unseen = [_BY_RS[k] for k in _BY_RS if k not in seen]
        for _ in range(8):
            self._rng.shuffle(unseen)
            pool, out, ok = list(unseen), {}, True
            for seat, n in sizes.items():
                vs = voids.get(seat, ()) if voids else ()
                need = min(mt.get(seat, 0), n)
                picked = []
                if need:
                    avail = [c for c in pool
                             if _is_trump(c, trump) and c.suit not in vs]
                    if len(avail) < need:
                        ok = False
                        break
                    picked = avail[:need]
                fill = n - len(picked)
                if fill:
                    pids = {id(c) for c in picked}
                    avail = [c for c in pool
                             if id(c) not in pids and c.suit not in vs]
                    if len(avail) < fill:
                        ok = False
                        break
                    picked = picked + avail[:fill]
                pids = {id(c) for c in picked}
                pool = [c for c in pool if id(c) not in pids]
                out[seat] = picked
            if ok:
                return out
        return super()._determinize(seen, sizes, voids)

    def step(self, state):
        raw = state['raw_obs']
        raw_legal = list(state.get('raw_legal_actions') or [])
        if raw.get('phase') != 4 or not raw_legal:
            env_legal = list(state['legal_actions'].keys())
            return min(env_legal) if env_legal else 0

        hand = raw['hand']
        trump_str = raw['trump_suit']
        trump = SUITS.index(trump_str)
        our = raw['current_player']
        ct = raw['current_trick']
        played = {s: c for s, c in enumerate(ct) if c is not None}
        k = len(played)
        leader = (our - k) % 4
        t = len(raw.get('trick_history') or [])

        sizes = {}
        for s in range(4):
            if s == our:
                continue
            sizes[s] = (5 - t) - (1 if ct[s] is not None else 0)

        voids = self._voids(raw, trump_str, leader) if self.constrained \
            else None
        seen = self._seen(hand, ct, raw.get('trick_history'))

        # Discard-count constraint context (lever 1), consumed by our
        # _determinize override. kept_s = 5 - drawn_s is all trump for
        # rule-based discarders; subtract trumps seat s already showed.
        self._trump_str = trump_str
        self._min_trumps = None
        rc = raw.get('replenish_counts')
        if self.discard_counts and rc is not None:
            pt = self._played_trumps(raw, trump_str)
            self._min_trumps = {s: max(0, (5 - rc[s]) - pt[s])
                                for s in sizes}

        # Hand context for the bid-aware payoff.
        bid_team = raw['highest_bidder'] % 2
        bid_kind = raw['highest_bid']
        tw = raw['tricks_won']
        ns_tr, ew_tr = tw[0] + tw[2], tw[1] + tw[3]
        htp = raw.get('highest_trump_played')
        if htp is not None:
            best_rank = get_card_rank(htp, trump_str)
            best_par = raw['highest_trump_player'] % 2
        else:
            best_rank, best_par = -1, -1

        trick_ids = tuple((ct[s].id if ct[s] is not None else -1)
                          for s in range(4))
        our_ids = tuple(c.id for c in hand)

        totals = {a: 0.0 for a in raw_legal if 0 <= a < len(hand)}
        for _ in range(self.n_worlds):
            opp = self._determinize(seen, sizes, voids)
            hands = tuple(
                our_ids if s == our
                else tuple(c.id for c in opp.get(s, []))
                for s in range(4))
            solver = DDSolver(trump, bid_team, bid_kind,
                              opponent=self.opponent, payoff=self.payoff)
            vals = solver.root_values(hands, leader, trick_ids,
                                      ns_tr, ew_tr, best_rank, best_par)
            for a in totals:
                totals[a] += vals[a]

        best = max(sorted(totals), key=lambda a: totals[a])
        return best + 9   # game hand index -> env play action id
