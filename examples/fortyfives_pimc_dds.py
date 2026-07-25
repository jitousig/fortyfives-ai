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
from fortyfives_pimc import PIMCAgent
from fortyfives_dds import DDSolver


class PIMCDDSAgent(PIMCAgent):

    def __init__(self, num_actions=18, n_worlds=20, seed=0,
                 constrained=True, opponent='minimax', payoff='delta'):
        # rollout is irrelevant here (no heuristic playout); pass
        # 'cheap' so the parent doesn't build a rule-based picker.
        super().__init__(num_actions=num_actions, n_worlds=n_worlds,
                         seed=seed, constrained=constrained,
                         rollout='cheap')
        self.opponent = opponent
        self.payoff = payoff

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
