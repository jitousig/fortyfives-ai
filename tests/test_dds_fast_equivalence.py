"""Bit-exactness gate: FastDDSolver must return IDENTICAL solve() and
root_values() results to the reference DDSolver (minimax model) at
every decision of randomly dealt hands, across trumps, bid contexts,
payoff modes, hand sizes, and mid-trick entry points.

This gate is what licenses using the fast solver inside PIMCDDSAgent
with no A/B: identical values => identical moves => identical eval
numbers, only faster. If this test fails, DO NOT ship the fast path.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'examples'))

from fortyfives_dds import (
    DDSolver, trick_winner, _RANK, _ISTRUMP, _EMPTY_TRICK,
)
from fortyfives_dds_fast import FastDDSolver


def _playout_compare(test, rng, hand_size, deals):
    """Deal `deals` random hands of `hand_size` cards/seat and play each
    to the end, comparing both solvers' root_values at EVERY decision
    (mid-trick states included by construction)."""
    for _ in range(deals):
        deck = list(range(52))
        rng.shuffle(deck)
        hands = [tuple(deck[i * hand_size:(i + 1) * hand_size])
                 for i in range(4)]
        trump = rng.randrange(4)
        bid_team = rng.randrange(2)
        bid_kind = rng.choice([1, 2, 3])
        payoff = rng.choice(['delta', 'raw'])
        ref = DDSolver(trump, bid_team, bid_kind, payoff=payoff)
        fast = FastDDSolver(trump, bid_team, bid_kind, payoff=payoff)

        trick = [-1, -1, -1, -1]
        leader = rng.randrange(4)
        ns = ew = 0
        br, bp = -1, -1

        test.assertEqual(
            ref.solve(tuple(hands), leader),
            fast.solve(tuple(hands), leader),
            msg=f'solve mismatch: {hands} t={trump} '
                f'{bid_team}/{bid_kind}/{payoff}')

        while any(hands):
            played = sum(1 for c in trick if c >= 0)
            seat = (leader + played) % 4
            args = (tuple(hands), leader, tuple(trick), ns, ew, br, bp)
            rv = ref.root_values(*args)
            fv = fast.root_values(*args)
            test.assertEqual(
                rv, fv,
                msg=f'root_values mismatch at {hands} trick={trick} '
                    f'leader={leader} ns={ns} br={br}/{bp} '
                    f't={trump} {bid_team}/{bid_kind}/{payoff}')

            # Advance with the reference argmax/argmin (lowest index
            # tie-break), replicating DDSolver._child_value's transition.
            sign = 1 if seat % 2 == 0 else -1
            i = max(sorted(rv), key=lambda k: sign * rv[k])
            card = hands[seat][i]
            hands[seat] = hands[seat][:i] + hands[seat][i + 1:]
            trick[seat] = card
            if _ISTRUMP[trump][card]:
                r = _RANK[trump][card]
                if r > br:
                    br, bp = r, seat % 2
            if sum(1 for c in trick if c >= 0) == 4:
                w = trick_winner(tuple(trick), leader, trump)
                if w % 2 == 0:
                    ns += 1
                else:
                    ew += 1
                trick = [-1, -1, -1, -1]
                leader = w


class TestFastDDSEquivalence(unittest.TestCase):

    def test_full_hands(self):
        _playout_compare(self, random.Random(1001), hand_size=5,
                         deals=30)

    def test_partial_hands(self):
        rng = random.Random(2002)
        _playout_compare(self, rng, hand_size=4, deals=10)
        _playout_compare(self, rng, hand_size=3, deals=10)
        _playout_compare(self, rng, hand_size=2, deals=10)
        _playout_compare(self, rng, hand_size=1, deals=10)

    def test_solver_reuse_across_positions(self):
        """One solver instance across many same-context deals (the TT
        clear-on-context-change path) must stay exact."""
        rng = random.Random(3003)
        ref = DDSolver(1, 0, 3)
        fast = FastDDSolver(1, 0, 3)
        for _ in range(10):
            size = rng.choice([2, 3, 4, 5])   # forces total change
            deck = list(range(52))
            rng.shuffle(deck)
            hands = tuple(tuple(deck[i * size:(i + 1) * size])
                          for i in range(4))
            leader = rng.randrange(4)
            self.assertEqual(ref.solve(hands, leader),
                             fast.solve(hands, leader))
            self.assertEqual(ref.root_values(hands, leader),
                             fast.root_values(hands, leader))


if __name__ == '__main__':
    unittest.main()
