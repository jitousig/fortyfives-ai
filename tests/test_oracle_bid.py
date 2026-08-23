"""
OracleBidder (examples/fortyfives_oracle_bid.py) structural gates:
auction replay up to NOW, auction continuation under the table model,
deal valuation plumbing (kitty -> declarer, discard policy, replenish
order, solve), determinism, and env-id plumbing.
"""
import os
import sys
import unittest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO)
sys.path.insert(1, os.path.join(_REPO, 'examples'))

import numpy as np
import rlcard
from rlcard.envs.registration import register, registry

if 'fortyfives' not in registry.env_specs:
    register(env_id='fortyfives',
             entry_point='fortyfives.envs.fortyfives_env:FortyfivesEnv')

from fortyfives.games.fortyfives.card import FortyfivesCard, RANKS, SUITS
from fortyfives.games.fortyfives.game import BID_PASS, BID_20, BID_25, BID_30, BID_HOLD
from fortyfives_oracle_bid import OracleBidder
from fortyfives_rule_based import RuleBasedAgent


def C(rank, suit):
    return FortyfivesCard(RANKS.index(rank) + SUITS.index(suit) * 13)


class TestAuctionReplay(unittest.TestCase):

    def test_turns_so_far_matches_live_game(self):
        """Drive real auctions with rule-based seats; at every NS turn the
        replay must reproduce the live state and the earlier seats'
        recorded actions."""
        ob = OracleBidder(n_worlds=1)
        rb = RuleBasedAgent(21)
        checked = 0
        for seed in range(40):
            env = rlcard.make('fortyfives')
            env.seed(seed)
            state, pid = env.reset()
            transcript = {s: [] for s in range(4)}
            guard = 0
            while env.game.phase == 1 and guard < 30:
                guard += 1
                raw = state['raw_obs']
                legal = tuple(sorted(env.game.get_legal_bids()))
                rep = ob._turns_so_far(raw)
                self.assertIsNotNone(rep, f'seed {seed}')
                turns, sim = rep
                self.assertEqual(turns, transcript)
                self.assertEqual(sim['bidder'], env.game.highest_bidder)
                self.assertEqual(sim['highest'], env.game.highest_bid)
                self.assertEqual(sim['passed'], list(env.game.passed))
                checked += 1
                a = rb.step(state)
                g = a - 13 if a >= 18 else a
                transcript[pid].append((legal, g))
                state, pid = env.step(a)
                if env.game.phase == 1 and sum(env.game.passed) == 0 \
                        and env.game.highest_bidder is None and guard > 3:
                    transcript = {s: [] for s in range(4)}   # all-pass redeal
        self.assertGreater(checked, 100)

    def test_finish_auction(self):
        ob = OracleBidder(n_worlds=1)
        # seat 1 to act first (dealer 0), everyone else hopeless hands
        hop = [C('2', 'S'), C('3', 'D'), C('4', 'C'), C('7', 'C'), C('8', 'D')]
        hands = {0: hop, 2: hop, 3: hop}
        sim = {'highest': None, 'bidder': None, 'passed': [False] * 4,
               'on_kitty': [False] * 4}
        self.assertEqual(ob._finish_auction(1, BID_20, hands, sim, 0), (1, BID_20, False))
        self.assertIsNone(ob._finish_auction(1, BID_PASS, hands, sim, 0))
        # seat 2 holds 5S+JS -> bids 25 over our 20; we pass later
        strong = [C('5', 'S'), C('J', 'S'), C('4', 'C'), C('7', 'C'), C('8', 'D')]
        hands = {0: hop, 2: strong, 3: hop}
        self.assertEqual(ob._finish_auction(1, BID_20, hands, sim, 0), (2, BID_25, False))
        self.assertEqual(ob._finish_auction(1, BID_30, hands, sim, 0), (1, BID_30, False))
        # dealer HOLD over a standing 20
        sim2 = {'highest': BID_20, 'bidder': 1, 'passed': [False, False, True, True],
                'on_kitty': [False] * 4}
        self.assertEqual(ob._finish_auction(0, BID_HOLD, {1: hop, 2: hop, 3: hop}, sim2, 0),
                         (0, BID_20, False))


class TestValuation(unittest.TestCase):

    def test_value_plumbing_and_bounds(self):
        ob = OracleBidder(n_worlds=1, payoff='net', declarer_penalty=0)
        rng = np.random.RandomState(0)
        deck = [FortyfivesCard(int(i)) for i in rng.permutation(52)]
        me = 0
        my = deck[:5]
        hands = {1: deck[5:10], 2: deck[10:15], 3: deck[15:20]}
        kitty, stock = deck[20:23], deck[23:]
        for declarer in range(4):
            for level in (1, 2, 3):
                for t in range(4):
                    v = ob._value(me, my, hands, kitty, stock, declarer, level, t, False)
                    self.assertIsInstance(v, int)
                    self.assertTrue(-90 <= v <= 90)
        # payoff='delta' values lie in the NS-delta range
        ob2 = OracleBidder(n_worlds=1, payoff='delta', declarer_penalty=0)
        v = ob2._value(me, my, hands, kitty, stock, 0, 3, 0, False)
        self.assertIn(v, list(range(-30, 31)) + [60])
        # declarer penalty shifts the declaring side by d
        ob3 = OracleBidder(n_worlds=1, payoff='net', declarer_penalty=15)
        v0 = ob._value(me, my, hands, kitty, stock, 0, 1, 0, False)
        self.assertEqual(ob3._value(me, my, hands, kitty, stock, 0, 1, 0, False), v0 - 15)
        v1 = ob._value(me, my, hands, kitty, stock, 1, 1, 0, False)
        self.assertEqual(ob3._value(me, my, hands, kitty, stock, 1, 1, 0, False), v1 + 15)

    def test_sample_world_partitions_deck(self):
        ob = OracleBidder(n_worlds=1)
        my = [C('5', 'S'), C('J', 'S'), C('A', 'H'), C('K', 'D'), C('2', 'C')]
        hands, kitty, stock = ob._sample_world(0, my, None)
        ids = [c.id for c in my] + [c.id for s in hands for c in hands[s]] \
            + [c.id for c in kitty] + [c.id for c in stock]
        self.assertEqual(sorted(ids), list(range(52)))
        self.assertEqual({len(h) for h in hands.values()}, {5})
        self.assertEqual(len(kitty), 3)
        # conditioned: seat 1 bid 25 -> must hold 5+J of a suit
        turns = {1: [((0, 1, 2, 3, 5, 6, 7), BID_25)]}
        for _ in range(20):
            hands, _, _ = ob._sample_world(0, my, turns)
            self.assertTrue(ob._seat_consistent(hands[1], turns[1]))


class TestAgentAPI(unittest.TestCase):

    def test_deterministic_and_legal(self):
        """Same seed -> same decisions; decisions always legal env ids;
        exercised through the real env for NS seats."""
        def run(seed):
            ob = OracleBidder(n_worlds=6, seed=seed)
            rb = RuleBasedAgent(21)
            env = rlcard.make('fortyfives')
            env.seed(7)
            state, pid = env.reset()
            acts = []
            guard = 0
            while env.game.phase in (1, 2) and guard < 40:
                guard += 1
                if pid in (0, 2):
                    a = ob.step(state)
                    self.assertIn(a, state['legal_actions'])
                    acts.append(a)
                else:
                    a = rb.step(state)
                state, pid = env.step(a)
            return acts
        self.assertEqual(run(3), run(3))
        # phase 3/4 delegate to rule-based (never raises)
        ob = OracleBidder(n_worlds=2)
        env = rlcard.make('fortyfives')
        env.seed(1)
        state, pid = env.reset()
        rb = RuleBasedAgent(21)
        guard = 0
        while not (env.game.phase == 4 and len(env.game.trick_history) >= 1) and guard < 200:
            guard += 1
            a = ob.step(state) if pid in (0, 2) else rb.step(state)
            self.assertIn(a, state['legal_actions'])
            state, pid = env.step(a)


if __name__ == '__main__':
    unittest.main()
