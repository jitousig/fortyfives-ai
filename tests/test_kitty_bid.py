"""
Issue #36 — rule variant "going on the kitty": a bid at any level made on
the unseen kitty under the normal auction rules. If it wins, the bidder
throws in their hand (keeping A♥), takes the kitty, declares trump after
seeing it, then discards/draws as normal.
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

from fortyfives.games.fortyfives.game import (
    FortyfivesGame, PHASE_AUCTION, PHASE_DECLARATION, PHASE_DISCARD,
    PHASE_GAMEPLAY, BID_PASS, BID_20, BID_25, BID_30, BID_HOLD,
    BID_20_KITTY, BID_25_KITTY, BID_30_KITTY, DISCARD_DONE,
)
from fortyfives.games.fortyfives.card import FortyfivesCard, RANKS, SUITS
from fortyfives_rule_based import RuleBasedAgent


def C(rank, suit):
    return FortyfivesCard(RANKS.index(rank) + SUITS.index(suit) * 13)


def _fresh(seed=0):
    g = FortyfivesGame()
    g.np_random = np.random.RandomState(seed)
    g.init_game()
    return g


def _is_ah(c):
    return c.rank == 'A' and c.suit == 'H'


class TestKittyBidAuction(unittest.TestCase):

    def test_legal_bids_include_kitty_variants(self):
        g = _fresh()
        legal = g.get_legal_bids()
        for a in (BID_PASS, BID_20, BID_25, BID_30,
                  BID_20_KITTY, BID_25_KITTY, BID_30_KITTY):
            self.assertIn(a, legal)
        self.assertNotIn(BID_HOLD, legal)
        g.process_auction(BID_20_KITTY)          # seat 1: 20 on the kitty
        self.assertEqual(g.highest_bid, BID_20)
        self.assertEqual(g.highest_bidder, 1)
        self.assertEqual(g.bids[1], BID_20)
        self.assertTrue(g.on_kitty[1])
        legal = g.get_legal_bids()               # seat 2
        self.assertEqual(sorted(legal), [BID_PASS, BID_25, BID_30,
                                         BID_25_KITTY, BID_30_KITTY])

    def test_kitty_bidder_outbid_keeps_hand(self):
        g = _fresh(1)
        h1 = list(g.hands[1])
        g.process_auction(BID_20_KITTY)   # 1
        g.process_auction(BID_25)         # 2 outbids normally
        g.process_auction(BID_PASS)       # 3
        g.process_auction(BID_PASS)       # 0 (dealer)
        g.process_auction(BID_PASS)       # 1 passes
        self.assertEqual(g.phase, PHASE_DECLARATION)
        self.assertEqual(g.highest_bidder, 2)
        self.assertFalse(g.on_kitty[2])
        self.assertTrue(g.on_kitty[1])    # its standing (losing) bid was on the kitty
        self.assertEqual(g.hands[1], h1)  # untouched
        self.assertEqual(len(g.dealer.pot), 3)   # kitty still waiting for declarer

    def test_winning_kitty_bid_swaps_hand(self):
        g = _fresh(2)
        # Force an A♥ into seat 1's hand so the keep-A♥ path is exercised.
        ah = C('A', 'H')
        holder = next(s for s in range(4) if any(_is_ah(c) for c in g.hands[s])) \
            if any(_is_ah(c) for s in range(4) for c in g.hands[s]) else None
        if holder is None:                     # A♥ in kitty or stock: swap it in
            src = g.dealer.pot if any(_is_ah(c) for c in g.dealer.pot) else g.dealer.deck
            src.remove(next(c for c in src if _is_ah(c)))
            src.append(g.hands[1].pop())
            g.hands[1].append(ah)
        elif holder != 1:
            i = next(i for i, c in enumerate(g.hands[holder]) if _is_ah(c))
            g.hands[holder][i], g.hands[1][0] = g.hands[1][0], g.hands[holder][i]
        original = list(g.hands[1])
        kitty = list(g.dealer.pot)
        self.assertTrue(any(_is_ah(c) for c in original))
        g.process_auction(BID_25_KITTY)   # 1
        g.process_auction(BID_PASS)       # 2
        g.process_auction(BID_PASS)       # 3
        g.process_auction(BID_PASS)       # 0
        self.assertEqual(g.phase, PHASE_DECLARATION)
        self.assertEqual(g.current_player_id, 1)
        new = g.hands[1]
        self.assertEqual(len(new), 4)     # A♥ + 3 kitty cards
        self.assertTrue(any(_is_ah(c) for c in new))
        for c in kitty:
            self.assertIn(c, new)
        for c in original:
            if not _is_ah(c):
                self.assertNotIn(c, new)
                self.assertIn(c, g.discard_pile)
        self.assertEqual(g.dealer.pot, [])
        # Declaration does not re-add the (now empty) kitty.
        g.process_declaration(0)
        self.assertEqual(len(g.hands[1]), 4)
        self.assertEqual(g.phase, PHASE_DISCARD)
        # Normal discard/draw: everyone done -> bidder refilled to 5.
        guard = 0
        while g.phase == PHASE_DISCARD and guard < 20:
            guard += 1
            g.process_discard(DISCARD_DONE)
        self.assertEqual(g.phase, PHASE_GAMEPLAY)
        self.assertEqual(len(g.hands[1]), 5)
        self.assertEqual(g.replenish_counts[1], 1)

    def test_kitty_bid_without_ace_swaps_whole_hand(self):
        g = _fresh(3)
        for s in range(4):
            g.hands[s] = [c for c in g.hands[s] if not _is_ah(c)]
            while len(g.hands[s]) < 5:
                g.hands[s].append(g.dealer.deck.pop())
        g.dealer.pot = [c for c in g.dealer.pot if not _is_ah(c)]
        while len(g.dealer.pot) < 3:
            g.dealer.pot.append(g.dealer.deck.pop())
        kitty = list(g.dealer.pot)
        g.process_auction(BID_20_KITTY)
        for _ in range(3):
            g.process_auction(BID_PASS)
        self.assertEqual(sorted(c.id for c in g.hands[1]), sorted(c.id for c in kitty))

    def test_dealer_hold_over_kitty_bid_is_a_normal_bid(self):
        g = _fresh(4)
        h0 = list(g.hands[0])
        g.process_auction(BID_20_KITTY)   # 1
        g.process_auction(BID_PASS)       # 2
        g.process_auction(BID_PASS)       # 3
        g.process_auction(BID_HOLD)       # dealer holds
        g.process_auction(BID_PASS)       # 1 passes
        self.assertEqual(g.highest_bidder, 0)
        self.assertFalse(g.on_kitty[0])
        self.assertEqual(g.hands[0], h0)
        self.assertEqual(len(g.dealer.pot), 3)

    def test_on_kitty_reset_on_new_hand(self):
        g = _fresh(5)
        g.process_auction(BID_20_KITTY)
        for _ in range(3):
            g.process_auction(BID_PASS)
        g.start_new_hand()
        self.assertEqual(g.on_kitty, [False] * 4)
        # all-pass redeal also resets
        for _ in range(4):
            g.process_auction(BID_PASS)
        self.assertEqual(g.on_kitty, [False] * 4)


class TestKittyBidEnv(unittest.TestCase):

    def test_action_space_and_mapping(self):
        env = rlcard.make('fortyfives')
        self.assertEqual(env.num_actions, 21)
        self.assertEqual(env.state_shape[0], 414)
        state, pid = env.reset()
        legal = state['legal_actions']
        for a in (18, 19, 20):
            self.assertIn(a, legal)
        self.assertEqual(env._decode_action(18), BID_20_KITTY)
        self.assertEqual(env._decode_action(20), BID_30_KITTY)
        self.assertEqual(env._game_to_env_action(BID_25_KITTY, PHASE_AUCTION), 19)
        self.assertEqual(env._game_to_env_action(BID_25, PHASE_AUCTION), 2)
        state, pid = env.step(18)          # seat 1: 20 on the kitty
        self.assertTrue(env.game.on_kitty[1])
        self.assertEqual(state['raw_obs']['on_kitty'], [False, True, False, False])
        obs = state['obs']
        self.assertEqual(obs[410 + 1], 1)
        self.assertEqual(int(obs[410:414].sum()), 1)
        # legacy bid encoding unchanged: seat 1 bid level 1 (20)
        self.assertEqual(obs[52 * 5 + 5 + 1 * 4 + BID_20], 1)


class TestRuleBasedKittyPolicy(unittest.TestCase):

    def test_policy(self):
        rb = RuleBasedAgent(21)
        OPEN = {0, 1, 2, 3, 5, 6, 7}
        # hopeless hand with A♥ -> 20 on the kitty
        h = [C('A', 'H'), C('2', 'S'), C('3', 'D'), C('4', 'C'), C('7', 'C')]
        self.assertEqual(rb.desired_bid(h, OPEN), BID_20_KITTY)
        # 20 already taken -> pass (never kitty at 25)
        self.assertEqual(rb.desired_bid(h, {0, 2, 3, 6, 7}), BID_PASS)
        # hopeless hand without A♥ -> pass
        h2 = [C('K', 'H'), C('2', 'S'), C('3', 'D'), C('4', 'C'), C('7', 'C')]
        self.assertEqual(rb.desired_bid(h2, OPEN), BID_PASS)
        # a hand that supports a bid never goes on the kitty
        h3 = [C('5', 'S'), C('A', 'H'), C('3', 'D'), C('4', 'C'), C('7', 'C')]
        self.assertEqual(rb.desired_bid(h3, OPEN), BID_20)
        # flag off reproduces the pre-#36 bidder
        self.assertEqual(RuleBasedAgent(21, kitty=False).desired_bid(h, OPEN), BID_PASS)
        # env-id plumbing: _bid_strategy returns env id 18
        self.assertEqual(rb._bid_strategy({'hand': h}, {a: None for a in (0, 1, 2, 3, 18, 19, 20)}), 18)


if __name__ == '__main__':
    unittest.main()
