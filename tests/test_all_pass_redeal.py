"""
Issue #28: when every player passes the auction, the hand is thrown in
and redealt by the SAME dealer — the deal must not rotate. (After a
played hand the dealer does rotate, as before.)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from fortyfives.games.fortyfives.game import (
    FortyfivesGame, PHASE_AUCTION, PHASE_GAMEPLAY, BID_PASS, BID_20,
    DISCARD_DONE,
)


def _fresh(seed=0):
    g = FortyfivesGame()
    g.np_random = np.random.RandomState(seed)
    g.init_game()
    return g


class TestAllPassRedeal(unittest.TestCase):

    def test_all_pass_keeps_dealer_and_redeals(self):
        g = _fresh(3)
        dealer = g.dealer_id
        first_hands = {p: [(c.rank, c.suit) for c in g.hands[p]]
                       for p in range(4)}
        self.assertEqual(g.current_player_id, (dealer + 1) % 4)
        for _ in range(4):
            self.assertEqual(g.phase, PHASE_AUCTION)
            g.step(BID_PASS)
        # Still in the auction of a NEW hand, dealt by the same dealer.
        self.assertEqual(g.phase, PHASE_AUCTION)
        self.assertEqual(g.dealer_id, dealer, 'dealer must not rotate on all-pass')
        self.assertEqual(g.auction_starting_player, (dealer + 1) % 4)
        self.assertEqual(g.current_player_id, (dealer + 1) % 4)
        self.assertIsNone(g.highest_bidder)
        self.assertEqual(g.passed, [False] * 4)
        new_hands = {p: [(c.rank, c.suit) for c in g.hands[p]] for p in range(4)}
        for p in range(4):
            self.assertEqual(len(new_hands[p]), 5)
        self.assertNotEqual(new_hands, first_hands, 'cards must be redealt')

    def test_dealer_rotates_after_a_played_hand(self):
        g = _fresh(5)
        dealer = g.dealer_id
        guard = 0
        while g.phase == PHASE_AUCTION and guard < 8:
            guard += 1
            g.step(BID_20 if g.highest_bid is None else BID_PASS)
        # declare, discard-done for everyone, play out the hand with any legal card
        guard = 0
        while not (g.phase == PHASE_AUCTION and g.trump_suit is None) and guard < 200:
            guard += 1
            legal = g.get_legal_actions()
            if g.phase == PHASE_GAMEPLAY:
                g.step(legal[0])
            elif DISCARD_DONE in legal:
                g.step(DISCARD_DONE)
            else:
                g.step(legal[0])
        self.assertEqual(g.phase, PHASE_AUCTION)
        self.assertEqual(g.dealer_id, (dealer + 1) % 4,
                         'dealer rotates after a completed hand')


if __name__ == '__main__':
    unittest.main()
