"""
Regression test: the auction must skip players who have already passed.

Bug (game.py):
  - process_auction advanced the turn with `(current+1) % num_players`
    and did NOT skip players who already passed, so once the pointer
    wrapped past a passed player that player was made current again.
  - get_legal_actions' auction fallback then manufactured `[BID_PASS]`
    for that already-passed player (get_legal_bids returns [] for a
    passed player), so a player who was out of the auction kept being
    re-prompted to "pass".

Correct behaviour: a player who has passed is out — the turn pointer
skips them, and they are never handed a forced pass.

Default game: 4 players, dealer P0, auction starts at P1; order P1 -> P2
-> P3 -> P0 -> ...
"""

import unittest

from fortyfives.games.fortyfives.game import (
    FortyfivesGame, BID_PASS, BID_20, BID_25, PHASE_AUCTION,
)


class TestAuctionSkipsPassed(unittest.TestCase):
    def test_advance_skips_already_passed_player(self):
        g = FortyfivesGame()
        self.assertEqual(g.phase, PHASE_AUCTION)
        self.assertEqual(g.current_player_id, 1)  # P1 (East) starts

        g.process_auction(BID_PASS)   # P1 passes            -> P2
        self.assertEqual(g.current_player_id, 2)
        g.process_auction(BID_20)     # P2 bids 20           -> P3
        self.assertEqual(g.current_player_id, 3)
        g.process_auction(BID_25)     # P3 bids 25           -> P0
        self.assertEqual(g.current_player_id, 0)
        g.process_auction(BID_PASS)   # P0 passes; auction not over

        # passed = {P0, P1}; bidders P2/P3 still active; not over yet.
        self.assertTrue(g.passed[1] and g.passed[0])
        self.assertFalse(g.is_bidding_over())
        self.assertEqual(g.phase, PHASE_AUCTION)

        # The bug: advance lands on P1, who already passed.
        self.assertFalse(
            g.passed[g.current_player_id],
            f"auction handed a turn to already-passed player "
            f"{g.current_player_id}")
        # It must skip passed P1 and go to the next active player, P2.
        self.assertEqual(g.current_player_id, 2)

    def test_fallback_does_not_force_pass_for_passed_player(self):
        # Defensive: even if an already-passed player were current, the
        # legal-actions fallback must not fabricate a BID_PASS for them.
        g = FortyfivesGame()
        cp = g.current_player_id
        g.passed[cp] = True

        self.assertEqual(g.get_legal_bids(), [],
                         "a passed player has no legal bids")
        self.assertNotIn(
            BID_PASS, g.get_legal_actions(),
            "an already-passed player must not be handed a forced pass")


if __name__ == '__main__':
    unittest.main()
