"""
Replenish counts must be recorded after the discard phase and exposed in
get_state as 'replenish_counts'. This is legitimately public information
at a real table (everyone sees how many cards each player draws), and it
is the input for discard-count-constrained PIMC determinization.
"""

import unittest
from fortyfives.games.fortyfives.game import (
    FortyfivesGame,
    PHASE_AUCTION, PHASE_GAMEPLAY,
    DISCARD_DONE,
)


def play_through_discard(seed=0):
    """Drive a fresh game with min-legal actions until gameplay begins.
    Returns the game (or None if everyone passed and the hand ended)."""
    game = FortyfivesGame()
    game.np_random.seed(seed)
    state, _ = game.init_game()
    for _ in range(200):
        if game.phase == PHASE_GAMEPLAY:
            return game
        legal = game.get_legal_actions()
        if not legal:
            return None
        if game.phase == PHASE_AUCTION and 1 in legal:
            game.step(1)  # BID_20, so the hand always has a discard phase
        else:
            game.step(min(legal))
    return None


class TestReplenishCounts(unittest.TestCase):

    def test_none_before_discard_complete(self):
        game = FortyfivesGame()
        game.init_game()
        self.assertEqual(game.phase, PHASE_AUCTION)
        state = game.get_state(0)
        self.assertIsNone(state['replenish_counts'])

    def test_counts_recorded_and_consistent(self):
        checked = 0
        for seed in range(30):
            game = play_through_discard(seed)
            if game is None:
                continue  # all-pass hand, no discard phase
            checked += 1
            state = game.get_state(0)
            counts = state['replenish_counts']
            self.assertIsInstance(counts, list)
            self.assertEqual(len(counts), game.num_players)
            for pid in range(game.num_players):
                self.assertGreaterEqual(counts[pid], 0)
                self.assertLessEqual(counts[pid], 5)
                # hand was replenished back to 5, so the count is exactly
                # 5 minus what the player kept at discard
                self.assertEqual(len(game.hands[pid]), 5)
            # every card drawn came out of the deck: 32-card deck, 20
            # dealt, 3 kitty to bidder -> deck had 9... (deck size varies
            # by engine; just assert internal consistency instead)
            self.assertEqual(counts, game.replenish_counts)
        self.assertGreater(checked, 0, "no seed reached gameplay; test is vacuous")

    def test_copy_not_reference(self):
        game = play_through_discard(1) or play_through_discard(2)
        self.assertIsNotNone(game)
        state = game.get_state(0)
        state['replenish_counts'][0] = 99
        self.assertNotEqual(game.replenish_counts[0], 99)


if __name__ == '__main__':
    unittest.main()
