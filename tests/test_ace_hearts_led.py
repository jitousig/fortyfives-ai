"""
Issue #37: the Ace of Hearts is ALWAYS trump, so leading it leads TRUMP.
Followers holding trump must follow with trump (subject to the renege
rule: a 5/J of trump outranks A♥ and may be withheld); followers with no
trump may play anything. Before the fix the engine treated an A♥ lead
under a non-hearts trump as a plain HEARTS lead.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'examples'))

from fortyfives.games.fortyfives.game import FortyfivesGame
from fortyfives.games.fortyfives.card import FortyfivesCard, RANKS, SUITS


def C(rank, suit):
    return FortyfivesCard(RANKS.index(rank) + SUITS.index(suit) * 13)


def _game(trump, lead_card, follower_hand):
    """Player 0 led `lead_card`; player 1 (holding follower_hand) to act."""
    g = FortyfivesGame.__new__(FortyfivesGame)
    g.num_players = 4
    g.phase = 4
    g.trump_suit = trump
    g.test_strict_suit_following = False
    g.hands = [[], list(follower_hand), [], []]
    g.current_trick = [lead_card, None, None, None]
    g.trick_starter = 0
    g.trick_lead_suit = lead_card.suit
    g.current_player_id = 1
    return g


class TestAceHeartsLed(unittest.TestCase):

    def test_must_follow_with_trump(self):
        hand = [C('K', 'H'), C('7', 'S'), C('9', 'C'), C('2', 'D')]
        g = _game('S', C('A', 'H'), hand)
        self.assertEqual(g.get_legal_plays(), [1], 'only the trump 7S is legal')

    def test_no_trump_may_play_anything(self):
        hand = [C('K', 'H'), C('9', 'C'), C('2', 'D'), C('Q', 'H')]
        g = _game('S', C('A', 'H'), hand)
        self.assertEqual(g.get_legal_plays(), [0, 1, 2, 3])

    def test_five_and_jack_may_be_withheld(self):
        # 5S / JS outrank A-hearts -> renege allowed -> any card
        for top in ('5', 'J'):
            hand = [C(top, 'S'), C('K', 'H'), C('9', 'C')]
            g = _game('S', C('A', 'H'), hand)
            self.assertEqual(g.get_legal_plays(), [0, 1, 2], top)

    def test_low_trump_plus_high_trump_must_play_a_trump(self):
        hand = [C('5', 'S'), C('7', 'S'), C('K', 'H'), C('9', 'C')]
        g = _game('S', C('A', 'H'), hand)
        self.assertEqual(g.get_legal_plays(), [0, 1],
                         'must follow with trump; either trump is fine')

    def test_hearts_trump_unchanged(self):
        hand = [C('K', 'H'), C('7', 'S'), C('9', 'C')]
        g = _game('H', C('A', 'H'), hand)
        self.assertEqual(g.get_legal_plays(), [0])

    def test_dds_solvers_agree(self):
        """Both solvers' legality replicas must match the engine on an
        A-hearts lead (they used to mirror the bug on purpose)."""
        from fortyfives_dds import legal_plays
        try:
            from fortyfives_dds_fast import FastDDSolver, _legal as fast_legal
        except ImportError:
            fast_legal = None
        cases = [
            [C('K', 'H'), C('7', 'S'), C('9', 'C'), C('2', 'D')],
            [C('K', 'H'), C('9', 'C'), C('2', 'D'), C('Q', 'H')],
            [C('5', 'S'), C('K', 'H'), C('9', 'C')],
            [C('5', 'S'), C('7', 'S'), C('K', 'H'), C('9', 'C')],
        ]
        trump = SUITS.index('S')
        ah = C('A', 'H').id
        for hand in cases:
            g = _game('S', C('A', 'H'), hand)
            want = tuple(g.get_legal_plays())
            ids = tuple(c.id for c in hand)
            self.assertEqual(legal_plays(ids, ah, trump), want, hand)
            if fast_legal is not None:
                mask = 0
                for c in hand:
                    mask |= 1 << c.id
                got = fast_legal(mask, ah, trump)
                got_idx = tuple(i for i, c in enumerate(hand)
                                if (got >> c.id) & 1)
                self.assertEqual(got_idx, want, hand)


if __name__ == '__main__':
    unittest.main()
