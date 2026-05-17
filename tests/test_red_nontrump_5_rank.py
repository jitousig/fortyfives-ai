"""
Regression test: the 5 of a RED non-trump suit must have a real rank.

Bug (card.py get_card_rank, red non-trump branch): the rank_map omitted
the '5' key, so a red non-trump 5 fell through to `rank_map.get(rank, 0)`
and ranked 0 — below every other card including the (low) red Ace, so a
red non-trump 5 could never win a trick it should. Black non-trump
correctly includes '5'; only red was affected.

Correct 45s red non-trump order (high -> low):
    K > Q > J > 10 > 9 > 8 > 7 > 6 > 5 > 4 > 3 > 2 > A
"""

import unittest

from fortyfives.games.fortyfives.card import FortyfivesCard, get_card_rank

RANKS = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
         '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
SUITS = {'S': 0, 'H': 1, 'D': 2, 'C': 3}


def card(rank, suit):
    return FortyfivesCard(RANKS[rank] + 13 * SUITS[suit])


class TestRedNonTrumpFiveRank(unittest.TestCase):
    def test_red_nontrump_5_is_not_zero(self):
        # Diamonds is red & non-trump when spades is trump.
        self.assertNotEqual(
            get_card_rank(card('5', 'D'), 'S'), 0,
            "red non-trump 5 must have a real rank, not the 0 fallback")

    def test_red_nontrump_5_beats_lower_cards(self):
        # 5 must outrank 4, 3, 2 and the low red Ace.
        r5 = get_card_rank(card('5', 'D'), 'S')
        for lower in ('4', '3', '2', 'A'):
            self.assertGreater(
                r5, get_card_rank(card(lower, 'D'), 'S'),
                f"red non-trump 5 must outrank {lower}")

    def test_red_nontrump_5_below_6(self):
        self.assertGreater(
            get_card_rank(card('6', 'D'), 'S'),
            get_card_rank(card('5', 'D'), 'S'),
            "red non-trump 6 must outrank 5")

    def test_red_nontrump_full_order_with_5(self):
        # Strict descending chain incl. 5, both red suits (spades trump
        # so both H and D are red non-trump; A-of-H is always trump so
        # use diamonds for the Ace position).
        order = ['K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4',
                 '3', '2', 'A']
        ranks = [get_card_rank(card(r, 'D'), 'S') for r in order]
        self.assertEqual(
            ranks, sorted(ranks, reverse=True),
            f"red non-trump order broken: {list(zip(order, ranks))}")
        self.assertEqual(len(set(ranks)), len(ranks),
                         "red non-trump ranks must be unique")

    def test_hearts_red_nontrump_5_also_fixed(self):
        # Symmetry: hearts is red non-trump when clubs is trump.
        self.assertGreater(
            get_card_rank(card('5', 'H'), 'C'),
            get_card_rank(card('4', 'H'), 'C'),
            "red non-trump 5 of hearts must outrank 4 of hearts")

    def test_black_nontrump_5_unaffected(self):
        # Guard: black non-trump already had '5'; fix must not disturb it.
        # Black non-trump order: K>Q>J>A>2>3>4>5>6>7>8>9>10, so 4 > 5 > 6.
        self.assertGreater(get_card_rank(card('4', 'C'), 'S'),
                           get_card_rank(card('5', 'C'), 'S'))
        self.assertGreater(get_card_rank(card('5', 'C'), 'S'),
                           get_card_rank(card('6', 'C'), 'S'))


if __name__ == '__main__':
    unittest.main()
