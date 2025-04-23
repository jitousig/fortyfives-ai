"""
Test to verify card creation and differentiation works correctly.
This ensures cards of different suits have different IDs and properties.
"""

import unittest
from fortyfives.games.fortyfives.card import FortyfivesCard, RANKS, SUITS, get_card_rank

def create_card(rank, suit):
    """Helper function to create a card of specific rank and suit."""
    suits = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
    ranks = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    return FortyfivesCard(ranks[rank] + 13 * suits[suit])

class TestCardCreation(unittest.TestCase):
    """Tests for card creation and differentiation."""
    
    def test_card_creation(self):
        """Test that cards are created with the correct properties."""
        diamonds_four = create_card('4', 'D')
        self.assertEqual(diamonds_four.rank, '4')
        self.assertEqual(diamonds_four.suit, 'D')
        
        clubs_four = create_card('4', 'C')
        self.assertEqual(clubs_four.rank, '4')
        self.assertEqual(clubs_four.suit, 'C')
        
        # Verify the cards have different IDs
        self.assertNotEqual(diamonds_four.id, clubs_four.id)
    
    def test_all_suits_different(self):
        """Test that cards of the same rank but different suits have different IDs."""
        cards = {}
        # Create cards of rank '4' for all suits
        for suit in ['S', 'H', 'D', 'C']:
            cards[suit] = create_card('4', suit)
        
        # Verify each suit has the expected suit value
        self.assertEqual(cards['S'].suit, 'S')
        self.assertEqual(cards['H'].suit, 'H')
        self.assertEqual(cards['D'].suit, 'D')
        self.assertEqual(cards['C'].suit, 'C')
        
        # Verify all IDs are different
        ids = [card.id for card in cards.values()]
        self.assertEqual(len(ids), len(set(ids)), "All card IDs should be unique")
    
    def test_card_rank_value(self):
        """Test that card rank values work correctly."""
        # Set trump suit for testing
        trump_suit = 'S'
        
        # Ace should have higher rank value than 2 in non-trump suit
        ace_hearts = create_card('A', 'H')
        two_hearts = create_card('2', 'H')
        
        self.assertGreater(get_card_rank(ace_hearts, trump_suit), 
                          get_card_rank(two_hearts, trump_suit))
        
        # Check special case: Ace of Hearts is always high trump
        ace_hearts = create_card('A', 'H')
        five_spades = create_card('5', 'S')  # 5 of trump
        
        # 5 of trump should be higher than Ace of Hearts
        self.assertGreater(get_card_rank(five_spades, trump_suit), 
                           get_card_rank(ace_hearts, trump_suit))
        
        # Verify Jack of trump is second highest
        jack_spades = create_card('J', 'S')
        self.assertGreater(get_card_rank(jack_spades, trump_suit), 
                           get_card_rank(ace_hearts, trump_suit))
        
        # Ace of trump should be lower than Jack of trump
        ace_spades = create_card('A', 'S')
        self.assertGreater(get_card_rank(jack_spades, trump_suit), 
                           get_card_rank(ace_spades, trump_suit))

if __name__ == "__main__":
    unittest.main() 