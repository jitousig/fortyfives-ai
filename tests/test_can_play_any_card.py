"""
Test to verify that when a player can't follow the lead suit, they can play any card from their hand.
This test ensures that the rule allowing players to play any card when they can't follow lead suit
is correctly implemented.
"""

import unittest
from fortyfives.games.fortyfives.game import FortyfivesGame
from fortyfives.games.fortyfives.card import FortyfivesCard

def create_card(rank, suit):
    """Helper function to create a card of specific rank and suit."""
    suits = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
    ranks = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    return FortyfivesCard(ranks[rank] + 13 * suits[suit])

class TestCanPlayAnyCard(unittest.TestCase):
    """Test cases for verifying the card playing rules when a player can/cannot follow suit."""

    def test_must_follow_suit_when_possible(self):
        """Test that a player must follow the lead suit when they have cards of that suit."""
        # Create a game instance
        game = FortyfivesGame()
        game.verbose = False
        
        # Set up a specific game state
        game.phase = 4  # Gameplay phase
        game.trick_lead_suit = 'C'  # Clubs is the lead suit
        game.trump_suit = 'H'  # Hearts is trump
        game.trick_starter = 0  # North led the trick
        
        # Set up a player who has a club (lead suit)
        player_hand = [
            create_card('2', 'S'),  # 2 of Spades - not lead suit, not trump
            create_card('Q', 'H'),  # Queen of Hearts - trump
            create_card('A', 'S'),  # Ace of Spades - not lead suit, not trump
            create_card('6', 'H'),  # 6 of Hearts - trump
            create_card('4', 'C')   # 4 of Clubs - lead suit
        ]
        
        # Set player 2 (South) as current player
        game.current_player_id = 2
        game.hands = [[], [], [], []]  # Initialize empty hands
        game.hands[2] = player_hand    # Assign hand to South
        game.current_trick = [None, None, None, None]  # Initialize empty trick
        game.current_trick[0] = create_card('7', 'C')  # North led 7 of Clubs
        
        # Get legal plays for the player
        legal_plays = game.get_legal_plays()
        
        # Test that player can play club (lead suit) or trump (hearts)
        self.assertIn(4, legal_plays)  # 4 of Clubs (lead suit) must be a legal play
        self.assertIn(1, legal_plays)  # Queen of Hearts (trump) must be a legal play
        self.assertIn(3, legal_plays)  # 6 of Hearts (trump) must be a legal play
        
        # Test that player cannot play non-lead, non-trump cards
        self.assertNotIn(0, legal_plays)  # 2 of Spades should not be a legal play
        self.assertNotIn(2, legal_plays)  # Ace of Spades should not be a legal play

    def test_can_play_any_card_when_cant_follow_suit(self):
        """Test that a player can play any card when they can't follow the lead suit."""
        # Create a game instance
        game = FortyfivesGame()
        game.verbose = False
        
        # Set up a specific game state
        game.phase = 4  # Gameplay phase
        game.trick_lead_suit = 'C'  # Clubs is the lead suit
        game.trump_suit = 'H'  # Hearts is trump
        game.trick_starter = 0  # North led the trick
        
        # Set up a player who doesn't have any clubs (lead suit)
        player_hand = [
            create_card('2', 'S'),  # 2 of Spades - not lead suit, not trump
            create_card('Q', 'H'),  # Queen of Hearts - trump
            create_card('A', 'S'),  # Ace of Spades - not lead suit, not trump
            create_card('6', 'H'),  # 6 of Hearts - trump
        ]
        
        # Set player 2 (South) as current player
        game.current_player_id = 2
        game.hands = [[], [], [], []]  # Initialize empty hands
        game.hands[2] = player_hand    # Assign hand to South
        game.current_trick = [None, None, None, None]  # Initialize empty trick
        game.current_trick[0] = create_card('7', 'C')  # North led 7 of Clubs
        
        # Get legal plays for the player
        legal_plays = game.get_legal_plays()
        
        # Test that player can play any card when they can't follow suit
        self.assertEqual(len(legal_plays), len(player_hand), 
                        "Player should be able to play any card when they can't follow the lead suit")
        
        # Verify each card is a legal play
        for i in range(len(player_hand)):
            self.assertIn(i, legal_plays, f"Card at index {i} should be a legal play")

if __name__ == "__main__":
    unittest.main() 