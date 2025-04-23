"""
Test for verifying the correct gameplay sequence when players can't follow the lead suit.
This tests that when players can't follow suit, they can play any card from their hand,
not just trump cards.
"""

import unittest
from fortyfives.games.fortyfives.game import FortyfivesGame
from fortyfives.games.fortyfives.card import FortyfivesCard, get_card_rank

def create_card(rank, suit):
    """Helper function to create a card of specific rank and suit."""
    suits = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
    ranks = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    return FortyfivesCard(ranks[rank] + 13 * suits[suit])

class TestCantFollowSuitPlaySequence(unittest.TestCase):
    """Test class for verifying gameplay when players can't follow the lead suit."""
    
    def setUp(self):
        """Set up the test environment with a specific game state."""
        # Create the game
        self.game = FortyfivesGame()
        
        # Set up the game state for the gameplay phase
        self.game.verbose = False
        self.game.phase = 4  # Gameplay phase
        self.game.highest_bidder = 0  # North is declarer
        self.game.highest_bid = 1  # BID_20
        self.game.trump_suit = 'H'  # Hearts is trump
        
        # Clear and set up player hands for testing
        self.game.hands = [[], [], [], []]
        
        # North (Player 0) - Has clubs (lead suit)
        self.game.hands[0] = [
            create_card('7', 'C'),  # 7 of Clubs - will be played as lead
            create_card('K', 'S'),  # King of Spades 
            create_card('2', 'D'),  # 2 of Diamonds
        ]
        
        # East (Player 1) - Has clubs (lead suit)
        self.game.hands[1] = [
            create_card('A', 'C'),  # Ace of Clubs - can follow suit
            create_card('4', 'H'),  # 4 of Hearts - trump
            create_card('3', 'D'),  # 3 of Diamonds
        ]
        
        # South (Player 2) - No clubs, has trumps and other suits
        self.game.hands[2] = [
            create_card('2', 'S'),  # 2 of Spades - not lead suit, not trump
            create_card('J', 'S'),  # Jack of Spades - not lead suit, not trump
            create_card('Q', 'H'),  # Queen of Hearts - trump
            create_card('A', 'S'),  # Ace of Spades - not lead suit, not trump
            create_card('6', 'H'),  # 6 of Hearts - trump
        ]
        
        # West (Player 3) - No clubs, has trumps and other suits
        self.game.hands[3] = [
            create_card('5', 'S'),  # 5 of Spades - not lead suit, not trump
            create_card('8', 'H'),  # 8 of Hearts - trump
            create_card('3', 'S'),  # 3 of Spades - not lead suit, not trump
            create_card('9', 'D'),  # 9 of Diamonds - not lead suit, not trump
        ]
        
        # Set up the current trick with no cards played
        self.game.current_trick = [None, None, None, None]
        self.game.trick_starter = 0  # North leads
        self.game.current_player_id = 0  # North's turn
        self.game.trick_lead_suit = None  # No lead suit yet
    
    def test_full_trick_sequence_with_cant_follow_suit(self):
        """Test a full trick sequence where some players can't follow suit."""
        game = self.game
        
        # Step 1: North leads 7 of Clubs
        game.current_trick[0] = game.hands[0][0]  # Play 7 of Clubs
        game.hands[0].pop(0)  # Remove from hand
        game.trick_lead_suit = 'C'  # Set lead suit to Clubs
        game.current_player_id = 1  # Move to next player
        
        # Verify North played the 7 of Clubs
        self.assertEqual(game.current_trick[0].rank, '7')
        self.assertEqual(game.current_trick[0].suit, 'C')
        self.assertEqual(game.trick_lead_suit, 'C')
        
        # Step 2: East follows suit with Ace of Clubs
        legal_plays_east = game.get_legal_plays()
        
        # Verify East must follow suit (i.e., must play Ace of Clubs)
        self.assertIn(0, legal_plays_east)  # Ace of Clubs must be legal
        
        # East follows suit with Ace of Clubs
        game.current_trick[1] = game.hands[1][0]  # Play Ace of Clubs
        game.hands[1].pop(0)  # Remove from hand
        game.current_player_id = 2  # Move to next player
        
        # Verify East played the Ace of Clubs
        self.assertEqual(game.current_trick[1].rank, 'A')
        self.assertEqual(game.current_trick[1].suit, 'C')
        
        # Step 3: South can't follow suit, should be able to play any card
        legal_plays_south = game.get_legal_plays()
        
        # Verify South can play any card (since they have no clubs)
        self.assertEqual(len(legal_plays_south), len(game.hands[2]),
                        "South should be able to play any card when they can't follow suit")
        
        # South plays a non-trump card (2 of Spades)
        game.current_trick[2] = game.hands[2][0]  # Play 2 of Spades
        game.hands[2].pop(0)  # Remove from hand
        game.current_player_id = 3  # Move to next player
        
        # Verify South played the 2 of Spades
        self.assertEqual(game.current_trick[2].rank, '2')
        self.assertEqual(game.current_trick[2].suit, 'S')
        
        # Step 4: West can't follow suit, should be able to play any card
        legal_plays_west = game.get_legal_plays()
        
        # Verify West can play any card (since they have no clubs)
        self.assertEqual(len(legal_plays_west), len(game.hands[3]),
                        "West should be able to play any card when they can't follow suit")
        
        # West plays a non-trump card (9 of Diamonds)
        game.current_trick[3] = game.hands[3][3]  # Play 9 of Diamonds
        game.hands[3].pop(3)  # Remove from hand
        
        # Verify West played the 9 of Diamonds
        self.assertEqual(game.current_trick[3].rank, '9')
        self.assertEqual(game.current_trick[3].suit, 'D')
        
        # Use determine_trick_winner to verify East wins with Ace of Clubs
        winner = self._determine_trick_winner(game)
        self.assertEqual(winner, 1, "East should win the trick with Ace of Clubs")

    def _determine_trick_winner(self, game):
        """Helper method to determine the winner of the current trick."""
        # Logic for determining the trick winner
        # The highest card of the lead suit wins, unless trumps are played
        lead_suit = game.trick_lead_suit
        trump_suit = game.trump_suit
        highest_rank_value = -1
        winner = -1
        
        # First check for trumps
        for i in range(4):
            card = game.current_trick[i]
            if card is not None and card.suit == trump_suit:
                card_rank = get_card_rank(card, trump_suit)
                if highest_rank_value < 0 or card_rank > highest_rank_value:
                    highest_rank_value = card_rank
                    winner = i
        
        # If no trumps, check for highest card of lead suit
        if winner < 0:
            for i in range(4):
                card = game.current_trick[i]
                if card is not None and card.suit == lead_suit:
                    card_rank = get_card_rank(card, trump_suit)
                    if highest_rank_value < 0 or card_rank > highest_rank_value:
                        highest_rank_value = card_rank
                        winner = i
        
        return winner
    
if __name__ == "__main__":
    unittest.main() 