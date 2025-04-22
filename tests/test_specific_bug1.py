"""
Test script for verifying the specific bug with West not being able to play K♣ as trump.
This tests the issue found in the gameplay review at Step 25.
"""

import sys
import os
# Add the parent directory to the path so we can import the fortyfives package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fortyfives.games.fortyfives.game import FortyfivesGame
from fortyfives.games.fortyfives.card import FortyfivesCard as Card, RANKS, SUITS

# Helper function to create a card with specific rank and suit
def create_card(rank, suit):
    card_id = RANKS.index(rank) + SUITS.index(suit) * 13
    return Card(card_id)

def test_bug1_west_can_play_trump():
    """Test the specific scenario where West (Player 3) should be able to play K♣ as trump when J♠ is led."""
    # Create the game and monkey patch the get_state method to avoid initialization errors
    game = FortyfivesGame.__new__(FortyfivesGame)
    game.get_state = lambda x: {}  # Dummy method
    
    # Initialize required attributes manually
    game.num_players = 4
    game.phase = 4  # Gameplay phase
    game.trump_suit = 'C'  # Clubs is trump
    
    # Disable test strict suit following to simulate normal gameplay
    game.test_strict_suit_following = False
    
    # Setup West's hand to match the scenario in Step 25
    game.hands = [[], [], [], []]
    game.hands[3] = [
        create_card('8', 'S'),  # 8 of Spades
        create_card('7', 'H'),  # 7 of Hearts
        create_card('K', 'C'),  # King of Clubs (trump)
        create_card('9', 'D'),  # 9 of Diamonds
        create_card('3', 'H')   # 3 of Hearts
    ]
    
    # Setup the current trick with South leading J♠
    game.current_trick = [None] * 4
    game.trick_starter = 2       # South is trick starter
    game.current_trick[2] = create_card('J', 'S')  # South leads J♠
    game.trick_lead_suit = 'S'   # Spades is the lead suit
    
    # Set current player to West (Player 3)
    game.current_player_id = 3
    
    # Get legal plays for West
    legal_plays = game.get_legal_plays()
    
    # Print the results for debugging
    print(f"West's hand: {[f'{card.rank}{card.suit}' for card in game.hands[3]]}")
    print(f"Lead suit: {game.trick_lead_suit}, Trump suit: {game.trump_suit}")
    print(f"Legal plays: {legal_plays}")
    print(f"Legal cards: {[game.hands[3][i].rank + game.hands[3][i].suit for i in legal_plays]}")
    
    # Verify that West can play both 8♠ (match lead suit) and K♣ (trump)
    assert 0 in legal_plays, "8S (matching lead suit) should be a legal play"
    assert 2 in legal_plays, "KC (trump) should be a legal play"
    
    # Make sure only those two cards are legal plays (not 7H, 9D, or 3H)
    assert len(legal_plays) == 2, f"Expected exactly 2 legal plays, got {len(legal_plays)}"
    assert 1 not in legal_plays, "7H should not be a legal play"
    assert 3 not in legal_plays, "9D should not be a legal play"
    assert 4 not in legal_plays, "3H should not be a legal play"
    
    print("Bug fix verified! West can play both the matching suit card and the trump card.")

if __name__ == "__main__":
    print("Testing the specific bug fix for West being able to play K♣ as trump...")
    test_bug1_west_can_play_trump()
    print("\nBug fix test passed!") 