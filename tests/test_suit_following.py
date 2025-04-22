"""
Test script for verifying the suit-following logic in the Fortyfives game.
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

def test_suit_following():
    """Test that players must follow suit when able."""
    # Create the game and monkey patch the get_state method to avoid initialization errors
    game = FortyfivesGame.__new__(FortyfivesGame)
    game.get_state = lambda x: {}  # Dummy method
    
    # Initialize required attributes manually
    game.num_players = 4
    game.phase = 4  # Gameplay phase
    game.trick_starter = 2  # Player 2 starts the trick
    game.trump_suit = 'D'
    
    # Enable strict suit following for this test
    game.test_strict_suit_following = True
    
    # Clear hands and add specific test cards
    game.hands = [[], [], [], []]
    
    # Player 0 has cards of different suits including diamonds
    game.hands[0] = [
        create_card('A', 'D'),  # Must be playable when diamonds led (idx 0)
        create_card('4', 'D'),  # Must be playable when diamonds led (idx 1)
        create_card('9', 'S'),  # Must not be playable when diamonds led and player has diamonds (idx 2)
        create_card('2', 'S')   # Must not be playable when diamonds led and player has diamonds (idx 3)
    ]
    
    # Setup the current trick with a diamond lead
    game.current_trick = [None] * 4
    game.current_trick[2] = create_card('3', 'D')  # Player 2 leads 3 of diamonds (a low trump)
    game.trick_lead_suit = 'D'
    
    # Test if player 0 can only play diamonds (should return indices 0 and 1)
    game.current_player_id = 0
    legal_plays = game.get_legal_plays()
    
    # Print the results
    print(f"Player 0's hand: {[f'{card.rank}{card.suit}' for card in game.hands[0]]}")
    print(f"Lead suit: {game.trick_lead_suit}")
    print(f"Legal plays: {legal_plays}")
    
    # Verify if the player must follow suit
    assert len(legal_plays) == 2, f"Expected 2 legal plays, got {len(legal_plays)}"
    assert 0 in legal_plays, "AD should be a legal play for Player 0"
    assert 1 in legal_plays, "4D should be a legal play for Player 0"
    assert 2 not in legal_plays, "9S should not be a legal play for Player 0"
    assert 3 not in legal_plays, "2S should not be a legal play for Player 0"
    
    print("Suit following logic is working correctly for diamonds!")

def test_high_trump_exemption():
    """Test that high trumps can be withheld."""
    # Create the game and monkey patch the get_state method to avoid initialization errors
    game = FortyfivesGame.__new__(FortyfivesGame)
    game.get_state = lambda x: {}  # Dummy method
    
    # Initialize required attributes manually
    game.num_players = 4
    game.phase = 4  # Gameplay phase
    game.trick_starter = 2  # Player 2 starts the trick
    game.trump_suit = 'D'
    
    # Enable strict suit following for this test
    game.test_strict_suit_following = True
    
    # Clear hands and add specific test cards
    game.hands = [[], [], [], []]
    
    # Player 0 has cards including J of Diamonds (a high trump)
    game.hands[0] = [
        create_card('J', 'D'),  # High trump that can be withheld (idx 0)
        create_card('5', 'D'),  # High trump that can be withheld (idx 1)
        create_card('9', 'S'),  # Non-trump (idx 2)
        create_card('A', 'H')   # A of Hearts is also a high trump that can be withheld (idx 3)
    ]
    
    # Setup the current trick with a low diamond lead
    game.current_trick = [None] * 4
    game.current_trick[2] = create_card('3', 'D')  # Player 2 leads 3 of diamonds (a low trump)
    game.trick_lead_suit = 'D'
    
    # Test if player 0 must follow suit but can withhold high trumps
    game.current_player_id = 0
    legal_plays = game.get_legal_plays()
    
    # Print the results
    print(f"Player 0's hand: {[f'{card.rank}{card.suit}' for card in game.hands[0]]}")
    print(f"Lead suit: {game.trick_lead_suit}")
    print(f"Legal plays: {legal_plays}")
    
    # Verify if the player can play any card when they only have high trumps
    assert len(legal_plays) == 4, f"Expected 4 legal plays, got {len(legal_plays)}"
    
    print("High trump exemption is working correctly!")

def test_no_suit_restriction():
    """Test that when a player cannot follow suit, they can play any card."""
    # Create the game and monkey patch the get_state method to avoid initialization errors
    game = FortyfivesGame.__new__(FortyfivesGame)
    game.get_state = lambda x: {}  # Dummy method
    
    # Initialize required attributes manually
    game.num_players = 4
    game.phase = 4  # Gameplay phase
    game.trick_starter = 2  # Player 2 starts the trick
    game.trump_suit = 'D'
    
    # Clear hands and add specific test cards
    game.hands = [[], [], [], []]
    
    # Player 0 has no diamonds
    game.hands[0] = [
        create_card('A', 'C'),  # Should be playable when diamonds led (idx 0)
        create_card('9', 'S'),  # Should be playable when diamonds led (idx 1)
        create_card('J', 'H'),  # Should be playable when diamonds led (idx 2)
        create_card('2', 'S')   # Should be playable when diamonds led (idx 3)
    ]
    
    # Setup the current trick with a diamond lead
    game.current_trick = [None] * 4
    game.current_trick[2] = create_card('5', 'D')  # Player 2 leads 5 of diamonds
    game.trick_lead_suit = 'D'
    
    # Test if player 0 can play any card
    game.current_player_id = 0
    legal_plays = game.get_legal_plays()
    
    # Print the results
    print(f"Player 0's hand (no diamonds): {[f'{card.rank}{card.suit}' for card in game.hands[0]]}")
    print(f"Lead suit: {game.trick_lead_suit}")
    print(f"Legal plays: {legal_plays}")
    
    # Verify that all cards are legal when the player can't follow suit
    assert len(legal_plays) == len(game.hands[0]), "Player should be able to play any card when they can't follow suit"
    
    print("No suit restriction logic is working correctly!")

def test_can_always_play_trump():
    """Test that players can always play trump cards, even when they have cards of the lead suit."""
    # Create the game and monkey patch the get_state method to avoid initialization errors
    game = FortyfivesGame.__new__(FortyfivesGame)
    game.get_state = lambda x: {}  # Dummy method
    
    # Initialize required attributes manually
    game.num_players = 4
    game.phase = 4  # Gameplay phase
    game.trick_starter = 2  # Player 2 starts the trick
    game.trump_suit = 'C'  # Clubs is trump
    
    # Disable test_strict_suit_following to test normal gameplay rules
    game.test_strict_suit_following = False
    
    # Clear hands and add specific test cards
    game.hands = [[], [], [], []]
    
    # Player 0 has both spades and clubs (trump)
    game.hands[0] = [
        create_card('8', 'S'),  # 8 of Spades (idx 0) - matching lead suit
        create_card('K', 'C'),  # King of Clubs (idx 1) - trump
        create_card('7', 'H'),  # 7 of Hearts (idx 2)
        create_card('9', 'D')   # 9 of Diamonds (idx 3)
    ]
    
    # Setup the current trick with a spade lead
    game.current_trick = [None] * 4
    game.current_trick[2] = create_card('J', 'S')  # Player 2 leads Jack of Spades
    game.trick_lead_suit = 'S'
    
    # Test if player 0 can play both spades (follow suit) and clubs (trump)
    game.current_player_id = 0
    legal_plays = game.get_legal_plays()
    
    # Print the results
    print(f"Player 0's hand: {[f'{card.rank}{card.suit}' for card in game.hands[0]]}")
    print(f"Lead suit: {game.trick_lead_suit}, Trump suit: {game.trump_suit}")
    print(f"Legal plays: {legal_plays}")
    
    # Verify that player can play both spades (lead suit) and clubs (trump)
    assert 0 in legal_plays, "8S (lead suit) should be a legal play"
    assert 1 in legal_plays, "KC (trump) should be a legal play"
    assert 2 not in legal_plays, "7H should not be a legal play (not lead suit or trump)"
    assert 3 not in legal_plays, "9D should not be a legal play (not lead suit or trump)"
    
    print("Trump card playability is working correctly!")

if __name__ == "__main__":
    print("Testing suit following logic...")
    test_suit_following()
    print("\nTesting high trump exemption...")
    test_high_trump_exemption()
    print("\nTesting no suit restriction logic...")
    test_no_suit_restriction()
    print("\nTesting trump card playability...")
    test_can_always_play_trump()
    print("\nAll tests passed!") 