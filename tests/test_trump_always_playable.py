"""
Test script to verify that trump cards are always playable in Fortyfives game,
even when a player has cards of the lead suit.
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

def test_can_play_trump_and_lead_suit():
    """Test that a player can play either lead suit cards or trump cards when both are available."""
    # Create a game with controlled state
    game = FortyfivesGame.__new__(FortyfivesGame)
    game.get_state = lambda x: {}  # Dummy method
    
    # Initialize required attributes manually
    game.num_players = 4
    game.phase = 4  # Gameplay phase
    game.trump_suit = 'S'  # Spades is trump
    
    # Disable test_strict_suit_following to simulate normal gameplay
    game.test_strict_suit_following = False
    
    # Setup player's hand with both clubs (lead suit) and spades (trump)
    game.hands = [[], [], [], []]
    game.hands[1] = [
        create_card('A', 'C'),  # Ace of Clubs (lead suit)
        create_card('T', 'S'),  # Ten of Spades (trump)
        create_card('9', 'H'),  # Nine of Hearts
        create_card('5', 'H'),  # Five of Hearts
    ]
    
    # Setup the current trick with Clubs led
    game.current_trick = [None] * 4
    game.trick_starter = 0  # Player 0 leads
    game.current_trick[0] = create_card('7', 'C')  # 7 of Clubs led
    game.trick_lead_suit = 'C'
    
    # Set current player to Player 1
    game.current_player_id = 1
    
    # Get legal plays for player 1
    legal_plays = game.get_legal_plays()
    
    # Get diagnostic information
    lead_suit_cards = [i for i, card in enumerate(game.hands[1]) if card.suit == game.trick_lead_suit]
    trump_cards = [i for i, card in enumerate(game.hands[1]) 
                 if card.suit == game.trump_suit or (card.rank == 'A' and card.suit == 'H')]
    
    # Print debugging information
    print(f"Trump suit: {game.trump_suit}")
    print(f"Lead suit: {game.trick_lead_suit}")
    print(f"Player's hand: {[f'{card.rank}{card.suit}' for card in game.hands[1]]}")
    print(f"Lead suit cards indices: {lead_suit_cards}")
    print(f"Trump cards indices: {trump_cards}")
    print(f"Legal plays: {legal_plays}")
    print(f"Legal cards: {[game.hands[1][i].rank + game.hands[1][i].suit for i in legal_plays]}")
    
    # Verify player can play both AC (lead suit) and TS (trump)
    assert 0 in legal_plays, "AC (lead suit) should be a legal play"
    assert 1 in legal_plays, "TS (trump) should be a legal play"
    assert 2 not in legal_plays, "9H should not be a legal play"
    assert 3 not in legal_plays, "5H should not be a legal play"
    
    print("Test passed! Player can play both lead suit and trump cards.")

def test_reproduce_east_bug():
    """Test that reproduces the specific bug where East couldn't play TS (trump) in Step 27."""
    # Create a game with controlled state
    game = FortyfivesGame.__new__(FortyfivesGame)
    game.get_state = lambda x: {}  # Dummy method
    
    # Initialize required attributes manually
    game.num_players = 4
    game.phase = 4  # Gameplay phase
    game.trump_suit = 'S'  # Spades is trump
    
    # Disable test_strict_suit_following to simulate normal gameplay
    game.test_strict_suit_following = False
    
    # Setup East's hand to match Step 27
    game.hands = [[], [], [], []]
    game.hands[1] = [
        create_card('Q', 'D'),  # Queen of Diamonds
        create_card('T', 'S'),  # Ten of Spades (trump)
        create_card('9', 'H'),  # Nine of Hearts
        create_card('A', 'C'),  # Ace of Clubs (lead suit)
        create_card('5', 'H'),  # Five of Hearts
    ]
    
    # Setup trick as in Step 27: North led 7♣, South played Q♣, West played A♥
    game.current_trick = [None] * 4
    game.trick_starter = 0  # North led
    game.current_trick[0] = create_card('7', 'C')  # North led 7♣
    game.current_trick[2] = create_card('Q', 'C')  # South played Q♣
    game.current_trick[3] = create_card('A', 'H')  # West played A♥
    game.trick_lead_suit = 'C'  # Clubs is lead suit
    
    # Set current player to East (Player 1)
    game.current_player_id = 1
    
    # Get legal plays for East
    legal_plays = game.get_legal_plays()
    
    # Get diagnostic information
    lead_suit_cards = [i for i, card in enumerate(game.hands[1]) if card.suit == game.trick_lead_suit]
    trump_cards = [i for i, card in enumerate(game.hands[1]) 
                 if card.suit == game.trump_suit or (card.rank == 'A' and card.suit == 'H')]
    
    # Print debugging information
    print(f"Trump suit: {game.trump_suit}")
    print(f"Lead suit: {game.trick_lead_suit}")
    print(f"East's hand: {[f'{card.rank}{card.suit}' for card in game.hands[1]]}")
    print(f"Lead suit cards indices: {lead_suit_cards}")
    print(f"Trump cards indices: {trump_cards}")
    print(f"Current trick: North:{game.current_trick[0].rank}{game.current_trick[0].suit}, " + 
          f"South:{game.current_trick[2].rank}{game.current_trick[2].suit}, " +
          f"West:{game.current_trick[3].rank}{game.current_trick[3].suit}")
    print(f"Legal plays: {legal_plays}")
    print(f"Legal cards: {[game.hands[1][i].rank + game.hands[1][i].suit for i in legal_plays]}")
    
    # Verify East can play both AC (lead suit) and TS (trump)
    assert 3 in legal_plays, "AC (lead suit) should be a legal play"
    assert 1 in legal_plays, "TS (trump) should be a legal play"
    assert len(legal_plays) == 2, f"Expected exactly 2 legal plays, got {len(legal_plays)}"
    
    print("Test passed! East can play both lead suit and trump cards.")
    
def test_full_gameplay_with_known_trump():
    """Test that tracks the trump suit through a full gameplay scenario."""
    # Create a regular game
    game = FortyfivesGame()
    
    # Set up a specific trump suit for testing
    game.trump_suit = 'S'  # Spades is trump
    
    # Print the trump suit
    print(f"Initial trump suit: {game.trump_suit}")
    
    # Simulate some actions that might affect the trump suit
    game.phase = 2  # Declaration phase
    game.highest_bidder = 0
    game.process_declaration(0)  # Declare spades as trump
    
    # Print the trump suit after declaration
    print(f"Trump suit after declaration: {game.trump_suit}")
    
    # Enter gameplay phase
    game.phase = 4
    
    # Print the trump suit in gameplay phase 
    print(f"Trump suit in gameplay phase: {game.trump_suit}")
    
    # Verify trump suit is still spades
    assert game.trump_suit == 'S', "Trump suit should still be Spades"

if __name__ == "__main__":
    print("Testing that players can play both lead suit and trump cards...")
    test_can_play_trump_and_lead_suit()
    
    print("\nTesting reproduction of the specific bug where East couldn't play trump in Step 27...")
    test_reproduce_east_bug()
    
    print("\nTesting trump suit consistency through game phases...")
    test_full_gameplay_with_known_trump()
    
    print("\nAll tests passed!") 