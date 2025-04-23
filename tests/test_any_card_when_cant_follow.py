"""
Test for verifying that a player can play any card when they cannot follow the lead suit.
This test validates that when a player doesn't have any cards of the lead suit,
they should be able to play any card (not just trump cards).
"""

from fortyfives.games.fortyfives.game import FortyfivesGame
from fortyfives.games.fortyfives.card import FortyfivesCard

def create_card(rank, suit):
    """Helper function to create a card of specific rank and suit."""
    suits = {'S': 0, 'H': 1, 'C': 2, 'D': 3}
    ranks = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    return FortyfivesCard(ranks[rank] + 13 * suits[suit])

def test_any_card_when_cant_follow_suit():
    """
    Test that when a player cannot follow the lead suit, they can play any card,
    even if they have trump cards.
    """
    # Create the game and monkey patch the get_state method to avoid initialization errors
    game = FortyfivesGame.__new__(FortyfivesGame)
    game.get_state = lambda x: {}  # Dummy method
    
    # Initialize required attributes manually
    game.num_players = 4
    game.phase = 4  # Gameplay phase
    game.trick_starter = 1  # East starts the trick
    game.trump_suit = 'H'  # Hearts is trump
    
    # Make sure test_strict_suit_following is not enabled
    game.test_strict_suit_following = False
    game.verbose = True  # Enable debugging output
    
    # Clear hands and add specific test cards
    game.hands = [[], [], [], []]
    
    # South (player 2) has Hearts (trump) and Spades, but no Clubs (lead suit)
    game.hands[2] = [
        create_card('2', 'S'),  # 2 of Spades - should be playable
        create_card('J', 'S'),  # Jack of Spades - should be playable
        create_card('Q', 'H'),  # Queen of Hearts - trump, should be playable
        create_card('A', 'S'),  # Ace of Spades - should be playable
        create_card('6', 'H')   # 6 of Hearts - trump, should be playable
    ]
    
    # Setup the current trick with a Club lead by East
    game.current_trick = [None] * 4
    game.current_trick[1] = create_card('9', 'C')  # East leads 9 of Clubs
    game.trick_lead_suit = 'C'
    
    # Test South's legal plays - should be able to play any card
    game.current_player_id = 2
    legal_plays = game.get_legal_plays()
    
    # Print the results
    print(f"South's hand: {[f'{card.rank}{card.suit}' for card in game.hands[2]]}")
    print(f"Current trick: East: {game.current_trick[1].rank}{game.current_trick[1].suit}")
    print(f"Legal actions:")
    for i in legal_plays:
        print(f"  {i}: Play: {game.hands[2][i].rank}{game.hands[2][i].suit}")
    
    # Verify that all cards are legal when the player can't follow suit
    assert len(legal_plays) == len(game.hands[2]), "Player should be able to play any card when they can't follow suit"
    
    # Check specific indices
    for i in range(len(game.hands[2])):
        assert i in legal_plays, f"Card at index {i} should be a legal play"
    
    print("Test passed: Player can play any card when they cannot follow suit, even if they have trump cards!")

if __name__ == "__main__":
    test_any_card_when_cant_follow_suit() 