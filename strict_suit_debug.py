from fortyfives.games.fortyfives.game import FortyfivesGame, PHASE_GAMEPLAY
from fortyfives.games.fortyfives.card import FortyfivesCard as Card, RANKS, SUITS

def create_card(rank, suit):
    card_id = RANKS.index(rank) + SUITS.index(suit) * 13
    return Card(card_id)

def main():
    game = FortyfivesGame()
    
    # Initialize manually
    game.phase = PHASE_GAMEPLAY
    game.trump_suit = 'D'
    
    # Setup player hands with cards
    game.hands = [[] for _ in range(4)]
    
    # Player 1 has diamonds, hearts and spades 
    game.hands[1] = [
        create_card('Q', 'D'),  # Must be playable when diamonds led
        create_card('6', 'D'),  # Must be playable when diamonds led
        create_card('A', 'H'),  # Special case: A♥ can be played anytime (high trump)
        create_card('K', 'S')   # Must not be playable when diamonds led and player has diamonds
    ]
    
    # Set up a trick with a diamond lead
    game.trick_starter = 2  # Player 2 leads
    game.current_trick = [None] * 4
    game.current_trick[2] = create_card('5', 'D')  # 5 of diamonds led
    game.trick_lead_suit = 'D'
    
    # Test player 1 with strict suit following
    game.current_player_id = 1
    
    # Test with strict suit following
    game.test_strict_suit_following = True
    strict_legal_plays = game.get_legal_plays()
    
    # Test without strict suit following
    delattr(game, 'test_strict_suit_following')
    normal_legal_plays = game.get_legal_plays()
    
    # Print results
    print(f"Player 1 hand: {[str(card) for card in game.hands[1]]}")
    print(f"Trump suit: {game.trump_suit}")
    print(f"Lead suit: {game.trick_lead_suit}")
    
    print("\nWith strict suit following:")
    print(f"Legal plays: {strict_legal_plays}")
    print(f"Legal cards: {[str(game.hands[1][i]) for i in strict_legal_plays]}")
    
    print("\nWithout strict suit following:")
    print(f"Legal plays: {normal_legal_plays}")
    print(f"Legal cards: {[str(game.hands[1][i]) for i in normal_legal_plays]}")
    
if __name__ == "__main__":
    main() 