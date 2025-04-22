from fortyfives.games.fortyfives.game import FortyfivesGame, PHASE_GAMEPLAY
from fortyfives.games.fortyfives.card import FortyfivesCard as Card, RANKS, SUITS

def create_card(rank, suit):
    card_id = RANKS.index(rank) + SUITS.index(suit) * 13
    return Card(card_id)

def main():
    game = FortyfivesGame()
    
    # Initialize manually
    game.phase = PHASE_GAMEPLAY
    
    # Setup player hands with cards
    game.hands = [[] for _ in range(4)]
    
    # Player 0 has diamonds and spades
    game.hands[0] = [
        create_card('A', 'D'), 
        create_card('J', 'D'), 
        create_card('9', 'S'), 
        create_card('2', 'S')
    ]
    
    # Player 1 has diamonds, hearts and spades
    game.hands[1] = [
        create_card('Q', 'D'),  # Must be playable when diamonds led
        create_card('6', 'D'),  # Must be playable when diamonds led
        create_card('A', 'H'),  # Special case: A♥ can be played anytime (high trump)
        create_card('K', 'S')   # Must not be playable when diamonds led and player has diamonds
    ]
    
    # Set trump suit
    game.trump_suit = 'D'
    
    # Set up a trick with a diamond lead
    game.trick_starter = 2  # Player 2 leads
    game.current_trick = [None] * 4
    game.current_trick[2] = create_card('5', 'D')  # 5 of diamonds led
    game.trick_lead_suit = 'D'
    
    # Test player 0
    game.current_player_id = 0
    legal_plays_0 = game.get_legal_plays()
    
    # Test player 1 
    game.current_player_id = 1
    legal_plays_1 = game.get_legal_plays()
    
    # Print results
    print("Player 0:")
    print(f'Hand: {[str(card) for card in game.hands[0]]}')
    print(f'Legal plays: {legal_plays_0}')
    print(f'Legal cards: {[str(game.hands[0][i]) for i in legal_plays_0]}')
    
    print("\nPlayer 1:")
    print(f'Hand: {[str(card) for card in game.hands[1]]}')
    print(f'Legal plays: {legal_plays_1}')
    print(f'Legal cards: {[str(game.hands[1][i]) for i in legal_plays_1]}')
    
if __name__ == "__main__":
    main() 