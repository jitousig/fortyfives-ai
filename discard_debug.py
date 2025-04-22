from fortyfives.games.fortyfives.game import (
    FortyfivesGame, 
    PHASE_DISCARD, PHASE_GAMEPLAY, DISCARD_DONE
)
from fortyfives.games.fortyfives.card import FortyfivesCard as Card, RANKS, SUITS

def create_card(rank, suit):
    card_id = RANKS.index(rank) + SUITS.index(suit) * 13
    return Card(card_id)

def main():
    # Create a game instance
    game = FortyfivesGame()
    game.verbose = True
    
    # Setup the game state
    game.phase = PHASE_DISCARD
    game.hands = [
        [create_card('A', 'S')], 
        [create_card('A', 'H')], 
        [create_card('A', 'D')], 
        [create_card('A', 'C')]
    ]
    game.auction_starting_player = 0
    game.current_player_id = 0
    game.dealer_id = 3
    game.dealer = game.dealer.__class__(game.np_random)
    
    # Process discard for each player
    print(f'Initial phase: {game.phase}')
    
    game.process_discard(DISCARD_DONE)
    print(f'Phase after player 0 discard: {game.phase}, Current player: {game.current_player_id}')
    
    game.process_discard(DISCARD_DONE)
    print(f'Phase after player 1 discard: {game.phase}, Current player: {game.current_player_id}')
    
    game.process_discard(DISCARD_DONE)
    print(f'Phase after player 2 discard: {game.phase}, Current player: {game.current_player_id}')
    
    game.process_discard(DISCARD_DONE)
    print(f'Phase after player 3 discard: {game.phase}, Current player: {game.current_player_id}')

if __name__ == "__main__":
    main() 