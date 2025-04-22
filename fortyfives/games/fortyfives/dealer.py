'''
Dealer module for Fortyfives game
'''

import random
from fortyfives.games.fortyfives.card import init_deck

class FortyfivesDealer:
    '''
    Dealer class for Fortyfives game
    '''
    
    def __init__(self, np_random=None):
        '''
        Initialize a dealer
        
        Args:
            np_random (numpy.random.RandomState): NumPy random generator
        '''
        self.np_random = np_random
        self.deck = init_deck()
        self.shuffle()
        self.pot = [] # Cards in the kitty/pot
        
    def shuffle(self):
        '''
        Shuffle the deck
        '''
        if self.np_random is not None:
            self.np_random.shuffle(self.deck)
        else:
            random.shuffle(self.deck)
            
    def deal_card(self, player_id):
        '''
        Deal a card to a player
        
        Args:
            player_id (int): The ID of the player to be dealt a card
        
        Returns:
            (FortyfivesCard): The card dealt
        '''
        return self.deck.pop()
    
    def deal_cards(self, n_players, num_cards_per_player):
        '''
        Deal cards to players at the beginning of a game
        
        Args:
            n_players (int): Number of players in the game
            num_cards_per_player (int): Number of cards to deal to each player
            
        Returns:
            (dict): A dictionary where the keys are player IDs and the values are
                  lists of FortyfivesCard objects
        '''
        hands = [[] for _ in range(n_players)]
        
        # Deal cards to players (clockwise, one at a time)
        for _ in range(num_cards_per_player):
            for player_id in range(n_players):
                hands[player_id].append(self.deck.pop())
        
        # Deal kitty (3 cards)
        self.pot = [self.deck.pop() for _ in range(3)]
        
        return hands
    
    def get_pot(self):
        '''
        Get the cards in the kitty/pot
        
        Returns:
            (list): A list of FortyfivesCard objects in the pot
        '''
        return self.pot
    
    def add_pot_to_hand(self, hand):
        '''
        Add the pot's cards to a player's hand (used when a player wins the bid)
        
        Args:
            hand (list): The player's hand
            
        Returns:
            (list): The updated hand with pot cards added
        '''
        hand.extend(self.pot)
        self.pot = []
        return hand 