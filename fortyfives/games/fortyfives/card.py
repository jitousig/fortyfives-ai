'''
Card utilities for Fortyfives game
'''

import numpy as np

# Define card ranks and suits
# Ranks from low to high
# For display purposes: A - Ace, T - 10, J - Jack, Q - Queen, K - King
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['S', 'H', 'D', 'C']  # Spades, Hearts, Diamonds, Clubs

# Points each card is worth
RANK_POINTS = {
    '5': 5,
    'J': 1,
    'A': 1,
    'K': 0,
    'Q': 0,
    'T': 0,
    '9': 0,
    '8': 0,
    '7': 0,
    '6': 0,
    '4': 0,
    '3': 0,
    '2': 0,
}

class FortyfivesCard:
    '''
    Card class for Fortyfives game
    '''
    
    def __init__(self, card_id):
        '''
        Initialize a card with card id
        
        Args:
            card_id (int): the id of a card, from 0 to 51
                           Card with id x has rank at x % 13 and suit at x // 13
                           For example, card with id 0 is the 2 of Spades
        '''
        self.id = card_id
        self.rank = RANKS[card_id % 13]
        self.suit = SUITS[card_id // 13]
        
    def get_index(self):
        '''
        Get the index of the card
        
        Returns:
            (int): the index of the card
        '''
        return self.id
    
    @staticmethod
    def rank_to_id(rank):
        '''
        Convert rank string to its id
        
        Args:
            rank (str): rank of the card, e.g., '2', '3', ..., 'A'
            
        Returns:
            (int): the id of the rank
        '''
        return RANKS.index(rank)
    
    @staticmethod
    def suit_to_id(suit):
        '''
        Convert suit string to its id
        
        Args:
            suit (str): suit of the card, e.g., 'S', 'H', 'D', 'C'
            
        Returns:
            (int): the id of the suit
        '''
        return SUITS.index(suit)
    
    @staticmethod
    def is_red_suit(suit):
        '''
        Check if a suit is red
        
        Args:
            suit (str): suit of the card, e.g., 'S', 'H', 'D', 'C'
            
        Returns:
            (bool): True if the suit is red (Hearts or Diamonds)
        '''
        return suit in ['H', 'D']
    
    def __str__(self):
        return self.rank + self.suit
    
    def __repr__(self):
        return self.rank + self.suit

def init_deck():
    '''
    Initialize a standard 52-card deck
    
    Returns:
        (list): A list of FortyfivesCard objects
    '''
    cards = []
    for card_id in range(52):
        cards.append(FortyfivesCard(card_id))
    return cards

def get_card_rank(card, trump_suit):
    '''
    Get the rank value of a card in Fortyfives
    Card rank depends on whether it's trump and if the suit is red or black
    
    Args:
        card (FortyfivesCard): The card to evaluate
        trump_suit (str): The current trump suit 'S', 'H', 'D', or 'C'
        
    Returns:
        (int): The rank value, higher value means stronger card
    '''
    # Special case: Ace of Hearts is always trump
    if card.rank == 'A' and card.suit == 'H':
        return 1001  # Always third highest trump (changed from 1002)
    
    # Check if card is trump
    if card.suit == trump_suit:
        # Special trump ranking
        rank_map = {
            '5': 1003,  # Highest trump
            'J': 1002,  # Second highest trump
            'A': 1000,  # A of Hearts handled above, this is for the other A
            'K': 999,
            'Q': 998
        }
        
        if card.rank in rank_map:
            return rank_map[card.rank]
        
        # For other trump cards
        if FortyfivesCard.is_red_suit(card.suit):
            # Red trump: 5, J, AH, A, K, Q, 10, 9, 8, 7, 6, 4, 3, 2
            base = {'T': 997, '9': 996, '8': 995, '7': 994, '6': 993, '4': 992, '3': 991, '2': 990}
            return base.get(card.rank, 989)  # Default low value if something is wrong
        else:
            # Black trump: 5, J, AH, A, K, Q, 2, 3, 4, 6, 7, 8, 9, 10
            base = {'2': 997, '3': 996, '4': 995, '6': 994, '7': 993, '8': 992, '9': 991, 'T': 990}
            return base.get(card.rank, 989)  # Default low value if something is wrong
    
    # Non-trump cards
    if FortyfivesCard.is_red_suit(card.suit):
        # Red non-trump: K, Q, J, 10, 9, 8, 7, 6, 4, 3, 2, A
        rank_map = {
            'K': 900, 'Q': 899, 'J': 898, 'T': 897, '9': 896, '8': 895,
            '7': 894, '6': 893, '4': 892, '3': 891, '2': 890, 'A': 889
        }
        return rank_map.get(card.rank, 0)
    else:
        # Black non-trump: K, Q, J, A, 2, 3, 4, 5, 6, 7, 8, 9, 10
        rank_map = {
            'K': 800, 'Q': 799, 'J': 798, 'A': 797, '2': 796, '3': 795,
            '4': 794, '5': 793, '6': 792, '7': 791, '8': 790, '9': 789, 'T': 788
        }
        return rank_map.get(card.rank, 0) 