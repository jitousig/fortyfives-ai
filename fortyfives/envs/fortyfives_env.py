'''
Fortyfives environment
'''

import numpy as np

from rlcard.envs import Env
from fortyfives.games.fortyfives.game import FortyfivesGame
from fortyfives.games.fortyfives.card import SUITS, RANKS

# Define action mappings
# Actions: [bid actions] + [trump declaration] + [play card]
# Bid actions: Pass (0), Bid 20 (1), Bid 25 (2), Bid 30 (3), Hold (4)
# Trump declaration: Spades (5), Hearts (6), Diamonds (7), Clubs (8)
# Play card: 9 (first card in hand) to 9+n (last card in hand)

class FortyfivesEnv(Env):
    '''
    Fortyfives Environment
    '''
    
    def __init__(self, config=None):
        '''
        Initialize the environment
        
        Args:
            config (dict): Environment configuration
        '''
        self.name = 'fortyfives'
        self.game = FortyfivesGame()
        super().__init__(config)
        
        # State shape and action shape
        # One-hot encoded cards (52*4) + phase (5) + bids (4*4) + points (2) + tricks (2) + dealer/current_player (2)
        # Update: increase space to 52*5 to accommodate all possible card indices
        self.state_shape = [52 * 5 + 5 + 4 * 4 + 2 + 2 + 2]  
        self.action_shape = []  # No additional action features
        
    def _extract_state(self, state):
        '''
        Extract state information from the game state
        
        Args:
            state (dict): State from the game

        Returns:
            (dict): State for RLcard environment
        '''
        extracted_state = {}
        extracted_state['obs'] = self._get_observation(state)
        extracted_state['legal_actions'] = self._get_legal_actions(state)
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = self._get_raw_legal_actions(state)
        return extracted_state
    
    def _get_observation(self, state):
        '''
        Get observation for a player
        
        Args:
            state (dict): State from the game

        Returns:
            (numpy.array): Observation as a numpy array
        '''
        # Initialize observation array
        obs = np.zeros(self.state_shape[0], dtype=int)
        
        # Encode player's hand
        self._encode_cards(obs, state['hand'], 0)
        
        # Encode trick cards
        for i, card in enumerate(state['trick']):
            if card is not None:
                self._encode_cards(obs, [card], 52 * (i + 1))
        
        # Encode phase
        phase_idx = 52 * 4 + state['phase']
        obs[phase_idx] = 1
        
        # Encode bids
        bid_info = state['bid_info']
        if bid_info['bids'] is not None:
            for i, bid in enumerate(bid_info['bids']):
                if bid is not None:
                    obs[52 * 4 + 5 + i * 4 + bid] = 1
        
        # Encode points
        points = state['points']
        if points is not None:
            obs[52 * 4 + 5 + 16] = points[0]
            obs[52 * 4 + 5 + 17] = points[1]
        
        # Encode tricks won
        tricks_won = state['tricks_won']
        if tricks_won is not None:
            obs[52 * 4 + 5 + 18] = tricks_won[0] + tricks_won[2]  # N/S tricks
            obs[52 * 4 + 5 + 19] = tricks_won[1] + tricks_won[3]  # E/W tricks
        
        # Encode dealer and current player
        obs[52 * 4 + 5 + 20] = state['dealer']
        obs[52 * 4 + 5 + 21] = state['current_player']
            
        return obs
    
    def _encode_cards(self, obs, cards, start_idx):
        '''
        Encode cards as one-hot vectors in observation
        
        Args:
            obs (numpy.array): Observation array
            cards (list): List of Card objects
            start_idx (int): Start index in observation array
        '''
        for card in cards:
            card_idx = RANKS.index(card.rank) + 13 * SUITS.index(card.suit)
            obs[start_idx + card_idx] = 1
    
    def _get_legal_actions(self, state):
        '''
        Get legal actions in a format usable by the RL agent
        
        Args:
            state (dict): State from the game

        Returns:
            (dict): Legal actions
        '''
        legal_actions = {}
        
        for action in state['legal_actions']:
            legal_actions[action] = None
            
        return legal_actions
    
    def _get_raw_legal_actions(self, state):
        '''
        Get raw legal actions for the current state
        
        Args:
            state (dict): State from the game

        Returns:
            (list): Raw legal actions
        '''
        return state['legal_actions']
    
    def get_payoffs(self):
        '''
        Get the payoffs for all players
        
        Returns:
            (list): Payoffs for all players
        '''
        winners = [p for p, score in enumerate(self.game.points) if score >= 125]
        
        # If the game is over, the winning partnership gets 1, the losing gets -1
        if winners:
            payoffs = [0, 0, 0, 0]
            if 0 in winners:  # N/S won
                payoffs[0] = 1
                payoffs[2] = 1
                payoffs[1] = -1
                payoffs[3] = -1
            else:  # E/W won
                payoffs[0] = -1
                payoffs[2] = -1
                payoffs[1] = 1
                payoffs[3] = 1
        else:
            # Game isn't over yet, so normalize current scores
            payoffs = [0, 0, 0, 0]
            ns_score = self.game.points[0] / 125  # Normalize to [0, 1]
            ew_score = self.game.points[1] / 125  # Normalize to [0, 1]
            payoffs[0] = ns_score
            payoffs[2] = ns_score
            payoffs[1] = ew_score
            payoffs[3] = ew_score
            
        return payoffs
    
    def _decode_action(self, action_id):
        '''
        Decode action id to a game action
        
        Args:
            action_id (int): Action id

        Returns:
            (int or tuple): Game action
        '''
        return action_id
    
    def _get_action_num(self):
        '''
        Get the number of possible actions
        
        Returns:
            (int): Number of possible actions
        '''
        # Bid actions (5) + Trump declaration (4) + Max cards in hand (8) + Done discarding (1)
        # 5 + 4 + 8 + 1 = 18
        return 18
    
    def seed(self, seed=None):
        '''
        Set the seed for the environment
        
        Args:
            seed (int): Seed for random number generation

        Returns:
            (list): List containing the seed
        '''
        self.np_random = np.random.RandomState(seed)
        self.game.np_random = self.np_random
        return [seed]
    
    def get_perfect_information(self):
        '''
        Get perfect information of the environment (all hands, etc.)
        
        Returns:
            (dict): Perfect information
        '''
        state = {}
        state['hands'] = self.game.hands
        state['current_trick'] = self.game.current_trick
        state['trump_suit'] = self.game.trump_suit
        state['points'] = self.game.points
        state['tricks_won'] = self.game.tricks_won
        state['highest_bidder'] = self.game.highest_bidder
        state['highest_bid'] = self.game.highest_bid
        state['bids'] = self.game.bids
        state['phase'] = self.game.phase
        
        return state 