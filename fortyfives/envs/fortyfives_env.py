'''
Fortyfives environment
'''

import numpy as np

from rlcard.envs import Env
from fortyfives.games.fortyfives.game import (
    FortyfivesGame,
    PHASE_AUCTION, PHASE_DECLARATION, PHASE_DISCARD, PHASE_GAMEPLAY,
    DISCARD_DONE,
)
from fortyfives.games.fortyfives.card import SUITS, RANKS

# Env action ID layout (18 total, non-overlapping across all phases):
# Bid actions:        0=PASS, 1=BID_20, 2=BID_25, 3=BID_30, 4=HOLD
# Trump declaration:  5=SPADES, 6=HEARTS, 7=DIAMONDS, 8=CLUBS
# Card play/discard:  9=card0 … 16=card7
# Discard done:       17

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
        # One-hot cards (52*5) + phase (5) + bids (4*4) + points (2) + tricks (2)
        # + dealer/current scalars (2) + trump suit (4) + lead suit (4)  [=295]
        # then APPENDED (indices 295+, never disturbing the above):
        #   + cards played in completed tricks, multi-hot (52)  [295..346]
        #   + highest trump played, one-hot                (52)  [347..398]
        #   + current trick standing [none/NS/EW]          (3)   [399..401]
        #   + dealer one-hot                               (4)   [402..405]
        #   + current player one-hot                       (4)   [406..409]
        # Trick history + winner + categorical seats were absent; with only
        # the current trick visible the policy was card-counting-blind, which
        # matched the flat ~rule-based plateau. New features go strictly past
        # 294 so existing offsets (and the rule-vs-rule canary) are unchanged.
        self.state_shape = [52 * 5 + 5 + 4 * 4 + 2 + 2 + 2 + 4 + 4
                            + 52 + 52 + 3 + 4 + 4]
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
        
        # Encode phase. Base is 52*5 (after all FIVE 52-card blocks: hand +
        # 4 trick slots). Using 52*4 collided every non-card feature with the
        # 4th trick card's one-hot — corrupting phase/trump during play.
        phase_idx = 52 * 5 + state['phase']
        obs[phase_idx] = 1
        
        # Encode bids
        bid_info = state['bid_info']
        if bid_info['bids'] is not None:
            for i, bid in enumerate(bid_info['bids']):
                if bid is not None:
                    obs[52 * 5 + 5 + i * 4 + bid] = 1
        
        # Encode points
        points = state['points']
        if points is not None:
            obs[52 * 5 + 5 + 16] = points[0]
            obs[52 * 5 + 5 + 17] = points[1]
        
        # Encode tricks won
        tricks_won = state['tricks_won']
        if tricks_won is not None:
            obs[52 * 5 + 5 + 18] = tricks_won[0] + tricks_won[2]  # N/S tricks
            obs[52 * 5 + 5 + 19] = tricks_won[1] + tricks_won[3]  # E/W tricks
        
        # Encode dealer and current player
        obs[52 * 5 + 5 + 20] = state['dealer']
        obs[52 * 5 + 5 + 21] = state['current_player']

        # Encode trump suit (one-hot over SUITS; all-zero until declared).
        # Decisive for play-phase ranking/strategy and previously omitted.
        trump = state.get('trump_suit')
        if trump in SUITS:
            obs[52 * 5 + 5 + 22 + SUITS.index(trump)] = 1

        # Encode current-trick lead suit (one-hot; all-zero if no card led yet)
        lead = state.get('trick_lead_suit')
        if lead in SUITS:
            obs[52 * 5 + 5 + 26 + SUITS.index(lead)] = 1

        # --- Appended features (base 295). Strictly past the legacy 0..294
        # block so existing offsets and the rule-vs-rule canary are intact. ---
        base = 52 * 5 + 5 + 30  # == 295

        def _card_idx(card):
            return RANKS.index(card.rank) + 13 * SUITS.index(card.suit)

        # Cards played in COMPLETED tricks (multi-hot). The current trick is
        # already encoded above; trick_history is the previously-missing
        # card-counting signal (which honors are still live).
        for trick in (state.get('trick_history') or []):
            for card in trick:
                if card is not None:
                    obs[base + _card_idx(card)] = 1

        # Highest trump played so far (one-hot); decisive for renege/honor.
        htp = state.get('highest_trump_played')
        if htp is not None:
            obs[base + 52 + _card_idx(htp)] = 1

        # Current trick standing relative to the NS partnership:
        # 0 = no winner yet, 1 = NS ahead, 2 = EW ahead.
        tw = state.get('trick_winner')
        if tw is None:
            obs[base + 104 + 0] = 1
        elif tw % 2 == 0:
            obs[base + 104 + 1] = 1
        else:
            obs[base + 104 + 2] = 1

        # Seat identities as one-hot (were raw 0..3 scalars at 285/286, which
        # the net read as magnitudes; legacy scalars left intact, superseded).
        dealer = state.get('dealer')
        if isinstance(dealer, int) and 0 <= dealer < 4:
            obs[base + 107 + dealer] = 1
        cur = state.get('current_player')
        if isinstance(cur, int) and 0 <= cur < 4:
            obs[base + 111 + cur] = 1

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
    
    def _game_to_env_action(self, game_action, phase):
        '''Map a game-internal action ID to its env action ID.'''
        if phase == PHASE_AUCTION:
            return game_action          # 0-4 → 0-4
        elif phase == PHASE_DECLARATION:
            return game_action + 5      # 0-3 → 5-8
        elif phase == PHASE_DISCARD:
            if game_action == DISCARD_DONE:
                return 17
            return game_action + 9      # 0-7 → 9-16
        elif phase == PHASE_GAMEPLAY:
            return game_action + 9      # 0-7 → 9-16
        return game_action

    def _get_legal_actions(self, state):
        '''
        Get legal actions in a format usable by the RL agent.
        Game action IDs are remapped to non-overlapping env action IDs so
        the same integer always means the same thing to the network.

        Returns:
            (dict): {env_action_id: None}
        '''
        legal_actions = {}
        phase = state['phase']
        for game_action in state['legal_actions']:
            env_action = self._game_to_env_action(game_action, phase)
            legal_actions[env_action] = None
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
        winners = [p for p, score in self.game.points.items() if score >= 125]
        payoffs = [0, 0, 0, 0]
        
        # If the game is over, the winning partnership gets 1, the losing gets -1
        if winners:
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
            # Game isn't over yet - provide intermediate rewards
            
            # Get partnership scores from the game (normalized to [0, 1])
            ns_score = self.game.points[0] / 125  # Normalize to [0, 1]
            ew_score = self.game.points[1] / 125  # Normalize to [0, 1]
            
            # Reward current score progress
            payoffs[0] = ns_score * 0.2  # Scale down to avoid overwhelming bid rewards
            payoffs[2] = ns_score * 0.2
            payoffs[1] = ew_score * 0.2
            payoffs[3] = ew_score * 0.2
            
            # Check if a bid was just made or lost
            if hasattr(self.game, 'bid_made') and self.game.highest_bidder is not None:
                bid_team = self.game.highest_bidder % 2  # 0 for NS, 1 for EW
                bid_value = 0
                if self.game.highest_bid is not None:
                    bid_value = self.game.highest_bid  # Use the bid index directly for reward scaling
                
                if hasattr(self.game, 'bid_made'):
                    # Add intermediate reward for making/losing a bid
                    if bid_team == 0:  # NS bid
                        if self.game.bid_made:
                            # NS made their bid - give positive reward
                            reward_value = 0.3 + (bid_value * 0.1)  # Higher bids give higher rewards
                            payoffs[0] += reward_value
                            payoffs[2] += reward_value
                            # Slight penalty to opponents when bid succeeds
                            payoffs[1] -= 0.1
                            payoffs[3] -= 0.1
                        else:
                            # NS failed their bid - give negative reward
                            reward_value = -0.3 - (bid_value * 0.1)  # Higher failed bids give larger penalties
                            payoffs[0] += reward_value
                            payoffs[2] += reward_value
                            # Reward opponents when bid fails
                            payoffs[1] += 0.2
                            payoffs[3] += 0.2
                    else:  # EW bid
                        if self.game.bid_made:
                            # EW made their bid - give positive reward
                            reward_value = 0.3 + (bid_value * 0.1)  # Higher bids give higher rewards
                            payoffs[1] += reward_value
                            payoffs[3] += reward_value
                            # Slight penalty to opponents when bid succeeds
                            payoffs[0] -= 0.1
                            payoffs[2] -= 0.1
                        else:
                            # EW failed their bid - give negative reward
                            reward_value = -0.3 - (bid_value * 0.1)  # Higher failed bids give larger penalties
                            payoffs[1] += reward_value
                            payoffs[3] += reward_value
                            # Reward opponents when bid fails
                            payoffs[0] += 0.2
                            payoffs[2] += 0.2
                
            # Add a small reward for winning tricks in the current hand
            if self.game.tricks_won and sum(self.game.tricks_won) > 0:
                ns_tricks = self.game.tricks_won[0] + self.game.tricks_won[2]
                ew_tricks = self.game.tricks_won[1] + self.game.tricks_won[3]
                total_tricks = ns_tricks + ew_tricks
                
                if total_tricks > 0:
                    # Small intermediate reward for winning tricks proportional to % of tricks won
                    ns_trick_reward = 0.1 * (ns_tricks / total_tricks)
                    ew_trick_reward = 0.1 * (ew_tricks / total_tricks)
                    
                    payoffs[0] += ns_trick_reward
                    payoffs[2] += ns_trick_reward
                    payoffs[1] += ew_trick_reward
                    payoffs[3] += ew_trick_reward
        
        return payoffs
    
    def _decode_action(self, action_id):
        '''
        Convert an env action ID back to the game-internal action expected by
        game.step().  Inverse of _game_to_env_action().
        '''
        phase = self.game.phase
        if phase == PHASE_AUCTION:
            return action_id            # 0-4 → 0-4
        elif phase == PHASE_DECLARATION:
            return action_id - 5        # 5-8 → 0-3
        elif phase == PHASE_DISCARD:
            if action_id == 17:
                return DISCARD_DONE     # → 16
            return action_id - 9        # 9-16 → 0-7
        elif phase == PHASE_GAMEPLAY:
            return action_id - 9        # 9-16 → 0-7
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