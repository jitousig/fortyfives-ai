'''
Unit tests for Fortyfives game
'''

import unittest
import numpy as np

import rlcard
from rlcard.agents import RandomAgent
from fortyfives.games.fortyfives.game import FortyfivesGame
from fortyfives.games.fortyfives.card import FortyfivesCard, get_card_rank, RANKS

# Register the environment
from rlcard.envs import register
register('fortyfives', 'fortyfives.envs.fortyfives_env:FortyfivesEnv')

class TestFortyfives(unittest.TestCase):
    '''
    Test cases for Fortyfives game
    '''
    
    def test_init_game(self):
        '''
        Test game initialization
        '''
        game = FortyfivesGame()
        state = game.get_state(0)
        
        # Check that the state has expected keys
        self.assertIn('hand', state)
        self.assertIn('current_player', state)
        self.assertIn('phase', state)
        self.assertIn('bid_info', state)
        
        # Check that each player has 5 cards
        self.assertEqual(len(game.hands[0]), 5)
        self.assertEqual(len(game.hands[1]), 5)
        self.assertEqual(len(game.hands[2]), 5)
        self.assertEqual(len(game.hands[3]), 5)
        
        # Check that bidding is set up properly
        self.assertEqual(game.phase, 1)  # Auction phase
        self.assertIsNone(game.highest_bidder)
        self.assertIsNone(game.highest_bid)
        
    def test_card_rank(self):
        '''
        Test card ranking
        '''
        # Test red trump cards
        five_hearts = FortyfivesCard(RANKS.index('5') + 13 * 1)  # 5 of Hearts
        jack_hearts = FortyfivesCard(RANKS.index('J') + 13 * 1)  # J of Hearts
        ace_hearts = FortyfivesCard(RANKS.index('A') + 13 * 1)   # A of Hearts
        ace_diamonds = FortyfivesCard(RANKS.index('A') + 13 * 2) # A of Diamonds
        
        trump_suit = 'H'  # Hearts as trump
        
        # Test that 5 > J > A of Hearts > A of other suits in trump
        self.assertGreater(get_card_rank(five_hearts, trump_suit), get_card_rank(jack_hearts, trump_suit))
        self.assertGreater(get_card_rank(jack_hearts, trump_suit), get_card_rank(ace_hearts, trump_suit))
        
        # Test that A of Hearts is always trump
        trump_suit = 'S'  # Spades as trump
        self.assertGreater(get_card_rank(ace_hearts, trump_suit), get_card_rank(ace_diamonds, trump_suit))
    
    def test_bidding_logic(self):
        '''
        Test bidding logic
        '''
        game = FortyfivesGame()
        
        # Player 0 bids 20
        game.step(1)  # BID_20
        self.assertEqual(game.highest_bidder, 0)
        self.assertEqual(game.highest_bid, 1)  # BID_20
        
        # Player 1 passes
        game.step(0)  # BID_PASS
        self.assertEqual(game.highest_bidder, 0)
        self.assertEqual(game.highest_bid, 1)
        
        # Player 2 bids 25
        game.step(2)  # BID_25
        self.assertEqual(game.highest_bidder, 2)
        self.assertEqual(game.highest_bid, 2)
        
        # Player 3 passes
        game.step(0)  # BID_PASS
        self.assertEqual(game.highest_bidder, 2)
        self.assertEqual(game.highest_bid, 2)
        
        # Player 0 passes
        game.step(0)  # BID_PASS
        self.assertEqual(game.highest_bidder, 2)
        self.assertEqual(game.highest_bid, 2)
        
        # Check that bidding is over and we've moved to the declaration phase
        self.assertEqual(game.phase, 2)  # Declaration phase
        self.assertEqual(game.current_player_id, 2)  # Highest bidder should be next
    
    def test_environment(self):
        '''
        Test environment integration with RLcard
        '''
        np.random.seed(42)
        
        env = rlcard.make('fortyfives')
        
        # Set random agents
        agents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)]
        env.set_agents(agents)
        
        # Play a game
        trajectories, payoffs = env.run(is_training=False)
        
        # Check that trajectories and payoffs have expected structure
        self.assertEqual(len(trajectories), 4)  # One for each player
        self.assertEqual(len(payoffs), 4)  # One for each player
        
        # Check that N/S have same payoff and E/W have same payoff
        self.assertEqual(payoffs[0], payoffs[2])
        self.assertEqual(payoffs[1], payoffs[3])

if __name__ == '__main__':
    unittest.main() 