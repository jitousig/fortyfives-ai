import unittest
import numpy as np
from fortyfives.games.fortyfives.game import FortyfivesGame
from fortyfives.games.fortyfives.game import BID_PASS, BID_20, BID_25, BID_30, BID_HOLD, PHASE_AUCTION

class TestBiddingLegalActions(unittest.TestCase):
    """Test the legal actions during the bidding phase."""
    
    def test_initial_bidding_options(self):
        """Test that the initial player has all bidding options."""
        game = FortyfivesGame()
        
        # Ensure we're in the auction phase and player 1 is the current player
        self.assertEqual(game.phase, PHASE_AUCTION)
        self.assertEqual(game.current_player_id, 1, "By default, player 1 (East) starts the auction")
        
        # Print legal actions for debugging
        legal_actions = game.get_legal_actions()
        print(f"Legal actions: {legal_actions}")
        print(f"BID constants: PASS={BID_PASS}, BID_20={BID_20}, BID_25={BID_25}, BID_30={BID_30}")
        print(f"Highest bid: {game.highest_bid}")
        
        # The initial player should be able to pass or bid any value
        expected_actions = [BID_PASS, BID_20, BID_25, BID_30]
        for action in expected_actions:
            self.assertIn(action, legal_actions, f"Action {action} should be legal for the initial player")
        
        # Player 1 shouldn't have HOLD yet (no bids made)
        self.assertNotIn(BID_HOLD, legal_actions, "HOLD should not be available when no bids have been made")
    
    def test_hold_only_for_dealer(self):
        """Test that HOLD is only available to the dealer."""
        game = FortyfivesGame()
        
        # Verify we start with player 1
        self.assertEqual(game.current_player_id, 1)
        
        # Simulate player 1 making a bid
        game.process_auction(BID_20)
        
        # Now it should be player 2's turn
        self.assertEqual(game.current_player_id, 2)
        legal_actions = game.get_legal_actions()
        
        # Player 2 should not have HOLD available
        self.assertNotIn(BID_HOLD, legal_actions, "HOLD should not be available to non-dealer")
        
        # Simulate player 2 and player 3 passing
        game.process_auction(BID_PASS)
        game.process_auction(BID_PASS)
        
        # Now it's dealer's turn (player 0)
        self.assertEqual(game.current_player_id, 0)
        legal_actions = game.get_legal_actions()
        
        # Dealer should have HOLD available
        self.assertIn(BID_HOLD, legal_actions, "HOLD should be available to dealer when there's a bid")
    
    def test_higher_bids(self):
        """Test that only higher bids are allowed after a bid is made."""
        game = FortyfivesGame()
        
        # Print the current player ID for reference
        print(f"Initial current_player_id: {game.current_player_id}")
        self.assertEqual(game.current_player_id, 1, "By default, player 1 starts the auction")
        
        # Simulate player 1 bidding 20
        game.process_auction(BID_20)
        
        # Check the player ID after player 1's action
        print(f"After player 1 bids, current_player_id: {game.current_player_id}")
        
        # Now it should be player 2's turn
        self.assertEqual(game.current_player_id, 2, "After player 1 bids, current player should be 2")
        legal_actions = game.get_legal_actions()
        
        # Player 2 should be able to pass or bid 25 or 30 (but not 20)
        expected_actions = [BID_PASS, BID_25, BID_30]
        for action in expected_actions:
            self.assertIn(action, legal_actions, f"Action {action} should be legal after a bid of 20")
        
        self.assertNotIn(BID_20, legal_actions, "Cannot bid 20 after a bid of 20")
        
        # Simulate player 2 bidding 25
        game.process_auction(BID_25)
        
        # Now it should be player 3's turn
        self.assertEqual(game.current_player_id, 3, "After player 2 bids, current player should be 3")
        legal_actions = game.get_legal_actions()
        
        # Player 3 should be able to pass or bid 30 (but not 20 or 25)
        expected_actions = [BID_PASS, BID_30]
        for action in expected_actions:
            self.assertIn(action, legal_actions, f"Action {action} should be legal after a bid of 25")
        
        self.assertNotIn(BID_20, legal_actions, "Cannot bid 20 after a bid of 25")
        self.assertNotIn(BID_25, legal_actions, "Cannot bid 25 after a bid of 25")

if __name__ == '__main__':
    unittest.main() 