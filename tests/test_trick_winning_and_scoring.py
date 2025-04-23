"""
Test script for verifying the trick winning logic, legal plays, and scoring issues
in the Fortyfives game.
"""

import unittest
from fortyfives.games.fortyfives.game import (
    FortyfivesGame, 
    SUIT_SPADES, SUIT_HEARTS, SUIT_DIAMONDS, SUIT_CLUBS,
    PHASE_AUCTION, PHASE_DECLARATION, PHASE_DISCARD, PHASE_GAMEPLAY,
    BID_20, BID_25, BID_30
)
from fortyfives.games.fortyfives.card import FortyfivesCard as Card, RANKS, SUITS

# Helper function to create a card with specific rank and suit
def create_card(rank, suit):
    card_id = RANKS.index(rank) + SUITS.index(suit) * 13
    return Card(card_id)

class TestTrickWinningAndScoring(unittest.TestCase):
    """Test cases for trick winning logic, legal plays, and scoring in Fortyfives game."""
    
    def test_trick_winning_with_trumps(self):
        """Test that the highest trump played wins the trick regardless of led suit."""
        # Create a game with controlled state
        game = FortyfivesGame()
        game.phase = PHASE_GAMEPLAY
        game.trump_suit = 'H'  # Hearts is trump
        
        # Set up a trick with specific cards
        game.current_trick = [None] * game.num_players
        game.trick_starter = 0  # North leads
        
        # North plays 10 of Spades (non-trump)
        game.current_trick[0] = create_card('T', 'S')
        game.trick_lead_suit = 'S'  # Lead suit is Spades
        
        # East plays Queen of Hearts (trump)
        game.current_trick[1] = create_card('Q', 'H')
        
        # South plays 5 of Clubs (non-trump)
        game.current_trick[2] = create_card('5', 'C')
        
        # West plays Ace of Spades (non-trump, but same as lead suit)
        game.current_trick[3] = create_card('A', 'S')
        
        # Determine the winner of the trick
        winner = game.get_trick_winner()
        
        # East should win because they played a trump
        self.assertEqual(winner, 1, "East should win the trick with Queen of Hearts (trump)")
        
        # Now let's test when multiple trumps are played
        game.current_trick = [None] * game.num_players
        
        # North plays 10 of Spades (non-trump)
        game.current_trick[0] = create_card('T', 'S')
        game.trick_lead_suit = 'S'  # Lead suit is Spades
        
        # East plays Queen of Hearts (trump)
        game.current_trick[1] = create_card('Q', 'H')
        
        # South plays 5 of Hearts (trump - highest trump)
        game.current_trick[2] = create_card('5', 'H')
        
        # West plays Jack of Hearts (trump - second highest)
        game.current_trick[3] = create_card('J', 'H')
        
        # Determine the winner of the trick
        winner = game.get_trick_winner()
        
        # South should win with 5 of Hearts (highest trump)
        self.assertEqual(winner, 2, "South should win the trick with 5 of Hearts (highest trump)")
    
    def test_legal_plays_with_trumps(self):
        """Test that players can always play trump cards, even when they have lead suit cards."""
        # Create a game with controlled state
        game = FortyfivesGame()
        game.phase = PHASE_GAMEPLAY
        game.trump_suit = 'H'  # Hearts is trump
        
        # Set up player's hand with both clubs and hearts (trump)
        game.hands = {0: []}
        game.hands[0] = [
            create_card('A', 'C'),  # Ace of Clubs
            create_card('9', 'H'),  # 9 of Hearts (trump)
            create_card('5', 'H'),  # 5 of Hearts (highest trump)
            create_card('K', 'C')   # King of Clubs
        ]
        
        # Set up a trick with Clubs led
        game.trick_starter = 3  # West leads
        game.current_trick = [None] * game.num_players
        game.current_trick[3] = create_card('2', 'C')  # 2 of Clubs led
        game.trick_lead_suit = 'C'
        
        # Get legal plays for player 0 (who has both clubs and hearts)
        game.current_player_id = 0
        legal_plays = game.get_legal_plays()
        
        # Player should be able to play both Clubs (following suit) AND Hearts (trump)
        legal_cards = [str(game.hands[0][i]) for i in legal_plays]
        
        self.assertIn('AC', legal_cards, "AC (lead suit) should be a legal play")
        self.assertIn('KC', legal_cards, "KC (lead suit) should be a legal play")
        self.assertIn('9H', legal_cards, "9H (trump) should be a legal play")
        self.assertIn('5H', legal_cards, "5H (trump) should be a legal play")
    
    def test_highest_trump_scoring(self):
        """Test that playing the highest trump earns 5 extra points for the partnership."""
        # Create a game with controlled state
        game = FortyfivesGame()
        
        # Set up a completed hand
        game.trump_suit = 'H'  # Hearts is trump
        game.tricks_won = [2, 2, 1, 0]  # N/S: 3 tricks, E/W: 2 tricks
        game.highest_trump_player = 2  # South played highest trump
        game.highest_trump_played = create_card('J', 'H')  # Jack of Hearts
        game.highest_bid = BID_20
        game.highest_bidder = 1  # East was highest bidder
        
        # Score the hand
        game.score_hand()
        ns_hand_points = game.hand_points[0]
        ew_hand_points = game.hand_points[1]
        
        # End the hand to apply bid penalties
        game.end_hand()
        
        # Check the hand scores (raw points before penalties):
        # N/S: 3 tricks × 5 points = 15 points + 5 for highest trump = 20
        # E/W: 2 tricks × 5 points = 10 points
        self.assertEqual(ns_hand_points, 20, "N/S should get 20 points (15 for tricks + 5 for highest trump)")
        self.assertEqual(ew_hand_points, 10, "E/W should get 10 points for tricks")
        
        # Check the game scores (after penalties):
        # N/S get their full 20 points
        # E/W get -20 because they bid 20 but only earned 10 points
        self.assertEqual(game.points[0], 20, "N/S game points should be 20")
        self.assertEqual(game.points[1], -20, "E/W game points should be -20 (failed to make bid of 20)")
    
    def test_all_players_get_to_play(self):
        """Test that all players get to play a card in a trick before it ends."""
        # Create a game with controlled state
        game = FortyfivesGame()
        game.phase = PHASE_GAMEPLAY
        game.trick_starter = 0  # North starts
        game.current_player_id = 0
        game.trick_lead_suit = 'S'  # Spades lead
        game.trump_suit = 'H'  # Hearts trump
        
        # Set up distinct cards for all players
        game.hands = {
            0: [create_card('T', 'S')],  # North: 10 of Spades
            1: [create_card('Q', 'H')],  # East: Queen of Hearts (trump)
            2: [create_card('A', 'S')],  # South: Ace of Spades
            3: [create_card('K', 'D')]   # West: King of Diamonds
        }
            
        # Set up empty current trick and trick history
        game.current_trick = [None] * game.num_players
        game.trick_history = []
        game.trick_winners = []
        
        # Capture the play process
        all_cards_played = []
        
        # Play a complete trick manually
        for i in range(game.num_players):
            print(f"Player {i} about to play a card")
            print(f"Current trick before play: {[str(c) if c else None for c in game.current_trick]}")
            
            # Set current player and play their card
            game.current_player_id = i
            card_before = game.hands[i][0] if game.hands[i] else None
            if card_before:
                all_cards_played.append(str(card_before))
                print(f"Player {i} playing {str(card_before)}")
            game.process_play(0)
            
            # Print state after play
            print(f"Current trick after play: {[str(c) if c else None for c in game.current_trick]}")
            print(f"Trick history length: {len(game.trick_history)}")
            if game.trick_history:
                print(f"Last trick recorded: {[str(c) if c else None for c in game.trick_history[-1]]}")
            print(f"Current player after play: {game.current_player_id}")
            print("-----")
        
        # Verify all cards were played
        print(f"All cards played: {all_cards_played}")
        
        # Verify the trick was completed and recorded in history
        self.assertEqual(len(game.trick_history), 1, "One trick should be recorded in history")
        
        # Get the recorded trick
        if game.trick_history:
            recorded_trick = game.trick_history[0]
            print(f"Recorded trick: {[str(c) if c else None for c in recorded_trick]}")
            
            # Verify all four cards were played and recorded
            self.assertIsNotNone(recorded_trick[0], "North's card should be in the trick")
            self.assertIsNotNone(recorded_trick[1], "East's card should be in the trick")
            self.assertIsNotNone(recorded_trick[2], "South's card should be in the trick")
            self.assertIsNotNone(recorded_trick[3], "West's card should be in the trick")
            
            # Check specific cards
            self.assertEqual(str(recorded_trick[0]), "TS", "North should have played 10 of Spades")
            self.assertEqual(str(recorded_trick[1]), "QH", "East should have played Queen of Hearts")
            self.assertEqual(str(recorded_trick[2]), "AS", "South should have played Ace of Spades")
            self.assertEqual(str(recorded_trick[3]), "KD", "West should have played King of Diamonds")
            
            # Check that East won the trick (played the only trump)
            if game.trick_winners:
                self.assertEqual(game.trick_winners[0], 1, "East should have won the trick with Queen of Hearts (trump)")
    
    def test_score_output_format(self):
        """Test that the score output includes partnership names and point breakdowns."""
        # Create a game with controlled state
        game = FortyfivesGame()
        game.verbose = True
        
        # Set up a completed hand
        game.trump_suit = 'H'  # Hearts is trump
        game.tricks_won = [2, 1, 1, 1]  # N/S: 3 tricks, E/W: 2 tricks
        game.highest_trump_player = 0  # North played highest trump
        game.highest_trump_played = create_card('5', 'H')  # 5 of Hearts
        game.highest_bid = BID_20
        game.highest_bidder = 0  # North was highest bidder
        
        # We'll capture the printed output to check formatting
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Score the hand
        game.end_hand()
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        # Check that the output contains partnership names and point breakdowns
        output = captured_output.getvalue()
        
        # Check for trick points
        self.assertIn("Trick points", output, "Output should include 'Trick points'")
        self.assertIn("North/South", output, "Output should include 'North/South'")
        self.assertIn("East/West", output, "Output should include 'East/West'")
        
        # Check for highest trump bonus
        self.assertIn("Highest trump bonus", output, "Output should include 'Highest trump bonus'")
        
        # Check for bid information
        self.assertIn("bid 20", output, "Output should include 'bid 20'")
        self.assertIn("made their bid", output, "Output should include 'made their bid'")
        
        # Check for clear distinction between hand points and game points
        self.assertIn("Hand points (trick points + bonuses)", output, 
                     "Output should include 'Hand points (trick points + bonuses)'")
        self.assertIn("Game points after penalties", output, 
                     "Output should include 'Game points after penalties'")

    def test_high_trump_reneging_when_low_trump_led(self):
        """Test that players can play any card when a lower trump is led (high trump reneging)."""
        # Create a game with controlled state
        game = FortyfivesGame()
        game.phase = PHASE_GAMEPLAY
        game.trump_suit = 'D'  # Diamonds is trump
        
        # Set up player's hand with various cards including high and low trumps
        game.hands = {0: []}
        game.hands[0] = [
            create_card('J', 'D'),  # Jack of Diamonds (high trump)
            create_card('5', 'D'),  # 5 of Diamonds (high trump)
            create_card('A', 'H'),  # Ace of Hearts (special high trump)
            create_card('K', 'H'),  # King of Hearts (non-trump, non-lead suit)
            create_card('Q', 'S')   # Queen of Spades (non-trump, non-lead suit)
        ]
        
        # Set up a trick with a low trump led (3 of Diamonds)
        game.trick_starter = 3  # West leads
        game.current_trick = [None] * game.num_players
        game.current_trick[3] = create_card('3', 'D')  # 3 of Diamonds led (low trump)
        game.trick_lead_suit = 'D'
        
        # Get legal plays for player 0
        game.current_player_id = 0
        legal_plays = game.get_legal_plays()
        
        # Player should be able to play ANY card when a low trump is led
        # This reflects the Nova Scotia 45s rule that high trumps can be reneged
        self.assertEqual(len(legal_plays), len(game.hands[0]), 
                        "Player should be able to play any card when a low trump is led")
        
        # Check that high trumps can be reneged
        legal_cards = [str(game.hands[0][i]) for i in legal_plays]
        self.assertIn('JD', legal_cards, "JD (high trump) should be a legal play despite reneging rules")
        self.assertIn('5D', legal_cards, "5D (high trump) should be a legal play despite reneging rules")
        self.assertIn('AH', legal_cards, "AH (special high trump) should be a legal play despite reneging rules")
        self.assertIn('KH', legal_cards, "KH (non-trump) should be a legal play when low trump is led")
        self.assertIn('QS', legal_cards, "QS (non-trump) should be a legal play when low trump is led")

    def test_highest_trump_wins_trick(self):
        """Test that the highest trump wins the trick regardless of which player plays it."""
        # Create a game with controlled state
        game = FortyfivesGame()
        game.phase = PHASE_GAMEPLAY
        game.trump_suit = 'D'  # Diamonds is trump
        
        # Set up a trick with a non-trump lead
        game.current_trick = [None] * game.num_players
        game.trick_starter = 1  # East leads
        
        # East leads with Queen of Diamonds (lower trump)
        game.current_trick[1] = create_card('Q', 'D')
        game.trick_lead_suit = 'D'
        
        # South plays Jack of Diamonds (high trump)
        game.current_trick[2] = create_card('J', 'D')
        
        # West plays 9 of Diamonds (lower trump)
        game.current_trick[3] = create_card('9', 'D')
        
        # North plays 7 of Diamonds (lower trump)
        game.current_trick[0] = create_card('7', 'D')
        
        # Determine the winner of the trick
        winner = game.get_trick_winner()
        
        # South should win with Jack of Diamonds (2nd highest trump)
        self.assertEqual(winner, 2, "South should win the trick with Jack of Diamonds (high trump)")
        
        # Another scenario with a non-trump lead and different highest trump
        game.current_trick = [None] * game.num_players
        game.trick_starter = 1  # East leads
        
        # East leads with 8 of Hearts (non-trump)
        game.current_trick[1] = create_card('8', 'H')
        game.trick_lead_suit = 'H'
        
        # South plays 3 of Hearts (following suit)
        game.current_trick[2] = create_card('3', 'H')
        
        # West plays 5 of Diamonds (highest trump)
        game.current_trick[3] = create_card('5', 'D')
        
        # North plays King of Hearts (highest in led suit)
        game.current_trick[0] = create_card('K', 'H')
        
        # Determine the winner of the trick
        winner = game.get_trick_winner()
        
        # West should win with 5 of Diamonds (highest trump)
        self.assertEqual(winner, 3, "West should win the trick with 5 of Diamonds (highest trump)")

    def test_highest_lead_suit_wins_when_no_trump(self):
        """Test that the highest card in the lead suit wins when no trump is played."""
        # Create a game with controlled state
        game = FortyfivesGame()
        game.phase = PHASE_GAMEPLAY
        game.trump_suit = 'D'  # Diamonds is trump
        
        # Set up a trick with Hearts lead and no trumps played
        game.current_trick = [None] * game.num_players
        game.trick_starter = 2  # South leads
        
        # South leads with 2 of Hearts (non-trump)
        game.current_trick[2] = create_card('2', 'H')
        game.trick_lead_suit = 'H'
        
        # West plays 9 of Hearts (non-trump)
        game.current_trick[3] = create_card('9', 'H')
        
        # North plays 5 of Hearts (non-trump)
        game.current_trick[0] = create_card('5', 'H')
        
        # East plays 7 of Spades (non-trump, non-lead suit)
        game.current_trick[1] = create_card('7', 'S')
        
        # Determine the winner of the trick
        winner = game.get_trick_winner()
        
        # In Hearts (a red suit), the ranking is K, Q, J, 10, 9, 8, 7, 6, 4, 3, 2, A
        # So West should win with 9 of Hearts (highest in led suit)
        self.assertEqual(winner, 3, "West should win with 9 of Hearts (highest in led suit)")
        
        # Another scenario with black suit lead and no trumps
        game.current_trick = [None] * game.num_players
        game.trick_starter = 0  # North leads
        
        # North leads with 10 of Spades (non-trump)
        game.current_trick[0] = create_card('T', 'S')
        game.trick_lead_suit = 'S'
        
        # East plays 4 of Spades (non-trump)
        game.current_trick[1] = create_card('4', 'S')
        
        # South plays King of Spades (non-trump)
        game.current_trick[2] = create_card('K', 'S')
        
        # West plays 6 of Hearts (non-trump, non-lead suit)
        game.current_trick[3] = create_card('6', 'H')
        
        # Determine the winner of the trick
        winner = game.get_trick_winner()
        
        # In Spades (a black suit), the ranking is K, Q, J, A, 2, 3, 4, 6, 7, 8, 9, 10
        # So South should win with King of Spades (highest in led suit)
        self.assertEqual(winner, 2, "South should win with King of Spades (highest in led suit)")

    def test_hand_scoring_vs_game_scoring(self):
        """Test that hand scores show raw points while game scores include bid penalties."""
        # Create a game with controlled state
        game = FortyfivesGame()
        
        # Set up a completed hand
        game.trump_suit = 'D'  # Diamonds is trump
        
        # Setup: N/S win 2 tricks (10 points) + highest trump bonus (5) = 15 points total
        # E/W win 3 tricks (15 points) but bid 30 and fail to make it
        game.tricks_won = [1, 2, 1, 1]  # N/S: 2 tricks, E/W: 3 tricks
        
        # South played the highest trump
        game.highest_trump_player = 2  # South
        game.highest_trump_played = create_card('J', 'D')  # Jack of Diamonds
        
        # East was the highest bidder with a bid of 30
        game.highest_bidder = 1
        game.highest_bid = BID_30
        
        # Score the hand (save hand_points values)
        game.score_hand()
        ns_hand_points = game.hand_points[0]
        ew_hand_points = game.hand_points[1]
        
        # Apply game score (penalties)
        game.end_hand()
        
        # Check the hand scores (raw points)
        # N/S: 2 tricks × 5 = 10 points + 5 for highest trump = 15 points
        # E/W: 3 tricks × 5 = 15 points
        self.assertEqual(ns_hand_points, 15, "N/S hand points should be 15 (10 for tricks + 5 for highest trump)")
        self.assertEqual(ew_hand_points, 15, "E/W hand points should be 15 (15 for tricks)")
        
        # Check the game scores (with penalties)
        # N/S get their full 15 points
        # E/W get -30 because they failed to make their bid of 30
        self.assertEqual(game.points[0], 15, "N/S game points should be 15")
        self.assertEqual(game.points[1], -30, "E/W game points should be -30 (failed to make bid of 30)")
        
        # Try another scenario where the bidding team makes their bid
        game = FortyfivesGame()
        game.trump_suit = 'H'  # Hearts is trump
        
        # Setup: N/S win 3 tricks (15 points) + highest trump bonus (5) = 20 points total and bid 20
        # E/W win 2 tricks (10 points)
        game.tricks_won = [2, 1, 1, 1]  # N/S: 3 tricks, E/W: 2 tricks
        
        # North played the highest trump
        game.highest_trump_player = 0  # North
        game.highest_trump_played = create_card('5', 'H')  # 5 of Hearts
        
        # North was the highest bidder with a bid of 20
        game.highest_bidder = 0
        game.highest_bid = BID_20
        
        # Score the hand (save hand_points values)
        game.score_hand()
        ns_hand_points = game.hand_points[0]
        ew_hand_points = game.hand_points[1]
        
        # Apply game score
        game.end_hand()
        
        # Check the hand scores (raw points)
        # N/S: 3 tricks × 5 = 15 points + 5 for highest trump = 20 points
        # E/W: 2 tricks × 5 = 10 points
        self.assertEqual(ns_hand_points, 20, "N/S hand points should be 20 (15 for tricks + 5 for highest trump)")
        self.assertEqual(ew_hand_points, 10, "E/W hand points should be 10 (10 for tricks)")
        
        # Check the game scores (with bid results)
        # N/S get their full 20 points (made the bid)
        # E/W get their full 10 points
        self.assertEqual(game.points[0], 20, "N/S game points should be 20 (made their bid)")
        self.assertEqual(game.points[1], 10, "E/W game points should be 10")

    def test_thirty_for_sixty_rule(self):
        """Test that the 30 for 60 rule is correctly applied in scoring."""
        # 1. Test successful 30 bid gets 60 points
        game = FortyfivesGame()
        
        # Set up a completed hand where a team made a 30 bid
        game.trump_suit = 'H'  # Hearts is trump
        game.tricks_won = [3, 0, 2, 0]  # N/S: 5 tricks (25 points), E/W: 0 tricks (0 points)
        game.highest_trump_player = 0  # North played highest trump
        game.highest_trump_played = create_card('5', 'H')  # 5 of Hearts
        game.highest_bid = BID_30
        game.highest_bidder = 0  # North was highest bidder (NS team)
        
        # Score the hand
        game.score_hand()
        
        # Verify bid was made (30 raw points is enough to make a 30 bid)
        self.assertTrue(game.bid_made, "Bid of 30 should be considered made with 30 raw points")
        
        # Check raw hand points (without special 30 for 60 rule)
        # N/S: 5 tricks × 5 = 25 points + 5 for highest trump = 30
        # E/W: 0 tricks × 5 = 0 points
        self.assertEqual(game.hand_points[0], 30, "N/S should have 30 raw hand points")
        self.assertEqual(game.hand_points[1], 0, "E/W should have 0 raw hand points")
        
        # End the hand to apply the 30 for 60 rule
        game.end_hand()
        
        # Check game points (after 30 for 60 rule applied)
        # N/S should get 60 points instead of 30
        # E/W still get their 0 points
        self.assertEqual(game.points[0], 60, "N/S should get 60 game points (30 for 60 rule)")
        self.assertEqual(game.points[1], 0, "E/W should get 0 game points")

        # 2. Test failed 30 bid loses 30 points
        game = FortyfivesGame()
        
        # Set up a completed hand where a team failed a 30 bid
        game.trump_suit = 'S'  # Spades is trump
        game.tricks_won = [1, 3, 1, 0]  # N/S: 2 tricks (10 points), E/W: 3 tricks (15 points)
        game.highest_trump_player = 1  # East played highest trump
        game.highest_trump_played = create_card('5', 'S')  # 5 of Spades
        game.highest_bid = BID_30
        game.highest_bidder = 2  # South was highest bidder (NS team)
        
        # Score the hand
        game.score_hand()
        
        # Verify bid was not made (15 raw points is not enough to make a 30 bid)
        self.assertFalse(game.bid_made, "Bid of 30 should be considered failed with only 15 raw points")
        
        # Check raw hand points
        # N/S: 2 tricks × 5 = 10 points + 0 for highest trump = 10
        # E/W: 3 tricks × 5 = 15 points + 5 for highest trump = 20
        self.assertEqual(game.hand_points[0], 10, "N/S should have 10 raw hand points")
        self.assertEqual(game.hand_points[1], 20, "E/W should have 20 raw hand points")
        
        # End the hand to apply penalty
        game.end_hand()
        
        # Check game points (after penalties)
        # N/S should get -30 points (penalty for failing a 30 bid)
        # E/W still get their 20 points
        self.assertEqual(game.points[0], -30, "N/S should get -30 game points (failed 30 bid)")
        self.assertEqual(game.points[1], 20, "E/W should get 20 game points")

    def test_pegging_restriction_after_100_points(self):
        """Test that teams with 100+ points can only peg when declaring team fails their bid."""
        
        # 1. Test when declaring team makes their bid - no pegging for defending team with 100+ points
        game = FortyfivesGame()
        
        # Set initial points: N/S team has 100+ points
        game.points = {0: 120, 1: 70, 2: 120, 3: 70}  # N/S: 120, E/W: 70
        
        # Set up a completed hand where E/W team made a 20 bid
        game.trump_suit = 'H'  # Hearts is trump
        game.tricks_won = [2, 3, 0, 0]  # N/S: 2 tricks (10 points), E/W: 3 tricks (15 points)
        game.highest_trump_player = 1  # East played highest trump
        game.highest_trump_played = create_card('5', 'H')  # 5 of Hearts
        game.highest_bid = BID_20
        game.highest_bidder = 1  # East was highest bidder (E/W team)
        
        # Score the hand
        game.score_hand()
        
        # Verify bid was made (20 raw points is enough to make a 20 bid)
        self.assertTrue(game.bid_made, "Bid of 20 should be considered made with 20 raw points")
        
        # Check raw hand points
        # N/S: 2 tricks × 5 = 10 points + 0 for highest trump = 10
        # E/W: 3 tricks × 5 = 15 points + 5 for highest trump = 20
        self.assertEqual(game.hand_points[0], 10, "N/S should have 10 raw hand points")
        self.assertEqual(game.hand_points[1], 20, "E/W should have 20 raw hand points")
        
        # End the hand to apply scoring rules
        game.end_hand()
        
        # Check game points:
        # N/S should NOT peg (still 120 points, no change)
        # E/W should get their 20 points (70 + 20 = 90)
        self.assertEqual(game.points[0], 120, "N/S should not peg when they have 100+ points and E/W makes bid")
        self.assertEqual(game.points[1], 90, "E/W should get 20 game points for making their bid")
        
        # 2. Test when declaring team fails their bid - pegging IS allowed for defending team with 100+ points
        game = FortyfivesGame()
        
        # Set initial points: N/S team has 100+ points
        game.points = {0: 120, 1: 70, 2: 120, 3: 70}  # N/S: 120, E/W: 70
        
        # Set up a completed hand where E/W team failed a 25 bid
        game.trump_suit = 'S'  # Spades is trump
        game.tricks_won = [3, 2, 0, 0]  # N/S: 3 tricks (15 points), E/W: 2 tricks (10 points)
        game.highest_trump_player = 0  # North played highest trump
        game.highest_trump_played = create_card('5', 'S')  # 5 of Spades
        game.highest_bid = BID_25
        game.highest_bidder = 3  # West was highest bidder (E/W team)
        
        # Score the hand
        game.score_hand()
        
        # Verify bid was not made (15 raw points is not enough to make a 25 bid)
        self.assertFalse(game.bid_made, "Bid of 25 should be considered failed with only 15 raw points")
        
        # Check raw hand points
        # N/S: 3 tricks × 5 = 15 points + 5 for highest trump = 20
        # E/W: 2 tricks × 5 = 10 points + 0 for highest trump = 10
        self.assertEqual(game.hand_points[0], 20, "N/S should have 20 raw hand points")
        self.assertEqual(game.hand_points[1], 10, "E/W should have 10 raw hand points")
        
        # End the hand to apply scoring rules
        game.end_hand()
        
        # Check game points:
        # N/S should be allowed to peg since E/W failed bid (120 + 20 = 140)
        # E/W should get penalty points (-25 for failing bid)
        self.assertEqual(game.points[0], 140, "N/S should be allowed to peg when E/W fails their bid")
        self.assertEqual(game.points[1], 45, "E/W should get -25 game points for failing their bid")

if __name__ == '__main__':
    unittest.main() 