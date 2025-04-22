import unittest
import numpy as np
from fortyfives.games.fortyfives.game import (
    FortyfivesGame, 
    SUIT_SPADES, SUIT_HEARTS, SUIT_DIAMONDS, SUIT_CLUBS,
    PHASE_AUCTION, PHASE_DECLARATION, PHASE_DISCARD, PHASE_GAMEPLAY, PHASE_SCORING,
    DISCARD_DONE,
    BID_20, BID_25, BID_30
)
from fortyfives.games.fortyfives.card import FortyfivesCard as Card, RANKS, SUITS
from fortyfives.games.fortyfives.dealer import FortyfivesDealer

# Helper function to create a card with specific rank and suit
def create_card(rank, suit):
    card_id = RANKS.index(rank) + SUITS.index(suit) * 13
    return Card(card_id)

class TestFortyfivesGame(unittest.TestCase):
    def setUp(self):
        self.game = FortyfivesGame()
        # Set random seed for reproducibility
        self.game.np_random = np.random.RandomState(42)
        
    def test_suit_following(self):
        """Test that players must follow suit when able."""
        # Setup a game with controlled state
        self.game.round = 1  # Set to gameplay phase
        self.game.current_player_id = 0
        self.game.highest_bidder = 2  # Player 2 is declarer
        self.game.highest_bid = 3  # Bid 30
        self.game.trump_suit = 'D'  # Diamonds is trump
        
        # Enable strict suit following for this test
        self.game.test_strict_suit_following = True
        
        # Clear and set up player hands for testing
        self.game.hands = [[], [], [], []]
        
        # Player 0 has cards of different suits including diamonds
        self.game.hands[0] = [
            create_card('A', 'D'),  # Must be playable when diamonds led
            create_card('J', 'D'),  # Must be playable when diamonds led
            create_card('9', 'S'),  # Must not be playable when diamonds led and player has diamonds
            create_card('2', 'S')   # Must not be playable when diamonds led and player has diamonds
        ]
        
        # Player 1 has cards including diamonds
        self.game.hands[1] = [
            create_card('Q', 'D'),  # Must be playable when diamonds led
            create_card('6', 'D'),  # Must be playable when diamonds led
            create_card('A', 'H'),  # Special case: A♥ can be played anytime (high trump)
            create_card('K', 'S')   # Must not be playable when diamonds led and player has diamonds
        ]
        
        # Player 2 has cards including one diamond
        self.game.hands[2] = [
            create_card('4', 'S'),  # Must not be playable when diamonds led and player has diamonds
            create_card('5', 'H'),  # Must not be playable when diamonds led and player has diamonds
            create_card('9', 'D'),  # Must be playable when diamonds led
            create_card('J', 'C')   # Must not be playable when diamonds led and player has diamonds
        ]
        
        # Player 3 has cards including one diamond
        self.game.hands[3] = [
            create_card('5', 'C'),  # Must not be playable when diamonds led and player has diamonds
            create_card('J', 'S'),  # Must not be playable when diamonds led and player has diamonds
            create_card('9', 'H'),  # Must not be playable when diamonds led and player has diamonds
            create_card('9', 'D')   # Must be playable when diamonds led
        ]
        
        # Test when diamond is led, all players must follow suit if they can
        self.game.current_trick = [None] * self.game.num_players
        self.game.trick_starter = 2  # Player 2 leads
        self.game.current_trick[2] = create_card('5', 'D')  # Player 2 leads 5 of diamonds
        self.game.trick_lead_suit = 'D'
        
        # Test player 3's legal plays (should only be 9D)
        self.game.current_player_id = 3
        legal_plays = self.game.get_legal_plays()
        self.assertEqual(len(legal_plays), 1, "Player 3 should only have one legal play")
        self.assertEqual(legal_plays[0], 3, "9D should be the only legal play for Player 3")
        
        # Test player 0's legal plays (should only be AD and JD)
        self.game.current_player_id = 0
        legal_plays = self.game.get_legal_plays()
        self.assertEqual(len(legal_plays), 2, "Player 0 should have exactly two legal plays")
        self.assertTrue(0 in legal_plays, "AD should be a legal play for Player 0")
        self.assertTrue(1 in legal_plays, "JD should be a legal play for Player 0")
        self.assertFalse(2 in legal_plays, "9S should not be a legal play for Player 0")
        
        # Test player 1's legal plays (should be QD and 6D only - not allowing AH here)
        self.game.current_player_id = 1
        legal_plays = self.game.get_legal_plays()
        self.assertEqual(len(legal_plays), 2, "Player 1 should have exactly two legal plays")
        self.assertTrue(0 in legal_plays, "QD should be a legal play for Player 1")
        self.assertTrue(1 in legal_plays, "6D should be a legal play for Player 1")
        self.assertFalse(2 in legal_plays, "AH should not be a legal play for Player 1 when diamonds are led")
        self.assertFalse(3 in legal_plays, "KS should not be a legal play for Player 1")
        
        # Clean up
        delattr(self.game, 'test_strict_suit_following')

    def test_no_suit_restriction(self):
        """Test that when a player cannot follow suit, they can play any card."""
        # Setup a game with controlled state
        self.game.round = 1  # Set to gameplay phase
        self.game.current_player_id = 0
        self.game.highest_bidder = 2  # Player 2 is declarer
        self.game.highest_bid = 3  # Bid 30
        self.game.trump_suit = 'D'  # Diamonds is trump
        
        # Clear and set up player hands for testing
        self.game.hands = [[], [], [], []]
        
        # Player 0 has no diamonds
        self.game.hands[0] = [
            create_card('A', 'C'),
            create_card('9', 'S'),
            create_card('J', 'H'),
            create_card('2', 'S')
        ]
        
        # Test when diamond is led but player has no diamonds
        self.game.current_trick = [None] * self.game.num_players
        self.game.trick_starter = 2  # Player 2 leads
        self.game.current_trick[2] = create_card('5', 'D')  # Player 2 leads 5 of diamonds
        self.game.trick_lead_suit = 'D'
        
        # Test player 0's legal plays (should be all cards in hand)
        self.game.current_player_id = 0
        legal_plays = self.game.get_legal_plays()
        
        # Special case: Ace of Hearts should be playable (it's considered a trump)
        # Player has JH which is not a diamond but can be played since player has no diamonds
        
        # Check number of legal plays
        self.assertEqual(len(legal_plays), 4, "Player 0 should be able to play any card")
        
        # Check that each card is playable
        for i in range(4):
            self.assertTrue(i in legal_plays, f"Card at index {i} should be a legal play for Player 0")

    def test_hand_replenishment(self):
        """Test that player hands are replenished to 5 cards after the discard phase."""
        # Create a controlled game instance
        game = FortyfivesGame()
        # Set verbose mode manually
        game.verbose = True
        
        # Create a deck with enough cards to replenish all hands
        controlled_deck = []
        # Add 20 cards to the deck (more than enough to replenish all players' hands)
        for rank in ['A', 'K', 'Q', 'J', '5']:
            for suit in ['S', 'H', 'D', 'C']:
                controlled_deck.append(create_card(rank, suit))
        
        # Initialize the game
        game.init_game()
        
        # Override the dealer's deck
        game.dealer.deck = controlled_deck.copy()
        
        # Set up players with fewer than 5 cards
        game.hands = [
            [create_card('A', 'S'), create_card('K', 'S')],  # Player 0 has 2 cards
            [create_card('A', 'H'), create_card('K', 'H'), create_card('Q', 'H')],  # Player 1 has 3 cards
            [create_card('A', 'D'), create_card('K', 'D')],  # Player 2 has 2 cards
            [create_card('A', 'C'), create_card('K', 'C'), create_card('Q', 'C')],  # Player 3 has 3 cards
        ]
        
        # Set up pot (kitty)
        game.dealer.pot = [create_card('5', 'S'), create_card('5', 'H'), create_card('5', 'D')]
        
        # Set bidding complete
        game.phase = PHASE_DECLARATION
        game.highest_bidder = 0  # Player 0 is the highest bidder
        game.highest_bid = 15
        
        # Set up the auction starting player for the discard phase logic
        game.auction_starting_player = 0
        
        # Process declaration
        game.process_declaration('S')  # Spades as trump
        self.assertEqual(game.phase, PHASE_DISCARD, "Phase should change to PHASE_DISCARD")
        
        # Process discard for all players
        for player_id in range(game.num_players):
            game.current_player_id = player_id
            # Discard the first card in hand
            card_to_discard = game.hands[player_id][0]
            game.process_discard(card_to_discard)
            game.process_discard(DISCARD_DONE)
        
        # All players have discarded, game should move to PHASE_GAMEPLAY
        self.assertEqual(game.phase, PHASE_GAMEPLAY, "Phase should change to PHASE_GAMEPLAY")
        
        # Check that all players now have 5 cards
        for player_id in range(game.num_players):
            print(f"Player {player_id} hand: {[str(card) for card in game.hands[player_id]]}")
            self.assertEqual(len(game.hands[player_id]), 5, f"Player {player_id} should have 5 cards")
        
        # Check that the highest bidder received the kitty cards
        highest_bidder_has_kitty = any(str(card) == '5S' for card in game.hands[0])
        self.assertTrue(highest_bidder_has_kitty, "Highest bidder should have received the kitty")

    def test_kitty_timing(self):
        """Test that the kitty is given to the declarer after trump declaration but before the discard phase."""
        # Setup game
        self.game.phase = PHASE_AUCTION
        self.game.dealer_id = 0
        self.game.auction_starting_player = 1  # Player 1 starts auction
        self.game.current_player_id = 1
        
        # Set up dealer pot (kitty)
        self.game.dealer = FortyfivesDealer()
        self.game.dealer.pot = [
            create_card('A', 'S'),
            create_card('K', 'H'),
            create_card('Q', 'D')
        ]
        
        # Set up player hands
        self.game.hands = {i: [] for i in range(4)}
        self.game.hands[1] = [
            create_card('5', 'S'),
            create_card('J', 'H'),
            create_card('9', 'D'),
            create_card('3', 'C'),
            create_card('2', 'S')
        ]
        
        # Player 1 makes a bid
        self.game.highest_bidder = 1
        self.game.highest_bid = 1  # BID_20
        
        # Player 1 declares trump
        self.game.process_declaration(SUIT_HEARTS)
        
        # Check that the kitty has been added to the declarer's hand
        self.assertEqual(len(self.game.dealer.pot), 0, "Kitty should be empty after declaration")
        self.assertEqual(len(self.game.hands[1]), 8, "Declarer should have 8 cards after receiving kitty")
        
        # Check that we're in the discard phase
        self.assertEqual(self.game.phase, PHASE_DISCARD, "Game should be in discard phase after declaration")
    
    def test_can_always_play_trump(self):
        """Test that players can always play trump cards even when following suit is possible."""
        # Setup game
        self.game.phase = PHASE_GAMEPLAY
        self.game.trump_suit = 'D'  # Diamonds is trump
        
        # Set up player hands
        self.game.hands = {i: [] for i in range(4)}
        self.game.hands[0] = [
            create_card('Q', 'H'),  # Can play when hearts is led
            create_card('3', 'D'),  # Can always play (trump)
            create_card('K', 'H'),  # Can play when hearts is led
            create_card('A', 'S')   # Can play when spades is led or no lead suit cards
        ]
        
        # Set up trick with hearts led
        self.game.current_trick = [None] * 4
        self.game.trick_starter = 2  # Player 2 leads
        self.game.current_trick[2] = create_card('J', 'H')  # Player 2 leads J of hearts
        self.game.trick_lead_suit = 'H'
        
        # Get legal plays for player 0
        self.game.current_player_id = 0
        legal_plays = self.game.get_legal_plays()
        
        # Check that both hearts and the trump can be played
        self.assertIn(0, legal_plays, "QH should be a legal play")
        self.assertIn(1, legal_plays, "3D (trump) should be a legal play")
        self.assertIn(2, legal_plays, "KH should be a legal play")
        self.assertNotIn(3, legal_plays, "AS should not be a legal play when hearts is led")
        
        # Now try with spades led
        self.game.current_trick[2] = create_card('J', 'S')  # Player 2 leads J of spades
        self.game.trick_lead_suit = 'S'
        
        legal_plays = self.game.get_legal_plays()
        
        # Check that spades and the trump can be played
        self.assertNotIn(0, legal_plays, "QH should not be a legal play when spades is led")
        self.assertIn(1, legal_plays, "3D (trump) should always be a legal play")
        self.assertNotIn(2, legal_plays, "KH should not be a legal play when spades is led")
        self.assertIn(3, legal_plays, "AS should be a legal play when spades is led")

    def test_high_trump_reneging(self):
        """Test that high trumps (5, J, A♥) can be withheld when a low trump is led."""
        # Setup game
        self.game.phase = PHASE_GAMEPLAY
        self.game.trump_suit = 'D'  # Diamonds is trump
        
        # Set up player hands with high and low trumps
        self.game.hands = {i: [] for i in range(4)}
        self.game.hands[0] = [
            create_card('5', 'D'),  # High trump that can be withheld
            create_card('J', 'D'),  # High trump that can be withheld
            create_card('A', 'H'),  # Special trump that can be withheld
            create_card('Q', 'H')   # Non-trump
        ]
        
        # Set up trick with a low trump led
        self.game.current_trick = [None] * 4
        self.game.trick_starter = 2  # Player 2 leads
        self.game.current_trick[2] = create_card('3', 'D')  # Player 2 leads 3 of diamonds (low trump)
        self.game.trick_lead_suit = 'D'
        
        # Get legal plays for player 0
        self.game.current_player_id = 0
        legal_plays = self.game.get_legal_plays()
        
        # Check that any card can be played when a low trump is led
        self.assertEqual(len(legal_plays), 4, "All 4 cards should be legal plays when a low trump is led")
        
        # Now try with a high trump led
        self.game.current_trick[2] = create_card('5', 'D')  # Player 2 leads 5 of diamonds (high trump)
        
        legal_plays = self.game.get_legal_plays()
        
        # Should only be able to play trumps when a high trump is led
        self.assertEqual(len(legal_plays), 3, "Only the 3 trump cards should be playable when a high trump is led")
        self.assertIn(0, legal_plays, "5D should be a legal play when a high trump is led")
        self.assertIn(1, legal_plays, "JD should be a legal play when a high trump is led")
        self.assertIn(2, legal_plays, "AH should be a legal play as a special trump when a high trump is led")
        self.assertNotIn(3, legal_plays, "QH should not be a legal play when a high trump is led")

    def test_highest_trump_bonus_scoring(self):
        """Test that the +5 points bonus is correctly awarded to the partnership that played the highest trump."""
        # Setup game
        self.game.phase = PHASE_GAMEPLAY
        self.game.trump_suit = 'D'  # Diamonds is trump
        
        # West (player 3) played the highest trump
        self.game.highest_trump_player = 3
        
        # Assign tricks
        self.game.tricks_won = [1, 1, 1, 2]  # NS: 2 tricks, EW: 3 tricks
        
        # East (player 1) was the highest bidder with a bid of 20
        self.game.highest_bidder = 1
        self.game.highest_bid = 1  # BID_20 (which is worth 20 points)
        
        # Score the hand
        self.game.score_hand()
        
        # Check the scores
        # NS: 2 tricks x 5 = 10 points
        # EW: 3 tricks x 5 = 15 points + 5 for highest trump = 20 points
        self.assertEqual(self.game.hand_points[0], 10, "NS should get 10 points (2 tricks x 5)")
        self.assertEqual(self.game.hand_points[1], 20, "EW should get 20 points (15 for tricks + 5 for highest trump)")

    def test_first_trick_leader(self):
        """Test that the player to the left of the highest bidder (declarer) leads the first trick."""
        # Setup game
        self.game.phase = PHASE_AUCTION
        self.game.dealer_id = 0
        self.game.auction_starting_player = 1  # Player 1 starts auction
        self.game.current_player_id = 1
        
        # Set up player hands
        self.game.hands = {i: [] for i in range(4)}
        for i in range(4):
            self.game.hands[i] = [create_card('K', 'S'), create_card('Q', 'H'), 
                                  create_card('J', 'D'), create_card('T', 'C'), 
                                  create_card('9', 'S')]
        
        # Set highest bidder to East (player 1)
        self.game.highest_bidder = 1
        self.game.highest_bid = BID_20
        
        # Declare trump
        self.game.process_declaration(SUIT_HEARTS)
        
        # Process discards
        for i in range(4):
            self.game.current_player_id = (self.game.dealer_id + 1 + i) % 4
            self.game.process_discard(0)  # Discard first card
            self.game.process_discard(DISCARD_DONE)
        
        # After discards, we should be in gameplay phase
        self.assertEqual(self.game.phase, PHASE_GAMEPLAY, "Game should be in gameplay phase after discards")
        
        # The trick starter should be South (player 2), which is to the left of East (player 1)
        self.assertEqual(self.game.trick_starter, 2, "South (player 2) should lead the first trick as they're to the left of East (declarer)")
        self.assertEqual(self.game.current_player_id, 2, "Current player should be South (player 2)")

    def test_can_play_any_trump_with_lead_suit(self):
        """Test that a player can play any of their trump cards even if they have cards of the lead suit."""
        # Setup game to replicate step 22 issue
        game = FortyfivesGame()
        game.phase = PHASE_GAMEPLAY
        game.trump_suit = 'S'  # Spades is trump, as in the example game
        
        # Set up player's hand exactly like West in step 22
        game.hands = {3: []}  # West's hand
        game.hands[3] = [
            create_card('5', 'S'),  # Trump card
            create_card('8', 'H'),  # Card of lead suit
            create_card('8', 'S'),  # Trump card
            create_card('3', 'S'),  # Trump card
            create_card('A', 'H')   # Special trump (AH)
        ]
        
        # Set up a trick with Hearts led, exactly as in the example
        game.current_trick = [None] * game.num_players
        game.trick_starter = 2  # South leads
        game.current_trick[2] = create_card('4', 'H')  # South leads 4 of Hearts
        game.trick_lead_suit = 'H'
        
        # Get legal plays for West (Player 3)
        game.current_player_id = 3
        legal_plays = game.get_legal_plays()
        
        # Print debugging info
        print(f"Hand: {[str(card) for card in game.hands[3]]}")
        print(f"Lead suit: {game.trick_lead_suit}")
        print(f"Trump suit: {game.trump_suit}")
        print(f"Legal plays: {legal_plays}")
        print(f"Legal cards: {[str(game.hands[3][i]) for i in legal_plays]}")
        
        # Player should be able to play all 5 cards in their hand
        self.assertEqual(len(legal_plays), 5, "West should be able to play all 5 cards in their hand")
        
        # Verify all cards are legal plays
        legal_cards = [str(game.hands[3][i]) for i in legal_plays]
        self.assertIn('5S', legal_cards, "5S (trump) should be a legal play")
        self.assertIn('8H', legal_cards, "8H (lead suit) should be a legal play")
        self.assertIn('8S', legal_cards, "8S (trump) should be a legal play")
        self.assertIn('3S', legal_cards, "3S (trump) should be a legal play")
        self.assertIn('AH', legal_cards, "AH (special trump) should be a legal play")
        
        # Now let's try to debug by checking exactly what the get_legal_plays function is doing
        # Check the internal state that affects legal plays
        lead_suit_cards = [i for i, card in enumerate(game.hands[3]) if card.suit == 'H']
        trump_cards = [i for i, card in enumerate(game.hands[3]) 
                      if card.suit == 'S' or (card.rank == 'A' and card.suit == 'H')]
        
        print(f"Lead suit cards indices: {lead_suit_cards}")
        print(f"Trump cards indices: {trump_cards}")
        
        # Check if the issue is with the combination logic
        combined_plays = sorted(list(set(lead_suit_cards + trump_cards)))
        print(f"Combined plays: {combined_plays}")
        print(f"Combined cards: {[str(game.hands[3][i]) for i in combined_plays]}")

if __name__ == '__main__':
    unittest.main() 