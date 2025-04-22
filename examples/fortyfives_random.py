#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
A toy example of running Fortyfives with random agents
'''

import os
import argparse

import rlcard
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import print_card
from fortyfives.games.fortyfives.card import SUITS

# Make environment
env = rlcard.make('fortyfives')

# Set random seed
env.seed(0)

# Enable verbose mode in the game
env.game.verbose = True

# Player names for direction
PLAYER_NAMES = ['North', 'East', 'South', 'West']

# Map suit indices to unicode suit symbols for pretty printing
SUIT_SYMBOLS = {
    0: '♠',  # Spades
    1: '♥',  # Hearts
    2: '♦',  # Diamonds
    3: '♣',  # Clubs
}

def format_card(card):
    """Format a card object to a readable string."""
    # Check if card is string or FortyfivesCard
    if hasattr(card, 'rank') and hasattr(card, 'suit'):
        suit_idx = SUITS.index(card.suit) if card.suit in SUITS else 0
        return f"{card.rank}{SUIT_SYMBOLS[suit_idx]}"
    return str(card)

def format_hand(hand):
    """Format a list of cards to a readable string."""
    if not hand:
        return "Empty hand"
    return ', '.join([format_card(card) for card in hand])

def get_phase_name(phase_id):
    """Convert phase ID to a readable name."""
    phases = {
        0: "Initial",
        1: "Auction",
        2: "Declaration",
        3: "Discard",
        4: "Gameplay",
    }
    return phases.get(phase_id, f"Unknown Phase ({phase_id})")

def get_bid_name(bid_id):
    """Convert bid ID to a readable name."""
    bids = {
        0: "Pass",
        1: "Bid 20",
        2: "Bid 25",
        3: "Bid 30",
        4: "Hold",
    }
    return bids.get(bid_id, f"Unknown Bid ({bid_id})")

def get_suit_name(suit_id):
    """Convert suit ID to a readable name."""
    suits = {
        0: "Spades",
        1: "Hearts",
        2: "Diamonds",
        3: "Clubs",
    }
    return suits.get(suit_id, f"Unknown Suit ({suit_id})")

def get_action_name(action, state):
    """Convert action ID to a readable description based on game phase."""
    if state['phase'] == 1:  # Auction
        return get_bid_name(action)
    elif state['phase'] == 2:  # Declaration
        return f"Declare trump: {get_suit_name(action)}"
    elif state['phase'] == 3:  # Discard
        if action == 16:  # DISCARD_DONE
            return "Done discarding"
        else:
            card = state['hand'][action]
            return f"Discard: {format_card(card)}"
    elif state['phase'] == 4:  # Gameplay
        if action < len(state['hand']):
            card = state['hand'][action]
            return f"Play: {format_card(card)}"
    
    return f"Unknown Action: {action}"

# Set up agents
agents = []
for i in range(env.num_players):
    agent = RandomAgent(num_actions=env.num_actions)
    agents.append(agent)

# Start the game
print(f"Number of actions: {env.num_actions}")
print(f"Number of players: {env.num_players}")
print(f"State shape: {env.state_shape}")
print(f"Action shape: {env.action_shape}")
print("")

# Play multiple episodes
num_episodes = 1
max_steps = 100  # Prevent infinite loops during development

for episode in range(num_episodes):
    print(f"==========\nEPISODE {episode}\n==========")
    print("Starting game...")
    
    # Initialize the game
    state, player_id = env.reset()
    dealer_id = state['raw_obs']['dealer']
    print(f"Dealer: {PLAYER_NAMES[dealer_id]} (Player {dealer_id})")
    print("")
    
    step = 0
    done = False
    
    while not done and step < max_steps:
        # Get player's name based on ID
        player_name = PLAYER_NAMES[player_id]
        
        # Current phase
        phase_name = get_phase_name(state['raw_obs']['phase'])
        
        print(f"Step {step}, {player_name} (Player {player_id}), Phase: {state['raw_obs']['phase']}")
        print(f"Phase: {phase_name}")
        print(f"{player_name}'s hand: {format_hand(state['raw_obs']['hand'])}")
        
        # Print current trick if in gameplay phase
        if state['raw_obs']['phase'] == 4:
            print("Current trick:", end=" ")
            trick = state['raw_obs']['current_trick']
            if all(card is None for card in trick):
                print("No cards played yet")
            else:
                played_cards = []
                for i, card in enumerate(trick):
                    if card is not None:
                        played_cards.append(f"{PLAYER_NAMES[i]}: {format_card(card)}")
                print(", ".join(played_cards))
        
        # Print legal actions
        print("Legal actions:")
        for action in state['legal_actions']:
            action_name = get_action_name(action, state['raw_obs'])
            print(f"  {action}: {action_name}")
        
        # Take a random action
        action = agents[player_id].step(state)
        print(f"Action taken: {action} ({get_action_name(action, state['raw_obs'])})")
        
        # Step the environment
        next_state, next_player_id = env.step(action, agents[player_id].use_raw)
        
        # Update for next step
        state = next_state
        player_id = next_player_id
        step += 1
        
        # Check if the game is over
        try:
            if callable(env.game.is_over):
                game_over = env.game.is_over()
            else:
                # Fallback if is_over is a property not a method
                game_over = env.game.is_over
            
            if game_over:
                done = True
                print(f"Game over after {step} steps")
                
                # Get payoffs
                payoffs = env.get_payoffs()
                print(f"Payoffs: {payoffs}")
                
                # Get perfect information
                perfect_info = env.get_perfect_information()
                print(f"Final game scores: {perfect_info['points']}")
        except Exception as e:
            print(f"Error checking game over: {e}")
        
        print("")  # Add a blank line between steps
    
    if step >= max_steps:
        print(f"Maximum steps reached ({max_steps}). Ending early.")

print("Done!") 