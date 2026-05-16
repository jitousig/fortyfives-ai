#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
A rule-based agent for playing Fortyfives
'''

import os
import argparse
import numpy as np
import rlcard
from rlcard.agents.random_agent import RandomAgent
from fortyfives.games.fortyfives.card import SUITS, RANKS

class RuleBasedAgent:
    '''
    A rule-based agent that follows basic Fortyfives strategy
    '''
    
    def __init__(self, num_actions):
        '''
        Initialize the agent
        
        Args:
            num_actions (int): Size of the action space
        '''
        self.use_raw = True
        self.num_actions = num_actions
        
    def step(self, state):
        '''
        Take a step in the game
        
        Args:
            state (dict): Current state of the game
            
        Returns:
            action (int): Action to take
        '''
        raw_obs = state['raw_obs']
        phase = raw_obs['phase']
        
        # Phase 1: Bidding
        if phase == 1:
            return self._bid_strategy(raw_obs, state['legal_actions'])
        
        # Phase 2: Trump declaration
        elif phase == 2:
            return self._choose_trump(raw_obs, state['legal_actions'])
        
        # Phase 3: Discarding
        elif phase == 3:
            return self._discard_strategy(raw_obs, state['legal_actions'])
        
        # Phase 4: Gameplay
        elif phase == 4:
            return self._play_strategy(raw_obs, state['legal_actions'])
        
        # Default: random action
        return np.random.choice(list(state['legal_actions'].keys()))
    
    def _bid_strategy(self, raw_obs, legal_actions):
        '''
        Strategy for bidding phase
        '''
        hand = raw_obs['hand']
        
        # Count high cards by suit
        suit_counts = {suit: 0 for suit in SUITS}
        for card in hand:
            suit_counts[card.suit] += 1
        
        # Count high cards (5s, Js, Aces)
        high_cards = [card for card in hand if card.rank in ['5', 'J', 'A']]
        num_high_cards = len(high_cards)
        
        # Find best suit for trump
        best_suit = max(suit_counts, key=suit_counts.get)
        best_suit_count = suit_counts[best_suit]
        
        # Choose bid based on suit count and high cards
        if best_suit_count >= 4 and num_high_cards >= 3:
            # Bid 30 if we have a good hand
            if 3 in legal_actions:  # Bid 30
                return 3
            elif 2 in legal_actions:  # Bid 25
                return 2
            elif 1 in legal_actions:  # Bid 20
                return 1
        elif best_suit_count >= 3 and num_high_cards >= 2:
            # Bid 25 with decent hand
            if 2 in legal_actions:  # Bid 25
                return 2
            elif 1 in legal_actions:  # Bid 20
                return 1
        elif best_suit_count >= 2 and num_high_cards >= 1:
            # Bid 20 with mediocre hand
            if 1 in legal_actions:  # Bid 20
                return 1
        
        # Pass with a poor hand
        if 0 in legal_actions:  # Pass
            return 0
        
        # If bidding has gone around and we're forced to hold
        if 4 in legal_actions:  # Hold
            return 4
            
        # If no suitable action found, take a random legal action
        return np.random.choice(list(legal_actions.keys()))
    
    def _choose_trump(self, raw_obs, legal_actions):
        '''
        Strategy for choosing trump
        '''
        hand = raw_obs['hand']
        
        # Count cards by suit
        suit_counts = {suit: 0 for suit in SUITS}
        suit_values = {suit: 0 for suit in SUITS}
        
        for card in hand:
            suit = card.suit
            suit_counts[suit] += 1
            
            # Assign values to high cards
            if card.rank == '5':
                suit_values[suit] += 5
            elif card.rank == 'J':
                suit_values[suit] += 3
            elif card.rank == 'A':
                suit_values[suit] += 4
            elif card.rank == 'K':
                suit_values[suit] += 2
            elif card.rank == 'Q':
                suit_values[suit] += 1
        
        # Find best suit based on count and high cards
        best_suit = max(suit_values, key=suit_values.get)
        
        # Map suit to action
        suit_to_action = {
            'S': 5,  # Spades
            'H': 6,  # Hearts
            'D': 7,  # Diamonds
            'C': 8,  # Clubs
        }
        
        # Choose the best suit if legal
        if suit_to_action[best_suit] in legal_actions:
            return suit_to_action[best_suit]
            
        # Fall back to legal action with most cards
        for suit, count in sorted(suit_counts.items(), key=lambda x: x[1], reverse=True):
            if suit_to_action[suit] in legal_actions:
                return suit_to_action[suit]
                
        # If no suitable action found, take a random legal action
        return np.random.choice(list(legal_actions.keys()))
    
    def _discard_strategy(self, raw_obs, legal_actions):
        '''
        Strategy for discarding
        '''
        hand = raw_obs['hand']
        trump_suit = raw_obs['trump_suit']
        
        # If done discarding is an option and we have 5 cards, we're done
        if 16 in legal_actions and len(hand) == 5:
            return 16
        
        # Assign values to cards
        card_values = {}
        for i, card in enumerate(hand):
            value = 0
            
            # Keep trump cards
            if card.suit == trump_suit:
                value += 10
                
                # High trump cards are valuable
                if card.rank == '5':
                    value += 10
                elif card.rank == 'J':
                    value += 8
                elif card.rank == 'A':
                    value += 6
                elif card.rank == 'K':
                    value += 4
                elif card.rank == 'Q':
                    value += 2
            else:
                # Non-trump high cards
                if card.rank == '5':
                    value += 5
                elif card.rank == 'A':
                    value += 4
                elif card.rank == 'K':
                    value += 2
                elif card.rank == 'Q':
                    value += 1
                    
            card_values[i] = value
        
        # Discard the lowest value card
        if not card_values:
            return np.random.choice(list(legal_actions.keys()))
        worst_card_idx = min(card_values, key=card_values.get)
        if worst_card_idx in legal_actions:
            return worst_card_idx

        # If no suitable action found, take a random legal action
        return np.random.choice(list(legal_actions.keys()))
    
    def _play_strategy(self, raw_obs, legal_actions):
        '''
        Strategy for gameplay
        '''
        hand = raw_obs['hand']
        current_trick = raw_obs['current_trick']
        trump_suit = raw_obs['trump_suit']
        
        # If we're leading
        if all(card is None for card in current_trick):
            # Lead with highest trump if possible
            for i, card in enumerate(hand):
                if card.suit == trump_suit and i in legal_actions:
                    return i
                    
            # Otherwise lead highest card
            for i, card in enumerate(hand):
                if i in legal_actions:
                    return i
        
        # Not leading - find cards that have been played in this trick
        led_suit = None
        highest_card = None
        highest_value = -1
        
        for i, card in enumerate(current_trick):
            if card is not None:
                if led_suit is None:
                    led_suit = card.suit
                
                # Calculate card value
                value = self._card_value(card, led_suit, trump_suit)
                if value > highest_value:
                    highest_value = value
                    highest_card = card
        
        # Try to follow suit
        follow_suit_options = []
        for i, card in enumerate(hand):
            if i in legal_actions:
                if card.suit == led_suit:
                    card_value = self._card_value(card, led_suit, trump_suit)
                    follow_suit_options.append((i, card_value))
        
        if follow_suit_options:
            # Try to win trick
            winning_options = [(i, val) for i, val in follow_suit_options if val > highest_value]
            if winning_options:
                # Play lowest winning card
                return min(winning_options, key=lambda x: x[1])[0]
            else:
                # Can't win, so play lowest card
                return min(follow_suit_options, key=lambda x: x[1])[0]
        
        # Can't follow suit - try to win with trump
        trump_options = []
        for i, card in enumerate(hand):
            if i in legal_actions:
                if card.suit == trump_suit:
                    card_value = self._card_value(card, led_suit, trump_suit)
                    trump_options.append((i, card_value))
        
        if trump_options:
            winning_options = [(i, val) for i, val in trump_options if val > highest_value]
            if winning_options:
                # Play lowest winning trump
                return min(winning_options, key=lambda x: x[1])[0]
        
        # Can't win - play lowest card
        return min(legal_actions)
    
    def _card_value(self, card, led_suit, trump_suit):
        '''
        Calculates the value of a card for determining trick winner
        '''
        # Rank values in Fortyfives
        rank_values = {
            '5': 14,
            'J': 13,
            'A': 12,
            'K': 11,
            'Q': 10,
            '10': 9,
            '9': 8,
            '8': 7,
            '7': 6,
            '6': 5,
            '4': 4,
            '3': 3,
            '2': 2
        }
        
        # Trump suit wins all
        if card.suit == trump_suit:
            return 100 + rank_values.get(card.rank, 0)
        # Led suit wins non-trump
        elif card.suit == led_suit:
            return rank_values.get(card.rank, 0)
        # Other suits are lowest
        else:
            return 0

def main():
    # Make environment
    env = rlcard.make('fortyfives')
    
    # Set random seed
    env.seed(42)
    
    # Set up agents
    agents = []
    for i in range(env.num_players):
        if i % 2 == 0:  # Players 0 and 2 are rule-based (team 1)
            agent = RuleBasedAgent(num_actions=env.num_actions)
        else:  # Players 1 and 3 are random (team 2)
            agent = RandomAgent(num_actions=env.num_actions)
        agents.append(agent)
    
    env.set_agents(agents)
    
    # Play multiple games
    num_games = 100
    team1_wins = 0
    team2_wins = 0
    
    print(f"Playing {num_games} games: Rule-based (team 1) vs Random (team 2)")
    
    for game in range(num_games):
        print(f"Game {game+1}/{num_games}")
        
        # Initialize the game
        state, player_id = env.reset()
        
        done = False
        while not done:
            action = agents[player_id].step(state)
            state, player_id = env.step(action)
            
            # Check if the game is over
            if env.game.is_over():
                done = True
        
        # Get the scores
        perfect_info = env.get_perfect_information()
        ns_score = perfect_info['points'].get(0, 0)
        ew_score = perfect_info['points'].get(1, 0)
        
        # Determine winner
        if ns_score >= 125 or ew_score <= -125:
            team1_wins += 1
            print(f"Game {game+1}: Team 1 (Rule-based) wins! Score: {ns_score} vs {ew_score}")
        elif ew_score >= 125 or ns_score <= -125:
            team2_wins += 1
            print(f"Game {game+1}: Team 2 (Random) wins! Score: {ns_score} vs {ew_score}")
    
    # Print overall results
    print("\n===== RESULTS =====")
    print(f"Team 1 (Rule-based) wins: {team1_wins}/{num_games} ({team1_wins/num_games*100:.1f}%)")
    print(f"Team 2 (Random) wins: {team2_wins}/{num_games} ({team2_wins/num_games*100:.1f}%)")
    print(f"Draws: {num_games - team1_wins - team2_wins}")

if __name__ == "__main__":
    main() 