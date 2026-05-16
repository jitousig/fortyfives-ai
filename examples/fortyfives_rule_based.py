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
from fortyfives.games.fortyfives.card import SUITS, RANKS, get_card_rank

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
        
        # Default: deterministic fallback (see note on benchmark determinism)
        return min(state['legal_actions'].keys())
    
    # Suit -> trump-declaration env action id.
    _SUIT_TO_DECL = {'S': 5, 'H': 6, 'D': 7, 'C': 8}

    def _supported_bid(self, hand):
        '''
        Top-trump bid model. For each suit S consider it as trump:
          5_S + J_S + A♥          -> 30
          5_S + J_S               -> 25
          5_S                     -> 20
          J_S + (>=1 other S card OR A♥) -> 20   (no-5 backup path)
          (none of the above)            -> pass
        A♥ is ALWAYS trump (any declared suit), so it counts for every S:
        in the 30 test, and as the J-path "backup card" for any suit
        (J♠ + A♥ with no other spades still bids 20 on spades, since
        J♠+A♥ are the 2nd/3rd best trumps there).
        Returns (suit or None, level in {30,25,20,0}). Deterministic:
        among suits reaching the top level, prefer the most effective
        trumps (length wins tricks in this 5-pts/trick game), then a
        fixed suit order.
        '''
        has_AH = any(c.rank == 'A' and c.suit == 'H' for c in hand)
        best = None  # ((level, trump_count, -suit_idx), suit, level)
        for s in SUITS:
            count_s = sum(1 for c in hand if c.suit == s)
            has_5 = any(c.rank == '5' and c.suit == s for c in hand)
            has_J = any(c.rank == 'J' and c.suit == s for c in hand)
            if has_5 and has_J and has_AH:
                level = 30
            elif has_5 and has_J:
                level = 25
            elif has_5:
                level = 20
            elif has_J and (count_s >= 2 or has_AH):
                level = 20
            else:
                continue
            tcount = sum(1 for c in hand
                         if c.suit == s or (c.rank == 'A' and c.suit == 'H'))
            key = (level, tcount, -SUITS.index(s))
            if best is None or key > best[0]:
                best = (key, s, level)
        if best is None:
            return None, 0
        return best[1], best[2]

    def _bid_strategy(self, raw_obs, legal_actions):
        '''Bid exactly the level the hand's top trumps support; if that
        level is unavailable (already outbid), pass.'''
        _, level = self._supported_bid(raw_obs['hand'])
        desired = {30: 3, 25: 2, 20: 1, 0: 0}[level]
        if desired in legal_actions:
            return desired
        # Supported bid taken / illegal -> pass; hold only if forced;
        # deterministic fallback (rule-based must stay reproducible).
        if 0 in legal_actions:
            return 0
        if 4 in legal_actions:
            return 4
        return min(legal_actions.keys())

    def _choose_trump(self, raw_obs, legal_actions):
        '''Declare the suit the bid was based on (same model as the bid,
        so they always agree). Fallback: longest suit, deterministic.'''
        hand = raw_obs['hand']
        suit, _ = self._supported_bid(hand)
        if suit is not None and self._SUIT_TO_DECL[suit] in legal_actions:
            return self._SUIT_TO_DECL[suit]

        counts = {s: sum(1 for c in hand if c.suit == s) for s in SUITS}
        for s in sorted(SUITS, key=lambda x: (-counts[x], SUITS.index(x))):
            if self._SUIT_TO_DECL[s] in legal_actions:
                return self._SUIT_TO_DECL[s]
        return min(legal_actions.keys())
    
    def _discard_strategy(self, raw_obs, legal_actions):
        '''
        Strategy for discarding.

        Throw EVERY non-trump card (draw fresh in the replenish step) and,
        if more than 5 trump remain, keep only the 5 highest-ranked trump.
        Ace of Hearts is always trump in this game (get_card_rank == 1001),
        so it is treated as trump here — discarding it would be a blunder.

        One card is discarded per call; DISCARD_DONE (16) ends the phase.
        Ordering uses the engine's authoritative get_card_rank so "lowest
        non-trump" / "5 highest trump" match real play, not a hand-rolled
        order. Deterministic (no np.random) — rule-based must stay a
        reproducible paired-eval benchmark.
        '''
        hand = raw_obs['hand']
        trump_suit = raw_obs['trump_suit']

        def is_trump(card):
            return card.suit == trump_suit or (card.rank == 'A' and card.suit == 'H')

        non_trump = [i for i, c in enumerate(hand) if not is_trump(c)]
        trump = [i for i, c in enumerate(hand) if is_trump(c)]

        # 1. Discard the weakest non-trump still in hand.
        if non_trump:
            non_trump.sort(key=lambda i: (get_card_rank(hand[i], trump_suit), i))
            for i in non_trump:
                if i in legal_actions:
                    return i

        # 2. Only trump left: if holding more than 5, shed the weakest
        #    until exactly the 5 highest-ranked remain.
        elif len(trump) > 5:
            trump.sort(key=lambda i: (get_card_rank(hand[i], trump_suit), i))
            for i in trump:  # weakest first
                if i in legal_actions:
                    return i

        # 3. Nothing to throw (<=5 trump, no non-trump): done.
        if 16 in legal_actions:
            return 16

        # Deterministic fallback (e.g. an index became illegal): never random.
        return min(legal_actions.keys())
    
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