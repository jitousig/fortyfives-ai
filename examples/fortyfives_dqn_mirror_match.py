#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Mirror match evaluation: DQN vs DQN for Fortyfives
'''

import os
import argparse
import numpy as np
import torch
import sys
import time

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed, get_device

# Add safe globals for PyTorch 2.6+
from torch.serialization import add_safe_globals
add_safe_globals(['rlcard.agents.dqn_agent', 'rlcard.agents.dqn_agent.DQNAgent'])

# Try to register the Fortyfives environment
from rlcard.envs.registration import register, registry
try:
    if 'fortyfives' not in registry.env_specs:
        register(
            env_id='fortyfives',
            entry_point='fortyfives.envs.fortyfives_env:FortyfivesEnv',
        )
        print("Successfully registered fortyfives environment")
    else:
        print("fortyfives environment already registered")
except Exception as e:
    print(f"Error registering environment: {e}")
    sys.exit(1)

def mirror_match(args):
    # Set the seed
    set_seed(args.seed)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        sys.exit(1)
    
    print(f"Loading model from {args.model_path}")
    
    # Try to make the environment
    try:
        env = rlcard.make('fortyfives', config={'seed': args.seed})
        print(f"Successfully created fortyfives environment")
        print(f"Number of players: {env.num_players}")
    except Exception as e:
        print(f"Error creating environment: {e}")
        sys.exit(1)
    
    # Load the trained agent
    try:
        device = get_device()
        print("Loading model with weights_only=False for safety")
        trained_agent = torch.load(args.model_path, map_location=device, weights_only=False)
        print(f"Successfully loaded trained agent")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Tracking wins per position
    position_wins = {i: 0 for i in range(env.num_players)}
    total_games = args.num_games
    rewards = {i: [] for i in range(env.num_players)}
    
    print(f"Running mirror match over {total_games} games...")
    start_time = time.time()
    
    # Run evaluation games
    for game_id in range(total_games):
        # All positions use the trained agent
        agents = [trained_agent for _ in range(env.num_players)]
        env.set_agents(agents)
        
        # Reset the environment
        state, player_id = env.reset()
        done = False
        step_count = 0
        
        # Play until done
        while not done and step_count < 1000:  # Safety limit
            step_count += 1
            
            # Get action from current agent
            action = env.agents[player_id].step(state)
            
            # Step the environment
            next_state, next_player_id = env.step(action)
            
            # Check if the game is done
            if step_count >= 1000 or (hasattr(env.game, 'is_over') and env.game.is_over()):
                done = True
                payoffs = env.get_payoffs()
                
                # Record rewards for all positions
                for pos in range(env.num_players):
                    reward = payoffs[pos]
                    rewards[pos].append(reward)
                    
                    # Check if this position won
                    if reward > 0:
                        position_wins[pos] += 1
            
            # Move to the next state
            state = next_state
            player_id = next_player_id
        
        # Print progress
        if (game_id + 1) % 10 == 0:
            print(f"Completed {game_id + 1}/{total_games} games")
    
    # Print evaluation results
    elapsed_time = time.time() - start_time
    print(f"\nMirror match complete!")
    print(f"Games played: {total_games}")
    print("\nWin rate by position:")
    
    for pos in range(env.num_players):
        win_rate = position_wins[pos] / total_games * 100 if total_games > 0 else 0
        avg_reward = np.mean(rewards[pos]) if rewards[pos] else 0
        print(f"Position {pos}: {position_wins[pos]} wins ({win_rate:.2f}%), avg reward: {avg_reward:.4f}")
    
    print(f"\nTime taken: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Mirror Match DQN vs DQN for Fortyfives")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model_path', type=str, default='experiments/fortyfives_dqn_selfplay_rewards/model.pth', 
                        help='Path to the trained model')
    parser.add_argument('--num_games', type=int, default=100, 
                        help='Number of games to evaluate')
    
    args = parser.parse_args()
    mirror_match(args) 