#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Evaluating a trained DQN agent against random agents for Fortyfives
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

def evaluate(args):
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
    
    # Create random agents for other positions
    random_agents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players - 1)]
    
    # Total number of wins and games
    wins = 0
    total_games = args.num_games
    rewards = []
    
    print(f"Evaluating over {total_games} games...")
    start_time = time.time()
    
    # Run evaluation games
    for game_id in range(total_games):
        # Put the trained agent in the specified position (default: 0)
        agents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)]
        agents[args.position] = trained_agent
        
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
                
                # Record reward for the trained agent
                reward = payoffs[args.position]
                rewards.append(reward)
                
                # Check if the agent won
                if reward > 0:
                    wins += 1
            
            # Move to the next state
            state = next_state
            player_id = next_player_id
        
        # Print progress
        if (game_id + 1) % 10 == 0:
            print(f"Completed {game_id + 1}/{total_games} games")
    
    # Print evaluation results
    elapsed_time = time.time() - start_time
    win_rate = wins / total_games * 100 if total_games > 0 else 0
    avg_reward = np.mean(rewards) if rewards else 0
    
    print(f"\nEvaluation complete!")
    print(f"Games played: {total_games}")
    print(f"Wins: {wins}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Time taken: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluate DQN Agent for Fortyfives")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model_path', type=str, default='experiments/fortyfives_dqn_selfplay/model.pth', 
                        help='Path to the trained model')
    parser.add_argument('--num_games', type=int, default=100, 
                        help='Number of games to evaluate')
    parser.add_argument('--position', type=int, default=0, 
                        help='Position of the trained agent (0-3)')
    
    args = parser.parse_args()
    evaluate(args) 