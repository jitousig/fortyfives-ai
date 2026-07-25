#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Evaluating a trained DQN agent for Fortyfives
'''

import os
import argparse
import numpy as np
import torch
import sys

import rlcard
from rlcard.agents import DQNAgent
from rlcard.utils import get_device, set_seed

# Allow loading the DQNAgent class
torch.serialization.add_safe_globals([rlcard.agents.dqn_agent.DQNAgent])

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
    # Check if cuda is available
    device = get_device()
    print(f"Using device: {device}")
    
    # Set the seed
    set_seed(args.seed)
    
    # Try to load the trained model
    try:
        print(f"Trying to load model from {args.model_path}")
        if args.use_unsafe_loading:
            # Use unsafe loading (for PyTorch 2.6+)
            agent = torch.load(args.model_path, weights_only=False, map_location=device)
        else:
            # Try to load with the safelist approach
            agent = torch.load(args.model_path, map_location=device)
        
        print(f"Successfully loaded model")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("To fix this, try running with --use_unsafe_loading flag")
        sys.exit(1)
    
    # Set up environment
    env = rlcard.make('fortyfives')
    print(f"Created environment with {env.num_players} players")
    
    # Set up agents (trained agent as player 0, random agents for others)
    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(rlcard.agents.RandomAgent(num_actions=env.num_actions))
    
    env.set_agents(agents)
    
    # Start evaluation
    rewards = []
    wins = 0
    losses = 0
    total_score_ns = 0
    total_score_ew = 0
    
    print(f'Starting evaluation with {args.eval_num} games')
    for episode in range(args.eval_num):
        # Reset the environment
        state, player_id = env.reset()
        
        # Play until the game is done
        done = False
        step_count = 0
        
        while not done and step_count < 1000:
            step_count += 1
            action = env.agents[player_id].step(state)
            next_state, next_player_id = env.step(action)
            
            # Check if the game is done
            if step_count >= 1000 or (hasattr(env.game, 'is_over') and env.game.is_over()):
                done = True
                payoffs = env.get_payoffs()
                rewards.append(payoffs[0])  # Record reward for player 0 (the DQN agent)
                
                # Get scores
                try:
                    perfect_info = env.get_perfect_information()
                    ns_score = perfect_info['points'].get(0, 0)
                    ew_score = perfect_info['points'].get(1, 0)
                    
                    total_score_ns += ns_score
                    total_score_ew += ew_score
                    
                    # Determine win/loss
                    if payoffs[0] > 0:
                        wins += 1
                    elif payoffs[0] < 0:
                        losses += 1
                    
                    print(f"Game {episode+1}: N/S: {ns_score}, E/W: {ew_score}, Reward: {payoffs[0]}")
                except Exception as e:
                    print(f"Error getting scores: {e}")
            
            # Move to the next state
            state = next_state
            player_id = next_player_id
        
    # Calculate statistics
    avg_reward = np.mean(rewards)
    win_rate = wins / args.eval_num * 100
    avg_score_ns = total_score_ns / args.eval_num
    avg_score_ew = total_score_ew / args.eval_num
            
    print("\n===== EVALUATION RESULTS =====")
    print(f"Games played: {args.eval_num}")
    print(f"Win rate: {win_rate:.1f}% ({wins} wins, {losses} losses, {args.eval_num - wins - losses} draws)")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Average N/S score: {avg_score_ns:.1f}")
    print(f"Average E/W score: {avg_score_ew:.1f}")
    print("=============================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluate DQN Agent for Fortyfives")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--eval_num', type=int, default=20, help='Number of games for evaluation')
    parser.add_argument('--model_path', type=str, default='experiments/fortyfives_dqn/model.pth', help='Path to saved model for evaluation')
    parser.add_argument('--use_unsafe_loading', action='store_true', help='Use unsafe loading for PyTorch 2.6+')
    
    args = parser.parse_args()
    evaluate(args) 