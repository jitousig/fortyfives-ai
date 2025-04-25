#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Training a DQN agent to play Fortyfives
'''

import os
import argparse
import numpy as np
import torch
import sys
import csv

import rlcard
from rlcard.agents import DQNAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve
)

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

# Custom Logger class since RLCard's Logger might have issues
class CustomLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.data = {'episode': [], 'performance': []}
        self.csv_path = os.path.join(log_dir, 'performance.csv')
        
        # Create CSV file with headers
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['episode', 'performance'])
            writer.writeheader()
    
    def log_performance(self, episode, performance):
        self.data['episode'].append(episode)
        self.data['performance'].append(performance)
        
        # Also save to CSV
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['episode', 'performance'])
            writer.writerow({'episode': episode, 'performance': performance})

def train(args):
    # Check if cuda is available
    device = get_device()
    print(f"Using device: {device}")
    
    # Set the seed
    set_seed(args.seed)
    
    # Try to make the environment with seed
    try:
        env = rlcard.make('fortyfives', config={'seed': args.seed})
        print(f"Successfully created fortyfives environment")
        print(f"Number of actions: {env.num_actions}")
        print(f"Number of players: {env.num_players}")
        print(f"State shape: {env.state_shape}")
    except Exception as e:
        print(f"Error creating environment: {e}")
        sys.exit(1)
    
    # Initialize the agent
    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape,
        mlp_layers=[64, 64],
        device=device
    )
    
    # Create random agents for other positions
    random_agents = [
        rlcard.agents.RandomAgent(num_actions=env.num_actions) 
        for _ in range(env.num_players)
    ]
    
    # Set DQN agent at position 0 and random agents at other positions
    env.set_agents([agent] + random_agents[1:])
    
    # Configure agents for evaluation
    eval_env = rlcard.make('fortyfives', config={'seed': args.seed})
    eval_random_agents = [
        rlcard.agents.RandomAgent(num_actions=eval_env.num_actions) 
        for _ in range(eval_env.num_players)
    ]
    eval_env.set_agents([agent] + eval_random_agents[1:])
    
    # Set up evaluator function
    def _eval_agent(agent, eval_env, num_games=1):
        # Custom evaluation function
        rewards = []
        for _ in range(num_games):
            # Reset the environment
            state, player_id = eval_env.reset()
            done = False
            step_count = 0
            
            while not done and step_count < 1000:
                step_count += 1
                action = eval_env.agents[player_id].step(state)
                next_state, next_player_id = eval_env.step(action)
                
                # Check if the game is done
                if step_count >= 1000 or (hasattr(eval_env.game, 'is_over') and eval_env.game.is_over()):
                    done = True
                    payoffs = eval_env.get_payoffs()
                    rewards.append(payoffs[0])  # Record reward for player 0 (the learning agent)
                
                # Move to the next state
                state = next_state
                player_id = next_player_id
        
        # Return average reward
        return float(np.mean(rewards)) if rewards else 0.0

    # Set up logger
    logger = CustomLogger(args.log_dir)
    
    for episode in range(args.num_episodes):
        # Generate data from the environment
        try:
            # Print progress
            if episode % 10 == 0:
                print(f"Episode {episode}/{args.num_episodes}")
            
            # Reset the environment
            state, player_id = env.reset()
            
            # Collect trajectories manually
            trajectories = [[] for _ in range(env.num_players)]
            done = False
            step_count = 0
            
            # Manual trajectory collection to avoid the int not subscriptable issue
            while not done and step_count < 1000:  # Safety limit
                step_count += 1
                
                # Get the action from the agent
                action = env.agents[player_id].step(state)
                
                # Record the state and action for the current player
                trajectories[player_id].append({
                    'state': state,
                    'action': action,
                    'reward': 0,  # Will be updated later
                    'next_state': None  # Will be updated later
                })
                
                # Step the environment
                next_state, next_player_id = env.step(action)
                
                # Check if the game is done
                if step_count >= 1000 or (hasattr(env.game, 'is_over') and env.game.is_over()):
                    done = True
                    # Get payoffs for all players
                    payoffs = env.get_payoffs()
                    
                    # Update rewards in all trajectories
                    for player in range(env.num_players):
                        for transition in trajectories[player]:
                            transition['reward'] = payoffs[player]
                
                # Update the next state for the previous action
                if len(trajectories[player_id]) > 0:
                    trajectories[player_id][-1]['next_state'] = next_state
                
                # Move to the next state
                state = next_state
                player_id = next_player_id
            
            # Print trajectory information
            if episode % 50 == 0:
                print(f"Generated {step_count} steps across {len(trajectories)} players")
                for p, traj in enumerate(trajectories):
                    print(f"Player {p}: {len(traj)} transitions")
            
            # Feed transitions into agent memory directly using feed_memory instead of feed
            for transition in trajectories[0]:
                if transition['next_state'] is not None:  # Skip the last transition if next_state is None
                    # Get the necessary components
                    state_obs = transition['state']['obs']
                    action = transition['action']
                    reward = transition['reward']
                    next_state_obs = transition['next_state']['obs']
                    legal_actions = list(transition['next_state']['legal_actions'].keys())
                    done_flag = done
                    
                    # Feed memory directly
                    agent.feed_memory(state_obs, action, reward, next_state_obs, legal_actions, done_flag)
            
            # Train the agent
            loss = agent.train()
            if episode % 10 == 0 and loss is not None:
                print(f"Episode {episode}, Loss: {loss:.6f}")
            
            # Evaluate the agent
            if episode % args.evaluate_every == 0:
                performance = _eval_agent(agent, eval_env, num_games=args.eval_num)
                logger.log_performance(episode, performance)
                print(f"Episode {episode}, Performance: {performance:.4f}")
                
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    # Save final model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print(f'Model saved to {save_path}')
    
    # Plot learning curve using matplotlib directly
    try:
        import matplotlib.pyplot as plt
        if len(logger.data['episode']) > 0:
            xs = logger.data['episode']
            ys = logger.data['performance']
            plt.figure()
            plt.plot(xs, ys)
            plt.title('Performance')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig(os.path.join(args.log_dir, 'learning_curve.png'))
            plt.close()
            print(f'Learning curve saved to {os.path.join(args.log_dir, "learning_curve.png")}')
        else:
            print("No performance data to plot")
            
    except Exception as e:
        print(f"Error plotting learning curve: {e}")

def evaluate(args):
    # Load pretrained model
    try:
        agent = torch.load(args.model_path, map_location=get_device())
        print(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Set up environment
    env = rlcard.make('fortyfives')
    
    # Set up agents (trained agent as player 0, random agents for others)
    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(rlcard.agents.RandomAgent(num_actions=env.num_actions))
    
    env.set_agents(agents)
    
    # Start evaluation
    rewards = []
    print('Start evaluation')
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
            
            # Move to the next state
            state = next_state
            player_id = next_player_id
            
        if episode % 10 == 0:
            print(f'Episode {episode}, average reward: {np.mean(rewards):.4f}')
            
    print(f'Final average reward after {args.eval_num} episodes: {np.mean(rewards):.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN Agent for Fortyfives")
    parser.add_argument('--mode', choices=['train', 'evaluate'], default='train', help='Train or evaluate the agent')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_episodes', type=int, default=5000, help='Number of episodes for training')
    parser.add_argument('--evaluate_every', type=int, default=100, help='Evaluate agent every N episodes')
    parser.add_argument('--eval_num', type=int, default=10, help='Number of games for evaluation')
    parser.add_argument('--log_dir', type=str, default='experiments/fortyfives_dqn', help='Directory for logs')
    parser.add_argument('--model_path', type=str, default='experiments/fortyfives_dqn/model.pth', help='Path to saved model for evaluation')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Create log directory if it doesn't exist
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        train(args)
    else:
        evaluate(args) 