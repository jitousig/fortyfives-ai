#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Training a DQN agent with self-play for Fortyfives
'''

import os
import argparse
import numpy as np
import torch
import sys
import csv
import time
from copy import deepcopy

import rlcard
from rlcard.agents import DQNAgent
from rlcard.utils import (
    get_device,
    set_seed
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

# Custom Logger class
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
    start_time = time.time()
    
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
    
    # Initialize main DQN agent that we'll train
    main_agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape,
        mlp_layers=[128, 128],  # Using larger network for better learning
        device=device
    )
    
    # Create random agents for other positions (simple self-play approach)
    random_agents = [
        rlcard.agents.RandomAgent(num_actions=env.num_actions) 
        for _ in range(env.num_players - 1)
    ]
    
    # Set the agents in the environment (main agent in position 0, random agents in others)
    env.set_agents([main_agent] + random_agents)
    
    # Configure agents for evaluation
    eval_env = rlcard.make('fortyfives', config={'seed': args.seed})
    eval_random_agents = [
        rlcard.agents.RandomAgent(num_actions=eval_env.num_actions) 
        for _ in range(eval_env.num_players)
    ]
    
    # Set up evaluator function
    def _eval_agent(agent, eval_env, num_games=1):
        # Set up evaluation environment with random agents
        eval_env.set_agents([agent] + eval_random_agents[1:])
        
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
    
    # Checkpoint directory
    checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for episode in range(args.num_episodes):
        # Generate data from the environment
        try:
            # Print progress
            if episode % 50 == 0:
                elapsed_time = time.time() - start_time
                print(f"Episode {episode}/{args.num_episodes} (Time elapsed: {elapsed_time:.1f}s)")
            
            # Reset the environment
            state, player_id = env.reset()
            
            # Play one episode
            trajectories = []
            done = False
            step_count = 0

            while not done and step_count < 1000:  # Safety limit
                step_count += 1

                # Take action
                action = env.agents[player_id].step(state)

                # Record state, action for main agent
                if player_id == 0:  # Main agent's turn
                    trajectories.append({
                        'state': state,
                        'action': action,
                        'reward': 0,      # Terminal transition updated below
                        'next_state': None,
                        'done': False,    # Terminal transition updated below
                    })

                # Step the environment
                next_state, next_player_id = env.step(action)

                # Update the next state for the previous transition
                if player_id == 0 and trajectories:
                    trajectories[-1]['next_state'] = next_state

                # Check if the game is done
                is_over = hasattr(env.game, 'is_over') and env.game.is_over()
                if step_count >= 1000 or is_over:
                    done = True
                    payoffs = env.get_payoffs()
                    # Only the terminal transition carries the outcome reward and done=True.
                    # All earlier transitions have reward=0 and done=False so that
                    # Q-value bootstrapping is not broken.
                    if trajectories:
                        trajectories[-1]['reward'] = payoffs[0]
                        trajectories[-1]['done'] = True

                # Move to the next state
                state = next_state
                player_id = next_player_id
            
            # Print trajectory information
            if episode % 500 == 0:
                print(f"Episode {episode}: {step_count} steps, {len(trajectories)} transitions for main agent")
            
            # Feed transitions into agent memory
            for transition in trajectories:
                if transition['next_state'] is not None:
                    main_agent.feed_memory(
                        transition['state']['obs'],
                        transition['action'],
                        transition['reward'],
                        transition['next_state']['obs'],
                        list(transition['next_state']['legal_actions'].keys()),
                        transition['done'],
                    )
            
            # Train the agent
            loss = main_agent.train()
            
            # Log loss
            if episode % 50 == 0 and loss is not None:
                print(f"Episode {episode}, Loss: {loss:.6f}")
            
            # Evaluate the agent
            if episode % args.evaluate_every == 0:
                performance = _eval_agent(main_agent, eval_env, num_games=args.eval_num)
                logger.log_performance(episode, performance)
                print(f"Episode {episode}, Performance: {performance:.4f}")
                
                # Save checkpoint
                if episode % args.checkpoint_every == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f'model_ep{episode}.pth')
                    torch.save(main_agent, checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path}")
            
            # Implement progressive self-play (position rotation)
            if episode % args.rotate_every == 0 and episode > 0:
                # Every rotate_every episodes, change the position of the main agent
                position = (episode // args.rotate_every) % env.num_players
                
                print(f"Rotating main agent to position {position} at episode {episode}")
                
                # Create a new set of agents with main agent at the new position
                new_agents = []
                for i in range(env.num_players):
                    if i == position:
                        new_agents.append(main_agent)
                    else:
                        new_agents.append(rlcard.agents.RandomAgent(num_actions=env.num_actions))
                
                # Set the new arrangement of agents
                env.set_agents(new_agents)
                
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    # Save final model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(main_agent, save_path)
    print(f'Final model saved to {save_path}')
    
    # Plot learning curve using matplotlib
    try:
        import matplotlib.pyplot as plt
        if len(logger.data['episode']) > 0:
            xs = logger.data['episode']
            ys = logger.data['performance']
            plt.figure(figsize=(10, 6))
            plt.plot(xs, ys)
            plt.title('Performance Over Time')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.savefig(os.path.join(args.log_dir, 'learning_curve.png'))
            plt.close()
            print(f'Learning curve saved to {os.path.join(args.log_dir, "learning_curve.png")}')
        else:
            print("No performance data to plot")
            
    except Exception as e:
        print(f"Error plotting learning curve: {e}")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN Agent with Self-Play for Fortyfives")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_episodes', type=int, default=10000, help='Number of episodes for training')
    parser.add_argument('--evaluate_every', type=int, default=200, help='Evaluate agent every N episodes')
    parser.add_argument('--checkpoint_every', type=int, default=1000, help='Save checkpoint every N episodes')
    parser.add_argument('--rotate_every', type=int, default=500, help='Rotate agent position every N episodes')
    parser.add_argument('--eval_num', type=int, default=20, help='Number of games for evaluation')
    parser.add_argument('--log_dir', type=str, default='experiments/fortyfives_dqn_selfplay', help='Directory for logs')
    
    args = parser.parse_args()
    
    # Create log directory if it doesn't exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        
    train(args) 