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
        rlcard.agents.RandomAgent(num_actions=env.num_actions) 
        for _ in range(env.num_players)
    ]
    
    # Set up evaluator
    def _eval_agent(agent, eval_env, random_agents):
        # Use the first position for the learning agent
        eval_env.set_agents([agent] + random_agents[1:])
        
        # Generate trajectories
        try:
            payoffs = tournament(
                env=eval_env,
                num=args.eval_num,
                is_training=False
            )[0]
            return float(payoffs)
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return 0.0

    # Set up logger
    logger = Logger(args.log_dir)
    
    for episode in range(args.num_episodes):
        # Generate data from the environment
        try:
            trajectories, _ = env.run(is_training=True)
            
            # Print progress
            if episode % 10 == 0:
                print(f"Episode {episode}/{args.num_episodes}")
            
            # Reorganize the trajectories to player-specific view
            trajectories = reorganize(trajectories, env.num_players)
            
            # Feed transitions into agent memory and train
            for ts in trajectories[0]:
                agent.feed(ts)
                
            # Train the agent
            agent.train()
            
            # Evaluate the agent
            if episode % args.evaluate_every == 0:
                performance = _eval_agent(agent, eval_env, eval_random_agents)
                logger.log_performance(episode, performance)
                print(f"Episode {episode}, Performance: {performance:.4f}")
                
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            continue
            
    # Save final model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print(f'Model saved to {save_path}')
    
    # Plot learning curve using matplotlib directly
    try:
        import matplotlib.pyplot as plt
        if hasattr(logger, 'data'):
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
        _, payoffs = env.run(is_training=False)
        rewards.append(payoffs[0])
        
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