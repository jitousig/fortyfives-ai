#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Training a DQN agent to play Fortyfives
'''

import os
import argparse
import numpy as np
import torch

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

def train(args):
    # Check if cuda is available
    device = get_device()
    
    # Set the seed
    set_seed(args.seed)
    
    # Make the environment with seed
    env = rlcard.make('fortyfives', config={'seed': args.seed})
    
    # Initialize the agent
    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape,
        mlp_layers=[64, 64],
        device=device
    )
    
    # Configure random agents for evaluation
    eval_env = rlcard.make('fortyfives', config={'seed': args.seed})
    random_agents = [
        rlcard.agents.RandomAgent(num_actions=env.num_actions) 
        for _ in range(env.num_players)
    ]
    
    # Set up evaluator
    def _eval_agent(agent, eval_env, random_agents):
        # Use the first position for the learning agent
        eval_env.set_agents([agent] + random_agents[1:])
        
        # Generate trajectories
        payoffs = tournament(
            env=eval_env,
            num=args.eval_num,
            is_training=False
        )[0]
        
        return float(payoffs)

    # Set up logger
    logger = Logger(args.log_dir)
    
    for episode in range(args.num_episodes):
        # Generate data from the environment
        trajectories, _ = env.run(is_training=True)
        
        # Reorganize the trajectories to player-specific view
        trajectories = reorganize(trajectories, env.num_players)
        
        # Feed transitions into agent memory and train
        for ts in trajectories[0]:
            agent.feed(ts)
            
        # Train the agent
        agent.train()
        
        # Evaluate the agent
        if episode % args.evaluate_every == 0:
            logger.log_performance(
                episode,
                _eval_agent(agent, eval_env, random_agents)
            )
            
    # Save final model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print(f'Model saved to {save_path}')
    
    # Plot learning curve
    logger.plot_curve(save_path=os.path.join(args.log_dir, 'learning_curve.png'))
    
    # Save model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print(f'Model saved to {save_path}')

def evaluate(args):
    # Load pretrained model
    agent = torch.load(args.model_path, map_location=get_device())
    
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
        
        if episode % 100 == 0:
            print(f'Episode {episode}, average reward: {np.mean(rewards):.4f}')
            
    print(f'Final average reward after {args.eval_num} episodes: {np.mean(rewards):.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN Agent for Fortyfives")
    parser.add_argument('--mode', choices=['train', 'evaluate'], default='train', help='Train or evaluate the agent')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_episodes', type=int, default=5000, help='Number of episodes for training')
    parser.add_argument('--evaluate_every', type=int, default=100, help='Evaluate agent every N episodes')
    parser.add_argument('--eval_num', type=int, default=100, help='Number of games for evaluation')
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