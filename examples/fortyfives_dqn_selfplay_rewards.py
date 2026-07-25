#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Train a DQN agent on Fortyfives with self-play and enhanced reward structure
'''

import os
import argparse
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import sys

import rlcard
from rlcard.utils import (
    get_device,
    set_seed,
    print_card,
)
from rlcard.agents import (
    RandomAgent,
)

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

# Custom logger that tracks performance in self-play
class CustomLogger:
    def __init__(self, log_dir=''):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.performance_data = []
        self.csv_path = os.path.join(log_dir, 'performance.csv')
        with open(self.csv_path, 'w') as f:
            f.write('episode,reward\n')
    
    def log_performance(self, episode, reward):
        self.performance_data.append((episode, reward))
        with open(self.csv_path, 'a') as f:
            f.write(f'{episode},{reward}\n')
    
    def plot_learning_curve(self, save_path=None):
        episodes, rewards = zip(*self.performance_data)
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Learning Curve')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(os.path.join(self.log_dir, 'learning_curve.png'))
        plt.close()

def train(args):
    set_seed(args.seed)
    
    print("Creating environment...")
    env = rlcard.make('fortyfives', config={'seed': args.seed})
    print(f"Number of actions: {env.num_actions}")
    print(f"Number of players: {env.num_players}")
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize DQN agent
    from rlcard.agents.dqn_agent import DQNAgent
    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape,
        device=device,
        save_path=args.log_dir,
        mlp_layers=[512, 512, 512],  # Deeper network for more complex game
        learning_rate=0.00005,  # Lower learning rate for stability
        replay_memory_init_size=1000,  # Collect more experiences before learning
        replay_memory_size=100000,  # Larger replay buffer
        batch_size=64,  # Larger batch size for more stable updates
        update_target_estimator_every=1000,  # Update target network less frequently
        discount_factor=0.99,  # Standard discount factor
    )
    
    # Initialize random agents
    random_agents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)]
    
    # Set up the evaluation environment
    eval_env = rlcard.make('fortyfives', config={'seed': args.seed + 1})
    eval_random_agents = [RandomAgent(num_actions=eval_env.num_actions) for _ in range(eval_env.num_players)]
    
    # Initialize logger
    logger = CustomLogger(log_dir=args.log_dir)
    
    # Checkpoint directory
    checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Start training
    print(f"Starting training for {args.num_episodes} episodes...")
    start_time = time.time()
    
    for episode in range(1, args.num_episodes + 1):
        try:
            # Print progress
            if episode % 50 == 0:
                elapsed_time = time.time() - start_time
                print(f"Episode {episode}/{args.num_episodes} (Time elapsed: {elapsed_time:.1f}s)")
            
            # Rotate the position of the DQN agent if needed
            if episode % args.rotate_every == 0 or episode == 1:
                position = ((episode - 1) // args.rotate_every) % env.num_players
                agents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)]
                agents[position] = agent
                env.set_agents(agents)
                if episode > 1:
                    print(f"Episode {episode}: Rotating agent to position {position}")
            
            # Reset the environment
            state, player_id = env.reset()
            
            # Play one episode
            trajectories = []
            done = False
            step_count = 0
            agent_pos = position  # Current position of the agent
            
            while not done and step_count < 1000:  # Safety limit
                step_count += 1
                
                # Take action
                action = env.agents[player_id].step(state)
                
                # Record state, action for agent
                if player_id == agent_pos:  # Agent's turn
                    trajectories.append({
                        'state': state,
                        'action': action,
                        'reward': 0,  # Updated when game ends
                        'next_state': None  # Updated on next step
                    })
                
                # Step the environment
                next_state, next_player_id = env.step(action)
                
                # Update the next state for the previous transition
                if player_id == agent_pos and trajectories:
                    trajectories[-1]['next_state'] = next_state
                
                # Check if the game is done
                if step_count >= 1000 or (hasattr(env.game, 'is_over') and env.game.is_over()):
                    done = True
                    payoffs = env.get_payoffs()
                    
                    # Update rewards for all transitions with actual payoff
                    for transition in trajectories:
                        transition['reward'] = payoffs[agent_pos]  # Agent's reward
                
                # Move to the next state
                state = next_state
                player_id = next_player_id
            
            # Print trajectory information
            if episode % 500 == 0:
                print(f"Episode {episode}: {step_count} steps, {len(trajectories)} transitions for agent")
            
            # Feed transitions into agent memory
            for transition in trajectories:
                if transition['next_state'] is not None:  # Skip if next_state is None
                    # Get the necessary components
                    state_obs = transition['state']['obs']
                    action = transition['action']
                    reward = transition['reward']
                    next_state_obs = transition['next_state']['obs']
                    legal_actions = list(transition['next_state']['legal_actions'].keys())
                    done_flag = done
                    
                    # Feed memory directly to the agent
                    agent.feed_memory(state_obs, action, reward, next_state_obs, legal_actions, done_flag)
            
            # Train the agent
            loss = agent.train()
            
            # Log loss occasionally
            if episode % 50 == 0 and loss is not None:
                print(f"Episode {episode}, Loss: {loss:.6f}")
            
            # Evaluate the agent
            if episode % args.evaluate_every == 0:
                # Set up evaluation with agent at position 0
                eval_agents = [agent] + eval_random_agents[1:]
                eval_env.set_agents(eval_agents)
                
                # Run multiple evaluation games
                rewards = []
                for _ in range(args.eval_num):
                    # Reset the environment
                    eval_state, eval_player_id = eval_env.reset()
                    eval_done = False
                    eval_step_count = 0
                    
                    while not eval_done and eval_step_count < 1000:
                        eval_step_count += 1
                        eval_action = eval_env.agents[eval_player_id].step(eval_state)
                        eval_next_state, eval_next_player_id = eval_env.step(eval_action)
                        
                        # Check if the game is done
                        if eval_step_count >= 1000 or (hasattr(eval_env.game, 'is_over') and eval_env.game.is_over()):
                            eval_done = True
                            eval_payoffs = eval_env.get_payoffs()
                            rewards.append(eval_payoffs[0])  # Record reward for the agent (at position 0)
                        
                        # Move to the next state
                        eval_state = eval_next_state
                        eval_player_id = eval_next_player_id
                
                # Calculate average reward
                avg_reward = float(np.mean(rewards)) if rewards else 0.0
                
                # Log performance
                logger.log_performance(episode, avg_reward)
                print(f"Episode {episode}, Average evaluation reward: {avg_reward:.4f}")
                
                # Save checkpoint
                if episode % args.checkpoint_every == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f'model_ep{episode}.pth')
                    torch.save(agent, checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path}")
                
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save the final trained model
    try:
        save_path = os.path.join(args.log_dir, "model.pth")
        torch.save(agent, save_path)
        print(f"Final model saved at {save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    # Plot learning curve
    try:
        logger.plot_learning_curve()
        print(f"Learning curve saved at {os.path.join(args.log_dir, 'learning_curve.png')}")
    except Exception as e:
        print(f"Error plotting learning curve: {e}")
    
    # Print final training summary
    elapsed_time = time.time() - start_time
    print(f"Training complete. Total time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN with self-play and enhanced rewards for Fortyfives")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_episodes', type=int, default=10000, help='Number of episodes to train')
    parser.add_argument('--evaluate_every', type=int, default=250, help='Evaluate the agent every N episodes')
    parser.add_argument('--checkpoint_every', type=int, default=1000, help='Save a checkpoint every N episodes')
    parser.add_argument('--rotate_every', type=int, default=500, help='Rotate agent positions every N episodes')
    parser.add_argument('--eval_num', type=int, default=20, help='Number of games for evaluation')
    parser.add_argument('--log_dir', type=str, default='experiments/fortyfives_dqn_selfplay_rewards', help='Directory to save models and logs')
    
    args = parser.parse_args()
    train(args) 