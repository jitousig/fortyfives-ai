#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Train a DQN agent on Fortyfives with self-play, enhanced rewards, and early stopping
'''

import os
import argparse
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import sys
import csv

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

# Custom logger that tracks performance and loss
class CustomLogger:
    def __init__(self, log_dir=''):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.performance_data = []
        self.loss_data = []
        
        # Create CSV files for logging
        self.perf_csv_path = os.path.join(log_dir, 'performance.csv')
        with open(self.perf_csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward'])
            
        self.loss_csv_path = os.path.join(log_dir, 'loss.csv')
        with open(self.loss_csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'loss'])
    
    def log_performance(self, episode, reward):
        self.performance_data.append((episode, reward))
        with open(self.perf_csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward])
    
    def log_loss(self, episode, loss):
        if loss is not None:
            self.loss_data.append((episode, loss))
            with open(self.loss_csv_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([episode, loss])
    
    def plot_learning_curve(self, save_path=None):
        if not self.performance_data:
            print("No performance data to plot")
            return
            
        episodes, rewards = zip(*self.performance_data)
        plt.figure(figsize=(15, 5))
        
        # Plot reward curve
        plt.subplot(1, 2, 1)
        plt.plot(episodes, rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Evaluation Performance')
        plt.grid(True)
        
        # Plot loss curve if available
        if self.loss_data:
            loss_episodes, losses = zip(*self.loss_data)
            plt.subplot(1, 2, 2)
            plt.plot(loss_episodes, losses)
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(os.path.join(self.log_dir, 'learning_curves.png'))
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
        learning_rate=0.0002,  # Higher learning rate for faster initial progress
        replay_memory_init_size=2000,  # Collect more experiences before learning
        replay_memory_size=200000,  # Larger replay buffer for better stability
        batch_size=128,  # Larger batch size for more stable updates
        update_target_estimator_every=2000,  # Update target network less frequently
        discount_factor=0.99,  # Standard discount factor
        epsilon_start=1.0,  # Start with full exploration
        epsilon_end=0.01,  # End with just 1% exploration (more exploitation)
        epsilon_decay_steps=100000,  # Very slow epsilon decay for better exploration
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
    
    # Early stopping variables
    best_loss = float('inf')
    best_eval_reward = float('-inf')
    best_model_path = os.path.join(args.log_dir, "best_model.pth")
    no_improvement_counter = 0
    reward_not_improved_counter = 0
    
    # Start training
    print(f"Starting training for up to {args.num_episodes} episodes with early stopping...")
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
                
                # Add intermediate rewards based on game state
                # Only for the agent's transitions
                if player_id == agent_pos and trajectories:
                    curr_transition = trajectories[-1]
                    
                    # Try to get perfect information to assess the current game state
                    try:
                        perfect_info = env.get_perfect_information()
                        
                        # Reward for winning tricks
                        if hasattr(env.game, 'trick_winner') and env.game.trick_winner is not None:
                            # If agent (or their partner) won the trick, give small positive reward
                            agent_team = agent_pos % 2  # 0 for N/S, 1 for E/W
                            trick_winner_team = env.game.trick_winner % 2
                            
                            if trick_winner_team == agent_team:
                                curr_transition['reward'] += 0.1  # Small positive reward for winning a trick
                        
                        # If this transition led to playing a card, give a tiny reward/penalty
                        # based on the card's value in the game
                        if hasattr(perfect_info, 'current_trick') and perfect_info.get('current_trick') is not None:
                            current_trick = perfect_info.get('current_trick', [])
                            if len(current_trick) > 0 and current_trick[-1] is not None:
                                # The agent just played a card
                                # Small reward for playing high cards at the right time
                                # This requires game-specific knowledge
                                pass  # Simplified for now
                    except:
                        # If we can't get perfect info, just continue without intermediate rewards
                        pass
                
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
            
            # Train the agent only if there are enough samples
            loss = None
            try:
                loss = agent.train()
            except ValueError as e:
                # This is likely due to not having enough samples in the replay buffer
                if "Sample larger than population" in str(e):
                    if episode % 50 == 0:
                        print(f"Episode {episode}: Not enough samples in replay buffer yet. " + 
                              f"Have {len(agent.memory.memory)}, need {agent.batch_size}")
                else:
                    # Re-raise if it's a different error
                    raise
            
            # Log loss every episode (if we have a loss)
            if loss is not None:
                logger.log_loss(episode, loss)
                
                # Print loss for every episode
                if episode % 10 == 0:  # Only print every 10 episodes to reduce output
                    print(f"Episode {episode}, Loss: {loss:.6f}")
                
                # Check if loss has improved
                if loss < best_loss:
                    best_loss = loss
                    no_improvement_counter = 0
                    if episode % 50 == 0:  # Only print improvement messages every 50 episodes
                        print(f"New best loss: {best_loss:.6f}")
                else:
                    no_improvement_counter += 1
                    if episode % 50 == 0:  # Only print no improvement message every 50 episodes
                        print(f"No loss improvement for {no_improvement_counter} episodes")
                
                # Check early stopping condition based on loss
                if no_improvement_counter >= args.patience:
                    print(f"Early stopping triggered after {episode} episodes due to no loss improvement for {args.patience} episodes")
                    # Save the current model as the final model
                    torch.save(agent, os.path.join(args.log_dir, "model.pth"))
                    break
            
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
                
                # Save the best model based on evaluation performance
                if avg_reward > best_eval_reward:
                    best_eval_reward = avg_reward
                    reward_not_improved_counter = 0
                    print(f"New best evaluation reward: {best_eval_reward:.4f}, saving best model")
                    torch.save(agent, best_model_path)
                else:
                    reward_not_improved_counter += 1
                    if episode % args.evaluate_every == 0:
                        print(f"No reward improvement for {reward_not_improved_counter * args.evaluate_every} episodes")
                    
                    # Check early stopping condition based on reward
                    if reward_not_improved_counter >= args.patience // args.evaluate_every:
                        print(f"Early stopping triggered after {episode} episodes due to no reward improvement for {args.patience} episodes")
                        # Save the current model as the final model
                        torch.save(agent, os.path.join(args.log_dir, "model.pth"))
                        break
                
                # Save checkpoint
                if episode % args.checkpoint_every == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f'model_ep{episode}.pth')
                    torch.save(agent, checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path}")
                
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save the final trained model if not already saved by early stopping
    try:
        final_model_path = os.path.join(args.log_dir, "model.pth")
        if not os.path.exists(final_model_path):
            print("Saving final model...")
            torch.save(agent, final_model_path)
        print(f"Final model saved at {final_model_path}")
        print(f"Best model saved at {best_model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    # Plot learning curve
    try:
        logger.plot_learning_curve()
        print(f"Learning curves saved at {os.path.join(args.log_dir, 'learning_curves.png')}")
    except Exception as e:
        print(f"Error plotting learning curve: {e}")
    
    # Print final training summary
    elapsed_time = time.time() - start_time
    print(f"Training complete. Total time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN with self-play, enhanced rewards, and early stopping for Fortyfives")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_episodes', type=int, default=30000, help='Maximum number of episodes to train')
    parser.add_argument('--evaluate_every', type=int, default=100, help='Evaluate the agent every N episodes')
    parser.add_argument('--checkpoint_every', type=int, default=500, help='Save a checkpoint every N episodes')
    parser.add_argument('--rotate_every', type=int, default=500, help='Rotate agent positions every N episodes')
    parser.add_argument('--eval_num', type=int, default=100, help='Number of games for evaluation')
    parser.add_argument('--patience', type=int, default=800, help='Early stopping patience (episodes with no reward improvement)')
    parser.add_argument('--log_dir', type=str, default='experiments/fortyfives_dqn_long_training', help='Directory to save models and logs')
    
    args = parser.parse_args()
    train(args) 