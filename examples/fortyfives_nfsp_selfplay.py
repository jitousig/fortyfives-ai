#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Train an NFSP agent on Fortyfives with self-play and early stopping
'''

import os
import argparse
import numpy as np
import torch
import time
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
import matplotlib.pyplot as plt
import sys
import csv
import json
from collections import defaultdict
import random

import rlcard
from rlcard.utils import (
    get_device,
    set_seed,
    print_card,
    Logger,
    plot_curve,
)
from rlcard.agents import (
    RandomAgent,
    NFSPAgent,
    DQNAgent,
)
from rlcard.envs.registration import register, registry

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local Fortyfives environment
from fortyfives.envs.fortyfives_env import FortyfivesEnv

# Register Fortyfives environment
if 'fortyfives' not in registry.env_specs:
    register(
        env_id='fortyfives',
        entry_point='fortyfives.envs.fortyfives_env:FortyfivesEnv',
    )

print("Successfully registered fortyfives environment")

class CustomLogger(Logger):
    def __init__(self, log_dir=''):
        super().__init__(log_dir)
        self.performance_data = []
        self.rl_loss_data = []
        self.sl_loss_data = []
        
        # Create CSV files for logging
        self.perf_csv_path = os.path.join(log_dir, 'performance.csv')
        with open(self.perf_csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward'])
            
        self.rl_loss_csv_path = os.path.join(log_dir, 'rl_loss.csv')
        with open(self.rl_loss_csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'loss'])
            
        self.sl_loss_csv_path = os.path.join(log_dir, 'sl_loss.csv')
        with open(self.sl_loss_csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'loss'])
    
    def log_performance(self, episode, reward):
        self.performance_data.append((episode, reward))
        with open(self.perf_csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward])
    
    def log_rl_loss(self, episode, loss):
        if loss is not None:
            self.rl_loss_data.append((episode, loss))
            with open(self.rl_loss_csv_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([episode, loss])
                
    def log_sl_loss(self, episode, loss):
        if loss is not None:
            self.sl_loss_data.append((episode, loss))
            with open(self.sl_loss_csv_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([episode, loss])
    
    def plot_learning_curve(self, save_path=None):
        if not self.performance_data:
            print("No performance data to plot")
            return
            
        episodes, rewards = zip(*self.performance_data)
        plt.figure(figsize=(15, 5))
        
        # Plot reward curve
        plt.subplot(1, 3, 1)
        plt.plot(episodes, rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Evaluation Performance')
        plt.grid(True)
        
        # Plot RL loss curve if available
        if self.rl_loss_data:
            rl_episodes, rl_losses = zip(*self.rl_loss_data)
            plt.subplot(1, 3, 2)
            plt.plot(rl_episodes, rl_losses)
            plt.xlabel('Episode')
            plt.ylabel('RL Loss')
            plt.title('RL Training Loss')
            plt.grid(True)
            
        # Plot SL loss curve if available
        if self.sl_loss_data:
            sl_episodes, sl_losses = zip(*self.sl_loss_data)
            plt.subplot(1, 3, 3)
            plt.plot(sl_episodes, sl_losses)
            plt.xlabel('Episode')
            plt.ylabel('SL Loss')
            plt.title('SL Training Loss')
            plt.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(os.path.join(self.log_dir, 'learning_curves.png'))
        plt.close()

class RuleBasedAgent(object):
    """A rule-based agent that follows basic card game strategies."""
    
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.use_raw = False
        
    def step(self, state):
        legal_actions = state['legal_actions']
        
        # Basic strategy: prefer playing higher cards when possible
        if legal_actions:
            # Get the ranks of legal cards
            card_ranks = [self._get_card_rank(card) for card in legal_actions]
            # Play the highest card
            return legal_actions[np.argmax(card_ranks)]
        return 0
        
    def _get_card_rank(self, card):
        """Get the rank of a card (higher is better)."""
        # In Fortyfives, the rank order is: 5 > King > Queen > Jack > 10 > 9 > 8 > 7 > 6
        rank_map = {
            5: 8,   # 5 is highest
            13: 7,  # King
            12: 6,  # Queen
            11: 5,  # Jack
            10: 4,  # 10
            9: 3,   # 9
            8: 2,   # 8
            7: 1,   # 7
            6: 0    # 6 is lowest
        }
        return rank_map.get(card % 13, 0)

class BiddingAgent:
    """Agent for the bidding phase of Fortyfives."""
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.use_raw = False

    def step(self, state):
        """Choose a random legal bid."""
        legal_actions = list(state['legal_actions'])
        return np.random.choice(legal_actions)

    def eval_step(self, state):
        """Same as step for evaluation."""
        return self.step(state)

class DeclarationAgent:
    """Agent for the declaration phase of Fortyfives."""
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.use_raw = False

    def step(self, state):
        """Choose a random legal declaration."""
        legal_actions = list(state['legal_actions'])
        return np.random.choice(legal_actions)

    def eval_step(self, state):
        """Same as step for evaluation."""
        return self.step(state)

class PlayAgent:
    """NFSP agent for the play phase of Fortyfives."""
    def __init__(self,
                 num_actions,
                 state_shape,
                 hidden_layers_sizes,
                 q_replay_memory_size,
                 q_replay_memory_init_size,
                 q_update_target_estimator_every,
                 q_discount_factor,
                 q_epsilon_start,
                 q_epsilon_end,
                 q_epsilon_decay_steps,
                 q_batch_size,
                 q_learning_rate,
                 min_buffer_size_to_learn,
                 sl_learning_rate,
                 sl_batch_size,
                 sl_epsilon,
                 device,
                 reservoir_buffer_capacity):
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.sl_epsilon = sl_epsilon
        self.device = device
        self.use_raw = False

        # Initialize RL agent (DQN)
        self._rl_agent = DQNAgent(
            num_actions=num_actions,
            state_shape=state_shape,
            mlp_layers=hidden_layers_sizes,
            replay_memory_size=q_replay_memory_size,
            replay_memory_init_size=q_replay_memory_init_size,
            update_target_estimator_every=q_update_target_estimator_every,
            discount_factor=q_discount_factor,
            epsilon_start=q_epsilon_start,
            epsilon_end=q_epsilon_end,
            epsilon_decay_steps=q_epsilon_decay_steps,
            batch_size=q_batch_size,
            learning_rate=q_learning_rate,
            device=device
        )

        # Initialize SL agent (Average Policy Network)
        self._sl_agent = DQNAgent(
            num_actions=num_actions,
            state_shape=state_shape,
            mlp_layers=hidden_layers_sizes,
            replay_memory_size=reservoir_buffer_capacity,
            replay_memory_init_size=min_buffer_size_to_learn,
            update_target_estimator_every=q_update_target_estimator_every,
            discount_factor=1.0,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_steps=0,
            batch_size=sl_batch_size,
            learning_rate=sl_learning_rate,
            device=device
        )

    def step(self, state):
        """Choose an action using either the RL or SL agent."""
        if np.random.random() < self.sl_epsilon:
            return self._sl_agent.step(state)
        else:
            return self._rl_agent.step(state)

    def eval_step(self, state):
        """Choose an action using the SL agent during evaluation."""
        return self._sl_agent.step(state)

    def feed(self, ts):
        """Feed experience to both RL and SL agents.
        
        Args:
            ts (dict): Transition containing:
                state (dict): Current state
                action (int): Action taken
                reward (float): Reward received
                next_state (dict): Next state
                done (bool): Whether the episode is done
        """
        # Skip incomplete transitions
        if ts['action'] is None or ts['next_state'] is None:
            return

        # Debug print transition info
        print("\nFeeding transition:")
        print(f"Action: {ts['action']}")
        print(f"Reward: {ts['reward']}")
        print(f"Done: {ts['done']}")
        print(f"State phase: {ts['state']['raw_obs']['phase']}")
        print(f"Next state phase: {ts['next_state']['raw_obs']['phase']}")

        # The state already contains 'obs' and 'legal_actions'
        self._rl_agent.feed_memory(
            ts['state']['obs'],
            ts['action'],
            ts['reward'],
            ts['next_state']['obs'],
            ts['next_state']['raw_legal_actions'],
            ts['done']
        )
        
        # Feed to SL agent with probability sl_epsilon
        if np.random.random() < self.sl_epsilon:
            # Get best response action from RL agent
            best_response_action = self._rl_agent.step(ts['state'])
            
            # Create SL transition
            self._sl_agent.feed_memory(
                ts['state']['obs'],
                best_response_action,
                0,  # SL agent doesn't use reward
                ts['next_state']['obs'],
                ts['next_state']['raw_legal_actions'],
                ts['done']
            )

    def get_state_dict(self):
        """Get the state dict of both agents."""
        return {
            'rl_agent': {
                'q_estimator': self._rl_agent.q_estimator.qnet.state_dict(),
                'target_estimator': self._rl_agent.target_estimator.qnet.state_dict()
            },
            'sl_agent': {
                'q_estimator': self._sl_agent.q_estimator.qnet.state_dict(),
                'target_estimator': self._sl_agent.target_estimator.qnet.state_dict()
            }
        }

    def load_state_dict(self, state_dict):
        """Load the state dict of both agents."""
        self._rl_agent.q_estimator.qnet.load_state_dict(state_dict['rl_agent']['q_estimator'])
        self._rl_agent.target_estimator.qnet.load_state_dict(state_dict['rl_agent']['target_estimator'])
        self._sl_agent.q_estimator.qnet.load_state_dict(state_dict['sl_agent']['q_estimator'])
        self._sl_agent.target_estimator.qnet.load_state_dict(state_dict['sl_agent']['target_estimator'])

def train(args):
    """Train NFSP agents for Fortyfives."""
    # Parse hidden layers sizes from string to list
    hidden_layers = eval(args.hidden_layers_sizes)

    # Create the environment
    env = rlcard.make('fortyfives')
    eval_env = rlcard.make('fortyfives')

    # Debug print environment info
    print("Environment info:")
    print(f"Number of players: {env.num_players}")
    print(f"State shape: {env.state_shape}")
    print(f"Number of actions: {env.num_actions}")

    # Initialize agents
    agents = []
    for player_id in range(env.game.get_num_players()):
        # Create phase-specific agents for each player
        player_agents = [
            BiddingAgent(num_actions=env.num_actions),
            DeclarationAgent(num_actions=env.num_actions),
            PlayAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape,
                hidden_layers_sizes=hidden_layers,
                q_replay_memory_size=args.q_replay_memory_size,
                q_replay_memory_init_size=args.q_replay_memory_init_size,
                q_update_target_estimator_every=args.q_update_target_estimator_every,
                q_discount_factor=args.q_discount_factor,
                q_epsilon_start=args.q_epsilon_start,
                q_epsilon_end=args.q_epsilon_end,
                q_epsilon_decay_steps=args.q_epsilon_decay_steps,
                q_batch_size=args.q_batch_size,
                q_learning_rate=args.q_learning_rate,
                min_buffer_size_to_learn=args.min_buffer_size_to_learn,
                sl_learning_rate=args.sl_learning_rate,
                sl_batch_size=args.sl_batch_size,
                sl_epsilon=args.sl_epsilon,
                device=args.device,
                reservoir_buffer_capacity=args.reservoir_buffer_capacity
            )
        ]
        agents.append(player_agents)

    # Ensure log directory exists
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(args.log_dir, 'best_model.pth')), exist_ok=True)

    # Initialize metrics
    total_timesteps = 0
    reward_buffer = []
    best_reward = float('-inf')
    patience_counter = 0

    # Create CSV file for logging rewards
    reward_log_path = os.path.join(args.log_dir, 'rewards.csv')
    with open(reward_log_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'phase', 'player', 'reward'])

    # Training loop
    for episode in range(args.num_episodes):
        print(f"\nStarting episode {episode}")

        # Rotate agent positions every rotate_every episodes
        if episode > 0 and episode % args.rotate_every == 0:
            agents = agents[1:] + [agents[0]]
            print(f"Rotated agents at episode {episode}")

        # Reset environment
        trajectories = [[] for _ in range(env.num_players)]
        state, player_id = env.reset()
        done = False
        current_phase = 'bidding'  # Start with bidding phase

        print(f"Initial state:")
        print(f"Phase: {current_phase}")
        print(f"Player: {player_id}")

        # Store initial state for each player
        for p_id in range(env.num_players):
            trajectories[p_id].append({
                'state': state,
                'action': None,
                'reward': 0,
                'next_state': None,
                'done': False,
                'phase': current_phase
            })

        step_count = 0
        while not done:
            step_count += 1
            # Get current phase agent
            phase_idx = {'bidding': 0, 'declaration': 1, 'play': 2}[current_phase]
            current_agent = agents[player_id][phase_idx]

            # Get action from agent
            action = current_agent.step(state)
            
            print(f"\nStep {step_count}:")
            print(f"Phase: {current_phase}")
            print(f"Player {player_id} taking action {action}")
            
            # Take action in environment
            next_state, done = env.step(action)
            
            # Get phase from state
            old_phase = current_phase
            if 'phase' in next_state['raw_obs']:
                phase_map = {0: 'bidding', 1: 'declaration', 2: 'play'}
                current_phase = phase_map[next_state['raw_obs']['phase']]
                if old_phase != current_phase:
                    print(f"Phase transition: {old_phase} -> {current_phase}")

            # Get intermediate rewards
            payoffs = env.get_payoffs()
            print(f"Payoffs: {payoffs}")

            # Update trajectories for all players
            for p_id in range(env.num_players):
                if trajectories[p_id][-1]['action'] is not None:
                    # Update the last transition
                    trajectories[p_id][-1]['next_state'] = next_state
                    trajectories[p_id][-1]['done'] = done
                    trajectories[p_id][-1]['reward'] = payoffs[p_id]

                # Add new transition
                if not done:
                    trajectories[p_id].append({
                        'state': next_state,
                        'action': None,
                        'reward': 0,
                        'next_state': None,
                        'done': False,
                        'phase': current_phase
                    })

            # Update the action for the current player's latest transition
            trajectories[player_id][-1]['action'] = action

            # Log rewards
            with open(reward_log_path, 'a') as f:
                writer = csv.writer(f)
                for p_id, payoff in enumerate(payoffs):
                    writer.writerow([episode, current_phase, p_id, payoff])

            # Move to next state
            state = next_state
            player_id = env.get_player_id()

            print(f"Next player: {player_id}")
            print(f"Done: {done}")

        print(f"\nEpisode {episode} completed after {step_count} steps")
        print(f"Final payoffs: {payoffs}")

        # Train agents
        for player_id, player_agents in enumerate(agents):
            play_agent = player_agents[2]  # Get PlayAgent
            if isinstance(play_agent, PlayAgent):
                play_transitions = [ts for ts in trajectories[player_id] if ts['phase'] == 'play' and ts['action'] is not None]
                print(f"Training player {player_id}'s PlayAgent with {len(play_transitions)} play phase transitions")
                for ts in play_transitions:
                    play_agent.feed(ts)

        # Evaluate every eval_num episodes
        if episode % args.eval_num == 0:
            rewards = []
            for eval_episode in range(args.evaluation_num):
                state, player_id = eval_env.reset()
                eval_done = False
                current_phase = 'bidding'
                episode_rewards = [0] * eval_env.num_players

                while not eval_done:
                    phase_idx = {'bidding': 0, 'declaration': 1, 'play': 2}[current_phase]
                    current_agent = agents[player_id][phase_idx]
                    action = current_agent.eval_step(state)
                    next_state, eval_done = eval_env.step(action)
                    
                    if 'phase' in next_state['raw_obs']:
                        phase_map = {0: 'bidding', 1: 'declaration', 2: 'play'}
                        current_phase = phase_map[next_state['raw_obs']['phase']]
                    
                    state = next_state
                    player_id = eval_env.get_player_id()

                    if eval_done:
                        eval_payoffs = eval_env.get_payoffs()
                        for p_id in range(eval_env.num_players):
                            episode_rewards[p_id] = eval_payoffs[p_id]

                rewards.append(np.mean(episode_rewards))

            mean_reward = np.mean(rewards)
            reward_buffer.append(mean_reward)

            print(f'Episode {episode}, evaluation average reward: {mean_reward:.4f}')

            # Save plots
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(range(0, episode + 1, args.eval_num), reward_buffer)
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            plt.title('Learning Curve')

            # Plot individual player rewards
            plt.subplot(1, 2, 2)
            player_rewards = {i: [] for i in range(4)}
            with open(reward_log_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    player_rewards[int(row['player'])].append(float(row['reward']))
            
            for player, rewards in player_rewards.items():
                plt.plot(rewards[-100:], label=f'Player {player}')
            plt.xlabel('Recent Episodes')
            plt.ylabel('Reward')
            plt.title('Recent Player Rewards')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(args.log_dir, 'learning_curves.png'))
            plt.close()

            # Save best model
            if mean_reward > best_reward:
                best_reward = mean_reward
                patience_counter = 0
                # Save only PlayAgent models
                state_dicts = []
                for player_agents in agents:
                    play_agent = player_agents[2]
                    if isinstance(play_agent, PlayAgent):
                        state_dicts.append(play_agent.get_state_dict())
                torch.save(state_dicts, os.path.join(args.log_dir, 'best_model.pth'))
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= args.patience:
                print(f'Early stopping triggered after {episode} episodes')
                break

    # Final evaluation
    rewards = []
    for eval_episode in range(args.evaluation_num):
        state, player_id = eval_env.reset()
        eval_done = False
        current_phase = 'bidding'
        episode_rewards = [0] * eval_env.num_players

        while not eval_done:
            phase_idx = {'bidding': 0, 'declaration': 1, 'play': 2}[current_phase]
            current_agent = agents[player_id][phase_idx]
            action = current_agent.eval_step(state)
            next_state, eval_done = eval_env.step(action)
            
            if 'phase' in next_state['raw_obs']:
                phase_map = {0: 'bidding', 1: 'declaration', 2: 'play'}
                current_phase = phase_map[next_state['raw_obs']['phase']]
            
            state = next_state
            player_id = eval_env.get_player_id()

            if eval_done:
                eval_payoffs = eval_env.get_payoffs()
                for p_id in range(eval_env.num_players):
                    episode_rewards[p_id] = eval_payoffs[p_id]

        rewards.append(np.mean(episode_rewards))

    mean_reward = np.mean(rewards)
    print(f'Final evaluation average reward: {mean_reward:.4f}')

    # Save final plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(0, episode + 1, args.eval_num), reward_buffer)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Final Learning Curve')

    # Plot individual player rewards
    plt.subplot(1, 2, 2)
    player_rewards = {i: [] for i in range(4)}
    with open(reward_log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            player_rewards[int(row['player'])].append(float(row['reward']))
    
    for player, rewards in player_rewards.items():
        plt.plot(rewards[-100:], label=f'Player {player}')
    plt.xlabel('Recent Episodes')
    plt.ylabel('Reward')
    plt.title('Final Player Rewards')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, 'final_learning_curves.png'))
    plt.close()

    return mean_reward

def evaluate(args):
    # Load pretrained model
    try:
        agent = NFSPAgent.from_checkpoint(torch.load(args.model_path, map_location=get_device()))
        print(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Set up environment
    env = rlcard.make('fortyfives')
    
    # Set up agents (trained agent as player 0, random agents for others)
    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    
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
            action, _ = env.agents[player_id].eval_step(state)
            next_state, next_player_id = env.step(action)
            
            # Check if the game is done
            if step_count >= 1000 or (hasattr(env.game, 'is_over') and env.game.is_over()):
                done = True
                payoffs = env.get_payoffs()
                rewards.append(payoffs[0])  # Record reward for player 0 (the NFSP agent)
            
            # Move to the next state
            state = next_state
            player_id = next_player_id
            
        if episode % 10 == 0:
            print(f'Episode {episode}, average reward: {np.mean(rewards):.4f}')
            
    print(f'Final average reward after {args.eval_num} episodes: {np.mean(rewards):.4f}')
    
    # Optional: visualize some game states
    if args.visualize:
        state, player_id = env.reset()
        print('Initial state:')
        env.render()
        
        # Play a few steps
        for _ in range(5):
            action, probs = env.agents[player_id].eval_step(state)
            print(f'Player {player_id} takes action: {action}')
            print(f'Action probabilities: {probs}')
            next_state, next_player_id = env.step(action)
            print('New state:')
            env.render()
            state = next_state
            player_id = next_player_id

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("NFSP example in RLCard")
    parser.add_argument('--env', type=str, default='fortyfives',
            choices=['blackjack', 'leduc-holdem', 'limit-holdem', 'doudizhu', 'mahjong', 'no-limit-holdem', 'uno', 'gin-rummy', 'fortyfives'])
    parser.add_argument('--algorithm', type=str, default='nfsp',
            choices=['dqn', 'nfsp'])
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--num_eval_games', type=int, default=2000)
    parser.add_argument('--evaluate_every', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default='experiments/fortyfives_nfsp_result/')
    parser.add_argument('--load_model', action='store_true',
                    help='Load an existing model')
    parser.add_argument('--load_model_path', type=str, default='')

    # NFSP parameters
    parser.add_argument('--hidden_layers_sizes', type=str, default='[512, 512]',
            help='The hidden layers sizes for the neural networks')
    parser.add_argument('--reservoir_buffer_capacity', type=int, default=100000,
            help='The capacity of the reservoir buffer')
    parser.add_argument('--q_replay_memory_size', type=int, default=100000,
            help='The size of the replay memory for Q-learning')
    parser.add_argument('--q_replay_memory_init_size', type=int, default=1000,
            help='The initial size of the replay memory for Q-learning')
    parser.add_argument('--q_update_target_estimator_every', type=int, default=1000,
            help='The frequency of updating the target estimator for Q-learning')
    parser.add_argument('--q_discount_factor', type=float, default=0.99,
            help='The discount factor for Q-learning')
    parser.add_argument('--q_epsilon_start', type=float, default=1.0,
            help='The starting epsilon for Q-learning')
    parser.add_argument('--q_epsilon_end', type=float, default=0.01,
            help='The ending epsilon for Q-learning')
    parser.add_argument('--q_epsilon_decay_steps', type=int, default=100000,
            help='The number of steps to decay epsilon for Q-learning')
    parser.add_argument('--q_batch_size', type=int, default=128,
            help='The batch size for Q-learning')
    parser.add_argument('--q_learning_rate', type=float, default=0.0005,
            help='The learning rate for Q-learning')
    parser.add_argument('--min_buffer_size_to_learn', type=int, default=1000,
            help='The minimum buffer size to start learning')
    parser.add_argument('--sl_learning_rate', type=float, default=0.0002,
            help='The learning rate for supervised learning')
    parser.add_argument('--sl_batch_size', type=int, default=128,
            help='The batch size for supervised learning')
    parser.add_argument('--sl_epsilon', type=float, default=0.1,
            help='The epsilon for supervised learning')
    parser.add_argument('--device', type=str, default='cpu',
            help='The device to run the training on')

    # Training parameters
    parser.add_argument('--rotate_every', type=int, default=100,
            help='The number of episodes between agent rotation')
    parser.add_argument('--eval_num', type=int, default=50,
            help='The number of episodes between evaluations')
    parser.add_argument('--evaluation_num', type=int, default=20,
            help='The number of evaluation episodes')
    parser.add_argument('--patience', type=int, default=10,
            help='The number of evaluations without improvement before early stopping')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train(args) 