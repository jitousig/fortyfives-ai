import os
import argparse
import numpy as np
import torch
from torch.serialization import safe_globals
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from rlcard.agents.dqn_agent import DQNAgent
from rlcard.utils.utils import set_seed, tournament
from rlcard.utils.logger import Logger
import rlcard
import time
import datetime
from collections import deque
import sys

# Register Fortyfives environment
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

class LivePlotLogger:
    """Logger for live plotting of loss"""
    def __init__(self, window_size=100):
        # Set up the figure and axis for loss
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_xlabel('Episodes')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('DQN Loss over Episodes')
        self.ax.grid(True)
        
        # Data storage
        self.episodes = []
        self.losses = []
        self.window_size = window_size
        self.moving_avg = deque(maxlen=window_size)
        self.moving_avg_line, = self.ax.plot([], [], 'r-', lw=2)
        
        # Set up legend
        self.ax.legend(['Loss', f'Moving Avg (window={window_size})'])
        
        # Initialize animation
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=1000, blit=True, cache_frame_data=False)
        plt.ion()  # Interactive mode on
        plt.show(block=False)
        
    def log_loss(self, episode, loss):
        """Log a loss value"""
        self.episodes.append(episode)
        self.losses.append(loss)
        self.moving_avg.append(loss)
        
        # Update plot data
        self.line.set_data(self.episodes, self.losses)
        
        # Update moving average if we have enough data
        if len(self.moving_avg) > 0:
            avg_data = [None] * (len(self.episodes) - len(self.moving_avg)) + list(np.convolve(list(self.moving_avg), np.ones(min(len(self.moving_avg), self.window_size))/min(len(self.moving_avg), self.window_size), mode='valid'))
            self.moving_avg_line.set_data(self.episodes[-len(avg_data):], avg_data)
        
        # Adjust limits if needed
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Draw and flush events
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
    def update_plot(self, frame):
        """Animation update function"""
        return self.line, self.moving_avg_line
        
    def save_plot(self, save_path):
        """Save the plot to a file"""
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

def train(args):
    # Check if model exists
    checkpoint_path = os.path.join(args.log_dir, 'model.pth')
    if not os.path.exists(checkpoint_path) and not os.path.exists(os.path.join(args.log_dir, 'best_model.pth')):
        raise ValueError(f"No model found at {checkpoint_path} or {os.path.join(args.log_dir, 'best_model.pth')}")
    
    # Create directory if it doesn't exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    # Initialize live plot logger
    live_logger = LivePlotLogger(window_size=50)
    
    # Set random seed
    set_seed(args.seed)

    # Make environment
    env = rlcard.make('fortyfives', config={'seed': args.seed, 'allow_step_back': True})
    
    # Allowlist DQNAgent for deserialization
    torch.serialization.add_safe_globals([rlcard.agents.dqn_agent.DQNAgent])
    
    # Load existing model checkpoint
    if os.path.exists(os.path.join(args.log_dir, 'best_model.pth')):
        print(f"Loading best model from {os.path.join(args.log_dir, 'best_model.pth')}")
        checkpoint = torch.load(os.path.join(args.log_dir, 'best_model.pth'), weights_only=False)
    else:
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Initialize agents
    agents = []
    for player_id in range(env.num_players):
        agent = DQNAgent.from_checkpoint(checkpoint)
        agents.append(agent)
    
    # Set up positioning
    for i, agent in enumerate(agents):
        position = (i + args.start_position) % env.num_players
        agent.set_device(torch.device(args.device))
        agent.use_raw = False
    
    # Get initial state
    state, _ = env.reset()
    
    # Create logger
    logger = Logger(args.log_dir)
    
    # Set up metrics
    total_episodes = checkpoint['total_t']
    start_time = time.time()
    best_reward = float('-inf')
    patience = args.patience
    patience_counter = 0
    
    # Find highest episode number in existing logs
    performance_file = os.path.join(args.log_dir, 'performance.csv')
    if os.path.exists(performance_file):
        with open(performance_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:  # Header + at least one data line
                try:
                    last_episode = int(lines[-1].split(',')[0])
                    total_episodes = max(total_episodes, last_episode)
                except:
                    pass
    
    print(f"Continue training from episode {total_episodes}")
    
    # Start training
    try:
        for episode in range(args.num_episodes):
            # Execute one episode
            current_episode = total_episodes + episode + 1
            
            # Rotate agents to different positions if needed
            if args.rotate_every > 0 and episode > 0 and episode % args.rotate_every == 0:
                temp = agents[0]
                for i in range(len(agents)-1):
                    agents[i] = agents[i+1]
                agents[-1] = temp
                print(f"\nRotating agents after episode {current_episode}")
                
            for player_id in range(env.num_players):
                env.set_agents([agents[(player_id + i) % env.num_players] for i in range(env.num_players)])
                
                trajectory, _ = env.run(is_training=True)
                
                # Feed transitions into agent memory
                for ts in trajectory:
                    agent = agents[(player_id + ts[0]) % env.num_players]
                    agent.feed(ts)
                
                # Train the agents
                if current_episode % args.train_every == 0:
                    for i, agent in enumerate(agents):
                        loss = agent.train()
                        if i == 0:  # Only log loss for the first agent
                            live_logger.log_loss(current_episode, loss)
            
            # Evaluate the performance
            if episode % args.evaluate_every == 0:
                env.set_agents([agents[i] for i in range(env.num_players)])
                reward = tournament(env, args.num_eval_games)[0]
                logger.log_performance(current_episode, reward)
                
                # Early stopping check
                if reward > best_reward:
                    best_reward = reward
                    patience_counter = 0
                    # Save best model
                    print(f"\nNew best reward: {best_reward} at episode {current_episode}, saving model")
                    for i, agent in enumerate(agents):
                        torch.save(agent.checkpoint_attributes(), os.path.join(args.log_dir, f'best_model.pth'))
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\nEarly stopping triggered after {patience} evaluations without improvement")
                        break
                
                # Print out results
                print(f'\rEpisode {current_episode}, Reward: {reward}, Best Reward: {best_reward}')
            
            # Save checkpoint
            if episode % args.checkpoint_every == 0:
                for i, agent in enumerate(agents):
                    torch.save(agent.checkpoint_attributes(), os.path.join(args.log_dir, f'model.pth'))
            
            # Print progress
            if episode % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f'\rEpisode {current_episode}/{total_episodes + args.num_episodes}, Time: {elapsed_time:.2f}s', end='')
                
    # Handle keyboard interrupt
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Final save
    for i, agent in enumerate(agents):
        torch.save(agent.checkpoint_attributes(), os.path.join(args.log_dir, f'model.pth'))
    
    # Save final plot
    live_logger.save_plot(os.path.join(args.log_dir, 'loss_visualization.png'))
    
    # Print training results
    elapsed_time = time.time() - start_time
    print(f'\nTraining completed in {elapsed_time:.2f}s')
    print(f'Best reward: {best_reward}')
    
    # Keep the plot window open until user closes it
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN Self-Play with Live Visualization")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_episodes', type=int, default=5000, help='Number of episodes to run')
    parser.add_argument('--num_eval_games', type=int, default=10, help='Number of games in each evaluation')
    parser.add_argument('--evaluate_every', type=int, default=100, help='Evaluate the agent every N episodes')
    parser.add_argument('--checkpoint_every', type=int, default=100, help='Save the model every N episodes')
    parser.add_argument('--rotate_every', type=int, default=500, help='Rotate agents every N episodes (0 to disable)')
    parser.add_argument('--log_dir', type=str, default='experiments/fortyfives_dqn_visualization', help='Directory to save the model and logs')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (cpu/cuda)')
    parser.add_argument('--start_position', type=int, default=0, help='Starting position of the main agent (0-3)')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience (evaluations)')
    parser.add_argument('--train_every', type=int, default=1, help='Train the agent every N episodes')
    args = parser.parse_args()
    
    train(args) 