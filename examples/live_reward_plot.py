#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time

def live_plot_reward(log_dir, update_interval=10, window_size=3):
    """
    Live plot of rewards from a performance.csv file
    
    Args:
        log_dir: Directory containing the performance.csv file
        update_interval: How often to update the plot (in seconds)
        window_size: Window size for the moving average calculation
    """
    # Setup the plot
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    line1, = ax.plot([], [], 'b-', label='Evaluation Reward')
    line2, = ax.plot([], [], 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
    
    # Add a horizontal line at 0 for reference
    ax.axhline(y=0, color='g', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Live Evaluation Reward over Episodes')
    ax.grid(True)
    ax.legend()
    
    # Set up for animation
    plt.tight_layout()
    plt.show(block=False)
    
    # Variables to track statistics
    best_reward = float('-inf')
    worst_reward = float('inf')
    
    # Main loop
    print(f"Starting live reward plotting. Press Ctrl+C to stop.")
    try:
        while True:
            # Check if the file exists
            perf_csv_path = os.path.join(log_dir, 'performance.csv')
            if os.path.exists(perf_csv_path):
                try:
                    # Read the CSV file
                    df = pd.read_csv(perf_csv_path)
                    
                    if not df.empty:
                        # Update the lines
                        episodes = df['episode'].values
                        rewards = df['reward'].values
                        
                        # Update statistics
                        current_best = rewards.max()
                        current_worst = rewards.min()
                        current_mean = rewards.mean()
                        
                        if current_best > best_reward:
                            best_reward = current_best
                            print(f"New best reward: {best_reward:.4f}")
                        
                        if current_worst < worst_reward:
                            worst_reward = current_worst
                        
                        # Update plot data
                        line1.set_data(episodes, rewards)
                        
                        # Calculate moving average if we have enough data
                        if len(df) > window_size:
                            ma = df['reward'].rolling(window=window_size).mean()
                            line2.set_data(episodes, ma)
                        
                        # Adjust plot limits
                        ax.relim()
                        ax.autoscale_view()
                        
                        # Update title with statistics
                        ax.set_title(f'Live Evaluation Reward - Best: {best_reward:.4f}, Mean: {current_mean:.4f}')
                        
                        # Redraw the figure
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                        
                        # Print status
                        print(f"Updated plot with {len(df)} data points. " +
                              f"Latest reward ({episodes[-1]:.0f}): {rewards[-1]:.4f}, " +
                              f"Mean: {current_mean:.4f}")
                except Exception as e:
                    print(f"Error updating plot: {e}")
            else:
                print(f"Waiting for performance.csv file at {perf_csv_path}...")
            
            # Sleep for the update interval
            time.sleep(update_interval)
    except KeyboardInterrupt:
        print("Live plotting stopped by user.")
        
    # Save the final plot
    plt.savefig(os.path.join(log_dir, 'live_reward_plot.png'))
    print(f"Final plot saved to {os.path.join(log_dir, 'live_reward_plot.png')}")
    
    # Wait for user to close the plot
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Live plot of reward vs episode')
    parser.add_argument('--log_dir', type=str, 
                        default='experiments/fortyfives_dqn_long_training',
                        help='Directory containing the performance.csv file')
    parser.add_argument('--update_interval', type=float, default=10.0,
                        help='Update interval in seconds')
    parser.add_argument('--window_size', type=int, default=3,
                        help='Window size for moving average')
    args = parser.parse_args()
    
    live_plot_reward(args.log_dir, args.update_interval, args.window_size) 