#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_reward(log_dir):
    """
    Plot the reward vs episode from a performance.csv file
    """
    # Load the performance data
    perf_csv_path = os.path.join(log_dir, 'performance.csv')
    if not os.path.exists(perf_csv_path):
        print(f"Performance file not found at {perf_csv_path}")
        return
    
    # Read the CSV file
    df = pd.read_csv(perf_csv_path)
    print(f"Loaded performance data with {len(df)} entries")
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['episode'], df['reward'], 'b-', label='Evaluation Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Evaluation Reward over Episodes')
    plt.grid(True)
    
    # Add a smoothed version of the reward to see the trend better
    window_size = 3  # Smaller window since we have fewer evaluation points
    if len(df) > window_size:
        smoothed_reward = df['reward'].rolling(window=window_size).mean()
        plt.plot(df['episode'], smoothed_reward, 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
        plt.legend()
    
    # Add a horizontal line at 0 for reference
    plt.axhline(y=0, color='g', linestyle='-', alpha=0.3)
    
    # Save the plot
    output_file = os.path.join(log_dir, 'reward_plot.png')
    plt.savefig(output_file)
    print(f"Reward plot saved to {output_file}")
    
    # Show some statistics
    print(f"Reward statistics:")
    print(f"  Min reward: {df['reward'].min():.6f}")
    print(f"  Max reward: {df['reward'].max():.6f}")
    print(f"  Mean reward: {df['reward'].mean():.6f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot reward vs episode')
    parser.add_argument('--log_dir', type=str, 
                        default='experiments/fortyfives_dqn_early_stopping',
                        help='Directory containing the performance.csv file')
    args = parser.parse_args()
    
    plot_reward(args.log_dir) 