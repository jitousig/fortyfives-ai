#!/usr/bin/env python3
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

def extract_loss_from_log(log_file):
    """
    Extract loss values from a log file with lines like:
    INFO - Step 0, rl-loss: 0.7766412496566772
    """
    episodes = []
    losses = []
    
    # Pattern to match episode numbers
    episode_pattern = re.compile(r'Episode (\d+)/')
    current_episode = None
    
    # Pattern to match loss values
    loss_pattern = re.compile(r'INFO - Step \d+, rl-loss: ([\d\.]+)')
    
    with open(log_file, 'r') as f:
        for line in f:
            # Check for episode updates
            episode_match = episode_pattern.search(line)
            if episode_match:
                current_episode = int(episode_match.group(1))
            
            # Check for loss values
            loss_match = loss_pattern.search(line)
            if loss_match and current_episode is not None:
                loss = float(loss_match.group(1))
                episodes.append(current_episode)
                losses.append(loss)
    
    return episodes, losses

def save_loss_to_csv(episodes, losses, output_file):
    """Save the loss data to a CSV file"""
    df = pd.DataFrame({'episode': episodes, 'loss': losses})
    df.to_csv(output_file, index=False)
    print(f"Saved loss data to {output_file}")
    return df

def plot_loss(df, output_file):
    """Create a plot of loss vs episode"""
    plt.figure(figsize=(12, 6))
    plt.plot(df['episode'], df['loss'])
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss over Episodes')
    plt.grid(True)
    
    # Add a smoothed version of the loss to see the trend better
    window_size = 10
    if len(df) > window_size:
        smoothed_loss = df['loss'].rolling(window=window_size).mean()
        plt.plot(df['episode'], smoothed_loss, 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
        plt.legend()
    
    plt.savefig(output_file)
    print(f"Loss plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract and plot loss values from a log file')
    parser.add_argument('--log_file', type=str, required=True,
                        help='Path to the log file containing training output')
    parser.add_argument('--output_dir', type=str, default='experiments/loss_analysis',
                        help='Directory to save the CSV and plot files')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract loss values from the log file
    episodes, losses = extract_loss_from_log(args.log_file)
    
    if not episodes:
        print("No loss values found in the log file")
        sys.exit(1)
    
    print(f"Extracted {len(episodes)} loss values")
    
    # Save to CSV
    csv_file = os.path.join(args.output_dir, 'extracted_loss.csv')
    df = save_loss_to_csv(episodes, losses, csv_file)
    
    # Create plot
    plot_file = os.path.join(args.output_dir, 'loss_plot.png')
    plot_loss(df, plot_file)
    
    # Print some statistics
    print(f"Loss statistics:")
    print(f"  Min loss: {df['loss'].min():.6f}")
    print(f"  Max loss: {df['loss'].max():.6f}")
    print(f"  Mean loss: {df['loss'].mean():.6f}") 