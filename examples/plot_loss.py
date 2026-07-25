#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_loss(log_dir):
    """
    Plot the loss vs episode from a loss.csv file
    """
    # Load the loss data
    loss_csv_path = os.path.join(log_dir, 'loss.csv')
    if not os.path.exists(loss_csv_path):
        print(f"Loss file not found at {loss_csv_path}")
        return
    
    # Read the CSV file
    df = pd.read_csv(loss_csv_path)
    print(f"Loaded loss data with {len(df)} entries")
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['episode'], df['loss'])
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss over Episodes')
    plt.grid(True)
    
    # Add a smoothed version of the loss to see the trend better
    window_size = 100
    if len(df) > window_size:
        smoothed_loss = df['loss'].rolling(window=window_size).mean()
        plt.plot(df['episode'], smoothed_loss, 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
        plt.legend()
    
    # Save the plot
    output_file = os.path.join(log_dir, 'loss_plot.png')
    plt.savefig(output_file)
    print(f"Loss plot saved to {output_file}")
    
    # Show some statistics
    print(f"Loss statistics:")
    print(f"  Min loss: {df['loss'].min():.6f}")
    print(f"  Max loss: {df['loss'].max():.6f}")
    print(f"  Mean loss: {df['loss'].mean():.6f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot loss vs episode')
    parser.add_argument('--log_dir', type=str, 
                        default='experiments/fortyfives_dqn_early_stopping',
                        help='Directory containing the loss.csv file')
    args = parser.parse_args()
    
    plot_loss(args.log_dir) 