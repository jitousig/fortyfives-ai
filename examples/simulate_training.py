"""
Simulate training data to demonstrate live loss plotting
"""

import os
import time
import random
import math
import argparse

def generate_noisy_loss(episode, start_value=0.5, decay_rate=0.01, noise_level=0.05):
    """Generate a synthetic loss value with noise"""
    # Exponential decay with some plateau
    base_loss = start_value * math.exp(-decay_rate * episode)
    # Add random noise
    noise = random.uniform(-noise_level, noise_level)
    # Ensure loss doesn't go negative
    return max(0.01, base_loss + noise)

def main():
    parser = argparse.ArgumentParser(description="Simulate DQN training data for visualization")
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes to simulate')
    parser.add_argument('--output_file', type=str, default='simulated_training.log', help='Output log file path')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between episodes (seconds)')
    parser.add_argument('--loss_start', type=float, default=0.5, help='Initial loss value')
    parser.add_argument('--decay_rate', type=float, default=0.005, help='Loss decay rate')
    parser.add_argument('--noise_level', type=float, default=0.05, help='Noise level for loss values')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    print(f"Simulating training for {args.num_episodes} episodes...")
    print(f"Output log file: {args.output_file}")
    print("Press Ctrl+C to stop simulation")
    
    try:
        with open(args.output_file, 'w') as f:
            for episode in range(1, args.num_episodes + 1):
                # Generate loss value
                loss = generate_noisy_loss(
                    episode, 
                    start_value=args.loss_start,
                    decay_rate=args.decay_rate,
                    noise_level=args.noise_level
                )
                
                # Write to log file
                log_line = f"INFO - Step {episode}, rl-loss: {loss:.6f}"
                f.write(log_line + '\n')
                f.flush()
                
                if episode % 10 == 0:
                    print(f"Episode {episode}/{args.num_episodes}, Loss: {loss:.6f}")
                
                # Simulate evaluations every 100 episodes
                if episode % 100 == 0:
                    eval_reward = random.uniform(-0.4, -0.1)  # Simulated reward
                    eval_line = f"Episode {episode}: Rotating agent to position {episode % 4}\nEpisode {episode}, Reward: {eval_reward}, Best Reward: {eval_reward}"
                    f.write(eval_line + '\n')
                    f.flush()
                    print(eval_line)
                
                # Wait
                time.sleep(args.delay)
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    print("Simulation complete!")

if __name__ == "__main__":
    main() 