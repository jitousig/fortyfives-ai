import os
import argparse
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
import signal
import sys
import re

class LossMonitor:
    def __init__(self, log_file, window_size=50, update_interval=1.0):
        self.log_file = log_file
        self.window_size = window_size
        self.update_interval = update_interval  # seconds
        
        # Data storage
        self.steps = []
        self.losses = []
        self.moving_avg = deque(maxlen=window_size)
        
        # Set up the figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.line, = self.ax.plot([], [], 'b-', lw=2, label='Loss')
        self.moving_avg_line, = self.ax.plot([], [], 'r-', lw=2, label=f'Moving Avg (window={window_size})')
        
        # Configure plot
        self.ax.set_xlabel('Training Steps')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('DQN Loss over Training')
        self.ax.grid(True)
        self.ax.legend()
        
        # Last read position
        self.last_position = 0
        self.last_file_size = 0
        
        # Counter for sequential steps if all steps are 0
        self.sequential_step = 0
        
        # Register signal handlers for clean exit
        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)
        
        # Animation
        self.anim = FuncAnimation(
            self.fig, self.update, interval=update_interval*1000, 
            cache_frame_data=False, save_count=100
        )
        
        plt.tight_layout()
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)
        
        print(f"Monitoring log file: {self.log_file}")
        print("Press Ctrl+C to exit.")
    
    def update(self, frame):
        """Update the plot with new data from the log file"""
        new_data = self.read_log_data()
        
        if new_data:
            print(f"Adding {len(new_data)} new data points")
            
            # Update data
            for step, loss in new_data:
                self.steps.append(step)
                self.losses.append(loss)
                self.moving_avg.append(loss)
            
            # Update plot data
            self.line.set_data(self.steps, self.losses)
            
            # Update moving average
            if len(self.moving_avg) > 0:
                window = min(len(self.moving_avg), self.window_size)
                moving_avg_data = []
                
                for i in range(len(self.losses)):
                    if i < window-1:
                        # Not enough data for full window
                        start_idx = 0
                        end_idx = i+1
                    else:
                        # Full window
                        start_idx = i-window+1
                        end_idx = i+1
                    
                    avg = sum(self.losses[start_idx:end_idx]) / (end_idx - start_idx)
                    moving_avg_data.append(avg)
                
                self.moving_avg_line.set_data(self.steps, moving_avg_data)
            
            # Adjust limits
            if self.steps:
                self.ax.set_xlim(min(self.steps), max(self.steps) + 10)
                
                # For y-axis, give some padding above and below
                if self.losses:
                    y_min = min(self.losses) * 0.9
                    y_max = max(self.losses) * 1.1
                    self.ax.set_ylim(y_min, y_max)
            
            # Update plot
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        
        return self.line, self.moving_avg_line
    
    def read_log_data(self):
        """Read new loss data from the log file"""
        if not os.path.exists(self.log_file):
            print(f"Waiting for log file {self.log_file} to be created...")
            return []
        
        new_data = []
        try:
            # Get file size
            file_size = os.path.getsize(self.log_file)
            
            # Only read if file has changed
            if file_size > self.last_file_size:
                print(f"File size changed: {self.last_file_size} -> {file_size}")
                with open(self.log_file, 'r') as f:
                    # Move to last position if file grew
                    f.seek(self.last_position)
                    
                    for line in f:
                        # Look for loss values
                        if "rl-loss:" in line:
                            try:
                                # Try to extract step number and loss value using regex
                                match = re.search(r'Step (\d+), rl-loss: ([0-9.]+)', line)
                                if match:
                                    step = int(match.group(1))
                                    loss = float(match.group(2))
                                    
                                    # If all steps are 0, use sequential numbering
                                    if step == 0:
                                        step = self.sequential_step
                                        self.sequential_step += 1
                                    
                                    new_data.append((step, loss))
                                    print(f"Found loss: {loss} at step {step}")
                                else:
                                    # Try alternate format with just the loss value
                                    match = re.search(r'rl-loss: ([0-9.]+)', line)
                                    if match:
                                        loss = float(match.group(1))
                                        step = self.sequential_step
                                        self.sequential_step += 1
                                        new_data.append((step, loss))
                                        print(f"Found loss: {loss} at sequential step {step}")
                            except (IndexError, ValueError) as e:
                                print(f"Error parsing line: {line.strip()} - {e}")
                    
                    # Update last position
                    self.last_position = f.tell()
                    self.last_file_size = file_size
            else:
                # Sleep briefly to avoid high CPU usage
                time.sleep(0.1)
        except Exception as e:
            print(f"Error reading log file: {e}")
        
        return new_data
    
    def handle_exit(self, sig, frame):
        """Handle clean exit when CTRL+C is pressed"""
        print("\nExiting gracefully...")
        plt.close(self.fig)
        sys.exit(0)
    
    def save_plot(self, save_path):
        """Save the current plot to a file"""
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Monitor and plot DQN loss in real-time")
    parser.add_argument('--log_file', type=str, required=True, 
                        help='Path to the log file to monitor')
    parser.add_argument('--window_size', type=int, default=50,
                        help='Window size for moving average calculation')
    parser.add_argument('--update_interval', type=float, default=1.0,
                        help='Update interval in seconds')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the final plot')
    
    args = parser.parse_args()
    
    # Create the monitor
    monitor = LossMonitor(
        log_file=args.log_file,
        window_size=args.window_size,
        update_interval=args.update_interval
    )
    
    try:
        print("Starting real-time monitoring. Press Ctrl+C to exit.")
        # Keep the reference to the monitor to prevent GC
        while True:
            plt.pause(0.1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if args.output:
            monitor.save_plot(args.output)

if __name__ == "__main__":
    main() 