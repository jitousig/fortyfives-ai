# Live Loss Visualization for DQN Training

This document explains how to visualize the loss in real-time while training your DQN agent for Fortyfives.

## Setup and Usage

### Option 1: Run Training with Output Redirection

1. First, start your training with output redirection to a log file:
   ```
   python examples/fortyfives_dqn_selfplay_early_stopping.py --num_episodes 5000 --log_dir experiments/fortyfives_dqn_training --evaluate_every 100 --checkpoint_every 200 --rotate_every 500 > training_output.log 2>&1 &
   ```

2. In a separate terminal, run the loss visualization script:
   ```
   python examples/live_loss_plot.py --log_file training_output.log --window_size 50 --update_interval 1.0 --output loss_plot.png
   ```

### Option 2: Use Simulated Data for Testing

If you want to test the visualization without running a full training:

1. Run the simulation script:
   ```
   python examples/simulate_training.py --num_episodes 1000 --output_file simulated_training.log --delay 0.05 &
   ```

2. In a separate terminal, run the loss visualization script:
   ```
   python examples/live_loss_plot.py --log_file simulated_training.log --window_size 30 --update_interval 0.5 --output loss_plot.png
   ```

## Script Parameters

### live_loss_plot.py

- `--log_file`: Path to the training log file (required)
- `--window_size`: Window size for moving average calculation (default: 50)
- `--update_interval`: Update interval in seconds (default: 1.0)
- `--output`: Path to save the final plot (optional)

### simulate_training.py

- `--num_episodes`: Number of episodes to simulate (default: 1000)
- `--output_file`: Output log file path (default: 'simulated_training.log')
- `--delay`: Delay between episodes in seconds (default: 0.1)
- `--loss_start`: Initial loss value (default: 0.5)
- `--decay_rate`: Loss decay rate (default: 0.005)
- `--noise_level`: Noise level for loss values (default: 0.05)

## How It Works

The live loss visualization works by:

1. Monitoring the training log file for new entries
2. Extracting loss values whenever they appear in the log
3. Updating a real-time plot showing both the raw loss values and a moving average
4. Saving the final plot when you close the visualization or press Ctrl+C

## Troubleshooting

- If the plot is not updating, make sure your training script is outputting loss values in the expected format (`INFO - Step X, rl-loss: Y`)
- If you get an error about the log file not existing, check the path and make sure the training script is running correctly
- If the plot window disappears immediately, try running with a longer update interval

## Notes

- The moving average helps smooth out noise in the loss values
- You can adjust the window size to control how much smoothing is applied
- The visualization will continue running until you close the window or press Ctrl+C 