# Forty Fives Agents

This directory contains various agent implementations for playing Forty Fives.

## Available Agents

1. **Random Agent** (`fortyfives_random.py`): An agent that selects random legal actions.
2. **Rule-Based Agent** (`fortyfives_rule_based.py`): An agent that follows basic Forty Fives strategy rules.
3. **DQN Agent** (`fortyfives_dqn.py`): A reinforcement learning agent that can be trained to play the game.

## How to Use the Agents

### Random Agent

The random agent provides a baseline for performance evaluation. To run a game with random agents:

```bash
python fortyfives_random.py
```

### Rule-Based Agent

The rule-based agent implements basic game strategies for bidding, trump selection, discarding, and gameplay. It can be evaluated against random agents:

```bash
python fortyfives_rule_based.py
```

By default, it plays 100 games with rule-based agents as North/South team against random agents as East/West team.

### DQN Agent (Reinforcement Learning)

The DQN agent uses deep reinforcement learning to learn optimal gameplay strategies. It can be trained and evaluated:

```bash
# Train a DQN agent
python fortyfives_dqn.py --mode train --num_episodes 5000 --log_dir experiments/fortyfives_dqn

# Evaluate a trained agent
python fortyfives_dqn.py --mode evaluate --model_path experiments/fortyfives_dqn/model.pth
```

Training parameters:

- `--num_episodes`: Number of games to play during training (default: 5000)
- `--evaluate_every`: Evaluate the agent every N episodes (default: 100)
- `--eval_num`: Number of games to play during each evaluation (default: 100)
- `--log_dir`: Directory for saving logs and model (default: experiments/fortyfives_dqn)
- `--seed`: Random seed for reproducibility (default: 42)

## Training Your Own Agents

### Basic Approach

1. Start with the random agent to understand the game flow and environment interface.
2. Use the rule-based agent to implement domain knowledge about the game.
3. Train a DQN agent against random or rule-based opponents to learn strategies.

### Advanced Approaches

For more advanced agent development:

1. **Self-Play**: Modify the DQN training to use self-play, where the agent plays against progressively updated versions of itself.
2. **PPO or A2C Agents**: Implement policy gradient methods like PPO (Proximal Policy Optimization) or A2C (Advantage Actor-Critic).
3. **Monte Carlo Tree Search**: Implement MCTS for more strategic gameplay, potentially combined with neural networks.

## Performance Comparison

You can evaluate different agents against each other by modifying the agent selection in the example scripts. For example, in `fortyfives_rule_based.py`, you could replace the random agents with trained DQN agents to compare performance.

## Customizing Agents

To create your own agent:

1. Implement a class with at least the following methods:
   - `__init__(self, num_actions)`: Initialize your agent
   - `step(self, state)`: Choose an action based on the current state
   
2. Set the `use_raw` attribute to indicate whether you want to use raw observations or encoded observations.

See the existing agent implementations for examples. 