# 45s Card Game

A reinforcement learning environment for the Nova Scotia variant of 45s card game built on RLcard.

## Game Description

45s is a trick-taking card game played with a standard 52-card deck. It's played with 4 players in partnership (North/South vs East/West). The objective is to be the first partnership to reach 125 points.

### Game Phases

1. **Deal**: 5 cards to each player, 3 cards to the kitty
2. **Auction**: Players bid 20, 25, 30, hold (dealer only), or pass
3. **Declaration**: Winning bidder declares the trump suit
4. **Discard**: Players discard unwanted cards, highest bidder must discard to have at most 5 cards
5. **Gameplay**: 5 tricks are played with complex card ranking system
6. **Scoring**: Partnerships score points based on tricks won and highest trump

## Features

- Full implementation of 45s game rules including:
  - Partnership play (North/South vs East/West)
  - Bidding system (20, 25, 30, hold, pass)
  - Complex card ranking system
  - Special trump rules (Ace of Hearts is always trump)
  - Reneging rules for high trump cards

## Project Structure

```
fortyfives/
├── fortyfives/
│   ├── __init__.py
│   ├── games/
│   │   ├── __init__.py
│   │   └── fortyfives/
│   │       ├── __init__.py
│   │       ├── card.py
│   │       ├── dealer.py
│   │       └── game.py
│   └── envs/
│       ├── __init__.py
│       └── fortyfives_env.py
├── examples/
│   └── fortyfives_random.py
├── tests/
│   ├── __init__.py
│   └── test_fortyfives.py
├── setup.py
└── requirements.txt
```

## Requirements

- Python 3.6+
- RLcard
- numpy
- matplotlib

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/fortyfives.git

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Usage

```python
import rlcard
from rlcard.agents import RandomAgent

# Make environment
env = rlcard.make('fortyfives')

# Set up agents
agents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)]
env.set_agents(agents)

# Play a game
trajectories, payoffs = env.run()
```

## Running Examples

```bash
# Run the random agent example
python examples/fortyfives_random.py
```

## Running Tests

```bash
# Run all tests
python -m unittest discover -s tests

# Run a specific test
python -m unittest tests.test_fortyfives
```

## Card Ranking

Card rankings in 45s are complex and depend on the trump suit and the suit color:

- **Red trump**: 5, J, A♥, A, K, Q, 10, 9, 8, 7, 6, 4, 3, 2
- **Black trump**: 5, J, A♥, A, K, Q, 2, 3, 4, 6, 7, 8, 9, 10
- **Red non-trump**: K, Q, J, 10, 9, 8, 7, 6, 4, 3, 2, A
- **Black non-trump**: K, Q, J, A, 2, 3, 4, 6, 7, 8, 9, 10

Note that the Ace of Hearts is always considered a trump card, regardless of the declared trump suit.

## License

MIT 