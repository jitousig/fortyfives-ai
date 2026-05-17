# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Fortyfives** is a reinforcement learning environment for the Nova Scotia variant of the 45s card game, built on RLCard. It implements a complete 4-player partnership card game with complex bidding, trump systems, and card ranking mechanics.

The project includes:
- A full game engine with rule enforcement
- An RLCard environment wrapper for RL agent training
- Multiple agent implementations (random, rule-based, DQN, NFSP)
- Comprehensive test suite covering game rules and card mechanics
- Training examples with self-play and evaluation utilities

## Installation & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

The project requires Python 3.6+, RLCard (>=1.0.7), NumPy, and Matplotlib.

## Running the Web GUI

The web GUI requires FastAPI and uvicorn. These live in the shared venv at `../fortyfives/fortyfives_env/`. `uvicorn` is not on the system PATH, so activate the venv first:

```bash
source ../fortyfives/fortyfives_env/bin/activate
uvicorn web.server:app --reload --port 8000
```

Then open http://localhost:8000 in a browser.

## Testing

Run all tests:
```bash
python -m unittest discover -s tests
```

Run a specific test file:
```bash
python -m unittest tests.test_fortyfives
```

The test suite is extensive with 12+ test files covering:
- Game initialization and state management
- Bidding logic and legal actions
- Card ranking with complex trump rules
- Trick playing, winning, and scoring
- Suit-following enforcement
- Special rules (30 for 60 bonus, pegging restrictions, renege rules)

## Project Structure

```
fortyfives/
├── fortyfives/                    # Main package
│   ├── games/
│   │   └── fortyfives/
│   │       ├── game.py           # FortyfivesGame class - core game engine
│   │       ├── card.py           # Card utilities, ranking logic
│   │       └── dealer.py         # Card dealing and deck management
│   └── envs/
│       └── fortyfives_env.py     # FortyfivesEnv - RLCard environment wrapper
├── examples/                      # Agent implementations and usage examples
│   ├── fortyfives_random.py       # RandomAgent baseline
│   ├── fortyfives_rule_based.py  # Rule-based agent with domain knowledge
│   ├── fortyfives_dqn*.py        # DQN agents with various training modes
│   ├── fortyfives_nfsp*.py       # Neural Fictitious Self-Play agents
│   └── *.py                       # Utilities: plotting, loss visualization, evaluation
├── tests/                         # Comprehensive test suite
│   ├── test_fortyfives.py        # Core game tests
│   ├── test_bidding_legal_actions.py
│   ├── test_trick_winning_and_scoring.py
│   ├── test_suit_following.py
│   ├── test_trump_always_playable.py
│   └── ... (12+ test files total)
└── experiments/                   # Training output directories
```

## Architecture Overview

### Game Engine (fortyfives/games/fortyfives/)

The game is driven by **FortyfivesGame**, which manages:
- **Game Phases** (5 phases):
  - PHASE_AUCTION (1): Bidding phase where players bid 20/25/30 or pass
  - PHASE_DECLARATION (2): Trump suit selection by highest bidder
  - PHASE_DISCARD (3): Players discard cards; highest bidder must reach ≤5 cards
  - PHASE_GAMEPLAY (4): Playing tricks with complex card ranking
  - PHASE_SCORING (5): Points awarded based on tricks and high trump

- **Game State Management**:
  - `hands`: Dict of player ID → card list
  - `bids`: Dict of player ID → bid value
  - `highest_bidder` / `highest_bid`: Track current leading bid
  - `trump_suit`: Declared trump (S/H/D/C)
  - `current_trick`: Cards played in current trick
  - `tricks_won`: Trick count per player
  - `points`: Partnership scores (0/2 vs 1/3)

### Card Ranking Logic (card.py)

The `get_card_rank()` function implements complex ranking based on:
- **Trump vs Non-Trump**: Trump cards always beat non-trump
- **Suit Color Dependency**: Red/Black trump and non-trump have different rankings
- **Ace of Hearts**: Always acts as a trump card (rank 1001), regardless of declared trump
- **Trump Card Order**: 5 > J > A♥ > A > K > Q > [suit-specific order]

Example rankings:
```
Red Trump: 5 (1003) > J (1002) > A♥ (1001) > A (1000) > K > Q > 10 > 9 > 8 > 7 > 6 > 4 > 3 > 2
Black Trump: 5 > J > A♥ > A > K > Q > 2 > 3 > 4 > 6 > 7 > 8 > 9 > 10
Red Non-Trump: K > Q > J > 10 > 9 > 8 > 7 > 6 > 4 > 3 > 2 > A (lowest!)
Black Non-Trump: K > Q > J > A > 2 > 3 > 4 > 6 > 7 > 8 > 9 > 10
```

### RLCard Integration (fortyfives_env.py)

**FortyfivesEnv** wraps the game for RL training:
- Extends `rlcard.envs.Env`
- Converts game state to neural network-friendly observation vectors (52*5 + features)
- Encodes legal actions dynamically based on game phase
- Manages agent-environment interaction loop

Key methods:
- `_extract_state()`: Converts raw game state to agent observations
- `_get_observation()`: One-hot encodes cards, phase, bids, points, tricks
- `_get_legal_actions()`: Returns valid actions for current phase

## Key Concepts & Rules

### Bidding System
- Players bid 20, 25, 30, or pass
- Dealer can "hold" (accept previous high bid)
- Auction ends when 3 players pass or all pass
- Highest bidder declares trump; others get 3 kitty cards

### Scoring Rules
- **30 for 60**: Making a 30 bid = 60 points (instead of standard 30)
- **Pegging**: Negative points end the game (goal is ≥125 to win)
- Partnership scoring: North/South (players 0/2) vs East/West (players 1/3)
- Points awarded: 5 per 5, 1 per Jack, 1 per Ace, 1 for high trump

### Card Playing Constraints
- **Suit Following**: Must follow suit if possible (except trump)
- **Trump Playable**: Trump always playable if in hand
- **Renege Rules**: High trump (5, J, A♥) must be played in some situations
- **Trick Winning**: Highest trump wins; highest suit card wins if no trump

## Development Workflow

### Adding New Agents

1. Implement a class with:
   ```python
   class CustomAgent:
       def __init__(self, num_actions):
           self.num_actions = num_actions
           self.use_raw = False  # or True for raw game state
       
       def step(self, state):
           # Return action index (0-17)
           pass
   ```

2. Test against baselines in examples:
   ```bash
   python examples/fortyfives_random.py
   ```

### Training RL Agents

The DQN and NFSP examples show training patterns:
- Self-play: Agent plays against copies of itself
- Opponent variety: Mix random, rule-based, and trained agents
- Evaluation: Periodic evaluation against fixed opponent set
- Logging: Loss/reward tracking with matplotlib visualization

Example training:
```bash
python examples/fortyfives_dqn.py --mode train --num_episodes 5000
```

### Modifying Game Rules

Game logic is in `FortyfivesGame`:
- Bidding: `get_legal_bids()`, `step()` during PHASE_AUCTION
- Declaration: PHASE_DECLARATION handling
- Discard: PHASE_DISCARD logic
- Play: `get_legal_actions()` during PHASE_GAMEPLAY
- Scoring: `get_payoffs()` at game end

Card evaluation uses `get_card_rank()` in card.py—modify this for ranking changes.

## Important Notes

- **State Representation**: Game state is a fixed-size vector (`state_shape`, currently **410**) built by `FortyfivesEnv._get_observation()`. Legacy non-card features are based at `52*5` (=260); enriched features (trick history, highest trump, trick standing, seat one-hots) are appended at index 295+. See invariants below before changing it.
- **Action Space**: 18 actions total (5 bid + 4 trump + 8 card play + 1 done). Actions are phase-dependent; illegal actions are filtered by `_get_legal_actions()`.
- **Random Seeding**: Use `env.seed()` for reproducible games; `game.np_random` controls shuffling.
- **Partnership Structure**: Always players 0/2 (North/South) vs 1/3 (East/West); hardcoded in payoff calculation.
- **Kitty Management**: 3 cards dealt to kitty; highest bidder gets them and must discard back to ≤5 cards.

## Hard-Won Invariants — READ BEFORE TOUCHING `fortyfives/` OR `examples/play_eval.py`

A long debugging session produced confident but **completely false** conclusions
because the measurement was silently broken. These invariants exist so that
never happens again. Treat a violation as "stop everything," not "investigate later."

### 1. Import shadowing (silent and severe)
`python examples/<script>.py` puts `examples/` on `sys.path[0]`, so bare
`import fortyfives` resolves to the **editable install in the shared venv**,
which points at the SIBLING repo `../fortyfives`, NOT this repo's package.
- The trainer/eval scripts prepend this repo's root to `sys.path[0]` to fix this. Do **not** remove that shim.
- Do **not** `pip install -e .` from this repo into the shared venv (breaks the sibling project).
- Before trusting ANY engine/env edit: confirm `fortyfives.__file__` resolves under `.../fortyfives-rl-training/` (or check `env.state_shape == 410`).

### 2. Evaluation canary (run after every env/eval change)
`play_eval.evaluate_paired(RuleBasedAgent(18))` (agent == baseline) MUST return
**diff 0.0000 / win 0.0%** — pure ties. If not, the eval is not actually paired
and every number it produces is noise. The rule-based agent MUST stay
deterministic (no `np.random`; use `min(legal_actions)` tie-break) because
`env.seed()` only seeds the dealer, never the global RNG.

### 3. Yardstick (how to judge an agent)
Trust `avg_diff` from `evaluate_paired` with its paired CI; ignore raw/unpaired
point means. Reference points: **random-legal ≈ −0.50, rule-based = 0.00.**
"Better than rule-based" = `avg_diff` significantly > 0 (CI excludes 0).

### 4. Observation correctness
Legacy non-card features start at `52*5` (=260), AFTER hand + 4 trick-card
blocks (0..259). Basing them at `52*4` collides with the 4th trick card and
corrupts phase/trump. The observation MUST encode trump suit and lead suit,
and (since the 410-wide enrichment) trick history / highest trump / trick
standing / seat one-hots — all decisive for play. Existing offsets 0..294 are
load-bearing for prior reasoning; ALWAYS append new features past 294 and bump
`state_shape` to match (never disturb 0..294, or the canary's meaning shifts).

### 5. Process
One change at a time, `git commit` per change, **plan-first** for non-trivial
work, and re-run invariant #2 after any `env`/`eval` edit. A result you cannot
trust is worse than no result.

## Repository Workflow & Branching Discipline — READ BEFORE ANY GIT OP

This is a **monorepo** with three components over one shared core. A
long-divergent branch + unpushed commits + a `git reset --hard` once nearly
destroyed an entire session's research. These rules are invariants, not
suggestions. The monorepo is correct (a core fix must reach research AND the
UIs atomically) — the failure mode is branch divergence, not repo structure.

### Components & the one-way dependency rule
- `fortyfives/` — core game logic. STABLE. Imported by everything; imports
  none of our other code.
- agent modules (`examples/fortyfives_pimc.py`,
  `examples/fortyfives_rule_based.py`, or an `agents/` dir) — released,
  UI-usable agents. Import only `fortyfives/`.
- research (`examples/` eval/training, experiments) — imports core + agents;
  churns freely; NOT imported by the UI.
- `web/` / mobile — UI. Imports core + agents. Nothing imports the UI.
- Imports flow ONE way: **core → agents → {research, UI}**. research and UI
  never import each other. A bug found via the UI is fixed in `fortyfives/`,
  never patched around in the UI.

### Branching — trunk-based, kept boring
- `main` is the single source of truth and MUST stay green (tests pass).
  Core fixes land on `main` FIRST.
- Work on SHORT-lived branches off `main`; push the branch immediately on
  creation; merge back when green; delete it after merge.
- Rebase/merge from `main` frequently. NEVER let a branch diverge far.
- Released agents are ARTIFACTS, not branches: pure-code agents live on
  `main` and are git-TAGGED at release (`agent-vN`); the UI imports them
  from `main`. Trained weights → GitHub Releases / git-LFS, never committed
  source.

### Git safety (each of these, alone, would have prevented the incident)
1. PUSH immediately. An unpushed commit is unrecoverable once the branch
   moves.
2. NEVER `git reset --hard`, `checkout -f`, or `branch -f` a branch that has
   unpushed commits or a dirty tree. Check `git status` and
   `git log @{u}..HEAD` first.
3. Before ANY destructive git op, confirm every commit is reachable from a
   pushed ref (`git branch -r --contains <sha>`); if not, push it first.
4. Do NOT propagate core fixes sideways between divergent branches via
   ad-hoc cherry-pick or reset+merge. Land the fix on `main`; other branches
   merge `main`.
5. Bug found while testing (including via the UI): add a FAILING test in
   `tests/` that reproduces it, fix in core on a short branch, merge to
   `main`, other branches merge `main`. `main` stays green.
6. Prefer git worktrees (already in use) to run components in parallel.
   Never branch-switch or reset a worktree that has a dirty tree or a
   running experiment.

### This document
This guidance is repo-wide and MUST live on `main` so every branch/worktree
inherits it. If you edited it on a feature or research branch, explicitly
flag that it still needs merging to `main`.

