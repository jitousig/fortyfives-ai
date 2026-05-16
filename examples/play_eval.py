"""
Plug-and-play evaluation for Fortyfives play-phase agents.

Any object with a step(state) -> action method works as a play agent.
The rule-based agent handles bidding, trump declaration, and discard for
all players. Only phase-4 card-play decisions differ between agents.

Usage
-----
from play_eval import evaluate, compare, load_model

# Evaluate one agent
result = evaluate(agent, num_hands=300)
print(result)

# Compare several agents on identical hands
from fortyfives_rule_based import RuleBasedAgent
compare(
    {'dqn_v1': load_model('experiments/v1/model.pth'),
     'dqn_v2': load_model('experiments/v2/model.pth'),
     'rule':   RuleBasedAgent(num_actions=18)},
    num_hands=500,
)
"""

import os
import sys
import numpy as np
import torch

import rlcard
from rlcard.envs.registration import register, registry

# Register env on import so callers don't have to
if 'fortyfives' not in registry.env_specs:
    register(
        env_id='fortyfives',
        entry_point='fortyfives.envs.fortyfives_env:FortyfivesEnv',
    )

sys.path.insert(0, os.path.dirname(__file__))
from fortyfives_rule_based import RuleBasedAgent


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_model(path):
    """Load a saved play-phase agent from disk."""
    return torch.load(path, map_location='cpu', weights_only=False)


def greedy(agent):
    """Context manager: set DQN epsilon to 0 for evaluation, then restore."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        old = getattr(agent, 'epsilon', None)
        if old is not None:
            agent.epsilon = 0.0
        try:
            yield agent
        finally:
            if old is not None:
                agent.epsilon = old

    return _ctx()


# ---------------------------------------------------------------------------
# Core hand runner
# ---------------------------------------------------------------------------

def _run_hand(env, play_agent, seed):
    """
    Play one hand with play_agent making phase-4 decisions for player 0.
    Rule-based handles everything else.

    The game is multi-hand (runs to 125 pts), so we terminate at the phase
    transition 4→1 (play→new auction) which marks the end of one hand.

    Returns (points_change, team_tricks) or (None, None) on timeout.
    points_change: raw game-point delta for team 0/2 this hand (e.g. +30, -20).
    team_tricks:   tricks won by players 0+2, excluding the final trick
                   (which is swallowed by the phase reset — see note in docs).
    """
    env.seed(seed)
    state, player_id = env.reset()
    rule_agent = RuleBasedAgent(num_actions=env.num_actions)

    init_points = env.game.points.get(0, 0) if env.game.points else 0
    prev_tricks = list(env.game.tricks_won)
    hand_tricks = [0, 0, 0, 0]   # accumulate trick wins for THIS hand
    in_play = False               # have we entered phase 4 yet?
    step = 0

    while step < 500:
        step += 1
        prev_phase = state['raw_obs']['phase']
        if prev_phase == 4:
            in_play = True

        if prev_phase == 4 and player_id == 0:
            action = play_agent.step(state)
        else:
            action = rule_agent.step(state)

        next_state, next_player_id = env.step(action)
        curr_phase = env.game.phase
        curr_tricks = list(env.game.tricks_won)

        # Track trick completions during play phase (tricks 1-4 are visible;
        # trick 5 is swallowed in the phase reset so hand_tricks sums to 4)
        if prev_phase == 4 and sum(curr_tricks) > sum(prev_tricks):
            for i in range(4):
                hand_tricks[i] += curr_tricks[i] - prev_tricks[i]
            prev_tricks = curr_tricks

        # Hand ended: play phase → new auction (or game over)
        if (in_play and prev_phase == 4 and curr_phase == 1) or env.game.is_over():
            points_now = env.game.points.get(0, 0) if env.game.points else 0
            return points_now - init_points, hand_tricks[0] + hand_tricks[2]

        state = next_state
        player_id = next_player_id

    return None, None   # timed out


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------

class EvalResult:
    """
    points: raw game-point delta per hand (e.g. +30 for making a 30 bid,
            -20 for failing a 20 bid). Range is roughly [-30, +60].
    tricks: tricks won by team 0/2 per hand (0-4; last trick not counted).
    """
    def __init__(self, name, points, tricks):
        self.name = name
        self.points = np.array(points, dtype=float)
        self.tricks = np.array(tricks, dtype=float)

    @property
    def num_hands(self):
        return len(self.points)

    @property
    def avg_points(self):
        return self.points.mean()

    @property
    def avg_tricks(self):
        return self.tricks.mean()

    @property
    def win_rate(self):
        return (self.points > 0).mean()

    @property
    def ci95(self):
        se = self.points.std() / self.num_hands ** 0.5
        return self.avg_points - 1.96 * se, self.avg_points + 1.96 * se

    # Keep avg_payoff as alias so training code still works
    @property
    def avg_payoff(self):
        return self.avg_points

    def __str__(self):
        lo, hi = self.ci95
        return (
            f"Agent: {self.name}\n"
            f"  Hands:      {self.num_hands}\n"
            f"  Avg points: {self.avg_points:+.2f}  (95% CI {lo:+.2f} to {hi:+.2f})\n"
            f"  Win rate:   {self.win_rate * 100:.1f}%\n"
            f"  Avg tricks: {self.avg_tricks:.2f} / ~4 (last trick not counted)"
        )


# ---------------------------------------------------------------------------
# evaluate — standalone (no pairing)
# ---------------------------------------------------------------------------

def evaluate(play_agent, num_hands=200, seed=0, name='agent', silent=False):
    """
    Evaluate a play-phase agent over num_hands independent hands.

    Cards are random (no pairing with a baseline). Use compare() for a
    fair head-to-head where luck is controlled.
    """
    env = rlcard.make('fortyfives')
    payoffs, tricks, timeouts = [], [], 0

    with greedy(play_agent):
        for i in range(num_hands):
            p, t = _run_hand(env, play_agent, seed + i)
            if p is not None:
                payoffs.append(p)
                tricks.append(t)
            else:
                timeouts += 1

    result = EvalResult(name, payoffs, tricks)
    if not silent:
        print(result)
        if timeouts:
            print(f"  (Timeouts: {timeouts}/{num_hands})")
    return result


# ---------------------------------------------------------------------------
# compare — paired evaluation on identical hands
# ---------------------------------------------------------------------------

def compare(agents, num_hands=300, seed=0, baseline_name=None):
    """
    Compare multiple play-phase agents on the same hands.

    agents: dict of {name: agent}
    baseline_name: key of the agent to use as baseline for diff columns.
                   Defaults to the first key.

    Prints a side-by-side table and returns a dict of {name: EvalResult}.
    """
    names = list(agents.keys())
    if baseline_name is None:
        baseline_name = names[0]

    env = {n: rlcard.make('fortyfives') for n in names}
    payoffs = {n: [] for n in names}
    tricks  = {n: [] for n in names}
    timeouts = {n: 0 for n in names}

    for i in range(num_hands):
        s = seed + i
        hand_results = {}
        for n, agent in agents.items():
            with greedy(agent):
                p, t = _run_hand(env[n], agent, s)
            hand_results[n] = (p, t)

        # Only record hands where all agents completed
        if all(p is not None for p, _ in hand_results.values()):
            for n, (p, t) in hand_results.items():
                payoffs[n].append(p)
                tricks[n].append(t)
        else:
            for n, (p, _) in hand_results.items():
                if p is None:
                    timeouts[n] += 1

    results = {n: EvalResult(n, payoffs[n], tricks[n]) for n in names}
    _print_comparison(results, baseline_name, timeouts, num_hands)
    return results


def _print_comparison(results, baseline_name, timeouts, num_hands):
    names = list(results.keys())
    base = results[baseline_name]

    col = 14
    name_col = max(len(n) for n in names) + 2

    # Header
    header = f"{'Agent':<{name_col}}  {'Payoff':>{col}}  {'Win%':>{col}}  {'Tricks':>{col}}  {'vs ' + baseline_name:>{col}}"
    print()
    print(header)
    print('-' * len(header))

    for n in names:
        r = results[n]
        lo, hi = r.ci95
        diff = r.avg_payoff - base.avg_payoff if n != baseline_name else 0.0
        diff_str = f"{diff:+.4f}" if n != baseline_name else "baseline"
        print(
            f"{n:<{name_col}}  "
            f"{r.avg_payoff:>+{col}.4f}  "
            f"{r.win_rate*100:>{col}.1f}  "
            f"{r.avg_tricks:>{col}.3f}  "
            f"{diff_str:>{col}}"
        )
        print(f"{'':>{name_col}}  95% CI [{lo:+.4f}, {hi:+.4f}]")

    if any(timeouts.values()):
        print()
        for n, t in timeouts.items():
            if t:
                print(f"  Timeouts — {n}: {t}/{num_hands}")
    print()
