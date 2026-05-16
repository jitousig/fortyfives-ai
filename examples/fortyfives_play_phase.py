#!/usr/bin/env python3
"""
Play-phase DQN trainer for Fortyfives.

Only trains on card-play decisions (phase 4). Rule-based handles everything
else. Define a TrainConfig to experiment with reward shaping, network size,
and hyperparameters, then evaluate results with play_eval.compare().

Quick start
-----------
# Train with defaults:
python examples/fortyfives_play_phase.py

# Custom config (edit TrainConfig below or subclass):
python examples/fortyfives_play_phase.py --name my_run --trick_reward 0.2

# Evaluate afterwards:
from play_eval import compare, load_model
from fortyfives_rule_based import RuleBasedAgent
compare({
    'my_run': load_model('experiments/my_run/model.pth'),
    'rule':   RuleBasedAgent(num_actions=18),
})
"""

import os
import sys

# Run as `python examples/...` puts examples/ on sys.path[0], so `import
# fortyfives` would resolve to the editable install in the shared venv
# (the SIBLING ../fortyfives repo), not this repo's package. Force this
# repo's root ahead of everything so env-code edits here actually apply.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import csv
from dataclasses import dataclass, field

import numpy as np
import torch
import rlcard
from rlcard.agents import DQNAgent
from rlcard.utils import get_device, set_seed

from rlcard.envs.registration import register, registry
if 'fortyfives' not in registry.env_specs:
    register(
        env_id='fortyfives',
        entry_point='fortyfives.envs.fortyfives_env:FortyfivesEnv',
    )

sys.path.insert(0, os.path.dirname(__file__))
from fortyfives_rule_based import RuleBasedAgent
from play_eval import evaluate_paired as eval_play


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Experiment identity
    name: str = 'play_phase'
    experiments_dir: str = 'experiments'
    seed: int = 42

    # Training loop
    num_episodes: int = 20000
    evaluate_every: int = 500
    eval_hands: int = 100

    # Network
    mlp_layers: list = field(default_factory=lambda: [128, 128, 64])

    # Exploration. rlcard's DQNAgent.step() reads epsilon from a linspace
    # schedule indexed by total_t (NOT self.epsilon). We advance total_t
    # once per fed transition, so decay_steps is in units of transitions
    # (~4-5 per hand). 30k ≈ epsilon floor reached around episode ~6k.
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 30000

    # Reward shaping. The dominant signal is the bid OUTCOME: in 45s the
    # bidding team must reach its bid or get "set" — grabbing tricks is not
    # the objective. So bid make/miss is the largest term, raw point delta a
    # secondary gradient, and per-trick shaping only a small dense aux to
    # help early credit assignment.
    bid_outcome_reward: float = 1.0   # +/- when our team makes/misses its bid
    point_reward_weight: float = 0.02 # scale on NS game-point delta this hand
    trick_reward: float = 0.02     # per trick won (+) or lost (-)
    trump5_bonus: float = 0.03     # 5 of trump in completed trick
    trumpJ_bonus: float = 0.02     # J of trump
    trumpA_bonus: float = 0.01     # A of trump

    @property
    def log_dir(self):
        return os.path.join(self.experiments_dir, self.name)


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_trick_reward(prev_tricks_won, game, cfg):
    """Reward for player 0 when a trick just completed."""
    curr = game.tricks_won
    winner = next(i for i in range(4) if curr[i] > prev_tricks_won[i])
    sign = 1.0 if winner in (0, 2) else -1.0
    reward = sign * cfg.trick_reward

    trump = game.trump_suit
    if trump and game.trick_history:
        for card in game.trick_history[-1]:
            if card is not None and card.suit == trump:
                if card.rank == '5':
                    reward += sign * cfg.trump5_bonus
                elif card.rank == 'J':
                    reward += sign * cfg.trumpJ_bonus
                elif card.rank == 'A':
                    reward += sign * cfg.trumpA_bonus

    return reward


def compute_hand_reward(bid_team, game, init_points, cfg):
    """Terminal reward from NS (team 0/2) perspective.

    Dominant term is the bid outcome: reward our team making its bid and
    setting the opponents; penalize being set / letting them make it. This
    is the actual 45s objective, independent of raw trick count. The point
    delta is a smaller secondary gradient.

    bid_made survives into the next auction (start_new_hand does not reset
    it), but highest_bidder IS reset — so bid_team must be captured during
    the hand and passed in here.
    """
    points_now = game.points.get(0, 0) if game.points else 0
    point_term = (points_now - init_points) * cfg.point_reward_weight

    bid_term = 0.0
    if bid_team is not None:
        made = bool(getattr(game, 'bid_made', False))
        if bid_team == 0:      # our team (NS) bid
            bid_term = cfg.bid_outcome_reward if made else -cfg.bid_outcome_reward
        else:                  # opponents (EW) bid — good for us if they fail
            bid_term = -cfg.bid_outcome_reward if made else cfg.bid_outcome_reward
    return point_term + bid_term


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, agent, rule_agent, cfg):
    """
    Play one hand. The shared policy controls BOTH NS seats (players 0 and
    2); transitions are collected from both. Each transition: dict with obs,
    action, reward, next_obs, legal_actions, done.

    Hand termination: phase 4→1 transition (same logic as play_eval._run_hand).
    is_hand_over() resets within the same env.step() call as the 5th trick and
    cannot be observed here; is_over() requires 125 points across many hands.
    """
    NS_SEATS = (0, 2)
    state, player_id = env.reset()
    init_points = env.game.points.get(0, 0) if env.game.points else 0
    prev_tricks_won = list(env.game.tricks_won)
    transitions = []
    pending = {s: None for s in NS_SEATS}  # per-seat (obs, action) awaiting trick
    in_play = False
    bid_team = None  # captured during the hand; highest_bidder is reset at 4->1
    step = 0

    while step < 500:
        step += 1
        prev_phase = state['raw_obs']['phase']
        if prev_phase == 4:
            in_play = True
        if bid_team is None and env.game.highest_bidder is not None:
            bid_team = env.game.highest_bidder % 2

        if prev_phase == 4 and player_id in NS_SEATS:
            action = agent.step(state)
            pending[player_id] = (state['obs'], action)
        else:
            action = rule_agent.step(state)

        next_state, next_player_id = env.step(action)
        curr_phase = env.game.phase

        # Detect trick completion (tricks 1-4; trick 5 resets within env.step).
        # Both NS seats are on the same partnership, so the team trick reward
        # is identical for each — compute once, credit each pending seat.
        curr_tricks = list(env.game.tricks_won)
        if prev_phase == 4 and sum(curr_tricks) > sum(prev_tricks_won):
            if any(pending[s] is not None for s in NS_SEATS):
                r = compute_trick_reward(prev_tricks_won, env.game, cfg)
                for s in NS_SEATS:
                    if pending[s] is not None:
                        transitions.append({
                            'obs': pending[s][0],
                            'action': pending[s][1],
                            'reward': r,
                            'next_obs': next_state['obs'],
                            'legal_actions': list(next_state['legal_actions'].keys()),
                            'done': False,
                        })
                        pending[s] = None
            prev_tricks_won = curr_tricks

        # Hand ended: play phase → new auction (or game truly over at 125 pts).
        # The 5th trick's plays are still pending here (trick 5 is swallowed in
        # the phase reset); give each still-pending NS seat the terminal reward.
        if (in_play and prev_phase == 4 and curr_phase == 1) or env.game.is_over():
            hand_reward = compute_hand_reward(bid_team, env.game, init_points, cfg)
            flushed = False
            for s in NS_SEATS:
                if pending[s] is not None:
                    transitions.append({
                        'obs': pending[s][0],
                        'action': pending[s][1],
                        'reward': hand_reward,
                        'next_obs': next_state['obs'],
                        'legal_actions': list(next_state['legal_actions'].keys()),
                        'done': True,
                    })
                    pending[s] = None
                    flushed = True
            if not flushed and transitions:
                transitions[-1]['reward'] += hand_reward
                transitions[-1]['done'] = True
            break

        state = next_state
        player_id = next_player_id

    return transitions


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.episodes, self.performances = [], []
        self.csv_path = os.path.join(log_dir, 'performance.csv')
        with open(self.csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=['episode', 'performance']).writeheader()

    def log(self, episode, performance):
        self.episodes.append(episode)
        self.performances.append(performance)
        with open(self.csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=['episode', 'performance']).writerow(
                {'episode': episode, 'performance': performance}
            )

    def save_plot(self, log_dir):
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(self.episodes, self.performances)
            plt.axhline(0, color='gray', linestyle='--')
            plt.title('Play Phase — Performance vs Rule-Based')
            plt.xlabel('Episode')
            plt.ylabel('Avg Payoff')
            path = os.path.join(log_dir, 'learning_curve.png')
            plt.savefig(path)
            plt.close()
            print(f"Learning curve saved to {path}")
        except Exception as e:
            print(f"Plot error: {e}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig):
    print(f"Experiment: {cfg.name}")
    print(f"Log dir:    {cfg.log_dir}")
    print(f"Episodes:   {cfg.num_episodes} | Eval every: {cfg.evaluate_every}")
    print(f"Network:    {cfg.mlp_layers}")
    print(f"Rewards:    bid={cfg.bid_outcome_reward}, point_w={cfg.point_reward_weight}, "
          f"trick={cfg.trick_reward}, 5={cfg.trump5_bonus}, J={cfg.trumpJ_bonus}, A={cfg.trumpA_bonus}")
    print()

    device = get_device()
    set_seed(cfg.seed)

    env = rlcard.make('fortyfives', config={'seed': cfg.seed})

    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape,
        mlp_layers=cfg.mlp_layers,
        device=device,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay_steps=cfg.epsilon_decay_steps,
    )
    rule_agent = RuleBasedAgent(num_actions=env.num_actions)
    logger = Logger(cfg.log_dir)
    total_transitions = 0

    for episode in range(cfg.num_episodes):
        try:
            transitions = run_episode(env, agent, rule_agent, cfg)

            for t in transitions:
                agent.feed_memory(
                    t['obs'], t['action'], t['reward'],
                    t['next_obs'], t['legal_actions'], t['done'],
                )
                # Advance the epsilon schedule. feed_memory() alone never
                # touches total_t, so without this the behavior policy
                # stays frozen at epsilon_start (fully random) forever.
                agent.total_t += 1
            total_transitions += len(transitions)

            try:
                loss = agent.train() if total_transitions >= 32 else None
            except ValueError:
                loss = None

            if episode % 100 == 0:
                loss_str = f"{loss:.5f}" if loss is not None else "n/a"
                print(f"Ep {episode:>6}/{cfg.num_episodes} | "
                      f"transitions: {len(transitions)} | loss: {loss_str}")

            if episode % cfg.evaluate_every == 0:
                result = eval_play(agent, num_hands=cfg.eval_hands,
                                   seed=cfg.seed + 10000, name=cfg.name, silent=True)
                logger.log(episode, result.avg_payoff)
                lo, hi = result.ci95
                print(f"  --> payoff {result.avg_payoff:+.4f} "
                      f"(CI {lo:+.4f}..{hi:+.4f}) | "
                      f"win {result.win_rate*100:.1f}% | "
                      f"tricks {result.avg_tricks:.2f}")

        except Exception as e:
            import traceback
            print(f"Error in episode {episode}: {e}")
            traceback.print_exc()

    save_path = os.path.join(cfg.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print(f"\nModel saved to {save_path}")
    logger.save_plot(cfg.log_dir)
    return agent


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Play-Phase DQN Trainer")
    parser.add_argument('--name',             type=str,   default='play_phase')
    parser.add_argument('--experiments_dir',  type=str,   default='experiments')
    parser.add_argument('--seed',             type=int,   default=42)
    parser.add_argument('--num_episodes',     type=int,   default=20000)
    parser.add_argument('--evaluate_every',   type=int,   default=500)
    parser.add_argument('--eval_hands',       type=int,   default=100)
    parser.add_argument('--mlp_layers',       type=int,   nargs='+', default=[128, 128, 64])
    parser.add_argument('--epsilon_start',    type=float, default=1.0)
    parser.add_argument('--epsilon_end',      type=float, default=0.05)
    parser.add_argument('--epsilon_decay_steps', type=int, default=30000)
    parser.add_argument('--bid_outcome_reward',   type=float, default=1.0)
    parser.add_argument('--point_reward_weight',  type=float, default=0.02)
    parser.add_argument('--trick_reward',     type=float, default=0.02)
    parser.add_argument('--trump5_bonus',     type=float, default=0.03)
    parser.add_argument('--trumpJ_bonus',     type=float, default=0.02)
    parser.add_argument('--trumpA_bonus',     type=float, default=0.01)
    args = parser.parse_args()

    cfg = TrainConfig(**vars(args))
    train(cfg)
