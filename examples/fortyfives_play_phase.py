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
from play_eval import evaluate as eval_play


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

    # Reward shaping
    trick_reward: float = 0.15     # per trick won (+) or lost (-)
    trump5_bonus: float = 0.20     # 5 of trump in completed trick
    trumpJ_bonus: float = 0.15     # J of trump
    trumpA_bonus: float = 0.05     # A of trump
    hand_reward_weight: float = 0.5  # scale on final payoff added to last trick

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


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, agent, rule_agent, cfg):
    """
    Play one hand. Returns transitions for player 0's phase-4 actions only.
    Each transition: dict with obs, action, reward, next_obs, legal_actions, done.

    Hand termination: phase 4→1 transition (same logic as play_eval._run_hand).
    is_hand_over() resets within the same env.step() call as the 5th trick and
    cannot be observed here; is_over() requires 125 points across many hands.
    """
    state, player_id = env.reset()
    init_points = env.game.points.get(0, 0) if env.game.points else 0
    prev_tricks_won = list(env.game.tricks_won)
    transitions = []
    pending = None   # (obs, action) awaiting trick resolution
    in_play = False
    step = 0

    while step < 500:
        step += 1
        prev_phase = state['raw_obs']['phase']
        if prev_phase == 4:
            in_play = True

        if prev_phase == 4 and player_id == 0:
            action = agent.step(state)
            pending = (state['obs'], action)
        else:
            action = rule_agent.step(state)

        next_state, next_player_id = env.step(action)
        curr_phase = env.game.phase

        # Detect trick completion (tricks 1-4; trick 5 resets within env.step)
        curr_tricks = list(env.game.tricks_won)
        if prev_phase == 4 and sum(curr_tricks) > sum(prev_tricks_won):
            if pending is not None:
                r = compute_trick_reward(prev_tricks_won, env.game, cfg)
                transitions.append({
                    'obs': pending[0],
                    'action': pending[1],
                    'reward': r,
                    'next_obs': next_state['obs'],
                    'legal_actions': list(next_state['legal_actions'].keys()),
                    'done': False,
                })
                pending = None
            prev_tricks_won = curr_tricks

        # Hand ended: play phase → new auction (or game truly over at 125 pts)
        if (in_play and prev_phase == 4 and curr_phase == 1) or env.game.is_over():
            points_now = env.game.points.get(0, 0) if env.game.points else 0
            hand_reward = (points_now - init_points) / 125 * cfg.hand_reward_weight
            if pending is not None:
                transitions.append({
                    'obs': pending[0],
                    'action': pending[1],
                    'reward': hand_reward,
                    'next_obs': next_state['obs'],
                    'legal_actions': list(next_state['legal_actions'].keys()),
                    'done': True,
                })
            elif transitions:
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
    print(f"Rewards:    trick={cfg.trick_reward}, 5={cfg.trump5_bonus}, J={cfg.trumpJ_bonus}, A={cfg.trumpA_bonus}, hand={cfg.hand_reward_weight}")
    print()

    device = get_device()
    set_seed(cfg.seed)

    env = rlcard.make('fortyfives', config={'seed': cfg.seed})

    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape,
        mlp_layers=cfg.mlp_layers,
        device=device,
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
    parser.add_argument('--trick_reward',     type=float, default=0.15)
    parser.add_argument('--trump5_bonus',     type=float, default=0.20)
    parser.add_argument('--trumpJ_bonus',     type=float, default=0.15)
    parser.add_argument('--trumpA_bonus',     type=float, default=0.05)
    parser.add_argument('--hand_reward_weight', type=float, default=0.5)
    args = parser.parse_args()

    cfg = TrainConfig(**vars(args))
    train(cfg)
