#!/usr/bin/env python3
"""
Stage A self-play trainer for Fortyfives (play phase only).

Builds on the PROVEN play-phase pipeline (fortyfives_play_phase.py):
the same trusted bid-outcome reward and the same phase-4-only episode
structure. The ONLY change vs the proven trainer is who sits in the
opponent (EW) seats during the play phase:

  - NS seats (0, 2):  the live learning policy (shared), collects transitions
  - EW seats (1, 3):  a FROZEN opponent sampled per-episode from a pool
                       (recent self snapshots + rule-based / random anchors)
  - every non-play phase (auction/declaration/discard), ALL seats:
                       deterministic rule-based — unchanged from the proven
                       pipeline, so play_eval's rule-vs-rule == 0.0000
                       canary stays valid for this trainer.

The reward and Logger are imported from fortyfives_play_phase so there is
exactly one source of truth for the trusted reward — self-play must not
silently drift to a different (untrusted) reward.

Quick start
-----------
python examples/fortyfives_selfplay.py --name sp_v1 --num_episodes 20000

Evaluate / rank afterwards:
python examples/elo_ladder.py --pool experiments/sp_v1/pool
"""

import os
import sys

# Run as `python examples/...` puts examples/ on sys.path[0], so `import
# fortyfives` would resolve to the editable install in the shared venv
# (a SIBLING worktree's old code), not this repo's package. Force this
# repo's root ahead of everything so env-code edits here actually apply.
# Removing this shim silently trains the wrong env — see CLAUDE.md.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import rlcard
from rlcard.agents import DQNAgent, RandomAgent
from rlcard.utils import get_device, set_seed

from rlcard.envs.registration import register, registry
if 'fortyfives' not in registry.env_specs:
    register(
        env_id='fortyfives',
        entry_point='fortyfives.envs.fortyfives_env:FortyfivesEnv',
    )

sys.path.insert(0, os.path.dirname(__file__))
from fortyfives_rule_based import RuleBasedAgent
from fortyfives_play_phase import compute_trick_reward, compute_hand_reward, Logger
from play_eval import evaluate_paired as eval_play, greedy


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SelfPlayConfig:
    name: str = 'selfplay'
    experiments_dir: str = 'experiments'
    seed: int = 42

    num_episodes: int = 20000
    evaluate_every: int = 500
    eval_hands: int = 100

    mlp_layers: list = field(default_factory=lambda: [128, 128, 64])

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 30000

    # Trusted reward (mirrors fortyfives_play_phase defaults; the reward
    # FUNCTIONS themselves are imported, only the knobs live here).
    bid_outcome_reward: float = 1.0
    point_reward_weight: float = 0.02
    trick_reward: float = 0.02
    trump5_bonus: float = 0.03
    trumpJ_bonus: float = 0.02
    trumpA_bonus: float = 0.01

    # Self-play opponent pool
    snapshot_every: int = 1000      # episodes between frozen self snapshots
    pool_recency_bias: float = 2.0  # >1 weights recent snapshots more
    p_rule_anchor: float = 0.20     # episodes vs deterministic rule-based
    p_random_anchor: float = 0.10   # episodes vs random-legal
    # remaining prob mass -> sampled frozen self snapshot (rule-based until
    # the first snapshot exists, i.e. a warm start identical to the proven
    # play-phase pipeline)

    @property
    def log_dir(self):
        return os.path.join(self.experiments_dir, self.name)

    @property
    def pool_dir(self):
        return os.path.join(self.log_dir, 'pool')


# ---------------------------------------------------------------------------
# Frozen snapshots & opponent pool
# ---------------------------------------------------------------------------

def make_frozen_snapshot(agent, env, cfg, device):
    """A greedy, no-learning copy of the current policy.

    Only the q-network weights are copied (not the replay buffer): a fresh
    DQNAgent with an all-zero epsilon schedule, so .step() is deterministic
    argmax over legal actions regardless of total_t.
    """
    frozen = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape,
        mlp_layers=cfg.mlp_layers,
        device=device,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
    )
    frozen.q_estimator.qnet.load_state_dict(
        copy.deepcopy(agent.q_estimator.qnet.state_dict())
    )
    frozen.q_estimator.qnet.eval()
    frozen.epsilons = np.zeros_like(frozen.epsilons)
    return frozen


class OpponentPool:
    """Frozen self snapshots plus two fixed anchors (rule-based, random).

    Per episode, sample_opponent() returns the EW play agent for that
    episode. Until the first snapshot exists it returns rule-based, which
    makes the early-training distribution identical to the proven
    play-phase pipeline (a deliberate warm start).
    """

    def __init__(self, cfg, num_actions, rng):
        self.cfg = cfg
        self.rng = rng
        self.snapshots = []  # list of frozen DQNAgents, oldest -> newest
        self._rule = RuleBasedAgent(num_actions=num_actions)
        self._random = RandomAgent(num_actions=num_actions)
        os.makedirs(cfg.pool_dir, exist_ok=True)

    def add(self, frozen, episode):
        self.snapshots.append(frozen)
        path = os.path.join(self.cfg.pool_dir, f'snap_ep{episode}.pth')
        torch.save(frozen, path)
        return path

    def sample_opponent(self):
        if not self.snapshots:
            return self._rule  # warm start == proven pipeline
        u = self.rng.random()
        if u < self.cfg.p_rule_anchor:
            return self._rule
        if u < self.cfg.p_rule_anchor + self.cfg.p_random_anchor:
            return self._random
        # recency-biased pick among frozen self snapshots
        n = len(self.snapshots)
        w = np.array([(i + 1) ** self.cfg.pool_recency_bias for i in range(n)],
                     dtype=float)
        w /= w.sum()
        return self.snapshots[self.rng.choice(n, p=w)]


def _opponent_act(agent, state):
    """Greedy action for a frozen/anchor opponent (never learns here)."""
    if hasattr(agent, 'eval_step'):
        out = agent.eval_step(state)
        return out[0] if isinstance(out, tuple) else out
    return agent.step(state)


# ---------------------------------------------------------------------------
# Episode runner (phase-4-only; mirrors fortyfives_play_phase.run_episode,
# differing ONLY in the EW play branch: frozen opponent instead of rule).
# ---------------------------------------------------------------------------

def run_episode(env, agent, rule_agent, opponent, cfg):
    NS_SEATS = (0, 2)
    state, player_id = env.reset()
    init_points = env.game.points.get(0, 0) if env.game.points else 0
    prev_tricks_won = list(env.game.tricks_won)
    transitions = []
    pending = {s: None for s in NS_SEATS}
    in_play = False
    bid_team = None
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
        elif prev_phase == 4:
            # EW play seat -> frozen opponent (the only deviation from the
            # proven pipeline; non-play phases stay rule-based below).
            action = _opponent_act(opponent, state)
        else:
            action = rule_agent.step(state)

        next_state, next_player_id = env.step(action)
        curr_phase = env.game.phase

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
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: SelfPlayConfig):
    print(f"Experiment: {cfg.name}")
    print(f"Log dir:    {cfg.log_dir}")
    print(f"Episodes:   {cfg.num_episodes} | Eval every: {cfg.evaluate_every}")
    print(f"Network:    {cfg.mlp_layers}")
    print(f"Pool:       snapshot_every={cfg.snapshot_every}, "
          f"recency_bias={cfg.pool_recency_bias}, "
          f"p_rule={cfg.p_rule_anchor}, p_random={cfg.p_random_anchor}")
    print()

    device = get_device()
    set_seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)

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
    pool = OpponentPool(cfg, env.num_actions, rng)
    logger = Logger(cfg.log_dir)
    total_transitions = 0

    for episode in range(cfg.num_episodes):
        try:
            opponent = pool.sample_opponent()
            transitions = run_episode(env, agent, rule_agent, opponent, cfg)

            for t in transitions:
                agent.feed_memory(
                    t['obs'], t['action'], t['reward'],
                    t['next_obs'], t['legal_actions'], t['done'],
                )
                # Advance the epsilon schedule. feed_memory() never touches
                # total_t, so without this the behavior policy stays frozen
                # at epsilon_start (fully random) forever.
                agent.total_t += 1
            total_transitions += len(transitions)

            try:
                loss = agent.train() if total_transitions >= 32 else None
            except ValueError:
                loss = None

            if episode % 100 == 0:
                loss_str = f"{loss:.5f}" if loss is not None else "n/a"
                print(f"Ep {episode:>6}/{cfg.num_episodes} | "
                      f"pool: {len(pool.snapshots)} | "
                      f"transitions: {len(transitions)} | loss: {loss_str}")

            if episode > 0 and episode % cfg.snapshot_every == 0:
                frozen = make_frozen_snapshot(agent, env, cfg, device)
                path = pool.add(frozen, episode)
                print(f"  [pool] snapshot ep{episode} -> {path} "
                      f"(pool size {len(pool.snapshots)})")

            if episode % cfg.evaluate_every == 0:
                result = eval_play(agent, num_hands=cfg.eval_hands,
                                   seed=cfg.seed + 10000, name=cfg.name,
                                   silent=True)
                logger.log(episode, result.avg_payoff)
                lo, hi = result.ci95
                print(f"  --> diff {result.avg_payoff:+.4f} "
                      f"(CI {lo:+.4f}..{hi:+.4f}) | "
                      f"beats rule {result.win_rate*100:.1f}% | "
                      f"tricks {result.avg_tricks:.2f}")

        except Exception as e:
            import traceback
            print(f"Error in episode {episode}: {e}")
            traceback.print_exc()

    save_path = os.path.join(cfg.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print(f"\nModel saved to {save_path}")
    # Always snapshot the final policy into the pool so the Elo ladder can
    # rank the end state even if it didn't land on a snapshot boundary.
    final_frozen = make_frozen_snapshot(agent, env, cfg, device)
    pool.add(final_frozen, cfg.num_episodes)
    logger.save_plot(cfg.log_dir)
    return agent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser("Fortyfives Stage-A Self-Play Trainer")
    p.add_argument('--name',            type=str,   default='selfplay')
    p.add_argument('--experiments_dir', type=str,   default='experiments')
    p.add_argument('--seed',            type=int,   default=42)
    p.add_argument('--num_episodes',    type=int,   default=20000)
    p.add_argument('--evaluate_every',  type=int,   default=500)
    p.add_argument('--eval_hands',      type=int,   default=100)
    p.add_argument('--mlp_layers',      type=int,   nargs='+', default=[128, 128, 64])
    p.add_argument('--epsilon_start',   type=float, default=1.0)
    p.add_argument('--epsilon_end',     type=float, default=0.05)
    p.add_argument('--epsilon_decay_steps', type=int, default=30000)
    p.add_argument('--bid_outcome_reward',  type=float, default=1.0)
    p.add_argument('--point_reward_weight', type=float, default=0.02)
    p.add_argument('--trick_reward',    type=float, default=0.02)
    p.add_argument('--trump5_bonus',    type=float, default=0.03)
    p.add_argument('--trumpJ_bonus',    type=float, default=0.02)
    p.add_argument('--trumpA_bonus',    type=float, default=0.01)
    p.add_argument('--snapshot_every',  type=int,   default=1000)
    p.add_argument('--pool_recency_bias', type=float, default=2.0)
    p.add_argument('--p_rule_anchor',   type=float, default=0.20)
    p.add_argument('--p_random_anchor', type=float, default=0.10)
    args = p.parse_args()

    cfg = SelfPlayConfig(**vars(args))
    train(cfg)
