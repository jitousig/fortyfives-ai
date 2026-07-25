#!/usr/bin/env python3
"""
Paired Elo ladder for Fortyfives play-phase agents.

Every match is PAIRED on identical deals using play_eval.evaluate_paired:
agent A and agent B each play the same hands (A or B controls both NS
seats; EW + all non-play phases are deterministic rule-based, same as the
trusted canary setup), so deal luck cancels and A's score vs B is the
fraction of hands A out-points B (ties = 0.5).

The ladder is ANCHORED: rule-based is pegged at Elo `--anchor_elo`
(default 1000) so ratings are absolute and comparable across runs, not
free-floating. random-legal is included as a second, lower reference
point. Each participant's trusted yardstick (avg_diff vs rule-based with
its paired 95% CI) is printed alongside its Elo so the scalar we trust
always travels next to the Elo number.

Usage
-----
python examples/elo_ladder.py --pool experiments/sp_v1/pool
python examples/elo_ladder.py --pool experiments/sp_v1/pool --num_hands 400
"""

import os
import sys

# See CLAUDE.md: force this repo's package ahead of the shared-venv
# editable install (sibling worktree) before importing fortyfives.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import glob
import re

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fortyfives_rule_based import RuleBasedAgent
from play_eval import evaluate_paired, load_model

try:
    from rlcard.agents import RandomAgent
except Exception:
    RandomAgent = None


# ---------------------------------------------------------------------------
# Participant discovery
# ---------------------------------------------------------------------------

def _snap_episode(path):
    """Sort key: episode number parsed from snap_ep<N>.pth, else by name."""
    m = re.search(r'ep(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_participants(pool_dir, num_actions):
    """Frozen self snapshots (sorted by episode) + rule/random anchors."""
    participants = {}

    snap_paths = sorted(glob.glob(os.path.join(pool_dir, 'snap_ep*.pth')),
                        key=_snap_episode)
    for p in snap_paths:
        name = os.path.splitext(os.path.basename(p))[0]  # snap_ep1000
        participants[name] = load_model(p)

    # Anchors. rule-based is the pegged reference; random is the floor.
    participants['rule'] = RuleBasedAgent(num_actions=num_actions)
    if RandomAgent is not None:
        participants['random'] = RandomAgent(num_actions=num_actions)

    return participants


# ---------------------------------------------------------------------------
# Round-robin (all matches paired on the SAME deals)
# ---------------------------------------------------------------------------

def round_robin(participants, num_hands, seed):
    """
    Returns:
      scores:  {(a,b): fraction of hands a out-points b}  (a != b)
      games:   {(a,b): hands actually scored}
      vs_rule: {name: PairedResult of name vs rule-based}
    """
    names = list(participants.keys())
    scores, games, vs_rule = {}, {}, {}

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i:
                continue
            res = evaluate_paired(
                participants[a], baseline=participants[b],
                num_hands=num_hands, seed=seed, name=a, silent=True,
            )
            d = res.diff
            n = len(d)
            if n == 0:
                continue
            a_score = float((d > 0).sum() + 0.5 * (d == 0).sum()) / n
            scores[(a, b)] = a_score
            scores[(b, a)] = 1.0 - a_score
            games[(a, b)] = games[(b, a)] = n
            if b == 'rule':
                vs_rule[a] = res

    # name-vs-rule for the rule row itself (the canary): rule vs rule.
    if 'rule' in participants:
        vs_rule['rule'] = evaluate_paired(
            participants['rule'], baseline=participants['rule'],
            num_hands=num_hands, seed=seed, name='rule', silent=True,
        )
    return scores, games, vs_rule


# ---------------------------------------------------------------------------
# Elo fit (anchored to rule-based)
# ---------------------------------------------------------------------------

def fit_elo(names, scores, games, anchor_name, anchor_elo,
            iters=2000, k0=8.0):
    """Iterative Elo fit on paired pairwise scores, then shift so
    anchor_name == anchor_elo. K decays over iterations for convergence."""
    R = {n: float(anchor_elo) for n in names}
    for it in range(iters):
        k = k0 * (1.0 - it / iters) + 0.05
        grad = {n: 0.0 for n in names}
        for n in names:
            for m in names:
                if n == m or (n, m) not in scores:
                    continue
                g = games[(n, m)]
                expected = 1.0 / (1.0 + 10 ** ((R[m] - R[n]) / 400.0))
                grad[n] += g * (scores[(n, m)] - expected)
        for n in names:
            R[n] += k * grad[n] / max(1, len(names) - 1)

    if anchor_name in R:
        shift = anchor_elo - R[anchor_name]
        for n in R:
            R[n] += shift
    return R


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_ladder(R, vs_rule, anchor_name):
    order = sorted(R, key=lambda n: R[n], reverse=True)
    print()
    print(f"{'Rank':<5} {'Agent':<16} {'Elo':>7}   "
          f"{'avg_diff vs rule':>16}   {'95% CI':>22}")
    print('-' * 74)
    for rank, n in enumerate(order, 1):
        tag = '  <- anchor' if n == anchor_name else ''
        r = vs_rule.get(n)
        if r is not None:
            lo, hi = r.ci95
            diff_s = f"{r.avg_diff:+.4f}"
            ci_s = f"[{lo:+.3f}, {hi:+.3f}]"
        else:
            diff_s, ci_s = 'n/a', 'n/a'
        print(f"{rank:<5} {n:<16} {R[n]:>7.1f}   "
              f"{diff_s:>16}   {ci_s:>22}{tag}")
    print()
    canary = vs_rule.get('rule')
    if canary is not None:
        ok = abs(canary.avg_diff) < 1e-9 and canary.win_rate == 0.0
        print(f"Canary (rule vs rule): avg_diff {canary.avg_diff:+.4f}, "
              f"win {canary.win_rate*100:.1f}%  "
              f"-> {'OK (paired eval trustworthy)' if ok else 'FAIL — ladder NOT trustworthy'}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("Fortyfives Paired Elo Ladder")
    p.add_argument('--pool', type=str, required=True,
                   help='Pool directory containing snap_ep*.pth snapshots')
    p.add_argument('--num_hands', type=int, default=300,
                   help='Paired hands per match (same deals for every pair)')
    p.add_argument('--seed', type=int, default=99,
                   help='Eval deal seed (shared by all matches -> fully paired)')
    p.add_argument('--anchor_elo', type=float, default=1000.0,
                   help='Elo pegged to the rule-based anchor')
    p.add_argument('--num_actions', type=int, default=18)
    args = p.parse_args()

    if not os.path.isdir(args.pool):
        sys.exit(f"Pool dir not found: {args.pool}")

    # Seed the global RNG so the random-legal anchor is reproducible
    # (play_eval.greedy() is a no-op for RandomAgent; it uses np.random).
    np.random.seed(args.seed)

    participants = load_participants(args.pool, args.num_actions)
    snaps = [n for n in participants if n.startswith('snap_')]
    print(f"Participants: {len(snaps)} snapshot(s) + anchors "
          f"({', '.join(n for n in participants if not n.startswith('snap_'))})")
    print(f"Matches paired on {args.num_hands} hands, seed {args.seed}, "
          f"rule anchored at Elo {args.anchor_elo:.0f}")

    scores, games, vs_rule = round_robin(
        participants, args.num_hands, args.seed)
    R = fit_elo(list(participants.keys()), scores, games,
                anchor_name='rule', anchor_elo=args.anchor_elo)
    print_ladder(R, vs_rule, anchor_name='rule')


if __name__ == '__main__':
    main()
