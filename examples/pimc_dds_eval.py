#!/usr/bin/env python3
"""
PIMC-DDS full evaluation, chunked for multi-core wall-clock.

worker mode (one chunk of paired hands, saved as .npz):
    python pimc_dds_eval.py worker --agent {v3|dds} --seed S --n N --out F
merge mode (combine chunks, print per-agent stats + per-hand DDS-v3 gap):
    python pimc_dds_eval.py merge --seed-base S --n-total N --dir D

Chunks partition the seed range [seed_base, seed_base + n_total); every
seed is one paired hand (agent run + rule-based baseline run on the
identical deal), so concatenating chunk arrays in seed order exactly
reproduces a single evaluate_paired(num_hands=n_total, seed=seed_base).
v3 and dds runs share seeds -> their per-hand diffs pair 1:1.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np


def worker(agent_name, seed, n, out):
    from play_eval import evaluate_paired
    if agent_name == 'v3':
        from fortyfives_pimc import PIMCAgent
        agent = PIMCAgent(n_worlds=20)
    else:
        from fortyfives_pimc_dds import PIMCDDSAgent
        agent = PIMCDDSAgent(n_worlds=20)
    r = evaluate_paired(agent, num_hands=n, seed=seed,
                        name=f'{agent_name}-{seed}', silent=True)
    assert len(r.diff) == n, f'timeouts in chunk {agent_name}-{seed}'
    np.savez(out, diff=r.diff, points=r.points, tricks=r.tricks,
             seed=seed, n=n)
    print(f'chunk {agent_name} seed={seed} n={n}: '
          f'avg_diff {r.diff.mean():+.3f}')


def _load(dir_, agent_name, seed_base, n_total):
    chunks = []
    for f in sorted(glob.glob(os.path.join(
            dir_, f'{agent_name}_{seed_base}_*.npz'))):
        z = np.load(f)
        chunks.append((int(z['seed']), z['diff']))
    chunks.sort()
    diff = np.concatenate([d for _, d in chunks])
    assert len(diff) == n_total, \
        f'{agent_name}: got {len(diff)} hands, want {n_total}'
    return diff


def merge(seed_base, n_total, dir_):
    out = {}
    for name in ('v3', 'dds'):
        d = _load(dir_, name, seed_base, n_total)
        m, se = d.mean(), d.std(ddof=1) / len(d) ** 0.5
        print(f'pimc-{name:3s} seed_base={seed_base} n={len(d)}: '
              f'avg_diff {m:+.3f} (95% CI {m - 1.96 * se:+.3f} to '
              f'{m + 1.96 * se:+.3f}) beats-base '
              f'{(d > 0).mean() * 100:.1f}%')
        out[name] = d
    g = out['dds'] - out['v3']
    m, se = g.mean(), g.std(ddof=1) / len(g) ** 0.5
    print(f'per-hand gap dds - v3: {m:+.3f} '
          f'(95% CI {m - 1.96 * se:+.3f} to {m + 1.96 * se:+.3f})')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='mode', required=True)
    w = sub.add_parser('worker')
    w.add_argument('--agent', choices=['v3', 'dds'], required=True)
    w.add_argument('--seed', type=int, required=True)
    w.add_argument('--n', type=int, required=True)
    w.add_argument('--out', required=True)
    m = sub.add_parser('merge')
    m.add_argument('--seed-base', type=int, required=True)
    m.add_argument('--n-total', type=int, required=True)
    m.add_argument('--dir', required=True)
    a = p.parse_args()
    if a.mode == 'worker':
        worker(a.agent, a.seed, a.n, a.out)
    else:
        merge(a.seed_base, a.n_total, a.dir)
