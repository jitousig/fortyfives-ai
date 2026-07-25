#!/usr/bin/env python3
"""
Play-phase decision-gate measurement (RESEARCH.md § Active work).

On identical deals (same seeds, paired through evaluate_paired):
  - PIMC v3            — the incumbent fair agent
  - oracle-br          — perfect-info best response vs the rule-based
                         table: the TRUE ceiling of the yardstick
  - oracle-minimax     — classic paranoid double-dummy (reported for
                         context; NOT an upper bound vs this table)

Reports each agent's avg_diff vs rule-based, plus the PER-HAND paired
gap oracle-br minus PIMC v3 (deal luck cancels; CI on the gap itself).
n=2000 x 2 seeds per the standing discipline.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from play_eval import evaluate_paired
from fortyfives_rule_based import RuleBasedAgent
from fortyfives_pimc import PIMCAgent
from fortyfives_dds import OracleAgent

import fortyfives
assert 'fortyfives-rl-training' in fortyfives.__file__, fortyfives.__file__

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEEDS = (0, 100000)


def paired_gap(a, b, label):
    """CI on per-hand a-b (both runs share seeds/deals)."""
    assert len(a.diff) == len(b.diff), 'timeout misalignment'
    d = a.diff - b.diff
    m = d.mean()
    se = d.std(ddof=1) / len(d) ** 0.5
    print(f'  {label}: {m:+.3f}  (95% CI {m - 1.96 * se:+.3f} '
          f'to {m + 1.96 * se:+.3f})')
    return m


for seed in SEEDS:
    print(f'\n================ seed {seed}, n={N} ================')
    t0 = time.time()

    canary = evaluate_paired(RuleBasedAgent(18), num_hands=min(N, 500),
                             seed=seed, name='canary', silent=True)
    assert canary.avg_diff == 0.0 and canary.win_rate == 0.0, 'CANARY FAILED'
    print('canary: +0.0000 / 0.0%  OK')

    results = {}
    for name, agent in (
        ('pimc-v3', PIMCAgent(n_worlds=20)),
        ('oracle-br', OracleAgent(opponent='rulebased')),
        ('oracle-minimax', OracleAgent(opponent='minimax')),
    ):
        t = time.time()
        r = evaluate_paired(agent, num_hands=N, seed=seed, name=name,
                            silent=True)
        results[name] = r
        lo, hi = r.ci95
        print(f'{name:15s} avg_diff {r.avg_diff:+.3f} '
              f'(95% CI {lo:+.3f} to {hi:+.3f}) '
              f'beats-base {r.win_rate * 100:.1f}%  '
              f'[{time.time() - t:.0f}s]')
        if hasattr(agent, 'rb_fallbacks') and agent.rb_fallbacks:
            print(f'  WARNING: {agent.rb_fallbacks} rb fallbacks')

    print('per-hand paired gaps:')
    paired_gap(results['oracle-br'], results['pimc-v3'],
               'oracle-br  - pimc-v3 ')
    paired_gap(results['oracle-minimax'], results['pimc-v3'],
               'oracle-mm  - pimc-v3 ')
    print(f'seed total: {(time.time() - t0) / 60:.1f} min')
