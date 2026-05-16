"""
Bidding-focused paired evaluation for Fortyfives bidding agents.

Mirror image of play_eval: there the play agent varied and bidding was
fixed rule-based. Here BIDDING (phase 1) + DECLARATION (phase 2) for the
NS seats (0, 2) is the agent under test, while everything that is NOT the
bidding variable is held constant and identical between the paired runs:

  - phase 4 (card play), ALL seats        -> play_agent  (default PIMC v3)
  - phase 3 (discard),  ALL seats         -> rule-based
  - phase 1/2, EW seats (1, 3)            -> rule-based
  - phase 1/2, NS seats (0, 2)            -> bid_agent   (THE variable)

Because play is held at PIMC v3 (≈ best known play, memory:
project-play-phase-ceiling) for both teams, an aggressive-but-makeable
bid is realized rather than wasted, so the bidding signal is not
confounded by weak play.

The per-hand point delta captures the real make/miss payoff
(made bid -> BID_SUCCESS_VALUES e.g. 30->60; miss -> -bid_value;
memory: project-scoring-rule), which is exactly the bidding signal.

Yardstick discipline (memory: project-play-phase-ceiling):
trust avg_diff with its paired CI; confirm any apparent win at
n>=2000 on >=2 independent seeds. Bidding canary: a rule-based
bidder vs itself MUST be diff 0.0000 (pairing intact + baseline
deterministic).

Usage
-----
from bid_eval import evaluate_bidding_paired, bidding_canary
from fortyfives_rule_based import RuleBasedAgent

bidding_canary()                                   # must be 0.0000
evaluate_bidding_paired(MyBidder(18), num_hands=2000, seed=2024)
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

import rlcard
from rlcard.envs.registration import register, registry

if 'fortyfives' not in registry.env_specs:
    register(
        env_id='fortyfives',
        entry_point='fortyfives.envs.fortyfives_env:FortyfivesEnv',
    )

sys.path.insert(0, os.path.dirname(__file__))
from fortyfives_rule_based import RuleBasedAgent
from play_eval import greedy, PairedResult


class RandomBidder:
    """Skill-free bidding baseline: uniform random over legal phase-1/2
    actions. Own RandomState (seeded once) so a run is reproducible —
    env.seed() only seeds the dealer, never global np.random (hard-won
    invariant). Variance across hands is part of the headroom signal;
    the paired CI accounts for it."""

    def __init__(self, num_actions=18, seed=12345):
        self.num_actions = num_actions
        self.use_raw = True
        self._rng = np.random.RandomState(seed)

    def step(self, state):
        keys = list(state['legal_actions'].keys())
        return keys[self._rng.randint(len(keys))]

    def eval_step(self, state):
        return self.step(state), {}


def _default_play_agent():
    """Default fixed play reference = canonical PIMC v3. Imported lazily
    so a missing torch/PIMC dep does not break canary-only use."""
    from fortyfives_pimc import PIMCAgent
    return PIMCAgent(num_actions=18)


def _run_hand_bid(env, bid_agent, play_agent, rule_agent, seed):
    """Play one hand. NS (0,2) phase-1/2 = bid_agent; everything else
    fixed (see module docstring). Multi-hand game runs to 125 pts, so we
    stop at the phase 4->1 transition that marks one hand's end.

    Returns NS (team 0/2) raw game-point delta for the hand, or None on
    timeout. The delta already encodes made/miss payoff.

    play_agent / rule_agent / bid_agent are reused across hands; if
    play_agent carries an _rng we reseed it per hand so the eval itself
    is reproducible (deal luck still cancels via env.seed)."""
    NS_SEATS = (0, 2)
    env.seed(seed)
    if hasattr(play_agent, '_rng'):
        play_agent._rng = np.random.RandomState(seed)
    # EV-style bidders need the live env to clone the true position.
    if hasattr(bid_agent, 'set_env'):
        bid_agent.set_env(env)
    state, player_id = env.reset()

    init_points = env.game.points.get(0, 0) if env.game.points else 0
    in_play = False
    step = 0

    while step < 500:
        step += 1
        phase = state['raw_obs']['phase']
        if phase == 4:
            in_play = True

        if phase == 4:
            action = play_agent.step(state)
        elif phase in (1, 2) and player_id in NS_SEATS:
            action = bid_agent.step(state)
        else:
            action = rule_agent.step(state)

        next_state, next_player_id = env.step(action)
        curr_phase = env.game.phase

        if (in_play and phase == 4 and curr_phase == 1) or env.game.is_over():
            points_now = env.game.points.get(0, 0) if env.game.points else 0
            return points_now - init_points

        state = next_state
        player_id = next_player_id

    return None


def evaluate_bidding_paired(bid_agent, baseline=None, num_hands=200,
                            seed=0, name='bidder', silent=False,
                            play_agent=None):
    """Paired bidding eval: bid_agent vs baseline on identical deals.

    Only NS phase-1/2 (bid + declaration) differs between the two runs;
    play (PIMC v3 by default), discard, and EW bidding are identical, so
    the per-hand point difference isolates NS bidding quality with deal
    luck cancelled. CI is on the paired differences (tight)."""
    if baseline is None:
        baseline = RuleBasedAgent(num_actions=18)
    if play_agent is None:
        play_agent = _default_play_agent()

    rule_agent = RuleBasedAgent(num_actions=18)
    env_a = rlcard.make('fortyfives')
    env_b = rlcard.make('fortyfives')
    diffs, agent_pts, timeouts = [], [], 0

    for i in range(num_hands):
        s = seed + i
        with greedy(bid_agent):
            pa = _run_hand_bid(env_a, bid_agent, play_agent, rule_agent, s)
        with greedy(baseline):
            pb = _run_hand_bid(env_b, baseline, play_agent, rule_agent, s)
        if pa is None or pb is None:
            timeouts += 1
            continue
        diffs.append(pa - pb)
        agent_pts.append(pa)

    # PairedResult expects (name, diffs, points, tricks); bidding eval
    # does not track NS tricks, pass zeros (tricks unused for bidding).
    result = PairedResult(name, diffs, agent_pts, [0.0] * len(diffs))
    if not silent:
        print(result)
        if timeouts:
            print(f"  (Timeouts: {timeouts}/{num_hands})")
    return result


def bidding_canary(num_hands=300, seed=0, play_agent=None):
    """Regression canary: a rule-based bidder vs itself MUST be
    diff 0.0000 / 0.0% beats. Non-zero => the eval is not actually
    paired or the bidder is non-deterministic; every number it
    produces is then noise. Run after any rule-based / eval edit."""
    res = evaluate_bidding_paired(
        RuleBasedAgent(num_actions=18), num_hands=num_hands, seed=seed,
        name='bidding-canary', silent=True, play_agent=play_agent)
    ok = abs(res.avg_diff) < 1e-9 and res.win_rate == 0.0
    print(f"BIDDING CANARY: avg_diff={res.avg_diff:+.4f} "
          f"beats={res.win_rate*100:.1f}% n={res.num_hands} "
          f"-> {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--canary', action='store_true')
    ap.add_argument('--num_hands', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fast_play', action='store_true',
                    help='use rule-based play instead of PIMC (fast smoke)')
    args = ap.parse_args()

    pa = RuleBasedAgent(num_actions=18) if args.fast_play else None
    if args.canary:
        bidding_canary(num_hands=args.num_hands, seed=args.seed,
                       play_agent=pa)
