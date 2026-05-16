#!/usr/bin/env python3
"""
EV-search bidding agent for Fortyfives (faithful engine rollout).

Categorically the PIMC paradigm lifted to the BID decision. At phase 1,
for each legal bid action it:
  1. clones the live env,
  2. re-determinizes the hidden cards (the 3 other hands + kitty + deck)
     from the cards not in our hand, keeping our real hand fixed,
  3. forces our seat to play the candidate bid,
  4. lets the held-constant rule-based policy finish the auction,
     declaration, discard, and (fast) play,
  5. reads the real engine NS game-point delta for the hand.
The mean delta over N worlds is EV(candidate); we bid the argmax.

This deliberately uses the REAL engine + real scoring for every rollout
(no hand-rolled P(make)/strength model) so the estimator cannot become a
silently-wrong instrument (project rule: a result you cannot trust is
worse than no result).

Single-variable design vs RuleBasedAgent: ONLY the phase-1 bid level is
chosen by lookahead. Declaration (phase 2), discard, and play are
delegated unchanged to RuleBasedAgent, so a bid_eval A/B isolates
"bid level by EV rollout" vs "bid level by the crude top-3-trump table".

Payoff context (engine, memory project-scoring-rule): made 20->+20,
25->+25, 30->+60; miss-> -bid. Under the bid_eval NS-delta metric,
passing is never negative for NS, so the rollout naturally learns to
bid only when the simulated make-EV beats collecting raw trick points.
"""

import copy
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from fortyfives.games.fortyfives.card import FortyfivesCard

sys.path.insert(0, os.path.dirname(__file__))
from fortyfives_rule_based import RuleBasedAgent

_DECK_BY_RS = {(c.rank, c.suit): c for c in (FortyfivesCard(i) for i in range(52))}


class EVBidder:
    """Faithful-rollout EV bidder. NS phase-1 only differs from
    RuleBasedAgent; everything else delegates to it."""

    def __init__(self, num_actions=18, n_worlds=20, seed=0):
        self.num_actions = num_actions
        self.n_worlds = n_worlds
        self.use_raw = True
        self._rb = RuleBasedAgent(num_actions)       # delegate + rollout policy
        self._rng = np.random.RandomState(seed)
        self._env = None

    # bid_eval injects the live env so we can clone the true position.
    def set_env(self, env):
        self._env = env

    # --- determinization -------------------------------------------------
    def _redeterminize(self, game, our_seat):
        """Re-deal the hidden cards in `game` (a clone) consistent with
        our observed hand. Phase-1 only: nothing is revealed except our
        hand (no plays; bids reveal no cards; kitty hidden). Preserves
        every container's size so engine invariants hold."""
        our_keys = {(c.rank, c.suit) for c in game.hands[our_seat]}
        pool = [_DECK_BY_RS[k] for k in _DECK_BY_RS if k not in our_keys]
        self._rng.shuffle(pool)
        i = 0
        for seat in range(len(game.hands)):
            if seat == our_seat:
                continue
            n = len(game.hands[seat])
            game.hands[seat] = pool[i:i + n]
            i += n
        n_pot = len(game.dealer.pot)
        game.dealer.pot = pool[i:i + n_pot]
        i += n_pot
        game.dealer.deck = pool[i:]

    # --- one rollout -----------------------------------------------------
    def _rollout(self, base_env, our_seat, forced_env_action):
        """Clone base_env, re-determinize, force our candidate bid, then
        drive the held-constant rule-based (incl. fast play) to the end
        of THIS hand. Returns NS (team 0/2) game-point delta."""
        cenv = copy.deepcopy(base_env)
        self._redeterminize(cenv.game, our_seat)
        g = cenv.game
        init_pts = g.points.get(0, 0) if g.points else 0

        # apply our forced action first (we are the current player)
        g.step(cenv._decode_action(forced_env_action))

        in_play = False
        step = 0
        while step < 500:
            step += 1
            if g.is_over():
                break
            phase = g.phase
            if phase == 4:
                in_play = True
            pid = g.current_player_id
            st = cenv._extract_state(g.get_state(pid))
            a_env = self._rb.step(st)             # rule-based for ALL seats
            g.step(cenv._decode_action(a_env))
            if in_play and phase == 4 and g.phase == 1:
                break
        pts_now = g.points.get(0, 0) if g.points else 0
        return pts_now - init_pts

    # --- agent API -------------------------------------------------------
    def step(self, state):
        raw = state['raw_obs']
        phase = raw.get('phase')

        # Only the phase-1 bid level is EV-chosen; everything else
        # (declaration/discard/play) delegates to rule-based unchanged.
        if phase != 1 or self._env is None:
            return self._rb.step(state)

        legal = sorted(state['legal_actions'].keys())
        if len(legal) <= 1:
            return legal[0] if legal else 0

        our_seat = raw['current_player']
        best_a, best_ev = legal[0], -1e18
        for a in legal:
            total = 0.0
            for _ in range(self.n_worlds):
                total += self._rollout(self._env, our_seat, a)
            ev = total / self.n_worlds
            if ev > best_ev:
                best_ev, best_a = ev, a
        return best_a

    def eval_step(self, state):
        return self.step(state), {}
