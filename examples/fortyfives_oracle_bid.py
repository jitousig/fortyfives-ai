#!/usr/bin/env python3
"""
Oracle bidder: bid / declare by exact double-dummy expected value.

The strong play engine (FastDDSolver) lifted to the AUCTION. At each
phase-1 decision (and, with declare=True, the phase-2 trump choice) it:
  1. samples N worlds for the hidden cards — the 3 other hands, the
     kitty and the stock order — from the cards not in our hand,
     rejection-conditioned (conditioned=True) on the auction so far
     under the table's rule-based bid model (each earlier bid/pass is a
     hard constraint on that seat's hand — lever-2 logic);
  2. for every legal candidate action, finishes the auction with the
     rule-based bid model for the other seats, gives the declarer the
     kitty, applies the table's discard policy (keep only trump, top 5)
     and the engine's replenish order, then SOLVES the resulting deal
     double-dummy with the chosen payoff ('net' = ΔNS − ΔEW per
     end_hand, the bid_eval yardstick; 'delta' = ΔNS);
  3. if WE declare, the value of a level is max over trump suits of the
     cross-world mean (declaration is made before the kitty is seen, so
     the suit is chosen once, not per world); a kitty bid (kitty=True)
     instead swaps our hand for the kitty and declares per world (the
     kitty is seen before declaring);
  4. bids the candidate with the best mean (ties -> lowest action id,
     i.e. PASS first — deterministic).

Why this should beat the faithful-rollout EVBidder (+0.5/+0.6, n.s.):
exact play-out instead of rule-based rollouts (make-probability is no
longer distorted by weak play), the net metric (an opponent's failed
contract counts), auction-conditioned worlds, and ~50x cheaper worlds.
Double-dummy values are optimistic for BOTH sides, as in Bridge DD
bidding simulations — a standard, consistent proxy.

Everything outside bid/declare delegates to RuleBasedAgent (discard,
play) so bid_eval's A/B isolates the auction policy.

Usage:
    from fortyfives_oracle_bid import OracleBidder
    from bid_eval import evaluate_bidding_paired
    evaluate_bidding_paired(OracleBidder(n_worlds=40), num_hands=2000)
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from fortyfives.games.fortyfives.card import SUITS, FortyfivesCard, get_card_rank
from fortyfives.games.fortyfives.game import (
    BID_PASS, BID_20, BID_25, BID_30, BID_HOLD, KITTY_BIDS,
)

sys.path.insert(0, os.path.dirname(__file__))
from fortyfives_rule_based import RuleBasedAgent
from fortyfives_pimc import _is_trump

try:
    from fortyfives_dds_fast import FastDDSolver as _Solver
except ImportError:                       # numba missing: reference solver
    from fortyfives_dds import DDSolver as _Solver

_DECK = [FortyfivesCard(i) for i in range(52)]
_LEVELS = (BID_20, BID_25, BID_30)


def _is_ah(c):
    return c.rank == 'A' and c.suit == 'H'


class OracleBidder:

    def __init__(self, num_actions=21, n_worlds=40, seed=0, declare=True,
                 kitty=False, conditioned=True, payoff='net',
                 cond_tries=200, rb_kitty=False, declarer_penalty=15.0):
        self.num_actions = num_actions
        self.n_worlds = n_worlds
        self.declare = declare
        self.kitty = kitty              # consider kitty bids as candidates
        self.conditioned = conditioned  # auction-condition the worlds
        self.payoff = payoff
        self.cond_tries = cond_tries
        # Double-dummy calibration: a perfect-information solve overrates
        # the DECLARER (controls trump, saw the kitty, plays with full
        # knowledge) relative to hidden-information play. Diagnosed
        # 2026-08-22: uncalibrated (0) the oracle over-declared (dealer
        # PASS->HOLD -4.6, PASS->25 -19.6 net/hand vs rb). Subtract
        # `declarer_penalty` points from whichever side declares in every
        # valuation (we declare: v - d; they declare: v + d).
        # CONFIRMED 2026-08-22 (bid_eval, n=2000 x 2 seeds, PIMC-DDS
        # lever-1 play, net metric, vs RuleBasedAgent bidder):
        #   d=10 pooled +1.51 (CI +0.97..+2.04); d=15 pooled +1.65
        #   (CI +1.10..+2.20); d15-d10 +0.14 n.s. Default 15.
        self.declarer_penalty = float(declarer_penalty)
        self.use_raw = True
        # Table model: how the OTHER seats bid/declare/discard. The
        # bid_eval / web tables run RuleBasedAgent with kitty=False.
        self._rb = RuleBasedAgent(num_actions, kitty=rb_kitty)
        self._rng = np.random.RandomState(seed)
        self.stats = {'decisions': 0, 'worlds': 0, 'cond_fallback': 0,
                      'solves': 0}

    # ------------------------------------------------------------------
    # table model helpers (mirror the engine / RuleBasedAgent exactly)
    # ------------------------------------------------------------------
    @staticmethod
    def _legal_bids(highest, is_dealer, kitty_allowed=True):
        levels = [l for l in _LEVELS if highest is None or l > highest]
        acts = [BID_PASS] + levels
        if is_dealer and highest is not None:
            acts.append(BID_HOLD)
        if kitty_allowed:
            acts += [l + 4 for l in levels]
        return acts

    def _rb_declare(self, hand):
        suit, _ = self._rb._supported_bid(hand)
        if suit is not None:
            return SUITS.index(suit)
        counts = {s: sum(1 for c in hand if c.suit == s) for s in SUITS}
        best = sorted(SUITS, key=lambda x: (-counts[x], SUITS.index(x)))[0]
        return SUITS.index(best)

    @staticmethod
    def _rb_keep(hand, trump_str):
        """RuleBasedAgent discard policy: keep only trump (A♥ incl.),
        and if more than 5 trump keep the 5 highest-ranked."""
        tr = [c for c in hand if _is_trump(c, trump_str)]
        if len(tr) > 5:
            tr.sort(key=lambda c: -get_card_rank(c, trump_str))
            tr = tr[:5]
        return tr

    def _solver(self, trump, bid_team, bid_kind):
        # Fresh solver per solve: the transposition table is only useful
        # within one deal; cached across deals it grows without bound
        # (PIMC-DDS also builds one solver per world).
        return _Solver(trump, bid_team, bid_kind, payoff=self.payoff)

    # ------------------------------------------------------------------
    # auction replay (constraints on the other seats' hands so far)
    # ------------------------------------------------------------------
    def _turns_so_far(self, raw):
        """Replay the auction record up to NOW. Returns (turns, sim) where
        turns = {seat: [(legal_game_ids, action)]} for actions already
        taken and sim = dict(highest, bidder, passed, done, on_kitty)
        describing the live auction state, or None if the record is not
        reproducible (then worlds are unconditioned)."""
        bids = raw.get('bids')
        passed = raw.get('passed') or []
        on_kitty = raw.get('on_kitty') or [False] * 4
        dealer = raw.get('dealer')
        me = raw['current_player']
        if bids is None or dealer is None or len(passed) != 4:
            return None

        def _bid(s):
            try:
                v = bids[s]
            except (KeyError, IndexError, TypeError):
                return 0
            return int(v) if v else 0
        level = {s: _bid(s) for s in range(4)}
        turns = {s: [] for s in range(4)}
        highest, bidder, done = None, None, set()
        sim_passed = [False] * 4
        cur = (dealer + 1) % 4
        for _ in range(16):
            legal = tuple(sorted(self._legal_bids(highest, cur == dealer)))
            L = level[cur]
            if cur not in done and L and L in legal:
                act = L + 4 if on_kitty[cur] else L
                turns[cur].append((legal, act))
                highest, bidder = L, cur
                done.add(cur)
            elif passed[cur]:
                turns[cur].append((legal, BID_PASS))
                sim_passed[cur] = True
            else:
                # this seat has not acted at this visit: it is NOW
                if cur != me:
                    return None
                break
            npass = sum(sim_passed)
            if (npass == 3 and bidder is not None) or npass == 4:
                return None          # auction would be over
            cur = (cur + 1) % 4
            while sim_passed[cur]:
                cur = (cur + 1) % 4
        else:
            return None
        hb = raw.get('highest_bidder')
        hbid = raw.get('highest_bid')
        if bidder != hb or (highest or None) != (int(hbid) if hbid is not None else None):
            return None
        return turns, {'highest': highest, 'bidder': bidder,
                       'passed': sim_passed, 'done': done,
                       'on_kitty': list(on_kitty), 'dealer': dealer}

    def _seat_consistent(self, hand, turns):
        for legal, act in turns:
            if self._rb.desired_bid(hand, legal) != act:
                return False
        return True

    # ------------------------------------------------------------------
    # world sampling
    # ------------------------------------------------------------------
    def _sample_world(self, me, my_hand, turns):
        """Returns (hands{seat: [cards]} for the 3 other seats, kitty
        list, stock list (engine deck order; pop() from the end))."""
        mine = {c.id for c in my_hand}
        pool = [c for c in _DECK if c.id not in mine]
        self._rng.shuffle(pool)
        pool = list(pool)
        hands = {}
        order = [s for s in range(4) if s != me]
        # constrained seats first (their rejection sampling is the
        # expensive part; unconstrained seats just take what is left)
        if turns:
            order.sort(key=lambda s: -len(turns.get(s, [])))
        for s in order:
            ts = turns.get(s) if turns else None
            if ts:
                ok = False
                for _ in range(self.cond_tries):
                    idx = self._rng.permutation(len(pool))[:5]
                    cand = [pool[i] for i in idx]
                    if self._seat_consistent(cand, ts):
                        ok = True
                        break
                if not ok:
                    self.stats['cond_fallback'] += 1
                    cand = pool[:5]
                    idx = list(range(5))
                chosen = set(int(i) for i in idx)
                hands[s] = cand
                pool = [c for i, c in enumerate(pool) if i not in chosen]
            else:
                hands[s], pool = pool[:5], pool[5:]
        kitty, stock = pool[:3], pool[3:]
        return hands, kitty, stock

    # ------------------------------------------------------------------
    # finish the auction for one candidate action, then value the deal
    # ------------------------------------------------------------------
    def _finish_auction(self, me, my_action, hands, sim, dealer):
        """Continue the live auction after we take `my_action` (game id),
        other seats per the table model, we PASS on any later turn.
        Returns (declarer, level, on_kitty) or None if all pass."""
        highest = sim['highest'] if sim else None
        bidder = sim['bidder'] if sim else None
        passed = list(sim['passed']) if sim else [False] * 4
        on_kitty = list(sim['on_kitty']) if sim else [False] * 4
        cur = me
        act = my_action
        for _ in range(20):
            if act == BID_PASS:
                passed[cur] = True
            elif act == BID_HOLD:
                bidder = cur
                highest = highest if highest is not None else BID_20
                on_kitty[cur] = False
            else:
                level = KITTY_BIDS.get(act, act)
                bidder, highest = cur, level
                on_kitty[cur] = act in KITTY_BIDS
            npass = sum(passed)
            if (npass == 3 and bidder is not None):
                return bidder, highest, on_kitty[bidder]
            if npass == 4:
                return None
            cur = (cur + 1) % 4
            while passed[cur]:
                cur = (cur + 1) % 4
            if cur == me:
                act = BID_PASS
            else:
                legal = self._legal_bids(highest, cur == dealer)
                act = self._rb.desired_bid(hands[cur], legal)
        return None

    def _value(self, me, my_hand, hands, kitty, stock, declarer, level,
               trump, declarer_on_kitty):
        """Double-dummy value (payoff) of the deal once `declarer` has won
        at `level` and declares `trump` (suit index): kitty -> declarer
        (or kitty swap), table discard policy, engine replenish order,
        solve from the seat left of the declarer."""
        trump_str = SUITS[trump]
        full = {s: list(hands[s]) for s in hands}
        full[me] = list(my_hand)
        if declarer_on_kitty:
            full[declarer] = [c for c in full[declarer] if _is_ah(c)] + list(kitty)
        else:
            full[declarer] = full[declarer] + list(kitty)
        deck = list(stock)
        post = {}
        for s in range(4):          # engine: seats 0..3 draw in order
            keep = self._rb_keep(full[s], trump_str)
            need = 5 - len(keep)
            drawn = []
            for _ in range(need):
                if deck:
                    drawn.append(deck.pop())
            post[s] = keep + drawn
        ids = tuple(tuple(c.id for c in post[s]) for s in range(4))
        solver = self._solver(trump, declarer % 2, level)
        self.stats['solves'] += 1
        v = solver.solve(ids, (declarer + 1) % 4)     # NS perspective
        if self.declarer_penalty:
            v = v - self.declarer_penalty if declarer % 2 == 0 \
                else v + self.declarer_penalty
        # The solver's payoff is ALWAYS dNS - dEW; an EW seat maximizes
        # the negative. (bid_eval only ever seats the bidder at NS, so
        # this was invisible there; at the web table it made EW bots bid
        # 30 on junk. Gated by tests/test_oracle_bid.py symmetry test.)
        return v if me % 2 == 0 else -v

    # ------------------------------------------------------------------
    # decisions
    # ------------------------------------------------------------------
    def _bid_decision(self, raw, legal_env):
        me = raw['current_player']
        my_hand = list(raw['hand'])
        dealer = raw.get('dealer', 0)
        rep = self._turns_so_far(raw) if self.conditioned else None
        turns, sim = (rep if rep else (None, None))
        if sim is None:
            # unconditioned: live auction state straight from raw
            hb, hbid = raw.get('highest_bidder'), raw.get('highest_bid')
            sim = {'highest': int(hbid) if hbid is not None else None,
                   'bidder': hb,
                   'passed': [bool(p) for p in (raw.get('passed') or [False] * 4)],
                   'on_kitty': list(raw.get('on_kitty') or [False] * 4)}
        # candidates in GAME ids
        cands = []
        for a in sorted(legal_env):
            g = a - 13 if a >= 18 else a
            if g in KITTY_BIDS and not self.kitty:
                continue
            cands.append(g)
        if len(cands) <= 1:
            return cands[0] if cands else BID_PASS

        # accumulators: for "we declare at level L": per suit sums;
        # for pass/outbid/kitty: scalar sums
        sums = {g: 0.0 for g in cands}
        suit_sums = {g: [0.0] * 4 for g in cands}
        self.stats['decisions'] += 1
        for _ in range(self.n_worlds):
            self.stats['worlds'] += 1
            hands, kitty, stock = self._sample_world(me, my_hand, turns)
            for g in cands:
                res = self._finish_auction(me, g, hands, sim, dealer)
                if res is None:
                    continue            # all pass: 0
                declarer, level, dk = res
                if declarer != me:
                    trump = self._rb_declare(hands[declarer])
                    sums[g] += self._value(me, my_hand, hands, kitty, stock,
                                           declarer, level, trump, dk)
                elif dk:
                    # kitty bid: we declare after seeing the kitty
                    swapped = [c for c in my_hand if _is_ah(c)] + list(kitty)
                    t = self._rb_declare(swapped)   # v1: table declare rule
                    sums[g] += self._value(me, my_hand, hands, kitty, stock,
                                           me, level, t, True)
                else:
                    for t in range(4):
                        suit_sums[g][t] += self._value(
                            me, my_hand, hands, kitty, stock, me, level, t, False)
        best_g, best_v = None, -1e18
        self.last_evs = {}
        for g in cands:
            if any(suit_sums[g]):
                v = max(suit_sums[g])       # declaration chosen once
            else:
                v = sums[g]
            v /= self.n_worlds
            self.last_evs[g] = v
            if v > best_v + 1e-9:
                best_v, best_g = v, g
        return best_g

    def _declare_decision(self, raw, legal_env):
        me = raw['current_player']
        my_hand = list(raw['hand'])
        level = int(raw['highest_bid'])
        on_kitty = bool((raw.get('on_kitty') or [False] * 4)[me])
        rep = self._turns_so_far_complete(raw) if self.conditioned else None
        turns = rep
        sums = [0.0] * 4
        self.stats['decisions'] += 1
        for _ in range(self.n_worlds):
            self.stats['worlds'] += 1
            hands, kitty, stock = self._sample_world(me, my_hand, turns)
            for t in range(4):
                if on_kitty:
                    # hand already swapped by the engine: my_hand IS
                    # A♥(+) + kitty, and the kitty is gone -> the
                    # sampled "kitty" is just 3 more stock cards.
                    v = self._value(me, my_hand, hands, [], kitty + stock,
                                    me, level, t, False)
                else:
                    v = self._value(me, my_hand, hands, kitty, stock,
                                    me, level, t, False)
                sums[t] += v
        return int(max(range(4), key=lambda t: (sums[t], -t)))

    def _turns_so_far_complete(self, raw):
        """Full-auction replay (phase 2): reuse the phase-1 replayer on a
        record where the auction is over. Returns turns or None."""
        # Same algorithm, but the auction is complete: walk until over.
        bids = raw.get('bids')
        passed = raw.get('passed') or []
        on_kitty = raw.get('on_kitty') or [False] * 4
        dealer = raw.get('dealer')
        if bids is None or dealer is None or len(passed) != 4:
            return None

        def _bid(s):
            try:
                v = bids[s]
            except (KeyError, IndexError, TypeError):
                return 0
            return int(v) if v else 0
        level = {s: _bid(s) for s in range(4)}
        turns = {s: [] for s in range(4)}
        highest, bidder, done = None, None, set()
        sim_passed = [False] * 4
        cur = (dealer + 1) % 4
        for _ in range(16):
            legal = tuple(sorted(self._legal_bids(highest, cur == dealer)))
            L = level[cur]
            if cur not in done and L and L in legal:
                turns[cur].append((legal, L + 4 if on_kitty[cur] else L))
                highest, bidder = L, cur
                done.add(cur)
            else:
                turns[cur].append((legal, BID_PASS))
                sim_passed[cur] = True
            npass = sum(sim_passed)
            if (npass == 3 and bidder is not None) or npass == 4:
                break
            cur = (cur + 1) % 4
            while sim_passed[cur]:
                cur = (cur + 1) % 4
        else:
            return None
        if bidder != raw.get('highest_bidder'):
            return None
        # our own turns are not constraints on OTHER hands
        return turns

    # ------------------------------------------------------------------
    # agent API
    # ------------------------------------------------------------------
    def step(self, state):
        raw = state['raw_obs']
        phase = raw.get('phase')
        legal_env = list(state['legal_actions'].keys())
        if phase == 1:
            g = self._bid_decision(raw, legal_env)
            return g + 13 if g >= 5 else g
        if phase == 2 and self.declare:
            t = self._declare_decision(raw, legal_env)
            env_a = t + 5
            return env_a if env_a in legal_env else min(legal_env)
        return self._rb.step(state)

    def eval_step(self, state):
        return self.step(state), {}
