"""
Lever 2 (auction-conditioned determinization) gates for PIMCDDSAgent.

1. The auction replay from the public record (bids/passed/dealer)
   reproduces the real turn-by-turn transcript of rule-based auctions.
2. Ground truth is never rejected: every seat's TRUE pre-discard hand
   satisfies the predicate the sampler applies to sampled hands.
3. Accepted worlds respect the structural invariants (sizes, disjoint
   cards, kept trumps, predicate) and the sampler rarely falls back.
4. Synthetic records: replay shapes, and a dealer HOLD disables the
   constraint (returns None).
"""
import os
import sys
import unittest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _REPO:
    sys.path.insert(0, _REPO)
sys.path.insert(1, os.path.join(_REPO, 'examples'))

import rlcard
from rlcard.envs.registration import register, registry

if 'fortyfives' not in registry.env_specs:
    register(env_id='fortyfives',
             entry_point='fortyfives.envs.fortyfives_env:FortyfivesEnv')

from fortyfives_rule_based import RuleBasedAgent
from fortyfives_pimc import _is_trump
from fortyfives_pimc_dds import PIMCDDSAgent

LEVEL = {1: 20, 2: 25, 3: 30}


def _drive(seed, agent=None, on_play=None):
    """Run one hand with rule-based everywhere (agent.step is called for
    seat 0's phase-4 decisions when `agent` is given, purely to exercise
    the sampler; rule-based still picks the action). Returns
    (transcript {seat: [(legal_levels, act)]}, pre_discard_hands {seat:
    cards}, first phase-4 raw state) ."""
    env = rlcard.make('fortyfives')
    env.seed(seed)
    state, pid = env.reset()
    rb = RuleBasedAgent(18)
    transcript = {s: [] for s in range(4)}
    pre = None
    first_raw = None
    for _ in range(400):
        g = env.game
        phase = g.phase
        if phase == 1:
            legal = tuple(LEVEL[a] for a in g.get_legal_bids() if a in LEVEL)
            npass_before = sum(g.passed)
        if phase == 2 and pre is None:
            pre = {s: list(g.hands[s]) for s in range(4)}
        if phase == 4:
            if first_raw is None:
                first_raw = g.get_state(pid)
            if agent is not None and pid == 0 and on_play is not None:
                on_play(env, state, pid)
        action = rb.step(state)
        if phase == 1:
            act = LEVEL.get(action, 0)
            transcript[pid].append((legal, act))
        state, pid = env.step(action)
        if phase == 1 and g.phase == 1 and npass_before == 3 \
                and sum(g.passed) == 0:
            # all passed -> redeal: the record restarts
            transcript = {s: [] for s in range(4)}
            pre = None
        if g.phase == 1 and phase == 4:
            break
        if g.is_over():
            break
    return transcript, pre, first_raw


class TestAuctionReplay(unittest.TestCase):

    def test_replay_matches_real_transcripts(self):
        agent = PIMCDDSAgent(n_worlds=1, discard_counts=True, auction=True)
        checked = 0
        for seed in range(60):
            transcript, _, raw = _drive(seed)
            self.assertIsNotNone(raw)
            turns = agent._auction_turns(raw)
            self.assertIsNotNone(turns, f'seed {seed}: replay failed')
            self.assertEqual(turns, transcript, f'seed {seed}')
            checked += 1
        self.assertEqual(checked, 60)

    def test_ground_truth_never_rejected(self):
        agent = PIMCDDSAgent(n_worlds=1, discard_counts=True, auction=True)
        n_bid_seats = 0
        for seed in range(150):
            _, pre, raw = _drive(seed)
            turns = agent._auction_turns(raw)
            self.assertIsNotNone(turns)
            trump = raw['trump_suit']
            for s in range(4):
                ok = agent._bid_consistent(pre[s], turns[s],
                                           s == raw['highest_bidder'], trump)
                self.assertTrue(ok, f'seed {seed} seat {s} rejected its '
                                    f'true hand {pre[s]} turns {turns[s]}')
                if any(a for _, a in turns[s]):
                    n_bid_seats += 1
        self.assertGreater(n_bid_seats, 100)

    def test_sampled_worlds_respect_invariants(self):
        agent = PIMCDDSAgent(n_worlds=4, discard_counts=True, auction=True,
                             seed=1)
        checks = {'worlds': 0}

        def on_play(env, state, pid):
            raw = state['raw_obs']
            agent.step(state)
            meta, cur = agent._last_world_meta, agent._last_world
            if meta is None:
                return
            checks['worlds'] += 1
            ctx = agent._ac_ctx
            trump = raw['trump_suit']
            seen = agent._seen(raw['hand'], raw['current_trick'],
                               raw.get('trick_history'))
            t = len(raw.get('trick_history') or [])
            ct = raw['current_trick']
            all_cur = []
            for s, cards in cur.items():
                self.assertNotEqual(s, pid)
                self.assertEqual(len(cards),
                                 (5 - t) - (1 if ct[s] is not None else 0))
                for c in cards:
                    self.assertNotIn((c.rank, c.suit), seen)
                all_cur += cards
            self.assertEqual(len({id(c) for c in all_cur}), len(all_cur))
            cur_ids = {id(c) for c in all_cur}
            disc_ids = set()
            for s, (kept, disc, hand) in meta.items():
                rc = ctx['rc'][s]
                self.assertEqual(len(kept), 5 - rc)
                R = list(cur[s]) + list(ctx['played'][s])
                rids = {(c.rank, c.suit) for c in R}
                for c in kept:
                    self.assertTrue(_is_trump(c, trump))
                    self.assertIn((c.rank, c.suit), rids)
                for c in disc:
                    self.assertNotIn(id(c), cur_ids)
                    self.assertNotIn(id(c), disc_ids)
                    disc_ids.add(id(c))
                    if s != raw['highest_bidder'] or rc > 0:
                        self.assertFalse(_is_trump(c, trump))
                n_disc = (3 + rc) if s == raw['highest_bidder'] else rc
                self.assertEqual(len(disc), n_disc)
                self.assertEqual(len(hand), 5)
                self.assertTrue(agent._bid_consistent(
                    hand, ctx['turns'][s], s == raw['highest_bidder'],
                    trump))

        for seed in range(8):
            _drive(seed, agent=agent, on_play=on_play)
        self.assertGreater(checks['worlds'], 20)
        st = agent.auction_stats
        self.assertGreater(st['worlds'], 0)
        self.assertGreater(st['accepted'] / st['worlds'], 0.8,
                           f'sampler mostly falling back: {st}')

    def test_synthetic_records(self):
        agent = PIMCDDSAgent(n_worlds=1, discard_counts=True, auction=True)
        base = {'dealer': 0, 'passed': [True, False, True, True]}
        # seat 1 bids 20, everyone else passes
        raw = dict(base, bids={0: 0, 1: 1, 2: 0, 3: 0},
                   highest_bidder=1, highest_bid=1)
        turns = agent._auction_turns(raw)
        self.assertEqual(turns[1], [((20, 25, 30), 20)])
        for s in (2, 3, 0):
            self.assertEqual(turns[s], [((25, 30), 0)])
        # seat 1 bids 20, seat 2 raises to 25, 3/0 pass, 1 passes
        raw = {'dealer': 0, 'passed': [True, True, False, True],
               'bids': {0: 0, 1: 1, 2: 2, 3: 0},
               'highest_bidder': 2, 'highest_bid': 2}
        turns = agent._auction_turns(raw)
        self.assertEqual(turns[1], [((20, 25, 30), 20), ((30,), 0)])
        self.assertEqual(turns[2], [((25, 30), 25)])
        self.assertEqual(turns[3], [((30,), 0)])
        self.assertEqual(turns[0], [((30,), 0)])
        # dealer HOLD: seat 1 bid 20, dealer (0) held -> bids[0]==1 too,
        # highest_bidder==0. Replay cannot reproduce -> None (skip lever 2)
        raw = {'dealer': 0, 'passed': [False, True, True, True],
               'bids': {0: 1, 1: 1, 2: 0, 3: 0},
               'highest_bidder': 0, 'highest_bid': 1}
        self.assertIsNone(agent._auction_turns(raw))

    def test_predicate_semantics(self):
        from fortyfives.games.fortyfives.card import FortyfivesCard
        agent = PIMCDDSAgent(n_worlds=1, discard_counts=True, auction=True)
        by = {(c.rank, c.suit): c for c in (FortyfivesCard(i) for i in range(52))}
        H = lambda *ks: [by[k] for k in ks]
        open_t = [((20, 25, 30), 0)]
        # passer with 20 available cannot hold any 5
        self.assertFalse(agent._bid_consistent(
            H(('5', 'S'), ('2', 'H'), ('3', 'D'), ('4', 'C'), ('7', 'C')),
            open_t, False, 'H'))
        self.assertTrue(agent._bid_consistent(
            H(('K', 'S'), ('2', 'H'), ('3', 'D'), ('4', 'C'), ('7', 'C')),
            open_t, False, 'H'))
        # J + A♥ backup bids 20 -> not a passer
        self.assertFalse(agent._bid_consistent(
            H(('J', 'S'), ('A', 'H'), ('3', 'D'), ('4', 'C'), ('7', 'C')),
            open_t, False, 'H'))
        # bidder: 5♥+J♥ no A♥ -> 25 on hearts
        t25 = [((20, 25, 30), 25)]
        self.assertTrue(agent._bid_consistent(
            H(('5', 'H'), ('J', 'H'), ('3', 'D'), ('4', 'C'), ('7', 'C')),
            t25, True, 'H'))
        self.assertFalse(agent._bid_consistent(        # wrong suit
            H(('5', 'H'), ('J', 'H'), ('3', 'D'), ('4', 'C'), ('7', 'C')),
            t25, True, 'S'))
        self.assertFalse(agent._bid_consistent(        # would be 30
            H(('5', 'H'), ('J', 'H'), ('A', 'H'), ('4', 'C'), ('7', 'C')),
            t25, True, 'H'))


if __name__ == '__main__':
    unittest.main()
