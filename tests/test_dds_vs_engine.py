"""
Correctness gate for the double-dummy solver (examples/fortyfives_dds.py).

Nothing downstream of the DDS (oracle numbers, PIMC-DDS results) may be
trusted until this file passes. Four independent layers:

1. Engine parity: on random play-outs driven through the REAL
   FortyfivesGame, the DDS state machine must agree with the engine on
   every legal-move set, every trick winner, the running highest-trump,
   and the final NS/EW game-point deltas (end_hand bid adjustments).
2. Search correctness: alpha-beta + transposition table must equal a
   plain brute-force minimax on random endgames, in both opponent modes
   and both payoff modes, including partial-trick entry states.
3. Rule-based reconstruction fidelity: the forced-EW move the solver
   predicts must equal the move the real RuleBasedAgent makes when fed
   the engine's own state presentation.
4. Engine-grounded best response: from real mid-hand positions, the
   solver's forced-mode value must equal the maximum NS game-point
   delta achievable by exhaustively rolling the REAL engine forward
   (deep-copied) with the real rule-based agent playing EW.
"""

import copy
import os
import random
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(1, os.path.join(_REPO_ROOT, 'examples'))

import numpy as np

from fortyfives.games.fortyfives.game import (
    FortyfivesGame, PHASE_AUCTION, PHASE_DECLARATION, PHASE_DISCARD,
    PHASE_GAMEPLAY, BID_PASS, DISCARD_DONE,
)
from fortyfives.games.fortyfives.card import SUITS, get_card_rank

from fortyfives_dds import (
    DDSolver, OracleAgent, legal_plays, trick_winner, _RANK, _ISTRUMP,
)
from fortyfives_rule_based import RuleBasedAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def drive_to_play(seed, bidder_seat=None):
    """Bring a fresh game to the start of the play phase. Random-ish
    auction/declaration/discard driven by `seed`; exactly one player
    bids (varied across seeds) so every hand has a bid context."""
    rng = random.Random(seed)
    game = FortyfivesGame()
    game.np_random = np.random.RandomState(seed)
    game.init_game()

    if bidder_seat is None:
        bidder_seat = rng.randrange(4)
    level = rng.choice([1, 2, 3])

    guard = 0
    while game.phase != PHASE_GAMEPLAY:
        guard += 1
        assert guard < 200, 'game failed to reach play phase'
        pid = game.current_player_id
        if game.phase == PHASE_AUCTION:
            if pid == bidder_seat and game.highest_bid is None:
                game.step(level)
            else:
                game.step(BID_PASS)
        elif game.phase == PHASE_DECLARATION:
            game.step(rng.randrange(4))
        elif game.phase == PHASE_DISCARD:
            legal = game.get_legal_discards()
            cards = [a for a in legal if a != DISCARD_DONE]
            if DISCARD_DONE in legal and (not cards or rng.random() < 0.6):
                game.step(DISCARD_DONE)
            else:
                game.step(rng.choice(cards))
    return game


def solver_state_from_game(game):
    """Extract (hands, leader, trick, ns_tr, ew_tr, best_rank, best_par)
    in solver id-space from a live game in the play phase."""
    hands = tuple(tuple(c.id for c in game.hands[s]) for s in range(4))
    trick = tuple((game.current_trick[s].id
                   if game.current_trick[s] is not None else -1)
                  for s in range(4))
    ns_tr = game.tricks_won[0] + game.tricks_won[2]
    ew_tr = game.tricks_won[1] + game.tricks_won[3]
    if game.highest_trump_played is not None:
        best_rank = get_card_rank(game.highest_trump_played, game.trump_suit)
        best_par = game.highest_trump_player % 2
    else:
        best_rank, best_par = -1, -1
    return hands, game.trick_starter, trick, ns_tr, ew_tr, best_rank, best_par


def rb_env_choice(rb, game):
    """The move the real rule-based agent makes, fed the engine's own
    state presentation (what env raw_obs / raw_legal_actions contain)."""
    raw_obs = game.get_state(game.current_player_id)
    return rb._play_strategy(raw_obs, list(game.get_legal_plays()))


def leaf_ref(bid_team, bid_kind, total_tricks, ns_tricks, best_par,
             payoff):
    """Independent re-implementation of the leaf payoff (game.end_hand
    with both scores below 100)."""
    bid_value = {1: 20, 2: 25, 3: 30}[bid_kind]
    ns_raw = 5 * ns_tricks + (5 if best_par == 0 else 0)
    ew_raw = (5 * (total_tricks - ns_tricks)
              + (5 if best_par == 1 else 0))
    if payoff == 'raw':
        return ns_raw - ew_raw
    if bid_team == 0:
        if ns_raw >= bid_value:
            return 60 if bid_kind == 3 else ns_raw
        return -bid_value
    return ns_raw


def brute_force(solver, hands, trick, leader, ns_tricks, best_rank,
                best_par, total_tricks):
    """Plain minimax, no pruning, no memo — independent of the solver's
    search. Transitions written out separately from DDSolver._child_value."""
    if not any(hands):
        return leaf_ref(solver._bid_team, solver._bid_kind, total_tricks,
                        ns_tricks, best_par, solver._payoff)
    played = sum(1 for c in trick if c >= 0)
    seat = (leader + played) % 4
    lead = trick[leader] if played else None
    legal = legal_plays(hands[seat], lead, solver._trump)
    if solver._forced and seat % 2 == 1:
        moves = (solver._rb_choice(hands[seat], trick, legal),)
    else:
        moves = legal
    vals = []
    for i in moves:
        card = hands[seat][i]
        nh = list(hands)
        nh[seat] = hands[seat][:i] + hands[seat][i + 1:]
        nh = tuple(nh)
        nt = list(trick)
        nt[seat] = card
        nt = tuple(nt)
        nbr, nbp = best_rank, best_par
        if _ISTRUMP[solver._trump][card] and _RANK[solver._trump][card] > nbr:
            nbr, nbp = _RANK[solver._trump][card], seat % 2
        if sum(1 for c in nt if c >= 0) == 4:
            w = trick_winner(nt, leader, solver._trump)
            vals.append(brute_force(
                solver, nh, (-1, -1, -1, -1), w,
                ns_tricks + (1 if w % 2 == 0 else 0), nbr, nbp,
                total_tricks))
        else:
            vals.append(brute_force(solver, nh, nt, leader, ns_tricks,
                                    nbr, nbp, total_tricks))
    return max(vals) if seat % 2 == 0 else min(vals)


# ---------------------------------------------------------------------------
# 1. Engine parity on random play-outs
# ---------------------------------------------------------------------------

class TestEngineParity(unittest.TestCase):

    def test_random_playouts_match_engine(self):
        for seed in range(60):
            game = drive_to_play(seed)
            rng = random.Random(1000 + seed)
            self._playout_and_check(game, rng)

    def _playout_and_check(self, game, rng):
        trump = SUITS.index(game.trump_suit)
        bid_team = game.highest_bidder % 2
        bid_kind = game.highest_bid
        self.assertEqual(game.points.get(0, 0), 0)
        for s in range(4):
            self.assertEqual(len(game.hands[s]), 5)

        my_hands = [[c.id for c in game.hands[s]] for s in range(4)]
        my_trick = [-1, -1, -1, -1]
        my_leader = game.trick_starter
        my_ns, my_ew = 0, 0
        my_br, my_bp = -1, -1

        while game.phase == PHASE_GAMEPLAY:
            pid = game.current_player_id
            self.assertEqual(game.trick_starter, my_leader)
            engine_legal = sorted(game.get_legal_plays())
            lead = my_trick[my_leader] if my_trick[my_leader] >= 0 else None
            mine = sorted(legal_plays(tuple(my_hands[pid]), lead, trump))
            self.assertEqual(engine_legal, mine,
                             f'legal mismatch seat {pid}')

            a = rng.choice(engine_legal)
            card = my_hands[pid].pop(a)
            my_trick[pid] = card
            if _ISTRUMP[trump][card] and _RANK[trump][card] > my_br:
                my_br, my_bp = _RANK[trump][card], pid % 2

            prev_winners = len(game.trick_winners)
            game.step(a)

            if game.phase == PHASE_GAMEPLAY:
                # running high-trump must match the engine mid-hand
                if game.highest_trump_played is None:
                    self.assertEqual(my_br, -1)
                else:
                    self.assertEqual(
                        get_card_rank(game.highest_trump_played,
                                      game.trump_suit), my_br)
                    self.assertEqual(game.highest_trump_player % 2, my_bp)

            if all(c >= 0 for c in my_trick):
                # trick complete — count locally; the engine's
                # trick_winners survives only mid-hand (start_new_hand
                # wipes it after the fifth trick).
                w = trick_winner(tuple(my_trick), my_leader, trump)
                if len(game.trick_winners) > prev_winners:
                    self.assertEqual(game.trick_winners[-1], w)
                if w % 2 == 0:
                    my_ns += 1
                else:
                    my_ew += 1
                my_trick = [-1, -1, -1, -1]
                my_leader = w

        # hand over: engine applied end_hand — compare both deltas
        self.assertEqual(my_ns + my_ew, 5)
        for payoff, expected in (
            ('delta', game.points[0]),
        ):
            got = leaf_ref(bid_team, bid_kind, 5, my_ns, my_bp, payoff)
            self.assertEqual(got, expected,
                             f'NS delta mismatch ({payoff})')
        # EW side, independent re-derivation of end_hand for team 1
        bid_value = {1: 20, 2: 25, 3: 30}[bid_kind]
        ew_raw = 5 * my_ew + (5 if my_bp == 1 else 0)
        if bid_team == 1:
            ew_expected = ((60 if bid_kind == 3 else ew_raw)
                           if ew_raw >= bid_value else -bid_value)
        else:
            ew_expected = ew_raw
        self.assertEqual(ew_expected, game.points[1], 'EW delta mismatch')


# ---------------------------------------------------------------------------
# 2. Alpha-beta + TT vs brute force
# ---------------------------------------------------------------------------

class TestSearchVsBruteForce(unittest.TestCase):

    def _random_case(self, rng, k):
        deck = list(range(52))
        rng.shuffle(deck)
        hands = tuple(tuple(sorted(deck[k * s:k * s + k]))
                      for s in range(4))
        rest = deck[4 * k:]
        trump = rng.randrange(4)
        prior = 5 - k
        pns = rng.randint(0, prior)
        trumps_out = [c for c in rest if _ISTRUMP[trump][c]]
        if rng.random() < 0.5 and trumps_out:
            c = rng.choice(trumps_out)
            br, bp = _RANK[trump][c], rng.randrange(2)
        else:
            br, bp = -1, -1
        ctx = dict(trump=trump, bid_team=rng.randrange(2),
                   bid_kind=rng.choice([1, 2, 3]))
        return hands, trump, ctx, pns, prior - pns, br, bp

    def test_solver_equals_brute_force(self):
        rng = random.Random(7)
        cases = [(1, 40), (2, 30), (3, 10)]
        for k, n in cases:
            for _ in range(n):
                hands, trump, ctx, pns, pew, br, bp = \
                    self._random_case(rng, k)
                leader = rng.randrange(4)
                mode = rng.choice(['minimax', 'rulebased'])
                payoff = rng.choice(['delta', 'raw'])
                s = DDSolver(ctx['trump'], ctx['bid_team'],
                             ctx['bid_kind'], opponent=mode,
                             payoff=payoff)
                got = s.solve(hands, leader, ns_tricks=pns, ew_tricks=pew,
                              best_rank=br, best_par=bp)
                want = brute_force(s, hands, (-1, -1, -1, -1), leader,
                                   pns, br, bp, pns + pew + k)
                self.assertEqual(got, want,
                                 f'k={k} mode={mode} payoff={payoff}')
                # root_values must be exact per-move values
                rv = s.root_values(hands, leader, ns_tricks=pns,
                                   ew_tricks=pew, best_rank=br,
                                   best_par=bp)
                agg = max if leader % 2 == 0 else min
                self.assertEqual(agg(rv.values()), want)
                self.assertEqual(s.rb_fallbacks, 0)

    def test_move_reduction_is_exact(self):
        """Equivalence-class move reduction must not change any value:
        full 5-card deals, reduce=True vs reduce=False, both modes,
        including all root_values entries."""
        rng = random.Random(23)
        for trial in range(12):
            deck = list(range(52))
            rng.shuffle(deck)
            hands = tuple(tuple(sorted(deck[5 * s:5 * s + 5]))
                          for s in range(4))
            trump = rng.randrange(4)
            bt, bk = rng.randrange(2), rng.choice([1, 2, 3])
            mode = 'minimax' if trial % 2 == 0 else 'rulebased'
            leader = rng.randrange(4)
            rv = {}
            for red in (True, False):
                s = DDSolver(trump, bt, bk, opponent=mode, reduce=red)
                rv[red] = s.root_values(hands, leader)
            self.assertEqual(rv[True], rv[False],
                             f'trial {trial} mode={mode}')

    def test_partial_trick_entry(self):
        rng = random.Random(11)
        for _ in range(25):
            hands, trump, ctx, pns, pew, br, bp = self._random_case(rng, 2)
            leader = rng.randrange(4)
            s = DDSolver(ctx['trump'], ctx['bid_team'], ctx['bid_kind'],
                         opponent=rng.choice(['minimax', 'rulebased']),
                         payoff='delta')
            # walk 1-3 plays into the trick with random legal moves
            hands_l = [list(h) for h in hands]
            trick = [-1, -1, -1, -1]
            nbr, nbp = br, bp
            steps = rng.randint(1, 3)
            for j in range(steps):
                seat = (leader + j) % 4
                lead = trick[leader] if j else None
                legal = legal_plays(tuple(hands_l[seat]), lead, trump)
                i = rng.choice(legal)
                c = hands_l[seat].pop(i)
                trick[seat] = c
                if _ISTRUMP[trump][c] and _RANK[trump][c] > nbr:
                    nbr, nbp = _RANK[trump][c], seat % 2
            hands2 = tuple(tuple(h) for h in hands_l)
            trick2 = tuple(trick)
            got = s.solve(hands2, leader, trick2, pns, pew, nbr, nbp)
            want = brute_force(s, hands2, trick2, leader, pns, nbr, nbp,
                               pns + pew + 2)
            self.assertEqual(got, want)


# ---------------------------------------------------------------------------
# 3. Rule-based reconstruction fidelity
# ---------------------------------------------------------------------------

class TestRuleBasedFidelity(unittest.TestCase):

    def test_forced_move_matches_real_agent(self):
        rb = RuleBasedAgent(18)
        checked = 0
        for seed in range(40):
            game = drive_to_play(200 + seed)
            trump = SUITS.index(game.trump_suit)
            solver = DDSolver(trump, game.highest_bidder % 2,
                              game.highest_bid, opponent='rulebased')
            while game.phase == PHASE_GAMEPLAY:
                pid = game.current_player_id
                actual = rb_env_choice(rb, game)
                hand_ids = tuple(c.id for c in game.hands[pid])
                trick_ids = tuple(
                    (game.current_trick[s].id
                     if game.current_trick[s] is not None else -1)
                    for s in range(4))
                legal = tuple(sorted(game.get_legal_plays()))
                predicted = solver._rb_choice(hand_ids, trick_ids, legal)
                self.assertEqual(predicted, actual,
                                 f'seed {seed} seat {pid}')
                checked += 1
                game.step(actual)
            self.assertEqual(solver.rb_fallbacks, 0)
        self.assertGreater(checked, 500)


# ---------------------------------------------------------------------------
# 4. Engine-grounded best response (end to end, no shared transition code)
# ---------------------------------------------------------------------------

def _engine_best_response(game, rb):
    """Max NS game-point delta achievable from this real game position,
    NS exploring every legal move on deep copies, EW forced through the
    real rule-based agent. Exact by construction; exponential, so only
    used from small positions."""
    if game.phase != PHASE_GAMEPLAY:
        return game.points[0]
    pid = game.current_player_id
    if pid % 2 == 1:
        g = copy.deepcopy(game)
        g.step(rb_env_choice(rb, g))
        return _engine_best_response(g, rb)
    best = None
    for a in game.get_legal_plays():
        g = copy.deepcopy(game)
        g.step(a)
        v = _engine_best_response(g, rb)
        if best is None or v > best:
            best = v
    return best


class TestEngineGroundedBestResponse(unittest.TestCase):

    def test_forced_mode_value_matches_engine_search(self):
        rb = RuleBasedAgent(18)
        done = 0
        for seed in range(30):
            game = drive_to_play(500 + seed)
            # play the first two tricks with all seats rule-based
            while (game.phase == PHASE_GAMEPLAY
                   and len(game.trick_winners) < 2):
                game.step(rb_env_choice(rb, game))
            if game.phase != PHASE_GAMEPLAY:
                continue
            trump = SUITS.index(game.trump_suit)
            solver = DDSolver(trump, game.highest_bidder % 2,
                              game.highest_bid, opponent='rulebased',
                              payoff='delta')
            st = solver_state_from_game(game)
            hands, leader, trick, ns_tr, ew_tr, br, bp = st
            got = solver.solve(hands, leader, trick, ns_tr, ew_tr, br, bp)
            want = _engine_best_response(game, rb)
            self.assertEqual(got, want, f'seed {seed}')
            done += 1
        self.assertGreater(done, 15)


if __name__ == '__main__':
    unittest.main()
