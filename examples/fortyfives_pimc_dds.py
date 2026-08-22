#!/usr/bin/env python3
"""
PIMC-DDS play agent: PIMC determinization with an EXACT double-dummy
solve per sampled world (the textbook Bridge/Skat recipe), replacing
PIMC v3's rule-based heuristic playout.

Two deliberate upgrades over PIMC v3, both downstream of the solver
(see RESEARCH.md § Active work):
  - Exact per-world play-out (fixes rollout-legality infidelity,
    "lever 2", and removes playout-policy noise entirely).
  - Bid-aware values: each world is scored as the final NS game-point
    delta (make/fail bid, 30-for-60), and the cross-world AVERAGE is
    taken in that space. E[delta] != f(E[raw]) exactly at
    make-the-bid vs risk-getting-set decisions — value PIMC v3
    structurally cannot see.

Determinization (seen cards, hard-void inference, world sizes) is
INHERITED from PIMCAgent unchanged, so PIMC v3 -> PIMC-DDS is a
single-variable change: the per-world evaluator.

opponent='minimax' (default): worlds are solved double-dummy — hidden
hands are known per world, policies are not, so paranoid play is the
principled choice. opponent='rulebased' is an ablation that models the
eval table's fixed EW policy inside each world (note: the forced policy
is hand-order-sensitive and sampled worlds have arbitrary order, so
this mode is approximate there by construction).

Usage:
    from fortyfives_pimc_dds import PIMCDDSAgent
    from play_eval import evaluate_paired
    evaluate_paired(PIMCDDSAgent(n_worlds=20), num_hands=2000)
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

from fortyfives.games.fortyfives.card import SUITS, get_card_rank

sys.path.insert(0, os.path.dirname(__file__))
from fortyfives_pimc import PIMCAgent, _is_trump, _BY_RS
from fortyfives_rule_based import RuleBasedAgent
from fortyfives_dds import DDSolver

try:
    # Numba/bitboard port — bit-identical to DDSolver for the minimax
    # model (gated by tests/test_dds_fast_equivalence.py), ~25x faster.
    from fortyfives_dds_fast import FastDDSolver
except ImportError:            # numba not installed: reference solver
    FastDDSolver = None


class PIMCDDSAgent(PIMCAgent):

    def __init__(self, num_actions=18, n_worlds=20, seed=0,
                 constrained=True, opponent='minimax', payoff='delta',
                 discard_counts=False, fast=True, auction=False,
                 auction_tries=64):
        # rollout is irrelevant here (no heuristic playout); pass
        # 'cheap' so the parent doesn't build a rule-based picker.
        super().__init__(num_actions=num_actions, n_worlds=n_worlds,
                         seed=seed, constrained=constrained,
                         rollout='cheap')
        self.opponent = opponent
        self.payoff = payoff
        # fast=True is NOT an A/B lever: FastDDSolver is bit-identical
        # to DDSolver (equivalence-gated), so results cannot differ.
        # It only applies to the minimax model; falls back silently
        # when numba is unavailable.
        self._solver_cls = (
            FastDDSolver if (fast and FastDDSolver is not None
                             and opponent == 'minimax') else DDSolver)
        # Estimator lever 1: constrain sampled worlds by each seat's
        # post-discard draw count (public at a real table). Rule-based
        # seats keep ONLY trump at discard, so kept = 5 - drawn is that
        # seat's trump count at replenish; minus trumps it has since
        # publicly played, it lower-bounds trumps still in hand.
        # Near-exact vs rule-based discarders; a heuristic prior vs
        # humans.
        self.discard_counts = discard_counts
        self._min_trumps = None   # per-step context for _determinize
        self._trump_str = None
        # Estimator lever 2: auction-conditioned worlds. Rule-based seats
        # bid deterministically from their PRE-DISCARD hand
        # (RuleBasedAgent._supported_bid), so every observed bid/pass is
        # a hard constraint on that hand. Worlds are sampled generatively
        # (current hands -> kept-trump labelling -> discards/kitty) and
        # rejected unless every hidden seat's implied pre-discard hand
        # reproduces its auction actions. Falls back to the lever-1
        # sampler after `auction_tries` rejections. Near-exact vs
        # rule-based bidders; a heuristic prior vs humans.
        self.auction = auction
        self.auction_tries = auction_tries
        self._rb = RuleBasedAgent(num_actions)
        self._ac_ctx = None        # per-step context (None = inactive)
        self.auction_stats = {'worlds': 0, 'accepted': 0, 'tries': 0,
                              'fallback': 0, 'ctx_skipped': 0}

    def _played_trumps(self, raw, trump_str):
        counts = {s: 0 for s in range(4)}
        for tr in (raw.get('trick_history') or []):
            for s, c in enumerate(tr):
                if c is not None and _is_trump(c, trump_str):
                    counts[s] += 1
        for s, c in enumerate(raw.get('current_trick') or []):
            if c is not None and _is_trump(c, trump_str):
                counts[s] += 1
        return counts

    def _deal_dc(self, unseen, sizes, voids, mt):
        """One lever-1 attempt: shuffle `unseen` in place, deal each seat
        >= mt[seat] trumps then fill. Returns {seat: cards} or None if
        over-constrained. (Body is the original lever-1 loop body; the
        RNG call sequence is unchanged so dc stays byte-identical.)"""
        trump = self._trump_str
        self._rng.shuffle(unseen)
        pool, out = list(unseen), {}
        for seat, n in sizes.items():
            vs = voids.get(seat, ()) if voids else ()
            need = min(mt.get(seat, 0), n)
            picked = []
            if need:
                avail = [c for c in pool
                         if _is_trump(c, trump) and c.suit not in vs]
                if len(avail) < need:
                    return None
                picked = avail[:need]
            fill = n - len(picked)
            if fill:
                pids = {id(c) for c in picked}
                avail = [c for c in pool
                         if id(c) not in pids and c.suit not in vs]
                if len(avail) < fill:
                    return None
                picked = picked + avail[:fill]
            pids = {id(c) for c in picked}
            pool = [c for c in pool if id(c) not in pids]
            out[seat] = picked
        return out

    def _determinize(self, seen, sizes, voids=None):
        """With discard-count constraints active, deal each seat at
        least its inferred minimum trump count, then fill the remaining
        slots from the full unseen pool (replenished cards are random,
        so trump stays in the fill pool). Greedy with reshuffled
        retries; falls back to the parent's sampler if over-constrained
        (e.g. a void conflict — rare/impossible vs rule-based).
        With the auction context active (lever 2) the lever-1 deal is
        further filtered by the auction predicate."""
        if self._ac_ctx is not None:
            return self._determinize_auction(seen, sizes, voids)
        mt = self._min_trumps
        if not mt or not any(mt.get(s, 0) for s in sizes):
            return super()._determinize(seen, sizes, voids)
        unseen = [_BY_RS[k] for k in _BY_RS if k not in seen]
        for _ in range(8):
            out = self._deal_dc(unseen, sizes, voids, mt)
            if out is not None:
                return out
        return super()._determinize(seen, sizes, voids)

    # ---- lever 2: auction-conditioned determinization -----------------
    _LEVEL = {1: 20, 2: 25, 3: 30}
    _KITTY = 4          # game id offset: BID_xx_KITTY == BID_xx + 4
    _HOLD = 4

    @staticmethod
    def _legal_bid_actions(highest, is_dealer):
        """Game-id legal auction set, mirroring game.get_legal_bids:
        PASS, open levels, HOLD (dealer, once there is a bid), and the
        kitty variant of every open level. Sorted tuple."""
        levels = [l for l in (1, 2, 3) if highest is None or l > highest]
        acts = [0] + levels
        if is_dealer and highest is not None:
            acts.append(4)
        acts += [l + 4 for l in levels]
        return tuple(sorted(acts))

    def _auction_turns(self, raw):
        """Replay the auction from the public record (bids / passed /
        on_kitty / dealer). Rule-based seats bid their level the first
        time it is legal and pass otherwise, so the turn sequence is
        determined by the final record. Returns {seat: [(legal_game_ids,
        action_game_id)]} with 0 = pass, or None if the replay does not
        reproduce highest_bidder/highest_bid (e.g. a dealer HOLD, which
        rule-based never chooses) — the caller then skips lever 2."""
        bids = raw.get('bids')
        if bids is None:
            return None
        passed = raw.get('passed') or []
        on_kitty = raw.get('on_kitty') or [False] * 4
        dealer = raw.get('dealer')
        hb, hbid = raw.get('highest_bidder'), raw.get('highest_bid')
        if dealer is None or hb is None or hbid is None or len(passed) != 4:
            return None
        # bids is a dict after init_game but a list of None after an
        # all-pass redeal (start_new_hand); read both shapes.
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
            legal = self._legal_bid_actions(highest, cur == dealer)
            L = level[cur]
            if cur not in done and L and L in legal:
                act = L + self._KITTY if on_kitty[cur] else L
                turns[cur].append((legal, act))
                highest, bidder = L, cur
                done.add(cur)
            else:
                turns[cur].append((legal, 0))
                sim_passed[cur] = True
            npass = sum(sim_passed)
            if (npass == 3 and bidder is not None) or npass == 4:
                break
            cur = (cur + 1) % 4
            while sim_passed[cur]:
                cur = (cur + 1) % 4
        else:
            return None
        if bidder != hb or highest != int(hbid):
            return None
        if any(bool(p) != sp for p, sp in zip(passed, sim_passed)):
            return None
        return turns

    def _bid_consistent(self, hand, turns, is_bidder, trump_str,
                        bidder_on_kitty=False):
        """Would RuleBasedAgent holding pre-discard `hand` have produced
        exactly these auction actions (and, if it declared from that
        hand, declared trump_str)? A kitty declarer declares from the
        kitty, so its suit is not a constraint on `hand`."""
        for legal, act in turns:
            if self._rb.desired_bid(hand, legal) != act:
                return False
        if is_bidder and not bidder_on_kitty:
            suit, _ = self._rb._supported_bid(hand)
            if suit != trump_str:
                return False
        return True

    def _played_cards(self, raw):
        out = {s: [] for s in range(4)}
        for tr in (raw.get('trick_history') or []):
            for s, c in enumerate(tr):
                if c is not None:
                    out[s].append(c)
        for s, c in enumerate(raw.get('current_trick') or []):
            if c is not None:
                out[s].append(c)
        return out

    def _determinize_auction(self, seen, sizes, voids):
        """Generative sampler: lever-1 deal of the CURRENT hidden hands ->
        label each seat's kept trumps (a (5 - drawn)-subset of its
        post-replenish trumps) -> draw its discards from the dead pool
        (unseen cards in no current hand: stock + discard pile) ->
        bidder: 8-card pre-discard set split uniformly into hand (5) +
        kitty (3) -> accept iff every hidden seat's pre-discard hand
        reproduces its auction actions. Rejection with a cap, then
        lever-1 fallback. self._last_world_meta records the accepted
        labelling ({seat: (kept, discards, pre_discard_hand)}) for tests.
        """
        ctx = self._ac_ctx
        mt = self._min_trumps or {}
        trump = self._trump_str
        rng = self._rng
        turns, rc, bidder, played = (ctx['turns'], ctx['rc'],
                                     ctx['bidder'], ctx['played'])
        unseen = [_BY_RS[k] for k in _BY_RS if k not in seen]
        st = self.auction_stats
        st['worlds'] += 1
        self._last_world_meta = None
        self._last_world = None
        last = None
        # bidder first: its constraint is the strongest -> earliest reject
        order = sorted(sizes, key=lambda s: (s != bidder, s))
        for _ in range(self.auction_tries):
            st['tries'] += 1
            cur = self._deal_dc(unseen, sizes, voids, mt)
            if cur is None:
                continue
            last = cur
            dealt = {id(c) for cs in cur.values() for c in cs}
            dead_nt = [c for c in unseen
                       if id(c) not in dealt and not _is_trump(c, trump)]
            dead_tr = [c for c in unseen
                       if id(c) not in dealt and _is_trump(c, trump)]
            rng.shuffle(dead_nt)
            rng.shuffle(dead_tr)
            meta, ok = {}, True
            for s in order:
                if s == bidder and ctx['kitty_declarer']:
                    # Declared from the kitty: the auction constrains
                    # only the thrown-in (dead) cards -> no constraint on
                    # this seat's live hand beyond lever 1.
                    continue
                if s in ctx['unmodelled']:
                    # Seat made a kitty bid the bot policy never makes
                    # (a human at a real table): no hand explains it ->
                    # lever 1 only for that seat instead of rejecting
                    # every world.
                    continue
                k = 5 - rc[s]                      # kept (all trump)
                R = list(cur[s]) + list(played[s])  # post-replenish hand
                tr = [c for c in R if _is_trump(c, trump)]
                if len(tr) < k:
                    ok = False
                    break
                if k < len(tr):
                    idx = rng.permutation(len(tr))[:k]
                    kept = [tr[i] for i in idx]
                else:
                    kept = list(tr)
                if s == bidder and rc[s] == 0:
                    # kept = the 5 highest trumps of the 8-card set; the
                    # 3 discards are non-trump or trump below the floor.
                    floor = min(get_card_rank(c, trump) for c in kept)
                    low = [c for c in dead_tr
                           if get_card_rank(c, trump) < floor]
                    elig = dead_nt + low
                    if len(elig) < 3:
                        ok = False
                        break
                    idx = rng.permutation(len(elig))[:3]
                    disc = [elig[i] for i in idx]
                    dids = {id(c) for c in disc}
                    dead_nt = [c for c in dead_nt if id(c) not in dids]
                    dead_tr = [c for c in dead_tr if id(c) not in dids]
                else:
                    n_disc = (3 + rc[s]) if s == bidder else rc[s]
                    if len(dead_nt) < n_disc:
                        ok = False
                        break
                    disc, dead_nt = dead_nt[:n_disc], dead_nt[n_disc:]
                pre = kept + disc
                if s == bidder:
                    # uniform split of the 8-card set into hand 5 / kitty 3
                    idx = rng.permutation(8)[:5]
                    hand = [pre[i] for i in idx]
                else:
                    hand = pre
                if not self._bid_consistent(hand, turns[s], s == bidder,
                                            trump):
                    ok = False
                    break
                meta[s] = (kept, disc, hand)
            if ok:
                st['accepted'] += 1
                self._last_world_meta = meta
                self._last_world = cur
                return cur
        st['fallback'] += 1
        if last is not None:
            return last
        return super()._determinize(seen, sizes, voids)

    def step(self, state):
        raw = state['raw_obs']
        raw_legal = list(state.get('raw_legal_actions') or [])
        if raw.get('phase') != 4 or not raw_legal:
            env_legal = list(state['legal_actions'].keys())
            return min(env_legal) if env_legal else 0

        hand = raw['hand']
        trump_str = raw['trump_suit']
        trump = SUITS.index(trump_str)
        our = raw['current_player']
        ct = raw['current_trick']
        played = {s: c for s, c in enumerate(ct) if c is not None}
        k = len(played)
        leader = (our - k) % 4
        t = len(raw.get('trick_history') or [])

        sizes = {}
        for s in range(4):
            if s == our:
                continue
            sizes[s] = (5 - t) - (1 if ct[s] is not None else 0)

        voids = self._voids(raw, trump_str, leader) if self.constrained \
            else None
        seen = self._seen(hand, ct, raw.get('trick_history'))

        # Discard-count constraint context (lever 1), consumed by our
        # _determinize override. kept_s = 5 - drawn_s is all trump for
        # rule-based discarders; subtract trumps seat s already showed.
        self._trump_str = trump_str
        self._min_trumps = None
        rc = raw.get('replenish_counts')
        if self.discard_counts and rc is not None:
            pt = self._played_trumps(raw, trump_str)
            self._min_trumps = {s: max(0, (5 - rc[s]) - pt[s])
                                for s in sizes}

        # Auction-conditioning context (lever 2), consumed by
        # _determinize_auction. Requires lever-1 data (rc) to label kept
        # trumps; the replay must reproduce the public outcome.
        self._ac_ctx = None
        if self.auction and rc is not None:
            turns = self._auction_turns(raw)
            if turns is None:
                self.auction_stats['ctx_skipped'] += 1
            else:
                if self._min_trumps is None:
                    pt = self._played_trumps(raw, trump_str)
                    self._min_trumps = {s: max(0, (5 - rc[s]) - pt[s])
                                        for s in sizes}
                ok = raw.get('on_kitty') or [False] * 4
                unmodelled = set()
                if not self._rb.kitty:
                    unmodelled = {s for s in range(4)
                                  if any(a >= 5 for _, a in turns[s])}
                self._ac_ctx = {'turns': turns, 'rc': list(rc),
                                'bidder': raw['highest_bidder'],
                                'kitty_declarer': bool(ok[raw['highest_bidder']]),
                                'unmodelled': unmodelled,
                                'played': self._played_cards(raw)}

        # Hand context for the bid-aware payoff.
        bid_team = raw['highest_bidder'] % 2
        bid_kind = raw['highest_bid']
        tw = raw['tricks_won']
        ns_tr, ew_tr = tw[0] + tw[2], tw[1] + tw[3]
        htp = raw.get('highest_trump_played')
        if htp is not None:
            best_rank = get_card_rank(htp, trump_str)
            best_par = raw['highest_trump_player'] % 2
        else:
            best_rank, best_par = -1, -1

        trick_ids = tuple((ct[s].id if ct[s] is not None else -1)
                          for s in range(4))
        our_ids = tuple(c.id for c in hand)

        totals = {a: 0.0 for a in raw_legal if 0 <= a < len(hand)}
        for _ in range(self.n_worlds):
            opp = self._determinize(seen, sizes, voids)
            hands = tuple(
                our_ids if s == our
                else tuple(c.id for c in opp.get(s, []))
                for s in range(4))
            solver = self._solver_cls(trump, bid_team, bid_kind,
                                      opponent=self.opponent,
                                      payoff=self.payoff)
            vals = solver.root_values(hands, leader, trick_ids,
                                      ns_tr, ew_tr, best_rank, best_par)
            for a in totals:
                totals[a] += vals[a]

        best = max(sorted(totals), key=lambda a: totals[a])
        return best + 9   # game hand index -> env play action id
