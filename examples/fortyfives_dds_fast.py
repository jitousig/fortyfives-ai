#!/usr/bin/env python3
"""
Numba/bitboard port of the exact double-dummy solver.

Same values, much faster: `FastDDSolver` is a drop-in for
`fortyfives_dds.DDSolver` restricted to the PIMC-DDS hot path
(opponent='minimax', reduce=True; payoff 'delta' or 'raw') and is
required to return BIT-IDENTICAL `solve`/`root_values` results — gated
by tests/test_dds_fast_equivalence.py, which plays out random deals
comparing every decision against the reference solver. The forced
rule-based opponent model needs Python callbacks and stays on the
reference solver.

Design notes:
  - Hands are int64 bitmasks (bit c = card id c held); the current
    trick is packed 7 bits/seat (127 = not yet played). All rule
    tables (_RANK/_ISTRUMP/pools/between-masks) are DERIVED from the
    reference module at import so ranking/legality can never drift.
  - The transposition-table key uses hand MASKS (order-insensitive) —
    sound because position value depends only on card sets; entries
    pack (value << 2 | bound_flag) into one int64.
  - Move-equivalence partition replicates the reference `_move_classes`
    exactly (same classes, same lowest-rank representative), so root
    value assignment per hand index is identical, not merely
    value-equal. Internal move ORDER may differ from the reference —
    sound alpha-beta + sound TT bounds make the exact root value
    order-independent (only speed varies).
  - The play-a-card transition exists twice (inlined in `_search`,
    and in `_child` for root calls) because numba supports self- but
    not mutual recursion. Keep both in sync; the equivalence gate
    enforces it.

First call per process pays the JIT compile (~seconds); long-running
eval workers amortize it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from numba import njit
from numba.core import types
from numba.typed import Dict

import fortyfives_dds as _ref
from fortyfives_dds import _BID_VALUE, _EMPTY_TRICK

# ---------------------------------------------------------------------------
# Tables, derived from the reference module (never rebuilt independently).
# ---------------------------------------------------------------------------

_RANK = np.array(_ref._RANK, dtype=np.int64)            # [4, 52]
_SUIT_OF = np.array(_ref._SUIT_OF, dtype=np.int64)      # [52]
_POOL_OF = np.array(_ref._POOL_OF, dtype=np.int64)      # [4, 52]
_AH = int(_ref._AH)

_tm = [0, 0, 0, 0]
_sm = [0, 0, 0, 0]
for _c in range(52):
    _sm[_ref._SUIT_OF[_c]] |= 1 << _c
    for _t in range(4):
        if _ref._ISTRUMP[_t][_c]:
            _tm[_t] |= 1 << _c
_TRUMPMASK = np.array(_tm, dtype=np.int64)
_SUITMASK = np.array(_sm, dtype=np.int64)

_BETWEEN = np.zeros((4, 52, 52), dtype=np.int64)
for _t in range(4):
    for (_a, _b), _m in _ref._BETWEEN[_t].items():
        _BETWEEN[_t, _a, _b] = _m

# Card ids ascending by rank under each trump; pool filtering happens at
# use sites (cross-pool rank ties are never compared).
_ORDER_ASC = np.array(
    [sorted(range(52), key=lambda c, t=_t: _ref._RANK[t][c])
     for _t in range(4)], dtype=np.int64)

_BIG = 1 << 40
_EMPTYP = 127 | (127 << 7) | (127 << 14) | (127 << 21)

_KEY_T = types.UniTuple(types.int64, 5)


# ---------------------------------------------------------------------------
# njit kernels
# ---------------------------------------------------------------------------

@njit(cache=False)
def _legal(hand, lead_card, trump):
    """Legal-move MASK; replica of fortyfives_dds.legal_plays."""
    if lead_card < 0 or hand == 0:
        return hand
    tm = hand & _TRUMPMASK[trump]
    lead_suit = _SUIT_OF[lead_card]
    if (_TRUMPMASK[trump] >> lead_card) & 1:   # trump led (A-hearts incl.)
        if tm == 0:
            return hand
        led_rank = _RANK[trump, lead_card]
        m = tm
        while m:
            c = 0
            while ((m >> c) & 1) == 0:
                c += 1
            m &= m - 1
            r = _RANK[trump, c]
            if not (r >= 1001 and r > led_rank):
                return tm          # an obligated trump exists
        return hand                # every trump withholdable
    lead_have = hand & _SUITMASK[lead_suit]
    if lead_have:
        return lead_have | tm
    return hand


@njit(cache=False)
def _partition(legal, trump, best_rank, other_live, cards, cls):
    """Equivalence-class partition of the legal mask; replica of the
    reference `_move_classes` (ascending rank within pool, merge
    adjacent unless A-hearts / a live card between / straddling the
    best-trump bonus threshold). Fills cards[i]/cls[i], returns n."""
    n = 0
    ncls = -1
    for pid in range(5):
        if pid == 1 + trump:
            continue
        prev = -1
        for k in range(52):
            c = _ORDER_ASC[trump, k]
            if ((legal >> c) & 1) == 0 or _POOL_OF[trump, c] != pid:
                continue
            merge = False
            if prev >= 0:
                merge = (prev != _AH and c != _AH
                         and (other_live & _BETWEEN[trump, prev, c]) == 0
                         and (pid != 0
                              or (_RANK[trump, prev] > best_rank)
                              == (_RANK[trump, c] > best_rank)))
            if not merge:
                ncls += 1
            cards[n] = c
            cls[n] = ncls
            n += 1
            prev = c
    return n


@njit(cache=False)
def _winner(trickp, leader, trump):
    """Winning seat of a completed packed trick; replica of
    fortyfives_dds.trick_winner."""
    lead_c = (trickp >> (7 * leader)) & 127
    lead_suit = _SUIT_OF[lead_c]
    best_seat = leader
    best_rank = -1
    any_trump = False
    for s in range(4):
        c = (trickp >> (7 * s)) & 127
        if c == 127:
            continue
        if ((_TRUMPMASK[trump] >> c) & 1) != 0:
            if not any_trump:
                any_trump = True
                best_seat = s
                best_rank = _RANK[trump, c]
            elif _RANK[trump, c] > best_rank:
                best_seat = s
                best_rank = _RANK[trump, c]
        elif (not any_trump) and _SUIT_OF[c] == lead_suit:
            if _RANK[trump, c] > best_rank:
                best_seat = s
                best_rank = _RANK[trump, c]
    return best_seat


@njit(cache=False)
def _leaf(ns, bp, bid_team, bid_kind, payoff_raw, total):
    ns_raw = 5 * ns + (5 if bp == 0 else 0)
    if payoff_raw != 0:
        ew_raw = 5 * (total - ns) + (5 if bp == 1 else 0)
        return ns_raw - ew_raw
    if bid_team == 0:
        bv = 20 if bid_kind == 1 else (25 if bid_kind == 2 else 30)
        if ns_raw >= bv:
            return 60 if bid_kind == 3 else ns_raw
        return -bv
    return ns_raw


@njit(cache=False)
def _search(h0, h1, h2, h3, trickp, leader, ns, br, bp, live,
            alpha, beta, trump, bid_team, bid_kind, payoff_raw,
            total, tt):
    """Fail-soft alpha-beta with TT; replica of DDSolver._search
    (minimax opponent model)."""
    if (h0 | h1 | h2 | h3) == 0:
        return _leaf(ns, bp, bid_team, bid_kind, payoff_raw, total)

    misc = (trickp | (leader << 28) | (ns << 30)
            | ((br + 1) << 34) | ((bp + 1) << 45))
    key = (h0, h1, h2, h3, misc)
    if key in tt:
        e = tt[key]
        val = e >> 2
        flag = e & 3
        if flag == 0:                           # exact
            return val
        if flag == 1:                           # lower bound
            if val >= beta:
                return val
            if val > alpha:
                alpha = val
        else:                                   # upper bound
            if val <= alpha:
                return val
            if val < beta:
                beta = val

    played = 0
    for s in range(4):
        if ((trickp >> (7 * s)) & 127) != 127:
            played += 1
    seat = (leader + played) & 3
    if seat == 0:
        hand = h0
    elif seat == 1:
        hand = h1
    elif seat == 2:
        hand = h2
    else:
        hand = h3
    lead_card = (trickp >> (7 * leader)) & 127 if played > 0 else -1
    legal = _legal(hand, lead_card, trump)
    other = live & ~hand

    cards = np.empty(16, dtype=np.int64)
    cls = np.empty(16, dtype=np.int64)
    n = _partition(legal, trump, br, other, cards, cls)

    # Class representatives (lowest rank in class), ordered by rank
    # descending for the search.
    reps = np.empty(16, dtype=np.int64)
    nr = 0
    prev_cls = -1
    for i in range(n):
        if cls[i] != prev_cls:
            prev_cls = cls[i]
            reps[nr] = cards[i]
            nr += 1
    for i in range(1, nr):                      # insertion sort, desc rank
        x = reps[i]
        j = i - 1
        while j >= 0 and _RANK[trump, reps[j]] < _RANK[trump, x]:
            reps[j + 1] = reps[j]
            j -= 1
        reps[j + 1] = x

    maximizing = (seat & 1) == 0
    a0 = alpha
    b0 = beta
    best = -_BIG if maximizing else _BIG
    for i in range(nr):
        c = reps[i]
        # --- play card c (keep in sync with _child) ---
        bit = 1 << c
        nh0, nh1, nh2, nh3 = h0, h1, h2, h3
        if seat == 0:
            nh0 = h0 & ~bit
        elif seat == 1:
            nh1 = h1 & ~bit
        elif seat == 2:
            nh2 = h2 & ~bit
        else:
            nh3 = h3 & ~bit
        nbr = br
        nbp = bp
        if ((_TRUMPMASK[trump] >> c) & 1) != 0:
            r = _RANK[trump, c]
            if r > nbr:
                nbr = r
                nbp = seat & 1
        ntrickp = (trickp & ~(127 << (7 * seat))) | (c << (7 * seat))
        if played + 1 == 4:
            w = _winner(ntrickp, leader, trump)
            nns = ns + (1 if (w & 1) == 0 else 0)
            nlive = live
            for s in range(4):
                cc = (ntrickp >> (7 * s)) & 127
                nlive &= ~(1 << cc)
            v = _search(nh0, nh1, nh2, nh3, _EMPTYP, w, nns, nbr, nbp,
                        nlive, alpha, beta, trump, bid_team, bid_kind,
                        payoff_raw, total, tt)
        else:
            v = _search(nh0, nh1, nh2, nh3, ntrickp, leader, ns, nbr,
                        nbp, live, alpha, beta, trump, bid_team,
                        bid_kind, payoff_raw, total, tt)
        # --- end transition ---
        if maximizing:
            if v > best:
                best = v
            if best > alpha:
                alpha = best
        else:
            if v < best:
                best = v
            if best < beta:
                beta = best
        if alpha >= beta:
            break

    if best <= a0:
        tt[key] = (best << 2) | 2               # upper bound
    elif best >= b0:
        tt[key] = (best << 2) | 1               # lower bound
    else:
        tt[key] = (best << 2)                   # exact
    return best


@njit(cache=False)
def _child(h0, h1, h2, h3, trickp, leader, seat, c, ns, br, bp, live,
           trump, bid_team, bid_kind, payoff_raw, total, tt):
    """Full-window value after `seat` plays card `c` (root helper;
    transition kept in sync with the inlined copy in _search)."""
    bit = 1 << c
    if seat == 0:
        h0 = h0 & ~bit
    elif seat == 1:
        h1 = h1 & ~bit
    elif seat == 2:
        h2 = h2 & ~bit
    else:
        h3 = h3 & ~bit
    if ((_TRUMPMASK[trump] >> c) & 1) != 0:
        r = _RANK[trump, c]
        if r > br:
            br = r
            bp = seat & 1
    ntrickp = (trickp & ~(127 << (7 * seat))) | (c << (7 * seat))
    played = 0
    for s in range(4):
        if ((ntrickp >> (7 * s)) & 127) != 127:
            played += 1
    if played == 4:
        w = _winner(ntrickp, leader, trump)
        nns = ns + (1 if (w & 1) == 0 else 0)
        nlive = live
        for s in range(4):
            cc = (ntrickp >> (7 * s)) & 127
            nlive &= ~(1 << cc)
        return _search(h0, h1, h2, h3, _EMPTYP, w, nns, br, bp, nlive,
                       -_BIG, _BIG, trump, bid_team, bid_kind,
                       payoff_raw, total, tt)
    return _search(h0, h1, h2, h3, ntrickp, leader, ns, br, bp, live,
                   -_BIG, _BIG, trump, bid_team, bid_kind,
                   payoff_raw, total, tt)


@njit(cache=False)
def _root(h0, h1, h2, h3, trickp, leader, ns, br, bp, trump, bid_team,
          bid_kind, payoff_raw, total, tt, out_cards, out_vals):
    """Exact value per legal CARD of the seat to move (class members
    share their representative's full-window value, exactly like the
    reference root_values). Fills out_cards/out_vals, returns n."""
    live = h0 | h1 | h2 | h3
    played = 0
    for s in range(4):
        c = (trickp >> (7 * s)) & 127
        if c != 127:
            live |= 1 << c
            played += 1
    seat = (leader + played) & 3
    if seat == 0:
        hand = h0
    elif seat == 1:
        hand = h1
    elif seat == 2:
        hand = h2
    else:
        hand = h3
    lead_card = (trickp >> (7 * leader)) & 127 if played > 0 else -1
    legal = _legal(hand, lead_card, trump)
    other = live & ~hand

    cards = np.empty(16, dtype=np.int64)
    cls = np.empty(16, dtype=np.int64)
    n = _partition(legal, trump, br, other, cards, cls)
    prev_cls = -1
    v = 0
    for i in range(n):
        if cls[i] != prev_cls:
            prev_cls = cls[i]
            v = _child(h0, h1, h2, h3, trickp, leader, seat, cards[i],
                       ns, br, bp, live, trump, bid_team, bid_kind,
                       payoff_raw, total, tt)
        out_cards[i] = cards[i]
        out_vals[i] = v
    return n


# ---------------------------------------------------------------------------
# Wrapper — reference-compatible API
# ---------------------------------------------------------------------------

class FastDDSolver:
    """Drop-in for DDSolver on the PIMC-DDS hot path. Restrictions:
    opponent='minimax' and reduce=True only (asserted)."""

    def __init__(self, trump, bid_team, bid_kind,
                 opponent='minimax', payoff='delta', reduce=True):
        assert trump in (0, 1, 2, 3)
        assert bid_team in (0, 1)
        assert bid_kind in _BID_VALUE
        assert opponent == 'minimax', 'FastDDSolver: minimax only'
        assert payoff in ('delta', 'raw')
        assert reduce, 'FastDDSolver: reduce=True only'
        self._trump = trump
        self._bid_team = bid_team
        self._bid_kind = bid_kind
        self._payoff_raw = 1 if payoff == 'raw' else 0
        self._tt = Dict.empty(_KEY_T, types.int64)
        self._total_tricks = None
        self.rb_fallbacks = 0                   # API parity; always 0

    # -- public API ---------------------------------------------------------

    def solve(self, hands, leader, trick=_EMPTY_TRICK,
              ns_tricks=0, ew_tricks=0, best_rank=-1, best_par=-1):
        self._prepare(hands, trick, ns_tricks, ew_tricks)
        h0, h1, h2, h3 = self._masks(hands)
        trickp, live = self._pack(hands, trick)
        return int(_search(h0, h1, h2, h3, trickp, leader, ns_tricks,
                           best_rank, best_par, live, -_BIG, _BIG,
                           self._trump, self._bid_team, self._bid_kind,
                           self._payoff_raw, self._total_tricks,
                           self._tt))

    def root_values(self, hands, leader, trick=_EMPTY_TRICK,
                    ns_tricks=0, ew_tricks=0, best_rank=-1, best_par=-1):
        self._prepare(hands, trick, ns_tricks, ew_tricks)
        h0, h1, h2, h3 = self._masks(hands)
        trickp, _ = self._pack(hands, trick)
        out_cards = np.empty(16, dtype=np.int64)
        out_vals = np.empty(16, dtype=np.int64)
        n = _root(h0, h1, h2, h3, trickp, leader, ns_tricks,
                  best_rank, best_par, self._trump, self._bid_team,
                  self._bid_kind, self._payoff_raw, self._total_tricks,
                  self._tt, out_cards, out_vals)
        played = sum(1 for c in trick if c >= 0)
        seat = (leader + played) % 4
        idx = {card: i for i, card in enumerate(hands[seat])}
        return {idx[int(out_cards[k])]: int(out_vals[k])
                for k in range(n)}

    def best_move(self, *args, **kwargs):
        vals = self.root_values(*args, **kwargs)
        leader = args[1]
        trick = kwargs.get('trick', args[2] if len(args) > 2
                           else _EMPTY_TRICK)
        played = sum(1 for c in trick if c >= 0)
        seat = (leader + played) % 4
        sign = 1 if seat % 2 == 0 else -1
        return max(sorted(vals), key=lambda i: sign * vals[i])

    # -- internals ----------------------------------------------------------

    def _prepare(self, hands, trick, ns_tricks, ew_tricks):
        played = sum(1 for c in trick if c >= 0)
        remaining = (sum(len(h) for h in hands) + played) // 4
        total = ns_tricks + ew_tricks + remaining
        if self._total_tricks is None:
            self._total_tricks = total
        elif self._total_tricks != total:
            self._tt = Dict.empty(_KEY_T, types.int64)
            self._total_tricks = total

    @staticmethod
    def _masks(hands):
        out = []
        for h in hands:
            m = 0
            for c in h:
                m |= 1 << c
            out.append(m)
        return out

    @staticmethod
    def _pack(hands, trick):
        trickp = 0
        live = 0
        for h in hands:
            for c in h:
                live |= 1 << c
        for s in range(4):
            c = trick[s]
            trickp |= (c if c >= 0 else 127) << (7 * s)
            if c >= 0:
                live |= 1 << c
        return trickp, live
