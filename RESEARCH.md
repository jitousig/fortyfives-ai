# RESEARCH — live plan / backlog (Track A)

This is the **plan**, not the record. Durable conclusions + the "why"
live in the auto-memory (`project_bidding_arc`,
`project_play_phase_ceiling`, `project_goals`,
`project_env_topology`). This file **links to** those; it does not
restate them (two docs that restate each other drift — that failure
mode already cost this project days). Keep it short; prune done items.

## Where things stand (pointers, not restated)
- Bidding: EV bid-level is **not a confirmed win** on the trustworthy
  engine (net, n=2000 ×2, CIs straddle 0). EV-trump falsified. Net
  yardstick: random ≈ −28, rule = 0.00. → see `project_bidding_arc`.
- Play: PIMC v3 robustly > rule-based, but **"play capped" is
  UNMEASURED** (no oracle). → see `project_play_phase_ceiling`.
- Engine on this branch == `main`'s (all 4 fixes); instrument
  trustworthy (both canaries +0.0000).

## Active work — DDS oracle + PIMC-DDS (play first)  [handoff 2026-07-25]
Decision (with user): attack **PLAY first**; bar = provably at/near the
double-dummy ceiling AND beats every baseline (paired CI excludes 0,
n≥2000 ×≥2 seeds). Concrete execution of decision-gate #2 below. Durable
context + faithfulness facts: memory `project-dds-handoff`.

Build ONE exact **double-dummy solver (DDS)** — solve a deal with all 4
hands face-up via exact alpha-beta minimax → optimal our−opp. Tractable:
each seat holds ≤5 cards in play, so the tree is tiny (~instant/deal).
Reuse 3 ways: (1) **play oracle** = upper bound + PIMC-v3 optimality gap;
(2) **PIMC-DDS agent** = swap PIMC's rule-based rollout for an exact
per-world DDS solve (fair, superhuman-grade); (3) later, bidding oracle.

Faithfulness — DDS must replicate THIS engine (verified vs current code):
- Objective = our−opp under 5/trick + 5/high (matches `PIMCAgent._simulate`
  + `game.score_hand`). Constant-sum ⇒ NS seats 0/2 maximize, EW 1/3 min.
- "High trump" = seat that **played** the highest trump (`game.py`
  ~L742–746 via `wins_over`), not who holds it; A♥ always trump.
- Legality = `game.get_legal_plays` (L432): follow lead if able, but trump
  always legal (renege); trump-led ⇒ follow with trump except a top-3
  trump (5/J/A♥) outranking the led trump MAY be withheld. PIMC's
  `_rollout_legal` only approximates this — DDS must be exact ("lever 2").
- Trick winner = `game.wins_over` (L899) + `get_card_rank` (`card.py`).

Correctness gate: unit-test the standalone DDS (legality+trick+scoring)
against the REAL engine on random rollouts BEFORE trusting a number;
re-run the paired canary after eval wiring. The oracle cheats (needs true
hands from `env.game`) → add a dedicated eval path; `play_eval.
evaluate_paired` is the yardstick, `fortyfives_pimc.PIMCAgent` the shell
to fork. Short-lived branch off `reconcile-bidding-engine`; promote only
validated results to `main` via small PR.

Plan review (Fable 5, 2026-07-25) — three upgrades, verified vs code:
1. **Minimax DDS is NOT an upper bound on the yardstick.** avg_diff is
   vs a FIXED deterministic rule-based EW; paranoid minimax forgoes
   exploiting its mistakes, so a fair agent can exceed it → "PIMC ≈
   minimax oracle" would NOT prove "capped". The decision-gate ruler is
   a **best-response oracle**: same solver, EW nodes FORCED to the real
   `RuleBasedAgent._play_strategy` move (import the real code; validate
   predicted == actual on replayed hands). Keep minimax mode for
   PIMC-DDS's per-world solves (opponent unknown there).
2. **Bid-aware leaves.** The yardstick is NS game-point delta after
   `end_hand` bid adjustments (`game.py` L947–1005: fail → −bid, made
   30 → flat 60; 100+ pegging rule unreachable in eval, hands start
   0-0 — assert). Per-world argmax is unchanged (monotone transform)
   but PIMC's cross-world AVERAGE is not: E[delta] ≠ f(E[raw]) exactly
   at make-vs-set decisions. Leaves return delta directly — also a
   plausible independent edge for PIMC-DDS over v3.
3. **Test the search, not just the state machine:** (a) brute-force
   minimax vs alpha-beta+TT on random endgames; (b) engine-grounded
   best-response cross-check on a subsample (roll the REAL env over NS
   move trees; exact by construction, validation-only); (c) forced-move
   fidelity (predicted rule-based moves == actual).
Minor: pair oracle vs PIMC v3 per-hand on identical seeds (CI on the
gap itself); ablation PIMC-DDS with per-world rule-based EW model;
profile solves/sec before committing to n=2000×2; if PIMC-DDS
disappoints, pre-registered suspects = strategy fusion / non-locality.
Working branch: `dds-play-oracle` (off reconcile-bidding-engine).

## Decision gate (do this BEFORE any new method)
No "capped / near-optimal" conclusion for **either phase** without a
perfect-information **oracle upper bound**. Two diagnostics:

1. **Bidding oracle** — perfect-info bidder (sees all 4 hands+kitty,
   picks bid by exact rollout) vs rule-bidder, net, n≥2000 ×2 seeds.
   - oracle ≈ rule (small gap) ⇒ rule-based bidding is near-ceiling
     for this engine ⇒ stop tuning bidders; reconsider scope (Track B
     / engine scoring is a product decision — `project-scoring-rule`).
   - oracle ≫ rule ⇒ real headroom; the EV estimator is the problem ⇒
     pursue the levers below.
2. **Play oracle** — double-dummy (full-visibility) play agent vs
   PIMC v3 AND rule-based, via `play_eval`. Gives the play upper bound
   + the PIMC-v3 optimality gap. Required before "play capped".
   (Oracle = upper bound, not the imperfect-info optimum; still the
   bound we lack.)

## Estimator levers (only if an oracle shows headroom; priority order)
1. **Discard-count-constrained determinization** (PIMC play + EV
   rollouts). Under the held-constant rule-based discard (keep trump,
   ditch non-trump, replenish to 5 from deck), each seat's kept-trump
   count `K_i = 5 − replenish_i` is observable and **near-exact**.
   Sampler = true generative model: give seat `K_i` trump, fill the
   remaining `replenish_i` slots from the **full unseen pool (trump
   included)**. Tighter & lower-variance than uniform/void-only.
   Caveats: (a) verify `replenish_i` is in the obs the agent consumes;
   (b) bidder's count is kitty-offset (8→5), special-case it;
   (c) "near-exact" only while opponents use rule-based discard.
2. **Auction-conditioned determinization** — EV bid belief is
   currently uniform/auction-ignorant; condition hidden hands on
   bids/passes + position (dealer seat, who passed, standing bid,
   can-I-hold, pass-risks-redeal-vs-concede).
3. **Rollout-legality fidelity ("lever 2")** — PIMC deep playout uses
   approx (must-follow + trump-always), not exact renege; untested.
   May mask determinization gains → test on EV bidder first (faithful
   engine, no legality confound) or pair with lever 1.
4. **Strategic-opponent continuation** — rollouts currently play all
   seats rule-based (position-blind); a smarter continuation policy
   models the strategic value of position.
- Verify: EV `_rollout` valuation of a "pass" that leads to all-pass
  → redeal (possible silent gap).

## Method stance
NOT PPO (generic suggestion; does not address this project's
bottleneck — instrument/representation, not the learner; play is
range-limited). See `project_play_phase_ceiling` correction. If a
*learned* bidder is ever pursued: imperfect-info family (Deep CFR /
NFSP / search-augmented), gated behind an oracle showing headroom.

## Standing discipline (non-negotiable)
- Confirm every apparent win at **n≥2000 on ≥2 independent seeds, CI
  excluding 0**. Never trust small-N, single-seed, or Elo-ladder.
- Re-run the canary after ANY engine/eval/rule-based edit (rule-vs-
  rule MUST be +0.0000). One change per commit; single-variable A/B.
- Two-track: research never blocks/contaminates `main`. Core fixes
  flow main→reconcile (merge); promote only *validated* results
  reconcile→main via small PRs. Never wholesale-merge reconcile→main.
