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
