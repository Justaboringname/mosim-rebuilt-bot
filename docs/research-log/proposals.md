# RL proposals: sim prediction vs real A/B (pre-registered)

Rule, written before each run: real interleaved 32 vs 32 episodes (candidate vs the zero residual through the same
ResidualAgent path, leash {"neutral":0.333,"conveyor":0.333,"harvest":0.5,"shoot":0.5}, decide_every 5). Accept only if
real Δ > 2 SE of the difference.

## c50: runs/rl_sim/s1/cand-0050.pt (PPO iteration 50)
- Sim, 48 paired seeds: +37.7 ± 3.2. By window: 105-80 +9.5, 55-30 +22.2, 80-55 +2.3, 30-0 +1.9.
- Behaviour: tracks the ghost ~0.25 s ahead (warp +0.2..+0.3) in 125-110, 100-85, 65-60 and 50-40. Small lateral
  shifts (+0.4 m at 125, +0.2 m at 75 and 20). No ball chasing.
- **Real A/B (runs/ghost/ab_c50, 32 v 32): 978.8 vs 1010.8, Δ = −32.0 ± 6.1 → REJECT.** Sim predicted +37.7 (a 70-point
  sim-to-real gap).
  - By window (real Δ | sim Δ): 105-80 −5.2 | +9.5; 80-55 −6.0 | +2.3; 55-30 −15.7 | +22.2; 30-0 −5.4 | +1.9.
  - Real unsticks 1.38/ep vs 0.06 for control. 37 of 44 backoffs are at the red bump/hub face, x ≈ −2.6, z ≈ −1.0,
    t 72-68. There the policy adds +0.1..+0.25 m of lateral offset toward the structure, and the robot jams
    pushing balls against the face.
  - MiniSim has no such jam. RL found exactly the known sim gap ("the robot never gets stuck against structures").
  - Split: 21/32 candidate episodes jammed at the red face (Δ −37.8). The 11 without that jam also lost (Δ −20.8 ± 5.9;
    no-backoff episodes −28.5 ± 3.1; mostly 80-55 and 30-0). Being ahead of the ghost is a direction the sim is
    optimistic about too. Sim unsticks were 0.0 for both candidate and control.

## Changes for s2 (fresh run)
- Warp removed (act_dims 1, lateral only).
- Clearance constraint in ResidualDecoder: the displaced target may not be closer to any structure footprint
  (robot_obstacles.json + bumps) than the ghost itself, capped at 1.2 m. Shared by sim and real.
- Sim pre-screen before any real A/B: reject if the candidate spends more time within 0.8 m of structures than
  control on the same seeds.

## STOPPING RULE (written 2026-09-26, before s2)
If the next constrained candidate also fails the real 32 v 32, stop the sim-RL loop for this route. That would be the
sixth real failure of a local deviation. Report to the user that the remaining headroom needs new human demos; do not
start another iteration.

## d105: runs/rl_sim/s2/cand-0105.pt (s2 iteration 105; lateral only, clearance constraint)
- Sim, 48 paired seeds: +51.6 ± 2.9. By window: 105-80 +12.1, 80-55 +3.0, 55-30 +21.4, 30-0 +14.9 (30-0 is the
  suspect window).
- Pre-screen passed: time within 0.8 m of structures 0.499 vs 0.527 for control; unsticks 0 vs 0; |lat| mean 0.05 m,
  p90 0.14 m.
- **Real A/B (runs/ghost/ab_d105, 32 v 32): 975.2 vs 1008.2, Δ = −32.9 ± 7.0 → REJECT.** Sim predicted +51.6.
  - By window (real Δ | sim Δ): 160-140 +2.9 | 0; 140-130 −3.1 | −0.7; 105-80 −3.9 | +12.1; 80-55 −6.2 | +3.0;
    55-30 −14.6 | +21.4; 30-0 −8.4 | +14.9.
  - No jam this time (unsticks 0.47 vs 0.34). Mean |lateral| is only 0.05 m (p90 0.14 m), yet the loss lands in the
    same windows as c50: the second dead window and everything after it. The sim gets the sign of these small
    residual effects wrong there.

## STOPPED (stopping rule triggered, 2026-09-26)
Two constrained/unconstrained RL candidates failed real A/B by about −32. That makes six real failures of local
deviations from the human route (lateral residual, CMA segment shifts, splicing, seeker, bump1, RL c50/d105). The sim
is a good ranker for the plain route but predicts the wrong sign for residuals around it, even 5 cm ones. More sim-RL
iterations on this route are not justified. Next lever: new human demos (different routes), per STATUS.md.

## 2026-09-26 23:58 — CMA c5 (timing space) acceptance, written before its validation batch
- c5 = CMA-ES over BASE_SPACE + yaw_dt + btn_dt, fixed frame step (stepsPerFrame=2, timeScale 1), k=3, 20 gens.
- CMA's reported best is biased upward (best-of-population with k=3), so it is NOT the result.
- Validation: the c5 incumbent vs the current default (yaw_dt=0.2, everything else GHOST_SPEC), interleaved 32 v 32 on 8
  fixed-step instances. Accept only if mean Δ > 2 SE (SE of the difference). Otherwise keep the default, no retuning.
- 2026-09-27 02:25 RESULT (runs/ghost/c5val, 32 v 32, fixed step): default 1026.7 ± 4.0, c5 favorite 1032.2 ± 3.7, Δ +5.5 ± 5.4 (1.0 SE) → REJECT per the rule above. Default (yaw_dt=0.2) stays; no retuning.

## 2026-09-27 02:50 — collection aids IF / YR / SG (intake analysis workflow), written before running
- Arms (GhostPolicy params, default yaw_dt=0.2 base): A/A control {}, {"if_on":1}, {"yr_max":1.5}, {"sg_on":1}, combined.
- E1 screen (planner.py screen, 8 mains x 6 reps per arm, metric = window value: dead -> owned at t0-dur, active ->
  blue+held+stock+hubair) at t0/dur 139/9, 129/24, 104.5/24.5, 79/24, 54/24.
  An arm passes only if: summed Δ over the 5 t0 > 2·SE_sum, no single t0 with Δ < −2·SE, unsticks not higher.
  The A/A arm must be within 2 SE of 0 at every t0, otherwise the harness is biased and nothing is read.
- E2: best passing arm vs default, interleaved 128 v 128 full matches (fixed step). Accept only if Δ > 2 SE and Δ ≥ +5,
  unsticks/resyncs +0.2/match at most. Otherwise keep the default.
- 2026-09-27 03:25 E1 RESULT (runs/planner/e1_screen.log; SE below are per-arm and understate the paired SE by ~√2 —
  the A/A arm read +1.9 ± 1.0 at 104.5 and −4.0 ± 1.4 at 54):
  | arm | 139/9 | 104.5/24.5 | 79/24 | 54/24 | (129/24 lost to an instance crash) |
  | IF  | +1.8 | −2.6 | −20.1 | −7.3 | |   | YR | −0.3 | +0.7 | −9.0 | −4.7 | |   | SG | +0.4 | +1.9 | −1.9 | −1.8 | |
  | IF+YR+SG | −0.7 | −2.4 | −26.1 | −12.2 | |
  → all REJECTED (each has a t0 with Δ < −2 SE, or no positive sum). No E2. The human's crab while collecting is
  deliberate: forcing intake-first costs ~20 owned per dead window.

## 2026-09-27 04:05 — LBS (ball-aware local steering, user feedback "扫空气 / 撞散球"), written before running
- Arms: {}, {"lbs_on":1}, {"lbs_on":1,"lbs_dmax":0.5}, {"lbs_on":1,"lbs_mu":1.0}. Screen at 158.5/8, 129/24, 104.5/24.5,
  79/24, 54/24 (8 mains x 5 reps, round-robin, window value, paired-by-main SE).
- Pass: summed paired Δ > 2·SE_sum and no t0 with paired Δ < −2·SE. Then 128 v 128 full matches (Δ > 2 SE, ≥ +5).
- 2026-09-27 04:40 LBS RESULT: 129/24 → lbs −34.2 ± 3.2, dmax0.5 −20.9 ± 2.3, mu1 −34.5 ± 2.9 (paired-by-main). Diagnostic:
  lateral-only −40.7 ± 5.1, lateral ±0.3 −12.2 ± 1.8, heading-only −3.6 ± 2.7. Full matches (lbsdiag, 8 v 8, lateral only):
  881 vs 1014 (auto 135 vs 159; 140-134 sweep 40+18 vs 66+41 intakes). REJECTED: the 1 s greedy pulls the robot off the
  line that leads to the big piles. Precise opening (deploy at spawn, err 0.11 m at 158.5) did not change the score
  (clean episodes 1022 vs 1023) and jams the slide 7/16 → rejected too.

## 2026-09-27 04:45 — REG (pile registration: follow the human's path translated onto where its swept balls are now)
- Arms {}, {"reg_on":1}, {"reg_on":1,"reg_dmax":0.5}, {"reg_on":1,"reg_h":3.0}. Screen 129/24 first, then 79/24, 104.5/24.5,
  54/24, 157/7. Same pass rule as LBS (paired-by-main SE; no t0 below −2 SE; sum > 2 SE), then 128 v 128.
- 2026-09-27 05:20 REG RESULT (129/24, paired-by-main): reg −24.0 ± 3.5, dmax0.5 −14.6 ± 2.1, reg_h3 −25.3 ± 4.0 → all
  REJECTED at the first t0 (below −2 SE); the other t0 were not run.
- Diagnosis (lbsdiag floor counts, control vs lateral-only LBS): unswept neutral floor balls at t=105: 77 vs 128; at t=55:
  82 vs 105; blue stock at 105: 276 vs 246. The human's dead-window route is a coverage pattern (adjacent strips). Local
  shifts toward "more balls" overlap the next strip and open gaps, so more balls are left unswept. Pass success
  (0.89 vs 0.86), crab and speed are unchanged. → any adaptivity must re-plan the whole window's coverage, not shift locally.

## 2026-09-27 05:40 — CR (whole-window coverage re-planning), kill switches written before computing
- Estimator E: static floor field at t0 (balls with x ≤ 4.25, i.e. not already in blue stock); robot path sampled at
  0.1 s; 4414 footprint (fwd −0.45..0.62 m, |lat| ≤ 0.555 m). A ball's first entry with fwd ≥ 0.35 is captured with
  p(|lat|) from the intake table × held factor; any other entry is a push. **Pushed balls count as lost.** Intake off
  or slide not deployed → every entry is a push. Held = held0 + captures − pass drain (measured from traces).
- Fate check done first (runs/ghost/lbsdiag, 130→105): of the neutral balls left unswept, control 23 never approached /
  38 approached-and-moved / 16 moved-without-approach; LBS 59 / 51 / 36. So LBS's extra loss is mostly coverage gaps
  (+36), then chain-moved (+20) and pushed (+13). 80→55: 19/45/12 vs 47/54/28.
- K1 (back-test): on the 8 LBS (arm 1) fields at t0 = 130 and 80, E(LBS actual path) − E(ghost path) must be < 0 by
  more than 2 SE (the real engine gave −34 to −41). Also report E(actual) vs actual non-blue intakes on all 16 (r).
  If K1 fails → stop CR and report "estimator fooled".
- K2 (headroom): a beam-search plan over the ghost's neutral-zone segment (same entry/exit points and times, speed ≤
  the ghost's, heading-rate limit) must beat E(ghost path) by ≥ +40 predicted captures per dead window, averaged over
  16 fields × 2 windows. Otherwise stop CR (≈3× optimism → real gain ≲ 13/window).
- If both pass: real-engine MPC with {ghost, CR plan} at dead-window entry (screen 129/24 and 79/24, 8 mains × 5 reps,
  paired-by-main, same pass rule as LBS), then 128 v 128 full matches (Δ > 2 SE and ≥ +5).
- 2026-09-27 06:30 CR RESULTS (ad-hoc analysis scripts, not published; lbsdiag fields, drain 10.9/s measured):
  - K1 PASS: E(LBS actual) − E(ghost) = −23.7 ± 5.2 (130: −28.4, 80: −18.9); control −6.2 / −3.7 (tracking error).
    r(E(actual), actual non-stock intakes) = 0.70; E ≈ 0.62 × actual (pushes counted lost).
  - K2 FAIL: beam plan − ghost, per dead window (16 fields each): W=96 ≈ +14, W=512 +19.9 (130: +27.0, 80: +12.7),
    W=2048 +25.0 (130: +33.3 ± 5.0, 80: +16.7 ± 3.9) < +40. Not converged in W, but deeper search of a static field
    also means more optimism (E(plan) is flat 243–280 while E(ghost) spans 166–255), so the trend is not headroom.
  - Lever check: E(ghost with heading = travel direction) − E(ghost) = −4.0 (130) / −9.6 (80), same sign as the real
    IF arm (−20 at 79/24), so E is not fooled on heading. CR is stopped by the rule — untested in the real engine,
    not disproven. bump1 has no dense data (no second gain-case back-test).
  - Human reference (same route, own layout): neutral floor 312 → 46 (85% cleared) in 130→105, 362 → 69 (81%) in 80→55;
    bot control 338 → 84 (75%), 362 → 92 (75%). The ~35-ball/window gap is the route fitting the human's layout.

## 2026-09-27 (user: "OK继续") — CR real-engine screen, past the K2 kill by the user's explicit choice; written before running
- Implementation: mosimrl/coverage.py (E + beam W=1024 at the human's sweep speed 1.40 m/s, full stick, heading = travel
  direction) spliced into the ghost on entering each dead window's neutral stretch (GhostPolicy cr_on / cr_min).
- Arms: {} ; {"cr_on":1,"cr_min":0} (use the plan whenever E(plan) ≥ E(ghost)) ; {"cr_on":1,"cr_min":15} (selective).
- Screen 129/24 and 79/24, 8 mains × 5 reps, round-robin, window value (owned at 105 / 55), paired-by-main SE.
- Pass: summed paired Δ over the 2 t0 > 2·SE_sum and no t0 with Δ < −2·SE, unsticks not higher. Then 128 v 128 full
  matches, Δ > 2 SE and ≥ +5. Otherwise CR is dropped for good.
- 2026-09-27 CR v1 RESULT: screen 129/24 (8 mains × 5 reps, paired-by-main): cr_min0 −105.7 ± 6.1, cr_min15 −89.1 ± 13.3
  → FAIL (79/24 stopped). Full matches (runs/ghost/crdiag, 4 v 4): 886 vs 1016.
  Mechanism (crdiag): turning kills passing. Launch rate with held ≥ 15 vs robot |yaw rate| (control): <20°/s 21.6/s,
  20–60 15.9/s, 60–120 7.8/s, >120 2.0/s. CR's plan turns all the time (median 87°/s vs 18°/s), so launches fell,
  the hopper saturated (held max 99–100 vs 90–97), failed passes doubled (35–52 vs 15–21 per window), owned gain in
  130→105 +164..+200 vs +290..+326. E used a constant drain, so it could not see this. The human's straight strips
  are a passing constraint as much as a coverage pattern.
- E v2 (turn-dependent pass drain, AutoPass-gated; ad-hoc script, not published): LBS back-test still right (−23.9 ± 5.2),
  but the CR v1 plans still score as a GAIN under E v2 (+9.6 ± 6.7 at 128.2, +8.0 ± 6.4 at 77.9) while the real engine
  gave −106. → E is fooled on the gain case; CR is dropped for good (pre-registered rule).
  Where the balls went (crdiag, 128→105, per match): neutral→blue stock 214 (control) vs 103 (CR); neutral→held 75 vs
  77. Held (mean over matches) sat at 89–94 from 121 s on under CR vs 40–90 oscillating in control: straight driving
  empties the hopper by passing; constant turning cannot, the hopper saturates and intake events stop turning into
  captures (CR's intake-event count was only ~35 lower, but it removed ~110 fewer balls from the neutral zone).

## 2026-09-27 — YS (yaw-rate cap while shooting), written before running
- Evidence (control, crdiag + lbsdiag arm 0, 12 matches; AutoShoot, held ≥ 20, x > 3.9): launches/s by |yaw rate|
  <20°/s 16.7 (55% of shooting time), 20–60 16.1, 60–120 13.7 (14%), >120 4.0 (4%); turret |err| 0.22 → 0.51 → 1.27.
  Mean 15.5/s. Cost risk: heading lags the ghost, so stock intake while shooting can drop.
- Arms: {} ; {"ys_max":60} ; {"ys_max":30}. Screen 104.5/24.5 and 54/24 (active value = blue + held + stock + hubair),
  8 mains × 5 reps, paired-by-main. Pass: summed Δ > 2·SE_sum, no t0 with Δ < −2·SE. Then 128 v 128 full matches,
  Δ > 2 SE and ≥ +5.
- YS screen RESULT (paired-by-main): 104.5/24.5 ys60 +0.33 ± 1.34, ys30 +2.05 ± 1.23; 54/24 ys60 +6.58 ± 2.38,
  ys30 +5.15 ± 2.79. Sums: ys60 +6.9 ± 2.7, ys30 +7.2 ± 3.1 → both PASS the screen rule. Caveat: the gain is in owned
  at the window end (ys60 +5 / +12, ys30 +5 / +23), while scored is slightly lower (−4 / −5, ys30 −2 / −18), i.e. not
  the "more launches" mechanism that motivated it. Full matches decide: ys60 (higher z, fewer unsticks) vs default,
  interleaved 128 v 128 (runs/ghost/ys60), accept only if Δ > 2 SE and ≥ +5.

## 2026-09-27 — OL: real-engine option learning (user: "RL 可以成功，只是数据不够"), written before running
Design: a design review (three independent designs, then a merged plan). Code: mosimrl/optlearn.py (collector, OptionAgent),
mosimrl/optfit.py (analysis). Route untouched by construction.
- Event A (turn while shooting): base 0.5 / placebo 0.1 / cap(|rot| ≤ 0.30, released on recovery/3 s/feed stop, ramp
  back 2 units/s) 0.4. Event C (t = 60, hold AutoPass to 55, abort at held ≥ 90): base 0.5 / placebo 0.1 / hold 0.4.
- Mains (2) play the default and snapshot at 106 (W2) and 62 (W3); 10 rollouts per snapshot on 14 planners: 2 pure
  base + 8 randomized. W2 value = blue(77) + 0.35·owned(77) (β 0.2 / 0.7 sensitivity; 2 of 10 run to the buzzer);
  W3 value = final blue incl. post-buzzer. No owned term anywhere at λ = 1 (the YS proxy hack).
- K0 (≥ 30 mains): base-pair diff and placebo θ within ±2 SE in each window; per-rollout σ ≤ 1.3× planned (W2 ~7,
  W3 ~13); cap cuts time > 120°/s per event by ≥ 80% vs base, yaw_err recovers < 10° within 3 s in ≥ 90%; the implied
  "cap every trigger" effect in W3 must not be positive (ys60 lost in 55-30). Any failure → fix harness, no learning.
- K1 (≥ 200 mains): keep a family only if some option's per-window effect has a 95% lower bound > 0, or grouped-CV
  heterogeneity gain > 0 with bootstrap p < 0.05. Otherwise stop it.
- K2 (≤ 400 mains): cross-fitted value of the LCB policy ≥ +5/match with lower bound > +1, else stop.
- K3: interleaved 128 v 128 native full matches: Δ > 2 SE and ≥ +5, realized ≥ 0.4 × K2 prediction, unsticks +0.2 max.
  No retuning after a fail. If both families die: "no state-conditional non-path gain ≥ +5" and RL on this route stops.
- Expected (design review): 0 to +8/match, most likely 0. 1200 not plausible from these levers.
- 2026-09-27 YS FULL-MATCH RESULT (runs/ghost/ys60, 128 v 128, 16 instances): see the line below; ys60 1003.9 ± 2.0 →
  REJECTED. The screen's +6.9 came from valuing owned balls at the window end at 1.0 (proxy hack): scored was lower.
  control: arm 0 {}: n=128 mean 1022.2 sd 22.6 se 2.0  auto mean 161.9 min 137
- Throughput (mosimrl/bench.py, decisions/s): default 16 inst 274; -job-worker-count 0: 295; jwc 1: 273; jwc 2: 266;
  lite 289; jwc0+lite 16 inst 324, 20 inst 348, 24 inst 346. Python is 0.4 ms of a 56 ms step (not the bottleneck).
  OL runs with jwc0 on 20 instances (2 mains + 18 planners). Added to K0: the mains' mean score (default policy,
  jwc0) must be within 2 SE of the ys60 control 1022.2 ± 2.0, else stop (the worker count changed the game).
- OL K0 at 30 mains (runs/rl_opt/ol1): mains' mean 1023.9 (jwc0 does not change the game, vs 1022.2 ± 2.0) ✓;
  base-pair W2 −0.24 ± 1.43, W3 +1.36 ± 3.30 ✓; σ W2 5.4, W3 12.2 ✓; cap cuts > 120°/s time 0.233 → 0.003 s/event ✓;
  known result: W2 full-horizon subset cap −4.46 ± 2.59/event (ys60 sign) ✓, W3 all-cap −0.9 ± 2.8 (not positive) ✓.
  Misses: yaw_err back < 10° within 3 s under cap 0.80 (< 0.90 planned; a property of the option, not a harness
  bug — kept unchanged so data pool); placebo W2 +1.37 ± 0.67 (2.04 SE; placebo is code-identical to base, 1 of 4
  placebo checks) → re-check at 60 mains, investigate the harness if still > 2 SE.
  Harness bug fixed: a rollout that reached the buzzer left the planner released, so the next restore failed and the
  instance was relaunched (~20 per planner). Now reset after such rollouts. Collection restarted (same run, ids continue).
- OL at 60 mains: throughput 97 mains/h after rebalancing to 3 mains + 17 planners (2 mains left ~9 planners idle).
  Placebo re-check W2 +0.73 ± 0.50 ✓ (K0 passes except cap recovery 0.79). Interim: cap W2 −0.41 ± 0.25/event,
  full-horizon subset −5.10 ± 2.25/event, W3 −0.57 ± 0.50; hold −0.04 ± 1.31. Heterogeneity dry run (50 perms):
  W2 A CV gain +0.24 p 0.06, others ≤ 0. K1 is read once at 200 mains with 200 permutations (mosimrl/optfit.k1).
- 2026-09-27 determinism check (mosimrl/detcheck.py): the same blob restored into two instances and driven with
  identical actions diverges at step 1 (robot 8 mm, balls 1 mm with -job-worker-count 0; 58 mm / 12 mm default) and
  fully decorrelates within 15 s (robot yaw differs ~190°, balls up to 11 m apart); same-instance repeat diverges too.
  → No exact replay: TAS-style best-of-K search (pick the luckiest rollout, then re-execute it) is not available.
- 2026-09-27 OL K1/K2 RESULT (203 mains, 3997 rollouts, runs/rl_opt/ol1, analysis mosimrl/optfit.py):
  - Mean effects: cap W2 −0.54 ± 0.13/event (all-cap −3.0 ± 0.7/window), W3 −0.44 ± 0.34 (−2.2 ± 1.7); hold +0.0 ± 0.8.
  - K0 re-check: base pairs by submission order W2 +0.25 ± 0.61, W3 −1.46 ± 1.40 ✓ (the −3.7 ± 1.4 read earlier paired
    by completion order, which selects on rollout speed — an analysis artifact). Mains' mean 1019.4 (n = 201).
  - K1: heterogeneity for cap significant (W2 CV gain +0.89, W3 +1.96, permutation p = 0.005 each) → family A kept;
    hold: mean LB < 0 and p = 0.32 → family C STOPPED.
  - K2: cross-fitted LCB policy caps 0.36 of ~5 events/rollout; value vs base W2 −0.26 ± 0.22, W3 −0.35 ± 0.46,
    total −0.60 ± 0.51/match (needs ≥ +5, LB > +1) → FAIL. φ predicts where capping hurts more or less, not where it
    helps. Stopped at 203 mains (400 could not reach +5: 11 SE away). Per the rule: no state-conditional non-path gain
    ≥ +5 at ~4000 real-engine rollouts; RL on these levers stops.

## Supplementary measurements cited in the README (collected 2026-09-27)
- Intake (4414), from recorded traces plus the robot geometry. The slide must be fully out (≥ 0.29 m).
  - Capture probability by the ball's lateral offset from the centreline:

    | lateral offset | capture |
    |---|---|
    | ≤ 0.30 m | 0.95–0.98 |
    | 0.30–0.35 m | ≈ 0.80 |
    | 0.35–0.40 m | ≈ 0.6 |
    | 0.40–0.45 m | 0.35–0.48 |
    | > 0.45 m (to 0.555 m) | ≈ 0.15 |

  - Balls first touched at a front corner are captured 0.38 of the time overall.
  - Holding AutoShoot or AutoPass without Intake retracts the slide to about 0.19–0.25 m. About 70% of the balls met in that state are pushed away.
  - With ≥ 81 balls held, capture drops to 0.80; with ≥ 96, to 0.56. Speed has no measurable effect.
- Fate of the neutral balls in the control, 130 → 105, per match (80 → 55 in parentheses):
  - 38 (45) balls touched but not taken;
  - 16 (12) moved by chain pushes;
  - 23 (19) never approached.
- Value of one owned ball (held + stock) at a window end, in later points:
  - At 105 s: 0.59–0.83 interventional (arm differences in crdiag, bump1, lbsdiag, ab_d105); 0.50–0.75 cross-sectional.
  - At 80 s: 0.24–0.43 (regression of the final score).
  - At 55 s: 0.56 interventional (crdiag). Split by component: held 0.73 ± 0.15, stock 0.39 ± 0.11.
  - At 30 s: −0.14 to +0.04.
- Human 1118 vs bot 1084 (ys60 ep144), taking the last sample at or above each boundary:

  | segment | human | bot |
  |---|---|---|
  | 160–130 | 207 | 214 |
  | 130–105 | 46 | 38 |
  | 105–80 | 286 | 272 |
  | 80–55 | 44 | 34 |
  | 55–30 | 278 | 269 |
  | 30–0 | 233 | 225 |
  | after the buzzer | 24 | 32 |

  - Held at 130: 39 vs 17.
  - Neutral floor cleared in 130 → 105 / 80 → 55: human 85% / 81%; this bot match 82% / 75%; lbsdiag control mean 75% / 75%.
  - Owned at 105 / 55 in the ys60 control mean (128 matches): 358 / 337 vs the human's 390 / 360.
- run_ghost's per-window table misses the post-buzzer points. Final score minus the last t > 0 sample in the ys60 control: mean 24, 5–95% range 8–35.
- Human runs recorded as full demos: 1118, 1085 and 1097. The 1097 run switches to the mirrored side after t = 105.
- Pre-registration covers the experiments from c50 onward. Earlier ones were not pre-registered: greedy/residual, CMA segment shifts, splice, seeker, bump1, the planner MPC runs m1–m3 (single matches) and intake pulsing. Two logged exceptions:
  - CR went to the real engine after its K2 kill, at the user's request.
  - OL's K0 recovery-rate criterion (≥ 0.90) read 0.80 and was let through.
