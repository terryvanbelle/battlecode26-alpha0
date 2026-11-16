# RL estimator plan after M1: head-to-head antithetic ES ("zdpes")

Status: implementation-ready. Synthesised from 3 designs and 3 judge reports (November 14, 2026). Replaces the PPO-lite estimator as the M1/M2 workhorse. PPO, V(trace) and the trailer machinery are kept for diagnostics and for stage B.

## 0. Decision

**Winner: Design 2's core.**
- Antithetic ES on low-dimensional policy offsets: per-point output-bias offsets plus one log-temperature.
- Fitness comes from **head-to-head** games between θ+ and θ−.
- Each unit is seat-swapped on the same map and seed.
- **Block-coordinate**: each unit perturbs one point's block.
- **Thresholded-t sign step**: no Adam.

Why it wins:
- It is outcome-only, pure self-play and needs no Java changes.
- It needs no gradient through the saturated softmax.
- It is the only method with direct game evidence on M1. The head-to-head score on the sabotaged coordinate was 11/16 for +30 vs +50 on ZMAPS, and 16/16 on 8 random training maps.

**Grafts**

| # | From | What |
|---|---|---|
| G1 | J1 | **Fixed σ = 10 logits on point 11 for M1.** D2's dP = 0.10 calibration gives σ ≈ 5, which stalls at about +28. For M2, σ is calibrated on the **centre's own recent records** (J2/D3: net-v2 records are off-distribution). The target is dP = 0.25, with a cap of 15. |
| G2 | J1 | **Corrected M1 target.** ks30 ≈ ks40 vs zdpf005 (2/16 against 3/16), so recovery needs about +20 to +25, not +28 to +33. The mechanism criterion uses ks40's iteration-0 own records for rounds < 200. |
| G3 | D1/J2 | Head-to-head (seat-swapped, same job) for all gates and promotions. It is about 2.8x cheaper than a common opponent. CRN is free: DNet's rng is seeded from (myId, round). |
| G4 | J2/J3 | **Map handling.** Each iteration takes a random permutation of the full training-map list, never chosen by score. Report s by map-length class (short/long, from round counts) and the share of exact-zero units (identical traj_hash in both seats). Flag maps where the seed is degenerate (thunderdome). |
| G5 | D2 | Pre-registered iteration-0 t check, a negative control, a null-coordinate drift report, and a stop rule for the bias space. |
| G6 | J1 | A cheap cliff-slope check before M1: ks20 vs ks30. |
| G7 | D1 | Fork hook, **as tooling only**: an exact-prefix regression test and an option-equivalence detector (hold ≡ S_CAP in 2 of 3 tries), used to mask dead ES coordinates. Never used for credit: z decorrelates within 50-100 rounds of a single-decision fork. |
| G8 | J3/D3/D1 | **Stage B for M2: "CPI proposes, head-to-head judges."** D3's ε-exploration plus TD-H100 R-learner/CPI produces a candidate Δ. A 1-D antithetic head-to-head (θ+αΔ vs θ−αΔ) accepts or rejects it. The outcome z judges every step, which removes the V-Goodhart risk. |
| G9 | J3 | An offline check on rl-m0 (zdpn2 mirror; V at chance before about round 500): how much TD-H100 per-decision signal survives without sabotage. It decides how stage B gets its proposals. |
| G10 | J2 | `box_request` ships N packages per iteration; prune on both hosts after each update. A package is 0.74 MB, so an iteration is about 47 MB per host. |

**Rejected:**
- D1's Adam: simulated null-coordinate drift of about 11 logits.
- D1's counter-based DNet.uniform: it changes play.
- Vine/fork credit.
- D3's CPI as the M1 workhorse: its game result was null (3/16 against 3/16), and opening P(hold) needs KL ≈ 4 against a 0.2 trust region.
- D3's Java per-point floor change: `W.FL` and `build_dpnet --floor-points` already exist.

## 1. Method specification

**Parameters**
- φ = {point p: {"b": [offset per option], "logT": x}} over a fixed **base net** (the run's start net.json).
- Candidate logits for a point in φ are (W·h + b_base)·exp(−logT) + b_offset. The scale happens first, as in desaturate.py; then the bias is added, as in sabotage.py.
- Points not in φ, and all hidden layers, are copied value-for-value, so their W.java entries stay byte-identical.

**Blocks**
- A block is one point's option biases plus its logT.
- Excluded:
  - value-digit points (9 BABY_VAL, 15 KING_VAL): a global bias over digits is meaningless;
  - options that are always masked;
  - options that the fork tool marks equivalent to another option. Equivalent options share one coordinate.
- KING_COMMS (81 options) is split into sub-blocks of at most 24 options by option group.

**Directions and jobs** (iteration k, n_dir directions)
- ε_i ~ N(0, I) over the chosen block, with seed 2_000_000 + 1000·k + i.
- θ±_i = φ ± σ⊙ε_i.
- Each direction gets `units_per_dir` units. A unit is (map, seed) played twice: (A=θ+, B=θ−) and (A=θ−, B=θ+).
- Maps come from a per-iteration permutation of `training_maps`; seed = seed_base + 1000·k + unit index.
- Jobs record no decisions (`record: ''`) and turn trace off (`trace: 0`).
- **Block assignment (M2):** direction i perturbs block (i + k) mod n_blocks, a round-robin. Every block is covered each iteration once n_dir ≥ n_blocks.
- **Pool units (M2 only, 25%):** θ+ and θ− each play the same pool snapshot on the same job, seat-swapped (4 games). This anchors against non-transitivity.

**Fitness per unit:** s_u = (mean score of the θ+ side over the unit's games) − 0.5, in [−0.5, 0.5]. A unit whose two games have identical traj_hash in both seats gives exactly 0; it is counted and kept, not dropped.

**Update** (per coordinate j, over the units whose block contains j)
- g_j = mean_u(ε_uj·s_u)
- se_j = sd_u(ε_uj·s_u)/√n_u
- t_j = g_j/se_j
- Δφ_j = 0.25·σ_j·sign(t_j)·min(2, max(0, |t_j| − 1)). The step is at most 0.5σ, and nothing moves when |t| < 1.
- M2 only: φ ← (1 − 0.02)·φ toward the base each iteration, the L2 leash. Cap |φ|∞ ≤ 30 and log-T in [−1, 1].
- The floor stays at 0.005 (plus per-point `--floor-points`), so every option stays sampleable and the start stays capability-complete.

**Calibration (M2):** every 4 iterations, play 16 recorded centre-vs-centre games (RL build, `--decisions`). σ_j is the offset at which the mean |ΔP_j| over those own records is 0.25, clipped to [3, 15]. logT gets σ = 0.3.

**Metrics** (`metrics.jsonl`, one row per iteration)
- per-coordinate g, t and φ
- |t| > 2 share
- split-half cosine of g
- share of exact-zero units
- s by map class
- fixed-record P(option) and own-record P(option) by round bin
- KL to base
- null-coordinate drift: the L∞ and median |φ| over coordinates with no planted effect

## 2. Ordered implementation list

1. **`scripts/rl_collect.py`** (small edit): `trace: 0` (or a missing key) omits `--trace`. `record: ''` already skips `--decisions`. Verify on 2 jobs that result.json and traj_hash are unchanged against a traced run.
2. **`scripts/es_net.py`** (new, about 120 lines)
   - `load_phi(path)`, `save_phi(path, phi)`, `zero_phi(base, blocks)`
   - `apply_phi(base_raw, phi) -> raw`: scale, then bias; this reuses the logic in `desaturate.desaturate` and `sabotage.sabotage`
   - `blocks(base_raw, points, exclude, equiv) -> [Block(point, coord_ids)]`
   - `calibrate_sigma(base_raw, phi, rec_dirs, target=0.25, cap=15)` via `dpnet_io` forward on p<point>.npz
   - `--self-test`: apply_phi(φ=0) must be byte-identical to base W.java for every point
3. **`scripts/es_schedule.py`** (new, about 150 lines)
   - `draw(k, n_dir, blocks, sigma) -> dirs`
   - `write_candidates(I, dirs)`: writes I/cand/<i><p|m>/net.json and runs `build_dpnet.py --name zdpe<k>_<i><p|m> --floor 0.005 [--floor-points …]` with no `--rl`
   - `jobs(cfg, k, dirs) -> jobs.json`: unit, dir, sign_a, map, seed and pool fields
   - `prune(k)`: removes the build/botsrc/zdpe<k>_* directories
4. **`scripts/es_update.py`** (new, about 150 lines)
   - `unit_scores(I) -> s_u`, `step(phi, dirs, s, sigma, cfg) -> phi', stats`
   - writes `I/phi.json`, the new centre `net.json` (via `apply_phi`) and the `metrics.jsonl` row
   - `--self-test`: on synthetic s with a planted linear effect, t has the right sign and null coordinates do not move when |t| < 1
5. **`scripts/es_loop.sh`** (new; a fork of `rl_loop.sh`'s stage, log, STOP and disk-guard helpers)
   - stages per iteration: schedule/build → collect (box slice plus VM slice, same `--deadline` and `--min-done` logic) → update → anchor/gate on schedule → prune
   - Generalise `collect()` to push every candidate package: build the `box_request` push list from the jobs.json package set, the same comprehension rl_loop already uses for pool members.
   - Builds happen only on the host that schedules. The box receives botsrc and compiles on first use, as now.
6. **`scripts/es_eval.sh`** (or reuse `eval_student.py`): a paired head-to-head gate (centre vs reference on ZMAPS × seeds × AB) and the 32-game anchor, printing the score, Wilson CI, and an exact-zero count for each.
7. **Fork tool** (optional; done after M1 starts and not on the critical path)
   - `Cfg.FORK_ID/FORK_ROUND/FORK_POINT/FORK_K` (default −1), applied in `DK.netPick` and `D.pick` after `DNet.sample`, so the uniform draw is still consumed.
   - `scripts/fork_equiv.py`: for each option pair at sampled states, fork, play +20 rounds, and compare the trace. It writes the `equiv` file read by `es_net.blocks`.
   - Regression check: the default build gives an identical traj_hash on 8 jobs, and a null fork gives an identical hash.
8. **Stage B (M2 only, after the stage-A stop rule fires):** `rl_train.py --estimator cpi` (D3's prototype `cpi3.py` plus the trace lookup from `td1.py`, about 150 lines), and `es_schedule.py --direction FILE` for 1-D judged steps.

Estimated effort: items 1-6 take about half a day; items 7-8 come later.

## 3. Pre-M1 checks

| Check | Games | Pass / use |
|---|---|---|
| C0 (offline): run D3's `td1.py` regressions on rl-m0 (zdpn2 mirror, all points) and on a 128-game unsabotaged iteration if one exists. Compute per-point TD-H100 t for the natural deviations. | 0 | Informational for M2 stage B (G9). If the median \|t\| on Phase-A points is < 1, stage B proposals come from low-rank ES, not CPI. |
| C1 cliff slope: ks20 vs ks30 head to head, 8 random training maps (fixed rng, not by score) × 2 seeds × AB. | 32 | Predict that ks20 scores ≥ 0.60. If it scores < 0.55, the signal near the target is weak: M1 is still run, but the pre-registered expectation drops to "reaches about +25", and criterion 2 is reported rather than binding. |
| C2: `es_net --self-test` and `es_update --self-test`; a 2-job dry run of the loop. | 2 | Byte-identical frozen points; correct job pairing. |

## 4. Revised M1 (known answer)

**Setup**
- Start: ks40 (desat-b5 + KING_SPAWN S_HOLD +40). It scores 20/128 = 0.156 vs zdpf005 (Wilson [0.103, 0.229]).
- φ covers point 11 only: 22 option biases plus logT. σ = 10 on biases, 0.3 on logT. No leash, no pool.
- Every game is θ+ vs θ− from the current centre.
- Each iteration has 32 directions × 2 units × 2 games = **128 games** and 64 packages, played on training maps (no ZMAPS).
- At most **8 iterations**.

**Pre-registered checks**
- After iteration 0, t_hold ≤ −2 (predicted about −3). If it fails, run iteration 1. If the pooled t over iterations 0-1 is still above −2, stop and diagnose (map mix, σ, exact-zero share) before spending more.
- Anchor after iterations 1, 3, 5 and 7: centre vs zdpf005 on 32 fixed ZMAPS jobs. This is for trend only.

**Pass** (all of the following)
1. **Binding:** the final centre vs zdpf005 on 128 ZMAPS jobs (seeds 31-38, AB) scores **≥ 0.45**, with a Wilson CI that does not overlap [0.103, 0.229].
2. **Mechanism:** P(hold) on **ks40's own iteration-0 records, rounds < 200**, recomputed under the final centre, is ≤ **0.05** (about +24). ≤ 0.02 (about +20) counts as full recovery. Fixed-record P(hold) on net-v2 states is reported but not used.
3. All points other than 11 and all hidden weights are byte-identical (W.java diff).
4. **Negative control:** 2 iterations (256 games) from the unsabotaged desat-b5, same settings.
   - The share of point-11 coordinates with |t| > 2 is ≤ 15% (chance is about 5%, i.e. about 1 of 23).
   - No coordinate's cumulative |Δφ| exceeds 1σ.
5. Reported, not gating: null-coordinate drift (expected median ≤ 3 logits, per J1's simulation, about 2), the exact-zero unit share, s on short vs long maps, and the hold-bias trajectory.

Expected outcome (J1's simulation, fixed σ = 10): the bias reaches about +30 by iteration 2 and about +17 by iteration 8 at the measured slope. At half the slope, 88% of runs reach ≤ +30.

Caveat: long ZMAPS (pipes, rift) favour holding, so criterion 1 may lag the training signal. That is why criterion 1 binds and the training maps are never tuned.

**Budget:** C1 32 + C2 2 + M1 ≤ 1,024 + anchors 128 + final gate 128 + negative control 256 = **about 1,570 games, about 5-6 h** at 250-350 games/h.

**M1b (generality / needle; required before M2, capped):**
- Same ks40 start, but φ covers all six Phase-A blocks (5, 7, 10, 11, 12, 13; about 145 coordinates), block round-robin.
- σ is calibrated (dP = 0.25 on ks40's own records, cap 15), except point 11, which keeps σ = 10.
- 128 games per iteration, **cap 2,000 games**.
- Pass:
  - criteria 1 and 2 as in M1;
  - no non-11 point's own-record option histogram moves by more than 0.1 TV;
  - non-11 |t| > 2 share ≤ 15%.
- If it fails only on budget, M2 proceeds with fewer blocks per stage: 11, 13, 5 first.

## 5. M2 plan (real self-play RL)

**Start and reference**
- The start is the capability-complete start net from the parallel capability pass (net-v2 desaturated + widened, with per-point floors).
- Snapshot 0 is that start, and the reference is zdpn2. Results are reported against both.
- zdb is unused (the bootstrap flag stays off). The field is used only in `strength_check` after a promotion, under the existing field rule; it never tunes anything.

**Stage A: bias/temperature ES**
- Blocks: 5, 7, 10, 11, 12, 13, 16 (KING_TURN), and the KING_COMMS sub-blocks; new points enter as more blocks.
- σ is calibrated every 4 iterations. There is a 2% leash toward the base.
- Each iteration has 128 games, of which 25% are pool units.
- Cheap anchor every 3 iterations: centre vs snapshot 0, 32 paired head-to-head games.
- **Promotion gate** every 6 iterations, or when the anchor reaches ≥ 0.56: 128 paired head-to-head games vs snapshot 0 (and vs zdpn2, if that differs). Promote if the Wilson lower bound is above 0.50, then run strength_check and the field rule.
- Bradley-Terry Elo over the pool, as already planned.
- **Stop rule:** no coordinate reaches |t| > 2.5 over two full sweeps of all blocks (about 3k games). That declares the bias space exhausted and moves the run to stage B.
- Expected cost per +5 pp: about 4-11k games (15-40 h), and only if such a gain exists in bias space. zdp's +12 pp over zdb came from state-dependent fixes, so a quick stage-A exhaustion is a likely, acceptable outcome.

**Stage B: state-dependent steps, with z as the judge**
- **Proposals.** If C0 shows usable TD signal, the proposal is a CPI step (D3): collect with ε-exploration via `--floor-points` (0.1 on king points, 0.02-0.05 on baby points; collection builds only); use the TD-H100 R-learner ensemble with LCB, then distil with KL ≤ 0.2. Otherwise the proposal is low-rank ES: add σ·(ε0 + Σ_r ε_r·(u_r·h)) to an option's logit, with u_r the top 2-4 PCA directions of the last hidden layer on own records.
- **Judging.** A 1-D antithetic head-to-head: θ+αΔ vs θ−αΔ, 128 games (seat-swapped, training maps).
  - If t > 2, apply α·Δ.
  - If 1 < t ≤ 2, keep Δ and double the games on the next try.
  - Otherwise halve α. After two halvings, discard Δ.
- **Guided ES (D1):** half of the stage-A directions are drawn from the span of the last 4 accepted proposals.
- Phase B baby points (MOVE, ACT, FACE, COMMS, FREE) are unlocked only through stage B, one stage at a time, as in rl-plan P3.

**M2 pass (unchanged from rl-plan):** a promoted version with Wilson lower bound > 0.50 vs zdpn2 (point estimate ≥ 0.55), and the field rule satisfied.

**Overall M2 budget:** 12k games (about 40 h) before the next design review. Stage A gets at most 6k of that, unless a coordinate with |t| > 2.5 is still moving.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Bias space holds no +5 pp | Stop rule → stage B; ES is the coarse layer only |
| Map-dependent sign (rush vs long maps) | Full-map permutation per iteration; per-class reporting; never select maps by score |
| Non-transitivity of "beat your sibling" | 25% pool units; gates against snapshot 0 and zdpn2; Elo; gate 3 round robin |
| Null-coordinate drift raises KL | \|t\| < 1 dead zone, leash, \|φ\|∞ cap, drift metric, negative control |
| Too-large σ (both siblings bad) | Block perturbation; calibration on own records at dP 0.25, cap 15 |
| V-Goodhart in stage B | z judges every accepted step; V is used only inside proposals |
| VM disk / package churn | 47 MB per host per iteration, pruned each iteration; the disk guard from rl_loop |
| Degenerate seeds (thunderdome) | Exact-zero share logged; such units contribute 0 and are counted |

## 7. Rules check
- **Self-play only:** training games are learner siblings (plus pool snapshots). zdb is not used. The field is used only for stop/rollback after a promotion.
- **Outcome-only:** fitness is z only. In stage B, V(trace) shapes proposals only, and z decides acceptance.
- **Capability-complete:** every option and point stays sampleable (floor 0.005, per-point floors). New points are new blocks, and nothing here depends on the capability pass's details.
- **Bytecode limits:** off.

## 8. Evidence pointers (box, ~/rltest/bc26a0 sandbox)
- /tmp/d1: forks, CRN tables
- /tmp/designer2: h2h +30 vs +50 = 11/16; ks20 to ks50 nets; crn.py
- /tmp/designer3: td1.py, cpi3.py, 16 CPI games
- /tmp/judge1: ks30 vs zdpf005 = 2/16; ph.py; sim2.py
- /tmp/judge2/h2h: +30 vs +50 on training maps = 16/16
- /tmp/judge3/vauc.py: V AUC at chance before round 500 in the zdpn2 mirror

## 9. es-m2 result and es-m3 (November 15, 2026)

**es-m2** ran stage A from rl-m2 v39 (05:14–11:50 PDT, 12 updates, about 1.9k games). It used per-point blocks with fixed σ = 5 on every bias and t_dead 1. It found nothing:
- Gates against zdpf005 (32 games each): v0 17, v3 14, v6 18, v9 17, v12 13. Pool games against the start net, iterations 0–11: 263/576 = 0.46.
- **Too few directions per coordinate.** The 32 directions were split over the 7 point blocks, so each coordinate's t came from 4–5 directions (3–4 degrees of freedom). With t clustered by direction (section 1), chance gives |t| > 1 about 37–39% of the time; 43% of the coordinates moved each iteration. The |t| > 2 share was 0.067, against 0.12–0.14 by chance at those degrees of freedom. The split-half cosine ranged from −0.09 to 0.15 and averaged about 0.04. |φ| grew steadily from 8 to 26 (the median null drift reached 1.3 logits): a noise walk. M1 never had this problem, because it had a single block (point 11), so all 32 directions hit every coordinate.
- **σ too small.** The calibration this section's plan asks for was never wired into `es_loop.sh`. Run afterwards on the start net's own records (rl-m2 iterations 37–38, 20k rows per point), it shows a nearly deterministic policy: for most options, a 5-logit offset moves P(option) by less than 0.05. The median σ for mean |dP| = 0.05 / 0.1 / 0.25 is 5–15 / 6–15 / 8–15 logits by point, and most coordinates reach the cap of 15 at the plan's 0.25.

**es-m3** ([config](../configs/es-m3.json)) restarts from rl-m2 v39 (φ = 0) with three changes. The budget, pool, maps and gates stay the same.
- `joint: true` (`es_schedule.cfg_blocks`): one block holds all 168 coordinates, so each direction perturbs all of them and every coordinate's t runs over all 32 directions (31 degrees of freedom). Cross-talk from the other coordinates' effects stays inside the bounded unit score s, so per-coordinate noise does not grow.
- σ is calibrated once, before iteration 0, on the start net's records (`runs/es/es-m3/sigma.json`). The target is mean |dP| 0.1 rather than 0.25, because all seven points are perturbed at once; values are clipped to [3, 15] (config `calib_target`, `calib_lo`, `calib_cap`). The median bias σ is 14.3; logT keeps 0.3. Recalibration every 4 iterations is still not automated.
- t_dead 2: at 31 degrees of freedom, chance moves about 5% of the coordinates per iteration, by at most about 0.1σ on average. The leash (2%) and caps are unchanged.

Early warning: if the candidates' pool score against the start net falls below about 0.35 in iterations 0–1, σ is too large (both siblings play badly); halve calib_target.

Stop rule (set in advance): if by iteration 18 (about 2.9k games) the last three gates pooled against zdpf005 do not have a Wilson lower bound above 0.52, bias space is exhausted and the run moves to stage B (section 5).

**Margin fitness (user suggestion, November 15, 2026; es-m3 from iteration 0).** A finer score than win/loss: a quick win beats a slow one, and a long loss beats a short one. `es_update._pts`, config `fitness`: score = 0.5 + 0.5 ((1 − α) o + α o m), with o = +1 for a win, −1 for a loss, 0 for a draw. The margin m is 1 − 0.5 r/2000 for a game that ends at round r before the limit, and 0.5 for one decided on points at the limit. Any win still outscores any loss. The `rp` variant gives a game decided at the limit m = 0.5 min(1, points gap / points total / 0.2) instead.
- In es-m2/es-m3 games, 51% end early by killing all the rat kings (median round 441) and 44% are decided on points at round 2000, usually narrowly (median gap 5% of the points total). So the round margin carries information for about half the games.
- Offline check on known-unequal pairs (the M1 sabotage pre-tests against zdpf005, 32 seat-swapped units each), paired t:
  - zdpk40 (17% wins): binary −6.17; rounds α 0.25 / 0.5 / 1: −6.38 / −6.59 / −6.91; rp: −6.51 / −6.80 / −7.04. That is about 7–14% more t, worth roughly 15–30% more games.
  - zdpk60 (2% wins): binary −31, already at its ceiling; the margin only adds noise.
  - zdps3 / zdps5 (44% / 47%): no metric separates them (|t| < 1). The margin variants move t toward 0, and rp at α 1 flips its sign. This is within noise, but a hint that margin and win rate can disagree on small changes.
- On es-m2's 12 iterations (no signal), all variants give the same |t| > 2 share and split-half cosine. The binary rescoring reproduces the logged metrics exactly.
- es-m3 drives the step with `rounds`, α 0.5. `shadow_fitness` logs binary and `rp` α 0.5 on the same games each iteration (metrics `shadow`: |t| > 2 share, split-half cosine, cosine of g with the driver). Gates stay pure win/loss, so a policy that buys speed at the cost of wins shows up there.

**es-m3 iteration 0 and the σ scale.** Iteration 0, at σ calibrated for mean |dP| 0.1 (median 14.3), showed no signal: the |t| > 2 share was 0.048 (chance at 31 degrees of freedom is about 0.05), and the split-half cosine was −0.06. Win/loss and both margin fitnesses gave nearly the same g (cosine with the driver 0.998 and 0.999). The candidates scored 8/32 against the start net, below the pre-set early-warning bar of 0.35 (es-m2: 0.46), and the head-to-head games got much shorter (median 500 rounds, against about 1,140 in es-m2). Halving calib_target to 0.05 would shrink σ by only about 15% (median 11.3), because the policy is saturated. So the change was measured directly: on the start net's own records (rl-m2 iteration 38, 5k rows per point), the share of decisions whose top choice a joint candidate flips, by point (5, 7, 10, 11, 12, 13, 16):

| σ | top-choice flips per point | sum |
| --- | --- | --- |
| dP 0.1 (iteration 0) | .02 .31 .70 .37 .55 .69 .46 | 3.1 |
| dP 0.05 | .04 .22 .52 .33 .51 .54 .56 | 2.7 |
| 0.5 × dP 0.1 | .01 .13 .30 .24 .15 .25 .18 | 1.3 |
| 0.35 × dP 0.1 | .01 .09 .16 .11 .06 .13 .04 | 0.6 |
| flat 5 (es-m2's σ, joint) | .01 .17 .24 .10 .29 .16 .46 | 1.4 |

es-m2 perturbed one point at a time (about 0.2 summed), and its candidates scored 0.46. From iteration 1, es-m3 uses the dP 0.1 calibration times `sigma_scale` 0.35 (a new config key, `es_schedule.sigma_of`; logT keeps 0.3). That gives a median bias σ of 5.0, with smaller σ where the policy is sensitive. Iteration 1, scheduled at the old σ, was discarded unplayed apart from 9 VM games and redrawn. Iteration 0's update stands (8 coordinates moved, the largest 11:b10 by −6.0). If the pool score stays below 0.35, the scale goes down again; the stop rule is unchanged.

## 10. es-m3 stopped; C0 and the stage-B exploration probe (November 15, 2026)

**es-m3 stopped after iteration 8** (16:34 PDT; iteration 9's 39 finished games were not applied). About 1.5k games:
- Gates against zdpf005: v0 17, v3 15, v6 16, v9 15 of 32. Pool games against the start net, iterations 0–8: 116/288 = 0.40.
- Every iteration: the |t| > 2 share was at chance (0.048–0.059), and the split-half cosine was 0.015–0.10. Both shadow fitnesses gave the same g (cosine with the driver above 0.99).
- The stop rule (iteration 18) needs the gates at 12, 15 and 18 to pool to at least 60/96. At the measured 0.47 that has probability about 0.002, and with no gradient signal the centre was not moving. So the remaining ~1.4k games went to stage B's deciding test instead.
- The box's memory-guard kills at 6 workers came from glibc malloc arenas: about half of a game JVM's 1.2 GB RSS. `student_java.sh` now caps them (MALLOC_ARENA_MAX=2), which gives the same traj_hash with about 40% less peak RSS on long games (commit 9da30e7).

**C0** (`scripts/td_check.py`, 0 games). Run on rl-m2 iterations 37–38: 256 games, the learner's recorded sides, natural deviations at floor 0.005 (0.001 on point 5). The only iterations that kept their records were 37 and 38. The critic was the rich one, cross-fitted, with iterations 32–36's trace rows added for training; held out, it scored Brier 0.212 (0.25 for a constant) and AUC 0.715.
- A deviation is a sampled option other than π's top choice. Its mean effect comes from the R-learner Σ Y (D − e) / Σ (D − e)², with a game-clustered t, on three targets:
  - td: V 100 rounds later minus V now (TD-H100);
  - mc: z minus V;
  - pre: V now minus V 100 rounds earlier, a placebo that cannot depend on the decision.
- CPI t_oof is the out-of-fold t of the state-dependent model: option advantage linear in 8 principal components of the playing net's last hidden layer, ridge, cross-fitted over halves of the games (`--cpi`).

| point | records | deviations (rate) | t td | t mc | t pre (placebo) | CPI t_oof (in sample) |
| --- | --- | --- | --- | --- | --- | --- |
| 5 BABY_ROLE | 1,378k | 2,960 (0.2%) | −1.13 | −1.23 | −0.54 | 0.65 (4.0) |
| 7 MINE_PICK | 178k | 8,980 (5.1%) | −1.80 | +0.79 | −2.94 | −2.32 (4.3) |
| 10 KING_MOVE | 600k | 48,545 (8.1%) | +1.63 | −0.69 | −1.02 | 1.31 (10.5) |
| 11 KING_SPAWN | 741k | 26,470 (3.6%) | −0.64 | −0.21 | +1.46 | −0.10 (9.6) |
| 12 KING_ACT | 697k | 10,480 (1.5%) | +0.66 | −0.17 | −0.73 | −0.05 (7.9) |
| 13 KING_FORM | 498k | 6,224 (1.3%) | +0.80 | +0.43 | +0.88 | 0.17 (9.3) |
| 16 KING_TURN | 726k | 9,688 (1.3%) | +0.18 | +0.04 | +0.91 | −0.64 (6.6) |

- The Phase-A median |t td| was 0.965, against the rule's threshold of 1. A rerun with a different critic fit gave 1.005, so the rule's verdict flips on the critic's random seed.
- The placebo's median |t| is 0.95. Under a correct se it would be about 0.67, so the clustered se is probably about 1.4× too small. Either way, the deviation effects cannot be told from the placebo.
- Per option, |t td| > 2 held for 3 of 127 tests, against 5.8 expected by chance.
- The state-dependent model fits in sample (t 4–10) and predicts nothing out of fold.
- Conclusion: in 256 games of natural deviations (0.2–8% of decisions) there is no usable per-decision signal. That holds for CPI and for PPO alike, since PPO's gradient rests on the same deviations.

**Exploration probe `cpi-x0`** ([config](../configs/cpi-x0.json)). The rule's two branches both lack evidence: C0 sits on its threshold, and the out-of-fold check is null. CPI was always meant to run on ε-explored data, which C0 did not test, and exploration is the RL-side lever. So one probe decides:
- Setup: 256 mirror games of rl-m2 v39, both sides recorded, at the plan's stage-B collection floors: 0.1 on the king points 10, 11, 12, 13 and 16; 0.05 on MINE_PICK; 0.03 on BABY_ROLE. lr is 0 and there are no gates.
- Per team-game, this gives about 3 times (KING_SPAWN), 7–8 times (KING_ACT, KING_FORM, KING_TURN) and 15 times (BABY_ROLE) the deviations, and 1–2 times on MINE_PICK and KING_MOVE.
- Then: `td_check.py --cpi` on its records, with rl-m2 iterations 32–38's trace rows as extra critic data.

Pre-set reading:
- If the out-of-fold t is at least 2 at 2 or more of the 7 points (chance about 0.01), stage B runs CPI steps. Each step is a last-layer update from the fitted advantages, judged by θ+αΔ against θ−αΔ.
- Otherwise, per-decision credit is out of reach at this game budget, and stage B runs the plan's low-rank ES.

## 11. Stage-B protocol, fixed before the probe's fit (November 15, 2026)

Written while cpi-x0 was still collecting, before any of its records were fitted.

**Pooled es-m3 gradient (0 games).** `es_propose.py` pools every direction of es-m3 iterations 1–8 (256 directions, one sigma scale) into one natural-gradient vector. Its |t| > 2 share is 0.042 (chance 0.046). Its split-half cosine (even against odd directions) is −0.14. A real gradient gives a positive cosine. So es-m3's whole bias space carries no direction worth a judge's 128 games, and it gets none.

**CPI branch (the probe finds signal).**
- Δ covers only the points whose out-of-fold t is at least 2. `cpi_step.py` maps the fit onto the playing net's last layer: option k's logit gains η_p (W[k,0] + Σ_r W[k,1+r] pc_r(h)). η_p is set so that the mean TV distance from π, on the probe's records under the deploy floors (rl-m2's), is 0.1 at α = 1. A check on C0's fit matched the edited nets' logits to the direct formula within 3e-5, with hidden layers and other points unchanged.
- Judge: packages zdpt<n>p against zdpt<n>m, on 32 maps (the first 32 of the training maps permuted by `numpy.random.default_rng(3800000 + n)`), seed 3800000 + 100n + 1 on the box and + 2 on the VM, both seats: 128 games, scored by `pair_score.py` (the paired t over units).
  - t > 2: guard, θ+αΔ against θ (zdpg39) on the same maps with seeds + 3 and + 4, 128 games. Accept if that score is at least 0.5. The antithetic judge measures the direction, while the guard catches a step that hurts both ways.
  - 1 < t ≤ 2: 128 more games at the same α (seeds + 5, + 6), then the pooled t decides as above.
  - Otherwise, halve α (1, 0.5, 0.25). After α = 0.25 fails, discard Δ.
- After an accepted step, the next step starts from a new exploration probe of θ+αΔ, with the same floors and 256 games.

**Low-rank ES branch (no signal).** [configs/es-lr0.json](../configs/es-lr0.json):
- Setup: rl-m2 v39 plus an `es_basis`, the top 3 PCs of each point's last hidden layer on the probe's records. Every option gets its bias plus 3 low-rank coordinates (`<p>:c<o>.<r>`), for 4 times es-m3's 168 coordinates in one joint block.
- σ: calibrated on the probe records (mean |dP| 0.1), times sigma_scale 0.2. That puts the total logit perturbation at about 1.1 times es-m3's.
- Everything else is as es-m3: 160 games per iteration, pool, gates and fitness.
- Futility, pre-set: after 6 updates, stop if the pooled gradient (es_propose over all directions) has a split-half cosine of at most 0.1 and a |t| > 2 share of at most 0.07. If the cosine is above 0.2, the pooled proposal goes to the 1-D judge above.

## 12. cpi-x0 probe result; the out-of-fold t was miscalibrated (November 15, 2026)

**Probe.** 256 mirror games of rl-m2 v39 under the stage-B floors, all valid (box 213, VM 43). Natural deviations
from the top choice, against C0's 256 games at floor 0.005: point 5 34.5k (11.7×), 7 15.6k (1.7×), 10 104k (2.1×),
11 90.8k (3.4×), 12 69.3k (6.6×), 13 49.6k (8.0×), 16 52.8k (5.4×). The critic (trained on rl-m2 iterations 32–38)
scores the probe's bins at AUC 0.696 (Brier 0.221 against 0.25).

**Phase A.** The deviation effect on the TD target is negative at all six Phase-A points (t −0.68, −1.11, −1.05,
−2.04, −0.42, −1.36; median |t| 1.08). That is the cost of exploring off a decent policy, not a usable direction.

**CPI fit (as pre-registered), pooled out-of-fold t:** 5 0.04, 7 −3.82, 10 2.76, 11 2.08, 12 −0.97, 13 0.93,
16 0.58 (in-sample t 5.9–12.9). On the mc target (secondary): 5 2.81, 7 0.88, 10 0.92, 11 −0.59, 12 0.10, 13 0.15,
16 −0.59. By section 11's rule (t ≥ 2 at 2 or more points), points 10 and 11 qualified. Step 0 was built
(`cpi_step.py`, η 152 and 607, logit offset sd 3.2 and 8.9; mean TV 0.100/0.100 plus, 0.102/0.157 minus; the top
choice flips in 9–16% of decisions) and its judge t0a1 started at 17:36 PDT (zdpt0p against zdpt0m; maps: row 0 of
the judge permutation, the first 28 on the box and the last 4 on the VM, seeds 3800001 and 3800002 on both: the
hosts split by map rather than by seed because the VM is about 6× slower per game).

**The statistic was wrong.** Point 7 at −3.82 (and −2.32 in C0) is far outside N(0, 1), so the t got a direct null
test: `td_check.py --sham N` redraws every record's option from π(·|x), choices that cannot affect Y, and reruns the
same out-of-fold check.
- Synthetic data (256 games, no leverage, effective games 253): the pooled t has sham sd 1.48–1.62; each fold's own t
  (fit on one half, test on the other) has sd 1.06–1.18; the two folds' t correlate at 0.99. A planted effect gives
  fold t ≈ 30, so the check has power.
- Why: the two halves' terms are bᴬ′M_B bᴮ and bᴮ′M_A bᴬ (b a half's score vector, M its ridge inverse), nearly the
  same number, but the pooled clustered standard error treats them as independent. The pooled t is about √2 too
  large. Section 11's "t ≥ 2" was in effect a 1.41 bar: about 8% false positives per point and about 10% for "2 or
  more of 7" with no signal at all, not the ~1% intended.
- Fix (`td_check.py`, `cpi_step.py`): `t_fold_mean`, the mean of the two folds' own t, replaces the pooled t.

**Corrected rule (written at 17:45 PDT, after seeing point 5's per-fold t and before points 7–16's):** section 11's
rule with t_fold_mean in place of the pooled t: t_fold_mean ≥ 2 at 2 or more of the 7 points means CPI, otherwise
low-rank ES. For the judge already running (t0a1, built under the old rule), its own paired t still counts:
- t > 2: the guard against zdpg39 as in section 11, whatever the fit says.
- 1 < t ≤ 2: 128 more games, as in section 11.
- t ≤ 1: if the corrected rule also fails, discard step 0 without the α = 0.5 and 0.25 retries (they were meant for
  a direction the fit supports) and start low-rank ES. If the corrected rule passes, continue section 11 as written.

**Per-fold t and sham null on the probe** (`td_check.py --cpi --sham 30`, 30 sham draws per point; fold t's, then
t_fold_mean; the sham's pooled-t sd, fold sd and fold correlation):

| point | fold t | t_fold_mean | sham pooled sd | sham fold sd | fold corr | effective games |
|---|---|---|---|---|---|---|
| 5 BABY_ROLE | −0.09, 0.18 | 0.05 | 1.24 | 0.89, 0.92 | 0.96 | 127 |
| 7 MINE_PICK | −5.06, −1.25 | −3.16 | 2.87 | 3.40, 1.07 | 0.36 | 1.5 |
| 10 KING_MOVE | 1.73, 2.21 | 1.97 | 1.39 | 0.99, 1.00 | 0.97 | 118 |
| 11 KING_SPAWN | 1.46, 1.58 | 1.52 | 1.68 | 1.17, 1.25 | 0.95 | 136 |
| 12 KING_ACT | −0.94, −0.47 | −0.71 | 1.44 | 1.18, 0.97 | 0.85 | 46 |
| 13 KING_FORM | 0.59, 0.70 | 0.65 | 1.17 | 0.90, 0.87 | 0.79 | 36 |
| 16 KING_TURN | 0.39, 0.54 | 0.47 | 1.32 | 0.97, 0.97 | 0.94 | 137 |

- Point 7's −3.8 is leverage: one game holds 81% of the out-of-fold Σ Â² (effective games 1.5 of 256), and its
  sham t has sd 2.9 with 16 of 30 draws beyond ±2. The statistic means nothing there (40 options, many rare).
- Elsewhere the sham confirms the √2: pooled sd 1.2–1.7, fold sd 0.9–1.25, fold correlation 0.79–0.97.
- **The corrected rule fails:** no point has t_fold_mean ≥ 2 (point 10 1.97, point 11 1.52). Across 7 points, the
  largest of 7 null fold means reaches 1.97 about 16% of the time, so the probe shows no state-dependent signal
  beyond chance.

**Judge t0a1: zdpt0p 75/128 against zdpt0m (0.586, paired t 2.10 over 64 units, Wilson 0.499–0.668; box 63/112,
VM 12/16).** t > 2, so step 0 goes to its guard (t0a1g: zdpt0p against zdpg39 on the same maps, seeds 3800003 and
3800004, 128 games, started 17:58 PDT). Set before its result:
- Score ≥ 0.5: accept. θ₁ = zdpt0p's net. Probe cpi-x1 ([configs/cpi-x1.json](../configs/cpi-x1.json), cpi-x0's
  probe of θ₁) feeds step 1's fit (t_fold_mean rule, `--sham 30`). θ₁ also plays the 32-game gate against zdpf005
  (seeds 41 and 42), for the chart against the earlier runs.
- Score < 0.5: the α = 0.5 attempt (zdpt01p against zdpt01m, seeds 3800011 and 3800012), then 0.25 (zdpt02*,
  seeds 3800021 and 3800022), as in section 11; discard after that and start low-rank ES.

**Guard t0a1g: zdpt0p 61/128 against zdpg39 (0.477, t −0.62; box 53/112, VM 8/16): step 0 at α = 1 rejected.** The
α = 0.5 attempt (same Δ, points 10 and 11; mean TV 0.03–0.06, top choice flips in 3–6% of decisions) is judged as
t0a05: zdpt01p against zdpt01m, seeds 3800011 and 3800012, started 18:19 PDT.

**α = 0.5 (t0a05): zdpt01p 66/128 against zdpt01m (0.516, t 0.35). α = 0.25 (t0a025, seeds 3800021 and 3800022,
TV 0.016–0.027): zdpt02p 64/128 against zdpt02m (0.500, t 0.00). Step 0 discarded at 19:06 PDT.** Over the three
judges and the guard (512 games), the fitted direction is consistent with no effect: the α = 1 judge's t 2.10 did
not hold up against the start net (0.477) or at smaller steps. This agrees with the corrected fit (no point with
t_fold_mean ≥ 2). Next, as set in sections 11–12: low-rank ES ([configs/es-lr0.json](../configs/es-lr0.json)).

## 13. es-lr0 futility check after 6 updates (November 15, 2026)

es-lr0 (low-rank ES, 651 coordinates: 161 biases, 483 low-rank, 7 log-temperatures; σ calibrated on the cpi-x0
records, median 12.8 for biases and 15 for low-rank coordinates, times 0.2) started at 19:08 PDT. Per iteration
(|t| > 2 share; split-half cosine): 0.044, −0.04; 0.042, −0.003; 0.066, +0.07; 0.055, +0.03; 0.048, −0.02;
0.054, −0.007. Gate against zdpf005 at v3: 16/32 (v0 17/32). Pool games against the start net, iterations 0–4:
65/160.

**Pooled over iterations 0–5** (`es_propose.py`, 192 directions): |t| > 2 share 0.049 (chance 0.046), max |t|
2.87, split-half cosine **+0.103**. The pre-set stop needs a cosine of at most 0.1 and a share of at most 0.07,
so the run continues; the judge needs a cosine above 0.2. Two independent noise vectors over 651 coordinates have a
cosine sd of about 1/√651 ≈ 0.04, so +0.10 is about 2.6 sd: perhaps a weak gradient spread over many coordinates,
too weak for any single coordinate's t.

Second checkpoint, set now (21:54 PDT) because section 11 set only one: after 12 updates (iterations 0–11, 384
directions), the same pooled check. Cosine above 0.2: judge the pooled proposal (`es_propose.py --build zdplj`,
section 11's judge and guard). Cosine at most 0.1: stop. In between: one last check after 18 updates, then stop
unless the cosine is above 0.2.

**After 12 updates (00:41 PDT November 16; iterations 0–11, 384 directions):** |t| > 2 share 0.049, max |t| 3.00,
split-half cosine **+0.170**: between the bars, so the last check comes after 18 updates (cosine above 0.2: judge the
pooled proposal from the start net; otherwise stop). The cosine rose from 0.103 as the directions doubled, as a weak
gradient spread over many coordinates would: no single coordinate stands out, but the two halves agree more as each
gets less noisy. It is about 4 sd from zero under no signal. Meanwhile the centre drifts down: gates against zdpf005
v3 16/32, v6 13/32, v9 10/32 (v0 17/32). The step rule moves coordinates that cross |t| > 2 by chance. That does not
touch the pooled proposal, which `es_propose.py` builds from the start net (θ = 0).

**After 18 updates (03:23 PDT November 16; iterations 0–17, 576 directions):** |t| > 2 share 0.068, max |t| 3.89,
split-half cosine **+0.311**, above the 0.2 bar, so the pooled proposal goes to the judge. The cosine rose with the
direction count (0.10, 0.17, 0.31), as a real gradient spread thin would. es-lr0 is stopped (STOP after iteration
17; its drifting centre has no further use, and the box is needed for the judge).

Proposal (`es_propose.py --iters 0-17 --build zdplj`, from the start net θ = 0, η 106.6 for mean TV 0.1 on the
cpi-x0 records): packages zdpljp (θ + ηΔ) and zdpljm (θ − ηΔ). The largest-t coordinates are 13:logT (step −1.21,
t −3.89), 10:b16, 11:c7.1, 13:c0.1, 13:b8 and 13:c19.2. The two sides are far from symmetric. Plus moves TV 0.03–0.30
per point; minus raises point 13's temperature e^1.21 ≈ 3.4-fold, so its KING_FORM policy flips 90% of decisions. The
antithetic judge will therefore mostly measure how bad the minus side is, and the guard (plus against the start net)
is the test that matters.

Judge, set now as in section 11 with n = 1: zdpljp against zdpljm on row 1 of the judge maps (28 maps on the box, 4 on
the VM), seeds 3800101 and 3800102, both seats, 128 games. Then: t > 2, guard zdpljp against zdpg39 on the same
maps with seeds 3800103 and 3800104 (accept if at least 0.5); 1 < t ≤ 2, 128 more games (seeds 3800105 and 3800106);
otherwise halve α (0.5, then 0.25) and then discard. Because the judge is weak evidence here, the guard runs even
if the judge passes, which section 11 already requires.

**Guard (04:05 PDT):** zdpljp against the start net zdpg39 scored 67/128 = 0.523 (t 0.49, 95% CI 0.44–0.61).
That meets the pre-set bar of at least 0.5, so the step is **accepted**. The evidence is the gradient check (cosine
+0.31) plus a guard that found no harm. The guard does not show an improvement. The judge itself, zdpljp against
zdpljm, scored 119/128 (t 17.5), mostly because the minus side is degenerate, as predicted above.

## 14. es-lr1: fixed-centre ES from the accepted step (November 16, 2026)

What es-lr0 showed: the per-iteration step rule (move each coordinate whose |t| exceeds 2) moves by chance and
degrades the centre (gates 17, 16, 13, 10, 11, 14 out of 32). The pooled gradient over 576 directions held a real
direction (cosine rising 0.10, 0.17, 0.31). So es-lr1 keeps the centre fixed and only collects directions for one
pooled proposal at a time.

- Centre θ1 = zdpljp (box runs/es/es-lr0/prop-0-17/plus/net.json, step baked into the weights, es_basis kept since
  the hidden layers are unchanged). σ is es-lr0's sigma.json, not recalibrated: θ1 is only TV 0.1 from θ0.
  `step_frac` 0, so the centre never moves. Same 160 games per iteration, fitness and floors as es-lr0. Pool
  packages zdpg39 and zdpljp. Package prefix zdpv, seed base 3950000. One anchor gate after iteration 0 (θ1
  against zdpf005, 32 games).
- Checks (es_propose over all directions so far, from θ1), pre-set:
  - After 6 updates: stop if the cosine is at most 0.1 and the |t| > 2 share at most 0.07.
  - After 12 updates: stop if the cosine is at most 0.1.
  - The first check, at 6, 12 or 18 updates, whose cosine is above 0.2 ends the run and the proposal is tested.
    After 18 updates without one, stop.
- Test of a proposal: the guard alone, θ1 + ηΔ (TV 0.1) against θ1, 128 games on row 2 of the judge maps (28 maps
  on the box, 4 on the VM), seeds 3800203 and 3800204. Accept if the score is at least 0.5; an accepted step starts
  es-lr2 under the same rules. The antithetic judge is dropped. It measures the direction, but a temperature
  coordinate makes the two sides so lopsided that it mostly scores the minus side's collapse (section 13: t 17.5,
  then a guard at 0.52). The cosine already supplies the direction evidence.
