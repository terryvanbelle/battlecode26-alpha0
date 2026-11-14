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
