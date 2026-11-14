# RL plan: self-play PPO-lite over the zdp decision points (zdpr)

Status: plan, 2026-11-14. It starts from **zdpr**, the design all three judges favoured as most buildable and most rule-compliant, and adds grafts from the other two designs. Design rationale lives in the judge verdicts. This document covers what we build and run.

## 0. Principles (binding)

- **Self-play only.** The opponents are the learner and its own past snapshots, with zdpn2 as snapshot 0. zdb is a bootstrap fallback only, behind an explicit flag that is noted in the README. Field bots are never trained against and are never used to pick hyperparameters or stages. A field result may only **stop or roll back** a run.
- **Outcome-only objective.** Reward is z = +1 win, -1 loss, 0 draw. Learned values are used only as baselines (state-only, so they add no bias). There are no count bonuses and no material reward.
- **Capability-complete start.** We start from zdpn2. All 11 points keep their primitive options, with a probability floor > 0. A KL leash to zdpn2 is never fully turned off, except in the M1 known-answer test.
- **Bytecode limits stay off.**
- **Lean.** No engine/Game.java changes, no GPU, no learned critic in v1, no exploiter. Each extra piece is added only when a measured failure calls for it (section 7).

## 1. Facts checked in the code that the plan relies on

| Fact | Consequence |
|---|---|
| `Decisions.java:38` accepts a record iff `len == 4 + charAt(3)` | A trailer counted inside n (`F[3]`) passes through unchanged. No harness change is needed. |
| `D.pick`/`D.pick2` call `DNet.sample`; **`DK.netPick` (DK.java:326) calls `DNet.sample` directly** | Stash k/p/mask inside `DNet.sample`. Copy them into the slot at all three call sites. |
| `D.NET_FLOOR` source default is 0.02f, but zdpn2 was built with `build_dpnet.py --floor 0.001` (pipeline_dp2.sh:20) | The zdpn2 reference is floor 0.001. M0 measures the cost of a higher RL floor against that. |
| `Game --trace N` already writes per-team babies, kings, cheese, xfer and rattraps every N rounds | The value model needs nothing new. Points, king HP and cat damage are skipped in v1. |
| net.json has normalisation folded into layer 1, and mu/sd are not saved | The trainer unfolds exactly: with any (mu, sd), `W1 = W1f*sd`, `b1 = b1f + W1f@mu`. It refolds on export. |
| Disk: 30 GB, 3.8 GB free | Use a disk guard and delete raw gz after npz conversion. |
| Point ids: MOVE 1, ACT_PRE 2, ACT_POST 3, FACE 4, ROLE 5 (R_GATHER 0, R_SCOUT 1, R_FIGHT 2, R_RAIDER 3, R_ATTACK 4), COMMS 6 (C_NONE 0), MINE_PICK 7, KING_MOVE 10, SPAWN 11, ACT 12, FORM 13 | These are used by the milestone tests. |

## 2. Algorithm

**Policy.** Each point keeps its current MLP (64,64). The behaviour policy is exactly what `DNet.sample` computes:
mu(k|x,mask) = (1-f)·softmax_mask(z/T)_k + f/m, with T = 1 and f = the RL floor.
The trained policy pi_theta has the same parametric form, so the ratio r = pi_theta(k)/mu_logged(k) is exact. The gradient flows through the softmax part.

**Loss per trainable point p** (each point is its own net with its own Adam, so no cross-point weighting is needed):

```
L_p = -mean_i[ min(r_i A_i, clip(r_i, 0.8, 1.2) A_i) ]
      + beta_p * mean_i KL(pi_theta(.|x_i) || pi_zdpn2(.|x_i))
      - c_ent * mean_i H(pi_theta(.|x_i))          (masked)
```

- Records with m = 1 allowed, or k = none, are dropped.
- **On-policy.** Only records from the current learner's sides are used. Snapshot-side records are not recorded at all.
- Each iteration starts with Adam reset.
- An epoch stops early for a point when the approximate KL(theta_k → theta) exceeds 0.02.

**Advantage (default, lambda = 1).** A_i = z_team − V(s_team(t_i)). V is a logistic regression on the team trace at the record's 10-round bin.

- Inputs for both us and them: babies, kings, cheese, transferred (cumulative and Δ over the last 50 rounds), rattraps, the us − them differences of each, round/2000, and map area.
- Fit V each iteration on a rolling window of the last ~3 iterations' learner team-games, with 10% of games held out. Log Brier score.
- Optional upgrade: a 32-unit MLP, used only if it improves Brier by more than 0.01.
- **Paired-game baseline.** Every learner-vs-snapshot job is played as a seat-swapped pair (same map and seed). For those records use A_i = (z − V) − mean over the pair of (z − V) for the learner's two games. This cancels map and seat luck. Mirror games are already zero-sum.
- Advantages are normalised per point per iteration (mean 0, sd 1) and clipped to ±5.
- All robots on a team share the team signal. There is no per-robot credit.

**Fallback.** When the sampled k cannot act and zdb's cascade runs, that is part of the environment, so the gradient still uses log pi(k). The fallback flag is logged per (point, option) for diagnostics. Options whose fallback rate is at least 99% become candidates for structural masking in a later build.

**Why not more.** A record critic, GAE with lambda < 1, and shaping are deferred to Phase B (section 5, P3) and are added only if the gradient-SNR diagnostic says the baby points need them.

## 3. Hyperparameters (start values)

| Name | Value |
|---|---|
| RL floor f | **0.005**. Change it per the M0 floor test: use 0.01 if it costs at most 3 pp vs zdb, or 0.002 if 0.005 costs more than 3 pp. |
| Temperature T | 1 (training and gates) |
| Epochs / minibatch / lr | 2 / 4096 / Adam 3e-4, decaying to 1e-4 after the first promotion |
| PPO clip / early-stop KL | 0.2 / 0.02 per point |
| Grad-norm clip | 0.5 per point |
| beta (KL to zdpn2) | 0.03, annealed linearly to 0.01 over 20 iterations, never 0 (except in M1) |
| Collapse rule | If a point's mean KL to zdpn2 exceeds 0.3, multiply beta_p by 2. If any point exceeds 0.5, refuse to export and roll back. |
| c_ent | 0.003 |
| Iteration size | 128 games = 64 seat-swapped pairs (Phase A). 256 games when the baby points unlock. |
| Game mix | 60% mirror (both sides recorded); 40% learner vs pool (learner side recorded) |
| Pool | zdpn2 (permanent), then every 2nd iteration's snapshot. Keep the last 8, plus zdpn2 and every promoted best. Choice is uniform until iteration 10, then PFSP-lite with weight ∝ (1 − winrate_vs_member)^1, where winrate is an EMA over the last 2 iterations. zdpn2 gets at least 10% of games. |
| Seeds | 1_900_000 + 1000·iteration + pair index |
| Trace | `--trace 10` |
| Training maps | 54 maps: all 74 minus teacher-dp2's `excluded_maps` (4 ancestral + 8 GMAPS) minus the 8 ZMAPS. Anchor and zdb gates run on ZMAPS, the field on GMAPS, so every reported number is on maps RL never saw. |

## 4. Code changes

### Java (java/bots/zdp). All recording only. The trajectory must stay identical to the non-RL build.

- **DNet.java.** `sample()` additionally sets the statics `lastK`, `lastP` (the floor-mixed probability of the returned k, exactly the p[] value used) and `lastMask` (reference to the `allowed` array). No behaviour change.
- **D.java.**
  - Add `static final boolean RL = false;` and `static final int NET_VERSION = 0;` (both rewritten by build_dpnet).
  - Add slot statics `pk, pp, pmask` (primary) and `pk2, pp2, pmask2` (secondary). They are set to `pk = -1` when a point opens. `pick()`/`pick2()` copy DNet's last* values after sampling.
  - `emit()`/`emit2()`: when `RL`, append the trailer before setting `F[3] = n - 4` (n now includes the trailer). The trailer is:
    - `k` (0xFFFF if no pick)
    - `q = min(65535, round(-ln(p_k)·4096))`
    - `flags`: bit0 fallback, bit1 secondary slot
    - `ceil(nOpt/16)` mask words (all ones if the mask is null; BABY_MOVE has 4)
    - `NET_VERSION`
    - `trailerLen` (last char, so the parser self-checks)
  - Fallback is computed at emit as follows. For BABY_MOVE it is `(lab & 0x8000) != 0`. For other points it is `k >= 0 && lab != k`. This is a diagnostic only, so no executor edits are needed in v1.
  - Size check: the largest record is BABY_MOVE, about 64 features + 1+1+1+4+1+1 = 73, which is under the 96-char F.
- **DK.java.** `netPick` copies DNet's last* into D's primary slot (king records go through D.emit).
- Build-time check: with `RL=true`, zdpn2 must replay `traj_hash`-identically to the non-RL zdpn2 build.

### Scripts

- **build_dpnet.py.**
  - Add `--rl` (sets `D.RL=true`) and `--version N` (sets `D.NET_VERSION`).
  - Write rl, floor, version and the net.json path into BUILD.json.
  - Copy net.json to `runs/rl/<run>/nets/<N>/net.json`, which is what the parity check reads.
- **decisions.py.** Add a `read_v2`/`split_trailer` path: if the last char equals `len − nFeat(point)`, split off the trailer. It stays backward compatible with v1 files. Vectorise it like `train_dp.read_fast`.
- **New `scripts/dpnet_io.py`.** Factored out of train_dp.py: `load_net`, `forward`, `fold`, `unfold`, `export`, plus the Java-identical floor-mixed probability in float32. train_dp.py imports it, and from now on also stores mu/sd in a sidecar `norm.npz`.
- **New `scripts/rl_schedule.py`.** Inputs: iteration, `runs/rl/<run>/pool.json` and `configs/rl-v1.json`. Output: `iter-NNNN/jobs.json`. Each job lists map, seed, sides, the A and B package names, the recorded sides and a pair id. It also updates the pool and PFSP weights from the previous iteration's results.
- **New `scripts/rl_collect.py`.** A fork of collect_teacher.py's runner. It runs jobs.json with two arbitrary bot packages, `--decisions` on the recorded sides, and `--trace 10`. It supports `--slice LO:HI:N` for box_split and `--deadline` (unix time), after which unstarted jobs are skipped. It is resumable.
- **New `scripts/rl_data.py`.** Per game it:
  - parses the trailers;
  - **runs the parity check**: recompute log p from `nets/<NET_VERSION>/net.json`, the recorded mask, floor and T. Pass requires a median |Δlogp| below 1e-3 and 99.9% of records below 2e-3, otherwise the iteration aborts. The 1/4096 quantisation is about 2.4e-4.
  - joins in the trace bin and z from result.json;
  - writes `iter-NNNN/data/p<point>.npz` with x, k, logq, mask, flags, team, round, game id, pair id, z and trace features, for **trainable points plus a 5% sample of the frozen points** (for diagnostics);
  - deletes the decisions-*.bin.gz files once the npz is verified.
- **New `scripts/rl_train.py`** (about 300 lines of numpy/torch, CPU, 8 threads). It:
  - fits V and computes A;
  - unfolds theta_k, runs PPO per trainable point, refolds and exports `nets/<k+1>/net.json`, refusing on the KL guard;
  - writes `metrics.json`;
  - takes `--lr 0` (identity test), `--points`, `--no-kl`, and `--fake-reward point:option` (planted test: A = +1 if k == option else 0 for that point, and other points get no gradient).
  - Frozen points are copied byte-for-byte from the input net.json.
- **New `scripts/rl_loop.sh`.** The driver. Each iteration is marked done per stage (marker files) and honours `runs/rl/STOP`:
  1. disk guard (abort if under 2 GB free)
  2. build `zdpr<k>`
  3. `box_request` 3/4 slice, then VM slice, `box_wait`, VM fill-in, all bounded by a deadline of T+16 min, with a minimum of 64 games, otherwise wait
  4. rl_data
  5. rl_train
  6. build `zdpr<k+1>`
  7. gates on schedule
  8. prune: keep compiled packages for pool members only, raw data for the current iteration only, and npz for the last 2 iterations
  9. append a row to `runs/rl/<run>/progress.jsonl`
- **New `scripts/rl_gate.sh`.** A thin wrapper over strength_check.sh / eval_student.py with `--player zdpr<k>`, the reference zdpn2, ZMAPS and Wilson CIs.
- **configs/rl-v1.json.** Maps, mix, pool rules and every hyperparameter in section 3, so a run is fully described by its config.
- **docs/rl-selfplay.md.** Results log. Update the README Current work block at each milestone, with times in PDT (UTC−7).
- **tests/.** Three checks:
  - trailer round-trip: a Java-format string through the Python parser;
  - the RL build replays the non-RL build on 2 jobs;
  - an lr=0 export is byte-identical, and frozen points are byte-identical.

Not reused: alpha0/ MCTS and replay code, which is tied to the macro-action format. Only patterns are reused (atomic json, STOP file, resume markers, Wilson CIs).

## 5. Phases, milestones and gates

### P0 — Plumbing (M0). About half a day of coding plus about 2 h of games. No learning claims.

| Check | Pass |
|---|---|
| a. RL build of zdpn2 vs non-RL zdpn2 | traj_hash identical on 8 jobs |
| b. One 64-game self-play iteration at lr = 0 | parity passes; all ratios r = 1 within quantisation; per-point record counts and fallback rates logged; V held-out Brier < 0.20; exported net.json and W.java byte-identical |
| c. Floor test: zdpn2 at floor 0.001 / 0.005 / 0.01 vs zdb, 64 games each (ZMAPS, seeds 31–34) | pick the largest floor costing ≤ 3 pp vs the 0.001 build |
| d. Saturation report on the trainable points | fraction of records with p_k > 0.999; if > 90% on a trainable point, see section 7 (desaturation) |
| e. Planted-reward direction test: 2 iterations of `--fake-reward 6:0` (C_NONE on COMMS), only COMMS trainable | P(C_NONE) rises on COMMS records; all other points byte-identical; the bot built from the new net shows the shift in its own records (proves sign, ratio and the W.java round trip) |
| f. V sanity on teacher-dp2 + M0 games | V's AUC predicting z at rounds 500/1000/1500 is above 0.7 (otherwise lambda = 1 still works; V is just a weak baseline) |

### P1 — Known-answer learning test (M1). About 3 h.

- Sabotage zdpn2: add +3 to BABY_ROLE's output bias for R_SCOUT.
- Precondition: the sabotaged net scores ≤ 0.35 vs zdpn2 over 64 games on ZMAPS. If it does not, raise the bias to +5. If that also fails, use the alternate sabotage: KING_SPAWN biased toward the "hold" option.
- Run: only BABY_ROLE is trainable; KL leash **off**; pool = the sabotaged net and its own snapshots only (no zdpn2, so the answer cannot leak); lambda = 1; 128 games/iteration; up to 8 iterations.
- **Pass** requires all of:
  - mean P(R_SCOUT) falls by at least half the gap to zdpn2's role distribution;
  - vs frozen zdpn2 on 128 ZMAPS games, the score rises from ≤ 0.35 to ≥ 0.45, with Wilson CIs not overlapping the start;
  - frozen points stay byte-identical.

### P2 — Phase A real RL (M2). About 30 iterations, about 8 h, about 3,800 games.

- Start from zdpn2, possibly desaturated (section 7), with the full section 3 settings.
- Trainable points: BABY_ROLE (5), MINE_PICK (7), KING_MOVE (10), KING_SPAWN (11), KING_ACT (12), KING_FORM (13). These are low-volume, high-leverage points. All other points stay frozen at zdpn2.
- Cheap anchor every 3 iterations: vs zdpn2, 32 games (ZMAPS × 2 seeds × 2 sides). Trend only.
- **Promotion gate** every 6 iterations, or when the anchor reaches ≥ 0.56: vs zdpn2, 128 games (ZMAPS × 8 seeds × 2 sides). **Promote** if the Wilson lower bound is above 0.50. Then run strength_check (zdb 64 games plus held-out field 192 games) on the promoted net.
- Field rule:
  - If the field score falls below zdpn2's field score minus 0.05, the version is not promoted, the run rolls back to best and lr is halved.
  - If this happens twice in a row, stop and investigate.
  - The field never changes PFSP, beta or staging.
- **M2 pass:** a promoted version with the Wilson lower bound above 0.50 vs zdpn2 (point estimate ≥ 0.55), plus the field rule satisfied.

### P3 — Phase B baby points. Unlock in stages; each stage starts only after the previous one passes a promotion gate.

1. BABY_ACT_PRE/POST and BABY_FACE.
2. BABY_COMMS.
3. BABY_MOVE.

- Iteration size: 256 games.
- The trainer subsamples MOVE/ACT/FACE records uniformly to keep at most 2M rows per point. This is unbiased because A is team-level.
- Before each unlock, run the **gradient-SNR diagnostic** per point: the cosine between policy gradients from two random halves of the iteration's games.
  - If the median cosine is below 0.05 for a newly unlocked point, apply the variance reducers in this order, A/B'd against the current setup for 10 iterations each:
    1. A per-record critic V_R (Python only): a 128,128 MLP over [point one-hot, the point's features zero-padded, the team trace at t]. It is a state-only baseline, A = z − V_R.
    2. lambda = 0.95 per 10-round bin, using V as a potential with gamma = 1, keeping lambda = 1 as the control arm.
    3. Double the games per iteration.

### Ongoing reporting

- **Bradley–Terry Elo** over the payoff matrix of pool members, filled for free from training games, with zdpn2 = 0. zdb and zdp-fixed are placed from gate games. This is the headline progress curve.
- **Gate 3, every 60 iterations:** round robin of promoted versions, 32 games per pair, to expose cycles.

## 6. Diagnostics (every iteration, one JSONL row per point)

- Per point: record count and share trainable; KL to zdpn2 and to theta_k; entropy (with an alarm on a drop of more than 50% vs zdpn2 → beta_p ×2); clip fraction; option histogram vs zdpn2 (alarm when one option exceeds 95% where zdpn2 had ≤ 70% → beta_p ×2); fallback rate per option; primitive-option share; parity median/p99; fraction of records with p_k > 0.999; gradient-SNR cosine.
- Per game: winner and reason, rounds, transferred, kings alive, babies at 500/1000/1500, time of the first king death.
- Learner win rate by opponent, V Brier score, games done by box vs VM, wall time per stage.

## 7. If a milestone fails

| Failure | Next step |
|---|---|
| M0 parity fails | Bug: float order, mask fit (`D.fit` resizing), wrong NET_VERSION mapping, or a trailer parse error. Do not train until it passes. |
| M0(d): saturated logits | **Desaturate cheaply first.** Scale each trainable point's last layer by 1/τ (τ = 2 or 3), which is the same as raising T and needs no training. Keep the KL anchor on the original zdpn2. Gate: vs zdb within 3 pp of zdpn2. If that costs strength, retrain with label smoothing 0.02 (`train_dp.py --label-smooth`, same data and seeds) and apply the same gate. |
| M1 fails (no recovery in 8 iterations) | Localised to the advantage/gradient path. Check, in order: (1) the sign of A per team in mirror games; (2) ratio distribution, clip fraction and early-stop rate; (3) rerun with 256 games/iteration; (4) the KING_SPAWN sabotage instead. If still flat with correct plumbing, the signal is too weak at this scale; use the per-record critic V_R and a 512-game iteration before anything else. |
| M2 stalls (no promotion in 30 iterations, no positive anchor slope) with M1 passed | (1) Restrict to KING_SPAWN, KING_FORM and BABY_ROLE (fewest options, highest leverage). (2) Add V_R. (3) Only then turn on the **zdb bootstrap fallback**: `--bootstrap-zdb 0.15` in the config, unrecorded zdb side, flagged in the README, removed once the learner beats zdp-fixed at ≥ 55%. |
| Field regression on promotion | Roll back to best and halve lr. Never tune against the field. |
| Box down | The deadline keeps the loop moving: VM only, 64-game minimum, about 25 min/iteration. |
| Last resort if PG cannot be made to work | Antithetic ES on the roughly 300 output biases of the Phase A points. Each pair is judged by 64 mirror games. Slow, but almost nothing can be wrong with it. |

## 8. Compute plan

Throughput is about 12 games/min combined: box 3/4 with 10 workers at -Xmx2g, VM 1/4 with 7 workers.

| Stage | Phase A (128 games) | Phase B (256 games) |
|---|---|---|
| Collect (box + VM) | ~11 min (deadline 16) | ~22 min (deadline 30) |
| rl_data + parity + npz | ~1 min | ~3 min |
| Train (VM CPU) | ~2 min (6 small points) | ~5 min |
| Build | < 1 min | < 1 min |
| Anchor (every 3rd iteration) | +3 min amortised to ~1 | same |
| **Per iteration** | **~16 min, ~3.5 it/h, ~85/day** | **~31 min, ~2 it/h** |

- Gates: a promotion gate (128 games, about 11 min) about every 6 iterations; strength_check (256 games, about 21 min) per promotion.
- Training runs sequentially after collection, so the loop stays strictly on-policy. Overlapping training with the next collection is a later optimisation only, if training time ever exceeds about 20% of the iteration.
- Disk:
  - raw gz: about 0.2 GB for the current iteration only;
  - npz: about 30–100 MB × 2;
  - nets: about 0.5 MB per iteration, all kept;
  - compiled packages: pool members only (≤ 12).
  - Steady state is under 1 GB. The 2 GB guard aborts collection.
- Memory: the trainer loads per point, at most 2M rows.

## 9. Timeline

| When | What |
|---|---|
| Day 1 | Java trailer + build_dpnet + decisions/dpnet_io + rl_schedule/collect/data/train/loop. M0 a–f. |
| Day 2 | M1 sabotage recovery (about 3 h), then start P2. |
| Days 2–3 | P2, about 30 iterations, promotion gates. M2 result, README update. |
| After M2 | P3 staged unlocks, with SNR-driven variance reducers as needed. |

## Results

### M0 on zdpn2 (November 14, 2026, 00:50 PDT; box sandbox; independently re-checked)

| Check | Result |
|---|---|
| a. RL build vs non-RL build | replay identical 8/8 |
| b. lr = 0 iteration (64 games) | Parity median 6.0e-5, max 1.22e-4 over 5.7M records. net.json is byte-identical and W.java's `P[]` is identical. V held-out Brier 0.18 on only 6 games; out-of-fold 0.255. |
| c. Floor vs zdb (64 games each) | 0.001: 33/64; **0.005: 33/64**; 0.01: 26/64. **The floor is 0.005.** |
| d. Saturation (share of records with p_k > 0.999) | 5: 0.99, 7: 0.86, 10: 0.43, 11: 0.88, 12: 0.995, 13: 0.96 |
| e. Planted `--fake-reward 6:0` (lr 1e-3, target_kl 0.1, 2 iterations) | Mean P(C_NONE) on the same records: 0.959 → 0.966 → 0.979. On matched seeds the bot's own C_NONE rate went from 0.973 to 0.988 (higher in 24 games, lower in 3; sign test p 5e-5). Only point 6 changed. |
| f. V AUC at rounds 500/1000/1500 | 0.51 / 0.64 / 0.69 out of fold. V is a weak baseline in mirror self-play; lambda = 1 still works. |

**Desaturation** (`scripts/desaturate.py`, scored vs zdb at floor 0.005, 64 games):

| Variant | Score |
|---|---|
| tau 2 on 5, 11, 12, 13 | 26/64 |
| tau 3 on 5, 11, 12, 13 | 27/64 |
| tau 2 on 11, 12, 13 | 27/64 |
| **tau 5 on 5 only** | **37/64**. Saturation on point 5 drops from 0.99 to 0.63. |

- Softening the king points costs strength.
- BABY_ROLE needs tau 5 to move at all.
- **RL start net:** net-v2 with point 5 scaled by tau 5. The KL anchor stays net-v2.
- Points 12 and 13 stay saturated and explore only through the floor. If they turn out not to learn, retrain them with label smoothing.
- None of these 64-game differences is significant (p ≥ 0.19).

### M1 known-answer test (November 14, 2026, 00:20–04:50 PDT): FAIL

The full report and an independent review are in [rl-m1-report.md](rl-m1-report.md).

**Sabotage.** The plan's BABY_ROLE R_SCOUT sabotage did not hurt: +3, +5 and even +15 still scored about 0.45 vs zdpn2, because point 5's logit margins are about 12. The test therefore used the alternate:
- KING_SPAWN S_HOLD at +40, giving P(hold) 0.28 on fixed records.
- Against frozen zdpn2 (floor 0.005) it scores **20/128 = 0.16**.

**Run.** 8 iterations × 128 self-play games, only point 11 trainable, no KL leash, pool of the sabotaged net and its own snapshots.
- P(hold) went the wrong way: 0.283 → 0.316.
- Score vs zdpn2: 11/64 → 12/64 on the same seeds.

**Plumbing is fine:**
- the advantage has the right sign per team, with corr(team hold fraction, z) between −0.2 and −0.4;
- parity passes and clipping is about 0.02;
- frozen points are byte-identical;
- the planted test passed in M0.

**Diagnosis: signal-to-noise.** The score-function gradient credits only a team's sampling *deviations* (realised minus expected holds, sd about 5 per team-game), and those barely correlate with z. The outcome effect of the sabotage comes from the states the policy reaches, which a state baseline absorbs. With the trainer's advantage, the pooled t for the sabotaged bias over all 1,024 games is −0.82. That works out to about 6,000 games per iteration for a reliable step. More games, desaturation (tau 2) and 256-game iterations did not help.

**Decision.** M2 does not start with the plain outcome-only PPO estimator. The next step is a design panel on the estimator (see below).
