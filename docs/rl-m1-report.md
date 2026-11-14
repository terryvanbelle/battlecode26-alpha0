# M1 known-answer test (rl-plan.md P1): FAIL

Run on 2026-11-14 from 00:20 to 04:50 PDT. Collection used only the box sandbox `~/rltest/bc26a0`. The GCloud VM ran only plain eval games, in a temp dir `~/m1tmp` that has since been deleted. The driver code is e68833a plus the edits listed under "Code edits".

## Verdict

**M1 fails.** After 8 iterations of 128 games the sabotage was not undone. On the fixed record set, P(sabotaged option) moved the wrong way, from 0.283 to 0.316. The score against frozen zdpf005 stayed flat: 11/64 at the start and 12/64 at the end, on the same seeds.

The plumbing checks pass:
- The sign of A is correct at team level.
- Parity holds and ratios start at 1.
- The clip fraction is about 0.02.
- Frozen points are byte-identical.
- No KL refusal happened.

The failure comes from signal-to-noise. The outcome-only score-function gradient only sees a team's sampling deviations: realised minus expected choices. That deviation has a standard deviation of about 5 decisions per team-game, and its correlation with z is -0.04. The strong team-level effect (in mirror games, the winner holds less in 76-90% of games) comes from the states the team is in, and the gradient cannot credit those states.

Pooled over all 1,024 games of the run, the gradient on the sabotaged bias has the right sign but t = -1.66. Per 128-game iteration, t is about ±0.5. To make the signal reliable you would need about 2,000 games per iteration, or a variance reducer worth about 16x.

That is ladder step 5 in section 7: "signal too weak at this scale; use the per-record critic V_R and a 512-game iteration". Desaturating the sabotaged point and using 256-game iterations (the ladder variants I tried) did not change this.

**What this means for M2:** with the current estimator and 128-256 games per iteration, M2 is very unlikely to find improvements smaller than this sabotage. This sabotage costs 0.5 → 0.17 in score. Don't start M2 as planned until the advantage estimator has a per-decision variance reducer (V_R / GAE), or switch to the section 7 last resort (antithetic ES on output biases). ES judges whole-policy perturbations by game outcome, and at team level this sabotage is very visible.

## 1. Sabotage and precondition

The tool is a new script, `scripts/sabotage.py`:
- `--net IN --out OUT --bias P:O:DELTA` adds DELTA to one output bias. Every other point and weight is copied value for value, norm.npz is copied, and the operation is recorded in a top-level `sabotaged` key.
- `--probs DATA... --net A --net B ... --point P` prints the mean floor-mixed option probabilities per net on the same records, plus the total variation distance to the first net.
- `--self-test` is wrapped by `tests/test_sabotage.py`, which passes on the box. It checks that the logit moves by exactly DELTA on that one option only, that nothing else changes, that norm is copied, that the probability report sees the shift, and that an out-of-range option is rejected.

Mean P(R_SCOUT) on point 5 uses 390,659 records from rl-m0 iteration 0 (net-v2 self-play, floor 0.005):

| Net | Mean P(R_SCOUT) | Total variation vs desat-b5 |
|---|---|---|
| net-v2 | 0.1545 | 0.004 |
| desat-b5 (base) | **0.1547** | 0 |
| +3 | **0.1652** | 0.011 |
| +5 | 0.1984 | 0.044 |
| +8 / +10 / +15 / +20 / +30 | 0.324 / 0.410 / 0.644 / 0.881 / 0.975 | n/a |

**Why +3 barely moves P(R_SCOUT).** Even after the tau-5 desaturation, point 5's logit margins are large. The gap between the best other option and R_SCOUT has quantiles 10/25/50/75/90 = -6.0 / 6.7 / 12.3 / 16.9 / 20.0. The plan's +3 / +5 assumed logits of order 1.

Precondition: each sabotaged build (non-RL, floor 0.005) against zdpf005, ZMAPS, seeds 31-34, both sides (AB), 64 games:

| Build | Score | Wilson 95% CI | Verdict |
|---|---|---|---|
| zdps3 (R_SCOUT +3) | 28/64 = 0.438 | [0.323, 0.559] | FAIL |
| zdps5 (R_SCOUT +5) | 30/64 = 0.469 | [0.352, 0.589] | FAIL |
| zdps15 (R_SCOUT +15; VM, stopped at 60 games) | 29/60 = 0.483 | [0.362, 0.607] | does not hurt (P(scout) 0.64) |
| **zdpk40** (KING_SPAWN S_HOLD = option 21, +40) | **11/64 = 0.172** | [0.099, 0.282] | **PASS: chosen** |
| zdpk60 (S_HOLD +60) | 1/64 = 0.016 | [0.003, 0.083] | too strong |
| zdpk40t2 (ks40 with point 11's last layer / 2; start of rl-m1b/c) | 12/64 = 0.188 | [0.111, 0.300] | PASS |

Even a large BABY_ROLE R_SCOUT bias does not hurt, so I used the plan's alternate: KING_SPAWN biased toward hold.

**Why +40 on hold.** S_HOLD (21) is a primitive that zdb never labels. Its logit sits 31 / 38 / 53 / 73 / 97 below the best option (quantiles 10/25/50/75/90). Biases of +2, +4 and +8 change nothing: P(hold) stays at the floor value of 0.00023. On the fixed record set:

| Hold bias | Mean P(hold) | KL to base |
|---|---|---|
| +30 | 0.087 | 0.68 |
| +40 | 0.283 | 2.30 |
| +60 | 0.603 | 5.0 |

**Start for M1:** `runs/m1/ks40/net.json`, which is desat-b5 with +40 on point 11 option 21. Against zdpf005 on 128 games (seeds 31-38, AB) it scores 11/64 + 9/64 = **20/128 = 0.156, Wilson [0.103, 0.229]**. Seeds 35-38 were played on the VM, whose build was checked: the W.java P[] md5 matches the box build, and a reference zdpf005-vs-zdb game replayed with an identical traj_hash.

## 2. Run rl-m1 (configs/rl-m1.json, sandbox only)

The config is a copy of rl-m0.json with these settings:
- name rl-m1, package_prefix zdpq
- start_net runs/m1/ks40/net.json; anchor_net net-v2, used only for diagnostics
- anchor_package zdpf005
- trainable_points [11]
- games_per_iter 128, min_games 128, mirror_fraction 0.6, floor 0.005, lambda 1 (z − V, paired baseline)
- train: lr 1e-3, epochs 2, batch 4096, clip 0.2, target_kl 0.1, beta 0, **no_kl true**
- gates disabled (anchor_every and promote_every 1000)
- workers_vm 15, box_slice "0:0:4", so everything runs on the box

target_kl 0.1 and lr 1e-3 follow the M0(e) finding that the defaults move the policy too slowly.

With `--no-kl`, beta is 0 and the collapse and max_anchor_kl refusal is off. The anchor is used only for the kl_anchor and option-histogram diagnostics. A new test shows that a far anchor neither refuses nor changes the export. The pool is v0, which is the sabotaged net (base_permanent), plus its own snapshots v2, v4 and v6. No zdpn2 or desat-b5 was in the pool.

Commands (box):
```
python scripts/sabotage.py --net runs/m0/desat-b5/net.json --out runs/m1/ks40/net.json --bias 11:21:40
python scripts/build_dpnet.py --net runs/m1/ks40/net.json --name zdpk40 --floor 0.005
python scripts/eval_student.py --player zdpk40 --opponents zdpf005 --maps <ZMAPS> --seeds 31,32,33,34 --sides AB --workers 8 --out runs/m1/pre/zdpk40
bash scripts/rl_loop.sh configs/rl-m1.json rl-m1 8
```

Time per iteration:
- collection: 12-16 min (the sabotaged mirror games are short)
- rl_data: about 35 s
- train: about 8 s

There was no box_wait idle, thanks to the fix below. The 8 iterations took 01:36 to 03:20 PDT.

### Per-iteration curve (point 11, P = P(S_HOLD))

Column meanings:
- **fixedP**: mean P(hold) of net v on the fixed record set: 230k net-v2 records from rl-m0 iteration 0. Base desat-b5 is 0.00023; sabotaged v0 is 0.2828.
- **own start/end**: the trainer's mean on the iteration's own records, before and after the update.
- **KL old→new**: the trainer's mean KL between the iteration's start and end nets (`kl_old_new`).
- **KL anchor**: KL to net-v2. Diagnostic only.
- **early stop**: the step at which the target_kl early stop fired, or "-" if it did not.
- **SNR cos**: the trainer's gradient-SNR cosine.
- **V Brier**: held-out Brier score of V; the constant predictor scores about 0.25.
- **Learner vs pool**: the learner's score per pool member.

| v | fixedP | own start → end | KL old→new | KL anchor | clip | steps | early stop | SNR cos | V Brier / AUC | Learner vs pool |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.2828 | 0.585 → 0.583 | 0.0033 | 4.83 | 0.025 | 162 | - | -0.05 | 0.207 / 0.75 | (all mirror) |
| 1 | 0.2810 | 0.531 → 0.530 | 0.0137 | 4.39 | 0.024 | 132 | - | 0.00 | 0.158 / 0.88 | v0 0.558 (52) |
| 2 | 0.2847 | 0.609 → 0.612 | 0.0033 | 5.07 | 0.022 | 118 | - | 0.00 | 0.156 / 0.90 | v0 0.577 (52) |
| 3 | 0.2876 | 0.638 → 0.645 | 0.0239 | 5.35 | 0.023 | 10 | step 10 | 0.30 | 0.186 / 0.78 | v0 0.54, v2 0.50 |
| 4 | 0.3071 | 0.654 → 0.653 | 0.0046 | 5.41 | 0.021 | 118 | - | 0.27 | 0.164 / 0.84 | v0 0.50, v2 0.42 |
| 5 | 0.2995 | 0.521 → 0.522 | 0.0075 | 4.32 | 0.025 | 126 | - | -0.01 | 0.188 / 0.77 | v0 0.56, v2 0.56, v4 0.69 |
| 6 | 0.3074 | 0.539 → 0.537 | 0.0065 | 4.46 | 0.023 | 146 | - | -0.03 | 0.164 / 0.83 | v0 0.61, v2 0.50, v4 0.50 |
| 7 | 0.3087 | 0.650 → 0.653 | 0.0146 | 5.43 | 0.022 | 106 | - | 0.26 | 0.169 / 0.84 | v0 0.57, v2 0.50, v4 0.58, v6 0.42 |
| 8 (final) | **0.3159** | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Learner-vs-pool scores are all within noise of 0.5 (12-52 games each).

Ratio and parity: rl_data parity passed every iteration (median about 9e-5). The trainer's start ratio max |log r| was 5e-4. The early stop fired once, at iteration 3.

### Pass criteria

1. **P(hold) falls by at least half the gap.** The gap is 0.2828 → 0.00023, so the target is ≤ 0.1415. Final v8 is **0.3159**, up 0.033. **FAIL.** On own records it went from 0.585 at the start of iteration 0 to 0.653 at the end of iteration 7.
2. **Score against frozen zdpf005 reaches at least 0.45 from at most 0.35.** The start is 20/128 = 0.156 [0.103, 0.229].
   - Final v8 (non-RL build at floor 0.005, played on the VM) scored **12/64 = 0.188 [0.111, 0.300]** on seeds 31-34. The start scored 11/64 on the same seeds.
   - I skipped seeds 35-38 for v8: criterion 1 had already failed and the VM plays only about 0.6 games/min.
   - A mid-run checkpoint of v4 was stopped at 0/11 to free the VM for the next precondition.
   - **FAIL**: no rise.
3. **Frozen points byte-identical.** **PASS.** In nets/8/net.json every point except 11 is JSON-equal to nets/0, and so are all top-level keys (version, hidden, desaturated, sabotaged). Only point 11 changed.

## 3. Ladder (section 7)

1. **Sign of A per team in mirror games.** Correct. On iterations 0-7, the team-game hold fraction correlates with z at -0.22, -0.39, -0.33, -0.36, -0.29, -0.20, -0.26, -0.23. In decided mirror games the winner holds less in 76% (iteration 0) and 89% (iteration 1) of games.
   - On iteration 1, mean A is -0.05 on hold records and +0.07 on the rest (z − V).
   - V is weak. On the iteration 0+1 window, out-of-loop V had val Brier 0.294 against 0.252 for a constant predictor. In-loop val Brier ranged 0.16-0.21 on 13 to 26 held-out games.
2. **Ratio, clip and early stop.** Healthy: start |log r| ≤ 5e-4, clip fraction 0.021-0.025, one early stop in 8 iterations. The steps are not limited by the trust region. KL old→new stays at 0.003-0.024 because the gradient is mostly noise:
   - The trainer's SNR cosine sits around 0, except iterations 3, 4 and 7 at 0.26-0.30, which pushed hold up rather than down.
   - Offline sweeps on iteration 1 data: larger steps (lr 3e-3 / 1e-2 / 3e-2) mostly move other options. Fixed-record P(hold) went 0.282 → 0.286 / 0.309 / 0.353. A no-value variant (A = z, new `--no-value`) is no better: fixed P went 0.282 → 0.283 / 0.294 / 0.320 at lr 1e-3 e2 / 1e-3 e8 / 3e-3 e8.
3. **Gradient on the sabotaged bias itself.** This is the decisive diagnostic. Per game I summed A·∂log π(k)/∂b_hold, with the exact floor mixture. Per 128-game iteration t was:
   - iterations 0-3: -0.48, -0.68, +0.38, -1.95
   - iterations 4-7: +0.08, -0.01, -0.11, -2.78
   - pooled over 1,024 games: **mean -0.29, se 0.17, t = -1.66**

   Two things explain it:
   - **Saturation.** At the start only 19% of point-11 records have the softmax P(hold) inside [0.01, 0.99]. The gradient of log π through the floor mixture is zero on saturated records, and also on floor-sampled rare options, because the IS factor s_k/π_k ≈ 0.
   - **Credit.** The score function credits only the sampling surprise: realised minus expected holds per team-game. Its standard deviation is 4.9 decisions and its correlation with z is -0.04, against -0.2 to -0.4 for the state-driven hold fraction. In most states hold is equivalent to the other non-spawn exits (S_CAP, S_NO_SURPLUS, S_FLOOR), so most decisions carry no outcome signal.
4. **Desaturated start (rl-m1b).** Same config, start `runs/m1/ks40t2` (ks40 with point 11 last layer / 2). It scores 12/64 = 0.188 vs zdpf005, so it is still sabotaged, and 54% of records now carry gradient. After 1 iteration (128 games): fixed P 0.2774 → 0.2753; own 0.368 → 0.367; KL old→new 0.002; SNR cos -0.14; bias-gradient t +0.54 (wrong sign). I stopped it to run step 5.
5. **256 games per iteration (rl-m1c).** Desaturated start ks40t2. Iteration 0 (256 mirror games, 41 min): fixed P 0.2774 → 0.2746; own 0.385 → 0.380; SNR cos -0.03; bias-gradient t +0.21; team corr(hold fraction, z) only -0.06. I stopped it during iteration 1 at 70/256 games. The trend did not change, and each iteration takes about 45 min.
6. **KING_SPAWN sabotage.** This is already the sabotage used, because R_SCOUT did not satisfy the precondition.
7. **Not done:** the per-record critic V_R, 512-game iterations, and ES on output biases. From the pooled t-stat, plain more data would need about 2,000 games per iteration to give t ≈ 2 per iteration.

## Code edits (driver working tree ~/projects/vibe/bc26a0, not committed; synced to the sandbox)

- **New `scripts/sabotage.py`:** the bias sabotage, the `--probs` report and `--self-test` (described in section 1).
- **New `tests/test_sabotage.py`:** pytest wrapper of the self-test. It passes on the box.
- **`scripts/rl_loop.sh`:**
  1. `collect()` skips `box_request`/`box_wait` when the config's box_slice is empty (LO ≥ HI, e.g. "0:0:4"). This removes the 300 s idle per iteration on a single host. rl-m1's loop.log has no BOX lines, and collection is only game time.
  2. `train()` passes `--no-kl` when the config has `train.no_kl: true` (M1 only). This sets beta to 0 and turns off the alarm-driven beta doubling and the max_anchor_kl refusal. The anchor stays a diagnostic.
- **`scripts/rl_train.py`:** new `--no-value` flag (V = 0, A = z before normalisation and the pair baseline). It was used in the offline diagnostics only.
- **`tests/test_rl_train.py`:** new check 5. Training with `--no-kl` against a far anchor (KL 1.30, above `--max-anchor-kl 0.01`) exits 0 with beta 0, and the export is byte-identical to the one trained with the anchor equal to the net. So the anchor neither refuses the export nor leaks into the gradient. The whole test passes on the box, as do test_rl_loop, test_desaturate and test_sabotage.

`.claude/` is untracked in the driver repo and is not mine.

## Paths (box ~/rltest/bc26a0)

- **Configs:** `configs/rl-m1.json`, `configs/rl-m1b.json`, `configs/rl-m1c.json` (sandbox only)
- **Nets:**
  - `runs/m1/sab{3,5,10,15,20}/net.json` (R_SCOUT)
  - `runs/m1/ks40`, `runs/m1/ks60` (S_HOLD)
  - `runs/m1/ks40t2`
- **Precondition games:** `runs/m1/pre/{zdps3,zdps5,zdpk40,zdpk60}/`. zdps15, zdpk40 seeds 35-38, zdpk40t2 and zdpq8n were played on the VM. Their scores are above, and their directories were deleted with the VM temp dir.
- **Runs:** `runs/rl/rl-m1/` (nets/0-8, progress.jsonl, loop.log, iter-0006 and 0007 data), `runs/rl/rl-m1b/` (1 iteration), `runs/rl/rl-m1c/` (iteration 0 plus a partial iteration 1)
- **Kept point-11 data for all rl-m1 iterations:** `runs/m1/keep/iter-000{0..7}/` (p11.npz, trace.npz, summary.json, games.jsonl)
- **Diagnostics:**
  - `runs/m1/curve.py RUN POINT OPTION FIXED_DATA` (per-iteration table)
  - `runs/m1/gbias.py RUN ITERS` (bias-gradient t-stat)
  - `runs/m1/score.py DIRS` (score with Wilson CI)
  - `runs/m1/offline/{a..g}` (offline lr/no-value sweeps)
- **Builds** (box build/botsrc): zdps3, zdps5, zdpk40, zdpk60, zdpk40t2, zdpq*, zdpu0/1, zdpw0/1
- **Disk:** about 450 MB under runs/m1 and runs/rl/rl-m1*
- **VM:** `~/m1tmp` deleted, no processes left. Another agent's `~/p3/bc26a0` RL run is writing to the VM disk, which now has 2.0 GB free.
- **Box:** all my loops and games are stopped, and the box is idle.
