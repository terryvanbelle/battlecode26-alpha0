# Training progress

Generated from repository JSONL measurements at `2026-11-10T03:35:41+00:00`.

Run `python scripts/plot_progress.py --progress-dir progress/campaign-v2` from the repository root to regenerate these figures and this report.

## Latest training measurements

| Measurement | Observed value |
|---|---:|
| `iteration` | 3 |
| `games_total` | 24 |
| `positions_total` | 11684 |
| `policy_loss` | 2.78214 |
| `value_loss` | 0.65177 |
| `total_loss` | 3.43391 |
| `entropy` | 2.80034 |
| `games_per_hour` | 42.0406 |
| `wall_seconds` | 1470.66 |

![Training loss, entropy, data volume and throughput](training.png)

## Learning health

These are **training-batch diagnostics, not held-out calibration**. Value BCE fits win logits; value MSE and mean use predictions on the −1 to +1 scale. Saturation is the fraction with |value| > 0.95.

Replay counts describe the current buffer, not cumulative data. Searched positions and actual decisions describe each training iteration; their ratio shows search coverage. Missing measurements remain gaps and are not filled with zero.

![Training-batch value diagnostics, replay size and search coverage](learning-health.png)

## Latest benchmark checkpoint

Iteration **2**, checkpoint `checkpoint-0002.pt [718f0843/macro_v2/greedy/ef6f893e]`.

**24 completed games; 0 errors; 12/12 panel opponents represented.**

Equal-weight mean game score: **16.7%** across all frozen panel opponents. This is not an Elo estimate.

Paired-map bootstrap interval unavailable: 1 shared map–seed cells have both sides completed against every panel opponent; at least 4 are required. A small smoke suite is insufficient for a strength claim.

| Opponent | Completed | W–D–L | Mean score | Descriptive 95% Wilson interval | Errors | Sides A/B |
|---|---:|---:|---:|---:|---:|---:|
| `Krak9n_robot` | 2 | 2–0–0 | 100.0% | 34.2%–100.0% | 0 | 1/1 |
| `seru9_TOTAL_AGGRO` | 2 | 2–0–0 | 100.0% | 34.2%–100.0% | 0 | 1/1 |
| `Camleaf_mk1` | 2 | 0–0–2 | 0.0% | 0.0%–65.8% | 0 | 1/1 |
| `CS50Crab_rapierV1` | 2 | 0–0–2 | 0.0% | 0.0%–65.8% | 0 | 1/1 |
| `spsquared_TSPAARK_QUALS` | 2 | 0–0–2 | 0.0% | 0.0%–65.8% | 0 | 1/1 |
| `HugoIngelsson_Xzibit` | 2 | 0–0–2 | 0.0% | 0.0%–65.8% | 0 | 1/1 |
| `abcdezyxwv_bestsofar` | 2 | 0–0–2 | 0.0% | 0.0%–65.8% | 0 | 1/1 |
| `IvanGeffner_basic51` | 2 | 0–0–2 | 0.0% | 0.0%–65.8% | 0 | 1/1 |
| `bc26-example` | 2 | 0–0–2 | 0.0% | 0.0%–65.8% | 0 | 1/1 |
| `vibe26b_zdb` | 2 | 0–0–2 | 0.0% | 0.0%–65.8% | 0 | 1/1 |
| `thewaxmango_collegejunior` | 2 | 0–0–2 | 0.0% | 0.0%–65.8% | 0 | 1/1 |
| `uravt_Version41` | 2 | 0–0–2 | 0.0% | 0.0%–65.8% | 0 | 1/1 |

![Benchmark scores and completion counts](evaluation.png)

![Per-opponent results with uncertainty](opponents.png)

## Interpretation

- Game score is win = 1, draw = 0.5, loss = 0. Errors/timeouts are excluded from the completed-game score and listed separately; inspect completion counts before comparing checkpoints.
- The panel mean weights opponents equally. A partial panel is labelled and is not comparable to a full panel. Different map/seed or side coverage can also change the score.
- Wilson intervals are descriptive per-game intervals. With draws they use fractional successes. Mirrored sides and repeated maps are correlated, so these are not paired-map uncertainty estimates or evidence of a statistically significant improvement.
- The panel bootstrap resamples complete shared map–seed cells, keeping both sides and every opponent together (2,000 resamples, fixed RNG seed). It uses the balanced intersection of completed cells; this can differ from the all-completed-game mean. Four cells is the calculation minimum, not a guarantee of adequate power. Bootstrap intervals can collapse when every observed cell has the same outcome.
- Distinct checkpoints are never pooled. Duplicate checkpoint/opponent/map/seed/side cells keep their final recorded outcome and do not increase sample size.
- Loss reductions describe fitting to self-play targets; benchmark outcomes measure playing performance. Neither a small smoke suite nor self-play win balance establishes strength against the wider field.
