# Training progress

Generated from repository JSONL measurements at `2026-11-10T04:20:44+00:00`.

Run `python scripts/plot_progress.py --progress-dir progress/campaign-v2` from the repository root to regenerate these figures and this report.

## Latest training measurements

| Measurement | Observed value |
|---|---:|
| `iteration` | 4 |
| `games_total` | 32 |
| `positions_total` | 15226 |
| `policy_loss` | 2.58595 |
| `value_loss` | 0.642089 |
| `total_loss` | 3.22804 |
| `entropy` | 2.58264 |
| `games_per_hour` | 44.935 |
| `wall_seconds` | 2111.58 |

![Training loss, entropy, data volume and throughput](training.png)

## Learning health

These are **training-batch diagnostics, not held-out calibration**. Value BCE fits win logits; value MSE and mean use predictions on the −1 to +1 scale. Saturation is the fraction with |value| > 0.95.

Replay counts describe the current buffer, not cumulative data. Searched positions and actual decisions describe each training iteration; their ratio shows search coverage. Missing measurements remain gaps and are not filled with zero.

![Training-batch value diagnostics, replay size and search coverage](learning-health.png)

## Latest benchmark checkpoint

Iteration **4**, checkpoint `checkpoint-0004.pt [1f04b952/macro_v2/greedy/ef6f893e]`.

Evaluation series: `standard [ef6f893e]`. Stages and evaluation configurations are reported separately.

**78 completed games; 0 errors; 5/12 panel opponents represented.**

Equal-weight mean game score: **50.0%** across the observed subset of opponents only. This is not an Elo estimate.

Paired-map bootstrap interval unavailable: 0 shared map–seed cells have both sides completed against every panel opponent; at least 4 are required. A small smoke suite is insufficient for a strength claim.

| Opponent | Completed | W–D–L | Mean score | Descriptive 95% Wilson interval | Errors | Sides A/B |
|---|---:|---:|---:|---:|---:|---:|
| `Krak9n_robot` | 16 | 16–0–0 | 100.0% | 80.6%–100.0% | 0 | 8/8 |
| `seru9_TOTAL_AGGRO` | 16 | 16–0–0 | 100.0% | 80.6%–100.0% | 0 | 8/8 |
| `Camleaf_mk1` | 16 | 8–0–8 | 50.0% | 28.0%–72.0% | 0 | 8/8 |
| `CS50Crab_rapierV1` | 16 | 0–0–16 | 0.0% | 0.0%–19.4% | 0 | 8/8 |
| `spsquared_TSPAARK_QUALS` | 14 | 0–0–14 | 0.0% | 0.0%–21.5% | 0 | 7/7 |
| `HugoIngelsson_Xzibit` | 0 | 0–0–0 | — | — | 0 | 0/0 |
| `abcdezyxwv_bestsofar` | 0 | 0–0–0 | — | — | 0 | 0/0 |
| `IvanGeffner_basic51` | 0 | 0–0–0 | — | — | 0 | 0/0 |
| `bc26-example` | 0 | 0–0–0 | — | — | 0 | 0/0 |
| `vibe26b_zdb` | 0 | 0–0–0 | — | — | 0 | 0/0 |
| `thewaxmango_collegejunior` | 0 | 0–0–0 | — | — | 0 | 0/0 |
| `uravt_Version41` | 0 | 0–0–0 | — | — | 0 | 0/0 |

![Benchmark scores and completion counts](evaluation.png)

![Per-opponent results with uncertainty](opponents.png)

## Interpretation

- Game score is win = 1, draw = 0.5, loss = 0. Errors/timeouts are excluded from the completed-game score and listed separately; inspect completion counts before comparing checkpoints.
- The panel mean weights opponents equally. A partial panel is labelled and is not comparable to a full panel. Different map/seed or side coverage can also change the score.
- Wilson intervals are descriptive per-game intervals. With draws they use fractional successes. Mirrored sides and repeated maps are correlated, so these are not paired-map uncertainty estimates or evidence of a statistically significant improvement.
- The panel bootstrap resamples complete shared map–seed cells, keeping both sides and every opponent together (2,000 resamples, fixed RNG seed). It uses the balanced intersection of completed cells; this can differ from the all-completed-game mean. Four cells is the calculation minimum, not a guarantee of adequate power. Bootstrap intervals can collapse when every observed cell has the same outcome.
- Distinct checkpoints, stages, panels and evaluation configurations are never pooled. Missing legacy identity is not treated as the current gate. Duplicate cells within the same identity keep their final recorded outcome and do not increase sample size.
- These descriptive panel averages do not establish the all-map victory gate. That gate separately requires every opponent–map pair and both seats to satisfy its standard and confirmation thresholds; averages across opponents or maps cannot substitute.
- Loss reductions describe fitting to self-play targets; benchmark outcomes measure playing performance. Neither a small smoke suite nor self-play win balance establishes strength against the wider field.
