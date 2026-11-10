# Training progress

Generated from repository JSONL measurements at `2026-11-10T03:00:29+00:00`.

Run `python scripts/plot_progress.py` to regenerate these figures and this report.

## Latest training measurements

No training measurements have been recorded. Empty axes are intentional; no synthetic baseline is plotted.

![Training loss, entropy, data volume and throughput](training.png)

## Latest benchmark checkpoint

Iteration **0**, checkpoint `checkpoint-0000.pt [936b66b3/macro_v2/greedy/ef6f893e]`.

**24 completed games; 0 errors; 12/12 panel opponents represented.**

Equal-weight mean game score: **12.5%** across all frozen panel opponents. This is not an Elo estimate.

Paired-map bootstrap interval unavailable: 1 shared map–seed cells have both sides completed against every panel opponent; at least 4 are required. A small smoke suite is insufficient for a strength claim.

| Opponent | Completed | W–D–L | Mean score | Descriptive 95% Wilson interval | Errors | Sides A/B |
|---|---:|---:|---:|---:|---:|---:|
| `Krak9n_robot` | 2 | 2–0–0 | 100.0% | 34.2%–100.0% | 0 | 1/1 |
| `seru9_TOTAL_AGGRO` | 2 | 1–0–1 | 50.0% | 9.5%–90.5% | 0 | 1/1 |
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
