# Value calibration on subsequent self-play games

Measured at `2026-11-10T03:45:37.555018+00:00` from **29 completed unique self-play games** identified when the input snapshot began at `2026-11-10T03:45:13.404498+00:00`.

Each game is scored using the exact checkpoint that generated it, before that game was available for training. These are future games relative to that checkpoint; they subsequently train newer checkpoints, so this is **not** a permanently held-out test of the final model. No benchmark data is read. Only retained searched positions are measured, with equal weight per game.

| Pre-game checkpoint | Completed / planned games | Positions | Brier | Value MSE | BCE (nats) | Mean value | Saturated |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8 / 8 | 4,096 | 0.2500 | 1.0000 | 0.6931 | 0.0000 | 0.00% |
| 1 | 8 / 8 | 3,492 | 0.2582 | 1.0326 | 0.7095 | -0.0318 | 0.00% |
| 2 | 8 / 8 | 4,096 | 0.2874 | 1.1497 | 0.7745 | 0.3646 | 0.00% |
| 3 | 5 / 8 | 2,259 | 0.2824 | 1.1297 | 0.7603 | 0.2259 | 0.00% |

Brier scores win probabilities against player-relative terminal outcomes; value MSE uses [-1,+1] and is exactly four times Brier. A neutral p=0.5 forecast has Brier 0.25, value MSE 1 and BCE ln(2)=0.6931 on decisive games. Saturation means |value|>0.95. Lower Brier/BCE is better. Reliability bins and plots are descriptive: positions from one game are correlated, and a few games cannot establish reliable generalization.

| Checkpoint | Mean predicted win probability | Mean terminal win label | Games against older checkpoints |
| ---: | ---: | ---: | ---: |
| 0 | 0.5000 | 0.4785 | 0 / 8 |
| 1 | 0.4841 | 0.7419 | 4 / 8 |
| 2 | 0.6823 | 0.4927 | 0 / 8 |
| 3 | 0.6130 | 0.3932 | 1 / 5 |

The latest measured checkpoint (3) has higher Brier error than the neutral baseline on its 5 completed future games. Its mean predicted win probability is 0.6130, versus a terminal win-label mean of 0.3932; 0.00% of predictions are saturated. This measures value forecasting, not playing strength. Self-play policies and the mix of historical opponents change between generations, so these scores do not isolate a causal effect of training. Historical-opponent games retain only current-policy positions, which can shift the label balance; equal-game weighting prevents longer retained trajectories from dominating.

## Same-policy and historical-opponent games

The same-policy group uses the generation checkpoint for both teams and retains searched positions from both. The historical-opponent (league) group searches and retains only the current checkpoint's team; the older opponent samples its policy without search. Consequently a decisive league game has either all positive or all negative retained labels, while same-policy games usually contribute labels from both teams. Each row below weights its games equally; a group's weight in the aggregate is its fraction of completed games.

| Checkpoint | Group | Games | Positions | Predicted win | Terminal win label | Brier | BCE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Same policy, both teams | 8 | 4,096 | 0.5000 | 0.4785 | 0.2500 | 0.6931 |
| 1 | League, current-only | 4 | 1,444 | 0.4838 | 1.0000 | 0.2664 | 0.7260 |
| 1 | Same policy, both teams | 4 | 2,048 | 0.4843 | 0.4839 | 0.2499 | 0.6929 |
| 2 | Same policy, both teams | 8 | 4,096 | 0.6823 | 0.4927 | 0.2874 | 0.7745 |
| 3 | League, current-only | 1 | 512 | 0.5815 | 0.0000 | 0.3388 | 0.8731 |
| 3 | Same policy, both teams | 4 | 1,747 | 0.6208 | 0.4915 | 0.2683 | 0.7321 |

The observation encoder supplies board state and side-relative game statistics, but no checkpoint identity, league flag, opponent search budget or opponent temperature. A board position's eventual outcome can depend on those omitted policy conditions, so a single value head learns a changing mixture of continuation regimes. This is a plausible source of forecast drift, together with small samples, changing visited states and fitting error; the observed changes alone do **not** establish overfitting or prove opponent aliasing caused the errors. See the unchanged [feature encoder](../alpha0/features_v2.py), [rollout selection](../alpha0/learn_v2.py) and [game-balanced replay](../alpha0/replay_v2.py).

A future controlled test should start two isolated runs from the same immutable checkpoint, keep optimizer/search budgets and paired map/seed schedules fixed, and compare pure current-policy self-play with a fixed 25% historical-opponent mixture over two generations of 16 games. Predeclare the league assignments rather than letting an eight-game binomial draw change the mixture sharply. Score each resulting checkpoint on fresh, completed future games, reporting same-policy and fixed historical-opponent strata separately with equal-game weighting. Include a matched-retention control that scores only the designated current team in both groups, to separate changes in opponents from changes in which labels are retained. Keep benchmark trajectories excluded. An alternative follow-up is a versioned value model conditioned on the simulator-known opponent policy/search regime, or separate value heads; that needs a documented default for unknown opponents at deployment. These are proposed tests, not changes to the active campaign, and success requires better future-game Brier/BCE rather than lower replay loss.

## Per-game measurements

| Game | Map | Checkpoint | Positions | Brier | Value MSE | BCE | Mean value | Saturated |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `iter-0001-game-000` | `tiny` | 0 | 512 | 0.2500 | 1.0000 | 0.6931 | 0.0000 | 0.00% |
| `iter-0001-game-001` | `DefaultSmall` | 0 | 512 | 0.2500 | 1.0000 | 0.6931 | 0.0000 | 0.00% |
| `iter-0001-game-002` | `closeup` | 0 | 512 | 0.2500 | 1.0000 | 0.6931 | 0.0000 | 0.00% |
| `iter-0001-game-003` | `knifefight` | 0 | 512 | 0.2500 | 1.0000 | 0.6931 | 0.0000 | 0.00% |
| `iter-0001-game-004` | `tiny` | 0 | 512 | 0.2500 | 1.0000 | 0.6931 | 0.0000 | 0.00% |
| `iter-0001-game-005` | `DefaultSmall` | 0 | 512 | 0.2500 | 1.0000 | 0.6931 | 0.0000 | 0.00% |
| `iter-0001-game-006` | `closeup` | 0 | 512 | 0.2500 | 1.0000 | 0.6931 | 0.0000 | 0.00% |
| `iter-0001-game-007` | `knifefight` | 0 | 512 | 0.2500 | 1.0000 | 0.6931 | 0.0000 | 0.00% |
| `iter-0002-game-000` | `tiny` | 1 | 239 | 0.2673 | 1.0693 | 0.7278 | -0.0340 | 0.00% |
| `iter-0002-game-001` | `DefaultSmall` | 1 | 512 | 0.2502 | 1.0009 | 0.6936 | -0.0310 | 0.00% |
| `iter-0002-game-002` | `closeup` | 1 | 512 | 0.2492 | 0.9966 | 0.6915 | -0.0323 | 0.00% |
| `iter-0002-game-003` | `knifefight` | 1 | 512 | 0.2504 | 1.0017 | 0.6940 | -0.0294 | 0.00% |
| `iter-0002-game-004` | `tiny` | 1 | 388 | 0.2666 | 1.0666 | 0.7265 | -0.0327 | 0.00% |
| `iter-0002-game-005` | `DefaultSmall` | 1 | 512 | 0.2666 | 1.0665 | 0.7264 | -0.0327 | 0.00% |
| `iter-0002-game-006` | `closeup` | 1 | 512 | 0.2497 | 0.9989 | 0.6926 | -0.0326 | 0.00% |
| `iter-0002-game-007` | `knifefight` | 1 | 305 | 0.2651 | 1.0606 | 0.7234 | -0.0298 | 0.00% |
| `iter-0003-game-000` | `tiny` | 2 | 512 | 0.3107 | 1.2426 | 0.8280 | 0.4558 | 0.00% |
| `iter-0003-game-001` | `DefaultSmall` | 2 | 512 | 0.2470 | 0.9880 | 0.6845 | 0.3531 | 0.00% |
| `iter-0003-game-002` | `closeup` | 2 | 512 | 0.3038 | 1.2154 | 0.8091 | 0.3343 | 0.00% |
| `iter-0003-game-003` | `knifefight` | 2 | 512 | 0.2948 | 1.1791 | 0.7868 | 0.3137 | 0.00% |
| `iter-0003-game-004` | `tiny` | 2 | 512 | 0.2707 | 1.0830 | 0.7433 | 0.4602 | 0.00% |
| `iter-0003-game-005` | `DefaultSmall` | 2 | 512 | 0.2934 | 1.1737 | 0.7876 | 0.3484 | 0.00% |
| `iter-0003-game-006` | `closeup` | 2 | 512 | 0.2968 | 1.1873 | 0.7940 | 0.3262 | 0.00% |
| `iter-0003-game-007` | `knifefight` | 2 | 512 | 0.2821 | 1.1282 | 0.7626 | 0.3251 | 0.00% |
| `iter-0004-game-000` | `tiny` | 3 | 512 | 0.3399 | 1.3595 | 0.8830 | 0.3090 | 0.00% |
| `iter-0004-game-001` | `DefaultSmall` | 3 | 512 | 0.3388 | 1.3553 | 0.8731 | 0.1630 | 0.00% |
| `iter-0004-game-002` | `closeup` | 3 | 512 | 0.2474 | 0.9894 | 0.6873 | 0.1609 | 0.00% |
| `iter-0004-game-003` | `knifefight` | 3 | 211 | 0.2616 | 1.0465 | 0.7176 | 0.1896 | 0.00% |
| `iter-0004-game-004` | `tiny` | 3 | 512 | 0.2244 | 0.8978 | 0.6404 | 0.3071 | 0.00% |

Pending or invalid planned games: **3**. Excluded canonical artifacts: **0**; their reasons are retained in the JSON report. Attempt journals are never counted as additional games.

The JSON report records every game/NPZ/checkpoint SHA-256, generation identity, per-game scores and reliability bins. It rejects wrong terminal labels, changed inference sources, corrupt tensors, mismatched checkpoints and checkpoints from the same or later training generation.

- [Machine-readable measurements](../progress/campaign-v2/value-calibration.json)
- [Calibration plot](../progress/campaign-v2/value-calibration.png)

Regenerate without changing training state:

```bash
.venv/bin/python scripts/audit_value.py --run-dir runs/campaign/v2
```
