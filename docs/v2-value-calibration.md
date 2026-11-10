# Value calibration on subsequent self-play games

Measured at `2026-11-10T03:34:53.413575+00:00` from **23 completed unique self-play games** identified when the input snapshot began at `2026-11-10T03:34:27.653939+00:00`.

Each game is scored using the exact checkpoint that generated it, before that game was available for training. These are future games relative to that checkpoint; they subsequently train newer checkpoints, so this is **not** a permanently held-out test of the final model. No benchmark data is read. Only retained searched positions are measured, with equal weight per game.

| Pre-game checkpoint | Completed / planned games | Positions | Brier | Value MSE | BCE (nats) | Mean value | Saturated |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8 / 8 | 4,096 | 0.2500 | 1.0000 | 0.6931 | 0.0000 | 0.00% |
| 1 | 8 / 8 | 3,492 | 0.2582 | 1.0326 | 0.7095 | -0.0318 | 0.00% |
| 2 | 7 / 8 | 3,584 | 0.2882 | 1.1527 | 0.7762 | 0.3702 | 0.00% |

Brier scores win probabilities against player-relative terminal outcomes; value MSE uses [-1,+1] and is exactly four times Brier. A neutral p=0.5 forecast has Brier 0.25, value MSE 1 and BCE ln(2)=0.6931 on decisive games. Saturation means |value|>0.95. Lower Brier/BCE is better. Reliability bins and plots are descriptive: positions from one game are correlated, and a few games cannot establish reliable generalization.

| Checkpoint | Mean predicted win probability | Mean terminal win label | Games against older checkpoints |
| ---: | ---: | ---: | ---: |
| 0 | 0.5000 | 0.4785 | 0 / 8 |
| 1 | 0.4841 | 0.7419 | 4 / 8 |
| 2 | 0.6851 | 0.4902 | 0 / 7 |

The latest measured checkpoint (2) has higher Brier error than the neutral baseline on its 7 completed future games. Its mean predicted win probability is 0.6851, versus a terminal win-label mean of 0.4902; 0.00% of predictions are saturated. This measures value forecasting, not playing strength. Self-play policies and the mix of historical opponents change between generations, so these scores do not isolate a causal effect of training. Historical-opponent games retain only current-policy positions, which can shift the label balance; equal-game weighting prevents longer retained trajectories from dominating.

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

Pending or invalid planned games: **1**. Excluded canonical artifacts: **0**; their reasons are retained in the JSON report. Attempt journals are never counted as additional games.

The JSON report records every game/NPZ/checkpoint SHA-256, generation identity, per-game scores and reliability bins. It rejects wrong terminal labels, changed inference sources, corrupt tensors, mismatched checkpoints and checkpoints from the same or later training generation.

- [Machine-readable measurements](../progress/campaign-v2/value-calibration.json)
- [Calibration plot](../progress/campaign-v2/value-calibration.png)

Regenerate without changing training state:

```bash
.venv/bin/python scripts/audit_value.py --run-dir runs/campaign/v2
```
