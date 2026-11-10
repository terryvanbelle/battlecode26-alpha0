# Initial experiment: measured results

The complete pipeline is working, but this small run **does not demonstrate an overall strength improvement**. The trained policy finished at 5 wins in 24 games, the same aggregate score as the untrained network. Individual matchups changed.

## Training performed

- Six complete official-engine games: three on `tiny`, three on `DefaultSmall`, with recorded seeds and actions.
- 46,538 actual search decisions; 43,779 retained positions after the per-game reservoir cap.
- Three iterations, 32 gradient updates each; 49,242-parameter model; four PUCT simulations per decision.
- About 18 minutes of training-loop wall time on the local CPU executor, with evaluation work sometimes sharing the host. `battlecode-dev` was not used.
- Final replay contains 10,000 positions. The small buffer can be dominated by a single long game; the larger configuration caps each game's contribution to preserve more game diversity.

## Frozen panel evaluation

Suite **`field-twelve-v2`**: 12 opponents × both seats on `tiny`, map seed 0. Each checkpoint uses greedy network decisions without MCTS. The panel, maps and settings are identical across these comparisons.

| Training iteration | Wins | Completed games | Mean game score | Unresolved failures |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 5 | 24 | 20.8% | 0 |
| 1 | 3 | 24 | 12.5% | 0 |
| 2 | 5 | 24 | 20.8% | 0 |
| 3 | 5 | 24 | 20.8% | 0 |

There were no draws. All 96 current-panel cells completed successfully. A superseded harness error remains in the raw ledger for auditability; the plotter uses its corrected rerun once, not as extra evidence.

The final checkpoint gained both seats against `CS50Crab_rapierV1`, but lost wins elsewhere:

| Opponent with changed results | Baseline wins | Final wins |
| --- | ---: | ---: |
| `seru9_TOTAL_AGGRO` | 2/2 | 1/2 |
| `Camleaf_mk1` | 1/2 | 0/2 |
| `CS50Crab_rapierV1` | 0/2 | 2/2 |

Both the official example player and accepted `vibe26b_zdb` beat the learned policy in both seats at the final checkpoint. With only one shared map–seed pair and two games per opponent, these observations are insufficient for a statistically meaningful strength claim or an Elo estimate. The final panel also overlaps smoke training maps.

## Failures and replacement

Early setup detected and corrected missing Java module access, voluntary disintegration incorrectly classified as a crash, and possible bot stdout interference with the JSON protocol. Invalid setup measurements are excluded from current results.

`AdamTan12_Finals` then exhibited a genuine uncaught off-map exception, reproduced on two seeds and observed in four of eight checkpoint/seat cells. Per the owner's instruction, it was replaced by similarly rated `thewaxmango_collegejunior`. **All** retired-opponent rows, including successful games, are excluded from the current panel. Provenance and diagnostic evidence are retained under `benchmarks/retired/AdamTan12_Finals/` and `progress/retired/`.

## Artifacts and validation

- `progress/training.jsonl` and `progress/evaluations.jsonl`: measurements underlying the graphs.
- `artifacts/initial/`: all four checkpoints, exact configuration, six game transcripts and a provenance/hash manifest.
- 30 tests passed, including native-scheduler equivalence through complete games, snapshot/replay branching, player-perspective backup, optimizer/RNG checkpoint recovery, benchmark fetching, reporting, and crash-versus-legal-termination detection.
- The exported final self-play game was replayed for 9,177 decisions and matched its recorded terminal engine state exactly.

The initial run is complete; it is not an always-running service. The local `runs/initial` directory retains replay/optimizer state for the README's resume command. Published checkpoints support evaluation, but resuming training also requires the matching local replay buffer. A sensible next experiment increases game diversity and data volume, then expands the action abstraction; downloading more compute alone does not remove those modeling limitations.
