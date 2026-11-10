# Warmstart iteration 1: prior held, no smoke gain

Checkpoint 1 retained **20/24 wins**, equal to checkpoint 0, after one genuine MCTS training iteration. Both panels completed all twelve frozen opponents in both seats with zero errors, using `tiny`, seed 0 and identical evaluation settings. 15 of the 24 paired terminal world states are exactly identical; this does not prove identical action histories. The four strongest-opponent losses show no terminal progress.

This is an explicit own-teacher policy prior followed by search-guided reinforcement learning. It is not a tabula-rasa AlphaZero result or a victory-gate pass. Checkpoint 0 and the original teacher-prior artifact remain preserved; no policy was promoted and no holdout was consumed.

The cycle took 14.8 minutes including both evaluations. Eight completed self-play games contributed 17,531 real decisions and 1,948 searched positions; only those genuine visit distributions and terminal outcomes entered the fresh, game-balanced replay. There were 48 optimizer updates. Collection/training took 9.4 minutes. No historical opponent was selected in this first iteration.

Mean training policy CE was 1.0551; terminal win-logit BCE 0.6891; value MSE 0.9919; saturated-value fraction 0.0%. These are in-sample optimization measurements, not value calibration or playing-strength estimates.

## Every smoke cell

| Opponent | CP0 A/B | CP1 A/B | Terminal states identical A/B |
| --- | --- | --- | --- |
| `CS50Crab_rapierV1` | W/W | W/W | no/no |
| `Camleaf_mk1` | W/W | W/W | yes/no |
| `HugoIngelsson_Xzibit` | W/W | W/W | yes/no |
| `IvanGeffner_basic51` | L/L | L/L | yes/yes |
| `Krak9n_robot` | W/W | W/W | yes/yes |
| `abcdezyxwv_bestsofar` | L/L | L/L | yes/yes |
| `bc26-example` | W/W | W/W | no/no |
| `seru9_TOTAL_AGGRO` | W/W | W/W | yes/yes |
| `spsquared_TSPAARK_QUALS` | W/W | W/W | yes/no |
| `thewaxmango_collegejunior` | W/W | W/W | yes/yes |
| `uravt_Version41` | W/W | W/W | no/no |
| `vibe26b_zdb` | W/W | W/W | yes/yes |

## Strong-opponent losses

Each quantity below is CP0 → CP1. Baby counts include all surviving friendly/enemy babies in the terminal engine state. All four defeats ended by losing every allied king.

| Opponent / seat | Round | Enemy king HP | Own / enemy babies | Own / enemy bank |
| --- | --- | --- | --- | --- |
| `IvanGeffner_basic51` A | 94 → 94 | 600 → 600 | 1/17 → 1/17 | 1/630 → 1/630 |
| `IvanGeffner_basic51` B | 60 → 60 | 600 → 600 | 8/16 → 8/16 | 845/1336 → 845/1336 |
| `abcdezyxwv_bestsofar` A | 42 → 42 | 470 → 470 | 11/19 → 11/19 | 986/1621 → 986/1621 |
| `abcdezyxwv_bestsofar` B | 54 → 54 | 590 → 590 | 11/16 → 11/16 | 697/1859 → 697/1859 |

## Evidence and continuation

- [Training and evaluation graphs](README.md), [paired JSON](iteration-001-comparison.json), [all 24 paired cells as CSV](iteration-001-cells.csv).
- [Protocol, prior lineage and restart command](../../docs/warmstart-v3.md).
- Both checkpoints share evaluation identity `ef6f893eb9563b88a33b6288c95b2fdeecd537db9eb16ed28040d52b0f97235c`.
- Checkpoint 0: `250c37e9d47ee647933d64c3988398042bb71f626754423900428cf878fe2f6a`.
- Checkpoint 1: `121acf1bd12d6f44c4909766167011664106605f1325371a585ed10a118d1360`.

A second iteration was authorized after this score held and the iteration-1 recovery archive was created, with launch conditional on archive verification. It tests learning with the now-trained critic while keeping this configuration fixed and preserving the prior champion. A future separate pure-current-self-play control would help isolate league target-distribution shifts: current collection retains both players in same-policy games but only current-learner positions against historical opponents, and the network has no explicit opponent identity. This observation does not alter the active immutable configuration.
