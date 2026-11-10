# Revised development v4 baseline: complete

Frozen warmstart cp2 `20d9e813349010630940e1cdfbaee5206490c7d323997688d26f47508bbeee7c`.

**83 wins / 173 losses / 0 draws: 32.421875% overall win rate.** All 256 scheduled games are valid, with zero errors. All eight maps, seeds 0/2026 and both seats are included.

[Audit](audit.json) verifies every terminal record, frozen panel/map/checkpoint identities and the exact schedule. This establishes the revised baseline, not a learned improvement over v3: the opponent panel changed.

| Opponent (prior Elo order) | Wins | Losses | Draws | Win rate |
|---|---:|---:|---:|---:|
| CS50Crab_rapierV1 | 26 | 6 | 0 | 81.2% |
| spsquared_TSPAARK_QUALS | 14 | 18 | 0 | 43.8% |
| HugoIngelsson_Xzibit | 17 | 15 | 0 | 53.1% |
| vibe26b_zdb | 4 | 28 | 0 | 12.5% |
| HugoIngelsson_YokoOno | 12 | 20 | 0 | 37.5% |
| abcdezyxwv_bestsofar | 2 | 30 | 0 | 6.2% |
| uravt_Version41 | 7 | 25 | 0 | 21.9% |
| IvanGeffner_basic51 | 1 | 31 | 0 | 3.1% |

The top-level README plots all opponent/map cells and the combined history. Historical v3 evidence remains unchanged. V8 starts from the same cp2 and uses fresh self-play, independent validation and fresh matched 256-game panels for each arm. No benchmark trajectories enter training.
