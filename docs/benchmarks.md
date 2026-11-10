# Frozen external benchmark opponents

The panel contains **eight external field bots** used by
[`terryvanbelle/battlecode26b-vibe` at `f4d6ec971b13a20f7f38fb56d249f62c0251ed05`](https://github.com/terryvanbelle/battlecode26b-vibe/tree/f4d6ec971b13a20f7f38fb56d249f62c0251ed05).
They are listed in that repository's `tools/galaxy26/field-teams.tsv`,
`research/field/FIELD.tsv`, and `research/field/candidates.txt`. **No version of
battlecode26b-vibe's own bot is in this panel.** Selected original roster rows
are retained under `benchmarks/evidence/`.

| External bot | Historical Elo | Approx. 95% interval | Recorded games | Selection rationale |
|---|---:|---:|---:|---|
| `Krak9n_robot` | 259.3 | ±33.7 | 1,218 | Weakest historical external-field endpoint. |
| `seru9_TOTAL_AGGRO` | 788.0 | ±18.8 | 1,983 | Lower-tier aggressive package. |
| `Camleaf_mk1` | 1006.2 | ±17.1 | 1,881 | Lower-middle field; bug-navigation implementation. |
| `CS50Crab_rapierV1` | 1398.7 | ±16.5 | 2,367 | Middle field; trap-and-combat family. |
| `spsquared_TSPAARK_QUALS` | 1584.6 | ±12.9 | 3,474 | Upper-middle qualifier strategy. |
| `HugoIngelsson_Xzibit` | 1695.8 | ±13.1 | 2,979 | Strong field; small-map economy provides a distinct challenge. |
| `abcdezyxwv_bestsofar` | 2010.4 | ±15.5 | 2,457 | Very strong grab/throw swarm, observed in historical replay analyses. |
| `IvanGeffner_basic51` | 2183.4 | ±24.5 | 2,319 | Strongest historical external-field endpoint. |

These span **1,924 historical Elo points** across eight separate authors. Elo and
intervals come from the source repository's batch Bradley–Terry fit; intervals
are 1.96 times its recorded standard errors. The fit mixes local score-share and
replica wins, accounts for map effects, and caps pair weights. It is a relative
historical scale, not official tournament Elo or a new measurement of this agent.
Recorded rows are in `benchmarks/evidence/historical_ratings.csv`.

## Acquisition and immutable identity

`configs/benchmarks.json` records each public upstream repository, full immutable
commit, package path, Java entry point, source-tree SHA-256, every downloaded file
hash, license location, and historical evidence. All eight selected packages
compile unchanged against Java 21 and the official engine 1.2.6. A limited static
scan found no network/process/filesystem/reflection/native-library API signatures;
this is a source check, not a claim of a security sandbox.

The original compiled/source downloads on the old VM are unavailable. We
reacquired the **same external bot names and source package paths** from their
public upstreams, and every pinned upstream commit predates the historical field
acquisition. However, the available historical repository does not record its
original VM artifact hashes, so exact byte-for-byte identity with that VM cannot
be proven. Historical ratings guide selection; new local results measure these
precisely pinned copies.

| Bot | Upstream | Pinned commit |
|---|---|---|
| `Krak9n_robot` | [Krak9n/battlecode-2026 ](https://github.com/Krak9n/battlecode-2026) | [`8a4af4fe0329`](https://github.com/Krak9n/battlecode-2026/tree/8a4af4fe0329bc7b6039b59104492eb65888c0d8/src/robot) |
| `seru9_TOTAL_AGGRO` | [seru9/BattleCode2026 ](https://github.com/seru9/BattleCode2026) | [`bf4823b6e049`](https://github.com/seru9/BattleCode2026/tree/bf4823b6e04912a3ceec38c56e1df1bb99d66f7f/src/TOTAL_AGGRO) |
| `Camleaf_mk1` | [Camleaf/battlecode26 ](https://github.com/Camleaf/battlecode26) | [`061afddd1d3d`](https://github.com/Camleaf/battlecode26/tree/061afddd1d3da0d4bd43c99af2e11cc15d67d3a2/src/mk1) |
| `CS50Crab_rapierV1` | [CS50Crab/The-Blades-Battlecode-2026 ](https://github.com/CS50Crab/The-Blades-Battlecode-2026) | [`f4b60966bb24`](https://github.com/CS50Crab/The-Blades-Battlecode-2026/tree/f4b60966bb2428e065e9d1781c9202b0a6da70e9/src/rapierV1) |
| `spsquared_TSPAARK_QUALS` | [spsquared/battlecode26 ](https://github.com/spsquared/battlecode26) | [`64c3257a929f`](https://github.com/spsquared/battlecode26/tree/64c3257a929f52c9824e60a330a7acf5a3670ef1/src/TSPAARK_QUALS) |
| `HugoIngelsson_Xzibit` | [HugoIngelsson/podemice-Battlecode-2026 ](https://github.com/HugoIngelsson/podemice-Battlecode-2026) | [`5414895a77f6`](https://github.com/HugoIngelsson/podemice-Battlecode-2026/tree/5414895a77f602d37034f06367223eb16a2e8ee6/Xzibit) |
| `abcdezyxwv_bestsofar` | [abcdezyxwv/Parmesan-bc2026 ](https://github.com/abcdezyxwv/Parmesan-bc2026) | [`0a0b266dd010`](https://github.com/abcdezyxwv/Parmesan-bc2026/tree/0a0b266dd010d31a66f2b4942853e4445dc36ace/src/bestsofar) |
| `IvanGeffner_basic51` | [IvanGeffner/BC26 ](https://github.com/IvanGeffner/BC26) | [`70b3f40cebef`](https://github.com/IvanGeffner/BC26/tree/70b3f40cebefcca9dc858771715d78088e7ae520/basic51) |

Run `python scripts/fetch_benchmarks.py` after cloning this project. It downloads
immutable GitHub archives, copies only manifest-listed files after SHA-256
verification, and verifies existing downloads without network access. A hash
mismatch fails loudly. Third-party sources stay under gitignored
`benchmarks/bots/`; they are not republished as this project's code. Available
upstream license documents are retained in `benchmarks/licenses/`, and originals
are kept beside local downloaded sources. No license was found for the selected
Krak9n, HugoIngelsson, or IvanGeffner upstreams; no license grant is inferred.

## Evaluation protocol

1. Run `python scripts/fetch_benchmarks.py` to fetch or verify the pinned local
   dependencies, then compile each unchanged package against the official Battlecode 2026 Java
   engine **1.2.6**. The entry point is `<package>.RobotPlayer.run(RobotController)`.
   Keep package names unchanged: they are part of source identity, and renaming
   a bot can alter identity-sensitive behavior.
2. Run every opponent from **both A and B** on every selected map and seed. Use
   deterministic candidate inference (no training exploration) and a fixed
   search budget, recorded with checkpoint hash and engine hash.
3. A smoke checkpoint uses `tiny`, seed `0`: **16 games**, two per opponent. This
   proves integration and offers only a noisy initial learning signal.
4. The standard suite uses `tiny`, `DefaultSmall`, `cheesefarm`, and
   `wallsofparadis`, seeds `0` and `2026`: **128 games**, 16 per opponent. These
   cover close combat, ordinary terrain, economy, and obstructed navigation.
   Never describe a smaller completed subset as the entire suite.
5. Store per-game opponent/source hash, map/hash, side, seed, checkpoint/hash,
   engine/hash, compute mode, winner, end reason, rounds, final points, duration,
   and failure status. Report wins/draws/losses and completion rate per opponent,
   with a macro average so a frequently played weak bot cannot dominate.
6. Use paired map–seed cells (both sides together) for uncertainty and checkpoint
   comparisons. `scripts/plot_progress.py` implements a paired-cell bootstrap for
   the panel mean when at least four shared map–seed cells have both sides against
   all eight opponents. Per-opponent Wilson intervals are labelled descriptive,
   not evidence of significance, because sides/maps are correlated. Do not treat
   robot decisions, self-play positions, repeated
   deterministic cells, or both seats as independent samples. A paired bootstrap
   interval needs enough map–seed cells; the smoke suite cannot provide one.
7. Benchmark games and benchmark code are **evaluation only**, never replay
   training targets or imitation examples. If training shares evaluation maps,
   label results in-distribution and use additional unseen maps for final claims.
   Do not optimize a claimed final model repeatedly on a held-out test suite.
8. Run candidate and benchmarks with identical engine limits. Local bytecode
   relaxation can change behavior (especially bots with bytecode-aware branches),
   so historical ratings are only selection evidence. Preserve a separately
   labelled official-limit evaluation mode for later distillation work.

Record crashes, timeouts and illegal-action failures explicitly. Do not silently
filter them out of graphs or present them as completed ordinary losses. Show
completed-game win rate alongside its denominator and the separate failure
count. An actual engine-rule forfeit is a result, labelled as a forfeit.

No engine-run strength results are claimed by this document. Live measurements
belong to the experiment's result files and regenerated progress graphs.
