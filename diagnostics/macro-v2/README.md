# Scripted macro-v2 capability diagnostic

These are **fixed rules-only controllers, not learned models**. Their games are **not training data** and do not establish a learned agent victory over the benchmark suite. The controller does not inspect opponent identity or copy opponent code.

The observed smoke experiment uses `tiny`, seed `0`, both seats against the 12 frozen opponents: 48 completed games across two controllers, with zero errors or timeouts. Each game ran in an isolated process with a 180-second game watchdog; at most two games ran concurrently.

| Scripted controller | Wins / completed | Opponents won against on both seats |
|---|---:|---:|
| assault | 20 / 24 | 10 / 12 |
| balanced | 10 / 24 | 5 / 12 |

Assault lost both seats against `abcdezyxwv_bestsofar` and `IvanGeffner_basic51`. The balanced script performed worse despite using more carrying and powered bites. This is evidence about the available macro actions and fixed priority choices on one small map, not general playing strength, statistical significance, or progress of a trained network.

- [Complete measured JSONL](results.jsonl) includes every cell, status, score, rounds, final resource/king summaries and action transcript.
- [Per-controller/opponent summary](summary.json) reports completed games and unscored failures separately.
- [Runtime and script identity](identity.json) pins the script, compiled bridge, engine JAR, map and suite. Each result also pins its opponent source and Java hash.

Reproduce this diagnostic from the repository root (after engine and benchmark setup):

```sh
.venv/bin/python scripts/probe_macros.py --controllers assault balanced \
  --map tiny --seed 0 --jobs 2 --timeout 180 --output runs/macro-probe
```

## Per-opponent observations

| Controller |Opponent | Wins / completed | Errors | Timeouts |
|---|---|---:|---:|---:|
| assault | CS50Crab_rapierV1 | 2 / 2 | 0 | 0 |
| assault | Camleaf_mk1 | 2 / 2 | 0 | 0 |
| assault | HugoIngelsson_Xzibit | 2 / 2 | 0 | 0 |
| assault | IvanGeffner_basic51 | 0 / 2 | 0 | 0 |
| assault | Krak9n_robot | 2 / 2 | 0 | 0 |
| assault | abcdezyxwv_bestsofar | 0 / 2 | 0 | 0 |
| assault | bc26-example | 2 / 2 | 0 | 0 |
| assault | seru9_TOTAL_AGGRO | 2 / 2 | 0 | 0 |
| assault | spsquared_TSPAARK_QUALS | 2 / 2 | 0 | 0 |
| assault | thewaxmango_collegejunior | 2 / 2 | 0 | 0 |
| assault | uravt_Version41 | 2 / 2 | 0 | 0 |
| assault | vibe26b_zdb | 2 / 2 | 0 | 0 |
| balanced | CS50Crab_rapierV1 | 2 / 2 | 0 | 0 |
| balanced | Camleaf_mk1 | 2 / 2 | 0 | 0 |
| balanced | HugoIngelsson_Xzibit | 0 / 2 | 0 | 0 |
| balanced | IvanGeffner_basic51 | 0 / 2 | 0 | 0 |
| balanced | Krak9n_robot | 2 / 2 | 0 | 0 |
| balanced | abcdezyxwv_bestsofar | 0 / 2 | 0 | 0 |
| balanced | bc26-example | 2 / 2 | 0 | 0 |
| balanced | seru9_TOTAL_AGGRO | 2 / 2 | 0 | 0 |
| balanced | spsquared_TSPAARK_QUALS | 0 / 2 | 0 | 0 |
| balanced | thewaxmango_collegejunior | 0 / 2 | 0 | 0 |
| balanced | uravt_Version41 | 0 / 2 | 0 | 0 |
| balanced | vibe26b_zdb | 0 / 2 | 0 | 0 |
