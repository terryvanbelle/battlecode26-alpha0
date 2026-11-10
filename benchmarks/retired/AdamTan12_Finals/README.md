# Retired: AdamTan12_Finals

Replaced by **`thewaxmango_collegejunior`** in **`field-twelve-v2`** under the
user's instruction to replace frequent crashers. Other eleven entries unchanged.

**Confirmed defect:** uncaught `GameActionException: Target location is not on
the map` in `ThrowOutsideCombat.throwAtWall`. The source senses the adjacent
square at [line 58](https://github.com/AdamTan12/BattleCode2026/blob/91fdddbb1c015a2e8cd7ebcb8eebc8319b7ce06d/src/Finals/Actions/ThrowOutsideCombat.java#L58)
before checking bounds at line 60. Independent engine reproductions on **tiny,
seed 0 / round 54** and **seed 1 / round 40** confirmed the bot fault after
harness validation. Legal disintegration and ordinary game endings are excluded.

The evaluation ledger had four failures in eight checkpoint/seat cells on tiny,
seed 0; these share a map/seed and are not eight independent map–seed samples:

| Checkpoint | Side A | Side B |
|---:|---|---|
| 0 | Completed | Completed |
| 1 | Error, round 54 | Completed |
| 2 | Error, round 29 | Error, round 21 |
| 3 | Completed | Error, round 21 |

Replacement historical Elo is **1767.0**, versus Finals' **1779.7**. The new
independent author is pinned at `f8f7864b5f12ac6c303339e89e06c4aa49102f8b` and passed
Java21/engine1.2.6 compilation and clean-download hash checks. These checks do
not guarantee future games will be crash-free.

Archive contents: `manifest-entry.json` (retired commit/file hashes),
`historical-rating.json`, `evaluation-summary.jsonl` (compact ledger and original
error hashes), `reproduction-evidence.json` (checkpoint hashes and short stack
traces), and upstream license. Full retired game rows are in
[`progress/retired/AdamTan12_Finals.jsonl`](../../../progress/retired/AdamTan12_Finals.jsonl).
No retired bot source is included here.

Reproduce with the saved checkpoint0001 and already compiled retired dependency:

```sh
PYTHONPATH=. .venv/bin/python benchmarks/retired/AdamTan12_Finals/reproduce.py
```

The script writes diagnostic evidence under `/tmp`, not experiment logs. Play
the replacement against every comparable checkpoint; exclude **all** Finals
rows from v2 aggregates, including its successful games.
