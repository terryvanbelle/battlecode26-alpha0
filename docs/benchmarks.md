# Frozen benchmark panel

**`field-twelve-v2` contains 12 opponents:** ten independent external field bots
used by `battlecode26b-vibe`, official `examplefuncsplayer`, and the project's
current accepted incumbent `zdb`. The original eight external IDs/hashes remain
unchanged. V2 replaces the confirmed crashing `AdamTan12_Finals` with
`thewaxmango_collegejunior`; the other eleven entries are unchanged.

| Opponent ID | Role | Historical Elo ± approximate 95% interval | Recorded games | Selection rationale |
|---|---|---:|---:|---|
| `Krak9n_robot` | External | 259.3 ± 33.7 | 1,218 | weakest external field endpoint. |
| `seru9_TOTAL_AGGRO` | External | 788.0 ± 18.8 | 1,983 | lower-tier aggressive package. |
| `Camleaf_mk1` | External | 1006.2 ± 17.1 | 1,881 | lower-middle field; bug-navigation implementation. |
| `CS50Crab_rapierV1` | External | 1398.7 ± 16.5 | 2,367 | middle field; trap-and-combat family. |
| `spsquared_TSPAARK_QUALS` | External | 1584.6 ± 12.9 | 3,474 | upper-middle qualifier strategy. |
| `HugoIngelsson_Xzibit` | External | 1695.8 ± 13.1 | 2,979 | strong field; lower-spawn small-map economy. |
| `abcdezyxwv_bestsofar` | External | 2010.4 ± 15.5 | 2,457 | very strong field; grab/throw swarm. |
| `IvanGeffner_basic51` | External | 2183.4 ± 24.5 | 2,319 | top historical external field endpoint. |
| `bc26-example` | Official starter | 1038.6 ± 16.1 | 2,218 | official starter reference. |
| `vibe26b_zdb` | Accepted incumbent | 1695.9 ± 52.9 | 213 | current accepted incumbent from battlecode26b-vibe. |
| `thewaxmango_collegejunior` | External | 1767.0 ± 12.7 | 3,084 | strong external collegejunior strategy; near-strength replacement for confirmed crashing Finals. |
| `uravt_Version41` | External | 2125.3 ± 21.8 | 2,163 | elite external field; second independent high-strength endpoint. |

Historical ratings are selection evidence, **not new results for this agent**.
They come from the source project's map-adjusted Bradley–Terry fit, combining
local score shares and replica wins; intervals are 1.96 times its recorded
standard errors. The ten external authors span 1,924 historical Elo points.
Exact rows and original roster excerpts are in `benchmarks/evidence/`, pinned to
[`f4d6ec971b13a20f7f38fb56d249f62c0251ed05`](https://github.com/terryvanbelle/battlecode26b-vibe/tree/f4d6ec971b13a20f7f38fb56d249f62c0251ed05).

## Current accepted bot and official starter

The latest source HEAD checked was
[`fc044c24c37d89d2581562e23439a2571bb2b300`](https://github.com/terryvanbelle/battlecode26b-vibe/tree/fc044c24c37d89d2581562e23439a2571bb2b300).
Its November 9, 2026, 18:02 PST handoff states “Incumbent still **zdb**
(resubmitted #332)”; `zgj` was rejected and newer experiments were pending.
`zdb` was accepted November 8 over `z3x`: 96 paired cells, +4/−0, t = 2.00.
We select frozen **`bots/zdb`**, not the differing working `bots/ratbot` or an
unaccepted experiment with a high fitted rating. The checked Java bytes also
match the older reference. See [incumbent evidence](../benchmarks/evidence/incumbent.md).

`bc26-example` is actual `examplefuncsplayer` from the official
[`battlecode/battlecode26-scaffold`](https://github.com/battlecode/battlecode26-scaffold/tree/f69e2ab872a0061c9d4a684aa1dd798a0829da85/src/examplefuncsplayer)
at `f69e2ab872a0061c9d4a684aa1dd798a0829da85`.

## Immutable downloads

Run `python scripts/fetch_benchmarks.py`. The [manifest](../configs/benchmarks.json)
records full upstream commits, package paths, entry points, every file's SHA-256,
Java-tree hashes, and license locations. Downloads copy only listed files after
hash verification; existing verified files need no network. All 12 packages
compile unchanged against Java 21/official engine 1.2.6.

Sources remain in gitignored `benchmarks/bots/`. Available upstream licenses are
retained in `benchmarks/licenses/`; no grant is inferred where none was found.
The original VM downloads have no available historical hashes. External names
and package paths match its roster, and pinned upstream commits predate its
acquisition, but exact identity with the old VM artifacts cannot be proven.
The own-project incumbent is directly pinned to its source commit.

## Evaluation protocol

- Keep package names unchanged. Benchmark source and games are **evaluation only**,
  never imitation examples or replay training targets.
- Use both sides at each map/seed, deterministic candidate inference, and a fixed
  recorded search budget. Candidate and benchmark use the same documented engine
  settings; relaxed limits can change historical benchmark behavior.
- **Smoke:** `tiny`, seed `0`, both sides = **24 games/checkpoint**. This is an
  integration and early-learning signal, insufficient for strength claims.
- **Standard:** `tiny`, `DefaultSmall`, `cheesefarm`, `wallsofparadis`, seeds `0`
  and `2026`, both sides = **192 games/checkpoint**. Keep schedules fixed. Label
  shared training maps as in-distribution and reserve unseen maps for final claims.
- Retain checkpoint/source/engine hashes, map/seed/side, outcome, end reason,
  rounds, points, duration and failure status. Report per-opponent W/D/L,
  completed counts and errors, plus an equal-weight opponent mean with coverage.
- The plotter bootstraps shared map–seed cells, keeping both sides and all panel
  opponents together; it needs at least four complete shared cells. Per-opponent
  Wilson intervals are descriptive because sides/maps are correlated. Repeated
  deterministic cells and robot decisions are not independent evidence.
- Errors are reported separately, never silently dropped or scored as ordinary
  losses. Legal deaths, disintegration, resignation and engine-rule forfeits are
  distinguished from runtime faults. A forfeit remains a labelled game result.

## Runtime failures and replacement

Confirm reproducible failures across independent cells after verifying the
harness and excluding legal game events. Then choose a different external author
at similar historical skill, freeze a new suite ID, retain failure/provenance
evidence, and complete that panel for every checkpoint being compared. Unchanged
cells may be reused only when their checkpoint, source, engine and schedule match;
replacement games must be played for every comparable checkpoint. Never combine
different panel aggregates into an unqualified progress trend.

**V2 replacement:** Finals produced four failures in eight checkpoint/seat cells.
Independent reproductions on tiny seeds 0 and 1 failed at rounds 54 and 40:
`ThrowOutsideCombat.throwAtWall:58` senses an off-map square before its bounds
check, causing an uncaught `GameActionException`. This was a genuine bot fault,
separate from the corrected harness detector issue. The replacement
`thewaxmango_collegejunior` has historical Elo 1767.0 versus Finals' 1779.7.
[Retired provenance and concise failure evidence](../benchmarks/retired/AdamTan12_Finals/README.md)
are retained; all Finals results, including completed games, are excluded from v2
aggregates. No retired bot source is republished.
