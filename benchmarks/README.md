# External benchmark dependencies

These are external bots from the battlecode26b-vibe field roster, not snapshots
of that project's own bot. See [the panel rationale](../docs/benchmarks.md) and
[the immutable manifest](../configs/benchmarks.json).

Download or verify all eight packages with:

```sh
python scripts/fetch_benchmarks.py
```

The downloaded `bots/` directory is ignored by Git. The manifest records exact
upstream commits and every file hash. `licenses/` preserves available upstream
license documents; missing license grants are not inferred. The files in
`evidence/` record the historical roster and ratings used for selection, not new
performance results. Benchmark sources and games are evaluation-only inputs.
