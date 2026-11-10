# Benchmark dependencies

The frozen panel contains ten external field bots from battlecode26b-vibe's
roster, the official examplefuncsplayer, and its current accepted incumbent zdb.
See [the panel rationale](../docs/benchmarks.md) and
[the immutable manifest](../configs/benchmarks.json).

Download or verify all 12 packages with:

```sh
python scripts/fetch_benchmarks.py
```

The downloaded `bots/` directory is ignored by Git. The manifest records exact
upstream commits and every file hash. `licenses/` preserves available upstream
license documents; missing license grants are not inferred. `evidence/` records
the historical roster, ratings and incumbent selection, not new performance
results. Benchmark sources and games are evaluation-only inputs.

Current suite: `field-twelve-v2`. Confirmed runtime-based replacements and their
frozen provenance are recorded under [`retired/`](retired/AdamTan12_Finals/README.md).
