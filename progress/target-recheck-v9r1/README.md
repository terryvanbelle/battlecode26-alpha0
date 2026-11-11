# V9r1 target reliability recheck

Running a bounded no-SGD mechanism test: all32 frozen V8 training roots, selected action versus cp2 greedy, four fresh continuation seeds and184 physically shared cells. No benchmark games or model updates. [Frozen protocol](../../docs/target-recheck-v9r1.md).

[Measured summary](summary.json) · [Root advantage and coverage graph](target-reliability.png). Missing roots do not contribute a fabricated score; no expansion decision until184/184 valid continuations. All32 roots remain in the completed denominator.

Original V9 stopped before a continuation because its wrapper omitted a required replay-task journal path. Its [failure record](../target-recheck-v9/launch-failure.json) and logs are retained. V9r1 isolates the infrastructure repair and keeps scientific settings/seeds unchanged.

Status: `runs/target-recheck-v9r1/status.json`; log `/tmp/target-recheck-v9r1.log`. Own plus seven shared locks/STOPs protect the run. Do not edit pinned collector/protocol/config while active. After quiescence use `scripts/export_target_recheck_v9r1.py`; it verifies the complete V8 recovery dependency and separately exports/isolated-restores V9r1 own state. To recover, first restore the referenced V8 archive without overwriting newer work, then verify the V9r1 archive against its per-file manifest and restore into an isolated directory before merging missing files. Never overwrite newer local state.
