# Follow V8 progress

**Current review (2026-11-10 22:25 UTC):** **414/768 valid benchmark games**, with both deadline-interrupted cells recovered on their second and final permitted attempts. Original error journals remain unchanged. The verified recovery archive has been published and a second bounded cycle has resumed. [Recovery proof](recovery.json) · [Interruption review](reviews/deadline-recovery-20261110.json).

Current panels: cp2 control **82 W / 174 L / 0 D (256/256)**; terminal candidate **77 W / 81 L / 0 D (158/256)**; PUCT pending. No unresolved errors in this snapshot. Partial candidate results are not a full-suite strength estimate. The original baseline remains 83/256 wins.

- Collection: all 1,508 selection continuations valid, zero unresolved errors.
- Fitting: both fixed 2,048-update heads match all 32 training targets. The terminal head changes 14 training actions; the matched PUCT head changes none.
- Independent first-action validation: 64 shared terminal games completed; mean advantage over cp2 and PUCT is **zero**.
- Whole-policy benchmark: **342/768 valid games** in this snapshot, zero errors. Fresh cp2 control: **82 W / 174 L / 0 D**, complete 256-game panel. Terminal candidate: **56 W / 30 L / 0 D**, only **86/256** games; PUCT control pending. Partial candidate results are not an estimate for the full suite. No improvement or promotion is established.

[Fitting curves and W/L/D coverage](reviews/20261110T201559Z/README.md) · [Input audit](reviews/input-audit-20261110.json) · [Saved-fit audit](reviews/fit-audit-20261110.json) · [All dated reviews](reviews/) · [Preregistered design](../../docs/terminal-target-head-v8.md)

The saved-fit auditor independently reconstructs the full deterministic batch stream, verifies optimizer steps and byte-identical frozen tensors, and recomputes training/anchor actions. This establishes fitting integrity, not game strength. Full recovery export and final integrity review wait until the bounded cycle is quiescent.

The [top-level README](../../README.md) now shows the partial candidate opponent-by-map dashboard and combined history. The original baseline remains 83/256 wins; its one-game difference from the fresh control is preserved in the [repeatability review](reviews/cp2-repeatability-20261110.json). Frozen rapierV1 seeds randomness from elapsed execution time, so matching game seeds do not guarantee identical games. No outcome-selected reruns are used.

A guarded continuation watcher is waiting for the current cycle’s budget boundary. It will independently verify, export and restore-check saved state before resuming unfinished benchmarks for one further bounded cycle. Operational status: `runs/research-sequence/boundary-status.json`.
