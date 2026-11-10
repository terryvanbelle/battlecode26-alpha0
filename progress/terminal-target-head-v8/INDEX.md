# Follow V8 progress

**Current published review:** [training complete, benchmark running](reviews/20261110T201559Z/README.md). The live run may be ahead of this snapshot.

- Collection: all 1,508 selection continuations valid, zero unresolved errors.
- Fitting: both fixed 2,048-update heads match all 32 training targets. The terminal head changes 14 training actions; the matched PUCT head changes none.
- Independent first-action validation: 64 shared terminal games completed; mean advantage over cp2 and PUCT is **zero**.
- Whole-policy benchmark: **190/768 games** in this snapshot, zero errors. These first games are the fresh cp2 control; candidate performance is still pending. No improvement or promotion is established.

[Fitting curves and W/L/D coverage](reviews/20261110T201559Z/README.md) · [Input audit](reviews/input-audit-20261110.json) · [Saved-fit audit](reviews/fit-audit-20261110.json) · [All dated reviews](reviews/) · [Preregistered design](../../docs/terminal-target-head-v8.md)

The saved-fit auditor independently reconstructs the full deterministic batch stream, verifies optimizer steps and byte-identical frozen tensors, and recomputes training/anchor actions. This establishes fitting integrity, not game strength. Full recovery export and final integrity review wait until the bounded cycle is quiescent.

The [top-level README](../../README.md) continues to show the completed cp2 baseline (83/256 wins). Candidate points will be added as their benchmark evidence becomes available, with incomplete coverage explicit.
