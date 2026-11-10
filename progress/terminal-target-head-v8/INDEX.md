# Follow V8 progress

**Current review (2026-11-10 23:13 UTC):** the fixed terminal candidate completed all256 games: **81 wins / 175 losses / 0 draws (31.6%)**. It fails the development improvement rule against fresh cp2 (82wins) and original cp2 (83wins). The PUCT control is still running; no model is promoted.

[Candidate review and next hypothesis](../../docs/v8-candidate-review.md) · [Independent terminal-panel audit](reviews/completed-panels-20261110T2313.json) · [Current dashboard](../../README.md)

- Collection: all1,508 selection continuations valid.
- Fitting: both fixed2,048-update heads match all32 training targets. The terminal head changes14 training actions; PUCT changes none.
- Independent first-action validation:64 shared terminal games; mean advantage over cp2 and PUCT is zero.
- Published benchmark snapshot: **668/768 valid games**. Fresh cp2:82W/174L, complete256. Terminal candidate:81W/175L, complete256. PUCT:73W/83L, partial156/256. The partial PUCT rate is not a full-suite strength estimate. The live run may be ahead of this snapshot.

[Training curves](reviews/20261110T201559Z/README.md) · [Input audit](reviews/input-audit-20261110.json) · [Saved-fit audit](reviews/fit-audit-20261110.json) · [All dated reviews](reviews/) · [Preregistered design](../../docs/terminal-target-head-v8.md)

The first cycle stopped at its time budget. Two interrupted cells were independently reviewed and recovered on attempt2; original SIGTERM error records remain unchanged. [Interruption evidence](reviews/deadline-recovery-20261110.json). The414-game recovery archive was verified and isolated-restored before the second bounded cycle began. [Recovery pointer and proof](recovery.json). Final experiment audit and a new recovery generation wait for completion and quiescence.

The one-game difference between original and fresh cp2 is documented in the [repeatability review](reviews/cp2-repeatability-20261110.json): frozen rapierV1 seeds randomness from elapsed execution time. All valid results are retained without outcome-selected reruns.

Operational status: `runs/terminal-target-head-v8/status.json` and `runs/research-sequence/boundary-status.json`. The continuation watcher currently owns the resumed cycle; do not launch a duplicate. Next research remains an unlaunched proposal until the full matched control and final audit are reviewed.

Preparation for the next review now includes an [exact 32-root action inventory](../../diagnostics/target-recheck-planning/action-inventory.json) and [preliminary seed scan](../../diagnostics/target-recheck-planning/seed-inventory.json). No new games or gradients were generated; the proposed184-cell diagnostic remains unlaunched.
