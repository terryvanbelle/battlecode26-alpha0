# V8 complete

All **768/768 benchmark games** are valid. Fresh cp2: **82W / 174L**; terminal candidate: **81W / 175L**; PUCT control: **79W / 177L**. No draws or unresolved errors. Both learned heads fail the improvement rule. No promotion or confirmation was triggered.

[Final terminal-panel audit and both baseline comparisons](reviews/completed-panels-final-20261111.json) · [Verified final recovery](recovery.json) · [Candidate analysis](../../docs/v8-candidate-review.md) · [Topline graphs](../../README.md)

All1,508 selection continuations, both2,048-update fits, and64 physically shared independent validation games completed. Independent first-action advantage was zero. Two deadline-interrupted benchmark attempts remain preserved; both recovered validly on attempt2 under the unchanged runtime. [Recovery review](reviews/deadline-recovery-20261110.json).

The complete archive is verified and isolated-restored separately from the earlier414-game generation. Both generations remain preserved. Final fitting and input audits passed again. No completed fit or successful game was rerun.

The next version is a [fresh-seed target-reliability recheck](../../docs/target-recheck-v9.md), without SGD, to distinguish noisy selected targets from poor transfer. Its outputs are separate from V8. Historical snapshots remain in [reviews](reviews/).
