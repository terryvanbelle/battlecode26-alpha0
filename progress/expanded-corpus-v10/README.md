# V10 phase A — expanded self-play sources

Running64 fixed own-self-play source games across eight maps. Both teams use frozen cp2 at temperature1.0. No target scoring, SGD, validation or benchmark games in this phase.

[Measured source coverage](summary.json) · [Frozen schedule](schedule.json) · [Runtime identity](identity.json) · [Protocol and downstream gates](../../docs/expanded-corpus-v10.md)

After64/64 valid sources, deterministic review selects64 training roots and1024 anchors, checks missing slots and the3200-cell target budget. Future validation and learner/evaluator code require a separate tested freeze before target outcomes or gradients. Coverage failure stops for review; no dropped maps or reduced denominators.

The prior [V9r2 mechanism test](../target-recheck-v9r2/README.md) passed with mean paired advantage0.1484375, but no benchmark improvement has been established. The top-level README continues to show actual playing strength.

Operational status: `runs/expanded-corpus-v10/status.json`; log `/tmp/expanded-corpus-v10.log`. Three one-thread collectors at most, separate Java engines, nine shared/own locks and STOP checks. Completion reviewer/exporter are separate from pinned source runtime. Never edit pinned collector/config/protocol while active or duplicate its owner.
