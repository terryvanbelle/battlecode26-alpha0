# V6: complete collection, fitted heads, no measured gain

The controlled v6 pilot completed correctly and **failed its research improvement
rule**. Fresh cp2, the terminal-ranking head and the matched PUCT head each won
**20/24** smoke games. All three still lost both seats to `bestsofar` and `basic51`.
No model was promoted; the all-opponent/all-map victory gate remains unmet.

[Independent final audit](../progress/terminal-target-head-v6/independent-final-audit.json)
SHA-256: `ab8f824a645432916417e8142d61c7cd13ec5e69561b96869115ff95288dd564`.

## Complete evidence

| Stage | Completed evidence |
|---|---|
| Own source self-play | 19 training + 9 validation games |
| Training inputs | 32 balanced roots + 512 anchors; 544 distinct inputs |
| Validation inputs | 16 distinct roots, disjoint from all 544 gradient inputs |
| Search controls | 32 fixed PUCT32/depth12 comparators |
| Terminal targets | 1,458/1,458 all-legal-action paired continuations |
| Fitting | Two fixed heads, 128 AdamW updates each, only 5,160 policy parameters |
| Independent validation | 64 unique terminal games, shared across identical actions |
| Whole-policy benchmarks | 72/72 fresh games: all 12 bots, both seats, three policies |

There were **zero errors, retries or incomplete attempts**. The 1,594 scored
games plus 28 source games are 1,622 terminal games; branching continuations do
not create additional independent training roots. V6 used 32 training roots
from 18 represented source games (19 generated), and 16 validation roots from
nine source games. All original experiments, candidates and failed coverage
evidence remain preserved.

The symmetric source-temperature change from .35 to 1.0 repaired v5's coverage
shortfall without dropping a map or slot. Same-seed initial coverage changed
24/32 to 30/32, and fixed reserves supplied the remaining two slots. This is
[source coverage evidence](../progress/terminal-target-head-v6/source-coverage-comparison.md),
not a learned strength gain. Continuations stayed at .35; no engine, action,
observation, inference or PUCT implementation changed.

## What the heads learned

Terminal labels differ from PUCT/cp2 on **9 of 32 roots**. Their selected-action
probabilities increased, but neither fitted head changed any of the 32 training
root greedy actions. Anchor actions changed on 3/512 inputs for the terminal
head and 1/512 for PUCT. Every one of the 16 validation roots also retained the
same action for all three policies. The planned physical deduplication therefore
required 64 games rather than 192; no redundant game was counted as independent
evidence.

The equal-root validation advantage was **0** against both PUCT and cp2, with
**zero improving roots** and worst difference 0. Per-opponent smoke nonregression
passed only as ties on all 12 opponents. The required positive validation
advantage and two improved roots were absent, so the expansion rule failed.
[Every opponent and seat](../progress/terminal-target-head-v6/README.md) is reported.

| Arm | Final target CE | Anchor KL (storage mean) | Target agreement |
|---|---:|---:|---:|
| Terminal ranking | 0.915429 | 0.050669 | 23/32 |
| PUCT control | 0.436549 | 0.015039 | 32/32 |

[Learning curves](../progress/terminal-target-head-v6/learning-curves.md) show all
128 recorded updates. Falling loss is not a strength result. The independent
auditor exactly replayed both head weights **and AdamW state** byte-for-byte;
encoders, fusion, critic and 544 cached critic predictions stayed unchanged.
Both complete candidate files and optimizer/RNG recovery state are retained.

## Observed resources

The invocation ran from 07:48:19 to 09:05:20 UTC on 2026-11-10: **4,620.99 seconds
(77.02 minutes)** within its three-hour cap. Terminal target collection took
3,760.61 seconds from first start to last end (1,395.73 valid cells/hour). The
[throughput curve](../progress/terminal-target-head-v6/collection-throughput.md)
uses actual end records. This is not a controlled measurement of a threefold
parallel speedup. Sampled aggregate worker/Java peak RSS was **3.44 GiB**.

| Supervised phase | Wall seconds | Reaped Python + Java CPU seconds |
|---|---:|---:|
| fit_heads | 3.12 | 3.38 |
| collect_selection | 3763.89 | 10518.55 |
| discover_training | 227.50 | 558.79 |
| discover_validation | 93.35 | 257.34 |
| collect_validation | 211.97 | 638.47 |
| freeze_comparators | 94.14 | 338.90 |
| benchmark | 216.06 | 635.81 |

The per-phase counters exclude some parent-side dataset/report work; their sum
need not equal total invocation wall time. CPU is elapsed processor time, not
a core reservation. No GCloud VM or paid infrastructure was used.

## Research decision and recovery

The [training-only diagnosis](v6-training-underfit.md) identifies remaining
target-fitting error, opposing root-label gradients and anchor pressure. The
32-root feature matrix has full row rank, so a hard linear interpolation barrier
on those roots is not established. This does not prove generalization, the
correctness of the two-seed labels or a stationary optimum.

The failed v6 rule blocks expansion of this learning recipe. The selected next
[controlled fitting diagnostic](cached-head-constraint-v7.md) reuses only the
fixed training data to separate update budget from anchoring. It collects no
new trajectories, inspects no validation/benchmark examples or outcomes for
its choices, and promotes no model. Fresh validation requires a later explicit
protocol; the previous validation set is not reused to tune a diagnostic head.

Raw terminal journals remain locally intact and are included in the separate
[verified v6 recovery archive](../artifacts/terminal-target-head-v6/README.md).
They are excluded from Git's loose progress tree to keep repository history
compact; restore the archive for complete raw evidence and exact fit recovery.
Current main, prior runs and later local work must never be overwritten.
