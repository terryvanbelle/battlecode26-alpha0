# V7: sufficient fitting budget with anchors retained

The preregistered cached-head diagnostic completed all six fixed snapshots in
20.99 wall seconds. Both arms fitted all 32 selected training actions at 2,048
updates. The ordered rule therefore yields **budget evidence**. No games were
run and no model was promoted. The all-map victory gate remains unmet.

The KL=1 control at 128 updates exactly reproduced the v6 terminal model,
AdamW state, loss history, Torch RNG and shared batch prefix. Both fresh-cp2
fits used the same complete 32-root, 512-anchor training dataset, batch stream,
optimizer and 5,160 trainable output parameters. Encoders and critic stayed
unchanged. Only the anchor coefficient differed.

| Arm | 128 updates | 512 updates | 2,048 updates | Final anchor changes |
|---|---:|---:|---:|---:|
| KL coefficient 1 | 23/32 | 27/32 | 32/32 | 3/512 |
| KL coefficient 0 | 23/32 | 28/32 | 32/32 | 73/512 |

Both arms retained all 23 initially agreeing labels at every snapshot. All nine
changed labels crossed the greedy action boundary by 2,048 updates. The
regularized arm's correctly game/team-weighted anchor KL was 0.039149 at the
final snapshot, versus 0.358944 without the anchor term. This supports retaining
anchoring while testing a larger fixed fit budget; it does not prove optimal
regularization, label correctness or generalization.

[Measured curves and all six snapshots](../progress/cached-head-constraint-v7/README.md)
include unsmoothed loss observations, target agreement, anchor changes, parameter
displacement and gradient alignment. The worker used 14.35 CPU seconds; parent
verification and reporting used 6.35 CPU seconds. The 900-second cumulative
limit was respected, with no errors, no Java engine and no validation or
benchmark data read for this diagnostic.

## Research boundary

The [independent final audit](../progress/cached-head-constraint-v7/independent-final-audit.json)
verified all 529 fit files, 256 recovery generations and six snapshots, exact
v6 reproduction, frozen tensors and all 544 cached critic outputs. It recomputed
the metrics and classification without any optimizer steps. Audit SHA-256:
`e3776f8fbde2d0ff65117c2efcf029a54265a734d00b1b069ada06dc70b6ebd2`.
The [separate recovery archive](../artifacts/cached-head-constraint-v7/README.md) passed staging and checkpoint verification; an isolated restore awaits baseline quiescence.
This is a training-fit result only. V6 remains the latest measured policy
comparison: cp2 and both v6 heads scored 20/24 smoke, with zero validation
advantage. V7 diagnostic heads are retained without benchmark evaluation or
promotion. A subsequent real-game learning experiment requires a fresh,
preregistered validation plan and review of selection noise and source coverage.
