# Unchanged warmstart losses: terminal evidence only

Warmstart checkpoints 0, 1 and 2 lost the same four smoke cells against `abcdezyxwv_bestsofar` and `IvanGeffner_basic51`. All twelve games were valid losses by `KILL_ALL_RAT_KINGS`, with zero benchmark failures. For each opponent/seat cell, the **complete terminal state and decision count are identical across all three checkpoints**. These are four distinct scenarios on `tiny`, seed 0, repeated under three policies.

The additional visible pattern is enemy capture of surviving babies. In the two `bestsofar` losses, enemies held 7/11 and 9/11 surviving learned-team babies. The learned team held no enemy babies in any of the four terminal states. These are endpoint observations, not a reconstruction of how the king died.

| Opponent / learned seat | Round | Own babies | Captured by enemy | Thrown | Neither carried nor thrown | Enemy babies | Enemy king HP | Own / enemy bank cheese |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `abcdezyxwv_bestsofar` A | 42 | 11 | 7 | 1 | 3 | 19 | 470 | 986 / 1621 |
| `abcdezyxwv_bestsofar` B | 54 | 11 | 9 | 0 | 2 | 16 | 590 | 697 / 1859 |
| `IvanGeffner_basic51` A | 94 | 1 | 1 | 0 | 0 | 17 | 600 | 1 / 630 |
| `IvanGeffner_basic51` B | 60 | 8 | 3 | 0 | 5 | 16 | 600 | 845 / 1336 |

Every learned king was dead; each enemy had one surviving king. No enemy baby was carried or thrown in these terminal states. “Captured” means `carried=true` with an opposing carrier resolved by robot ID; “neither carried nor thrown” does not mean ready to act or tactically effective. Bank cheese excludes cheese held by robots.

Three defeats left 697–986 cheese in the learned bank, while `basic51` seat A left only 1. This does not support a universal inability to spend. The records do not show whether spending, rescue, positioning or another legal choice would have changed the result.

## Contrast with independently collected own self-play

The existing [target audit](../diagnostics/warmstart-plateau/targets.json) covers 3,751 retained search targets from 16 own-self-play games. Among those targets:

| Macro | Visit-target argmax count |
|---|---:|
| `assault_king` (21) | 2,700 |
| `retreat` (24) | 3 |
| `king_relocate` (25) | 0 |
| `spawn_assault` (27) | 262 |
| `carry_enemy` (32) | 0 |
| `rescue_ally` (33) | 1 |
| `throw_damage` (34) | 0 |
| `throw_transport` (35) | 0 |
| `drop_safe` (36) | 0 |
| `powered_bite` (37) | 0 |

This table lists selected macros only. Counts describe **search-target argmax**, not executed actions, and are **not conditioned on legality**. Zero argmax count does not establish that an action was legal, useful or necessary in every state. The existing [tactical probe](../diagnostics/warmstart-plateau/counterfactual-v1/summary.json) did expose four own-self-play baby roots with `carry_enemy` legal, but tested only the greedy action, powered bite and retreat. It did not test capture at those roots.

The probe also found a context-dependent retreat contrast: the `tiny` king’s two continuations changed from losses to wins, whereas the `closeup` baby’s two changed from wins to losses. Those selected own-self-play states and two continuation seeds do not establish a blanket tactical rule. See the [plateau analysis](warmstart-plateau.md).

These observations motivate measuring capture/control and king-survival coverage in separately generated own self-play. They do not establish a causal explanation of the benchmark defeats or justify copying benchmark behavior. No current experiment, actor, policy, feature, runtime or training target was changed for this audit.

## Provenance and limits

- [Machine-readable audit](../diagnostics/warmstart-plateau/benchmark-terminal-audit.json): per-checkpoint model hashes, all twelve complete terminal-state hashes, both teams’ exact counts, result-artifact paths and canonicalization definitions.
- [Source evaluations](../progress/warmstart-v3/evaluations.jsonl): terminal states and total decision counts; **no per-turn benchmark action traces**. All twelve inspected worker logs are empty. Identical terminal states do not prove identical action histories.
- Benchmark observations in this audit are descriptive evaluation evidence only. **No benchmark state, action or outcome enters replay, policy/value targets or gradients.**

Hashes below bind the snapshot inspected for this document:

- Source JSONL file SHA256: `57ee9ff0194d4c0b490bde11618b0544afb72a098c49378e87323fa9f0f5555b`.
- Canonical complete evaluation-row list SHA256: `69c1542cf8a46a6059a52cbf6a21b26de6d059bae434b7746c97e9da369b6697`.
- Canonical twelve selected rows SHA256: `ac3c86861829bb0c7cef17df84827c9927f8abddbbe25f990fff0a10eafb2f0b`.

| Checkpoint | Model SHA256 |
|---|---|
| CP0 | `250c37e9d47ee647933d64c3988398042bb71f626754423900428cf878fe2f6a` |
| CP1 | `121acf1bd12d6f44c4909766167011664106605f1325371a585ed10a118d1360` |
| CP2 | `20d9e813349010630940e1cdfbaee5206490c7d323997688d26f47508bbeee7c` |

Model hashes were checked against the local checkpoint bytes. Terminal hashes use compact, sorted-key JSON of the entire terminal object, not the summary columns above.
