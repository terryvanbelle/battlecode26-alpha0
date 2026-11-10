# Terminal-root-ranking-v4 mechanism test

This experiment asks whether complete root-action coverage and real terminal
rollouts can rank actions more usefully than the existing shallow PUCT search.
It follows the [precommitted plateau investigation](warmstart-plateau.md), using
frozen warmstart checkpoint 2. It does not train a network, change the engine,
or test benchmark strength. The prior's 20/24 smoke score remains unchanged
until a new learned checkpoint is separately evaluated.

## Frozen comparison

[`configs/root-ranking-v4.json`](../configs/root-ranking-v4.json) pins checkpoint
SHA-256 `20d9e813349010630940e1cdfbaee5206490c7d323997688d26f47508bbeee7c`.
Four completed own-self-play transcripts, `iter-0001-game-004` through `007`,
supply the first baby and king turn where powered-bite macro 37 is legal. This
selection depends on recorded state, role and legality, not on continuation or
benchmark outcomes. All roots and their replay-prefix/state hashes are frozen
before any action is scored. Source, map, checkpoint, configuration and runtime
hashes bind the experiment.

An independent replay found eight qualifying roots on `tiny`, `DefaultSmall`,
`closeup` and `knifefight`, with 18, 29, 22, 19, 22, 26, 21 and 30 legal actions.
This creates **374 selection continuations** before independent validation.
The same states are reconstructed by the runner and checked against their
frozen identities; engine snapshots preserve physical and random-number state.

| Component | Fixed setting |
| --- | --- |
| Selection | Every legal first action; policy RNG seeds 103003 and 103013 |
| Continuation | Frozen checkpoint 2 for both teams, temperature 0.35, through a real terminal outcome |
| Ranking | Highest mean score; ties use the original policy prior, then action ID |
| Comparator | Existing PUCT, 32 simulations, depth 12 robot turns, root noise enabled; seed 20261114 + root ordinal |
| Validation | The selected action and PUCT action, paired seeds 103021, 103043, 103067 and 103091 |
| Score | Win 1, actual draw 0.5, loss 0, from the root actor's team perspective |
| Allocation | One actor and one Torch thread; at most two hours per invocation |

If both selectors choose the same action, the four validation continuations are
shared. They are not run twice or counted as independent evidence. Thus 32–64
unique validation games are needed, giving 406–438 total terminal continuations.
Selection outcomes are never included in validation scores. These are policy
sampling seeds, not changes to the archived engine/map seeds. They are disjoint
from the frozen benchmark confirmation seeds.

All eight roots must have complete valid selection and validation evidence before
the research decision can pass. Compute a paired validation score difference for
each root, then weight roots equally. The precommitted requirements are:

- Mean score advantage of at least 0.10 over PUCT.
- Strict improvement on at least two distinct roots.
- No root's mean score difference below −0.25.

A pass permits review of a separately versioned training pilot. It is not a
promotion, a statistical superiority claim or any part of the all-map victory
gate. The roots come from a narrow training distribution, and only four shared
validation RNG seeds are used. Preserve failures and incomplete results as well
as favorable outcomes. No SGD is authorized within this diagnostic.

## Run, stop and recover

```bash
.venv/bin/python -u -m alpha0.root_ranking_v4 --seconds 7200
```

The supervisor takes its own `runs/root-ranking-v4/experiment.lock` plus the
neutral campaign and warmstart training locks. Inspect process commands and
locks before starting a second invocation. The runner respects all three STOP
files: `runs/campaign/STOP`, `runs/warmstart-v3/STOP` and
`runs/root-ranking-v4/STOP`. It never removes them automatically.

The supervisor enforces a wall-clock deadline and per-task watchdogs on its one
Python/Java process group. A continuation interrupted by a deadline, STOP or
error has no score. Successful cells are immutable and reused on resume; errors
and start records remain as evidence. Missing or corrupt identity-bound inputs
require restoration or a new documented version, not silent replacement.

All essential recovery data are in `progress/root-ranking-v4/`: the identity,
configuration, exact runner source, frozen roots, PUCT comparator records,
per-cell attempts, selections, summaries and invocation records. The ignored
`runs/root-ranking-v4/` directory holds replaceable process-control files and
logs. Publish the progress directory at review boundaries. Restore a missing
warmstart dependency from its verified checkpoint-2 archive following
[the recovery caveat](warmstart-v3.md#checkpoint-2-current-gate-and-next-research-decision),
preserving current code/gate and newer work. Re-run the same command to complete
missing cells; do not rerun successful cells or change seeds to improve a score.

Regenerate graphs from observed records only:

```bash
.venv/bin/python scripts/plot_root_ranking.py
```

The [progress report](../progress/root-ranking-v4/README.md) separates selection
from independent validation, counts unique completed games, and shows unresolved
coverage explicitly. Passing score arithmetic is provisional until the supervisor
has completed its final input/runtime checks. The final benchmark requirement
remains all **888 opponent–map pairs** under both stages of
[`alpha0.victory`](../alpha0/victory.py).
