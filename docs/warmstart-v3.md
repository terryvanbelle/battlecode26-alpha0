# Explicit teacher prior followed by MCTS

`winner-prior-mcts32-v3` tests whether actual search-guided self-play can improve
the separately published winner-only teacher initialization. It uses the
unchanged v2 model, observations, macro actions and engine. It is **AlphaZero-style
learning with an explicit rule-policy prior**, not learning from random weights.
The neutral v2 campaign remains a separate control.

The initialization artifact is
[`candidate-002-winners/warmstart.pt`](../artifacts/bootstrap-v3/candidate-002-winners/warmstart.pt),
SHA-256 `c1473de81dda381c9f14868ea246c11f5bbd644fb59b0b0420e547598b26aa0e`.
Its six own-rule self-play games, winner-only behavior-cloning recipe, limitations
and source hashes are described in [the bootstrap report](bootstrap-proposal.md).
No benchmark source, actions, trajectories or outcomes become training examples.

The runner loads **model weights only** into a newly constructed learner. Search
replay starts empty; AdamW state is fresh; the final value linear layer is exactly
zero. It verifies all three facts, writes a checksum-bound initialization marker,
and publishes checkpoint 0 before evaluation or self-play. Teacher data retains
its separate provenance and is never relabeled as MCTS visits.

## Bounded experiment

[`configs/warmstart-v3.json`](../configs/warmstart-v3.json) freezes the training
configuration and teacher lineage. The first requested cycle is one iteration:

| Setting | Value |
| --- | --- |
| Training maps | `tiny`, `DefaultSmall`, `closeup`, `knifefight` |
| Seed | `20261112` |
| Completed games per iteration | 8 |
| Search | 32 simulations, depth cap 12 robot turns |
| Search decision probability | 10% babies; 25% kings |
| Replay | Game-balanced; at most 512 positions/game and 6,000 total |
| Optimizer | 48 minibatches of 32; AdamW learning rate 0.0003 |
| Runtime allocation | One actor, one Torch thread, one evaluation worker |
| Evaluation | Checkpoint 0 and checkpoint 1; all 12 frozen bots, both seats, `tiny`, seed 0 |
| Evaluation watchdog | 1,800 seconds/cell, matching the neutral v2 control |
| Invocation budget | At most two hours, including evaluation |

Only searched positions produce policy targets: legal PUCT visit distributions.
Values use completed self-play outcomes from each position's team perspective.
Nonsearched actions are sampled from the current network. The existing learner
also supplies root noise, map/seat variation, past-checkpoint opponents, episode
reservoir sampling and game-balanced replay. Failed or interrupted games do not
become draws or training examples.

The policy prior, search budget and search fractions differ from the neutral v2
control, so a comparison between their final scores compares whole recipes. The
checkpoint 0 versus checkpoint 1 comparison within this experiment isolates the
effect of its completed MCTS training iteration under the same evaluation
runtime. One development map and one seed cannot establish general strength.
Neither a teacher fit nor a smoke score passes the standard and untouched
holdout victory gates.

## Run, stop and resume

```bash
.venv/bin/python -u scripts/run_warmstart_experiment.py \
  --hours 2 --target-iteration 1
```

The process owns `runs/warmstart-v3/experiment.lock`; another invocation exits
without starting duplicate work. Status, PID and logs are in that directory.
Evaluation cells, reports and progress graphs are in `progress/warmstart-v3`.
Create `runs/warmstart-v3/STOP` for a cooperative stop; explicitly remove it
before an intended restart. The runner checks initial CPU headroom, and its
deadline/STOP cleanup tracks only its own evaluation descendants, including
workers that start separate process sessions.

Re-running the same command resumes the exact immutable configuration, saved
plans, completed episodes, optimizer, RNG and replay generation. It validates
the original checkpoint 0 before constructing the learner, so a missing latest
checkpoint cannot silently replace the published baseline. Successful evaluation
cells are reused only under their exact checkpoint and evaluation identities.
Baseline and trained reports must have identical evaluation configuration and
frozen-suite hashes before a comparison is accepted.

`--target-iteration` is an operational bound, not part of the immutable training
configuration. After checkpoint 1 retained 20/24 valid smoke wins, continuation
to iteration 2 was authorized, conditional on verifying the iteration-1 recovery
archive. It uses the same configuration and one-actor allocation:

```bash
.venv/bin/python -u scripts/run_warmstart_experiment.py \
  --hours 2 --target-iteration 2
```

This second cycle can use the critic learned during the first cycle. The prior
champion stays preserved. There is no automatic promotion, holdout evaluation or
change to neutral v2; the runner stops at the requested operational target.

Seven targeted tests cover exact model import, empty replay/fresh optimizer,
initialization crash recovery, checkpoint protection, checksum and comparison
identity checks, exclusive locking, operational iteration targets and cleanup of
thread-spawned workers in separate sessions. The first launch passed these checks
and independent review before starting real self-play.

## First completed cycle and recovery

The first cycle completed in 14.8 minutes: eight terminal self-play games,
17,531 real decisions, 1,948 genuine search targets and 48 optimizer updates.
Checkpoint 1 retained the initial **20/24** smoke wins with no errors. The four
losses against `bestsofar` and `basic51` had exactly unchanged terminal states,
including king HP, surviving armies and round counts. This is preservation of
the prior's smoke performance, not an observed strength gain. See the
[all-cell comparison and terminal diagnostics](../progress/warmstart-v3/iteration-001.md)
and [progress graphs](../progress/warmstart-v3/README.md).

After the runner exits, create its durable resume archive with:

```bash
.venv/bin/python scripts/export_warmstart.py --iteration 1
```

The exporter requires the runner lock to be free and both completed checkpoint
evaluations to match. It preserves checkpoints, latest optimizer/RNG, active
replay and its RNG, initialization dependency, plans, game transcripts,
project-owned sources, maps, notices and progress evidence under
`artifacts/warmstart-v3/iteration-001/`. The archive and each file have SHA-256
checksums. Redundant consumed game arrays and older replay generations are
omitted with recorded hashes; local files remain intact. Six focused tests
verify corruption rejection, locking and exact state recovery; an additional
regression rejects dependency payloads. Restore at the original checkout path
and run the archived hash-pinned bootstrap scripts as documented in `RESUME.md`.
Downloaded benchmark sources, all compiled classes, vendor engine JARs, JDK
files and the Python environment are excluded. The archive retains repository
commits, source/runtime hashes and setup scripts to reproduce those local
dependencies, together with `LICENSE` and `THIRD_PARTY.md`.

A future separate pure-current-self-play control could isolate league-related
target shifts. Same-policy games retain searched positions from both players;
historical-opponent games retain only the current learner's searched positions.
Winning those league games therefore changes the positive-label proportion.
Game-balanced replay does not balance wins, sides or opponent identities, and
the network observes physical state rather than an explicit opponent identity.
No active immutable configuration was changed for this observation.
