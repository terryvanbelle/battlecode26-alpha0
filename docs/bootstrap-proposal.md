# Optional curriculum initialization: separate bootstrap-v3 experiment

This proposal initializes a fresh network from **our own rule-controlled
self-play**, then returns to search-guided reinforcement learning. It is an
explicit expert prior, not tabula-rasa AlphaZero and not a change to the neutral
v2 experiment. The architecture, observations, official engine and 40-action
macro schema remain v2; the experiment lineage is `macro40-bootstrap-v3`.

## Why test a prior

Eight simulations among as many as 40 legal macros cannot even visit every root
action once. The sparse-search campaign searches only 15% of baby decisions and
35% of king decisions. A tree depth of twelve robot turns may not reach the next
king turn in a populated game. Until values become useful, a nearly uniform
policy may spend many complete games discovering a basic sequence such as
spawn, navigate, attack, return cheese. A small rule curriculum can supply those
sequences and let subsequent MCTS refine them.

This is a hypothesis about sample efficiency, not a claim that scripted play is
optimal or that imitation guarantees improvement. Fixed teachers restrict the
initial exploration distribution and can pass their mistakes to the student.
The neutral v2 run must remain a separate, comparable baseline.

## Teacher and data boundaries

[`alpha0/bootstrap_v2.py`](../alpha0/bootstrap_v2.py) contains the complete teacher
rules. Both teams use the same macro interface and physical engine as the
learned policy. No benchmark source, evaluation log, opponent trajectory or
benchmark outcome is loaded. Only the victory-gate metadata is read, to reject
reserved holdout maps. Teacher decisions depend on present engine state and a
fixed style, not future engine randomness or an opposing private program.

Three opening styles are paired in both seat assignments:

- **Assault:** kings prefer assault-oriented spawning; babies approach enemy
  kings, returning carried cheese when holding at least ten.
- **Forage:** kings prefer economic spawning and relocation; babies gather
  cheese and occupy mines.
- **Guard:** kings build defenders; babies guard allied kings or hunt cats.

Babies at 20 HP or less prefer retreat when legal. Forage and guard openings
switch to assault at round 250; resource return still takes priority when a baby
holds at least ten cheese. Every priority falls back to another legal macro.
The teacher distribution assigns 98% to its chosen legal action and mixes 2%
uniform exploration over legal actions. These constants are published rules,
not hidden learned weights. Different styles are not an extra network input, so
the student learns their pooled behavior where state alone does not distinguish
them.

Only complete games produce labels. A uniform reservoir keeps at most 256
positions from each game. Policy targets are the teacher's distributions,
**not MCTS visits**. Value targets are actual terminal outcomes, converted to
each position's team perspective. An interrupted or failed game is preserved as
an error transcript and does not produce fabricated draws or a trained output.

Teacher data uses `source="scripted_selfplay"` and separate files, deliberately
different from the search replay's accepted `source="selfplay"`. It must not be
silently mixed into the neutral/search replay. The module records source,
engine, patch, Java-source, map and data hashes alongside complete action
transcripts. Benchmark data never enters this path.

## Default optional configuration

| Setting | Value |
| --- | --- |
| Maps | `tiny`, `closeup`, `knifefight`; all disjoint from reserved confirmation maps |
| Games | Six; each adjacent pair swaps teacher styles on the same map and seed |
| Teacher pairs | Assault/forage, assault/guard, forage/guard, each reversed |
| Storage | At most 256 positions/game; numeric arrays without object pickle |
| Sampling | Choose a completed game uniformly, then a retained position |
| Student | Fresh 87,973-parameter dual-scale v2 network; neutral value initialization |
| Fitting | 256 minibatches of 32; AdamW at 0.0003; gradient norm cap 1 |
| Objective | Teacher-policy cross entropy + 0.25×terminal win-logit BCE |
| Local watchdog | 900 seconds/game and 100,000 decisions/game; incomplete games rejected |

The reduced value weight emphasizes the initialization's purpose: learning
basic action sequences without trusting a few teacher games as a calibrated
long-horizon value function. Later MCTS training should restore the ordinary
terminal value objective. Fit diagnostics use a fixed **training-data sample**;
they are not validation or a strength estimate.

Run a new optional experiment:

```bash
.venv/bin/python -m alpha0.bootstrap_v2 \
  --out runs/bootstrap-v3/own-rules-001
```

A small plumbing check:

```bash
.venv/bin/python -m alpha0.bootstrap_v2 \
  --out runs/bootstrap-v3/smoke-001 --games 1 --steps 8 \
  --positions-per-game 64 --batch-size 16 --maps tiny --game-timeout 180
```

The destination must be new or empty. The module never overwrites a current
run, edits its optimizer, modifies production checkpoints, or activates the
student. It exports both `neutral-initialization.pt` and `warmstart.pt`, plus
configuration, transcripts, data shards and fit diagnostics. The warmstart
checkpoint is visibly tagged with teacher lineage and `training_stage`.

The actual smoke check used `tiny`, seed 20261111, assault versus forage, 64
retained positions and eight optimizer steps. The official game finished at
round 20 after 405 decisions, with team A winning. On the fixed training sample,
policy cross entropy fell from 3.155 to 2.401, teacher argmax agreement rose from
0% to 87.5%, and value MSE changed from 1.000 to 0.989 with no saturated values.
This validates generation, terminal labeling and student fitting only. It is
one game, not evidence of stronger Battlecode play. Independent checks verified
that illegal policy targets are zero, value labels have the correct team
perspective, holdout maps and existing output directories are rejected, and
checkpoints load with `weights_only=True`. The local evidence is under
`runs/bootstrap-v3/smoke-002/`; it has not been activated as a campaign.

## Returning to AlphaZero-style learning

The artifact is intended as an **initialization**, not a replacement for MCTS.
After review, start a new campaign lineage, load only `model_state_dict` into a
fresh v2 learner, reset its optimizer, and start an empty search-target replay.
Preserve the warmstart checkpoint hash and teacher metadata in every descendant
checkpoint and report. The existing production run is not resumed with these
weights. This module does not wire automatic campaign activation; that remains
an explicit experiment setup step.

Continue self-play with real terminal outcomes, legal PUCT visit targets, root
noise, map/seat variation and past learned opponents. Teachers can be discarded
after initialization; benchmark bots remain evaluation-only. Compare the new
run against its exported neutral initialization and against the untouched v2
baseline under the same runtime identity and evaluation policy. Report teacher
fit gains separately from post-bootstrap reinforcement-learning gains.

Do not claim improvement because teacher agreement increased. A new policy
still has to qualify against every frozen opponent under the existing standard
and untouched holdout rules. If initialization improves only early fitting or a
single training map, keep that as a failed or limited experiment rather than
silently promoting it.
