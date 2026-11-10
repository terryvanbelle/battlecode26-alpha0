# Design decisions

## What is being learned

One policy controls every rat on both teams during self-play. It learns a 17-way decision for each non-forced robot turn: hold, eight movement/dig directions, or eight king-spawn directions. Every option also invokes fixed legal pickup, transfer and opportunistic attack routines. Those routines prioritize enemy rats and may trigger backstabbing; the network cannot explicitly veto an available automatic attack.

This action abstraction is a substantial inductive bias. The model does not yet choose traps, promotion, ratnapping, throwing, squeaks, shared-array messages, attack cheese spending, or the order of every primitive action. A small action space makes an initial full-game CPU experiment feasible. A stronger experiment should expand it and re-evaluate the frozen panel, rather than interpret these results as full-action AlphaZero.

## Observation and topology

The input is a **24 × 17 × 17** actor-centered tensor plus **32** scalar features. Ally/enemy channels are relative to the acting team. Coordinates retain the engine's compass orientation, so directional actions align with the input. There is no memory or recurrent hidden state.

Spatial channels encode on-map mask, wall, dirt, mine, loose cheese, allied/enemy traps, acting robot, allied/enemy babies and kings, cats, normalized health, carried cheese, facing direction, and three cooldowns. Global scalars encode round, cooperation, map size, actor coordinates/type/health/resources/readiness, both teams' cheese and transfers, cat damage, king/baby counts and health, cat count/health, ground cheese, and legal-action count. Exact ordered names and normalization constants are in `alpha0/env.py`.

These are privileged observations: local terrain/entities are omniscient even outside the official vision cone, while aggregate team/enemy information is global. Legality and all physical action effects still use the official controller. Carried and airborne robots contribute to global counts but are omitted from ground occupancy; dedicated carried/airborne status planes are deferred. The encoding is not a complete Markov state even though search branches the full engine state.

The default network uses a 32-channel convolution followed by three residual blocks (two 3×3 convolutions each), GroupNorm and SiLU. Its outputs combine an 8-channel spatial projection pooled to 4×4, global mean/max trunk features, and a 32-unit scalar embedding, through a 128-unit fused layer. Heads produce 17 policy logits and a 64-unit/tanh scalar value. There are **103,290 parameters**. The initial smoke model uses 16 channels and two blocks, **49,242 parameters**, with the same input/output schema.

GroupNorm supports single-position CPU inference without batch-statistic drift. Illegal actions are masked before softmax and loss. The network predicts the win/loss result from the acting team's perspective.

## Search and learning

PUCT uses network priors and leaf values, with exploration constant 1.5. Self-play adds Dirichlet noise (alpha 0.3, mixture 0.25) at every root. Targets are normalized visit counts; move selection applies temperature to the visits. Early rounds use temperature 1, then a lower configured value. Stored targets always retain the original normalized counts.

Battlecode turn order follows spawn order, so the same team can act repeatedly. Search stores all edge values in absolute team-A coordinates and converts them to the current player's sign during selection. It never blindly negates at every depth. Cats and forced turns advance inside the official engine. Snapshot copies include mutable world state, object identity relationships, RNGs and turn cursor; immutable pathfinding tables are shared.

Training minimizes visit-target cross entropy plus mean-squared error against the actual terminal winner, using AdamW, weight decay and gradient clipping. Search-depth cutoffs bootstrap the value network; wall-clock/decision-limit aborts never become fake draw labels. All retained positions from a completed game receive its final outcome in their team's perspective. There is no reward shaping, benchmark imitation or benchmark-game training.

Replay is bounded. Position tensors use float16 storage and float32 computation. A per-game reservoir bounds memory even before a long game finishes; it samples uniformly when a game exceeds its configured position capacity. The initial smoke run uses up to 10,000 retained positions per game and a 10,000-position replay, so a long game can displace earlier games; that is a limitation of the tiny integration run. The larger configuration caps each game at 5,000 positions in a 50,000-position replay to retain more game diversity. Checkpoints store network, optimizer, iteration, RNGs, game/sample counts and configuration. Replay is committed before the matching checkpoint. Continuation requires the same configuration; a different experiment belongs in a separate run/progress directory.

## Evaluation and interpretation

The frozen external panel spans historical skill and different authors. Each map/seed is played from both seats. Baseline iteration 0 and trained checkpoints use identical greedy-network evaluation settings. Evaluation does not apply root noise or train on its games. Legacy opponent execution cannot be cloned safely, so benchmark games measure the network alone; they do not measure the full search-enhanced self-play player.

The smoke panel is only 24 games per checkpoint and the training smoke maps overlap it. It establishes integration and exposes failures, not generalization or statistical strength. The standard panel has 192 games; it is still a development set, so reserve unseen maps/seeds for final claims. Error/timeout rows are reported separately and never quietly converted to game losses. Per-opponent Wilson intervals are descriptive; the aggregate paired bootstrap requires enough complete shared map-seed cells.

Main research risks are sparse terminal feedback, short search horizons relative to 2000-round games, small CPU-scale data volume, action abstraction, privileged observations, and the difference between search self-play and network-only evaluation. These are visible limitations to improve in subsequent experiments, not evidence that a small initial loss reduction implies stronger play.

## Engine and simulator decisions

We pin official Java engine **1.2.6** because that is the version used for the source project's historical benchmark data. The binary, source and Javadoc archives are downloaded from the official release host and SHA-256 verified. Game mechanics are not reimplemented in Python: actions go through `RobotControllerImpl`, cats use the original engine AI, and the original world determines winners. The bridge's manual turn stepping has been compared against native `GameWorld.runRound()` through complete games on two maps.

Search uses deep snapshots instead of replaying every prefix because prefix replay scales quadratically with episode length. Mutable world graphs, aliased references, turn order/cursor, random generators, mines and IDs are copied; immutable all-pairs cat pathfinding tables are shared. Search snapshots are released after every actual move. An independent reset-and-replay path validates branch restoration. Live benchmark bot threads are deliberately excluded from snapshotting.

Three patches govern local execution: raise legacy robot bytecodes to one billion per turn, disable the cumulative team execution timeout, and replace the untracked `Math.random()` final tie-break with the world's seeded RNG. The last change preserves a 50/50 tie-break but changes its reproducibility. Learned actions bypass instrumentation entirely. Instrumentation remains for external Java bots so `Clock.yield()` and per-robot static isolation still work. Five Java module-opening arguments are required by the official instrumented standard library on Java 21.

The no-op replay recorder reduces allocation during search. This version records actions and terminal state instead of generating `.bc26` visual replays. A future replay exporter can reproduce action transcripts using the official engine. A fourth, diagnostic-only patch records voluntary robot termination. Automatic diagnostics distinguish unexpected sandbox termination or initialization errors from legitimate game deaths/resignations; infrastructure failures never become candidate wins.

The repository contains 74 hash-pinned maps from the source project: 43 match maps embedded in the official engine and 31 are additional corpus maps. Their origins are recorded in `docs/rules/PROVENANCE.md`. The source project's engine-checked rules digest is retained alongside the original rule text; actual engine behavior is authoritative when prose differs.

## Benchmark selection decisions

The 12-bot panel combines **ten external authors**, the official example player, and the source project's current accepted champion. It spans the source project's historical rating distribution and includes different strategic families:

| Opponent | Historical Elo used for selection |
| --- | ---: |
| Krak9n / robot | 259.3 |
| seru9 / TOTAL_AGGRO | 788.0 |
| Camleaf / mk1 | 1006.2 |
| CS50Crab / rapierV1 | 1398.7 |
| spsquared / TSPAARK_QUALS | 1584.6 |
| HugoIngelsson / Xzibit | 1695.8 |
| abcdezyxwv / bestsofar | 2010.4 |
| IvanGeffner / basic51 | 2183.4 |
| thewaxmango / collegejunior | 1767.0 |
| uravt / Version41 | 2125.3 |
| Official examplefuncsplayer | Reference baseline |
| battlecode26b-vibe / accepted zdb | Current accepted champion |

Those numbers are **historical selection evidence**, not our measurements. Original downloaded VM artifacts are unavailable here. The named repositories/packages match the prior roster, and their newly pinned public commits predate the historical acquisition; exact equality with the original VM's bytes cannot be proven. The new panel is reproducible through commit IDs and per-file hashes in `configs/benchmarks.json`. The accepted champion was verified against the source project's latest handoff at commit `fc044c24c37d89d2581562e23439a2571bb2b300`: `zdb` remains the incumbent, while the working `ratbot` includes unaccepted changes. The official example comes from a pinned official scaffold commit.

`AdamTan12/Finals` was retired after repeated uncaught off-map errors in its throwing code, confirmed on two seeds. It was replaced with `thewaxmango/collegejunior` at a similar historical skill level, creating suite `field-twelve-v2`. The retired manifest and raw failure evidence remain available, and the replacement is evaluated at every comparable checkpoint. Repeated genuine bot crashes trigger replacement; legal deaths and harness defects do not.

Benchmark sources are downloaded into ignored directories, keeping third-party code separate from our training data and avoiding redistribution of repositories without explicit licenses. Available licenses and provenance are retained. Compilation and every-file hash checks run before evaluation. No benchmark code, game trajectory or target enters replay learning.

## Experiment, compute and persistence decisions

The first run uses `configs/smoke.json`: two games per iteration on `tiny` and `DefaultSmall`, four simulations per decision, depth limit eight, 32 gradient steps of batch size 32, AdamW learning rate 0.001, and replay capacity 10,000. Its purpose is to exercise the complete pipeline and obtain a measured starting point. The standard configuration increases model size, simulations, map diversity and training volume, but remains an unvalidated starting configuration.

The attached executor has a four-core CPU quota and 16 GiB memory limit, with no detected GPU. PyTorch uses one thread for each training/inference process to avoid oversubscription with Java. Current game generation is serial inside an experiment; separate evaluation can run concurrently. Parallel actor workers and batched inference are future throughput improvements, not implemented capabilities.

`battlecode-dev` is **not used**. It is historically an eight-vCPU/31GB machine with another active project, and this executor lacks its SSH/GCloud credentials. Moving to it requires working access and coordination with the owner before changing workloads. Nothing in this repository starts/stops VM services or submits to the Battlecode service.

No tournament-sized JVM heap cap is set. Host memory limits and JVM ergonomic heap sizing remain. The driver has finite request/game watchdogs and decision caps; a timed-out game is an error and cannot train the value head. Checkpoint and replay persistence support recovery, but this is a bounded command-line experiment rather than an always-running hosted training service. Resuming requires the original run configuration; new experiments need separate output directories.

The public GitHub repository was created by the owner after the integration token lacked repository-creation permission. Source, manifests, measured progress JSONL, charts and selected small checkpoint artifacts are published there. Large replay buffers, downloaded dependencies, full run directories and all credentials stay out of Git. Charts are regenerated from measurements; they reach GitHub when committed and pushed, not through an implicit background sync.

## Next decisions to revisit after measurement

1. Expand the policy action space, particularly explicit attacks/cooperation, traps, king promotion and carry/throw interactions; fix king leading-edge digging as part of that versioned interface.
2. Increase search depth/simulations and actor throughput after profiling; compare network-only and search-enabled play with opponents whose state can be safely replayed or cloned.
3. Add global spatial context, temporal memory and carried/airborne status features; then test whether privileged inputs materially improve learning.
4. Use more training games and held-out maps/seeds before making strength claims; preserve the external panel as evaluation-only.
5. Only after useful results, design tournament-compatible observations and distillation/compression within actual bytecode and memory budgets.
