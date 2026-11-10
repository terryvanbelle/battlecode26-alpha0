# Design decisions

## Versioned experiments and evidence

This is AlphaZero-style policy iteration over the official Battlecode 2026 engine, with a deliberately restricted, hand-authored action interface. It is not a faithful implementation of full-action AlphaZero: observations are privileged, action macros contain tactical/navigation heuristics, the game is multi-agent and stochastic, and the CPU campaign searches only some decisions. Each policy controls all rats on its team with shared weights.

| Version | Action interface | Observations | Model | Value objective |
| --- | --- | --- | --- | --- |
| Initial v1 | `macro_v1`, 17 actions | Local 24×17×17 + 32 scalars | 49,242 parameters in the measured smoke run; 103,290 in the unvalidated larger configuration | Tanh value + terminal-outcome MSE |
| Current v2 | `macro_v2`, 40 actions | Local 33×17×17 + global 33×60×60 + 64 scalars | Dual-scale residual network, 87,973 parameters | Raw win logit + terminal-outcome BCEWithLogits |

The first experiment completed **six self-play games, three iterations and 96 optimizer updates**. It retained 43,779 positions cumulatively, with only 10,000 in each final replay snapshot. All four checkpoints completed the final 12-bot smoke panel: **96 evaluation games**. Both untrained and final checkpoints won **5/24 games (20.8%)**. There is no demonstrated aggregate strength improvement. The final recorded value MSE was **2.74076**, worse than the MSE of 1 from a constant-zero predictor.

An independent replay audit found the final value predictions near −0.998, with all sampled predictions saturated beyond magnitude 0.95 even though the last replay contained 58.75% positive labels. Between 90.9% and 100% of each replay came from its last long game. This motivates a new value objective and game-balanced replay; it does not prove either change will yield a stronger policy. The audit, hashes and limitations are in [the research plan](docs/RESEARCH_PLAN.md). Original v1 checkpoints, action transcripts and results remain under `artifacts/initial/` and `progress/`; no v2 data silently replaces that history.

V1 and v2 action meanings, feature shapes, model classes and run directories are separate. V2 checkpoints record `model_version="dual_scale_v2"`, `feature_version=2`, `action_version="macro_v2"` and their network configuration. V1 loaders remain supported. Replay tensors and weights cannot be reinterpreted across versions. An archived 4,610-action v1 game still reproduces its prior terminal physical state after the versioned bridge changes.

## V2 actions and fixed controller behavior

The 40 choices are complete robot turns. The policy chooses when to invoke a macro; the macro implements its legal primitive actions. Exact ordered IDs, priorities and limits are in [the macro-v2 specification](docs/macro-v2.md).

| IDs | Choices |
| --- | --- |
| 0 | Hold, preserving cooperation when possible |
| 1–8 | Compass movement and digging |
| 9–16 | Compass king-spawn directions |
| 17–27 | Gather, return, claim a mine, guard a king, assault a king, hunt a baby, hunt cats, retreat, relocate a king, spawn for economy, spawn for assault |
| 28–31 | Place rat/cat traps, clear an own trap, form a king |
| 32–36 | Carry an enemy, rescue an ally, throw for damage, throw for transport, drop safely |
| 37–39 | Powered bite, clear a route, cooperating economy |

Free legal cheese pickup occurs before and after the primary action. Economic/support macros may transfer cheese; combat/carry/throw macros preserve action cooldown for their primary purpose. Goal macros can remain selectable when a target exists but movement is temporarily blocked or on cooldown. The mask therefore represents available macro intentions, not a promise that every choice immediately changes position. Each primitive still checks the official controller's legality.

Navigation uses fixed Dijkstra route costs over robot-center positions, with footprint walls, dirt, congestion, traps and contextual threat costs. Retreat, target selection, trap placement, king clustering and throw-ray scoring are fixed heuristics. These are substantive inductive biases, authored from rules and APIs rather than copied from benchmark bots. There are still no learned primitive action orderings, communication actions, or recurrent private robot programs.

V2 also corrects limitations in the v1 controller: baby bites and cheese transfers target legal king footprint tiles instead of unreachable centers; cat attacks use its footprint; king digging uses its leading edge; redundant turning is avoided; and combat can reserve cooldown instead of automatically transferring first. These are **controller improvements, not learned gains**. Compare trained v2 checkpoints with a fresh untrained v2 baseline under identical engine, macro and evaluation settings. A cross-version score jump alone cannot establish learning. Any later macro or engine change must receive a new runtime identity and a new comparable baseline.

## V2 observations and network topology

Both spatial tensors share **33 channels**: on-map mask, walls, dirt, mines, logarithmically scaled ground cheese, separate allied/enemy rat and cat traps, the acting robot, ground babies/kings/cats and normalized HP, allied/enemy carried cheese, facing, three cooldowns, allied/enemy carried and airborne robots, and coordinates relative to the actor. Carried/airborne status has dedicated planes and does not overwrite a ground carrier's health or direction.

The local view is actor-centered at **17×17**. The global view retains the full official map, padded to **60×60** with an explicit validity mask. Coordinates keep the engine's compass orientation; NORTH is increasing y. Ally/enemy channels are relative to the acting team. The **64 scalars** describe actor state, map/round/cooperation, both teams' economy and armies, cat statistics, legal choices, and presence/direction/distance to the nearest mine, loose cheese, allied king, enemy king, cat and enemy baby. Nearest-target inputs are Euclidean geometry, not a predicted opponent strategy. Exact ordered names and normalizers are in [`alpha0/features_v2.py`](alpha0/features_v2.py).

These observations are omniscient research inputs, including opposing traps and entities beyond official vision. They are not tournament-compatible and are not a complete Markov description: for example, not every engine timer or historical rule flag is encoded. Search still branches the full physical state. The model has no recurrent memory. Tournament-compatible observation design and distillation are deferred until useful strength is demonstrated.

The local trunk has a 3×3 convolution to 24 channels and two residual blocks, each containing two 3×3 convolutions. The global trunk uses two stride-2 convolutions to 16 channels, reducing 60×60 to 15×15, followed by two residual blocks. GroupNorm and SiLU support batch-one inference without moving batch statistics. Fusion combines local actor-center/valid-mean/valid-max features, global actor/valid-mean/valid-max features, a four-channel coarse 4×4 global projection, and a 32-unit scalar embedding. A 128-unit fused layer feeds 40 policy logits and a 64-unit value head. Total: **87,973 parameters**.

Illegal actions are masked before softmax and policy loss. The last value linear layer is initialized to zero; the initial value is exactly neutral. The value head returns an **unbounded win logit** during training. Inference converts it to `2*sigmoid(logit)-1`, preserving the PUCT evaluator's [−1,1] contract. A one-thread CPU smoke measurement on `DefaultSmall` took approximately **1.50 ms for inference and 0.35 ms for feature encoding**; this is not end-to-end game throughput, which also includes engine execution, serialization, navigation and snapshots.

## Search and sparse self-play targets

PUCT uses network priors and leaf values with exploration constant 1.5. Searched self-play roots receive Dirichlet noise with alpha 0.3 and mixture 0.25. Policy targets are normalized visit counts. Temperature changes actual action sampling but never rewrites those visit targets. Search cutoffs bootstrap the network; only real terminal game results provide outcome labels.

Battlecode follows robot spawn order, so multiple consecutive turns can belong to one team. Edge values are stored in absolute team-A coordinates and converted for the acting team during selection. Search never blindly negates at each ply. One ply means a robot decision, **not a complete round**; depth 12 can therefore remain a short horizon in a populated game. Cats and forced turns advance through the engine.

The current configuration is [`configs/campaign-v2.json`](configs/campaign-v2.json), implemented by [`alpha0/learn_v2.py`](alpha0/learn_v2.py):

| Setting | Current value |
| --- | --- |
| Training maps | `tiny`, `DefaultSmall`, `closeup`, `knifefight` |
| Parallel actors / games per iteration | 2 / 8 |
| Search per selected decision | 8 simulations, maximum depth 12 |
| Probability of searching a candidate-policy decision | 35% for kings, 15% for babies; the first eligible decision is searched |
| Current-policy action temperature | 1 through round 80; 0.35 afterward |
| Historical-opponent games | 25% probability when earlier checkpoints exist; candidate seat alternates |
| Historical opponent sampling temperature | 0.3 |
| Updates per iteration | 48 minibatches of 32 |
| Optimizer | AdamW, learning rate 0.0003, weight decay 0.0001, gradient norm cap 1 |

On decisions without search, the relevant checkpoint's policy is sampled with the configured temperature. Those decisions advance the real game but **are not stored as imitation targets**. Only searched candidate-policy positions enter replay; in current-versus-current games both sides are eligible. In historical-opponent games the other side uses an earlier learned checkpoint, and PUCT uses the appropriate side's checkpoint for priors and leaf evaluation. No frozen benchmark enters the self-play league.

Sparse search trades target density for complete-game throughput. It is a deliberate departure from searching every AlphaZero move, and eight simulations may still cover few actions among 40. Its benefit requires measurement. Raising search budgets, adding batched frontier inference, or measuring search-enabled evaluation are future experiments rather than assumed improvements.

## Terminal objective, game-balanced replay and recovery

For a completed game, `z` is +1 for the winning team, −1 for the losing team, or 0 only for an actual draw. Each stored position gets its own player's `z`. The v2 objective is visit-distribution cross entropy plus `BCEWithLogits(value_logit, (z+1)/2)`. There is no reward shaping or discounted surrogate score. MSE after inference conversion, BCE, mean value, saturation fraction, entropy and losses are reported separately; the v1 MSE curve and v2 BCE curve must not be interpreted as the same quantity.

BCE fixes a specific optimization defect: wrong-sign logits of magnitude 30 retain finite corrective gradients, while tanh followed by MSE has effectively zero gradient there. Regression tests establish that behavior and the zero-valued initialization. They do **not** establish calibrated values, resistance to forgetting or competitive play. Game diversity and independent episode-level validation remain necessary.

[`alpha0/replay_v2.py`](alpha0/replay_v2.py) stores each completed self-play game separately. A uniform reservoir retains at most **512 searched positions per game**. Minibatches select a retained game uniformly, then a position uniformly within that game, so a long game cannot dominate solely through its length. Limits are **6,000 positions and 32 games**, with whole-game FIFO eviction; if every game retains 512 positions, the position limit binds before 32 games. Inputs use float16 storage and float32 computation. Benchmark/evaluation source markers and incomplete games are rejected, and saved policy masks and terminal-label perspectives are validated.

Each completed actor saves its action transcript, terminal state, metadata and hashed numeric-array shard. The learner can reuse completed jobs after interruption. Immutable replay game files are published before an atomic index; model/optimizer/RNG/configuration checkpoints follow the replay and completed optimizer iteration. Restarting requires the same configuration. Different configurations belong in separate versioned experiments, and partial games are errors rather than fabricated draws. Hashes and generation checkpoints keep the evidence attributable.

## Own-teacher initialization experiments

The neutral v2 campaign remains a control. A separate `bootstrap-v3` experiment uses the **same v2 network and macro schema**, initialized from games played only by our own simple assault, forage and guard controllers. This is an explicit expert prior, not tabula-rasa AlphaZero. No downloaded bot code or benchmark trajectory supplies examples. Six full games, three development maps and swapped controller seats supply terminal outcomes and teacher decisions; confirmation maps remain excluded. Source snapshots, shards, selections and checkpoints are published for reproduction.

Candidate 001 fitted both winning and losing teachers with policy CE plus outcome BCE for 256 updates and scored 12/24 in the smoke panel. Candidate 002 selected the 917 winning-team positions from the same 1,536 retained examples, fitted policy only for 256 updates, and explicitly reset the value output layer to zero. It scored 20/24, winning both seats against ten opponents and losing both to `bestsofar` and `basic51`. This is a comparison of two initialization recipes: both example selection and shared-trunk value gradients changed, so it does not isolate a causal effect of winner filtering. Neither result establishes broad strength or meets the victory gate.

The winner-only prior is selected for a separate MCTS continuation, with a fresh optimizer and **empty search-target replay**. Teacher labels never masquerade as MCTS visit targets. Initialization artifact hashes and teacher lineage remain in the run configuration. This continuation uses one actor, 32 simulations per searched root, 10% baby-turn and 25% king-turn search, eight complete games and 48 gradient batches per iteration. The critic starts neutral because a value head fitted to six scripted games saturated and has no demonstrated calibration on the learned policy's future games. The baseline and resulting search-trained checkpoint receive separate complete evaluations. These choices and results are expanded in [the bootstrap proposal](docs/bootstrap-proposal.md), [winner-prior report](progress/bootstrap-v3-winners/README.md), and `configs/warmstart-v3.json`.

## Development evaluation and the all-opponent victory gate

The frozen panel contains 12 bots spanning different authors and historical skill. Both seats are played for each map/seed. Evaluation currently uses the **greedy network with no root noise and no search**, because live Java opponent threads cannot be cloned safely. It is not a measurement of the full search-enhanced self-play player. Benchmark code, trajectories, observations and outcomes never enter gradient replay or the historical-checkpoint league; development results can inform research decisions.

The original 24-game smoke panel overlaps training and has just one paired map/seed cell per opponent. It checks integration and exposes weaknesses; it cannot establish generalization. V2 checkpoint comparisons require the same checkpoint family, action/observation schema, engine/patch/bridge identity, opponent hashes and evaluation settings. Errors and timeouts remain explicit and never count as candidate wins.

The machine-readable success definition is [`configs/victory_gate.json`](configs/victory_gate.json), explained in [the research plan](docs/RESEARCH_PLAN.md):

| Stage | Schedule | Required against every opponent |
| --- | --- | --- |
| Development qualification | `tiny`, `DefaultSmall`, `cheesefarm`, `wallsofparadis`; seeds 0 and 2026; both seats; **192 games** total | Score at least 60%, and at least 50% in each seat: at least 10/16 decisive wins |
| Fresh-map confirmation | `Stash`, `Hike`, `heart`, `ratsareants`; seeds 90101, 90107, 90121, 90133; both seats; **384 games** total | Score at least 60%, and at least 50% in each seat: at least 20/32 decisive wins |

The **same frozen checkpoint and evaluation policy** must pass both stages against **all twelve** opponents. Aggregate panel score cannot substitute for beating one difficult opponent. Every expected cell must have a valid result, with no unresolved errors. Actual draws earn half credit. These are empirical qualification thresholds, not proof of superiority on all maps or simultaneous statistical significance.

The confirmation maps are excluded from training. Before its first holdout game, lock candidate, panel, policy and runtime identities in a persistent ledger. Only that candidate may finish missing or failed infrastructure cells; successful cells cannot be rerun to choose favorable outcomes. A failed candidate consumes this holdout version. Later independent confirmation needs a newly declared disjoint holdout and version, rather than repeated checkpoint selection on the same data. Report paired map-seed bootstrap intervals and every seat's coverage; pairs, not robot turns or independently resampled seats, are the uncertainty unit. A configured gate is not evidence that it has passed.

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

The initial `configs/smoke.json` run remains a historical integration experiment: two games per iteration, four simulations at every decision, depth eight, 32 gradient steps, learning rate 0.001, and a 10,000-position FIFO replay. Its larger v1 `configs/train.json` configuration was an unvalidated starting point. The v2 campaign above is a new experiment addressing observed controller, observation and value-learning failures; success must be established from its own baseline and evaluations.

The executor has a four-core CPU quota and 16 GiB memory limit, with no detected GPU. V2 uses two independent actor processes and one PyTorch thread per process. Training follows completed iteration games. Separate evaluation may run concurrently within measured resource capacity. Batched inference is available at the model API, but shared actor inference batching is not implemented. The benchmark/campaign runners retain finite timeouts and decision caps; these are local watchdogs, not tournament bytecode constraints.

`battlecode-dev` is **not used**. It is historically an eight-vCPU/31GB machine with another active project, and this executor lacks its SSH/GCloud credentials. Nothing here starts or stops that VM's services or submits to the Battlecode service. No tournament-sized JVM heap cap is set; host limits and JVM ergonomic sizing remain.

Campaign work is organized into finite recoverable cycles: at most four training iterations of eight games per configured cycle, with an explicit wall-time budget. The launcher defaults to three hours; the research gate records four hours as a cycle ceiling. A stop sentinel prevents new scheduling and is never silently deleted. Already running games have their own watchdogs, so a wall-time scheduling boundary is not a hard instantaneous process-kill deadline. Completed game shards, plans, replay and model checkpoints survive between cycles. Each cycle should publish measured failure modes and the next hypothesis before continuing; operational cycle limits are not a promise of victory or an overall cap on the authorized research.

The public GitHub repository was created by the owner after the integration token lacked repository-creation permission. Source, manifests, measured JSONL, graphs and selected compact checkpoint artifacts are published there. Dependencies and live run directories stay out of the main branch; credentials are never archived. Resumable checkpoint/replay archives prefer release assets, but the current upload endpoint rejects the integration credentials. An append-only `campaign-checkpoints` branch provides a tested fallback with a 90 MiB per-archive cap. New archives retain the latest replay generation, model history and pending episode shards, omitting redundant older replay data without deleting local files or previous backups. Recovery verifies hashes and merges evidence; metrics ahead of the restored model require review rather than silently overwriting an iteration. Plots must label model/action versions, losses, evaluation coverage and invalid attempts. Files reach the main branch when committed and pushed, not merely when a local run refreshes a graph.

## Decisions to revisit after measurement

1. Evaluate whether v2 learns beyond its untrained macro-controller baseline; separate heuristic/controller gains from optimizer gains.
2. Audit held-out self-play episode calibration, replay diversity and forgetting; a lower replay loss alone is insufficient.
3. Compare 8/16/32 simulations, searched-decision fractions and completed-game throughput under a fixed development protocol. Measure depth in rounds as well as robot decisions.
4. Add learned-champion promotion and search-enabled comparisons only with explicit reproducible opponent semantics. An approximate neural model of a Java opponent is not its exact private program state.
5. Keep the 12-bot panel evaluation-only and reserve confirmation maps until one frozen candidate qualifies. Preserve failures rather than tuning the holdout retrospectively.
6. After useful strength is demonstrated, design tournament-compatible sensing, memory and distillation under actual submission budgets.
