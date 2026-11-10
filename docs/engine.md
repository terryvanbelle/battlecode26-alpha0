# Official engine integration

`python scripts/setup_engine.py` downloads the official Battlecode 2026 Java
engine **1.2.6**, source and Javadoc JARs with fixed SHA-256 checksums. It installs
a hash-pinned Temurin Java 21 JDK under ignored `vendor/jdk` if needed, applies
`patches/local-research.patch`, and compiles the bridge plus four patched engine
classes under `build/java`. `JAVA_HOME` can select an existing Java 21 JDK.
The runtime classpath puts these classes before the unmodified engine JAR.
The GPL-3.0 upstream engine is fetched rather than committed; the patch and
adapter source are included for reproducibility.

## Local limits and reproducibility

- Learned policies call the engine's `RobotControllerImpl` directly; no Battlecode
  bytecode instrumentation, robot classloader memory accounting or tournament
  execution-time limits apply to network inference, search or macro execution.
- Frozen Java benchmark bots retain the official sandbox/classloader and
  `Clock.yield()` scheduling. Their per-turn budget is raised to **1,000,000,000
  bytecodes**, and the cumulative team execution-time limit is disabled. The
  instrumenter's integer counter remains finite; this is a practical raised
  budget, not a claim of infinite bytecodes. `-Dbc.alpha0.tournamentLimits=true`
  restores original benchmark bytecode and team-time limits.
- The bridge does not impose a project `-Xmx` or tournament memory cap. JVM
  ergonomic heap sizing, Python allocation and the host's 16 GiB cgroup limit
  still apply in this environment. The training runner retains infrastructure
  watchdogs: the smoke configuration allows 1,800 seconds per game, and the
  JSON adapter allows 120 seconds per response. A timeout is an infrastructure
  failure, not a game loss or training result. Configure these for larger hosts
  and search budgets; they are separate from Battlecode tournament limits.
- The official final random tie-break used `Math.random()`, outside game state.
  The patch uses the game's seeded `rand.nextDouble()` instead. The 50/50 rule
  remains, while snapshots and reseeded games become reproducible.
- Recording is replaced by a no-op `GameMaker.MatchMaker` sink. Physics never
  reads replay data; no `.bc26` visual replay is produced by this bridge. Raw
  episode actions, seeds, summaries and checkpoints are the reproducible records.

## Turn stepping and exact search states

The bridge follows the official `GameWorld.runRound()` lifecycle: beginning of
round, a snapshot of spawn-order IDs, each robot's beginning of turn, controller,
end of turn, termination handling, and end of round. New robots start on the next
round; killed robots are skipped. Cats execute their built-in engine behavior
inside `InternalRobot.processEndOfTurn()`.

For self-play, `StateCopier` deep-copies the mutable engine object graph,
preserving shared references and cycles (including robot/controller/world,
carried robots, mines, traps and ID generators). Both world RNG and the engine's
mutable static `InternalRobot.rand` are saved. Only precomputed immutable cat
BFS tables and the recording-free sink are shared. This implementation is tied
to the hash-pinned engine and Java 21 and deliberately rejects unsupported Java
runtime objects instead of silently shallow-copying them. It uses Java's
`sun.misc.Unsafe.allocateInstance`; changing engine or JDK versions requires
rerunning deterministic transition tests.

Snapshotting live benchmark sandboxes is unsupported: they contain executing
Java threads and arbitrary private bot state. Benchmark evaluation therefore
uses the learned policy without MCTS, against the actual frozen Java bot. This
measures the network policy, not policy-plus-search strength.

The Java instrumenter requires five module openings on Java 21:
`jdk.internal.misc`, `jdk.internal.math`, `jdk.internal.util`,
`jdk.internal.access` and `sun.security.action`, each passed as
`--add-opens=java.base/PACKAGE=ALL-UNNAMED`. The Python adapter supplies them.
Without these flags, common code such as instrumented `java.util.Random` fails
at class initialization; compiling a benchmark alone is insufficient validation.
The bridge monitors initialization and unexpected sandbox exits while a robot
is still alive. A failure produces `valid_game:false`, `benchmark_failures`, a
bounded diagnostic tail and a top-level `error`, causing the runner to exclude
the game from scores. A diagnostic-only sandbox flag records termination via
`RobotDeathException`, including `disintegrate()` before the scheduler removes
the robot. This distinguishes normal voluntary death from an exception or
invalid return. A legitimate `resign()` or physical death also remains a normal
game event. Returning from a bot's
`run()` method is treated as a failed benchmark rather than evidence of policy
strength. Induced initialization/return/module failures and legitimate resigns
are covered by `tests/test_benchmark_failures.py`.
Engine exception diagnostics and bot output have separate bounded tails, so
verbose bot chatter cannot hide a crash stack. Bot console mirroring to stdout
is disabled; stdout belongs exclusively to the JSONL protocol, including when
a bot prints text without a trailing newline.

## Initial 17-action abstraction

One learned action controls **one complete robot turn**:

| ID | Macro |
|---|---|
| 0 | Hold position |
| 1–8 | Move N, NE, E, SE, S, SW, W, NW; clear adjacent dirt if movement is blocked and digging is legal |
| 9–16 | Rat king builds at the legal location nearest the requested compass direction |

Each macro performs deterministic cheese pickup and transfer to a reachable
friendly king before and after its main action. It then opportunistically
attacks a legal sensed target, preferring enemy rats over cats, then lower HP,
then ID. Consequently it can backstab automatically. Movement/build macros turn
toward the requested direction when legal. All execution uses official
`can*`/action methods. A legal mask contains hold and currently executable macro
choices. Turns with hold as the only choice automatically execute it and advance;
they do not create meaningless policy samples.

This is **AlphaZero-style learning over a restricted action abstraction**. It is
not learning every Battlecode primitive from scratch. Trap placement, carrying,
throwing, voluntary king conversion, squeaks and shared-memory protocols are
not policy choices in this first version. The fixed attack rule also excludes
learning when to maintain cooperation. Expand this interface before drawing
conclusions about the game's attainable strength.

## JSONL protocol

Launch `vendor/jdk/bin/java -cp build/java:vendor/engine/battlecode26-java-1.2.6.jar
alpha0.EngineBridge`. Send one JSON object per stdin line; stdout is one JSON
response per line and errors/diagnostics go to stderr.

- `{"cmd":"reset","map":"tiny","seed":42,"maps_dir":"maps"}` starts self-play.
- Add `opponent_package`, `opponent_classes` and `learned_team` (0=A, 1=B) for
  policy-only benchmark evaluation.
- `{"cmd":"step","action":1}` applies a complete macro turn.
- `{"cmd":"snapshot"}` returns `{ "snapshot": id }`.
- `{"cmd":"restore","snapshot":id}` restores an exact state.
- `{"cmd":"release","snapshots":[1,2]}` frees stored states; release them after
  each real move to bound search memory.
- Reset/step can include `"snapshot":true` to return a state and snapshot together.
- `{"cmd":"replay","map":"tiny","seed":42,"actions":[0,1,9]}` deterministically
  replays a transcript using the engine without intermediate IPC observations.
- `{"cmd":"close"}` releases live benchmark sandboxes and exits.

State responses include map dimensions, flattened `y*width+x` terrain arrays,
all robot entities/footprints, active robot and team, legal action IDs, round,
bank/transferred cheese, cat damage, cooperation state and terminal winner.
Directions use the compass order above (north means increasing game y).
Teams are 0=A, 1=B, 2=neutral. Trap arrays use 0=none, 1=rat, 2=cat. These are
centralized omniscient training observations and exceed tournament sensing.
