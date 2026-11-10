# Research log

## 2026-11-10: independent continuation and v2 baseline

The user authorized continued independent research until the learned agent beats
all twelve opponents. The predeclared empirical victory criterion is in
[RESEARCH_PLAN.md](RESEARCH_PLAN.md) and `configs/victory_gate.json`. No checkpoint
has passed it. The managed CPU environment is the execution host; `battlecode-dev`
has not been used or changed.

The first experiment revealed two concrete limitations: baby rats attempted to
attack a king's inaccessible center rather than its footprint, and the value head
collapsed to nearly -1 while the replay buffer was dominated by its latest long
game. V2 fixes the attack adapter, adds global observations and 40 macros, trains
raw win logits with BCE, and samples complete games uniformly. These are versioned
engineering changes, not evidence of learning. V1 remains reproducible.

The neutral v2 checkpoint completed **24/24** `tiny`, seed 0, both-seat benchmark
cells: **3 wins, 21 losses, zero unresolved errors**. Its SHA-256 is
`936b66b3435dc42ba028225c7732fc32d4d769378530e3d8e14b79b00a30b963`.
This is the baseline for subsequent v2 learning comparisons. Its lower score than
the v1 baseline is not a controlled architecture ablation: features, action schema
and initialization all changed.

A separate fixed-controller diagnostic completed **48/48** games without errors.
The assault script won **20/24** and the balanced script **10/24**. Assault beat
ten opponents in both seats but lost both games to `abcdezyxwv_bestsofar` and
`IvanGeffner_basic51`. These are scripted capability tests, **not learned-policy
results, victory evidence or training data**. Complete measurements are in
[`diagnostics/macro-v2`](../diagnostics/macro-v2). They show that the corrected
macro space contains useful behavior, while indiscriminately prioritizing carrying
and powered bites can hurt the simple rush.

The active next experiment is neutral-initialization v2 self-play under
`configs/campaign-v2.json`: eight complete games per iteration, eight-simulation
search on 15% of baby turns and 35% of king turns, and 48 optimizer batches. Only
searched decisions receive MCTS visit targets. Complete-game outcomes supply the
value labels. Historical learned checkpoints provide a quarter of self-play
opponents when available. No benchmark trajectory enters replay.

An optional [scripted self-play warmstart](bootstrap-proposal.md) is implemented
but **has not been activated in the production v2 run**. Its tiny validation
established only that a fresh student can fit teacher decisions from our own
completed game. If used later, it must start a separately labeled experiment,
reset optimizer/replay, preserve teacher lineage and compare against neutral v2.

Operational changes include identity-bound plans and per-cell evaluation records,
versioned replay generations, resumable deadlines, a process lock and STOP marker,
immutable GitHub release recovery assets, and an enabled hourly continuation with
the same managed cloud configuration. An active schedule is not proof of
uninterrupted compute. See [autonomy.md](autonomy.md).

At the next review, inspect full-panel smoke outcomes, value mean/MSE/saturation,
replay game diversity, game lengths and search coverage. If neutral sparse-search
training remains weak, test the explicitly labeled own-teacher warmstart and
larger useful search budgets; do not repeat a failed recipe indefinitely or
consume reserved confirmation maps merely to inspect progress.

### First v2 iteration and recovery verification

Eight complete self-play games generated **42,230 decisions**, including **6,927
searched decisions**; the reservoir retained **4,096 positions** across all eight
games. Training and game generation took 394 seconds. After 48 updates, mean
training-batch value BCE was **0.69254**, MSE **0.99879**, predicted value **-0.01933**
and saturation fraction **0%**. The value head is close to the uninformed baseline;
it has not demonstrated useful held-out calibration, but has not repeated the
initial run's saturation failure at this checkpoint.

Checkpoint 1 completed **24/24** smoke cells: **4 wins**, versus 3 for its neutral
baseline, with no errors. That one-game difference on a single map/seed is not a
strength claim. The campaign continued to its next self-play iteration.

The separately labeled all-teacher warmstart completed six own-controller games,
256 distillation updates, and all 24 benchmark cells: **12 wins, zero errors**.
This is prior initialization, not search-guided improvement. It remains weaker
than the fixed assault diagnostic and has not replaced the neutral v2 experiment.

GitHub release creation succeeded, but the asset-upload endpoint returned HTTP
401 with the current credentials. The recovery helper now uses an append-only
`campaign-checkpoints` branch as an authentication-specific fallback. A clean
remote download and isolated restore of the **trained checkpoint 1** successfully
loaded its optimizer/RNG state, eight replay games, 4,096 positions and all 48
baseline/checkpoint evaluation rows. Checkpoint SHA-256:
`37f989790ddcb0768dafdc8fa3d887ec9504689646c99f73d047865dd2d58c90`.
The active training checkout was untouched. Ordinary Git progress can be ahead
or behind the archive: recovery merges evidence safely and requests review if
training metrics refer to a later checkpoint than the available model.
