# Research log

The current goal is to beat all twelve frozen opponents on **every one of the
74 available maps**. Schema 2 of [the gate](../configs/victory_gate.json) and
[the all-map protocol](all-maps-victory.md) supersede the earlier four-map
qualification scope. Routine evaluations may use small map subsets; final
success requires every opponent–map pair to pass both full-map stages. Entries
below retain their historical measurements and experimental scope.

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

### Winner-only prior selected for search continuation

Candidate 002 used the same six own-controller games and tensor-identical initial
weights as candidate 001. It selected 917 winning-team examples, disabled value
loss during its 256 initialization updates, and reset the final value layer to
zero. It completed **24/24** smoke cells with **20 wins and no errors**: both-seat
wins against ten opponents, both-seat losses to `bestsofar` and `basic51`. The
policy-example filter and value objective changed together; this comparison
supports the combined recipe, not a causal claim about either change alone.
All-teacher and winner-only comparisons use identical evaluation configuration;
their 600-second watchdog differs from the neutral campaign's 1,800 seconds,
with every game finishing well below either limit.

The next test initializes a **separate** search learner from this prior, with an
empty replay and new optimizer. It raises the root budget to 32 simulations while
searching 10% of baby turns and 25% of king turns, uses one actor alongside the
neutral control, and restores the common 1,800-second evaluation watchdog. The
full panel must be re-evaluated before claiming search-guided improvement. The
reserved confirmation suite remains untouched. Details and artifacts are in
[`progress/bootstrap-v3-winners`](../progress/bootstrap-v3-winners).

### Calibration on subsequent self-play games

A separate read-only audit evaluates each game's **pre-game checkpoint** against
its eventual outcome, before that model has learned from that game. The snapshot
includes 23 completed games: eight generated by checkpoint 0, eight by checkpoint
1 and seven by checkpoint 2. Equal-game win-probability Brier scores are
**0.2500, 0.2582 and 0.2882**, respectively. No predictions are saturated, but the
trained heads forecast worse than the neutral baseline on these future games;
checkpoint 2 is optimistic. Lower replay fitting loss has therefore not yet
translated into useful value prediction. These are small, changing self-play
populations, not a permanent final-model holdout or a controlled cross-generation
comparison. The [audit report](v2-value-calibration.md) records all source hashes,
label distributions and reliability plots. This supports keeping the warmstart
critic neutral initially and requiring future-game calibration before assuming
that deeper search has a good leaf evaluator.

### Calibration refresh: 2026-11-10 03:45 UTC

The 23-game measurements above remain the historical snapshot. The
[latest audit report](v2-value-calibration.md) now contains **29 completed games**,
with matching audit-script provenance and no excluded inputs. Checkpoints 0, 1
and 2 each have eight subsequent games; checkpoint 3 has five of eight. Their
equal-game Brier scores are **0.2500, 0.2582, 0.2874 and 0.2824**. No predictions
are saturated. The JSON and plot were regenerated from immutable pre-game
checkpoints; three planned games were pending at this snapshot.

Separating same-policy and league games explains the sharp label-prevalence
change in checkpoint 1's subsequent games. Its four same-policy games have a
**0.4839** mean terminal win label, Brier **0.2499** and BCE **0.6929**. Its four
league games were all current-checkpoint wins; because only that team's searched
positions are retained, their label mean is **1.0000**, Brier **0.2664** and BCE
**0.7260**. Equal game weights therefore produce the aggregate label mean
**0.7419**. Checkpoint 2's eight subsequent games are all same-policy, with label
mean **0.4927**, while its mean forecast is **0.6823**.

The encoder has no opponent identity or policy/search-regime input. The same
board observation can therefore be associated with different continuation
policies, and the mixture of retained labels changes with league outcomes.
These facts make mixture shift and omitted opponent conditions plausible
contributors; they do not prove overfitting or identify the cause of the forecast
errors. The audit records a future controlled comparison of pure current-policy
self-play and a fixed league mixture, including matched side retention and
separate future-game calibration. Opponent-conditioned value prediction is a
separate versioned experiment. No active training configuration or runtime was
changed for this investigation.

### Goal expanded to every available map

The user expanded the objective to all opponents on all available maps, then
clarified that routine evaluations can use small subsets and full-map checks
should be occasional. Gate `field-twelve-v2-all-maps-v2` now freezes all **74**
available map files and the existing **12** opponents. Its map-inventory
fingerprint is
`c93f589749c7ff4b922f8a42b8cd60368958a31910e20da1379fdd2a66b472b7`,
computed from sorted `{name, sha256}` entries. No opponent or difficult map is
removed to improve the result.

Each of **888 opponent–map pairs** must independently score at least 60%, with
at least 50% in each seat, in both stages: **3,552 standard games** using two
seeds per map, followed by **7,104 fresh-seed confirmation games** using four.
The same frozen checkpoint and evaluation policy must pass both. Scores cannot
be pooled across maps, opponents or stages. Both stages cover the complete
inventory; confirmation therefore means fresh seeds, not unseen maps.
`Stash`, `Hike`, `heart` and `ratsareants` remain excluded from gradient training
through initialization ancestry as an additional restraint. All benchmark
trajectories remain excluded from training.

The implementation is separate in [`alpha0.victory`](../alpha0/victory.py), with
`standard` and `confirmation` stages. One candidate is locked per confirmation
version; failure consumes its four-seed set. A subsequent independent attempt
needs a new version and four predeclared unused seeds on all maps. The earlier
192/384-game specification and smoke measurements remain historical evidence,
not a pass under the expanded goal.

Most evaluations remain inexpensive subset diagnostics. Automatic full
standard qualification requires a complete smoke panel scoring at least 50%
against every opponent, rather than triggering after every four weak training
iterations. A periodic research review may also request an occasional all-map
check. The winner-only initialization and its first two MCTS checkpoints each
won **20/24** smoke games, with both-seat losses to `bestsofar` and `basic51`.
They neither meet that scheduling screen nor demonstrate all-map victory or
improvement over the prior. The hourly continuation instructions were updated
to the expanded goal and this evaluation cadence. The user's current budget
caps each invocation at **three hours**; a full gate may span recoverable cycles.

The old neutral run stopped at checkpoint 4 with 32 completed self-play games.
A plotting endpoint-roundoff error was repaired: an all-loss Wilson interval
produced a tiny negative error-bar length. The graph code now also separates
smoke, old standard and all-map evidence by stage and configuration, instead of
pooling overlapping schedules. Raw outcomes are unchanged. The pre-migration
failure status is retained under `progress/campaign-v2/operational-history/`;
the neutral control is paused for research review while the stronger warmstart
is investigated. The temporary migration STOP was created by this maintenance
work, not by the user.

Warmstart checkpoint 2 has a separately verified recovery export with exact
model, optimizer, Torch/NumPy RNG and replay RNG restoration (16 games, 3,751
retained searched positions). Its 20/24 smoke score still matches initialization.
The new all-map scope report starts at **0/3,552 valid qualification games**;
none of the historical subset games is silently relabeled as full-gate evidence.

The neutral checkpoint-4 recovery snapshot was independently restored and
verified before publication: SHA-256
`b13fee70a33962fcf338c0c5d399fb31f1ae318dcaee7123da31f804c5d2c8a9`.
It preserves exact model/optimizer/RNG and the iteration-4 replay generation.
The [published pointer](../progress/campaign-v2/snapshot.json) uses the tested
`campaign-checkpoints` branch fallback because release-asset upload credentials
remain insufficient. No newer local run was overwritten.
