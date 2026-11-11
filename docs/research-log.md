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

### Warmstart plateau diagnosis and next bounded experiment

The [recorded target audit and counterfactual probe](warmstart-plateau.md) completed without changing training state. Across 16 own self-play games, search visit argmax matched its source policy on 99.20%/98.41% of sampled states. Eight reconstructed tactical roots produced 48 completed terminal continuations; 48 searches spanning depth/budget and zero-critic controls kept the original preferred action. A retreat improved both sampled continuations at one king root and worsened another baby root, supporting context-dependent action ranking rather than a blanket heuristic. These are diagnostic observations, not benchmark gains.

The approved next review cycle is the precommitted `terminal-root-ranking-v4` mechanism test in that report: new recorded roots, all legal first actions, separate fixed selection/validation seeds, one actor and a two-hour resumable limit. It must meet its declared research decision rule before any subsequent rollout-guided SGD pilot. The recurring continuation now prioritizes implementing and running that versioned test; it does not blindly repeat the unchanged target-3 warmstart. Both existing runs and the original champion remain preserved. The maintenance-owned gate migration STOP was removed only after migration checks passed; no user stop marker was removed.

### Terminal-root-ranking-v4 completed; policy-transfer pilot selected

The [versioned mechanism test](root-ranking-v4.md) was implemented and committed
as `fd84126` before any scored continuation. Thirty-nine runner/renderer tests
and 37 subtests passed, including process-tree cleanup, STOP/deadline handling,
immutable recovery, exact action/seed coverage and rejection of premature passes.
An independent official-engine replay matched all eight frozen root states,
prefixes, actors and legal-action lists before scoring.

One actor completed **374 selection continuations plus 44 unique validation
continuations**, all valid, with zero errors, retries or incomplete attempts.
The run took **2,752.86 seconds** (45 minutes 53 seconds), within its two-hour
budget. All legal root actions received both fixed selection seeds. Five roots
selected the same action as PUCT and shared their four validation games; no
shared game was counted twice as independent evidence.

The exact equal-root paired validation difference was **7/32 = 0.21875**,
with **two improving roots** and worst difference **−0.25**. Deltas in frozen
root order were `[0, 1, -0.25, 0, 1, 0, 0, 0]`, so the precommitted research
rule passed. At `tiny` king round 8, build-west beat spawn-assault 1.00 to 0.00;
at `closeup` baby round 49, southeast beat assault-king 1.00 to 0.00. The same
southeast choice lost 0.00 to 0.25 at `DefaultSmall` baby round 20. That adverse
result is retained and argues against a universal action rule.

The [independent final audit](../progress/root-ranking-v4/independent-final-audit.json)
held all three exclusion locks while checking quiescence, every cell identity
and terminal proof, exact selection tie-breaking, separate validation seeds,
chronology, source/checkpoint/runtime hashes and rational decision arithmetic.
The audit SHA-256 is
`b69616abb0bb802e386c7f5b00d5b7a061f188f4dccd8e7b5d1c09970bce1394`.
Its archived standalone source was rerun successfully. Original chronology
observations are hash-bound and preserved, so restoration does not mistake Git
checkout modification times for experiment times. [Graphs, immutable cells,
source snapshots and recovery records](../progress/root-ranking-v4/README.md)
are tracked in the main branch; the unchanged cp2 dependency remains in its
previously verified separate warmstart archive.

This is evidence about eight selected own-self-play roots under a frozen
continuation policy. It does not isolate action coverage from terminal horizon,
establish compute efficiency, or show general playing strength. No weights
changed and no benchmark games occurred. The prior and cp2 remain **20/24** in
the existing smoke panel; neither all-map stage has passed. The separately
published [benchmark terminal audit](warmstart-terminal-failures.md) found
identical failure endpoints across cp0/1/2 and many captured surviving babies.
Those descriptive evaluation states are not training examples or causal proof.

The recorded research decision is to implement
[terminal-target-head-v5](terminal-target-head-v5.md) in the next bounded cycle.
It preregisters 32 fresh balanced training roots, exhaustive two-seed terminal
targets, a matched PUCT selected-action control, and identical conservative
updates to only the existing 5,160-parameter policy output layer. Sixteen fresh
validation roots must exclude exact training-root and anchor input duplicates.
Fresh complete all-12 smoke panels compare cp2 with both learned arms. No v4
root, continuation or validation outcome enters fitting, and no benchmark data
enter gradients. The original architecture, critic, champion, controls and
888-pair all-map gate remain fixed. This protocol is selected for implementation;
it has not yet been trained or evaluated, and no automatic promotion is implied.


### Terminal-target-head-v5 coverage review (2026-11-10)

The preregistered pilot was implemented and committed as `aadc029` before real
source games. The integrated checks passed 78 tests and 32 subtests. Its bounded
invocation completed **20 own-self-play games** (16 initial plus four prescribed
reserves) in **188.42 seconds**, with zero failed or incomplete source attempts
and unchanged cp2/runtime. It froze **24 of 32 required training roots**; all
eight missing slots were on `closeup`. The coverage barrier correctly ended
the invocation as `review_required`: no target continuation, SGD, validation
or benchmark game ran. Missing roots were not discarded from the denominator.

[Coverage graphs and exact manifests](../progress/terminal-target-head-v5/README.md)
retain every reserve assignment. The separate [verified recovery archive](../artifacts/terminal-target-head-v5/README.md)
contains all completed source transcripts, lossless cached inputs and the unchanged
cp2 dependency. No candidate was produced and no benchmark improvement is claimed.
This is a pre-target coverage failure, requiring an explicit versioned source-policy
review before further learning; repeating the same seed schedule cannot fill the
frozen missing slots. V4 and both earlier learning campaigns remain unchanged.


The independent coverage audit verified all 20 terminal worlds and transcripts,
684 lossless cached positions and their canonical input hashes, current model/
runtime/map hashes, and the exact reserve allocation while holding all exclusion
locks. [Passive diagnosis](terminal-v5-coverage.md) found an assault-intent loop
behind impassable king geometry. Seven sources never created a baby; the eighth
did produce combat states but only for the opposite role/team parity from its
assigned slots. Retaining those slots as unavailable was correct.

A separately versioned [terminal-target-head-v6](terminal-target-head-v6.md) is
selected before any new source games: source temperature 1.0 symmetrically on
all maps/teams, continuation temperature 0.35 unchanged. The full source/reserve
seed inventory is retained for a paired coverage comparison; every data artifact
is generated anew in v6. Sampled stalled inputs assign assault mean probability
99.9775% at T=.35 versus 77.2487% at T=1. This is a coverage hypothesis, not proof
that the planned roots will exist. Both matched arms still require exactly
32 training roots, 512 anchors, 16 input-disjoint validation roots, the 1,600-cell
ceiling and 72 complete fresh smoke games. All fitting and decision rules stay
fixed; another shortfall returns to review without SGD. V5 source, config and
preregistered protocol remain byte-identical.

The v6 implementation passed **92 tests and 32 subtests** before collection,
including separate source/continuation temperature call sites, immutable v5
hashes, the complete paired seed inventory, all four locks/STOP markers, exact
fit recovery and process-tree deadline cleanup. An operational-only rendering
fix regenerates final graphs; rendering failure still publishes durable final
status. The independent reviewer found no change to optimizer math or frozen
engine, network, inference and PUCT behavior.


During v6 collection, an operational check reproduced a managed-sandbox issue:
`/proc/PID/cwd` was denied for live actors whose command lines remained readable.
The pinned runner's scan could falsely return no actors; actual process commands
and all four flocks confirmed the existing run, and no duplicate was started.
The separate, unpinned v6 recovery helper now blocks matching processes when
cwd is unreadable; its 17 offline tests pass, including a PermissionError
regression. No learner, collector, inference, engine or pinned runtime changed.
The operator instructions require command/lock inspection and the corrected
external scan; future supervisors should adopt that behavior.


### V6 completed and independently audited; fitting diagnostic selected

[Terminal-target-head-v6](terminal-v6-results.md) completed in 77.02 minutes with
28 source games, 32 training roots, 512 anchors, 16 input-disjoint validation roots,
1,458 valid target continuations,64 physically shared validation continuations
and 72 complete fresh benchmark games. Both heads received 128 matched AdamW
updates; only 5,160 policy parameters changed. No error, retry or incomplete
attempt occurred. The independent final audit exactly reproduced both model
and optimizer states and verified all terminal evidence, hashes and chronology.
Its SHA is `ab8f824a645432916417e8142d61c7cd13ec5e69561b96869115ff95288dd564`.

All three policies won 20/24 with the same four losses to bestsofar/basic51.
All 16 validation actions were identical; mean paired advantage and number of
improved roots were both 0. The research expansion rule FAILED; no model was
promoted and neither all-map victory stage passed. Terminal labels differed
from PUCT on 9/32 training roots, but neither head changed a root greedy action.

[Training-only diagnosis](v6-training-underfit.md) found increased target
probabilities without argmax crossings, full row rank of the 32-root feature
matrix, and strong gradient opposition among labels and from anchor KL. The
selected next experiment is [cached-head-constraint-v7](cached-head-constraint-v7.md):
two fresh-cp2, fixed-data head fits, KL 1 versus 0, a shared deterministic 2,048-update
stream and fixed 128/512/2,048 snapshots. The 128-step KL=1 result must exactly
reproduce v6 before interpretation. It is a 15-minute, training-only mechanism
diagnostic: no new games, held validation or benchmark tuning, and no automatic
promotion. All v6 artifacts remain preserved in their separate recovery export.


## 2026-11-10: v7 fitting-budget diagnostic completed

Preregistration and implementation were published at `fe407fb` before real SGD.
The fixed two-arm diagnostic completed in 20.99 seconds, with all six snapshots
and all recovery generations retained. KL=1 at 128 updates reproduced v6 exactly.
Both arms reached all 32 selected training actions only at the fixed 2,048
snapshot, giving the preregistered budget classification. Anchoring limited
final changed anchor actions to 3/512 versus 73/512 for KL=0. No validation,
benchmark or engine games were run, and no policy was promoted.

[Results and resource measurements](cached-head-v7-results.md) separate fitting
from playing strength. Independent audit and separate verified recovery are
being finalized before selecting the next real-game transfer protocol.


## User amendment: broader development evaluation and improvement

The user replaced all-opponent/all-map success with measured improvement. New
suite development-eight-v3 uses eight retained mid/strong opponents and eight
maps (256 games per checkpoint). Original twelve-bot history is preserved.
Cp2 needs a fresh full baseline; no tiny-only score is reused. The new rule and
paired map bootstrap are documented in development-benchmarks-v3.md. The README
now leads with the opponent-by-map W/L/D dashboard, and evaluation refreshes it.
The stopped all-map zdb run remains stopped: 0W/60L/0D, incomplete, no errors.


## Broader baseline running; V8 implementation review resumed

The new cp2 baseline is actively collecting the 256-game revised panel under
all predecessor locks. An 80-game partial W/L/D dashboard was published at
`ec1c0e1`; partial coverage is explicit. No complete-panel improvement is claimed.
The V8 preflight suite initially found four failures (61 tests passed): an
undefined plotting variable and unimplemented diagnostic metrics. These are
being repaired in the unlaunched draft. The draft still needs its old 12-bot
smoke evaluator and decision rule migrated to the new development protocol
before it can be frozen or launched. No active evaluation runtime was changed.

## Broader baseline crash review and recovery publication

The v3 baseline exhausted its schedule with 252 valid games (78W/174L/0D)
and four unscored collegejunior runtime errors, all on wallsofparadis. Two
independent fresh reruns reproduced the same internal BugSim assertion. This
is not a valid complete baseline and cannot support an improvement claim.
All eight maps remain required. Rating-matched replacement qualification is
recorded in diagnostics/collegejunior-wallsofparadis; candidate selection uses
historical ratings and crash validity, not learner wins. Icarus already failed
with repeated runtime errors; YokoOno is preregistered as the next candidate.

V7 recovery was published and independently restored into an isolated directory;
all 573 restored files and 264 checkpoint payloads verified. Its diagnostic heads
remain excluded from benchmark evaluation. The unfinished V8 draft passed 65
tests plus 41 subtests after the earlier metric/plot fixes, but its old smoke
protocol still requires migration before freezing or launching. No V8 SGD or
games have occurred. Version-selectable development reporting/evaluation keeps
historical v3 evidence intact when a qualified replacement panel is adopted.

## User priority: tournament conversion at 70% overall wins

Recorded configs/research-roadmap.json and docs/tournament-transition.md. At least
180 wins in a complete valid 256-game current panel triggers a focus switch to
Java-only tournament play under all official limits and APIs. This does not
reintroduce universal opponent/map wins. The current displayed original-panel
rate is 78/252 valid games (30.95%), with four crashes excluded; it is not a
complete valid milestone panel. The qualified replacement still needs a fresh
full baseline.

## Revised v4 panel frozen and baseline launch

YokoOno completed 32/32 qualification games with zero errors. Freeze v4 replacing
only collegejunior; retain all maps, seeds and success criteria. Start fresh cp2
256-game baseline under shared locks and a three-hour bound. Qualification games
are not baseline games. No improvement claim is available yet.

## V8 migration tested while the revised baseline runs

V8 now schedules 768 fresh development games (cp2 plus both fixed heads),
pins the v4 panel and improvement rule, and uses the approved score-gain,
paired-map bootstrap and zdb-nonregression criterion. First-action validation
remains a separate mechanism diagnostic. No old tiny-only result substitutes
for broad-panel evidence. All seven locks/STOPs are respected.

The full V8 suite passed 82 tests and 41 subtests, including synthetic complete
768-cell comparisons, missing-data rejection, zdb regression veto and isolated
2048-update model/optimizer/RNG recovery round trips. These are implementation
checks, not real game improvements. V8 has not yet generated games or gradients.
A bounded baseline-to-V8 handoff checks complete valid terminal evidence, runtime
identity, STOPs and quiescence before starting the preregistered three-hour cycle.
It stops for review on invalid baseline evidence or existing V8 state, and shifts
to tournament-compatibility review if baseline wins already reach 70%.

## Revised baseline completed; V8 handoff executed

V4 cp2 completed all 256 games: 83W/173L/0D (32.421875%), zero errors.
A separate read-only audit verified every terminal cell and frozen identities.
The updated README now shows v4 coverage, opponent/map W/L/D and combined
per-bot/overall history. This is a new baseline, not improvement over v3.

The bounded coordinator automatically started V8 after validation and lock
checks. V8 froze 32 training roots, 512 anchors, 16 input-disjoint validation
roots and its PUCT comparators, then began the fixed 1,508-cell selection panel.
At this publication review it was healthy in collect_selection with three
workers; no SGD or V8 benchmark result was available. No pinned actor runtime
was edited. Final counts and recovery must be reviewed after its bounded cycle.

## Independent V8 input audit during active collection

A separate read-only auditor, without importing the collector/learner hashing
helpers, reconstructed all 560 cached model inputs (32 training roots, 512
anchors, 16 validation roots). It verified source/cache checksums, legal masks,
seed formulas, balanced role/team slots and matching-parity reserves. The
544 distinct current gradient inputs plus 560 historical inputs form an
1,104-input exclusion union; all 16 validation inputs are disjoint. All 30
referenced source games are terminal and valid. The fixed selection budget is
1,508 continuations. See progress/terminal-target-head-v8/reviews/input-audit-20261110.json.

This is an input-integrity audit, not a final experiment audit or stronger-play
claim. Completed source caches remain in the live run and will enter the separate
verified recovery export after quiescence. V8 remains healthy in collection with
its pinned identity unchanged; no model or experiment settings were changed.

## V8 fits and first-action validation complete; broad benchmark active

All 1,508 planned selection games completed validly. Both 2,048-update fixed
heads fit all 32 targets. The terminal head changes 14/32 training actions and
7/512 anchor actions; PUCT changes 0/32 and 1/512. A separate auditor rebuilt
the deterministic batch stream, checked optimizer steps and byte-identical
frozen tensors, and recomputed these actions from cached hidden features.

All 64 physically shared first-action validation continuations are complete.
Mean terminal-head advantage over cp2 and PUCT is zero; its mechanism criterion
fails. The precommitted full 768-game benchmark remains running and cannot yet
establish a playing-strength result. At the published review 190 games were
valid, all fresh cp2 control games, with no errors. No outcome-dependent change
to the candidate, training settings or benchmark schedule was made.

New nonpinned reporting helpers derive dashboard views from canonical immutable
benchmark attempts and plot saved fitting histories. They do not launch games
or train. Final export/verification remains deferred until actor quiescence.

## Fresh cp2 control shows timing-dependent variation

A partial repeatability check found one changed outcome (rapierV1, knifefight,
seed 2026, learned seat B) with identical evaluation identity and opponent
class hashes: original cp2 win in 57 rounds, fresh control loss in 33. The
frozen opponent initializes its RNG from Clock.getTimeElapsed(), which exposes
accumulated execution nanoseconds. Fixed game seeds therefore do not promise
identical opponent randomness. This is an observed nondeterministic mechanism,
not a traced proof of the exact divergent action. Both valid outcomes remain;
no opponent/harness patch, outcome-selected rerun or threshold change is made.
Use the preregistered fresh matched control and retain the original baseline
comparison too; fresh-seed confirmation remains required.


### V8 bounded continuation guard — 2026-11-10

The first three-hour cycle is still evaluating the preregistered panel. A separate nonpinned watcher, `scripts/continue_v8_after_budget.py`, waits for its verified deadline boundary, checks all seven locks and STOPs, reruns independent input/fit audits, exports and verifies recovery plus an isolated restore, and publishes it before one further bounded cycle. It resumes only unfinished benchmark work; completed fitting is not repeated. Genuine errors, changed runtime, STOP, or more than two first-attempt watchdog interruptions require review. Status is `runs/research-sequence/boundary-status.json`; its own `boundary.lock` prevents duplicate watchers. This is operational continuity, not evidence of stronger play.

### V8 deadline classification repair — 2026-11-10 21:04 UTC

The first cycle completed 412 benchmark games and paused with unchanged runtime. Two in-flight YokoOno/wallsofparadis games received the supervisor's SIGTERM at the deadline. The worker caught the interruption and saved `evaluation_error` before the supervisor could write `worker_timeout`. The continuation guard therefore stopped, correctly refusing an unreviewed generic failure. Both original journals have no terminal outcome and are retained unchanged. `reviews/deadline-recovery-20261110.json` pins their exact hashes and timing evidence before any retry. A separate recovery runner invokes the unchanged benchmark worker for attempt 2 only, under all seven locks/STOPs and a process-group watchdog; the two-attempt cap, game settings, checkpoint, runtime identity and prior successful results remain unchanged. This repairs infrastructure classification, not bot behavior or scoring. Any failed second attempt requires review.


### V8 verified recovery and continuation — 2026-11-10 22:25 UTC

Both hash-pinned deadline cells completed validly on attempt 2; original attempts were preserved. At this boundary there are 414/768 valid benchmark games: cp2 82W174L, terminal 77W81L across158 games, PUCT pending. Independent fit/input audits passed again. Recovery SHA256 `5c19fe28088d2e394d58197180344f9ca44042ec2735dd6c131d372a28655adc` was verified and isolated-restored (6,296 files, 261 checkpoint payloads), then published at d385a2c. The guarded supervisor resumed the unchanged V8 identity for one further three-hour cycle at22:24:50UTC, owner151472. Resume revalidates cached stages before unfinished benchmark work; no outcome-selected reruns or new SGD are intended. No playing-strength conclusion from partial panels.

### Target-reliability planning while V8 control finishes

V8 remains active with its original runtime. Read-only planning reconstructed all32 training target/cp2 action pairs and their source/input hashes. Four fresh continuation seeds require184 unique cells because18 roots retain the cp2 action. Original selection-seed mean advantage0.328125 is in-sample maximization evidence, not independent improvement. The next test remains unlaunched until the full matched control and final V8 audit are reviewed; no gradients or new outcomes were generated by this preparation. See `diagnostics/target-recheck-planning` and `docs/v8-candidate-review.md`.


### V8 complete; V9 selected — 2026-11-11

All768 benchmark cells are valid: cp2 82W174L, terminal81W175L, PUCT79W177L. Both learned arms fail the frozen improvement rule against original and fresh cp2. Final terminal audit is `reviews/completed-panels-final-20261111.json`; complete recovery export was verified and isolated-restored before further actors. No confirmation or promotion. Select the preregistered V9 no-SGD target-reliability recheck:32fixed roots, four disjoint continuation seeds,184shared-action cells. Comprehensive seed scan covered16153localJSON files and7available recovery archives without collisions; reserved formulas reviewed. One-thread serial collection keeps the bounded diagnostic simple. Implementation/protocol are frozen before preflight and launch.


### V9 wrapper repairs and retained-game continuation

V9 stopped before a game because `worker_task_path` was absent. V9r1 fixed that in a separate namespace and completed eight valid terminal continuations; reporting then rejected missing redundant `decisions` metadata. V9r2 uses an exact SHA-pinned compatibility reader deriving only the count from retained action lists, passes full terminal/action-hash validation (including corruption-rejection tests), and imports all eight without rerunning them. Both prior versions and raw journals remain preserved. New records explicitly include decisions. V9r2 preflight passed and its bounded three-hour cycle is live under identity289f0b1eb06f27fe1c6afe5c28dfe53a49d054c1954b7a79eed5046ca874d35c, owner175428, with actual new continuations and no SGD. Early coverage was16/184; no mechanism decision yet.
