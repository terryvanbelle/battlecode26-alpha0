# Terminal-target-head-v8 preregistered transfer experiment

**Selected on 2026-11-10 after independent verification of v7.**
The [independent v7 audit](../progress/cached-head-constraint-v7/independent-final-audit.json)
has SHA-256 `e3776f8fbde2d0ff65117c2efcf029a54265a734d00b1b069ada06dc70b6ebd2`.
It confirms budget evidence, all six fixed snapshots, 256 recovery generations,
529 retained files and exact reproduction of the 128-update control. The v8
config and runtime identity must pin this prerequisite. Commit this protocol, config and implementation
before generating any source game, fitting a head or running an evaluation.

[V6](terminal-v6-results.md) completed its allocation but failed the improvement
rule: neither 128-update head changed a training-root greedy action, all sixteen
validation actions matched cp2, and all three policies scored 20/24 on smoke.
[V7](cached-head-constraint-v7.md) is a training-only diagnostic. Its recorded
KL=1 arm reaches all 32 selected actions at 2,048 updates, retaining all 23 old
labels and changing only 3/512 anchor actions. Its KL=0 arm also reaches all 32,
but changes 73/512 anchor actions. The independently audited result supports
testing a larger fixed update budget while retaining KL=1. It does not establish
the correctness of two-seed labels, generalization, game strength or victory.

V8 is a separate experiment on newly generated source trajectories. It is not
additional SGD on v6, a deployment of a v7 diagnostic head, or a re-evaluation on
the old validation set. The unchanged scientific rules below are inherited from
the frozen [v6 protocol](terminal-target-head-v6.md). This amendment changes the
fixed fitting budget to 2,048 updates, allocates new randomness, and expands the
validation input-exclusion set. It does not weaken the research decision rule.

## Frozen baseline and allocation

Initialize both heads freshly from warmstart cp2, SHA-256
`20d9e813349010630940e1cdfbaee5206490c7d323997688d26f47508bbeee7c`.
Keep the engine, local resource-limit patches, macro actions, observation
encoding, network topology, inference adapter and PUCT implementation unchanged.
Preserve all earlier models and artifacts. New state belongs only under
`runs/terminal-target-head-v8` and `progress/terminal-target-head-v8`.
The four training-excluded maps in the unchanged victory gate remain excluded
from gradients throughout the checkpoint's initialization/warmstart ancestry.

The complete planned allocation remains:

| Component | Fixed allocation |
|---|---|
| Development maps, in order | `tiny`, `DefaultSmall`, `closeup`, `knifefight` |
| Training roots | 32: eight per map, balanced by baby/king role and team |
| KL anchors | 512 from all completed training discovery sources |
| Terminal targets | Every legal action, two paired continuations; at most 1,600 cells |
| PUCT comparators | 32 roots; unchanged PUCT32, depth 12 |
| Candidate fits | Two fresh cp2 heads; exactly 2,048 updates each; KL coefficient 1 |
| Validation roots | 16: four per map, balanced by role/team; complete exclusion guard |
| Validation continuations | Four paired seeds per distinct root/action; at most 192 games |
| Whole-policy development | Three policies × 8 frozen opponents × 8 maps × 2 seeds × both seats =768 games |

Thirty-two roots remain a small, correlated training sample. Thousands of
continuations or optimizer updates do not create additional independent roots.
The four self-play maps and eight-map development panel are distinct. The old
12-opponent/74-map victory requirement is historical and superseded.

## New source and continuation randomness

For map index `m=0..3`, source index `j`, and team `t=0,1`, use exactly:

| Stream | Formula or fixed values |
|---|---|
| Training engine seed | `20263200 + 100*m + j` |
| Training policy seed, team t | `2*(330000 + 100*m + j) + t` |
| Training initial sources | `j=1..4` per map |
| Training reserves | `j=5..8` per map |
| Validation engine seed | `20264200 + 100*m + j` |
| Validation policy seed, team t | `2*(340000 + 100*m + j) + t` |
| Validation initial sources | `j=1..2` per map |
| Validation reserves | `j=3..6` per map |
| Training terminal continuations | `205003`, `205013` |
| Validation terminal continuations | `205021`, `205043`, `205067`, `205091` |
| Optimizer/shared batch stream | `205101` |
| Anchor sampling | `205151` |
| PUCT noise | `205201 + root_ordinal`, ordinal 0..31 |

Audit all new source-engine, source-policy, continuation, anchor, optimizer and
PUCT streams before launch against earlier experiment seeds and reserved
benchmark-confirmation seeds. Do not replace any seed based on an outcome.
The fixed eight-map/seed0+2026 development schedule is a repeated development screen;
the new randomness above does not make that known screen a fresh holdout.

All training and validation source games use unchanged cp2 on both teams,
source temperature 1.0, and no league opponents. All terminal continuations use
cp2 on both teams at temperature 0.35. Generate new source records, prefixes,
inputs, caches and manifests. No earlier trajectory or cache supplies v8 data.
Naturally recurring exact training inputs are measured as overlap, not silently
claimed to be novel or removed to improve a result.

Odd initial `j` assigns `(A baby, B king)`; even `j` assigns `(B baby, A king)`.
Training uses the first turn of each assigned role/team with macro 37 legal.
Validation uses the first such turn whose canonical input is outside the full
exclusion union below. Slot selection cannot use winners, scores, priors,
values or future continuation outcomes.

## Fixed reserves and coverage checks

Retain the v6 deterministic assignment exactly. Process maps in fixed order,
then original source groups by increasing `j`. For a group with missing slots,
take the lowest unused reserve of matching parity, assign it exclusively to
that group, and fill only that group's still-missing roles/teams. Once filled,
a slot is never replaced. Training odd reserves are 5,7 and even reserves 6,8;
validation odd reserves are 3,5 and even reserves 4,6. A consumed reserve cannot
be reassigned to another group. No physical `(source_id, decision_index)` fills
two slots.

For validation, skip excluded inputs chronologically within the same source
before declaring its slot unavailable. The same guard applies to reserves.
The larger exclusion union may exhaust the available sources: this is a
coverage failure, not permission for `j>6`, a smaller denominator or weaker
exclusions. Require all 32 training slots, all 512 anchors and all 16 validation
slots before target scoring or fitting. Freeze the entire selection plan
`C = 2 * sum(number_of_legal_actions)` and require `C<=1600`. An unmet requirement
stops the version for research review without SGD or outcome-driven pruning.

Report initial and reserve source counts separately, attempted/completed games,
accepted roles/teams by map, unavailable slots and distinct represented source
games. A source with no accepted root is not an additional validation scenario.

## Complete historical and current input exclusion

Keep the canonical `gradient-input-v1` encoding from v6 unchanged: ordered
local/global/scalar feature arrays in C-contiguous little-endian float32,
canonical positive zero, finite-value checks, then the uint8 legal mask, with
the length-prefixed schema/header and SHA-256. Public-state hashes, RNG seeds,
hidden embeddings and action labels are not substitutes for this input hash.

Define three frozen sets:

- `G_new`: canonical hashes of every v8 training root and all 512 v8 anchors.
- `G_old`: all 544 distinct v6 gradient inputs: its 32 roots and 512 anchors.
- `V_old`: all 16 v6 validation input hashes, disjoint from `G_old`.

The historical set is exactly `H = G_old union V_old`, containing 560 distinct
inputs. V7 reused the same v6 training inputs and contributes no additional
inputs. Pin the original hash inventories by file SHA-256. Read their input
hashes and provenance only; previous validation actions and outcomes do not
select a new root, label, hyperparameter or checkpoint.
This explicit guard covers v6/v7 inputs and the current v8 gradient set. It does
**not** enumerate every input used in earlier cp2 bootstrap/warmstart ancestry;
do not describe it as universal historical training-input independence.

Freeze training roots and anchors before validation discovery. The complete
validation exclusion set is **`U = G_new union H`**. Freeze every component list,
the sorted unique union and their canonical JSON SHA-256 fingerprints. Attach
the union digest to validation discovery jobs, selected-root manifests, dataset
metadata and fit evidence. Every selected validation input must be absent from
`U`. Recheck both the current-gradient and full-union intersections before fitting,
before validation scoring and in the final report. Checking only `G_new` does
not satisfy v8, even when a copied older proof reports zero intersection.

For every skipped input, record source, decision index, role/team, input hash
and all matching categories: current training root, current anchor, prior v6
gradient input and/or prior v6 validation input. Report counts and reserve use.
Keep all predeclared training roots and anchors regardless of overlap with
earlier inputs. Report old/new training-input intersections and distinct input
counts descriptively; do not reject training slots, reweight them, or select
labels based on historical outcomes. New source trajectories do not guarantee
previously unseen arrays. Exact exclusion also does not make nearby states or
multiple roots from one source statistically independent.

## Matched target construction and longer fixed fitting

Freeze roots, source prefixes, input/state hashes, legal-normalized cp2 priors
and PUCT comparators before terminal scoring. Preserve v6 selectors:

- Terminal arm: score every legal first action using both fixed training seeds;
  choose highest mean root-team terminal score, with ties resolved by cp2 prior,
  then action ID. Win/draw/loss scores are 1/0.5/0. Failed or partial games are
  unscored. Retain the entire action table, including losing alternatives.
- PUCT arm: unchanged 32 simulations, depth 12, `cpuct=1.5`, Dirichlet alpha 0.3,
  exploration fraction 0.25 and root noise enabled; use the fixed per-root noise
  seed and existing stable legal-action order. Retain complete visit counts.

Both arms use `q=.75*one_hot(selected_action)+.25*p0`, restricted to and normalized
over the recorded legal mask. Neither arm uses raw visit counts as its target.
Initialize both from cp2 and train only `policy.weight` and `policy.bias`, exactly
5,160 parameters. Encoders, fusion and value head remain byte-identical to cp2;
verify cached critic predictions too.

Fit both candidates for exactly **2,048 AdamW updates**. Retain learning rate
`3e-4`, weight decay `1e-4`, betas `(.9,.999)`, epsilon `1e-8`,32 roots and 32 anchors
per update, and anchor-KL coefficient 1. The optimizer/sample seed is 205101.
Freeze one complete 2048-batch schedule and use it in both arms. Root sampling
is balanced by map/role/team. Anchor sampling is uniform over source games,
then teams, then anchors within the game/team stratum.

The 512 anchors still come from **all completed training discovery sources**,
including sources with no accepted roots. Use v6's per-source/team Algorithm-R
reservoir of at most 16 positions, deterministic hash-derived reservoir seed
with `anchor_seed=205151`, ordered integer quotas and a separate
`numpy.random.default_rng(205151)` quota-selection stream. Preserve lossless
float32 inputs. A short reservoir is a coverage failure; never duplicate an
anchor or silently redistribute sampling weight. No validation source supplies
an anchor.

The objective remains target CE plus forward `KL(p0 || p_theta)`, each with
coefficient 1. Report correctly game/team-weighted full-set KL in addition to
the sampled-minibatch trace; label any plain storage mean separately.
Retain optimizer/RNG states and immutable recovery generations. Record all
observed losses and training target agreement, changed-label agreement,
retained-label agreement, parameter movement and anchor action changes.
The fresh set need not contain exactly nine changed labels: derive those group
sizes from the new frozen targets and retain the full32-root denominator.

**Only the final 2,048-update head from each arm is evaluated.** No early stop,
best-checkpoint selection, optimizer restart, coefficient branch or additional
update budget is permitted after inspecting training, validation or bots.
Intermediate recovery state is not a candidate. Preserve failed fits and all
data. A fresh set that remains underfit is evidence, not permission to enlarge
the budget inside v8.

## Fresh first-action validation and full-policy development evaluation

Freeze the complete 16-root validation manifest before target scoring or fitting.
After both final heads are fixed, record each head's and cp2's masked greedy
action on every validation root. Do not choose only roots on which policies
differ. Force each distinct selected first action and use cp2 on both teams at
temperature 0.35 for the remainder. Use all four fixed validation seeds and
physically deduplicate identical root/action/seed cells while preserving every
arm label. There are at most 192 games. All incomplete/error games are unscored.

Report paired means with equal weight per root, all 16 root comparisons, the
number of changed actions, and descriptive groups by map, role and team.
Report the actual distinct source games as the clustering units; continuations
are not independent roots. Do not reuse v6 validation roots, arrays, actions or
scores. New validation observes transfer of one decision under cp2 continuation,
not deployment of the learned policy for a full game.

Then run 768 fresh whole-policy development games: cp2, terminal head and PUCT
head against the frozen eight-bot v4 panel on all eight maps, seeds 0/2026,
both seats. Use configs/improvement-v4.json and benchmarks-development-v4.json,
unchanged greedy adapter and 1,800-second per-game timeout. Freeze all three
candidates before validation. Every wrapper binds experiment, checkpoint,
opponent, map and runtime; all three panels are fresh. No qualification or
historical smoke outcomes substitute for these games. Benchmark trajectories
never enter gradients, targets, anchors or root selection.

## Research decision under the user's broader-map amendment

Require complete collection, both fixed fits, all 16 valid validation roots,
all 768 benchmark games, zero unresolved errors and identity/integrity checks.
The terminal-ranking head is the preregistered candidate; PUCT is the matched
control. Report both comparisons, without choosing whichever arm wins afterward.
The terminal candidate must pass the unchanged development-v4 rule: at least
0.05 mean score gain against fresh cp2, paired-map 95% bootstrap lower bound
above zero and aggregate zdb nonregression. No individual-opponent required win
or nonregression is imposed. Passing leads to frozen fresh-seed confirmation,
not automatic promotion. The 70% overall-win milestone shifts research priority
to tournament compatibility as documented in tournament-transition.md.

The original first-action mechanism criteria (mean terminal-minus-PUCT ≥0.10,
at least two improving roots, worst root delta ≥−0.25, and nonnegative mean
terminal-minus-cp2) remain reported as a separate diagnostic. They no longer
veto an independently measured development improvement. This is a pre-launch
protocol amendment following the user's new objective, not an outcome-based
change. Report all roots/map means even if the diagnostic fails. Two-seed
training ranking is noisy; fitting its labels does not prove stronger play.

The small source/root sample, exact-input overlap beyond the guarded set and
limited map coverage remain explicit limitations.

Failure preserves both heads and all evidence and returns to research review.
Passing permits a recorded review of whether a larger independently validated
learning experiment is justified. **No model is automatically promoted.** No
v8 result establishes all-opponent/all-map victory; that goal was superseded by
the user and its gate remains historical.

## Bounded supervision and recovery

Retain the v6 three-hour process-tree watchdog and fixed data ceilings. Use at
most three one-Torch-thread collectors, one fitting worker, and at most two
benchmark workers. Monitor aggregate Python/Java CPU and RSS. Inspect process
commands and lock ownership before launch; unavailable process metadata is not
proof that no actor exists. Do not modify pinned files while an actor is active.

Hold all seven experiment locks and honor all seven STOP files:

| Owner | Lock | STOP |
|---|---|---|
| Campaign | `runs/campaign/campaign.lock` | `runs/campaign/STOP` |
| Warmstart | `runs/warmstart-v3/experiment.lock` | `runs/warmstart-v3/STOP` |
| V5 | `runs/terminal-target-head-v5/experiment.lock` | `runs/terminal-target-head-v5/STOP` |
| V6 | `runs/terminal-target-head-v6/experiment.lock` | `runs/terminal-target-head-v6/STOP` |
| V7 | `runs/cached-head-constraint-v7/experiment.lock` | `runs/cached-head-constraint-v7/STOP` |
| V8 | `runs/terminal-target-head-v8/experiment.lock` | `runs/terminal-target-head-v8/STOP` |
| Development | `runs/development-v3/experiment.lock` | `runs/development-v3/STOP` |

Later user stop instructions also apply. Workers must have isolated process
groups, parent-death handling and hard descendant watchdogs. Atomic per-attempt
journals preserve errors and incomplete work. Resume only unchanged identities
and verified recovery state; interrupted work never receives a score or an
invented completed update. A pause does not authorize new source seeds, reserve
slots, target cells, hyperparameters or candidate selection.

Publish measured source coverage, collection throughput, actual fitting curves,
separate collected-versus-scored validation coverage, every root comparison and
every benchmark seat. Reports must distinguish unavailable data from observed
losses and historical references from fresh results. Preserve exact source,
input, target, checkpoint, optimizer/RNG, runtime and map/opponent hashes in a
separate verified v8 recovery export. Keep all v6/v7 archives and current/newer
local work intact during restoration.


## Recovery command

After all seven locks are released and no matching actor remains, use
`.venv/bin/python scripts/export_terminal_v8.py export`. Verify the archive and
perform an isolated restore before publishing its pointer. It verifies final
2,048-update heads, frozen tensors, optimizer/RNG/batch state and all owned
raw data. Historical v6 exclusion metadata and v7 audit are archived as
references, never restored over current files. Original v6/v7 archives remain
separate dependencies. Never bypass actor/lock checks to export a running cycle.
