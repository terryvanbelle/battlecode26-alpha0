# Terminal-target-head-v5 preregistered protocol

**Selected for the next bounded cycle by the 2026-11-10 research review.**
[V4 completed 418 valid continuations and passed its mechanism rule](root-ranking-v4.md),
with a separate [independent final audit](../progress/root-ranking-v4/independent-final-audit.json).
This pilot has not yet been implemented or run. Verify that evidence and the
frozen cp2 dependency before collection. Do not train on any v4 root,
selection continuation, or validation continuation.

This pilot asks whether useful terminal-rollout action rankings can transfer to
new decisions through a small policy update. It is a controlled policy-learning
experiment, not another full AlphaZero iteration or a victory claim. Thirty-two
fresh target states are a deliberately small pilot; hundreds of continuations
from those states do **not** become hundreds of independent training states.

## Frozen baseline and two matched arms

Initialize both arms from unchanged warmstart cp2, SHA-256
`20d9e813349010630940e1cdfbaee5206490c7d323997688d26f47508bbeee7c`.
Preserve cp2, the original prior champion, neutral control, and all v4 evidence.
The engine, action macros, observations, inference adapter and network topology
remain unchanged. Use new `runs/terminal-target-head-v5` and
`progress/terminal-target-head-v5` paths.

Both arms receive the **same fresh states**, prior distributions, sample order,
optimizer seed, loss and number of updates. They differ only in the action
selector that supplies the target:

- **Terminal-ranking arm:** every legal first action gets two paired frozen-cp2
  terminal continuations. Choose highest mean root-team score; ties use cp2 prior,
  then action ID.
- **PUCT control:** ordinary unchanged 32-simulation/depth-12 PUCT on that same
  state: `cpuct=1.5`, Dirichlet alpha `0.3`, exploration fraction `0.25`, root
  noise enabled, and noise RNG seed `203201 + root_ordinal`. Choose its visit
  argmax using the existing stable legal-action order for ties. Record the
  complete visit distribution too.

For both arms construct the same target form:
`q = 0.75 * one_hot(selected_action) + 0.25 * p0`. Here `p0` is the frozen cp2
prior restricted to the recorded legal action mask and normalized to sum to
one. Set every illegal-action probability to zero; reject nonfinite priors or
zero legal mass. Apply the same legal mask and normalization to the candidate
policy. This rule applies to **every training target and every KL anchor**.
This isolates **selected-action transfer**. The PUCT control uses the ordinary
search's action, but this pilot deliberately does not train one arm on raw visit
counts and the other on a hard label. It therefore is not an isolated comparison
against the full v3 visit-distribution training recipe. Any later return to full
MCTS training needs a separately stated protocol.

## Fresh, symmetric data and fixed allocation

Use the same four development maps in fixed order: `tiny`, `DefaultSmall`,
`closeup`, `knifefight`. Generate new complete self-play games with **frozen cp2
on both teams, temperature 0.35 on both teams, and no league opponents**. Both
seats supply data. No benchmark policy, trajectory, action, or outcome enters
collection, target selection, filtering, or fitting.

For map index `m=0..3`, start with four training source games, engine seeds
`20261200 + 100*m + j`, `j=1..4`. Source-policy RNG streams for team `t=0,1`
are seeded `2*(310000 + 100*m + j)+t`. Assign two training root slots per game:
odd `j` selects `(A baby, B king)`, even `j` selects `(B baby, A king)`.
Each slot is the first turn of that role/team where macro 37 is legal. This
gives exactly **32 planned roots from 16 initial source games**, eight roots
per map, balanced across role and team. Predeclared training reserves use
`j=5..8`, with the same seed formulas and role/team rules. Reserves replace only
unavailable slots under the deterministic assignment below. Require all 32
slots before target collection; otherwise record the coverage shortfall and
return to research review without SGD.

Freeze every root, complete source prefix, state hash, prior and PUCT comparator
before terminal scoring. Training continuation seeds are **203003 and 203013**.
The root-team outcome is win 1 / draw 0.5 / loss 0. No partial or failed game
receives a score. Keep the full action table, including unfavorable alternatives.

Freeze the actual number `C = 2 * sum(number_of_legal_actions)` before collection.
The observed v4 mean is 23.375 legal actions, implying about **1,496 training
continuations** for 32 roots. Set a precommitted collection ceiling of **1,600
cells**. If the discovered plan exceeds that ceiling, pause for budget review
before scoring rather than silently removing expensive roots. Do not increase
the target count because early results are favorable or collection is fast.

For fresh validation, generate two source games per map using engine seeds
`20262200 + 100*m + j`, `j=1..2`, and per-team policy seeds
`2*(320000 + 100*m + j)+t`. Use the same alternating role/team root rule.
Predeclared validation reserves use `j=3..6` with these validation seed formulas.
This requires exactly **16 validation roots**, from eight initial source games
plus any reserves actually used. Fresh RNG seeds alone do not guarantee fresh
model inputs. Freeze the training roots **and the complete KL-anchor sample**
first, then apply the exact input-overlap guard below during validation root
selection. Resolve reserves and freeze the complete validation manifest before
target scoring or fitting. Run no validation continuations until both candidates
are fixed. Missing validation slots leave the pilot incomplete; never reduce
its denominator. Validation states, priors and outcomes are excluded from
target fitting, anchors and checkpoint selection.

## Exact gradient-input overlap guard

Build the anchor sample described under [Conservative fitting](#conservative-fitting)
after training-source discovery is complete. Freeze the anchor positions and
their inputs before selecting the final validation roots. Define `G` as the set
of canonical input hashes for **all 32 training roots and all 512 KL anchors**.
Training root selection and anchor sampling remain unchanged; this guard does
not remove or replace training examples.

The hash identifies the actual model input, not its source seed, public-world
hash, hidden-state embedding, policy prior or outcome. In fixed order, serialize
`local`, `global_board`, and `scalars` as C-contiguous, little-endian float32
arrays, followed by the legal-action mask as C-contiguous uint8 values 0/1.
Use the exact float32 values supplied to the model after encoding and any
storage casts; reject nonfinite values and canonicalize signed zero to positive
zero. Encode a compact, sorted-key UTF-8 JSON header containing schema
`gradient-input-v1`, feature version, action version, and ordered field names,
shapes and dtypes (`<f4` for features, `|u1` for the mask). Prepend its byte length
as an unsigned 8-byte little-endian integer, then the header, then the ordered
array bytes; hash that complete sequence with SHA-256. Retain inputs losslessly so later
storage changes cannot invalidate the guard silently. Apply this same encoding
to roots and anchors; a legal-mask difference is part of the input identity.

For each validation role/team slot, scan its source game chronologically. A
candidate must have macro 37 legal **and an input hash absent from `G`**. Skip
an exact gradient-input duplicate and continue scanning the **same source game**
for the first eligible unmatched turn. If no such turn exists, mark that slot
unavailable and use the fixed matching-parity reserves. Use the same exclusion
and chronological scan in reserve games. Do not use game outcomes, continuation
results, priors, values or other model outputs to select or skip a candidate.

Record every skipped exact duplicate with source game, decision index,
role/team, input hash and whether it matched a training root, an anchor, or both.
Freeze `G` as a sorted unique hash list and its compact-JSON SHA-256 digest with
the training/anchor manifests; attach that digest
and each selected input hash to the validation manifest. Recheck disjointness
before fitting and before validation scoring. Report skip counts, unavailable
slots, reserve assignments, actual source-game counts and distinct validation
input counts. The allocation remains exactly **32 training and 16 validation
roots**. This prevents exact gradient-input reuse; it does not make nearby or
structurally similar states independent.

## Deterministic reserve assignment and seed separation

Process maps in the fixed order above, then original source-game groups in
ascending `j`. Within a group, slots retain their declared baby/king order.
A group with missing slots takes the lowest **unused reserve j of matching
parity** for that map and split. Assign that reserve game exclusively to the
group and fill only its still-missing role/team slots, using each slot's first
qualifying turn. For validation, "qualifying" includes the input-overlap guard
and continued chronological scanning described above. Once filled, a slot is
never replaced. If slots remain missing, try the next unused matching-parity
reserve. A consumed reserve is never
reassigned to another group. Training odd reserves are `5,7`, even `6,8`;
validation odd reserves are `3,5`, even `4,6`.

No physical root `(source_game_id, decision_index)` can fill two slots. Training
selection uses role, team, legality and chronological order; validation adds
the exact gradient-input exclusion. Source-game winners and all continuation
outcomes are irrelevant. Exhausting the prescribed
reserves leaves missing coverage, not permission to invent more seeds or drop
slots. Record attempted/completed source games, qualifying/missing slots,
assignments, and distinct source games represented by accepted roots for both
splits. These are separate counts: a completed game with no accepted root does
not increase the root-validation scenario count. The allocation stays exactly
**32 training and 16 validation roots** even when additional source games are
needed. Freeze both manifests and the anchors before target scoring or fitting.

The source engine/policy seed formulas, continuation seeds, anchor seed, optimizer
seed and PUCT seeds are fixed before collection. Audit their split separation
and exclude every v4 diagnostic seed and every reserved benchmark confirmation
seed. Do not reuse a training source prefix or state as validation. The
additional canonical input guard excludes an exact root/anchor input even when
it comes from a different seed, prefix or physical game.

## Conservative fitting

Train only `policy.weight` and `policy.bias`: **5,160 parameters**. Freeze local,
global and scalar encoders, fusion, and the complete value head. This allows
context-dependent action changes while preventing a tiny dataset from moving
the shared representation or confounding the result with critic changes. Verify
all frozen tensors byte-for-byte after fitting.

Cache cp2 features and priors once. Use 128 AdamW updates, with 32 target roots
and 32 anchors per update, learning rate `3e-4`, weight decay `1e-4`, betas
`(0.9, 0.999)`, epsilon `1e-8`, and fixed optimizer/sample seed `203101`.
Sample training roots with equal map/role/team weight. Add a cp2-preservation
term with coefficient 1: forward `KL(p0 || p_theta)` on a deterministic
512-position sample from **all completed training source games generated for
initial/reserve discovery**, balanced by game and team, with anchor-reservoir
RNG seed `203151`. This includes a training source game even if none of its
slots qualified; validation source games are always excluded. Store each anchor
with its legal mask and normalized `p0`. In each anchor batch choose source
game uniformly, team uniformly, then an anchor uniformly within that stratum;
this preserves equal game/team weight even if integer storage quotas differ.
Compute KL only on legal actions, treating zero-probability terms as zero.
These anchor positions use cp2 probabilities, not winning labels.

The implementation pins the previously unspecified integer storage convention
before source generation. Each completed training source uses an independent
Algorithm-R reservoir of at most 16 positions per team. Its NumPy PCG64 seed is
the integer value of the first 16 hexadecimal characters of SHA-256 of compact,
sorted-key JSON `{"anchor_seed":203151,"source_id":ID,"player":TEAM}`.
After all initial and required reserve games are known, order strata by source
ID, then team 0/1. Give each `floor(512 / number_of_strata)` positions and give
one extra to each of the first remainder strata. In this fixed order, use one
separate `numpy.random.default_rng(203151)` stream to select each quota without
replacement from that stratum's chronologically sorted reservoir. This nested
uniform sampling preserves equal selection within each source/team. A stratum
shorter than its quota is an explicit coverage failure, not permission to
duplicate positions or redistribute its weight. Record all quotas, reservoir
seeds and chosen decision indices. All retained source inputs use lossless
float32 storage; cached head features are also float32.

The loss is mean target cross-entropy plus mean anchor forward KL, each with
coefficient 1. Precompute and freeze the sampled target/anchor batch indices;
use the same anchors and minibatches in both arms. Do not tune hyperparameters,
choose an intermediate checkpoint, or stop early based on validation or bots.

Report target-action differences, fitting loss, anchor KL, and changed-action
rate. A failure to fit or generalize is useful evidence; do not unfreeze the
trunk or add more updates within this version to rescue the result. Value
calibration is a separate experiment, since this pilot deliberately performs no
value training.

## Independent validation and benchmark check

On each frozen validation root, take the greedy action from each fitted head
and original cp2. Force that first action, then use frozen cp2 for **both** teams
at temperature 0.35 for the remainder. This tests transfer of the first decision
under the same continuation regime as the targets. Greedy policy ties use
lowest action ID. It does not test full-game deployment of the fitted policy;
the benchmark stage below does that.

Use paired continuation seeds **203021, 203043, 203067, 203091**. Deduplicate
identical actions physically and reuse their four outcomes, retaining separate
comparison labels. There are at most **192 validation continuations**
(`16 roots * 3 distinct actions * 4 seeds`). Report equal-root paired scores
against the fitted PUCT control and against cp2, grouped by map, role and team.
Report the actual distinct source games represented by validation roots,
including reserves; without reserves it is eight, not 192. Also report all
completed source games considered during discovery. Multiple roots or
continuations from one source game are correlated.

Then evaluate cp2 and both fixed candidates against **all 12 frozen bots on
tiny in both seats**, using the existing smoke identity, greedy adapter, seed
and timeout settings: 72 games in total, with zero unresolved errors required.
Fresh cp2 evaluation avoids depending on a historical watchdog difference.
These games are evaluation only. No benchmark outcome selects targets or
changes a candidate. No final all-map claim follows from this screening panel.

Precommit a **review-for-expansion** rule: terminal-ranking head has mean paired
validation advantage at least 0.10 over the matched PUCT head, improves at least
two roots, and has no root disadvantage below -0.25; its mean versus cp2 is
nonnegative; and all three smoke panels complete without errors. Additionally,
for **each of the 12 opponents separately**, the terminal-ranking candidate
must have a two-seat mean score at least as high as fresh cp2 on that opponent.
Report all 12 comparisons and both seat outcomes, plus aggregates and the
historical 20/24 baseline; an aggregate tie cannot hide forgetting against one
bot. Failure preserves both candidates and all data without automatic extra
SGD. Passing permits a recorded autonomous research review to choose whether
to expand collection or resume full MCTS learning. It does not automatically
promote a model. The all-74-map victory gate is unchanged.

## Two-to-three-hour operational budget

Use at most **three collectors**, each with one Torch thread and its own Java
engine, on the four-CPU host. Monitor combined Python/Java CPU and memory use;
the worker count does not guarantee a reserved idle CPU. Partition roots
deterministically across workers; never schedule the same cell twice. Fitting
uses one thread after collection. Benchmark evaluation uses at most two workers
after fitting. Hold `runs/campaign/campaign.lock`,
`runs/warmstart-v3/experiment.lock` and
`runs/terminal-target-head-v5/experiment.lock`. Honor `runs/campaign/STOP`,
`runs/warmstart-v3/STOP`, `runs/terminal-target-head-v5/STOP` and later user stop
instructions; enforce a three-hour process-tree watchdog. Inspect actual
commands and lock ownership before launching, and never modify an active
actor's recorded runtime files.

| Work | Allocation / estimate |
| --- | --- |
| Source games, discovery, freezing inputs | Reserve 10 minutes |
| Training target cells | At most 1,600; ideal three-worker estimate about 22–107 minutes at 2.5–12 s/cell |
| PUCT comparators, feature caching and two head fits | Reserve 5 minutes |
| Fresh paired validation | At most 192; ideal three-worker estimate about 3–13 minutes |
| Three complete smoke panels | Reserve 15 minutes |
| Integrity audit, plots and export | Reserve 10 minutes |

The sum is roughly **65–160 minutes**, including the stated overhead reserves,
within a 180-minute cap under a planning band of 2.5–12 seconds per continuation,
informed by the observed single-worker root timings. **Threefold throughput
has not been measured.** Those rows divide
single-worker cost by three for planning, not as an empirical speedup claim.
V4 timings may be a favorable sample; source reserves, longer games, shared
CPU and engine overhead can make the cycle longer. Report actual worker/core
time and wall time. If the deadline is reached, retain immutable completed
cells and unscored interruptions. A subsequent recorded research review may
allocate another bounded resumable cycle under the existing authorization.
Never fit on an outcome- or completion-selected subset to meet the clock.

Before launch, pin a separate v5 config, root/seed manifests, exact code/runtime
hashes, dependency checkpoint, and loss protocol. After collection, publish
all counts and per-stage elapsed/core time, not merely the number of SGD steps.
Publish a separately verified v5 recovery export containing both completed
candidate checkpoints, optimizer/RNG and batch state, source transcripts,
root/anchor inputs or exact reconstruction dependencies, targets, cell evidence,
and evaluation identities. Keep benchmark records separate from gradient data.
Do not use the neutral snapshot or warmstart export as if it contained this new
run, and never restore over newer work. This document changes no v4 runtime.
