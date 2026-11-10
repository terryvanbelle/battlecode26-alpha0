# Cached-head-constraint-v7 preregistration

**Selected on 2026-11-10 after the completed, independently audited v6 review.**
No v7 numerical fit may run until this protocol, config and implementation are
committed. The failed v6 expansion rule blocks expanding that learning recipe;
this is a separate, training-only mechanism diagnostic, not automatic extra
v6 SGD, new rollout collection, or a promoted policy.

## Question and fixed evidence

[V6](terminal-v6-results.md) completed every planned game and both fits, but all
three policies scored 20/24 and its validation advantage was zero. Terminal
labels differed from cp2/PUCT on nine roots; no training-root greedy action
changed. [The training-only diagnosis](v6-training-underfit.md) found substantial
probability movement, opposing label and anchor gradients, and full row rank
of the 32-root feature matrix. These observations do not identify a stationary
optimum, establish generalization or prove the two-seed action labels correct.

This test separates **update budget** from **anchor pressure within that budget**.
Use only the unchanged v6 training dataset: all 32 roots, all 512 anchors and all
32 terminal-selected labels, including the 23 labels agreeing with cp2. Do not
collect new games or read validation feature arrays, actions, outcomes, benchmark
examples or benchmark outcomes to choose a branch, hyperparameter or checkpoint.
V6 metadata contains exclusion-hash strings; they are integrity checks, not
fitting inputs. The completed audit and diagnosis are pinned as opaque hashes.
No v4 root/continuation or benchmark trajectory enters gradients.

`configs/cached-head-constraint-v7.json` pins every dependency by file SHA-256:
cp2, the complete training NPZ/metadata, the original v6 identity, the original
128-batch schedule, the v6 terminal checkpoint, final audit and training-only
diagnosis. Initialize both arms freshly from cp2 SHA
`20d9e813349010630940e1cdfbaee5206490c7d323997688d26f47508bbeee7c`.
Preserve every prior model and run. Use only `runs/cached-head-constraint-v7`
and `progress/cached-head-constraint-v7` for new state.

## Two arms and exact fitting schedule

The sole between-arm difference is the forward anchor-KL coefficient:

| Arm | Anchor coefficient | Updates | Recorded snapshots |
|---|---:|---:|---|
| `kl_1` | 1 | 2,048 | 128, 512, 2,048 |
| `kl_0` | 0 | 2,048 | 128, 512, 2,048 |

Train only existing `policy.weight` and `policy.bias`, exactly 5,160 parameters.
Encoders, fusion and critic remain byte-identical to cp2. Use the original target
`q=.75*one_hot(selected_action)+.25*p0`, restricted to the recorded legal mask and
normalized. Retain the original AdamW learning rate 3e-4, weight decay 1e-4,
betas (.9,.999), epsilon 1e-8, batch sizes 32 roots/32 anchors and seed 203101.
There is no optimizer, topology, feature, target, label or engine change.

Extend the **same deterministic batch stream** to 2,048 steps for both arms.
Its first 128 root/anchor batches must equal the saved v6 schedule exactly.
Root sampling gives equal map/role/team weight; anchor sampling is uniform over
source games, then teams, then anchors within that stratum. Integer storage
quotas do not silently change the sampler's game/team weights. Even the KL=0
arm retains the same anchor sampling stream and diagnostic KL measurements.

Before interpreting later results, the KL=1 snapshot at step 128 must reproduce
the original v6 terminal-head **model weights and AdamW state exactly**. Check
shared batch prefix, loss history and Torch RNG too. The extended schedule's
full digest and final NumPy sampling state need not equal those of a 128-step
schedule. A failed reproduction guard invalidates interpretation and stops work
for implementation review. It does not authorize changing labels or seeds.

Do not stop early when a target is fitted, tune on an intermediate result, or
select a preferred diagnostic checkpoint. Finish both arms to 2,048 and retain
all six fixed snapshots, their optimizer/RNG state, sampled batch arrays and
immutable recovery generations every 16 completed updates. Interrupted in-memory
steps must not be saved as complete evidence. Resume only verified completed
updates, without overwriting later work.

## Reports and precommitted classification

At each fixed snapshot report all 32 selected greedy actions, agreement on the
changed nine and retained 23 separately, mixed-target CE, correctly game/team-
weighted anchor KL, anchor greedy-action changes, parameter displacement and
CE/KL gradient alignment. Retain complete loss curves and partial agreement;
never shrink the denominator or treat disagreement as unavailable data.

Classify using **only steps 128, 512 and 2,048**, in this mutually exclusive order:

1. **Budget evidence:** KL=1 first reaches all 32 selected greedy actions at 512
   or 2,048 after passing the exact 128-step control. A success at 512 followed
   by regression at 2,048 still supplies budget evidence, with the regression
   reported; it does not become anchoring evidence.
2. **Anchoring evidence within the tested budget:** KL=0 reaches all 32 at any
   fixed snapshot while KL=1 never does through 2,048. This does not prove the
   stationary regularized optimum or establish a deployable KL=0 policy.
3. **Neither criterion:** neither arm reaches all 32 at a fixed snapshot.
   Review optimization and conditioning before changing the feature trunk.

An incomplete arm, missing snapshot, failed exact-reproduction check or integrity
failure permits no completed classification. Agreement means the existing masked
greedy rule, including its stable action-ID tie break. No diagnostic head is
promoted, evaluated against benchmarks, or sent back to the old validation set.
Any later learning experiment requires a recorded review and a newly preregistered
fresh validation plan. The 12-opponent/74-map victory gate is unchanged.

## Bounded execution, STOP and recovery

Use one Torch/inter-op thread and one worker; set OMP/MKL/OpenBLAS/NumExpr thread
limits to one. Run no Java engine. The hard limit is **900 total wall seconds
across resumptions**, including worker startup and fitting; report processor
seconds separately. This is not a 900-second budget reset on every launch.
Closed pauses charge actual elapsed wall time and exclude downtime. An unclosed
attempt (for example SIGKILL or host loss) conservatively consumes its entire
reserved remaining budget; it can exhaust the allowance and require a new
recorded operational review, rather than silently reclaiming uncertain time.

The parent supervisor holds five locks: campaign, warmstart, v5, v6 and its own
experiment lock. Honor their five corresponding STOP files and every later user
stop instruction. Inspect commands and flocks first. Matching actor commands
with unreadable `/proc/PID/cwd` remain possible live actors, not evidence of
quiescence. Use an isolated worker process group, a parent-death signal, and a
hard process-tree watchdog. Do not modify pinned code/config/protocol while
active. The audited v6 data and checkpoints remain unchanged.

Publish a separate verified v7 recovery export with all six snapshots, optimizer
and RNG states, the extended schedule, partial recovery state if interrupted,
input hashes and exact dependency locations. The v6 archive remains the source
of original rollout evidence. Preserve current main and newer local work during
restoration; no neutral or warmstart archive substitutes for these run states.

This diagnostic establishes neither learning progress in games nor victory.
Its only possible result is a bounded explanation of the observed fitting failure
and evidence for a subsequent research decision.
