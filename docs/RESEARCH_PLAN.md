> **Current objective amendment (November 10, 2026):** the user replaced the
> all-opponent/all-map success requirement with improvement on a broader-map,
> smaller opponent panel. See [active protocol](development-benchmarks-v3.md). This supersedes conflicting historical
> instructions below. Publish opponent-by-map W/L/D and coverage in the README.

# Research plan: qualify against every opponent on every map

The target is a single frozen learned policy that beats each member of the
12-bot panel on **every one of the 74 available maps**, with both seats covered.
Routine evaluations use small map subsets; the complete inventory is reserved
for occasional full checks and the final qualification stages. The initial experiment
does not meet that target: the final checkpoint won **5/24** smoke games, equal
to the untrained checkpoint's aggregate score. Its value estimator also
collapsed. More iterations of the same small experiment are not an adequate
response to those observations.

This document separates measured evidence from requirements for subsequent
experiments. The gate is machine-readable in
schema 2 of [`configs/victory_gate.json`](../configs/victory_gate.json), enforced
by [`alpha0.victory`](../alpha0/victory.py). Training and
evaluation implementations must enforce those requirements before declaring
success; the existence of a configuration file alone is not evidence of a pass.

## What the first experiment established

The official-engine integration, snapshot branching, complete self-play games,
optimizer updates, frozen-bot execution and progress reporting work. Independent
tests compare full games with the native `GameWorld.runRound()` scheduler and
verify deterministic branching and optimizer/RNG restoration.

Only six self-play games and 96 gradient updates were performed. The network's
49,242 parameters were fitted to highly correlated decisions from two maps.
Four PUCT simulations per decision provide a pipeline test, not strong search.
Evaluation used the greedy network without search because live Java opponent
threads cannot be cloned. Keep that evaluation policy fixed when comparing
checkpoints, or start a separately labelled series when it changes.

All 24 final smoke results ended by king destruction. There were 19 losses;
five of those had zero cheese in the learned team's bank at termination. These
are diagnostic observations, not evidence that starvation caused all losses.
The existing smoke map overlaps training and supplies only one map–seed pair
per opponent. No strength or Elo claim follows from it.

## The value failure is measured, not hypothetical

The recorded final iteration value MSE is **2.74076**. With targets in
`{-1,+1}`, the maximum possible error of a bounded value is 4; a constant-zero
predictor has MSE 1. The apparent low losses in earlier iterations were not
reliable evidence of generalization.

An independent audit sampled 768 positions without replacement from each saved
replay, using `numpy.random.default_rng(12345)` successively for replays 1, 2 and
3. It evaluated every saved checkpoint with one PyTorch CPU thread:

| Replay | Positive terminal labels, all 10,000 positions | Share from last game | Checkpoint 2 MSE | Checkpoint 3 MSE | Checkpoint 3 mean value |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 59.14% | 100% | 0.0274 | 2.3668 | -0.9985 |
| 2 | 50.57% | 90.90% | 0.2254 | 1.9460 | -0.9981 |
| 3 | 58.75% | 91.77% | 3.8587 | 2.2792 | -0.9979 |

Every sampled checkpoint-3 prediction had `abs(value) > 0.95`. The final
replay is not overwhelmingly negative; the collapsed output is therefore not
explained by marginal label imbalance. Checkpoint 2 was almost maximally wrong
on replay 3 despite fitting the earlier replay. The dominant `DefaultSmall`
game's winner changed from A to A to B across the three iterations. This is
consistent with memorizing game/geographic correlations followed by a sharp
distribution change, then saturating the tanh value head. It does not establish
that any one input feature caused the failure.

The replay SHA-256s for this audit are:

```text
9252c397f692c5bd21dbe6aca8b189dacb8d8a49352c02526ff48cb057a2e301  replay-0001.npz
a0148e173fe624565d34e538856b707a30ab2aa6e282cb459d52230ebda3f469  replay-0002.npz
cd8fbed2b1915b97cb103bda69047696339b56b241f80161827c282352547152  replay-0003.npz
```

These buffers are retained under `runs/initial/`; model checkpoints and game
transcripts are also exported under `artifacts/initial/`. The old run remains
an immutable baseline, not a source of invented improvement claims.

## Highest-leverage changes

1. **Balance replay by completed game.** Keep separately identified episode
   shards, cap each game's contribution, sample games uniformly before sampling
   their positions, and retain many games. A smaller per-game reservoir alone
   prevents one-game replacement but does not equalize unequal game lengths.
   Preserve the real terminal result and player-relative labels. Do not balance
   outcomes by falsifying labels or add unfinished games as draws. Record game,
   side, map, checkpoint and round metadata so diversity can be audited.

2. **Make confident value errors recoverable.** Initialize the final value head
   at zero, lower the initial learning rate to `3e-4`, and use a raw win logit
   with `BCEWithLogitsLoss(logit, (z+1)/2)`. Convert it to
   `2*sigmoid(logit)-1` for MCTS. This is a documented change from the initial
   tanh-plus-MSE objective, retaining the same terminal win target. Report value
   MSE, BCE, predicted mean, saturation fraction and calibration on complete
   held-out self-play episodes. Wrong-sign logits of magnitude 20 must retain
   finite nonzero gradients in a regression test. A small loss on the replay
   used for updates is insufficient.

3. **Remove avoidable action and observation bottlenecks.** The initial 17
   actions cannot choose traps, ratnapping, throwing, king conversion or whether
   to preserve cooperation. Fixed opportunistic attacks can force backstabbing.
   A local 17×17 crop plus aggregate counters cannot locate a distant king or
   cheese source. The next version should add a coarse full-map view, tactical
   state and richer legal action choices. Version the observation/action schema
   and start a new checkpoint series; do not silently reinterpret old weights
   or replay tensors. Keep every action inside official controller legality.

4. **Measure search in useful game time.** A search ply is one robot turn, not
   one round. With many rats, a shallow tree may not reach the next king turn.
   Four simulations often supervise a narrow prior-driven choice. Log visited
   actions, depth in decisions and rounds, root value and policy entropy. Raise
   the budget only after value calibration and throughput are acceptable.
   Compare 8/16/32 simulations on the same development setting rather than
   assuming a larger number automatically helps.

5. **Increase self-play diversity.** Rotate maps, seeds, seats and start a
   league of the current policy and earlier self-play checkpoints. Evaluate
   whether the policy is forgetting previous opponents. Early curriculum maps
   may be `tiny`, `DefaultSmall`, `closeup` and `knifefight`; the four reserved
   maps `Stash`, `Hike`, `heart` and `ratsareants` must never enter gradient
   training, including warmstarts and checkpoint ancestry. Other maps may enter
   a versioned curriculum; fresh confirmation seeds must remain unused.
   Benchmark bot code, action traces and benchmark outcomes never enter
   gradient training or the self-play opponent pool.

6. **Retain a measured champion.** Compare a challenger against the previous
   learned champion using a separate self-play validation schedule. Publish
   both results and the promotion rule. Do not replace the champion merely
   because the training loss fell. The development benchmark panel can guide
   research decisions, but final fresh-seed confirmation has a different role.

## Frozen all-map success gate

The panel is exactly `field-twelve-v2`, with the twelve source commits and Java
tree hashes copied into the gate. Its canonical fingerprint is
`f18d20f65e086a216c53613e8aaf1de6b714c0a5cea8c1ffcf5505ce72c3a43d`.
No strong opponent is removed to improve a score. Repeated uncaught failures
under a validated harness can justify an explicitly versioned replacement,
with the failed source and all measurements retained. Legal resignations,
disintegration, physical deaths and ordinary losses do not qualify.

The gate freezes all 74 entries in `maps/`, fingerprint
`c93f589749c7ff4b922f8a42b8cd60368958a31910e20da1379fdd2a66b472b7`.
This hashes the sorted `frozen_maps` list of `{name, sha256}` entries using compact
sorted-key JSON. Any map addition, removal or byte change requires a new explicit
inventory version. The full specification is in [all-maps-victory.md](all-maps-victory.md).

| Stage | Maps | Seeds | Games per opponent–map | Required result for **each** opponent–map | Total games |
| --- | --- | --- | ---: | --- | ---: |
| Standard qualification | All 74 | 0, 2026 | 4, both seats | Score ≥60%, and ≥50% in each seat | 3,552 |
| Fresh-seed confirmation | All 74 | 90101, 90107, 90121, 90133 | 8, both seats | Score ≥60%, and ≥50% in each seat | 7,104 |

All **888 opponent–map pairs** must pass each stage independently with the same
checkpoint and evaluation policy. With decisive games, standard requires at
least **3/4** wins and at least **1/2** in each seat; confirmation requires at
least **5/8** wins and at least **2/4** in each seat. Draws earn half credit, so
standard's attainable minimum score is 2.5/4. No pooling across maps, opponents
or stages can rescue a failure. Every expected cell needs a valid terminal
result, with zero unresolved infrastructure errors. A timeout or bot crash is
not a candidate win. A 20/24 smoke result is not all-map qualification.

The former 192-game development and 384-game disjoint-map gate is superseded
by this user-requested expansion; any historical results retain their original
scope. Both revised stages cover all maps, so confirmation is **fresh-seed**,
not unseen-map. The four originally reserved maps remain excluded from gradient
training as an additional restraint, and benchmark trajectories never enter
replay. Preserve unused confirmation seeds through training and development.

Freeze the checkpoint, policy configuration, map inventory, panel and
engine/patch/bridge hashes before the first confirmation game. Persist that
identity in a confirmation ledger. Only that candidate may finish interrupted
cells; never rerun a successful cell to choose a favorable result. One candidate
may attempt confirmation per version. A failed confirmation consumes its
four-seed set. A later attempt requires a new gate version with four predeclared
unused seeds across the same complete current map inventory, not disjoint maps.

This is an empirical requirement on the frozen schedule, not proof of true
superiority on all maps. Each opponent–map pair has only two standard and four
confirmation seed pairs, and there are 888 such comparisons. Report every pair's
scores, both seats and coverage. Any bootstrap must resample complete two-seat
seed pairs and be labeled descriptive; small sample sizes, shared maps and
checkpoint selection prevent a simultaneous 95% superiority claim.

## Finite, recoverable research cycles

The [terminal-root-ranking-v4 comparison](root-ranking-v4.md) completed
418 valid terminal continuations at eight frozen roots, with no gradients.
Its independently verified mean paired validation advantage was 0.21875,
with two improving roots and worst difference −0.25, passing the precommitted
mechanism rule. This is not benchmark improvement.
The recorded review selects [terminal-target-head-v5](terminal-target-head-v5.md):
fresh symmetric self-play, matched terminal-ranking/PUCT labels, only the
existing policy output layer trained, fresh validation inputs, and all-12-bot
smoke checks. That protocol is published for implementation in the next cycle.
The limits below do not authorize blindly repeating the stagnant warmstart
configuration or treating the diagnostic's continuations as training data.

Each cycle is bounded by four training iterations, eight complete games per
iteration, and **three hours** under the user's current instruction. These are operational limits on one
recoverable job, not tournament restrictions and not an overall limit on the
authorized research. Measure throughput before increasing concurrency. Use cheap
small-map subsets for most checkpoint comparisons. Automatic full standard
qualification is screened by a complete smoke evaluation with at least 50%
against **each** of all twelve opponents, not by merely reaching four weak
iterations. An explicit periodic research review may also schedule an occasional
all-map check to examine coverage and weaknesses. Neither route changes the
final gate, and partial diagnostics never count as a pass. A complete passing
3,552-game standard stage is required before the 7,104-game confirmation.

Persist each completed episode, its map/seed/action transcript, generation
checkpoint, terminal outcome and retained training examples before starting the
next episode. Save model, optimizer, RNG state, replay episode identities and
configuration atomically at recoverable boundaries. An interrupted episode
must resume deterministically or be recorded as incomplete and excluded from
training. Restarting must not duplicate completed evaluation cells or combine
different runtime identities into one apparent sample.

At each cycle boundary publish the configuration and hashes, number of completed
and failed games, replay diversity, calibrated value diagnostics, per-opponent
scores with coverage, throughput, elapsed time and the next concrete hypothesis.
Use run IDs and schema versions on graphs so results from different models are
not silently connected as one learning curve. Keep all raw evidence and label
failed experiments. Continue with another finite cycle after reviewing the
measurements; do not leave an unbounded silent process or promise that a fixed
compute budget will necessarily reach the victory gate.
