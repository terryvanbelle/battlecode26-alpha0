# Victory across all available maps

The success claim is now: **one immutable candidate passes every one of the 12
frozen opponents on every one of the 74 available maps, in both development
qualification and fresh-seed confirmation**. A good average over maps,
opponents, or stages cannot compensate for a failing opponent–map pair.
Earlier smoke and four-map evaluations remain useful development evidence;
they do not satisfy this expanded gate.

This document specifies schema 2, gate `field-twelve-v2-all-maps-v2`, in the
versioned [victory configuration](../configs/victory_gate.json).
[`alpha0.victory`](../alpha0/victory.py) evaluates its `standard` and
`confirmation` stages separately from the legacy evaluator's four-map gate.

## Frozen scope and provenance

The inventory checked on 2026-11-10 contains exactly **74 `maps/*.map26` files**.
Names are unique, including under case folding, and all file contents match
[maps/SHA256SUMS](../maps/SHA256SUMS). The complete names and per-file hashes are
in that manifest; neither filename filtering nor a short list of easy maps
defines the gate.

| Frozen input | SHA-256 |
| --- | --- |
| Exact `maps/SHA256SUMS` file bytes | `32999a7d563a1f6b9d2b143b9f604e1521b458dc6ddcafa63218c0cbaf6dbbe2` |
| Canonical map inventory | `c93f589749c7ff4b922f8a42b8cd60368958a31910e20da1379fdd2a66b472b7` |
| Frozen `field-twelve-v2` opponent panel | `f18d20f65e086a216c53613e8aaf1de6b714c0a5cea8c1ffcf5505ce72c3a43d` |

The canonical map fingerprint hashes UTF-8 JSON of a list sorted by the exact
map name, with each entry `{"name": name_without_extension, "sha256": file_hash}`,
using Python `json.dumps(entries, sort_keys=True, separators=(',', ':'))`.
The panel fingerprint follows the existing gate definition: sorted opponent
IDs with their pinned source commit and Java tree hash. The twelve identities
remain those in [the benchmark manifest](../configs/benchmarks.json).

Freeze the full map set and byte hashes before scheduling. A missing, modified,
duplicate, or additional available map requires explicit inventory review and
a new gate version; it must not silently change an existing ledger. The old
version still describes its original 74-map experiment. A later claim about
*all then-available maps* must pass the expanded inventory. In particular,
discovering a difficult new map does not justify ignoring it or substituting a
map on which the candidate has already won.

## Exact operational criteria

For each opponent–map pair separately, score a valid win as 1, a valid draw as
0.5, and a valid loss as 0. There are **888 distinct opponent–map pairs**.
The explicit seed arrays and candidate evaluation settings are frozen in the
gate configuration before running a stage.

| Stage | Seeds on each of 74 maps | Seats per seed | Games per opponent–map | Minimum total score per opponent–map | Minimum score in each seat | Total games |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard qualification | 0, 2026 | A and B | 4 | 60%, hence at least 2.5/4 attainable score | 1/2 | 3,552 |
| Fresh-seed confirmation | 90101, 90107, 90121, 90133 | A and B | 8 | 60%, hence at least 5/8 attainable score | 2/4 | 7,104 |

With decisive games these thresholds require **at least 3/4 wins** in standard
and **at least 5/8 wins** in confirmation, also meeting each seat's 50% floor.
Actual draws receive half credit: two wins, a draw and a loss score 2.5/4 and
can pass standard if the seat floors also pass. Four draws do not pass.
Use exact half-score units for the threshold checks: standard needs at least
5 units out of 8, and confirmation at least 10 out of 16. Do not round a
displayed percentage to decide a pass.

All 888 pairs must pass **each stage independently**, with the same checkpoint,
policy, search budget, temperature and engine identity. Standard and
confirmation scores cannot be pooled to rescue a failure. Opponent-level
averages, map-level averages and the overall average are descriptive only.
An especially favorable seat cannot rescue the other seat's score below 50%.
The full qualifying experiment therefore contains **10,656 valid games**.

Most evaluations use inexpensive small map subsets. The campaign automatically
schedules standard for a candidate with a complete smoke evaluation scoring
at least 50% against every frozen opponent. An explicit periodic research
review may also request an occasional full-map check; no full scan is required
at every checkpoint. This screen saves evaluation budget; it is not a
substitute for all-map qualification.
The 20/24 winner-only initialization and its first two MCTS checkpoints still
lose both smoke seats to two opponents and do not meet the screen.

## Fresh-seed confirmation and the four reserved maps

Disjoint-map confirmation is impossible when both stages must cover every
available map. The revised confirmation tests **fresh random seeds on the
same complete map inventory**. It must be named fresh-seed confirmation, never
unseen-map confirmation. Seed arrays must be distinct within each stage and
disjoint between stages. Before opening confirmation, check its seeds against
recorded self-play, development, and earlier confirmation usage; reserve fresh
integers, not just renamed seed entries. Newly generated training jobs must
not consume the reserved confirmation seeds.

Keep `Stash`, `Hike`, `heart` and `ratsareants` excluded from gradient training
as an additional restraint. They remain part of **both** all-map evaluation
stages, so their geometry is no longer an unseen development holdout. The
no-gradient rule applies to checkpoint ancestry, warmstarts, replay and other
supervised training data, not merely the current configuration's map list.
All benchmark trajectories remain excluded from training on every map.

Only a candidate that completes and passes the full standard stage may open
confirmation. Before its first confirmation game, persist a lock containing
the candidate checkpoint hash, gate/config hash, map inventory hash, frozen
panel hash, inference policy settings and engine/runtime identity. Exactly
**one candidate may attempt confirmation in each version**. Publishing a
new checkpoint while confirmation runs must not replace the locked candidate.

If that candidate fails, the confirmation version is consumed. Its results
may inform subsequent development, but a later independent confirmation needs
a new version with four newly predeclared, unused seeds on **all 74 maps**.
There is no requirement to invent a disjoint map set, and no permission to
reuse a failed seed set under a new label. Preserve the failed ledger and
report how many confirmation attempts preceded any eventual pass.

## Completion, recovery and failure cases

Every expected opponent–map–seed–seat cell needs one unambiguous valid terminal
result with matching stage, checkpoint, policy, runtime and input hashes.
Coverage is exact set equality, not merely the expected number of rows.
Duplicate rows cannot fill a missing cell; conflicting successful results
for the same identity require investigation. Reject unknown scores, NaNs,
invalid terminal games and records from a different map hash or candidate.

Infrastructure errors, worker timeouts and interrupted jobs leave cells
incomplete. They do not become wins, draws or ordinary losses. Retry only
those failed infrastructure cells with identical identities, preserving all
attempts. Keep the first valid published result; never replay a successful
cell until a favorable outcome appears. A legal in-game loss or resignation
remains a valid loss. Resume a partial stage using its locked candidate and
schedule, rather than starting a fresh denominator.

A map-dependent benchmark crash does not justify dropping that map, shrinking
the panel or calling a default win. Validate the harness and reproduce the
failure under the documented panel-change policy. Any justified opponent
replacement creates a new panel/gate version and requires the complete gate
again. A strong but functioning opponent is never a replacement candidate.

An evaluation may stop early once a valid result makes its stage impossible
to pass. Such a stage is **failed/incomplete**, never passed; an opened
confirmation remains consumed. Otherwise use finite recoverable batches with
persisted results and explicit remaining coverage, within the user's current
maximum three-hour invocation budget. Full stages may span multiple cycles.
A deadline can postpone
the remaining work but cannot relax the gate. Full completion is required
for a success claim even if every completed pair currently passes.

## Fairness and stochastic interpretation

Each map has the same number of seeds and both candidate seats. Pair the two
seats on the same map and seed, and record per-seat scores. This reduces some
starting-position bias; it does not make the two games independent or ensure
that a map is symmetric. Use identical documented local engine and bytecode
settings for candidate and opponent throughout. Fast maps must not receive
more weight because more of their games finish within a wall-clock budget.

The gate is an empirical requirement on the frozen schedule, **not proof of a
true win probability above 60% on every map**. Each opponent–map pair has only
two development seed pairs and four confirmation seed pairs. There are 888
such pairs, checkpoint selection uses standard results, and repeated
confirmation versions introduce further selection. Opponent randomness may
also survive engine seed control. Hashes identify what ran; they do not turn
random outcomes into statistical guarantees.

Report exact win/draw/loss counts, scores, coverage and both seats for every
pair. If intervals are shown, preserve each seed's two-seat pair when
resampling and label them descriptive; with two or four pairs, bootstrap
intervals are particularly unstable. A pooled interval cannot certify all
888 opponent–map cases. A failed gate establishes that the predeclared
operational requirement was not met, not that the candidate is universally
weaker. Any future increase in sample size must be declared before observing
the new results, with no retrospective threshold or map changes.
