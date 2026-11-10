# Terminal-head v5: source coverage failure

The bounded v5 run stopped with **24 of 32 required training roots**. All eight missing roots belonged to `closeup`. It performed **zero gradient updates**. This diagnosis reads completed source records and their lossless feature caches; it did not launch games, replay an engine, or run model inference. The machine-readable evidence is [coverage-diagnosis.json](../progress/terminal-target-head-v5/coverage-diagnosis.json).

The evidence supports a frozen-policy/action-abstraction deadlock. The collector retained complete terminal games and correctly applied the preregistered reserve allocation. This is a coverage failure, with no claim about learned strength or head fitting.

## What completed

| Map | Complete source games | Accepted training roots |
|---|---:|---:|
| `tiny` | 4 | 8 |
| `DefaultSmall` | 4 | 8 |
| `closeup` | 8: four initial, four reserves | 0 |
| `knifefight` | 4 | 8 |
| Total | 20 | 24 / 32 |

The accepted roots came from 12 source games. Validation-source discovery, terminal target scoring, and SGD were not reached. The experiment identity is `f28ce105ff9c6e9da536aef14e1686ef70c401739c5bdceb24a5d596928d1313`; the incomplete training-manifest SHA-256 is `75661c3cab6f36278954d95e250f519c35e6597ea545522b8cec28203127e402`.

Seven closeup games reached round 1,310 after 2,620 decisions. Their terminal banks and king counts were both `[0,0]`, and only the two cats remained. No baby rats were built. The eighth game, `training-m2-j6`, escaped the deadlock and must be described separately.

| Source | Decisions | Round | Action histogram |
|---|---:|---:|---|
| `training-m2-j1` | 2,620 | 1,310 | `21:2619, 19:1` |
| `training-m2-j2` | 2,620 | 1,310 | `21:2619, 23:1` |
| `training-m2-j3` | 2,620 | 1,310 | `21:2619, 1:1` |
| `training-m2-j4` | 2,620 | 1,310 | `21:2619, 4:1` |
| `training-m2-j5` | 2,620 | 1,310 | `21:2618, 7:1, 39:1` |
| `training-m2-j6` | 2,218 | 414 | `21:1795, 18:369, 27:38, 20:4, 19:2, 29:2`; one each of `0,3,4,5,6,8,23,38` |
| `training-m2-j7` | 2,620 | 1,310 | `21:2619, 1:1` |
| `training-m2-j8` | 2,620 | 1,310 | `21:2619, 4:1` |

The action transcript identifies the rare non-assault decisions exactly. A non-assault action at the final decision, after the other king has died, is not evidence of useful early exploration. Per-source engine/policy seeds, complete histograms, terminal hashes, candidate records, and sampled legal masks are in the diagnosis JSON.

## Why assault did nothing

Action 21 is `assault_king`. Its intent mask permits it when a mobile robot has an enemy king to target; it does not require an achievable path. The macro's route search considers the king's full 3×3 footprint. If no reachable approach goal exists, this action can consume a turn without moving or attacking. See [MacroV2.java](../java/alpha0/MacroV2.java), particularly the legal-action mask, `Routes.bestCell`, `toward`, and action 21.

A static calculation on the existing `training-m2-j1.npz` global wall plane confirms that geometry matters. Starting from either king center, `(1,1)` or `(28,28)`, there are exactly **57 wall-free reachable king centers**. This deliberately generous calculation ignores dirt and robot occupancy, and allows all eight neighboring directions. The closest such center remains squared distance **514** from the enemy king's nearest footprint tile; the official king attack radius squared is **8**. A one-tile actor can connect the corners, but a king cannot approach closely enough even in this relaxed graph.

Dirt also surrounds the initial king footprints and blocks immediate spawning. Near team A's start, rows `y=0..5` are:

```text
KKKdd.
KKKdd.
KKKdd.
ddddd.
ddddd.
######
```

Here `K` is the king footprint, `d` dirt, and `#` wall. The 224 retained reservoir anchors from the seven stalled games are all king turns at the starting centers. None permits spawning actions 26/27 or attack action 37. The collector's chronological scan found no eligible combat root in those games; absence of baby roots is consistent with the full action transcripts, rather than a sampling or serialization omission.

The end at round 1,310 is explained exactly by official resource consumption: 2,500 initial cheese / 2 cheese per king-turn = 1,250 fed turns, followed by 600 HP / 10 damage per unfed turn = 60 starvation turns. The recorded `WON_BY_DUBIOUS_REASONS` is the official terminal tie-break after both kings die, using the already pinned seeded-RNG patch. These are completed games, not watchdog expirations or fabricated terminal rewards.

## Temperature suppressed the available alternatives

The 224 stored anchors were selected by the existing uniform source/team reservoirs. Their exact input hashes were recomputed from lossless float32 arrays and legal masks, and their cache hashes were verified. The following are **sampled-state probabilities**, not averages over every decision:

| Action probability | Mean | Range |
|---|---:|---:|
| Frozen legal-normalized prior for assault 21 | 0.77249 | 0.69871–0.85054 |
| Sampling probability for assault 21 at T=0.35 | 0.999775 | 0.999364–0.999945 |
| Frozen legal-normalized prior for clear-route 38 | 0.013501 | 0.007703–0.018156 |
| Sampling probability for clear-route 38 at T=0.35 | 0.000010676 | 0.000001490–0.000029307 |

For example, at `training-m2-j1` decision 59, round 30, team B's king remains at `(28,28)`. Its assault prior is about 0.76562; sharpening at T=0.35 raises the sampling probability to about 0.99977. At T=1, the legal-normalized prior itself is the action distribution, leaving materially more probability for dirt clearing and other alternatives. This comparison is arithmetic on recorded priors; no T=1 game result or coverage guarantee is implied.

## Reserve j6 is not an allocator bug

`training-m2-j6` selects clear-route action 38 at decision 674 and first selects build action 27 at decision 678. It subsequently produces babies and ends at round 414 with team A winning by `KILL_ALL_RAT_KINGS`. Its qualifying candidates are:

| Candidate | Decision index | Round |
|---|---:|---:|
| `RAT_KING:1` — team B king | 1,487 | 393 |
| `BABY_RAT:0` — team A baby | 1,522 | 394 |

However, **j6 is an even reserve**, assigned to the still-missing even group j2. That group requires `BABY_RAT:1` and `RAT_KING:0`: team B baby and team A king. Neither is available. The two actual candidates form the opposite, odd pair. Accepting them for a different group after seeing the trajectory would violate the fixed allocation. The collector and allocator correctly left the prescribed slots unavailable.

## Recommended amendment

Use a separately identified **v6** source-coverage protocol with **T=1.0 for both teams on all four maps**, for both training and validation source discovery. Regenerate the complete paired source/reserve seed inventory under that identity. Preserve v5 as failed-coverage evidence and do not reuse its caches or fit on its incomplete 24-root set.

Keep the frozen checkpoint, macro definitions, official engine physics, 32 training / 16 validation quotas, role/team/map strata, deterministic reserve assignment, canonical gradient-input exclusion, and 512-anchor requirement. Keep terminal selection and validation continuations at **T=0.35**, with the same comparator and fitting protocol. Coverage must pass a bounded discovery stage before target scoring or SGD.

Increasing reserves at T=0.35 would tend to repeat the observed suppression of exploration. Relaxing attack-action eligibility alone cannot produce baby roots in games with no babies. Changing assault's fallback behavior would change action semantics and confound this experiment; source exploration can be amended while keeping those semantics fixed. T=1.0 is a reasoned remedy to test, not an asserted success.

## Source record fingerprints

These JSON records remain under `runs/terminal-target-head-v5/sources/`; the recovery archive retains their transcripts and caches. Their SHA-256 values are also in the published diagnosis JSON, together with each cache SHA-256.

| Source JSON basename | SHA-256 |
|---|---|
| `training-m2-j1.json` | `273ebbf0173ed84f0e0bfb241e459874b3acf58e614a421fbe350811cbbf1c4b` |
| `training-m2-j2.json` | `43d24d4994c9e016ae1ad29afa19be1c869f753b80a355fdb67f1301f7624b74` |
| `training-m2-j3.json` | `285938c6073cd0d034ad660e60000651f88cfe0c45fb9d9c086fd490fddd3ae3` |
| `training-m2-j4.json` | `e2e3bce39fc0769907c53d59a557fefd07e0022c65a76fd33b7f2406092a05ae` |
| `training-m2-j5.json` | `a66f7889cc7ebab7c9a57380e77c6b7d06f8aef6a9c242c92f36c9384fd3a594` |
| `training-m2-j6.json` | `bcddab707fe41adf72c36060d8e07c2d4b798e9274e4c1f95ac32e4fd8a7265d` |
| `training-m2-j7.json` | `994164fc3b2addb952ddd9d6158d06560e036013a5a5fbc095759953982acb3d` |
| `training-m2-j8.json` | `151a0caa6c127774c6cf2cd7bd6fc144c2676afe65e868db87c3b84b3ed399ac` |
