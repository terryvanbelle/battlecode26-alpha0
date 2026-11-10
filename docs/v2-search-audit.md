# V2 search audit: first self-play generation

This read-only audit covers the **eight completed generation-0 self-play games** used by campaign iteration 1. It does not assess the whole campaign or benchmark strength. No benchmark code, trajectories or outcomes were used. Running training code, configuration and engine files were left unchanged.

## What the data shows

- **42,230 decisions; 6,927 searched roots (16.40%); 4,096 retained targets.** Every game contributed its 512-position reservoir. All games used the same untrained v2 checkpoint against itself; no historical opponent was used yet.
- Retained roots have **21.34 legal actions on average** (median 21, range 12–31), but only **6.87 actions with a visit** (median 7, maximum 8). **67.82% of legal-action incidences receive zero target probability.**
- **86.06% of visited root edges have exactly one of eight visits.** Target entropy averages **1.871 nats** (2.699 bits), versus 3.052 nats for uniform distributions over the same legal masks. Mean largest target probability is 0.2423. The sharp targets mainly reflect a small visit budget; sharpness alone does not demonstrate good decisions.
- All stored targets sum to one, assign zero mass to illegal actions, and are exact multiples of 1/8. There are no malformed visit targets in this sample.
- Kings account for **637/4,096 retained positions (15.55%)**. King roots average 24.51 legal actions and 6.92 visited actions; baby roots average 20.76 and 6.86. King search is oversampled by design (35% search probability versus 15% for babies); the transcript does not record actor type at every unsearched decision, so those exact conditional coverage rates cannot be independently reconstructed here.
- Team A won six games and B won two. Player-relative retained terminal labels are **1,960 positive / 2,136 negative (47.85% positive)**. The initial class balance is healthy; these eight outcomes are not evidence of a seat advantage.

| Game suffix | Map | Seed | Winner | Rounds | Decisions | Searched roots |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| 000 | `tiny` | 1656982212 | A | 181 | 3,810 | 620 |
| 001 | `DefaultSmall` | 1344494084 | A | 137 | 5,657 | 881 |
| 002 | `closeup` | 1769747038 | A | 175 | 6,786 | 1,058 |
| 003 | `knifefight` | 305196229 | B | 140 | 3,531 | 641 |
| 004 | `tiny` | 477670760 | A | 167 | 4,209 | 691 |
| 005 | `DefaultSmall` | 1369528673 | B | 176 | 6,797 | 1,123 |
| 006 | `closeup` | 139961480 | A | 194 | 7,703 | 1,280 |
| 007 | `knifefight` | 1126652752 | A | 143 | 3,737 | 633 |

## Assault macros versus compass actions

These are selections, not counts of successful moves, spawns or attacks. Compass actions may only turn while movement is cooling down; calling all of them wasted movement would exceed the evidence. Whole-game selection shares and retained-root target statistics have different sampling denominators.

| Action | Selected in all 42,230 decisions | Retained roots where legal | Visited when legal | Mean target mass | Uniform-legal reference mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| 21: `assault_king` | 2,153 (5.10%) | 4,083 | 33.50% | 4.907% | 4.746% |
| 27: `spawn_assault` | 51 (0.12%) | 361 | 25.48% | 0.323% | 0.301% |
| 26: `spawn_economy` | 70 (0.17%) | 361 | 25.48% | 0.317% | 0.301% |
| 17: `gather_cheese` | 1,428 (3.38%) | 3,766 | 25.15% | 3.351% | 4.344% |
| 20: `guard_king` | 3,110 (7.36%) | 3,524 | 64.47% | 7.706% | 4.171% |

The eight compass actions, IDs 1–8, account for **14,500 decisions (34.34%)**. Compass build actions, IDs 9–16, were selected 392 times; economy and assault spawning were selected 70 and 51 times. Thus `spawn_assault` is 9.94% of the 513 spawn-labelled selections, while it is rarely legal in the overall robot-turn population. Neither action 21 nor 27 has target mass below its uniform-legal reference. The evidence supports **broadly sparse exploration**, not a specific bug suppressing these two macros.

## Why this generation contains little outcome-informed search

The source checkpoint’s final value-layer weights and bias are exactly zero, verified from its saved tensors. Its inferred value is therefore zero for every nonterminal position. In this generation, PUCT can obtain outcome information only by reaching a true terminal state; otherwise its choices are driven by the initial policy and root Dirichlet noise. This neutral start avoids the earlier saturated-value failure, but it means the first policy targets should not be described as demonstrated strategic improvements.

The games report 62,336 allocated search nodes, including one root per search. The maximum with eight simulations is 62,343. This does **not** establish search depth or terminal-leaf frequency: a first terminal expansion still creates a node. Root depth in decisions/rounds, terminal-leaf counts and root values are not stored. With eight simulations, a tree cannot reach the configured 12-decision depth cap anyway; and a search ply advances one chosen robot turn plus any forced turns, not a fixed number of game rounds.

The first optimizer summary reports value BCE 0.69254, MSE 0.99879, mean value −0.01933 and zero saturation. Those are measurements on training batches, not held-out calibration. They show the new value head has not immediately collapsed, but do not yet show a useful critic.

## Next controlled test, after this bounded run

Keep the active campaign unchanged. Once a trained checkpoint has been checked on whole held-out **self-play** episodes, freeze it and compare **8 versus 32 simulations** on the same 32 self-play states, balanced over the four training maps, robot type, early/late game and acting team. Reuse each root’s exploration seed and keep every other search parameter fixed. No benchmark positions enter this test.

Record legal/visited support, prior-to-target divergence, root value, terminal leaves, deepest reached round, chosen action, latency and engine transitions. Report action-21/27 visitation conditional on legality, alongside all other actions. If 32 simulations mainly broaden zero-value visits without reaching useful outcomes, more search alone is not the fix: prioritize critic calibration and temporal credit assignment. If it exposes useful alternatives, test a separately versioned training configuration under a matched wall-time budget before adopting it. Action-group priors or fewer redundant compass alternatives are later ablations, not changes justified by this sample alone.

## Exact audit inputs

Paths below are relative to `runs/campaign/v2/games/`; `.attempt-*` duplicates were excluded. Every NPZ hash matched its canonical game JSON’s `data_sha256` before analysis. Only policy, legal-mask, scalar and player arrays were read; game actions supplied the full-trajectory counts.

- Source checkpoint SHA-256: `936b66b3435dc42ba028225c7732fc32d4d769378530e3d8e14b79b00a30b963`
- Training-runtime fingerprint: `60baf63fb6e74aecbd4c3d93e77f264a32d7a963409ea326be90ec217c031d5c`
- `runs/campaign/v2/plan-0001.json` SHA-256: `7a0fe2ba54a089a638b50361aab0075c9f2734a90e05d8aceb992928fb8ea394`

```text
c96e7ddac0c85eacee51cea34057296462dbbd2ea43d6d01bae6165ac94834fc  iter-0001-game-000.json
33c8c5821c548282d980ee08bdc2e5e980a0d2f0659ebc03d46e9ae13ca78b88  iter-0001-game-000.npz
8cc6cd32f270521fcbfafd0e95a66fd97140280c42f59ebdc7bc23c96033245c  iter-0001-game-001.json
853f08da8cdf76e5b7a75593fbe8166adc98bd98cd2f35f0ec8dc049b22949af  iter-0001-game-001.npz
fab612f5361c33beab74dedff4b9c3911bd0b0f9f8f8b4fb5b87b0d6cb21eeff  iter-0001-game-002.json
c12aa4fa2d019ea809fc084f093d3377f13e4e00894462db88c4e0ea0311c6cc  iter-0001-game-002.npz
5b8fd28b13ad43e9d33407100e425f61bf0f4dd3b9559b6c32e157a1bf893b8d  iter-0001-game-003.json
568be1deb078cd83816222c8ec5b9160301b5469598764a33be3088ae3206046  iter-0001-game-003.npz
d360d45901d23a674cdf9fceeded5d3e381726834ff43f3d59d20cd75f10bebc  iter-0001-game-004.json
a84e4584bb414ef35fe556094915ccdbd4cad8a24332c21d22b8ec5134805063  iter-0001-game-004.npz
28804e41ba35b66920f5f7f610f959838d0053f928b933a6f47a5ce90f4af76b  iter-0001-game-005.json
7c577d79141d0f19eb1c209cbf9f484ed68bfa0caf3858a0b07ac69026965c36  iter-0001-game-005.npz
1f448ea259c5f2579598ca15c7af824a006826b4965fc0d1b61072c351860069  iter-0001-game-006.json
1138281c3adb68d891c216e60879919fecb26341bddd310b69a9f30edbe07c55  iter-0001-game-006.npz
364dd4f1458e726be64d7d145627fff070073876a446fde9aa052be80e3d3629  iter-0001-game-007.json
c0e43694aad3d26247911a54dae280721ddcb8a98007fc1ef1c20f6c2a082c25  iter-0001-game-007.npz
```
