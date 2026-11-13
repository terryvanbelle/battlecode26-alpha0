# Design principle: the RL starting model must be capability-complete

**Principle (user direction, November 13, 2026).** The network that starts RL must be at least theoretically able to do anything a hand-coded bot can do. If the student lacks an action or an input that a hand-coded bot has, RL can never discover a strategy that needs it, and could be blocked from surpassing the hand-coded bots.

How this applies:
- The reference is the **full game API** (`RobotController`, engine 1.2.6), not just the calls zdb happens to make.
- Hand-engineered inputs, such as the obs-v2 team board and the obs-v3 enemy memory, may stay as extra features because they are computed from legal information. They never replace the raw capability.
- Any gap left open on purpose is listed below with the reason.

## Audit of student-actions-v1 / student-obs-v3

### Actions

| API call | Student today | Gap |
|---|---|---|
| `move(dir)`, `turn(dir)` | yes (8 directions each) | — |
| `moveForward()` | via `move(facing)` | to check: whether the cooldown is the same as `move` toward the facing direction |
| `buildRat`, `placeDirt`, `removeDirt`, `place/removeRatTrap`, `place/removeCatTrap`, `carryRat` | yes, at any tile within ±3 | none: every engine action range is distance² ≤ 9, so the 7×7 target window covers all of them |
| `attack(loc)` | yes | — |
| `attack(loc, cheeseAmount)` | **no**; recorded as a plain bite | **missing**: cheese-boosted bites. zdb uses them in fights (`Baby.java:970`, `Baby.java:1017`, `GrabMicro.java:308`) |
| `pickUpCheese(loc, amount)` | full pickup only | **missing**: the amount |
| `transferCheese(loc, amount)` | always transfers all | **missing**: the amount |
| `throwRat`, `dropRat(dir)`, `becomeRatKing` | yes | — |
| `squeak(int)` | **no**; ignored by the hook (117k zdb calls in teacher-v5) | **missing**: 32-bit message content |
| `writeSharedArray(index, value)` | **no**; ignored by the hook (1.39M zdb calls in teacher-v5) | **missing**: 64 indices × 10-bit values |
| `disintegrate`, `resign` | no | missing (rarely useful, but cheap to include) |
| indicator and timeline calls | no | no game effect; excluded |
| Turn length | at most 12 primitives per turn | must be long enough for the shared-array writes a king makes in one turn |

### Observation

| Information a bot can get | Student today | Gap |
|---|---|---|
| Map tiles in vision | yes; the local window (radius 5) covers the largest vision (king, distance² 25) | — |
| `RobotInfo` of sensed robots | type, team, health, and "facing me" | **missing**: exact direction, raw cheese carried, what it carries, cat chirality |
| `readSharedArray` (64 × 10 bits) | no; obs-v2 board is a fixed substitute | **missing**: the raw array |
| `readSqueaks` (last 5 rounds: content, sender, round, source location) | no | **missing** |
| Own `getID`, `getTurningCooldownTurns`, `getAllCheese` | no | **missing** (bots use IDs to assign roles) |
| Other getters (health, cheese, cooldowns, cost, traps, dirt, cooperation, backstab, map size, round) | yes | — |

### Memory and computation

| A hand-coded bot can… | Student today | Gap |
|---|---|---|
| keep any per-robot state across turns (roles, plans, timers) | fixed features only: map flags, kings, enemies from the last 6 rounds | **missing**: writable memory |
| remember the exact map | a 16×16 summary beyond the 11×11 local window | **missing**: full-resolution map memory (maps are at most 60×60) |
| run iterative algorithms (pathfinding, search) within 17.5k–20k bytecodes per turn | one fixed-depth network pass per primitive | partly covered: multi-step turns plus writable memory allow iterative computation in principle. This residual gap is accepted. |

## Plan: student-actions-v2 / student-obs-v4

1. **Parametric actions.** Some actions get a parameter head conditioned on the state and on the chosen action:
   - attack, pickup and transfer: a cheese amount;
   - squeak: a 32-bit message;
   - shared-array write: an index action plus a 10-bit value;
   - a write to the robot's own memory register.

   Each parameter is a vector of independent bits. That covers every integer value, so any API argument is reachable. Add `disintegrate` and `resign`. Raise the per-turn step cap.
2. **Writable private memory.** Each robot gets registers (planned: 32 × 10 bits) that it can write as actions and reads back in its observation. This is the network's equivalent of a bot's variables. Imitation can't label them, because zdb's internal state isn't observable, so they start unused and RL can learn to use them.
3. **Complete observation:**
   - the raw shared array;
   - heard squeaks (up to 16 messages from the last 5 rounds: content bits, sender offset, age);
   - full `RobotInfo` per sensed robot;
   - own ID bits and the missing getters;
   - full-resolution map memory (60×60 planes, read by a small convolutional encoder in place of the 16×16 summary).
4. **Imitation benefit.** zdb's squeaks, shared-array writes and boosted bites become ordinary training labels, so the student can learn zdb's own communication protocol instead of a hand-made substitute.
5. Recollect teacher data, retrain, and pass the gate. **RL starts only from a model built to this spec.**

The obs-v3 pipeline started at 04:41 PDT keeps running as a diagnostic: it shows whether enemy memory fixes small-map defence. It cannot produce the RL starting model.
