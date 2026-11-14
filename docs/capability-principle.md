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
| `moveForward()` | via `move(facing)` | none: in engine 1.2.6 `moveForward()` is exactly `move(getDirection())` |
| `buildRat`, `placeDirt`, `removeDirt`, `place/removeRatTrap`, `place/removeCatTrap`, `carryRat` | yes, at any tile within ±3 | none: every engine action range is distance² ≤ 9, so the 7×7 target window covers all of them |
| `attack(loc)` | yes | — |
| `attack(loc, cheeseAmount)` | **no**; recorded as a plain bite | **missing**: cheese-boosted bites. zdb uses them in fights (`Baby.java:970`, `Baby.java:1017`, `GrabMicro.java:308`) |
| `pickUpCheese(loc, amount)` | full pickup only | **missing**: the amount |
| `transferCheese(loc, amount)` | always transfers all | **missing**: the amount |
| `throwRat`, `dropRat(dir)`, `becomeRatKing` | yes | — |
| `squeak(int)` | **no**; ignored by the hook (117k zdb calls in teacher-v5) | **missing**: 32-bit message content |
| `writeSharedArray(index, value)` | **no**; ignored by the hook (1.39M zdb calls in teacher-v5) | **missing**: 64 indices × 10-bit values |
| `disintegrate` | no | missing (rarely useful, but cheap to include) |
| `resign` | no | deliberately excluded: it only forfeits the game |
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

   Each parameter is a vector of independent bits. That covers every integer value, so any API argument is reachable. Add `disintegrate`. Raise the per-turn step cap.
2. **Writable private memory.** Each robot gets registers (planned: 32 × 10 bits) that it can write as actions and reads back in its observation. This is the network's equivalent of a bot's variables. Imitation can't label them, because zdb's internal state isn't observable, so they start unused and RL can learn to use them.
3. **Complete observation:**
   - the raw shared array;
   - heard squeaks (up to 16 messages from the last 5 rounds: content bits, sender offset, age);
   - full `RobotInfo` per sensed robot;
   - own ID bits and the missing getters;
   - full-resolution map memory (60×60 planes, read by a small convolutional encoder in place of the 16×16 summary).
4. **Imitation benefit.** zdb's squeaks, shared-array writes and boosted bites become ordinary training labels, so the student can learn zdb's own communication protocol instead of a hand-made substitute.
5. Recollect teacher data, retrain, and pass the gate. **RL starts only from a model built to this spec.**

## Implemented (November 13, 2026): student-actions-v2 / student-obs-v4 / student-mlp-v2

**Actions (664 in actions-v3; 665 in v2, which also had `RESIGN`).** The v1 actions plus:
- `SQUEAK`, 64 `WRITE_SHARED i` (kings only, as in the engine), 32 `WRITE_REG i`, `DISINTEGRATE` (no `RESIGN`; see below).
- A parameter is chosen after the action, as independent bits from a head conditioned on the trunk and on the chosen action:
  - 16-bit amount for attack, pickup and transfer. 0 means the default: a plain bite, the whole tile, or all raw cheese. Amounts above what the engine allows are clamped.
  - 32-bit squeak content.
  - 10-bit value for shared-array and register writes.
- At most 128 primitives per turn (zdb writes about 1.4 shared-array slots per king turn).
- `DISINTEGRATE` is in the space, but a student team is only offered it with `Game --end-actions 1`: an imitation net would pick it by accident about once per game. RL can switch it on.
- `resign()` is deliberately not an action (student-actions-v3, 664 actions). It does nothing but forfeit the game: every robot on the team is destroyed and the opponent wins by `RESIGNATION`. A capability-complete bot loses nothing by never being able to call it. actions-v2 records (665 actions) still load: the layout is identical apart from that trailing index, which was never a label.

**Observation (9,081 bytes).**
- Dense values:
  - the local window grows to 25 channels per tile, adding exact facing, raw cheese, what the robot carries, cat chirality, ID and exact HP;
  - exact cheese amounts per tile;
  - scalars add own ID, turning cooldown, `getAllCheese`, exact HP/cheese/round/position, the carried robot's details, whether it squeaked this turn, and the step counter;
  - metadata for up to 16 heard squeaks: sender offset, absolute source, age and sender ID.
- Bits:
  - full-resolution map memory, 60×60 tiles × 8 planes: seen, wall, dirt, mine, cheese, rat trap, cat trap, self;
  - the raw shared array (64 × 10 bits);
  - 32 private registers × 10 bits;
  - the content of the heard squeaks (16 × 32 bits).
- The obs-v2 team board and obs-v3 enemy memory stay as extra inputs.

**Network.**
- The map bits go through their own 64-unit layer, computed sparsely over the bits that are set.
- That output joins the dense values and the other bits in the 384→256 trunk, which feeds the policy, value and parameter heads.
- Java/PyTorch parity is tested for the policy, the value and the parameter logits.

**Recording.** Every zdb controller call now maps to a label: squeaks, shared-array writes, boosted bites and amounts included. A 400-round smoke game had no unmapped or ignored calls.

Pipeline: [scripts/pipeline_obs_v4.sh](../scripts/pipeline_obs_v4.sh) runs teacher-v9 (500 pure zdb games), student-v9, a DAgger round (teacher-v10-dagger, student-v10), the zdb check and the frozen gate. The trainer now streams records in chunks of games, because obs-v4 records are about 9 KB each.

## Update: student-mlp-v3 (November 13, 2026)

Capability includes being able to *use* the actions well enough to learn from: student-v9 could write any shared-array value,
but wrote its king's position correctly only 2–17% of the time, so its babies could not find the king and it starved
([results](imitation-results.md#student-v9-obs-v4-capability-complete-132-against-zdb-the-king-starves-november-13-2026)).
mlp-v3 gives the parameter head binary copies of the integers values are usually built from (`Net.ctx`) and a second hidden
layer. This adds no restriction: every bit is still chosen freely.

## Update: zdp pass 3 (November 14, 2026)

The decision-point line (zdp, [decision-points.md](decision-points.md#pass-3-capability-completion)) now covers the
gaps the pass-2 points left: free shared-array writes and squeaks (KING_COMMS, KING_VAL, BABY_COMMS C_RAW/C_RELAY,
BABY_VAL), pickup and trap removal (BABY_FREE), strafe and risky steps, bite amounts, drops, transfers and
disintegrate, king facing (KING_TURN) and a keep/review role. Still open: per-robot writable memory registers, babies
reading the whole shared array (only BABY_ROLE and KING_COMMS see all 64 slots), more than the newest heard squeak as a
feature, the comms block on BABY_FACE / MINE_PICK, king pickup control, bite and transfer amounts as ladders rather
than every integer, `resign` (excluded on purpose) and bytecode budgets for the hooks.
