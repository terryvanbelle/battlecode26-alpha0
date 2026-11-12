# Lean zdb imitation (student v1)

User direction (November 12, 2026): bootstrap RL from a model that imitates zdb well enough to be a competent starting point; exact fidelity is not required. This lineage replaces the exact-equivalence recorder programme in `docs/zdb-imitation-next-experiment.md` for practical purposes. The historical diagnostics there remain preserved but are not prerequisites.

Compute: GCloud VM `battlecode-dev` (user-authorized for this project on November 12).

## Pipeline

| Step | Code | Notes |
| --- | --- | --- |
| Build | `scripts/build_student.py` | Wraps every public primitive in `RobotControllerImpl` with `alpha0.student.Hook` calls; compiles with `java/student/**` into `build/student/classes` (atomic swap). |
| Game harness | `alpha0.student.Game` (`scripts/student_java.sh`) | One full game; each team is a sandboxed bot (`bot:PKG:CLASSES`) or the student net (`net:WEIGHTS[:T]`). Optional decision recording. |
| Teacher collection | `scripts/collect_teacher.py configs/teacher-v1.json` | zdb vs itself (30%) and 12 field bots; 63 maps (excludes ancestral four and the eight development-v4 maps); official bytecode limits; every 10th game is validation. Resumable. |
| Training | `python -m alpha0.student.train` | Behaviour cloning with legal-action-masked cross-entropy, king records upweighted, value head on game outcome. Exports Java weights. |
| Evaluation | `scripts/eval_student.py` | Any player vs opponent groups on maps × seeds × sides. |
| Gate | `configs/student-gate-v1.json` | Pass rule frozen before any student was trained. |

## Student interface

- **Observation `student-obs-v1`** (4,016 uint8): an 11×11 egocentric window with 16 channels (terrain from memory, current cheese/traps/units with HP, facing-toward-me, flying, memory age), a 16×16 coarse map from the robot's own memory with 8 channels (walls, dirt, mines, last-seen own/enemy kings, three symmetry guesses of the enemy base from the robot's first location, self, explored), and 32 scalars. Built only from the public `RobotController` API and what that robot itself sensed earlier; no communication, no teacher globals.
- **Actions `student-actions-v1`** (566): one primitive per step, repeated until END (at most 12 per turn): END, MOVE×8, TURN×8, 11 targeted types × 7×7 offsets (attack, carry, build, rat/cat trap, place/remove dirt, pickup, transfer, remove traps), THROW, DROP×8, BECOME_KING.
- **Explicit approximations:** `attack(loc, cheese)` is treated as a plain attack; amount pickups as full pickups; transfers move all raw cheese; `squeak` and `writeSharedArray` are not imitated (the student has no communication). Turns ended by bytecode exhaustion get no END record. The teacher's memory and communication are richer than the student's observation, so some labels are not predictable from the inputs.
- **Network `student-mlp-v1`:** 4016 → 384 → 256 (ReLU) with policy (566) and value (tanh) heads. Java inference uses a sparse first layer. Greedy play picks the action group with the largest total probability, then the best action in that group; plain argmax would favour END, whose probability is not split across targets.

## Status

- The smoke test and the 7-game pilot were clean: zero unmapped teacher calls, and almost all teacher actions fell inside the student's legality mask.
- The Java and PyTorch forward passes agree (`tests/test_student_parity.py`).
- The 400-game teacher collection is running.
