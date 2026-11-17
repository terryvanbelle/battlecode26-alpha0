# zdn: a neural controller with no decision tree (November 17, 2026)

## Why

Every action-space redesign so far kept zdb's cascade as the executor. zdp's decision points, grouped floors, the
smoothed per-turn start (rl-s2) and the plan heads (rl-p3) all let a net choose among zdb's own branches, errands or
constants, while zdb's code still decides targets, thresholds and fallbacks. The landscape stays the one section 17 of
[rl-es-plan.md](rl-es-plan.md) measured: good policies are near-deterministic, and policies with entropy play badly.

The user's direction (November 17): discard zdb's decision tree entirely, and replace it with a mechanism that can
do the same computation but doesn't use if/then decisions as its basis.

## The boundary: what stays code

Code keeps only what decides nothing:
- **Perception.** This turn's sensing, plus a memory of what was seen (map tiles, mines, cheese, kings, enemies, cats,
  each with its age), decoded team messages, and symmetry candidates for the enemy king.
- **Path distances.** Breadth-first search over the known map, given to the net as inputs. Navigation is the net
  picking the step that shortens the path to where it wants to go.
- **Legality.** The engine's rules, as masks over the options.
- **Message encoding.** The net picks what to say (a typed fact or call); code packs it into the wire format and the
  receivers' perception decodes it.
- **Execution** of the primitive calls the net picks.

Everything zdb's tree decides becomes the net's choice: what to do this turn, where to step, which way to face, what
to bite, grab, throw, trap, dig or spawn, what to commit to (role, target), what to remember, and what to say.

**Rule for inputs.** Every quantity that zdb's decision code compares, thresholds, counts or optimises over is an input
to the net, as a state feature or as a feature of the option it scores. Every comparison and every selection is the
net's job. A function of these inputs computed by zdb's tree is then representable by the net, so the net can do zdb's
computation, and imitation can find it.

## Why this should escape the trap

- The options are plain actions and targets, each with its own features, not errands run out of context. Choosing a
  different option means doing a different thing, not running a branch where it was never meant to run.
- A smooth net trained by imitation is uncertain mainly where zdb's choice was a near tie in the inputs, so its
  entropy sits near decision boundaries, where both choices are about equally good and where the gradient on the
  boundary lives. A uniform floor spreads deviations everywhere instead.
- Commitments (role, target) are registers the net writes, kept until it changes them. A different commitment is a
  coherent alternative plan, not a one-turn twitch.

## Architecture

**Turn structure.** At the start of a turn the net reads the state and writes its memory registers. Then it picks
primitive calls one at a time from the currently legal ones, until it picks END (or hits a step cap). Options are
re-generated after every call, so a step taken after a move sees the new position.

**Inputs.**
- State vector S: own unit (type, health, raw cheese, cooldowns, facing, carrying), team (global cheese, kings,
  babies, rat cost, traps, dirt, cooperation and backstab), the round, map size, decoded team messages, memory
  summaries, and the registers.
- Local grid L: an egocentric window of channels (terrain, cheese, mines, own traps, allies, enemies, cats, flying
  rats, visibility and age of the memory).
- Options: each option c has a kind (move, turn, bite, pickup, transfer, carry, drop, throw, dirt, trap, spawn, become
  king, squeak, publish, END, or a register value) and a feature row r_c that describes that option in the current
  state (e.g. for a step: path distance to each target that matters after the step, threats at the tile, trap risk).

**Net.** A trunk h = MLP([S, L]) is computed once per turn. Each head scores every option with one shared hidden layer:
logit_c = w · relu(A h + B r_c + C ctx + b) + b0, where ctx is the step context (step index, what has been done this
turn). A categorical head is the same scorer with one-hot rows. Softmax over the legal options.

**Memory registers.** A few registers per robot carry its intent across turns (mode/role and committed targets).
Pointer registers point at an option from a target list (known mines, kings, enemy-king candidates, cheese, explore
points, sites). Code keeps "rounds since this register changed" as an input. In imitation each register is labelled
with zdb's own intent variable (teacher forcing: input = zdb's value at the turn start, label = its value at the end).
The primitive student lacked exactly this: it copied 83% of zdb's single actions, yet could not carry zdb's multi-turn
plans ([imitation-results.md](imitation-results.md)).

## Imitation

**Teacher records.** The teacher package is zdp (zdb, net mode off) plus zdn's perception and a recorder. Its
RobotController is wrapped (RcLog, generated by `scripts/gen_rclog.py`), so every state-changing call zdb makes is seen
just before it runs. The recorder then builds the options of that moment, labels the one zdb's call matches, and emits
a record. Each sampled turn yields a register record (S, L, register options and zdb's intent before and after) and
one step record per call plus END.

**Build and sampling.** `scripts/build_zdn.py --teacher zdnt --baby-pm 40 --king-pm 150` writes the teacher package
(java/bots/zdp + java/bots/zdn + java/bots/zdn-teacher, one package). Its patches to zdp must each match exactly once:
RcLog around the RobotController, ZT.start/end around the turn, ZT.goal at Nav.goTo (the GOAL register), ZT.mode after
moveDecision (the MODE register), and ZT.sample at every exploration draw (the EXPL register's candidates). Perception
and register bookkeeping run every turn. Records are written on a hashed sample of turns, per mille of baby and king
turns. An exception the recorder or the player catches is sent as an ERROR record (point 299), so collection and tests
can count them. `scripts/build_zdn.py --player NAME [--baby net.json --king net.json] [--record]` writes the player; with
no net it chooses uniformly.

**Smoke (09:20 PDT).** On four games (one rush map, three gather maps), the teacher's trajectory hash equals plain
zdp's for the same seed: the recorder does not change zdb's play. Every register value and every primitive zdb used
matched one of zdn's options in 99.8 to 100% of the records. That covers moves, turns, bites, carries, throws, traps,
digs, transfers, squeaks, builds, alerts, king moves, and all registers but CASH. CASH was NONE throughout: zdb's cash
mode is rare.

**Collection.** configs/teacher-zdn1.json: 600 games against the 12 training opponents plus 30% mirror, the same
excluded maps as before, no bytecode limits (scripts/pipeline_zdn1.sh).

**Training.** Cross-entropy per head over the legal options with label smoothing, on the box's GPU. The smoothing keeps
the start soft enough for RL. Register r's scorer also sees one-hots of the kinds chosen for registers 0..r-1, so the
registers form a chain, not independent guesses.

## Tests, readings set before their games

**Code tests.**
- tests/test_zdn.py: the teacher build's patches (each applied once; a drifted patch is an error; no class silently
  dropped), the player build, the Java-literal escaping of packed nets, the pack layout (a reader mirroring
  ZNet.java), and the record reader on synthetic records (including the ERROR record).
- tests/test_zdn_train.py: the register context, the export's folding of the standardisation (packed-net math on raw
  inputs equals the torch net on standardised inputs), and a small end-to-end training run on synthetic records with a
  learnable rule.
- tests/test_zdn_java.py: ZNet.java's logits equal torch's (export, pack, then the Java parser). Engine games on
  DefaultSmall: the teacher plays exactly like zdp (same trajectory hash), raises nothing, and labels at least 98% of
  register values and steps, with one END per recorded turn. The player runs and records without errors, both with no
  net and with a net trained on the teacher game's records.

**Readings for zdn1** (the first imitation net), set before its games:
- Label coverage on teacher-zdn1: at least 98% of register and step records labelled, and no ERROR records. Below that,
  fix the option builders before training.
- Fit: validation step accuracy is reported, not gated (fidelity is not the goal). Below 0.70, check the inputs
  before playing: the primitive student reached 0.83 on single actions without registers.
- Strength (T = 1.0, the sampling RL starts from): score against the five held-out field bots and against zdb.
  0.40 or more against the held-out field counts as competent. RL starts from zdn1 whatever the score, unless it is
  below 0.15, which would point at a bug (ERROR records, an input mismatch between teacher and player) rather than a
  weak policy.

## RL

To be filled in: PPO over all heads, self-play only, the same gates as before.
