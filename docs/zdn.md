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
can count them. `scripts/build_zdn.py --player NAME [--baby net.json --king net.json] [--record PM] [--version V]`
writes the player; with no net it chooses uniformly.

**Smoke (09:20 PDT).** On four games (one rush map, three gather maps), the teacher's trajectory hash equals plain
zdp's for the same seed: the recorder does not change zdb's play. Every register value and every primitive zdb used
matched one of zdn's options in 99.8 to 100% of the records. That covers moves, turns, bites, carries, throws, traps,
digs, transfers, squeaks, builds, alerts, king moves, and all registers but CASH. CASH was NONE throughout: zdb's cash
mode is rare.

**Collection.** configs/teacher-zdn1.json: 600 games against the 12 training opponents plus 30% mirror, the same
excluded maps as before, no bytecode limits (scripts/pipeline_zdn1.sh).

**Training.** Cross-entropy per head over the legal options with label smoothing, on the box's GPU. The smoothing keeps
the start soft enough for RL. Register r's scorer also sees one-hots of the kinds chosen for registers 0..r-1, so the
registers form a chain, not independent guesses. Memory, not GPU time, limits the data: one baby turn holds about
8 KB of features (int16, option lists padded to the longest), and a training run peaks near twice that per turn (the box
has 8 GB). zdn_train.py therefore concatenates in place, keeps no float64 or full-width copies, and serves large
allocations by mmap (heap fragmentation had added a third). --keep-frac keeps a random share of every game's turns, so
a capped run still sees every game.

## Tests, readings set before their games

**Code tests.**
- tests/test_zdn.py: the teacher build's patches (each applied once; a drifted patch is an error; no class silently
  dropped), the player build, the Java-literal escaping of packed nets, the pack layout (a reader mirroring
  ZNet.java), and the record reader on synthetic records (including the ERROR record).
- tests/test_zdn_train.py: the register context, the export's folding of the standardisation (packed-net math on raw
  inputs equals the torch net on standardised inputs), the low-memory helpers (equal to plain numpy), the scorer's two
  paths (gathered and per-kind), --keep-frac (whole turns with all their steps, seeded), and a small end-to-end
  training run on synthetic records with a learnable rule.
- tests/test_zdn_java.py: ZNet.java's logits equal torch's (export, pack, then the Java parser). Engine games on
  DefaultSmall: the teacher plays exactly like zdp (same trajectory hash), raises nothing, and labels at least 98% of
  register values and steps, with one END per recorded turn. The player runs and records without errors, both with no
  net and with a net trained on the teacher game's records. Its random draws differ across game seeds (they repeated
  before the fix below). Every recorded q is the probability the player sampled with: log(1/n) with no net, and with
  a net what zdn_rl recomputes in Python from the record (median gap under 1e-3).
- tests/test_zdn_rl.py: the decision table (every recorded choice, grouped by turn, with its options), the net loader
  (net_from_json reproduces the packed net's logits from raw inputs; unfold and export keep the function), the data
  stage's parity check (passes on true q, exit 2 on corrupted q), and the trainer (--lr 0 leaves the nets
  byte-identical; a planted outcome moves the policy toward the rewarded choice, and the credit check reports it).
- tests/test_rl_loop_zdn.py: rl_loop.sh's zdn stages call the zdn scripts with the config's values, and a KL refusal
  rolls both units back (stubbed scripts, no games).

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

**Policy randomness.** The player samples from its own stream, seeded from Math.random, which our runner reseeds per
game, team and robot (alpha0.student.Game). Robot IDs and the opening repeat on every game of a map, so a stream
seeded from them repeats its draws across seeds. That was rl-p2's bug, and zdn's first player had it too: its
policy drew from zdb's birth-seeded stream, and its kings from a constant one. The recorder's turn sampling has a
stream of its own, so recording never changes play. The VM's main checkout ran the runner from before the reseed
fix; it was rebuilt on November 17 (the teacher collection is unaffected: zdb is deterministic, and the engine
varies with the game seed; only 7 of teacher-dp2's 1000 games were duplicates).

**Records.** RL builds record a random sample of turns (`--record PM`, `--record-king PM`). Each record is VER 2: every
choice carries q = round(-4096 log p) under the policy that sampled it (0 when forced), and each turn carries the net
version. The teacher's records carry no q.

**Pipeline** (scripts/rl_loop.sh with config "kind": "zdn"; configs/rl-zdn1.json). Each iteration is the same as for
the zdp nets, with these stages swapped:
- Build: `build_zdn.py --player zdnr<k>`, with nets/<k>/{baby,king}.json, 20 per mille of baby turns and 250 per mille
  of king turns recorded, and version k.
- Data (`zdn_rl_data.py`): the learner sides' records become one decision table per unit. A row is one sampled choice
  (register or step) with its options stored ragged. The tables are joined with z, the seat-swapped pairs and the
  trace. Parity: log p is recomputed from each record with the version's net and compared with q (median under
  1e-3, 99.9th percentile under 2e-3, else stop). At most 150,000 turns per unit.
- Train (`zdn_rl_train.py`): PPO over every head, registers and steps, babies and kings. A turn's advantage comes from
  its team-game and 10-round bin through the critic (rl_critic's rich critic, GAE lambda 0.9), and all of the turn's
  choices share it. Loss per sampled multi-option choice: the clipped surrogate with r = pi(choice) / exp(-q / 4096)
  and a KL leash to the imitation nets (beta 0.1 annealed to 0.03). No entropy bonus: in rl-s2 it moved mass onto
  rare options against their advantages (below); the leash to the label-smoothed imitation nets keeps the policy soft. An update stops early when the
  approximate KL to the playing net exceeds 0.05. If a unit's KL to the anchor ends above 0.5, nothing is exported
  and the net is rolled back. The standardisation is the imitation run's (norm.npz), carried with the nets.
- Credit check, every update: per chosen step kind, the change of the step policy's mass against the kind's mean
  advantage, and their correlation over kinds (credit_corr, in the log and progress.jsonl). It should be positive;
  rl-s2's drift had it near -1.
**Why no entropy bonus (rl-s2's credit check, November 17).** rl-s2 pooled 47/96 = 0.49 against its start over
the v9, v12 and v15 gates (38/96 against zdb). On iteration 14's states, options the start played under 2% of the
time gained probability by v15 at five of seven points (log ratio +0.26 to +0.43), although their games ended no
better. Where the advantage per option is logged (point 16), it ranks options as their results do (correlation
+0.95), and the policy moved against it (-0.95). The systematic part of the update was the entropy bonus (c_ent
0.01); the advantage signal per option was a few thousandths of its spread.

- Gates: as for rl-s2. The anchor gate plays the imitation player zdn1, every third version, with zdb tracked. The
  field never feeds back into training.
