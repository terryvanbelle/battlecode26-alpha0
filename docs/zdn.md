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

**State margins (zdn2).** zdb's spawn rule is a set of sharp thresholds. Below its core count (12 babies on maps
of up to 441 tiles, else 18), it builds as soon as global cheese minus the rat's cost reaches its floor: 100 below
8 babies, 300 from 8 on, 20 for a 10-cheese rat, and never more than 2 · (2000 − round). Past the core count, it builds
when its surplus estimate clears the cost by 100. In teacher-zdn1 the first rule is exact: in boot zdb builds on 99 to
100% of the turns at or above its floor, and on 0.1% below it. Standardised, global cheese ranges over thousands, so the
few cheese between waiting and building are nearly invisible to the net. zdn1's king put 0.37 on building where zdb
built in boot (0.08 where it waited) and built on a third of those turns at argmax; past boot, 0.16. The state now
ends with 13 margin columns: cheese minus cost against 0, 20, 100, 300 and the floor, the surplus against cost + 100
(clipped to ±50), the baby count against 8, 18, 40, 80 and the core count (clipped to ±3), and global cheese at or
above 1000 and 3000. ZO.state computes them in the game. They are a function of the older columns
([scripts/zdn_margins.py](../scripts/zdn_margins.py)), so the teacher-zdn1 records train with the margins appended
(`python scripts/zdn_margins.py scripts/zdn_train.py ...`), with no re-collection. The king trained on all 145k turns
went from validation step loss 0.397 to 0.240 (accuracy 0.929 to 0.963; exact build tile 0.18 to 0.67). Where zdb
builds, its build probability went from 0.37 to 0.77 in boot (argmax 0.34 to 0.80) and from 0.16 to 0.77 past boot;
where zdb waits, from 0.08 to 0.04. A trial on 36k turns had learned no build rule with or without the margins: too
little data to show the change.

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
  register values and steps, with one END per recorded turn. The state's margin columns equal scripts/zdn_margins.py's
  function of the columns before them, and the king's cheese margins come within their clipping window. The player runs and records without errors, both with no
  net and with a net trained on the teacher game's records. Its random draws differ across game seeds (they repeated
  before the fix below). Every recorded q is the probability the player sampled with: log(1/n) with no net, and with
  a net what zdn_rl recomputes in Python from the record (median gap under 1e-3).
- tests/test_zdn_margins.py: the margin columns on hand-worked states (both floors, the end-of-game cap, the core
  count at 441 tiles), and the read hook (appends the margins once, keeps the dtype, refuses states that have them).
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

**zdn1 against its readings.** Labels 99.6% or more, no ERROR records; validation step accuracy 0.93 (king) and 0.88
(baby). Strength 51/192 = 0.27 against the held-out field and 0/64 against zdb. By the reading above RL should have
started from zdn1. I held it for one fix instead: the kings' under-building (State margins, above) was a diagnosed input
defect, cheap to repair and clearly limiting (zdn1 ended games with 6 babies to the opponent's 26).

**Readings for zdn2** (zdn1 plus the margins; same teacher records, same training settings), set before its games:
RL starts from zdn2 if its held-out field score is at least zdn1's 0.27, and from zdn1 otherwise. Either way RL starts
after this check, with no further imitation fixes first.

**zdn2 against its reading** (14:40 PDT). 52/192 = 0.27 against the held-out field (95% CI 0.21–0.34; zdn1 51/192)
and 0/64 against zdb, so RL starts from zdn2. The better build decisions did not change the score, because building was
not the limit. Game statistics (our side, mean per game):

| Opponent | zdb: rounds, cheese delivered | zdn2: rounds, cheese delivered |
|---|---|---|
| AlexT101 | 427, 1865 | 162, 94 |
| VarunVejalla | 632, 2821 | 162, 100 |
| daannte | 340, 1178 | 126, 49 |
| quesswho | 40 (win) | 79 (81% wins) |
| Guzzi | 72 (win) | 313 (53% wins) |

zdn's robots per round roughly match zdb's. Against the aggressive bots it delivers about 0.6 cheese per round to
zdb's 4.5, and its kings die early; against the weak ones its attack is slow. Both are sequences of many choices (gather,
carry, deliver; converge on a king). A per-step imitation fit of 0.88 compounds over those sequences, and RL from self-play
is meant to repair exactly that.

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

**rl-zdn1 through iteration 5** (November 17, 17:02 PDT). Started 14:42 from zdn2; about 20 minutes per iteration,
30 with gates. Parity passes every iteration (median 5e-5). Against zdn2: v3 20/32, v6 14/32 (pooled 34/64 = 0.53).
Against zdb: 0/32 at both. Pool games against v0: 96/174 = 0.55. The policy has hardly moved: KL to the anchor
0.02 for both units (cap 0.5), entropy flat, and each update uses about 0.007 of its 0.05 KL budget. Baby
credit_corr is positive on every update (0.27 to 0.73); king's is mixed (-0.68 to +0.57, four of six negative) over
only nine step kinds. The critic's held-out Brier fell from 0.227 to 0.174 (0.25 is chance). Self-play games are
getting shorter: 28.8k recorded-sample baby turns in iteration 0, 21.7k in iteration 5.

**Reading for rl-zdn1 at v12** (set before the v9 and v12 gates; about 19:30 PDT). Pool v9 and v12 against zdn2.
- Wilson lower bound above 0.5: RL is working; leave it alone.
- Otherwise, if KL to the anchor is still under 0.05: the update is too small to show anything in 12 iterations.
  Double it (4 epochs instead of 2; the 0.05 per-update KL stop stays) and continue the same run.
- If king credit_corr is negative on most of iterations 6–11 while baby's stays positive: before changing the update,
  look at the king's advantage and mass change per kind for the kinds that drive the correlation.

**rl-zdn1 at v12, against its reading** (read at 21:00 PDT, an hour and a half late: nothing checked it between
turns). v9 19/32 and v12 20/32 against zdn2, so 39/64 = 0.61 with a Wilson lower bound of 0.49, not above 0.5. KL to
the anchor was 0.03 for both units, and king credit_corr was negative on three of iterations 6–11, not most. By the
reading, then: 4 epochs instead of 2 from iteration 16's update on, in the same run (rl_loop reads the config each
iteration). By the time of the read, v15 had played its gate too (19/32; v9–v15 pooled 58/96 = 0.60, lower bound
0.50). Pool games against v0 had risen to 28/32 over iterations 11–15, against 23/52 and 29/52 in iterations 1 and 2.
These are about six games an iteration since PFSP started at iteration 10. The 32-game gate against the same net reads
lower. Against zdb: 0/32 at every gate so far.

**rl-zdn1's king under 4 epochs** (22:30 PDT, after the v18 gate). v18 scored 20/32 against zdn2 and 0/32 against
zdb. Pooled over v9–v18, the learner has 78/128 = 0.61 against zdn2 (Wilson 0.52–0.69). The first four 4-epoch king
updates (nets 17–20) raised entropy inside each update by +0.028, +0.016, +0.007 and +0.006, and their credit_corr
was negative every time (-0.18, -0.29, -0.06, -0.36). That is not a runaway. Entropy at the start of each update, on
fresh games, was 0.59, 0.62, 0.63 and 0.63. The start net had 0.63 to 0.64 at nets 1 and 2, and the 2-epoch updates
ranged from 0.59 to 0.62. The within-update rise is shrinking. Per step kind over the four updates:

| King kind | rows/update | mean raw A | mass | sum of dlog, 4 epochs (nets 17–20) | sum of dlog, 2 epochs (nets 7–16) |
|---|---|---|---|---|---|
| END | 17,296 | -0.001 | 0.757 | -0.032 | -0.030 |
| MOVE | 1,760 | -0.002 | 0.078 | +0.073 | +0.276 (A +0.014) |
| BUILD | 1,810 | -0.045 | 0.079 | +0.147 | -0.047 (A -0.025) |
| TURN | 675 | -0.052 | 0.030 | +0.154 | -0.041 |
| ALERT | 521 | -0.128 | 0.023 | +0.030 | +0.002 |
| DIG | 254 | -0.006 | 0.011 | +0.031 | +0.586 (A +0.009) |
| PICK | 241 | +0.023 | 0.011 | -0.015 | +0.016 |
| RTRAP | 78 | -0.005 | 0.004 | +0.087 | -0.188 |
| BITE | 66 | -0.227 | 0.003 | +0.044 | +0.039 |

Mass leaves END, which has about zero advantage, for nearly every other kind, including BUILD, TURN, ALERT and BITE,
whose mean advantages are negative. Under 2 epochs the largest moves (DIG +0.59, MOVE +0.28) went with positive
advantage. Under 4 epochs BUILD reversed, from -0.05 to +0.15, and kings now build more although building turns score
below average. Unlike in rl-s2, a kind's mean advantage here is a weak measure for the king. All of a turn's choices
share its advantage, and the critic sees only team, game and 10-round bin. So kinds that kings use when attacked
(BITE -0.23, ALERT -0.13) score low whatever their merit, and a turn that goes worse than the bin predicts pushes mass
from the chosen kind onto the others. Nothing changed for now. Pool games against v0 dipped for the 4-epoch versions:
10/20 over iterations 17–19, against 28/32 over iterations 11–15 for 2-epoch versions. Those are about six games an
iteration, and v18's 32-game gate against zdn2 held at 20/32.

**Reading at v24** (about 00:20 PDT, set before the v21 and v24 gates). Compare the 4-epoch versions with the 2-epoch
ones: v21+v24 against zdn2 with v9–v15 (58/96 = 0.60), and pool games against v0 over iterations 17–24 with
iterations 11–16 (32/38).
- If both are lower, with v21+v24 under 0.55 and pool under 0.70, go back to 2 epochs and keep the run.
- Otherwise keep 4 epochs. If king BUILD mass keeps rising against its advantage, measure BUILD against END in matched
  king states (same round bin, cheese and baby count) before changing anything.

**rl-zdn1 at v24, against its reading** (00:25 PDT, November 18). v21 scored 11/32 and v24 10/32 against zdn2, so
21/64 = 0.33, against 58/96 = 0.60 for v9–v15. Both were 0/32 against zdb. Pool games against v0 over iterations 17–24
came to 29/54 = 0.54, against 32/38 over iterations 11–16. Both fell below the reading's thresholds (0.55 and 0.70), so
iteration 26's update and later ones use 2 epochs again, in the same run. The learner is now clearly worse than zdn2,
and worse than v9–v18 were.
- **The king is drifting.** Its credit_corr on nets 21–26 was +0.04, +0.24, -0.08, -0.34, -0.42 and -0.75. Its
  entropy at the start of each update was 0.64 and 0.65 on nets 25 and 26, above the start net's 0.63 and rising.
  BUILD keeps gaining mass (dlog +0.045, +0.047, +0.053 on nets 24–26) although its mean advantage is -0.04 to -0.07.
  The babies look healthy: entropy steady at 0.71–0.72, and credit positive on every update but one.
- **A likely mechanism, not yet tested.** For a peaked policy (king END about 0.76 of the step mass, and near 1 in many
  states), noise in the update raises entropy on average. Entropy is convex in the logit gap near a corner, so a
  zero-mean random walk in the logits spreads mass from the dominant option onto rare ones. A bigger update (4 epochs)
  means a bigger walk. The anchor leash is what holds this back, and it weakened in the same window. beta anneals from
  0.1 to 0.03 over iterations 0–20, so it was 0.044 at iteration 16 and reached the floor at 20. The 4 epochs and the
  weaker leash are confounded.

**Reading at v30** (set before the v27 and v30 gates; about 02:00 PDT). Pool v27 and v30 against zdn2; both versions
were trained at 2 epochs throughout.
- 0.5 or more: the drop was the 4 epochs. Keep the run.
- 0.45 to 0.5: keep it and read again at v33.
- Under 0.45: the weaker leash, or the drift it lets through, is the likely cause. Stop rl-zdn1. Start rl-zdn2 from
  v18's nets (20/32 against zdn2, the last good gate), with the same anchor zdn2, 2 epochs, and beta held at 0.1 (no
  annealing).
