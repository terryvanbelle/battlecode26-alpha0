# Decision-point imitation (zdp)

**Why.** Primitive-level imitation stalled. student-v9 and student-v10 each won 1/32 against zdb. In student-v10,
single-action accuracy was 83% and the shared-array writes were 94–100% correct, yet its babies still died without a kill
and delivered a fraction of zdb's cheese. Copying zdb one primitive at a time doesn't add up to zdb's multi-turn plans
(see [imitation-results.md](imitation-results.md)).

**Idea (the user's fallback plan).** Keep zdb's high-level behaviours (deliver, rescue, raid, go to mine, and so on) and
its communication code. Replace only its control logic, the choice of which behaviour runs, with a small net.
- That net is small enough for the engine's per-turn bytecode limits (baby 17,500, king 20,000), so it can play in
  real matches. The primitive student was about 1000× over budget.
- Its starting policy can copy zdb closely, because it chooses among zdb's own behaviours.

## Pass 1: bug-for-bug zdb (`java/bots/zdp`)

zdp is zdb (2026b `bots/zdb`, commit 75c15b7b) in package `zdp`, with decision points added and nothing else changed.
zdb's bugs and its near-duplicate code paths stay in. Pass 2 cleans them up, checked against zdb by test runs.

At each decision point:
1. A hook snapshots the point's features before zdb decides (`D.babyMove()` etc.).
2. zdb's code decides as before. Each branch sets the option it took (`D.lab = D.M_...`).
3. A hook sends (point, label, features) to the research harness (`D.emit()`). The harness intercepts indicator
   strings that start with U+DEC0, and `Game --decisions AB` writes them to `decisions-{A,B}.bin.gz`
   ([reader](../scripts/decisions.py)).

**Exactness.** zdb's play depends on its bytecode budget. Flow.work spends whatever the turn leaves, and 50 extra
bytecodes per baby turn changed all 8 test games. Every hook therefore uses zdb's ledger: the bytecodes it spends are
added to `G.debt`, which zdb's budget reads add back. A label costs 6 bytecodes (`D.lab = k; G.debt += 6;`), and a
wrapped call is credited with K = 9.

**Check.** [replay_check.py](../scripts/replay_check.py) plays zdp and zdb on the same jobs under tournament limits.
It compares each game's per-round trajectory hash (both teams' cheese, transfers, units, traps and cat damage).
The plain copy is identical in 8/8 games.

## Decision points

| Point (id) | Where | Options | Features |
|---|---|---|---|
| BABY_MOVE (1) | `Baby.moveDecision` | 39 branches of zdb's movement cascade (`D.M_*`); net mode adds plain steps/turns (56) | 49 |
| BABY_ACT_PRE (2) / BABY_ACT_POST (3) | action phase before / after the move | 21 (`D.A_*`: pass, throw, bites, traps, deliver, cash king, raw forward actions) | 53 |
| BABY_FACE (4) | `guardTurn` | 13 (none, seen/remembered/heard enemy, guard, 8 directions) | 29–31 |
| BABY_ROLE (5) | birth and every 10 rounds | 5 (gather, scout, fight, raider, attack) | 32 |
| BABY_COMMS (6) | `cashTick` / `report` | 13 squeak kinds (`D.C_*`) | 37 |
| MINE_PICK (7) | `pickMine` | candidate rank (40) | 45 |
| KING_MOVE (10) | king movement | 37 | 40 |
| KING_SPAWN (11) | `King.spawn` | 22 (no surplus, floor, cap, build site, dig, blocked, hold) | 33 |
| KING_ACT (12) | king action | 11 | 30 |
| KING_FORM (13) | king formation | 30 | 29 |

The features are a compact, zdb-computed summary: role, cheese, health, cat, enemies, targets, kings, mines, the alert,
counts in sight, position and so on (small integers). Many options never fire because zdb's `Cfg` flags turn those
branches off.

**Verification (no bytecode limits).** With all 11 points recording, zdp replays zdb exactly in 32/32 games
(3.86M records, all labelled). Under `--limits`, any extra bytecodes per turn shift zdb's own bytecode-dependent
choices (50 extra bytecodes per baby turn changed 8/8 games), so for now this work runs without limits; the budget
comes later.

## Net mode (pass 1b)

A separate build (`D.NET` true, a compile-time constant, so the recording build has none of its code). At a point,
the net picks an option. zdp runs that option's behaviour; if the behaviour can't act, zdb's cascade runs as before.
Options zdb never takes (e.g. plain moves in each direction) get a small probability floor, so RL can find them.
Each point has its own MLP (hidden 64,64, about 7k float32 weights), compiled into the bot as packed strings in
`W.java` ([build_dpnet.py](../scripts/build_dpnet.py), [train_dp.py](../scripts/train_dp.py)); input normalisation is folded into layer 1.
Imitation trains it on the labels; RL then trains it in self-play (with RL against zdb allowed as a bootstrap).

## Results

**zdpn1 (teacher-dp1: 600 zdp games vs the train opponents, no limits).** Validation accuracy per point: BABY_MOVE
0.985, ACT_PRE 0.995, ACT_POST 0.999, FACE 1.0, ROLE 0.9996, COMMS 0.996, MINE_PICK 0.995, KING_MOVE 0.90,
KING_SPAWN 0.97, KING_ACT 1.0, KING_FORM 0.9996. Sampling at temperature 1 with a 0.001 floor.

| Match (no limits) | zdpn1 | zdp (= zdb) |
|---|---|---|
| vs zdb, 8 maps × 2 seeds × 2 sides | **16/32 (0.50)** | (mirror) |
| held-out field (5 bots × 8 maps × 2 sides) | **40/80 (0.50)** | 41/80 (0.51) |
| examplefuncsplayer | 16/16 | 16/16 |

Per held-out bot (zdpn1 / zdp): Guzzi 16/16 / 16/16, quesswho 16/16 / 16/16, Varun 6/16 / 5/16, AlexT101 1/16 / 3/16,
daannte 1/16 / 1/16. The net-mode bot plays at zdb's level: the bootstrap goal (~40% vs the field) is met.

**Debugging notes.** The first net-mode builds lost every game to zdb. Causes, in order: (1) per-tensor 16-bit
weight quantisation after folding the normalisation (constant features got ~1000× weights, zeroing the rest; the king
never spawned) — weights are now exact float32 and constant features get unit scale; (2) the MINE_GO/WANDER executors
skipped zdb's idle-mine shift. With nets trained on only 32 games, each point alone was fine (3–6/8 vs zdb) but all
together lost 0/8: small per-point errors compound. 600 games fixed that.

## Pass 2: cleanup, checked against zdb

**2A, behaviour-exact cleanup (commit 0ac0a97).** zdp went from 10,403 to 5,948 lines. The removed code was dead
under zdb's constant `Cfg` values in both builds: 7 whole files (KPath, BitFlow, GrabMicro, Seal, TrapRows, Trail,
PathOracle), about 1.1k lines of Baby, 380 of King, 580 of Flow and 227 unused `Cfg` constants. Duplicates between
the cascade and the net-mode executors are now shared helpers. Check: zdp against zdb was 60/60 identical games,
and a net-mode build of the cleaned source with the zdpn1 nets replayed zdpn1 24/24 identically.

**2B, bug fixes (commit 0bdf161).** 21 suspected zdb bugs were found by six area reviewers; 16 survived adversarial
verification. Each fix sits behind its own `Cfg.FIX_*` flag (false gives zdb's behaviour), so it can be bisected.
- Babies: bite any reachable tile of an enemy king (not just the nearest tile); throw enemies into cats, which kills
  them; THROW_HOLD after a carry swap; forget a stale enemy-king sighting; wander inside the mine box that `atMine`
  counts; change mine when a stalled one is abandoned; end a seek only on real news; FORM_KING centre rats promote
  as soon as they are action-ready.
- Kings: pickup income reaches the income estimate; skip piles under the king; bite enemy kings on a reachable tile;
  the shared alert keeps the most-threatened king's alert.
- Navigation and traps: DIG_FIRST only applies to an old target; the robot-blocked counter resets on a new search;
  trap marks fix the arrival round and ignore carried rats.

| Check (no limits) | Result |
|---|---|
| all `FIX_*` false vs zdb | replay identical 24/24 |
| fixes on vs zdb (8 maps × 4 seeds × 2 sides) | **40/64 (0.625, 95% CI 0.50–0.73)** |
| fixes on vs held-out field (6 bots × 8 maps × 2 seeds × 2 sides) | 114/192, the same as zdb (114/192) |
| zdpn1 (nets) on the same jobs | vs zdb 26/64; field 109/192 |

Test tooling: [strength_check.sh](../scripts/strength_check.sh) (box split, Wilson intervals);
`build_dpnet.py --name X --cfg FLAG=value,...` builds a variant without nets.

**2C, net-mode fixes and facing/comms control (commit e6120e9).** In net mode the executors now match the cascade: mine
blocking, the idle features, the stall/hunt split, attacker bookkeeping, and BABY_FACE opens on every turn where the
robot can turn. Choosing "no squeak" on BABY_COMMS now really mutes zdb's squeak. Check: the recording build replays
the 2B build 40/40 identically.

### zdpn2 (November 13, 2026, 23:07 PDT)

zdpn2 is trained on teacher-dp2: 1000 games of zdp after pass 2 with fixes on, decision records only. It uses the
same 11 points and (64,64) MLPs as before, built with floor 0.001 and T 1.
- Validation accuracy: BABY_MOVE 0.992, KING_MOVE 0.935, KING_SPAWN 0.964; every other point is at least 0.994.

| Check (no limits; same jobs as above) | Result |
|---|---|
| zdpn2 vs zdb (64) | **33/64 (0.516, 95% CI 0.40–0.63)**, up from zdpn1's 26/64 |
| zdpn2 vs held-out field (192) | **113/192 (0.589)**; the zdb copy scores 114/192 |
| teacher (zdp, fixes on) vs zdb | 40/64 |

zdpn2 is the starting point and KL anchor for self-play RL ([rl-plan.md](rl-plan.md)).

Open capability gap at the time: king comms (shared-array alerts, budget, enemy-king and cat publishing) were
hard-coded rather than a decision point; pass 3 adds KING_COMMS (below).

## Pass 3: capability completion

The RL starting net must be able to do anything a hand-coded bot can ([capability-principle.md](capability-principle.md)).
After pass 2 the 11 points could only choose among zdb's own branches, and several API actions, the king's shared-array
writes and free actions (pickup, trap removal) were not decisions at all. Pass 3 adds five points and appends options
and features to five others. Every addition is **appended** (old option and feature indices keep their meaning), and
every new behaviour is reachable only in net mode through an option the build makes reachable.

| Id | Point | Options (old → new) | Features (old → new) | zdb label |
|---|---|---|---|---|
| 1 | BABY_MOVE | net 56 → **72**: strafe (step without turning) 56..63, risky step past the cat lane 64..71 | 60 → **76** (+comms block) | unchanged |
| 2/3 | BABY_ACT_PRE/POST | 21 → **39**: bite with cheese 1/4/9/16/36/100, drop held rat in 8 directions, transfer 10/50/half, disintegrate (Cfg.DP_ALLOW_DISINTEGRATE, off) | 53 → **74** (+5, +comms block) | unchanged |
| 5 | BABY_ROLE | 5 → **6**: R_KEEP; reviews every Cfg.DP_ROLE_EVERY rounds, current role masked | 32 → **112** (+comms block, +all 64 shared-array slots) | review: R_KEEP when unchanged |
| 6 | BABY_COMMS | 13 → **15**: C_RAW (a free 32-bit squeak, BABY_VAL digits), C_RELAY (re-squeak the newest heard net-type squeak) | 37 → **53** (+comms block) | unchanged |
| 8 | **BABY_FREE** (new) | 16: zdb's pickup, take all, take none, cap 10/25/50/100/200; remove own rat/cat trap on this or fan tiles 0–2 (chained, ≤ 2) | 36 | FR_ZDB (0) |
| 9 | **BABY_VAL** (new, secondary) | 32: a base-32 digit | 30 | none (net only) |
| 12 | KING_ACT | 11 → **17**: bite the weakest reachable enemy with cheese 1/4/9/16/36/100 | 30 → **38** (+king mini block) | unchanged |
| 10/11/13 | KING_MOVE/SPAWN/FORM | unchanged | +8 (king mini block) | unchanged |
| 14 | **KING_COMMS** (new) | 81: keep zdb's writes, drop all, drop a group (slots, alert, free registers R0–R2, formation, mines, cats, budget), budget band 0/100/1023, call/clear the alert, squeak, remove a rat/cat trap, set any of the 64 slots (KING_VAL digits); an edit loop of ≤ 12 continue steps | 94 | KC_ZDB (0) |
| 15 | **KING_VAL** (new, secondary) | 32: a base-32 digit | 22 | none |
| 16 | **KING_TURN** (new) | 9: keep the facing, or turn to one of 8 directions (opens when a spawn is possible: a newborn faces the king's way) | 24 | KT_ZDB (0) |

BABY_FACE (4) and MINE_PICK (7) are unchanged. The "comms block" (16 features) is what the robot hears: R0–R2, BUDGET,
ALERT, FORM and the newest net-type squeak with its age; the "king mini block" (8) is the king's view of the same.
Value points (9, 15) emit a value as base-32 digits (`D.digits`), one secondary record per digit; they are reached
only from net-only options, so they never record in a recording build.

**Mechanisms.**
- *Option 0 = zdb* at every new point, and new options are appended, so a recording build labels exactly zdb's choice.
- *W.A / W.FL.* build_dpnet.py writes `W.A[p]` (the active option count, from `net.json["active"]`) and `W.FL[p]`
  (per-point floor, `--floor-points P:F,...`); DNet samples only the first `reach(p)` options. Both default empty, which
  is byte-for-byte the old behaviour. BUILD.json records `active` and `floor_points`; rl_data, rl_train and desaturate
  read the per-point floors.
- *Second RNG stream.* The new points (8, 9, 14, 15, 16) sample from `DNet.uniform2()`, so giving them nets never shifts
  the old points' draws.
- *Free-action chains.* BABY_FREE and KING_COMMS reopen after a continue option (each step its own record, step is
  feature 0), with hard caps; only terminal options are allowed at the cap.
- *Record buffers* grow to 192 chars; every point has `NAME_FEATS`; [dp_points.py](../scripts/dp_points.py) reads the
  catalogue from the Java constants, `decisions.py --check-feats` checks every record's feature count.
- *widen_dpnet.py* (logic in `dpnet_io.widen`) grows a pass-2 net to the new points: zero input columns (old logits
  bit-identical), new output rows at bias median(max old logit) − ln(10·m/floor) on calibration records (a new option
  starts at about 0.1·floor/m but keeps a gradient), R_KEEP at p99.9 + 8 so reviews keep the role, `--active old` (the
  build reaches only the old options: exact replay) or `full`, and `--init-new zdb-hard` (option 0 at bias 0, the rest
  −30: replays zdb) or `calibrated` (the rest at −ln(10·m/floor): the RL start) for the new points; value points start
  uniform. norm.npz is padded.

**Checks (VM, no limits).**

| Check | Result |
|---|---|
| zdp with all `FIX_*` false vs zdb (mincemeat, jail, rift, pipes × seeds 21–23 × AB, vs zdb) | replay identical **24/24** |
| zdp (fixes on, recording) vs the pre-pass-3 zdp | identical **24/24** |
| net-v2 built from the pass-3 source (floor 0.001) vs the current zdpn2 | identical **16/16** |
| net-v2 widened (`--active old --init-new zdb-hard`, nets on all 16 points, `--floor-points 8:0,14:0,16:0`) vs zdpn2 | identical **16/16** |
| RL build (net-v2, floor 0.005, `--rl`) vs the pre-pass-3 RL build | identical **16/16**; rl_data parity on 4 games: median 8.3e-5, p99.9 1.2e-4 (n 130k), passed |
| recording build: `decisions.py --check-feats` on 24 games (6.66M records) | every point OK; new labels: BABY_FREE 100,771 × FR_ZDB, KING_COMMS 54,613 × KC_ZDB, KING_TURN 54,255 × KT_ZDB, BABY_ROLE reviews 135,793 × R_KEEP |
| train_dp.py on 8 of those games (3 epochs) | trains all 14 labelled points, including 8, 14, 16 (72-option BABY_MOVE, 112-feature ROLE) |
| smoke: every option live (`--active full --init-new calibrated`, floor 0.05, RL build), 4 games vs zdb | 0 errors, 0 exceptions, 4 valid games; every point incl. 9 and 15 fires and samples new options; rl_data parity median 5.4e-5, p99.9 1.2e-4 (n 67k), passed; `--check-feats` OK; test_trailer on the real files OK |

The smoke build loses all 4 games: every point at floor 0.05 with all options live is much noisier than the RL
start (floor 0.005, new options staged). It checks plumbing only.

**Open issues.**
- KING_VAL's KC_DECOMP (zdb's own slot writes decomposed into KING_VAL digit labels) is not implemented, so KING_VAL
  and BABY_VAL have no imitation labels; they start uniform and only RL can shape them.
- widen's calibration uses the max over all old logits (not only the allowed ones) and m = the point's full option
  count, both approximations of the design's formula.
- widen's new-option bias is calibrated against unmasked logits: with all options live at floor 0.05 the new
  BABY_ACT_PRE rows hold about 14% of the unmasked probability (BABY_MOVE 4%, KING_ACT 2%); the real share is lower
  because the masks drop most of them, but unlock stages should check it.
- configs/rl-v1.json has no `floor_points` or unlock stages yet; rl_loop builds do not pass `--floor-points`. The
  design's staging (phase A: 5 with R_KEEP, 14, 15, 16, 12; phase B: 8, 9, 6, 2/3, then 1) is for the next RL run.
- Gaps left open (see capability-principle.md): per-robot writable memory, babies' read of the full shared array,
  more than the newest heard squeak, comms block on FACE / MINE_PICK, king pickup control, bite / transfer amounts
  as ladders, resign excluded, hook bytecode budgets (limits are off).
