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

Open capability gap: king comms (shared-array alerts, budget, enemy-king and cat publishing) are still hard-coded
rather than a decision point. A KING_COMMS point is planned.
