# Design brief: a new action space for zdp (bc26a0)

Repo: /home/terryvanbelle/projects/vibe/bc26a0 (read-only for this task). Bot under redesign: java/bots/zdp
(zdb, a ~5k-line hand-coded Battlecode 2026 bot, wrapped as 16 net "decision points"). Evidence files are in
diagnostics/zdp-action-space/ (maps in maps/, analysis scripts and outputs in analysis/).

## 1. The question and the owner's instructions

After imitation (zdp copies zdb) every RL and ES attempt stalled: PPO (rl-m2, 39 iterations), one CPI step
(discarded), and low-rank ES (found a real gradient direction, split-half cosine 0.31, but a line search along it scored
129/256 = 0.504, t 0.13). Every perturbation direction is flat or downhill.

The owner then asked (2026-11-16): "If we're truly stuck on a sharp peak on the fitness landscape, maybe we should be
changing the fitness landscape. Is it possible that the mapping of network to actions is too overconstrained, so that
only one particular controller stands a chance of performing well?" ... "I give you full latitude to refactor zdb
however you see fit" ... "You're also free to break the constraint that your refactored zdb needs to fully imitate zdb".

Standing constraints from the owner:
- RL means reinforcement learning (policy gradient family). Fixing estimators inside RL (critic, GAE, exploration,
  shaping, per-robot credit) is allowed; swapping in ES is not the plan.
- All RL progress comes from self-play; field bots are only benchmarks.
- Capability-complete start: the net that starts RL must be at least theoretically able to do anything a hand-coded
  bot could do (full RobotController API with parameters, sensing, comms, memory). Gaps must be deliberate and
  documented (docs/capability-principle.md). Owner's earlier idea (2026-11-13): factor zdb's control logic (mode and
  target choices) out of its high-level actions, let the net replace the control logic, keep zdb-never-used actions
  available at small probability, and put an abstraction over comms (net picks what to say; code encodes it).
- Competence bar: ~40% against the public field is a competent RL start (zdb is ~60%). Do not chase fidelity to zdb.
- Bytecode limits are off for now (each turn gets 1e9 bytecodes in research games). Note it, don't design around it.
- When a fix is accepted, old paths and fix flags are deleted (no dead code).

## 2. Measured evidence

### 2a. Sensitivity: floor 0.2 at one point (20% of that point's decisions uniform over its allowed options), all other
points at v39's normal floors (0.005; point 5 at 0.001). 64 games each against zdpg39 (= v39 at normal floors).

| point | name | score | t | reading |
|---|---|---|---|---|
| 1 | BABY_MOVE | 17/64 = 0.27 | -4.3 | sharp |
| 2 | BABY_ACT_PRE | 14/64 = 0.22 | -4.8 | sharpest |
| 3 | BABY_ACT_POST | 18/64 = 0.28 | -3.7 | sharp |
| 4 | BABY_FACE | 27/64 = 0.42 | -1.3 | mild |
| 5 | BABY_ROLE | 32/64 = 0.50 | 0.0 | flat |
| 6 | BABY_COMMS | 26/64 = 0.41 | -1.6 | mild |
| 7 | MINE_PICK | pending | | |
| 8 | BABY_FREE | pending | | |
| 10 | KING_MOVE | 34/64 = 0.53 | +0.4 | flat |
| 11 | KING_SPAWN | 17/64 = 0.27 | -3.7 | sharp |
| 12 | KING_ACT | 28/64 = 0.44 | -1.3 | mild |
| 13 | KING_FORM | 24/64 = 0.375 | -2.0 | moderate |
| 14 | KING_COMMS | pending | | |
| 16 | KING_TURN | 31/64 = 0.48 | -0.3 | flat |

Caveat: uniform noise measures how costly undirected choices are at a point. That depends on the option set (e.g. 16 of
KING_SPAWN's 22 options mean "spawn now"; BABY_ACT's 39 options include cheese-spending bites, traps, transfers,
disintegrate and coop-ending bites), not only on how precise the decision must be.

### 2b. Gradient signal (analysis/fisher.py; cpi-x0 states, v39 net). rl_train's PPO ratio uses the floor-mixed
probability p = (1-f) softmax(z) + f/m (scripts/rl_train.py ~380-384). The score of a sampled option is
(1-f) s_k (e_k - s) / p(k): for a floor-sampled option (s_k ~ 0) and for a saturated top choice (s_top ~ 1) it is ~0.
F = expected squared score per record (Fisher trace in logit space); a uniform policy has F = 1 - 1/m.

| point | F at floor .005, % of uniform | saturated rows (s_top>0.999) | share of F from floor picks | F with logits/3, % of uniform |
|---|---|---|---|---|
| 1 | 1.1% | 0.80 | 0.009 | 11.1% |
| 2 | 1.5% | 0.90 | 0.003 | 4.7% |
| 3 | 0.0% | 0.99 | 0.009 | 0.3% |
| 4 | 0.0% | 1.00 | 0.61 | 0.0% |
| 5 | 0.2% | 0.98 | 0.03 | 1.6% |
| 7 | 9.0% | 0.66 | 0.01 | 30.1% |
| 10 | 12.5% | 0.39 | 0.005 | 44.7% |
| 11 | 3.9% | 0.84 | 0.005 | 11.0% |
| 12 | 1.4% | 0.84 | 0.04 | 23.1% |
| 13 | 2.9% | 0.79 | 0.02 | 31.3% |
| 16 | 1.7% | 0.01 | 0.15 | 65.4% |

At cpi-x0's floor 0.1 the deviation rate at KING_SPAWN rose from 0.03 to 0.12 while F fell (0.037 -> 0.032): floor
exploration produces deviations the gradient cannot see. Saturated rows carry ~0 of F.

**Central diagnosis (the trap):** in this action space the policies that play well are near-deterministic and give
PPO almost no gradient; the policies with gradient (more entropy) play badly, because undirected choices at the sharp
points are costly. ES sees the same thing as a flat-then-cliff landscape: with huge logit gaps, small weight changes do
nothing until a decision flips. A redesign must make **stochastic policies competent** ("entropy tolerance"), so RL can
keep entropy, get gradient, and still play well.

### 2c. Fallback (cpi-x0 probe). When a sampled option cannot act, zdb's own code for that phase runs instead, but the
RL record keeps the sampled option and trains on it (D.fallback is a diagnostic flag only; rl_train.py ~525-541).

| point | deviation rate | deviations that fall back |
|---|---|---|
| BABY_ACT_POST (3) | 0.5% | 89% |
| KING_ACT (12) | 9.7% | 79% |
| BABY_ACT_PRE (2) | 1.5% | 64% (90% of PRE fallbacks: THROW while carrying nothing, caused by an untrained input) |
| KING_MOVE (10) | 17% | 58% (17.5% of all records fall back) |
| KING_SPAWN (11) | 12% | 39% |
| KING_FORM (13) | 11% | 37% |
| BABY_MOVE (1) | 1.2% | 13% |
| 5, 7, 8, 9, 14, 15, 16 | | 0% by construction (labels written after execution, or failures drop the record) |

### 2d. Code maps (read these). maps/map-*.json: seven read-only maps of every point (option kinds, why
fallback happens, context dependence, features, coupling, a generalization proposal with cost). maps/verify-*.json:
adversarial verifications of the maps (corrections, missed facts, proposal concerns); not all may exist yet, and a
correction overrides the map. Plain-text renderings of the maps: summary-1.txt (baby-move, baby-act-face),
summary-2.txt (baby-role-comms-mine-free, king-move-turn, king-spawn-act-form), summary-3.txt
(king-comms-and-plumbing, hidden-strategy).

Main structural findings (details and line numbers in the maps):
1. Most options are zdb cascade branches ("zdb's errand X") with zdb's target choice, thresholds and gates inside the
   executor. The net picks which zdb errand runs, not where to go or what to target. Alternatives a stronger player
   would want (another mine, contesting enemy-half mines, re-anchoring a king, intercepting a carrier, when to end
   cooperation, a bigger army) are mostly inexpressible. What can be expressed is mostly zdb errands run outside the
   region of the cascade they were written for (WAR at close range, DELIVER with 1 cheese, ATTACK for a gatherer).
2. Masks are looser than executors (fallback above); duplicates split one behaviour across labels (8 identical KING_MOVE
   stays, 5 identical KING_SPAWN holds, KING_FORM no-op/close families, BABY_MOVE near-duplicates); "shortcut + masked
   twin" encodings (R_KEEP with the current role masked; FR_ZDB). CORRECTED by the verifier: BABY_ROLE's flip-flopping
   is NOT caused by R_KEEP's bias; see 2e.
3. Uniform floors over act-heavy option sets explore "act more": 16/22 spawn options build, 17/25 form options open,
   10/11 KING_ACT options act; BABY_ACT's random picks spend cheese, kill own rats, or end cooperation for both teams.
4. Decisions with multi-turn consequences are re-decided every turn with no commitment (formation sites last a median
   of 1 extra record after a net open; a one-turn target switch wipes Nav's bug/flow progress; roles flip-flop).
5. Untrained inputs: columns constant in imitation (team rat-trap count, raider flag) keep random first-layer weights;
   widened columns (comms block, 80 BABY_ROLE columns) are zero. States only RL creates are met with random responses.
   Packed bitmask integers (canMove, dirtStep, bm/dm, rm) and direction ordinals cannot be decoded by an MLP.
6. Frozen and untrained points inject noise into every RL game (KING_COMMS floor: ~7.8 random protocol edits per
   team-game, ~3 into MINE slots; uniform raw squeaks that parse as zdb messages 5/16 of the time; random KING_TURN).
7. Hard-coded strategy no point controls: the round-30 backstab (Baby.hostile, BREAK_ROUND 30), the 18-baby cap and
   spawn floors, the king's home mine chosen once and never revised, home-half-only mine candidates, zdb's role mix at
   birth, delivery thresholds, engagement radii, the BUDGET economy signal, formation gates.
8. Credit: rl-m2/cpi-x0 use a "rich" team-trace critic (held-out AUC ~0.70) with GAE (gamma 1, lambda 0.8) over 10-round
   bins; every decision in a bin shares the team advantage (~1,200 BABY_MOVE decisions per team-game). Per-turn
   movement choices are nearly unidentifiable in that signal.
9. The king's single action per turn is split across KING_SPAWN, KING_ACT and KING_MOVE's digs, which compete for the
   same action, ring tiles and bank without seeing each other.
10. Only points 5, 7, 10, 11, 12, 13 (16 dropped after iteration 0) were ever trained by RL. Baby tactics (1-4), comms
   (6, 9, 14, 15) and BABY_FREE (8) have always been frozen.

### 2e. Verifier corrections (added 10:30 PDT; these override the maps and 2d)
From maps/verify-baby-move.json, verify-baby-act-face.json, verify-baby-role-comms-mine-free.json (the king and
hidden-strategy verifications are still running; check maps/ for verify-king-*.json and verify-hidden-strategy.json):
- BABY_ROLE flip-flopping comes from ONE poisoned input, not from R_KEEP: col 27 = cap(round - Baby.ekSeen, 255)
  (D.java:544) is written only by the net's R_ATTACK (Baby.java:1915) and never cleared, so it was always 255 in zdb-mode
  imitation data; after the net makes an attacker it reads 0-254 for 255 rounds and drives role logits to 100+ (86% of
  all flips). In normal states role logits are ~10 (point 5 was desaturated by tau 5), far below KEEP's 60.
- General training flaw: train_dp.py:120-121 computes mu/sd in float32 along the slow axis, so large-valued constant
  features escape the sd < 1e-3 guard and train as unit-scale biases. Every feature constant in zdb-mode data but
  written by net executors (ekSeen, raider, cash state from C_CASH_LEAD) is a latent out-of-range input. Fix: float64
  mu/sd or a max == min test; neutralise such columns exactly (zero the folded W1 column, fold its value into b1).
- zdb itself is hostile in cooperation from round 30 (Cfg.BREAK_ROUND = 30): it bites and grabs enemy rats in coop on
  purpose (hostile is true in ~96% of coop rows). Masking enemy bites/grabs in coop would mask zdb's own labels.
- Cat traps: catTrapsAllowed(team) = coop || (roundsSinceBackstab <= 100 && backstabber != team). The backstabber never
  places cat traps again; the victim only within 100 rounds. Backstab state (who, when) is not an input anywhere.
- Rat traps are backstab triggers: an enemy rat entering our rat trap calls backstab(our team) (GameWorld.java:639-642),
  and rat traps are legal in coop, so floor-sampled trap options end cooperation (~8.6 floor rat traps per team-game
  from the babies' 0.005 floor alone in cpi-x0).
- BABY_FREE's options (pickup, own-trap removal) are free engine actions (no cooldown); POST still opens after them.
- Facing is mostly set by BABY_MOVE (P_STEP, P_TURN, M_MINE_STILL and Nav/Micro turn; only P_STRAFE does not); FACE
  only controls the leftover turn. Cascade trap options use the front fan of the current facing (Baby.java:724, 857).
- ~60% of BABY_MOVE deviations come from the net's own T=1 softmax, not the floor (FLANK and AIM_CARRY >= 95%).
  Micro.act is a complete scoring micro and never falls back. Nav.guard is reset every turn.
- 90% of BABY_ACT_PRE fallbacks are THROW with nothing carried, caused by the untrained team rat-trap count (input 32);
  zeroing it brings argmax THROW back to the teacher's rate.
- Most zdb gates are already inputs (allCheese, coop, cat distance, king-to-cat distance, team cat traps, BUDGET); the
  missing ones are THROW's flight score, the cat's heading with fan-tile placeability, and the backstab state.
- MINE_PICK reproduces zdb's rule at 99.9% on zdb's distribution (81.7% in self-play, worse at large m); 80% of picks
  have enemy-half mines excluded. A pointer scorer is motivated by coverage and generalisation, not capacity.
- Enemy rats never hear squeaks; cats hear both teams but react only in attack mode.
- C_CASH_LEAD skips zdb's endgame gate (round >= 1880, one king, bank), so it is a king-formation trigger at any round.
- Dead zdb code: the ATTACK-only king bites at Baby.java:141 and 218-221 can never run.

## 3. Infrastructure facts (verify in code when it matters)
- Java: java/bots/zdp/D.java (point ids, features, take/emit/fallback, records), DNet.java (per-point MLP n_in-64-64-n_out,
  normalisation folded into layer 1; sample = (1-floor) softmax(z/T) + floor/m; separate RNG streams for points 8, 9, 14+),
  Baby.java (baby turn, cascade, netExec/netAllowed), King.java and DK.java (king turn, king points), Nav.java/Flow.java
  (movement), Micro.java (a complete scoring micro, net-reachable only through M_MICRO), Comms.java, Econ.java, Cfg.java.
- Records: one indicator string per decision (<= 256 chars, GameConstants), trailer = sampled k, q = round(-ln p * 4096)
  capped at 65535, recorded mask, flags (bit0 fallback, bit1 secondary).
- Python: scripts/dp_points.py and decisions.py (point table, record decoding), train_dp.py (imitation), widen_dpnet.py
  and dpnet_io.py (widening, probs, fold), build_dpnet.py (packages), rl_data.py, rl_critic.py, rl_train.py (PPO with
  clip, target_kl, entropy bonus, KL leash to an anchor with alarms), es_eval.sh / eval_student.py (games).
- Docs: docs/decision-points.md (the 16 points and their history), docs/capability-principle.md, docs/rl-es-plan.md
  (section 16 = this investigation), docs/engine.md, docs/rules/specs.txt and docs/rules/engine-checked-digest.md
  (game rules: rats, kings, cats, cheese, traps, dirt, throws, cooperation and backstab, scoring).
- Compute: a 20-CPU box (~4x the VM's game throughput) and a VM; a 64-game head-to-head takes ~12 minutes.

## 4. Requirements for the new action space
- R1 Entropy tolerance: a policy carrying substantial entropy at every trainable decision (e.g. Fisher trace >= 20% of
  uniform) must still play competently. Undirected choices should be coherent (sensible candidates, commitment), and
  destructive or irreversible actions must not be one uniform draw away.
- R2 Every offered option executes and has one meaning; no silent fallback; the record holds what was executed.
- R3 Capability-complete (owner's principle), with deliberate, documented exceptions only.
- R4 Temporal abstraction: multi-turn decisions commit until a termination condition (reached, invalid, interrupt event,
  timeout) instead of being re-decided every turn.
- R5 Credit: fewer, more consequential decisions, and a decision structure the RL estimator can credit (state any
  estimator changes you need, staying within RL).
- R6 A start policy: imitation of zdb is optional and may be approximate; the start must be competent (~40% vs field)
  and soft enough to have gradient (bounded logit gaps, e.g. label smoothing or temperature).
- R7 An incremental path in this codebase: what can reuse DNet/records/rl_train, what needs new infrastructure, and in
  which order to build and test it.
- R8 Measurable: the same floor-0.2 sensitivity table, the Fisher table, and a short RL run must be able to show the
  landscape changed.

## 5. What to deliver
A markdown design (aim for 2,000-4,000 words) with:
1. The core idea in one paragraph, and why it removes the trap in 2b.
2. The new decision structure: a table of points (new, kept, merged, dropped) with when each opens, its options or
   candidates, masks, key features, executor, commitment/termination, and the imitation label source.
3. Infrastructure changes (Java, records, Python), with file names.
4. How it meets R1-R8, point by point, including what stays hard-coded and why.
5. Start policy and RL plan (what is trainable first, floors/temperature, estimator changes).
6. Test plan with pass/fail criteria, cheapest first.
7. A staged implementation plan with rough effort, and the first stage small enough to test within a day.
8. Risks, and what you would cut if time ran short.
Check feasibility claims against the code; cite file:line for anything load-bearing.
