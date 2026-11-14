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

| Point | Where | Options |
|---|---|---|
| BABY_MOVE (1) | `Baby.moveDecision` | 39 branches of zdb's movement cascade, in priority order (`D.M_*`) |
| planned: king move, spawn, formation, baby role, action phase | | |

The features are a compact, zdb-computed summary: role, cheese, health, cat, enemies, targets, kings, mines, the alert,
counts in sight, position and so on (~45 small integers).

## Net mode (pass 1b)

A separate build (`D.NET` true, a compile-time constant, so the recording build has none of its code). At a point,
the net picks an option. zdp runs that option's behaviour; if the behaviour can't act, zdb's cascade runs as before.
Options zdb never takes (e.g. plain moves in each direction) get a small probability floor, so RL can find them.
The net is small: hidden 16–32, a few thousand weights, compiled into the bot.
Imitation trains it on the labels; RL then trains it in self-play (with RL against zdb allowed as a bootstrap).
