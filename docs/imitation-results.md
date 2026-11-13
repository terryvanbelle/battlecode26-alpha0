# zdb imitation: results log

All games use official bytecode limits for the bot side. Times are PDT (UTC-7).

## Sanity check against zdb

The check is 32 games: 8 training-distribution maps (mincemeat, jail, rift, pipes, gridlock, thunderdome, sunandmoon, ZeroDay) × 2 seeds × both sides. A perfect copy would score about 50%, so a result well below that means a problem.

| Student | Data | Val acc | vs zdb T=0 | vs zdb T=1.0 |
|---|---|---|---|---|
| student-v1 | teacher-v1 (400 games, kings sampled 6%) | 0.848 | 0/32 | 1/32 |
| student-v2 | + teacher-v2 (400 games, kings every turn) | 0.863 | 0/32 | 1/32 |
| student-v3 | + teacher-v3-dagger (400 DAgger games, student-v2 at T=1.0 plays 50% of zdb turns) | 0.854* | 2/32 | 0/32 |
| student-v4 | + teacher-v4-dagger (DAgger with student-v3, rare baby states always recorded) | running | | |

\* Validation for v3 includes the harder states the student itself reaches, so the accuracies are not directly comparable.

## What each round found

**student-v1: the king built itself to death.** It built a rat every round from round 1 to 41 while the cost rose from 10 to 110, then starved. zdb builds every round only while the cost is 10. With kings sampled at 6%, there were about 3k build examples, almost all from the opening, so the model learned "build when spawn tiles are free". Ruled out: Java/PyTorch parity (100%), action encode/decode, and memory parity.

**student-v2: the economy works.** Kings are now recorded every turn. In the traced rift game it won, transferring 11,675 cheese (v1: 515). Two problems remained:
- At T=0, greedy babies loop and the king starves, even against examplefuncsplayer.
- At T=1.0 it survives 2000 rounds on rift, ZeroDay, pipes and jail with transfers at 70–95% of zdb's, but loses on points. On small maps zdb kills its king early. It never builds a second king.

Selection panel, wins out of 8 (T = 0 / 0.3 / 1.0):

| Opponent | T=0 | T=0.3 | T=1.0 |
|---|---|---|---|
| examplefuncsplayer | 2 | 4 | 6 |
| uravt_Version35 | 0 | 0 | 0 |
| AlexT101_finalsbot | 0 | 0 | 0 |

T=1.0 is the deployment temperature.

**student-v3: DAgger round 1 made little difference.** It lasts longer on small maps (mincemeat about 270 rounds vs 210; sunandmoon 130–480 vs 33–99), but still loses to zdb.

**teacher-v4-dagger, now running.** A second king and defending fights are rare, decisive states. With babies sampled at 6%, there were almost no become-king examples. Babies that can become king or that see an enemy are now recorded every turn. The rule depends only on the state, so the learned P(action | state) stays unbiased.
