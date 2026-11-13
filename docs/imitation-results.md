# zdb imitation: results log

All games use official bytecode limits for the bot side. Times are PDT (UTC-7).

## Sanity check against zdb

The check is 32 games: 8 training-distribution maps (mincemeat, jail, rift, pipes, gridlock, thunderdome, sunandmoon, ZeroDay) × 2 seeds × both sides. A perfect copy would score about 50%, so a result well below that means a problem.

| Student | Data | Val acc | vs zdb T=0 | vs zdb T=1.0 |
|---|---|---|---|---|
| student-v1 | teacher-v1 (400 games, kings sampled 6%) | 0.848 | 0/32 | 1/32 |
| student-v2 | + teacher-v2 (400 games, kings every turn) | 0.863 | 0/32 | 1/32 |
| student-v3 | + teacher-v3-dagger (400 DAgger games, student-v2 at T=1.0 plays 50% of zdb turns) | 0.854* | 2/32 | 0/32 |
| student-v4 | + teacher-v4-dagger (DAgger with student-v3, rare baby states always recorded) | 0.846* | – | 0/32 |
| student-v5 | obs-v2 (team board); teacher-v5, 500 pure zdb games | 0.839 | – | 0/32 |
| student-v7 | obs-v3 (enemy memory); teacher-v7, 500 pure zdb games | 0.836 | – | 1/32 |

\* Validation for v3 and v4 includes the harder states the student itself reaches, so the accuracies are not directly comparable.

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

**student-v4: long maps nearly even, small maps still lost.** DAgger round 2 always recorded babies that could become king or saw an enemy. It was trained on all four collections (4.56M records, 8 epochs; val acc 0.846). Against zdb it scored 0/32 at T=1.0, but the losses changed:
- It now makes a second king. On rift it ends with 2–4 kings, against zdb's 2 (v3 always had 1).
- On the 2000-round maps it loses on points by much less. pipes: 48–51, 48–51, 49–50 and 38–61 (v3: about 38–61 in every game). rift: 40–59 to 47–52 (v3: 30–69 to 33–66). jail: 40–59 to 42–57.
- On ZeroDay it is unchanged at 30–36 points of 99, mainly because zdb does 3–5× its cat damage.
- On the small maps (gridlock, mincemeat, sunandmoon, thunderdome) and one jail seed, zdb still kills its king, in 27–530 rounds.

Selection panel at T=1.0, wins out of 8: examplefuncsplayer 6, uravt_Version35 0, AlexT101_finalsbot 0.

**Frozen gate, student-v4 at T=1.0** (held-out maps and opponents; [rule](../configs/student-gate-v1.json)). It **fails** on the held-out field.

| Group | Rule | student-v4 | zdb (reference) |
|---|---|---|---|
| untrained net | ≥ 0.80 | 0.97 | – |
| examplefuncsplayer + lectureplayer | ≥ 0.70 | 0.91 | 1.00 |
| held-out field (5 bots) | ≥ 0.20 | **0.15** (23/150) | 0.33 (52/156) |

Per held-out bot (student / zdb): VarunVejalla 0.00 / 0.47, GuzziFlipFlops 0.91 / 1.00, quesswho_quals 0.06 / 0.07, daannte_v47 0.00 / 0.00, AlexT101_usqualsbot 0.00 / 0.16. Games where the opponent's own code crashed are excluded. Against VarunVejalla, zdb wins most games on the small maps (tiny, knifefight, closeup), where the student's babies are wiped out and its king dies.

**Diagnosis: the student could not see what zdb's babies react to.** Per-group accuracy rules out tactics: attack 0.94, carry 0.96, transfer 0.98, move 0.92. zdb's babies, however, act on the team shared array, which the kings write:
- a fresh threatened-king alert sends nearby babies to defend (`Baby.threatenedKing`);
- live king positions from heartbeats set where babies regroup and deliver;
- a shared enemy-king sighting sends healthy babies to press.

Observation v1 held only what each robot had seen itself, so these decisions looked like noise in the data. That fits the early king deaths on small maps.

**Observation v2** adds a fixed team board that fits the kings-write shared array:
- Each king publishes its centre and a heartbeat every turn.
- A king that sees enemies publishes an alert, with threat = enemy babies within distance² 18.
- Enemy-king sightings are published.

Every robot reads the board as relative vectors to the nearest live king, the alert and the enemy king, plus a "within defend range of a fresh alert" flag. The student can run the same protocol itself in a tournament. Babies in defend range are now always recorded. The obs-v1 data cannot be reused, so the pipeline restarts: teacher-v5 (500 pure zdb games), student-v5, a DAgger round (teacher-v6-dagger, 400 games with student-v5), student-v6, then the zdb check and the gate ([script](../scripts/pipeline_obs_v2.sh)).

**student-v5 (obs-v2): still 0/32.** Long maps are about as before: jail 46–53 to 48–51, pipes 48–51 in two games. The small maps were still lost. Changing the sampling temperature (0, 0.3, 0.6, 1.0) didn't change the outcome on thunderdome, sunandmoon or gridlock. The obs-v2 DAgger round was stopped at 35/400 games after the next finding.

**Per-round traces found the mechanism.** Game.java gained `--trace N`, which writes a per-round table plus the student's death log. On gridlock (seed 23, T=1.0):
- Both sides hold 12 babies at round 20. By round 45 the student is down to 2, while zdb still has 12 and has spent only about 80 cheese.
- Most dead student babies had "turns since thrown or dropped" = 0, so zdb was grabbing and throwing them.

The engine lets a rat grab an enemy that can't see it, or one that is weaker by more than a health threshold. zdb guards against this by turning to face the nearest enemy that could grab it (`Baby.guardTurn`). That enemy may not be in view: zdb remembers enemies seen at the start of the turn and recent threats (`threatRound`, glance back), and it notices unseen damage (`blindHurt`). The student had none of this memory. Once it turned away, the enemy vanished from its observation. This fits its worst-copied action, baby turn, at 0.44 accuracy.

**Observation v3** adds:
- a local channel for enemy babies seen in the last 6 rounds, fading with age;
- the time since any enemy rat was seen;
- health lost since the robot's last turn;
- a count of remembered enemies that aren't in view now.

Everything restarts on obs-v3: teacher-v7 (500 pure zdb games), student-v7, teacher-v8-dagger, student-v8, then the zdb check and the gate ([script](../scripts/pipeline_obs_v3.sh)).

**student-v7 (obs-v3): 1/32 against zdb; the small maps are still lost.** Val acc 0.836, value MSE 1.39. Results by map:
- The only win was gridlock seed 22 (B side), where zdb's king died at round 308. The other small-map games (gridlock, mincemeat, sunandmoon, thunderdome) were all lost to a king kill, in 34–534 rounds.
- thunderdome is lost by round 60 every time, with no cheese moved by either side.
- The long maps are no better than v4–v5: pipes 38–40 points of 99, rift 28–40, ZeroDay 28–39, jail 38–43 or a king kill.

Enemy memory alone doesn't close the gap. The obs-v3 DAgger round (teacher-v8, student-v8, gate) keeps running as a diagnostic. The RL starting model will come from the capability-complete spec in [capability-principle.md](capability-principle.md).
