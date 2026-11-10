# Battlecode 2026 Alpha0

## Benchmark performance over time

![Win rate against every benchmark bot over time](progress/overview/benchmark-history.png)

**Best measured smoke result: 20/24 wins (83.3%).** Each column above tests all 12 bots on `tiny`, seed 0, in both seats. The lower panel shows each opponent, including **vibe26b_zdb**, separately. Experiment boundaries distinguish controller changes, initialization and subsequent learning. This small-map result is **not** an all-map strength claim.

[Chart data and interpretation](progress/overview/README.md) · [Partial all-map zdb head-to-head](progress/zdb-allmaps-cp2/README.md)

| Results to inspect | What they contain |
|---|---|
| [Benchmark history](progress/overview/README.md) | One overview of all complete smoke panels |
| [Warmstart / MCTS](progress/warmstart-v3/README.md) | Current measured champion, learning curves and opponent results |
| [Latest completed game experiment: V6](docs/terminal-v6-results.md) | Controlled head training, validation, all-bot evaluation and audit |
| [Neutral RL control](progress/campaign-v2/README.md) | Separate baseline experiment |
| [Design](DESIGN.md) | Architecture, observations, training and evaluation decisions |


Search-guided self-play in MIT Battlecode 2026's **official Java engine 1.2.6**. This is an initial CPU research experiment: an AlphaZero-style policy/value network, PUCT search, terminal game outcomes, replay training, and a frozen **12-bot panel**: ten external competitors from `battlecode26b-vibe`'s downloaded pool, the official example player, and that project's current accepted champion.

The active v2 experiment uses an **87,973-parameter dual-scale policy/value network**, 40 turn-level macros, and privileged local plus full-map observations. Search guides self-play; evaluation uses the greedy network. It is **not tournament compatible**. The original 17-action experiment remains available for reproducibility.

## Active autonomous campaign

[Neutral v2 results](progress/campaign-v2/README.md) · [Warmstart results](progress/warmstart-v3/README.md) · [Future-game value calibration](docs/v2-value-calibration.md) · [All-map victory criteria](docs/all-maps-victory.md) · [Resume and stop instructions](docs/autonomy.md)

The strongest measured initialization so far wins **20/24 smoke games**: [winner-only own-teacher prior](progress/bootstrap-v3-winners/README.md). It beats ten opponents in both seats on `tiny`, but still loses to `bestsofar` and `basic51`. Its first two [search-guided checkpoints](progress/warmstart-v3/README.md) also scored **20/24**, so these measurements have not shown improvement on that prior. The warmstart experiment and neutral v2 control have separate checkpoints and progress records; consult their status files for current execution state.

The completed [terminal root-ranking experiment](docs/root-ranking-v4.md) passed its precommitted mechanism rule: 418 valid continuations, eight roots, and a mean paired validation advantage of **0.21875** over shallow PUCT. Its [graphs and audited evidence](progress/root-ranking-v4/README.md) separate selection from independent validation. No weights changed and no benchmark games were played. The [policy-head learning pilot](docs/terminal-target-head-v5.md) stopped at its preregistered coverage check: 20 complete source games supplied 24 of 32 required roots, with all eight missing slots on `closeup`. [The coverage evidence and graphs](progress/terminal-target-head-v5/README.md) preserve this no-SGD result; the 20/24 benchmark score above remains unchanged. The [completed v6 pilot](docs/terminal-v6-results.md) filled all required inputs, ran 1,458 terminal target games, trained both heads and completed 72 fresh benchmark games. Both learned heads still scored **20/24**, with zero validation advantage. [Audited results and learning curves](progress/terminal-target-head-v6/README.md) preserve the negative result. The completed [v7 training-only diagnostic](docs/cached-head-constraint-v7.md) fitted all 32 selected actions at 2,048 updates with anchoring retained. Its six fixed snapshots establish fitting-budget evidence, not stronger play. Fresh-data transfer research is being reviewed next.

The goal is one frozen learned checkpoint that beats **each of the twelve opponents on every one of the 74 available maps**. Schema 2 of [`configs/victory_gate.json`](configs/victory_gate.json) requires all **888 opponent–map pairs** to score at least 60%, with at least 50% in each seat, independently in **3,552 standard games** and **7,104 fresh-seed confirmation games**. Scores cannot be pooled across maps, opponents or stages to rescue a failure. Both stages use the complete hash-pinned map inventory. No checkpoint has passed this gate.

Most evaluations use cheap small map subsets. The campaign automatically opens the expensive standard stage only after a complete smoke evaluation scores at least 50% against **every** opponent; an explicit periodic research review can also schedule an occasional all-map check. That screen is not qualification: the prior and its 20/24 checkpoints still fail it against two opponents. Confirmation locks one candidate per gate version; a failed confirmation consumes its four-seed set. A later attempt needs a new version and four unused seeds across all maps. `Stash`, `Hike`, `heart` and `ratsareants` remain excluded from gradient training, while all benchmark trajectories remain evaluation-only. See the [full protocol](docs/all-maps-victory.md).

Detailed [search-mechanism graphs](progress/root-ranking-v4/README.md) and [neutral control graphs](progress/campaign-v2/README.md) remain available separately.

```bash
bash scripts/bootstrap.sh
bash scripts/run_campaign.sh --hours 3
# Graceful stop; a later restart requires intentionally removing this marker.
touch runs/campaign/STOP
```

The campaign resumes model, optimizer, RNG, complete-game replay and evaluation cells. Each cycle has a time budget and a review boundary. Verified archives provide recovery through GitHub release assets or the separate `campaign-checkpoints` branch; the current credentials require the Git fallback. `nohup` alone cannot survive a cloud environment being reclaimed. See the [operating instructions](docs/autonomy.md) for current continuation setup and persistence limits.

## Initial v1 measurements

Initial run: **6 self-play games, 43,779 retained positions, 3 training iterations**. All four checkpoints completed the final 12-bot smoke panel: **96 evaluation games, no unresolved failures**. Final score is **5/24 wins (20.8%)**, matching the untrained baseline; this run does not demonstrate an overall strength improvement.

[Initial experiment report](docs/initial-results.md) · [Latest results and sample counts](progress/README.md) · [Benchmark provenance](docs/benchmarks.md) · [Model and experiment design](DESIGN.md) · [Engine changes](docs/engine.md)

The original graphs remain in [the initial progress report](progress/README.md). Every point comes from committed JSONL measurements; v2 has its own series. Evaluation uses the greedy network without search because legacy Java bots contain live threads that cannot be snapshotted. Search remains active in self-play. Graphs refresh after each iteration and evaluation stage; GitHub updates when those files are pushed.

## Run locally

Linux x86-64, Python 3.11+, and a JDK 21 are supported. The setup script downloads hash-pinned official engine assets and a local JDK if needed.

```bash
bash scripts/bootstrap.sh
.venv/bin/python -m pytest -q
.venv/bin/python -m alpha0.train train --config configs/smoke.json --run-dir runs/initial
```

The legacy v1 smoke experiment is deliberately small: 49,242 parameters, four search simulations per decision, two full games and 32 gradient updates per iteration. The active v2 configuration is [`configs/campaign-v2.json`](configs/campaign-v2.json); its architecture, corrected footprint attacks, sparse search and game-balanced replay are documented in [DESIGN.md](DESIGN.md).

```bash
# Continue the same run with its original configuration.
.venv/bin/python -m alpha0.train train --config configs/smoke.json --run-dir runs/initial --resume --iterations 20

# A larger, separate experiment (use a separate progress directory too).
.venv/bin/python -m alpha0.train train --config configs/train.json --run-dir runs/main --progress-dir runs/main/progress

# Evaluate a saved checkpoint against all 12 frozen bots, both seats.
.venv/bin/python -m alpha0.train evaluate runs/initial/latest.pt --suite smoke

# Rebuild the committed plots from measured logs.
.venv/bin/python scripts/plot_progress.py
```

Checkpoints, complete action histories, and bounded replay buffers live under ignored `runs/`. The first experiment's four checkpoints, six complete action transcripts and provenance are published under [`artifacts/initial`](artifacts/initial). To verify a published game transcript, run `.venv/bin/python scripts/replay_game.py artifacts/initial/games/iter-0003-game-001.json`. Do not load untrusted checkpoints. The engine and JDK downloads are excluded from Git; their URLs and hashes are recorded by the setup.

## Compute limits and portability

Learned turns bypass bytecode instrumentation. Legacy bots receive a one-billion-bytecode per-turn allowance, and the engine's cumulative team execution timeout is disabled. Java/Python retain ordinary process/host memory limits; no tournament-sized JVM heap cap is imposed. Per-request and per-game wall-clock watchdogs prevent stalled experiments from hanging indefinitely. [Exact changes and remaining limits](docs/engine.md).

Training currently runs on the attached CPU machine. **`battlecode-dev` is not being used.** Its historical configuration and the missing access prerequisites are in [compute notes](docs/compute.md). Existing VM workloads have not been changed.

## Sources

- [Official Battlecode 2026 repository](https://github.com/battlecode/battlecode26), engine, and [downloaded rules](docs/rules/specs.txt).
- [battlecode26b-vibe](https://github.com/terryvanbelle/battlecode26b-vibe), pinned at `f4d6ec971b13a20f7f38fb56d249f62c0251ed05`, supplies the benchmark roster, historical skill evidence, and map collection.
- External opponents are pinned individually in [the manifest](configs/benchmarks.json). Historical ratings are selection evidence, not ratings from this experiment.

See [third-party notices](THIRD_PARTY.md) for engine, map, and bot source attribution.
