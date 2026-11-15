# Battlecode 2026 Alpha0

A research repository for learning a [Battlecode 2026](https://github.com/battlecode/battlecode26) bot in the official
Java engine (1.2.6). The long-term goal is an AlphaZero-style bot that keeps improving through self-play. Games run
at engine speed: the learned policy executes inside the Java engine, not in a separate process.

Current status, results and charts: **[PROGRESS.md](PROGRESS.md)**.

## Approach

The current line bootstraps from a strong hand-coded bot, then improves it by self-play.

1. **Teacher.** `zdb`, the best hand-coded bot from
   [battlecode26b-vibe](https://github.com/terryvanbelle/battlecode26b-vibe), wins about 60% against other
   hand-coded competition bots.
2. **Decision-point bot.** `java/bots/zdp` is a cleaned-up and bug-fixed copy of zdb. Its control choices (baby
   movement, actions, facing, role, comms, mine choice; king movement, spawning, actions, formation, comms and facing)
   go through 16 decision points. A small network per point, trained to imitate the hand-coded choice, can make each
   choice instead. The networks are compiled into the Java bot. The starting model can do anything the hand-coded
   bot can. [Decision points](docs/decision-points.md) · [Capability principle](docs/capability-principle.md)
3. **Self-play improvement.** Only self-play games train the networks. Bots from other competitors are used only
   as benchmarks. Methods tried so far: PPO with a learned critic and GAE
   ([plan](docs/rl-plan.md)), and antithetic evolution strategies on per-point output biases
   ([plan](docs/rl-es-plan.md)).

Earlier lines, kept for reproducibility:
- **Macro-action AlphaZero (V1–V11):** a policy/value network with PUCT search over 17–40 turn-level macros and
  privileged full-state features (`alpha0/`, [DESIGN.md](DESIGN.md)). It reached about 32% against the development
  benchmark panel.
- **Primitive-action imitation (student v1–v10):** a per-robot network choosing one primitive API action at a time
  from its own observations ([pipeline](docs/lean-imitation.md), [results](docs/imitation-results.md)). It could not
  reproduce zdb's play well enough, which led to the decision-point design.

## Repository layout

| Path | Contents |
| --- | --- |
| `java/bots/zdp` | The decision-point bot: zdb's logic, the decision points, and the in-engine network runtime (`DNet`). |
| `java/student`, `java/imitation`, `java/alpha0` | Engine-side code for the student (API hooks, in-engine net), the primitive recorder and replay audit, and the macro-action line. |
| `alpha0/` | Python: macro-action AlphaZero (`train`, `mcts`, `network*`), student imitation (`student/`, `imitation/`). |
| `scripts/` | Game running (`eval_student.py`, `replay_check.py`), net builds (`build_dpnet.py`, `widen_dpnet.py`), the RL and ES loops (`rl_loop.sh`, `es_loop.sh` and their stages), multi-host game splitting (`box_split.sh`, `vm_helper.sh`), plotting. |
| `configs/` | Experiment configurations (RL and ES runs, benchmark panels, teacher collection). |
| `docs/` | Designs, plans and result write-ups; `docs/img/` holds the progress charts. |
| `progress/`, `artifacts/`, `diagnostics/` | Committed measurements, graphs, selected checkpoints and audits of the earlier lines. |
| `benchmarks/`, `maps/` | Opponent provenance and licences; the map collection. |
| `patches/` | The local research patch to the engine ([engine changes](docs/engine.md)). |
| `tests/` | Unit and plumbing tests (`pytest`). |

Run state (games, nets, checkpoints) lives in the ignored `runs/` and `build/` directories.

## Run locally

Linux x86-64, Python 3.11+ and JDK 21 are supported. The setup script downloads hash-pinned official engine assets
and a local JDK if needed. zdb and the field bots are read from a local checkout of battlecode26b-vibe at
`~/projects/vibe/2026b`.

```bash
bash scripts/bootstrap.sh
.venv/bin/python -m pytest -q

# Play zdp against zdb on one map, both sides (no bytecode limits unless --limits).
.venv/bin/python scripts/eval_student.py --player zdp --opponents zdb --maps jail --seeds 21 --sides AB --out runs/eval/try

# Build a net-mode bot from trained decision nets, then play it like any other bot.
.venv/bin/python scripts/build_dpnet.py --net runs/dp/net-v2/net.json --name zdpn2

# Self-play runs (resumable; stop with touch runs/rl/RUN/STOP or runs/es/RUN/STOP).
bash scripts/rl_loop.sh configs/rl-m2.json
bash scripts/es_loop.sh configs/es-m2.json
```

## Compute limits and portability

Experiments run without tournament bytecode limits for now, so the bots are **not yet tournament compatible**.
[Tournament transition](docs/tournament-transition.md) lists what that requires. The engine patch and remaining
limits are in [docs/engine.md](docs/engine.md). A run's games can be split across two hosts; the loops wait for
the second host's share and fill in anything it doesn't finish.

## Sources

- [Official Battlecode 2026 repository](https://github.com/battlecode/battlecode26), engine, and
  [downloaded rules](docs/rules/specs.txt).
- [battlecode26b-vibe](https://github.com/terryvanbelle/battlecode26b-vibe) supplies zdb, the benchmark roster,
  historical skill evidence and the map collection.
- External opponents are pinned individually in [the manifest](configs/benchmarks.json). Historical ratings are
  selection evidence, not ratings from this experiment.

See [third-party notices](THIRD_PARTY.md) for engine, map and bot source attribution.
