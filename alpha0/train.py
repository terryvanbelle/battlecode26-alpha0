"""Resumable search-guided self-play and frozen-panel evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from .env import ACTION_DIM, IN_CHANNELS, SCALAR_DIM, EngineEnv, encode, make_evaluator
from .mcts import SearchConfig, search
from .network import NetworkConfig, PolicyValueNet, policy_value_loss
from .replay import Position, ReplayBuffer, label_game

ROOT = Path(__file__).resolve().parents[1]


def append_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path: Path, net, optimizer, iteration: int, config: dict,
                    rng, games_total: int, positions_total: int, wall_seconds: float) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(net.checkpoint(optimizer_state_dict=optimizer.state_dict(), iteration=iteration,
               config=config, numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
               games_total=games_total, positions_total=positions_total, wall_seconds=wall_seconds), temporary)
    temporary.replace(path)


def load_model(path: Path):
    # Load only checkpoints produced by this project, never an untrusted pickle.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    net = PolicyValueNet(NetworkConfig(**payload["network_config"]))
    net.load_state_dict(payload["model_state_dict"])
    net.eval()
    return net, payload


def refresh_graphs(progress: Path) -> None:
    subprocess.run([os.sys.executable, str(ROOT / "scripts/plot_progress.py"),
                    "--progress-dir", str(progress)], check=True, cwd=ROOT)


def self_play(net, config: dict, rng, map_name: str, seed: int, game_path: Path):
    started = time.monotonic()
    positions, actions, search_nodes = [], [], 0
    env = EngineEnv()
    try:
        state = env.reset(map_name, seed)
        evaluate = make_evaluator(net)
        cfg = SearchConfig(simulations=config["simulations"], max_depth=config["max_search_depth"])
        while not state.terminal:
            if len(actions) >= config["max_decisions"] or time.monotonic() - started > config["game_timeout_seconds"]:
                raise TimeoutError("self-play cutoff reached; incomplete game is excluded from training")
            result = search(state, evaluate, env.transition, ACTION_DIM, config=cfg, rng=rng, add_root_noise=True)
            board, scalars = encode(state)
            mask = np.zeros(ACTION_DIM, dtype=bool)
            mask[list(state.legal_actions)] = True
            position = Position(board.astype(np.float16), scalars, result.policy, mask, state.player)
            # Bound even a very long unfinished game's RAM. Reservoir selection is
            # uniform over its decisions; game outcomes still label only completed games.
            cap = config.get("game_position_capacity", config["replay_capacity"])
            if len(positions) < cap:
                positions.append(position)
            else:
                replace = int(rng.integers(len(actions) + 1))
                if replace < cap:
                    positions[replace] = position
            temperature = 1.0 if state.raw.get("round", 0) <= config["temperature_rounds"] else config["temperature_late"]
            action = result.select_action(rng, temperature)
            actions.append(action)
            search_nodes += result.nodes
            state = env.transition(state, action)
            env.release_except(state)
            if len(actions) % 1000 == 0:
                print(json.dumps({"selfplay_decisions": len(actions), "map": map_name,
                                  "round": state.raw.get("round"), "seconds": time.monotonic() - started}), flush=True)
        row = {"status": "ok", "map": map_name, "seed": seed, "actions": actions,
               "rounds": state.raw.get("round"), "terminal_value_a": state.terminal_value_a,
               "winner": state.raw.get("winner"), "terminal": state.raw,
               "seconds": time.monotonic() - started, "search_nodes": search_nodes,
               "decisions": len(actions), "retained_positions": len(positions)}
        atomic_json(game_path, row)
        return label_game(positions, state.terminal_value_a), row
    except Exception as exc:
        atomic_json(game_path, {"status": "error", "map": map_name, "seed": seed,
                    "actions": actions, "error": str(exc), "seconds": time.monotonic() - started})
        raise
    finally:
        env.close()


def compile_benchmarks(manifest: dict) -> None:
    javac = Path(os.environ["JAVA_HOME"]) / "bin/javac" if os.environ.get("JAVA_HOME") else ROOT / "vendor/jdk/bin/javac"
    compiler = str(javac) if javac.exists() else "javac"
    jar = ROOT / "vendor/engine/battlecode26-java-1.2.6.jar"
    for bot in manifest["bots"]:
        for file in bot["files"]:
            actual = hashlib.sha256((ROOT / file["path"]).read_bytes()).hexdigest()
            if actual != file["sha256"]:
                raise ValueError(f"frozen benchmark changed: {file['path']}")
        dest = ROOT / "build/benchmarks" / bot["id"]
        stamp = dest / ".source_hash"
        if stamp.exists() and stamp.read_text() == bot["java_tree_sha256"]:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run([compiler, "--release", "21", "-proc:none", "-nowarn", "-cp", str(jar),
                        "-d", str(dest), *map(str, sorted((ROOT / bot["path"]).rglob("*.java")))],
                       check=True, capture_output=True, text=True)
        stamp.write_text(bot["java_tree_sha256"])


def evaluate_checkpoint(net, checkpoint: Path, iteration: int, suite: str, progress: Path,
                        timeout: float = 1800, opponents: list[str] | None = None) -> None:
    manifest = json.loads((ROOT / "configs/benchmarks.json").read_text())
    compile_benchmarks(manifest)
    schedule = manifest["evaluation_protocol"][suite]
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    existing = []
    output = progress / "evaluations.jsonl"
    if output.exists():
        existing = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    done = {(r.get("checkpoint_sha256"), r["opponent"], r["map"], r["seed"], r["side"])
            for r in existing if r["status"] == "ok"}
    inference = make_evaluator(net)
    for bot in manifest["bots"]:
        if opponents and bot["id"] not in opponents:
            continue
        for map_name in schedule["maps"]:
            for seed in schedule["seeds"]:
                for player, side in enumerate(("A", "B")):
                    key = (checkpoint_hash, bot["id"], map_name, seed, side)
                    if key in done:
                        continue
                    row = {"iteration": iteration, "checkpoint": checkpoint.name,
                           "checkpoint_sha256": checkpoint_hash, "opponent": bot["id"], "map": map_name,
                           "seed": seed, "side": side, "suite": suite, "policy": "greedy-network-no-search",
                           "engine_version": "1.2.6", "status": "error", "score": None, "rounds": None}
                    started = time.monotonic()
                    env = None
                    try:
                        env = EngineEnv(opponent_package=bot["package"],
                                        opponent_classes=ROOT / "build/benchmarks" / bot["id"], learned_team=player)
                        state = env.reset(map_name, seed)
                        decisions = 0
                        while not state.terminal:
                            if time.monotonic() - started > timeout:
                                raise TimeoutError("benchmark game wall-clock limit reached")
                            policy, _ = inference(state)
                            action = max(state.legal_actions, key=lambda a: policy[a])
                            state = env.transition(state, int(action))
                            env.release_except(state)
                            decisions += 1
                        value = state.terminal_value_a * (1 if player == 0 else -1)
                        row.update(status="ok", score=(value + 1) / 2,
                                   rounds=state.raw.get("round"), decisions=decisions,
                                   winner=state.raw.get("winner"), terminal=state.raw)
                    except Exception as exc:
                        row["error"] = str(exc)
                    finally:
                        if env is not None:
                            env.close()
                    row["seconds"] = time.monotonic() - started
                    append_json(output, row)
                    print(json.dumps({k: row.get(k) for k in ("iteration", "opponent", "side", "map", "status", "score", "rounds", "error")}), flush=True)
        refresh_graphs(progress)


def run_training(args):
    config = json.loads(Path(args.config).read_text())
    run = Path(args.run_dir).resolve()
    progress = Path(args.progress_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    (run / "games").mkdir(exist_ok=True)
    torch.set_num_threads(config["torch_threads"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    replay = ReplayBuffer(config["replay_capacity"])
    iteration, games_total, positions_total, previous_wall = 0, 0, 0, 0.0
    if args.resume:
        net, payload = load_model(run / "latest.pt")
        # Continuations retain the experiment definition; --iterations is only the stop boundary.
        if config != payload["config"]:
            raise ValueError("resume config differs from checkpoint; use original config or start a new run")
        iteration, games_total, positions_total = payload["iteration"], payload["games_total"], payload["positions_total"]
        previous_wall = payload.get("wall_seconds", 0.0)
        rng.bit_generator.state = payload["numpy_rng"]
        torch.set_rng_state(payload["torch_rng"])
        replay = ReplayBuffer.load(run / f"replay-{iteration:04d}.npz")
    else:
        if (run / "latest.pt").exists():
            raise ValueError("run already exists; use --resume or a fresh --run-dir")
        net = PolicyValueNet(NetworkConfig(IN_CHANNELS, SCALAR_DIM, ACTION_DIM,
                                           channels=config["channels"], blocks=config["blocks"]))
    optimizer = torch.optim.AdamW(net.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    if args.resume:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    atomic_json(run / "config.json", config)
    started = time.monotonic()
    initial = run / "checkpoint-0000.pt"
    if not args.resume:
        save_checkpoint(initial, net, optimizer, 0, config, rng, 0, 0, 0)
        save_checkpoint(run / "latest.pt", net, optimizer, 0, config, rng, 0, 0, 0)
        replay.save(run / "replay-0000.npz")
    if not args.skip_eval:
        evaluate_checkpoint(net, run / f"checkpoint-{iteration:04d}.pt", iteration,
                            config["eval_suite"], progress, config["game_timeout_seconds"])
    stop = args.iterations if args.iterations is not None else config["iterations"]
    for step in range(iteration + 1, stop + 1):
        iteration_start = time.monotonic()
        game_rounds = []
        for game_index in range(config["games_per_iteration"]):
            map_name = config["maps"][(games_total) % len(config["maps"])]
            seed = int(rng.integers(1, 2**31 - 1))
            examples, game = self_play(net, config, rng, map_name, seed,
                                       run / "games" / f"iter-{step:04d}-game-{game_index:03d}.json")
            replay.extend(examples)
            games_total += 1
            positions_total += len(examples)
            game_rounds.append(game["rounds"])
            print(json.dumps({"selfplay_game": games_total, "iteration": step, "positions": len(examples),
                              "rounds": game["rounds"], "winner": game["winner"], "seconds": game["seconds"]}), flush=True)
        losses = []
        net.train()
        for _ in range(config["train_steps"]):
            batch = {k: torch.as_tensor(v) for k, v in replay.sample(config["batch_size"], rng).items()}
            logits, values = net(batch["board"], batch["scalars"], batch["legal_mask"])
            loss, metrics = policy_value_loss(logits, values, batch["policy"], batch["value"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            losses.append(metrics)
        elapsed = time.monotonic() - iteration_start
        wall = previous_wall + time.monotonic() - started
        metrics = {k: float(np.mean([row[k] for row in losses])) for k in losses[0]}
        row = {"iteration": step, "games_total": games_total, "positions_total": positions_total,
               "replay_positions": len(replay), "policy_loss": metrics["policy_loss"],
               "value_loss": metrics["value_loss"], "total_loss": metrics["loss"], "entropy": metrics["policy_entropy"],
               "games_per_hour": config["games_per_iteration"] * 3600 / elapsed,
               "wall_seconds": wall, "rounds_mean": float(np.mean(game_rounds)),
               "simulations": config["simulations"], "seed": config["seed"]}
        # Replay publishes before checkpoint: a committed latest always has its matching buffer.
        replay.save(run / f"replay-{step:04d}.npz")
        checkpoint = run / f"checkpoint-{step:04d}.pt"
        save_checkpoint(checkpoint, net, optimizer, step, config, rng, games_total, positions_total, wall)
        save_checkpoint(run / "latest.pt", net, optimizer, step, config, rng, games_total, positions_total, wall)
        append_json(progress / "training.jsonl", row)
        print(json.dumps(row), flush=True)
        refresh_graphs(progress)
        if not args.skip_eval and step % config["eval_every"] == 0:
            evaluate_checkpoint(net, checkpoint, step, config["eval_suite"], progress, config["game_timeout_seconds"])
    refresh_graphs(progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", default=str(ROOT / "configs/smoke.json"))
    train.add_argument("--run-dir", default=str(ROOT / "runs/initial"))
    train.add_argument("--progress-dir", default=str(ROOT / "progress"))
    train.add_argument("--iterations", type=int)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--skip-eval", action="store_true")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--suite", choices=("smoke", "standard"), default="smoke")
    evaluate.add_argument("--progress-dir", default=str(ROOT / "progress"))
    evaluate.add_argument("--opponents", nargs="+")
    evaluate.add_argument("--timeout", type=float, default=1800)
    commands.add_parser("compile-benchmarks")
    args = parser.parse_args()
    if args.command == "train":
        run_training(args)
    elif args.command == "evaluate":
        torch.set_num_threads(1)
        net, payload = load_model(args.checkpoint)
        evaluate_checkpoint(net, args.checkpoint, payload["iteration"], args.suite,
                            Path(args.progress_dir).resolve(), args.timeout, args.opponents)
    else:
        compile_benchmarks(json.loads((ROOT / "configs/benchmarks.json").read_text()))


if __name__ == "__main__":
    main()
