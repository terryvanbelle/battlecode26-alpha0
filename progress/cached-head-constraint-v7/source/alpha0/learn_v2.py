"""Game-balanced, checkpointed policy iteration for the versioned macro40 agent."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import torch

from .env import EngineEnv
from .features_v2 import ACTION_DIM, encode, make_evaluator
from .mcts import SearchConfig, search
from .network_v2 import NetworkConfig, PolicyValueNetV2, policy_value_loss
from .replay_v2 import GameReplayBuffer
from .train import append_json, atomic_json

ROOT = Path(__file__).resolve().parents[1]
PLAN_IDENTITY_VERSION = 2
TRAINING_RUNTIME_FILES = (
    "alpha0/learn_v2.py", "alpha0/replay_v2.py", "alpha0/mcts.py", "alpha0/env.py",
    "alpha0/features_v2.py", "alpha0/network_v2.py", "alpha0/network.py",
    "java/alpha0/MacroV2.java", "java/alpha0/EngineBridge.java",
    "java/alpha0/StateCopier.java", "java/alpha0/NoopGameMaker.java",
    "patches/local-research.patch", "vendor/engine/battlecode26-java-1.2.6.jar",
)


class RolloutInterrupted(Exception):
    """A cycle boundary pauses a game without fabricating a terminal result."""


def _file_hash(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _training_runtime_identity() -> dict:
    """Hash only inputs capable of changing self-play or its training targets.

    Source hashes alone cannot detect stale compiled Java classes. Record the
    executed class bytes too, without coupling training to evaluation/reporting
    implementation changes.
    """
    classes = sorted((ROOT / "build/java").rglob("*.class"))
    if not classes:
        raise FileNotFoundError("compiled engine is missing; run scripts/setup_engine.py")
    jdk = Path(os.environ.get("JAVA_HOME", ROOT / "vendor/jdk"))
    release = jdk / "release"
    return {"identity_version": PLAN_IDENTITY_VERSION,
            "files": {name: _file_hash(ROOT / name) for name in TRAINING_RUNTIME_FILES},
            "compiled_classes": {path.relative_to(ROOT).as_posix(): _file_hash(path) for path in classes},
            "jdk_release_sha256": _file_hash(release),
            "numpy_version": str(np.__version__), "torch_version": str(torch.__version__)}


def _job_identity(job: dict, runtime: dict | None = None) -> dict:
    runtime = _training_runtime_identity() if runtime is None else runtime
    return {"identity_version": PLAN_IDENTITY_VERSION,
            "runtime_sha256": _config_hash(runtime),
            "game_id": job["game_id"], "map": job["map"], "seed": job["seed"],
            "map_sha256": _file_hash(ROOT / "maps" / f"{job['map']}.map26"),
            "current_team": job["current_team"], "config_sha256": _config_hash(job["config"]),
            "checkpoint_sha256": _file_hash(job["checkpoint"]),
            "opponent_checkpoint_sha256": _file_hash(job["opponent"]) if job.get("opponent") else None}


def _write_rollout_result(destination: Path, row: dict) -> None:
    """Keep failures/interruption evidence as well as the current job result."""
    atomic_json(destination.with_name(f"{destination.stem}.attempt-{row['attempt']:04d}.json"), row)
    atomic_json(destination, row)


def load_checkpoint(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("model_version") != "dual_scale_v2":
        raise ValueError("v2 training requires a dual_scale_v2 checkpoint")
    model = PolicyValueNetV2(NetworkConfig(**data["network_config"]))
    model.load_state_dict(data["model_state_dict"])
    return model, data


def save_checkpoint(path, model, optimizer, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(model.checkpoint(optimizer_state_dict=optimizer.state_dict(),
                               torch_rng=torch.get_rng_state(), **metadata), temporary)
    temporary.replace(path)


def rollout(job: dict) -> dict:
    """One independent full game. Only searched decisions become policy targets."""
    config, destination = job["config"], Path(job["output"])
    torch.set_num_threads(config["torch_threads"])
    torch.manual_seed(job["seed"])
    rng = np.random.default_rng(job["seed"])
    positions, actions = [], []
    searched_count, search_nodes = 0, 0
    started = time.monotonic()
    cap = config["game_position_capacity"]
    row = {"game_id": job["game_id"], "map": job["map"], "seed": job["seed"],
           "source": "selfplay", "action_version": "macro_v2", "checkpoint": job["checkpoint"],
           "opponent_checkpoint": job.get("opponent"), "current_team": job["current_team"],
           "identity": job.get("identity"), "training_runtime": job.get("training_runtime"),
           "attempt": job.get("attempt", 1),
           "failure_count": job.get("failure_count", 0)}
    env = None

    def check_budget():
        if time.monotonic() >= job.get("deadline_monotonic", float("inf")):
            raise RolloutInterrupted("cycle deadline reached; unfinished game excluded from replay")
        if job.get("stop_file") and Path(job["stop_file"]).exists():
            raise RolloutInterrupted("stop requested; unfinished game excluded from replay")
        if time.monotonic() - started > config["game_timeout_seconds"]:
            raise TimeoutError("self-play game exceeded wall-clock budget; no value labels produced")
        if env is not None:
            remaining = min(job.get("deadline_monotonic", float("inf")) - time.monotonic(),
                            config["game_timeout_seconds"] - (time.monotonic() - started))
            env.response_timeout = min(config["response_timeout_seconds"], max(0.01, remaining))

    try:
        check_budget()
        if _job_identity(job) != job.get("identity"):
            raise ValueError("rollout checkpoint/opponent/configuration/runtime changed after scheduling")
        model, _ = load_checkpoint(job["checkpoint"])
        opponent = load_checkpoint(job["opponent"])[0] if job.get("opponent") else model
        evaluators = [make_evaluator(model), make_evaluator(model)]
        if job.get("opponent"):
            evaluators[1 - job["current_team"]] = make_evaluator(opponent)
        evaluate = lambda state: evaluators[state.player](state)
        search_config = SearchConfig(simulations=config["simulations"], max_depth=config["max_search_depth"])
        env = EngineEnv(action_version="macro_v2", response_timeout=config["response_timeout_seconds"])
        check_budget()
        state = env.reset(job["map"], job["seed"])

        def transition(position, action):
            check_budget()
            return env.transition(position, action)

        while not state.terminal:
            check_budget()
            if len(actions) >= config["max_decisions"]:
                raise TimeoutError("incomplete self-play game; no value labels produced")
            actor = next(r for r in state.raw["robots"] if r["id"] == state.raw["current_id"])
            current_player = not job.get("opponent") or state.player == job["current_team"]
            probability = config["king_search_fraction"] if actor["type"] == "RAT_KING" else config["search_fraction"]
            should_search = current_player and (searched_count == 0 or rng.random() < probability)
            temperature = 1.0 if state.raw["round"] <= config["temperature_rounds"] else config["temperature_late"]
            if should_search:
                result = search(state, evaluate, transition, ACTION_DIM, config=search_config,
                                rng=rng, add_root_noise=True)
                observation = encode(state)
                mask = np.zeros(ACTION_DIM, dtype=bool)
                mask[list(state.legal_actions)] = True
                position = {"local": observation.local.astype(np.float16),
                            "global_board": observation.global_board.astype(np.float16),
                            "scalars": observation.scalars, "policy": result.policy,
                            "legal_mask": mask, "player": state.player}
                searched_count += 1
                search_nodes += result.nodes
                if len(positions) < cap:
                    positions.append(position)
                else:
                    index = int(rng.integers(searched_count))
                    if index < cap:
                        positions[index] = position
                action = result.select_action(rng, temperature)
            else:
                probabilities, _ = evaluate(state)
                legal = np.asarray(state.legal_actions, dtype=int)
                t = temperature if current_player else config["opponent_temperature"]
                if t <= 0:
                    action = int(legal[np.argmax(probabilities[legal])])
                else:
                    weights = np.log(np.maximum(probabilities[legal], 1e-12)) / t
                    weights = np.exp(weights - weights.max())
                    action = int(rng.choice(legal, p=weights / weights.sum()))
            state = transition(state, action)
            env.release_except(state)
            actions.append(action)
        if not positions:
            raise RuntimeError("completed game produced no search targets")
        if _job_identity(job) != job["identity"]:
            raise ValueError("rollout inputs changed during execution; result excluded")
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.stack([p[key] for p in positions]) for key in positions[0]}
        npz = destination.with_suffix(".npz")
        temporary = npz.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(npz)
        row.update(status="ok", terminal_value_a=state.terminal_value_a, winner=state.raw["winner"],
                   rounds=state.raw["round"], decisions=len(actions), searched_positions=searched_count,
                   retained_positions=len(positions), search_nodes=search_nodes,
                   seconds=time.monotonic() - started, actions=actions, terminal=state.raw,
                   data_sha256=hashlib.sha256(npz.read_bytes()).hexdigest())
    except Exception as error:
        interrupted = (isinstance(error, RolloutInterrupted) or
                       time.monotonic() >= job.get("deadline_monotonic", float("inf")) or
                       bool(job.get("stop_file") and Path(job["stop_file"]).exists()))
        row.update(status="interrupted" if interrupted else "error", error=str(error),
                   failure_count=row["failure_count"] + (0 if interrupted else 1),
                   decisions=len(actions), actions=actions,
                   seconds=time.monotonic() - started)
    finally:
        if env is not None:
            env.close()
    _write_rollout_result(destination, row)
    return {k: v for k, v in row.items() if k not in ("actions", "terminal")}


class Learner:
    def __init__(self, run: Path, config: dict, progress: Path):
        self.run, self.config, self.progress = Path(run).resolve(), config, Path(progress).resolve()
        self.run.mkdir(parents=True, exist_ok=True)
        self.progress.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(config["torch_threads"])
        torch.manual_seed(config["seed"])
        self.rng = np.random.default_rng(config["seed"])
        self.iteration, self.games_total, self.positions_total, self.elapsed = 0, 0, 0, 0.0
        latest = self.run / "latest.pt"
        if latest.exists():
            self.model, data = load_checkpoint(latest)
            if data["config"] != config:
                raise ValueError("campaign config changed; start a versioned experiment, never silently resume")
            self.iteration, self.games_total = data["iteration"], data["games_total"]
            self.positions_total, self.elapsed = data["positions_total"], data["wall_seconds"]
            self.rng.bit_generator.state = data["numpy_rng"]
            torch.set_rng_state(data["torch_rng"])
            self.buffer = GameReplayBuffer.load(self.run / data.get("replay_path", "replay"))
        else:
            self.model = PolicyValueNetV2(NetworkConfig(**config["model"]))
            self.buffer = GameReplayBuffer(capacity=config["replay_capacity"], max_games=config["replay_max_games"],
                                           max_positions_per_game=config["game_position_capacity"], seed=config["seed"])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["learning_rate"],
                                           weight_decay=config["weight_decay"])
        if latest.exists():
            self.optimizer.load_state_dict(data["optimizer_state_dict"])
        atomic_json(self.run / "config.json", config)
        if not latest.exists():
            self.checkpoint()

    @property
    def checkpoint_path(self):
        return self.run / f"checkpoint-{self.iteration:04d}.pt"

    def checkpoint(self):
        # The checkpoint always names its immutable replay generation. An interrupted
        # next iteration cannot evict data required by the published latest model.
        replay_path = f"replay/iteration-{self.iteration:04d}"
        self.buffer.save(self.run / replay_path)
        metadata = dict(iteration=self.iteration, config=self.config, numpy_rng=self.rng.bit_generator.state,
                        games_total=self.games_total, positions_total=self.positions_total, wall_seconds=self.elapsed,
                        replay_path=replay_path)
        save_checkpoint(self.checkpoint_path, self.model, self.optimizer, **metadata)
        # Copy preserves exact checkpoint hash for deduplication and candidate identity.
        temporary = self.run / "latest.tmp"
        temporary.write_bytes(self.checkpoint_path.read_bytes())
        temporary.replace(self.run / "latest.pt")

    def _reload_published(self):
        """Discard a partial optimizer pass when a cycle stops between updates."""
        self.model, data = load_checkpoint(self.run / "latest.pt")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"],
                                           weight_decay=self.config["weight_decay"])
        self.optimizer.load_state_dict(data["optimizer_state_dict"])
        self.buffer = GameReplayBuffer.load(self.run / data.get("replay_path", "replay"))
        self.iteration, self.games_total = data["iteration"], data["games_total"]
        self.positions_total, self.elapsed = data["positions_total"], data["wall_seconds"]
        self.rng.bit_generator.state = data["numpy_rng"]
        torch.set_rng_state(data["torch_rng"])

    def _plan(self):
        path = self.run / f"plan-{self.iteration+1:04d}.json"
        runtime = _training_runtime_identity()
        if path.exists():
            plan = json.loads(path.read_text())
            if (plan.get("identity_version") != PLAN_IDENTITY_VERSION or
                    plan.get("training_runtime") != runtime or
                    plan.get("config_sha256") != _config_hash(self.config) or
                    plan.get("checkpoint_sha256") != _file_hash(self.checkpoint_path)):
                raise ValueError("saved game plan does not match this checkpoint/configuration/runtime version")
            if len(plan["jobs"]) != self.config["games_per_iteration"]:
                raise ValueError("saved game plan has an unexpected number of games")
            for job in plan["jobs"]:
                if job.get("training_runtime") != runtime or job.get("identity") != _job_identity(job, runtime):
                    raise ValueError("saved game plan inputs changed")
            return plan
        history = [path for path in sorted(self.run.glob("checkpoint-*.pt"))
                   if int(path.stem.split("-")[-1]) < self.iteration]
        jobs = []
        for index in range(self.config["games_per_iteration"]):
            game_id = f"iter-{self.iteration+1:04d}-game-{index:03d}"
            opponent = None
            if history and self.rng.random() < self.config["past_opponent_fraction"]:
                opponent = str(history[int(self.rng.integers(len(history)))])
            jobs.append({"game_id": game_id, "map": self.config["maps"][(self.games_total+index) % len(self.config["maps"])],
                         "seed": int(self.rng.integers(1, 2**31-1)), "config": self.config,
                         "checkpoint": str(self.checkpoint_path), "opponent": opponent,
                         # Alternate each repeated map's seat, including historical-opponent games.
                         "current_team": (index % len(self.config["maps"]) +
                                          index // len(self.config["maps"]) + self.iteration) % 2,
                         "output": str(self.run / "games" / f"{game_id}.json")})
            jobs[-1]["training_runtime"] = runtime
            jobs[-1]["identity"] = _job_identity(jobs[-1], runtime)
        plan = {"jobs": jobs, "numpy_rng_after_plan": self.rng.bit_generator.state,
                "identity_version": PLAN_IDENTITY_VERSION, "training_runtime": runtime,
                "config_sha256": _config_hash(self.config), "checkpoint_sha256": _file_hash(self.checkpoint_path)}
        atomic_json(path, plan)
        return plan

    def train_iteration(self, deadline: float, stop_file: Path) -> dict | None:
        started = time.monotonic()
        plan = self._plan()
        self.rng.bit_generator.state = plan["numpy_rng_after_plan"]
        pending = []
        attempts, failures, active = {}, {}, {}
        for job in plan["jobs"]:
            output = Path(job["output"])
            row = json.loads(output.read_text()) if output.exists() else None
            if row and row.get("identity") != job["identity"]:
                raise ValueError("saved rollout belongs to different training inputs")
            attempts[job["game_id"]] = row.get("attempt", 0) if row else 0
            failures[job["game_id"]] = row.get("failure_count", 0) if row else 0
            if row and row["status"] == "ok":
                continue
            if failures[job["game_id"]] >= self.config["max_game_attempts"]:
                raise RuntimeError(f"self-play failure budget exhausted: {row}")
            pending.append(job)
        # Launch only available slots; stop scheduling new games at a safe cycle boundary.
        with ProcessPoolExecutor(max_workers=self.config["workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
            while pending or active:
                while pending and len(active) < self.config["workers"] and time.monotonic() < deadline and not stop_file.exists():
                    job = pending.pop(0)
                    attempts[job["game_id"]] += 1
                    attempt_job = {**job, "attempt": attempts[job["game_id"]],
                                   "failure_count": failures[job["game_id"]],
                                   "deadline_monotonic": deadline, "stop_file": str(stop_file.resolve())}
                    active[pool.submit(rollout, attempt_job)] = attempt_job
                if not active:
                    break
                finished, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in finished:
                    job = active.pop(future)
                    try:
                        row = future.result()
                    except Exception as error:
                        row = {"status": "error", "game_id": job["game_id"], "error": str(error),
                               "identity": job["identity"], "attempt": job["attempt"],
                               "failure_count": job["failure_count"] + 1}
                        _write_rollout_result(Path(job["output"]), row)
                    print(json.dumps(row, allow_nan=False), flush=True)
                    failures[job["game_id"]] = row["failure_count"]
                    if row["status"] != "ok":
                        if row["status"] == "interrupted" or failures[job["game_id"]] < self.config["max_game_attempts"]:
                            pending.append(job)
                        else:
                            raise RuntimeError(f"repeated self-play failure: {row}")
        if pending:
            return None
        if time.monotonic() >= deadline or stop_file.exists():
            return None  # Keep completed episode shards; optimize on the next cycle.
        if _training_runtime_identity() != plan["training_runtime"]:
            raise ValueError("training runtime changed while episode collection was active")
        rows = []
        for job in plan["jobs"]:
            output = Path(job["output"])
            row = json.loads(output.read_text())
            if row["status"] != "ok":
                raise RuntimeError(f"uncompleted game {job['game_id']}")
            npz = output.with_suffix(".npz")
            if hashlib.sha256(npz.read_bytes()).hexdigest() != row["data_sha256"]:
                raise ValueError("rollout data hash mismatch")
            if job["game_id"] not in self.buffer:
                with np.load(npz, allow_pickle=False) as data:
                    arrays = {k: data[k] for k in data.files}
                    positions = [{k: v[index] for k, v in arrays.items()} for index in range(len(arrays["player"]))]
                    self.buffer.add_game(job["game_id"], positions, row["terminal_value_a"],
                                         {"source": "selfplay", "map": job["map"], "seed": job["seed"]}, completed=True)
            rows.append(row)
        self.model.train()
        metrics = []
        for _ in range(self.config["train_steps"]):
            if time.monotonic() >= deadline or stop_file.exists():
                self._reload_published()
                return None
            batch = {k: torch.as_tensor(v) for k, v in self.buffer.sample(self.config["batch_size"], self.rng).items()}
            logits, values = self.model(batch["local"], batch["global_board"], batch["scalars"], batch["legal_mask"])
            loss, measured = policy_value_loss(logits, values, batch["policy"], batch["value"])
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite policy/value loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
            self.optimizer.step()
            metrics.append(measured)
        if _training_runtime_identity() != plan["training_runtime"]:
            self._reload_published()
            raise ValueError("training runtime changed during optimization; partial update discarded")
        self.iteration += 1
        self.games_total += len(rows)
        self.positions_total += sum(row["retained_positions"] for row in rows)
        seconds = time.monotonic() - started
        self.elapsed += seconds
        self.checkpoint()
        means = {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]}
        summary = {"iteration": self.iteration, "games_total": self.games_total, "positions_total": self.positions_total,
                   "policy_loss": means["policy_loss"], "value_loss": means["value_loss"], "total_loss": means["loss"],
                   "entropy": means["policy_entropy"], "value_mse": means["value_mse"], "value_mean": means["value_mean"],
                   "value_saturated_fraction": means["value_saturated_fraction"],
                   "replay_games": self.buffer.num_games, "replay_positions": len(self.buffer),
                   "games_per_hour": len(rows)*3600/max(seconds, 1e-6), "wall_seconds": self.elapsed,
                   "actual_decisions": sum(row["decisions"] for row in rows),
                   "searched_positions": sum(row["searched_positions"] for row in rows),
                   "value_loss_type": "BCEWithLogits", "model_version": "dual_scale_v2"}
        append_json(self.progress / "training.jsonl", summary)
        print(json.dumps(summary), flush=True)
        return summary
