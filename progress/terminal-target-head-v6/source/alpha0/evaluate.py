"""Identity-safe, resumable evaluation of v1/v2 checkpoints against the frozen panel.

Each cell runs in a subprocess and publishes one atomic artifact per attempt.
Only infrastructure failures may retry; successful cells are never replayed.
The existing alpha0.train evaluation API and legacy progress rows are untouched.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_VERSION = 2
RETRYABLE_ERRORS = {"worker_timeout", "worker_process_error", "engine_process_error", "runtime_changed"}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for index, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{index}; refusing to discard evidence") from exc
    return rows


def panel_fingerprint(manifest: dict) -> str:
    return fingerprint(sorted(({key: bot[key] for key in ("id", "source_commit", "java_tree_sha256")}
                               for bot in manifest["bots"]), key=lambda item: item["id"]))


def policy_schema(payload: dict) -> dict:
    version = payload.get("model_version", "residual_v1")
    if version == "dual_scale_v2":
        expected_dim, feature_version, action_version = 40, 2, "macro_v2"
    elif version == "residual_v1":
        expected_dim, feature_version, action_version = 17, 1, "macro_v1"
    else:
        raise ValueError(f"Unsupported checkpoint model_version {version!r}")
    action_dim = int(payload["network_config"]["action_dim"])
    if (action_dim != expected_dim or int(payload.get("feature_version", feature_version)) != feature_version or
        payload.get("action_version", action_version) != action_version):
        raise ValueError("Checkpoint architecture and feature/action schema disagree")
    return {"model_version": version, "feature_version": feature_version,
            "action_version": action_version, "action_dim": action_dim,
            "network_config": payload["network_config"]}


def load_policy(checkpoint: Path):
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    schema = policy_schema(payload)
    if schema["model_version"] == "dual_scale_v2":
        from .network_v2 import NetworkConfig, PolicyValueNetV2
        from .features_v2 import make_evaluator
        net = PolicyValueNetV2(NetworkConfig(**payload["network_config"]))
    else:
        from .network import NetworkConfig, PolicyValueNet
        from .env import make_evaluator
        net = PolicyValueNet(NetworkConfig(**payload["network_config"]))
    net.load_state_dict(payload["model_state_dict"])
    net.eval()
    return net, make_evaluator(net), payload, schema


def runtime_identity(schema: dict, root: Path = ROOT) -> dict:
    import numpy
    import torch
    jar = root / "vendor/engine/battlecode26-java-1.2.6.jar"
    classes = sorted((root / "build/java").rglob("*.class"))
    if not classes:
        raise FileNotFoundError("No compiled bridge classes; run scripts/setup_engine.py")
    bundle = {p.relative_to(root).as_posix(): file_hash(p) for p in classes}
    adapters = ["alpha0/env.py", "alpha0/evaluate.py"]
    adapters += (["alpha0/network_v2.py", "alpha0/features_v2.py"]
                 if schema["action_version"] == "macro_v2" else ["alpha0/network.py"])
    python_sources = {name: file_hash(root / name) for name in adapters}
    jdk = Path(os.environ.get("JAVA_HOME", root / "vendor/jdk"))
    release = jdk / "release"
    if not release.exists():
        raise FileNotFoundError("A pinned JDK or JAVA_HOME with a release file is required")
    patch = root / "patches/local-research.patch"
    return {"engine_version": "1.2.6", "engine_jar_sha256": file_hash(jar),
            "engine_patch_sha256": file_hash(patch), "bridge_sha256": fingerprint(bundle),
            "compiled_classes": bundle, "python_adapter_sha256": fingerprint(python_sources),
            "python_sources": python_sources, "jdk_release_sha256": file_hash(release),
            "torch_version": str(torch.__version__), "numpy_version": str(numpy.__version__),
            "physics": {"engine_rules": "official-1.2.6", "benchmark_bytecode_limit": 1_000_000_000,
                        "team_execution_timeout_disabled": True}}


def make_context(checkpoint_hash: str, schema: dict, runtime: dict, *, timeout_seconds: float,
                 search_mode: str = "greedy", search_config: dict | None = None) -> dict:
    if search_mode != "greedy":
        raise ValueError("Only greedy evaluation is currently supported; search modes must never silently fall back")
    if search_config:
        raise ValueError("Greedy evaluation does not accept a search configuration")
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise ValueError("timeout_seconds must be finite and positive")
    config = {"policy_schema": schema, "runtime": runtime, "search_mode": search_mode,
              "search_config": {}, "timeout_seconds": float(timeout_seconds), "torch_threads": 1,
              "tie_break": "first-legal-action", "policy_random_seed": 0}
    return {"checkpoint_sha256": checkpoint_hash, "evaluation_config_sha256": fingerprint(config),
            "evaluation_config": config}


def build_cell_specs(manifest: dict, schedule: dict, context: dict, *, checkpoint: Path,
                     iteration: int, suite: str, stage: str, maps_dir: Path,
                     opponents: list[str] | None = None) -> list[dict]:
    selected = set(opponents) if opponents is not None else {b["id"] for b in manifest["bots"]}
    unknown = selected - {b["id"] for b in manifest["bots"]}
    if unknown or not selected:
        raise ValueError(f"Invalid opponent selection: {sorted(unknown)}")
    maps, seeds = schedule["maps"], schedule["seeds"]
    sides = schedule.get("sides", ["A", "B"])
    if not maps or not seeds or set(sides) != {"A", "B"}:
        raise ValueError("Evaluation schedules require maps, seeds and both sides")
    if len(set(maps)) != len(maps) or len(set(seeds)) != len(seeds) or len(sides) != 2:
        raise ValueError("Duplicate scheduled cells would overstate completion")
    map_hashes = {name: file_hash(maps_dir / f"{name}.map26") for name in maps}
    cells = []
    for bot in manifest["bots"]:
        if bot["id"] not in selected:
            continue
        for map_name in maps:
            for seed in seeds:
                for side in sides:
                    identity = {"artifact_version": ARTIFACT_VERSION, **context,
                                "benchmark_suite_id": manifest["suite_id"],
                                "panel_fingerprint_sha256": panel_fingerprint(manifest),
                                "opponent": bot["id"], "opponent_java_sha256": bot["java_tree_sha256"],
                                "opponent_source_commit": bot["source_commit"],
                                "map": map_name, "map_sha256": map_hashes[map_name],
                                "seed": int(seed), "side": side, "stage": stage}
                    cell_id = fingerprint(identity)
                    cells.append({"evaluation_cell_id": cell_id, "identity": identity,
                                  "checkpoint_path": str(checkpoint.resolve()), "iteration": int(iteration),
                                  "generation": int(iteration), "suite": suite, "stage": stage,
                                  "maps_dir": str(maps_dir.resolve()), "bot": bot})
    return cells


def prepare_benchmark(bot: dict, runtime: dict, root: Path = ROOT) -> Path:
    # Content-addressed build directories cannot retain stale classes from an older bot.
    for file in bot["files"]:
        if file_hash(root / file["path"]) != file["sha256"]:
            raise ValueError(f"Frozen benchmark source changed: {file['path']}")
    build_id = fingerprint({"source": bot["java_tree_sha256"], "engine": runtime["engine_jar_sha256"],
                            "jdk": runtime["jdk_release_sha256"]})
    destination = root / "build/evaluation-benchmarks" / bot["id"] / build_id
    with locked(destination.with_suffix(".lock")):
        if (destination / ".complete").exists():
            expected = json.loads((destination / ".class-hashes.json").read_text())
            actual = {p.relative_to(destination).as_posix(): file_hash(p) for p in sorted(destination.rglob("*.class"))}
            if actual != expected:
                raise RuntimeError("runtime identity changed: compiled opponent classes were modified")
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(dir=destination.parent, prefix=".compile-"))
        try:
            jdk = Path(os.environ.get("JAVA_HOME", root / "vendor/jdk"))
            subprocess.run([str(jdk / "bin/javac"), "--release", "21", "-proc:none", "-nowarn",
                            "-cp", str(root / "vendor/engine/battlecode26-java-1.2.6.jar"),
                            "-d", str(temporary), *map(str, sorted((root / bot["path"]).rglob("*.java")))],
                           check=True, capture_output=True, text=True)
            hashes = {p.relative_to(temporary).as_posix(): file_hash(p) for p in sorted(temporary.rglob("*.class"))}
            atomic_json(temporary / ".class-hashes.json", hashes)
            (temporary / ".complete").write_text(build_id)
            os.replace(temporary, destination)
        finally:
            import shutil
            if temporary.exists():
                shutil.rmtree(temporary)
    return destination


def result_base(spec: dict, attempt: int) -> dict:
    identity = spec["identity"]
    config = identity["evaluation_config"]
    runtime = config["runtime"]
    return {"artifact_version": ARTIFACT_VERSION, "evaluation_cell_id": spec["evaluation_cell_id"],
            "identity": identity, "attempt": attempt, "iteration": spec["iteration"],
            "generation": spec["generation"], "checkpoint": f"{Path(spec['checkpoint_path']).name} [{identity['checkpoint_sha256'][:8]}/{config['policy_schema']['action_version']}/{config['search_mode']}/{identity['evaluation_config_sha256'][:8]}]",
            "checkpoint_path": spec["checkpoint_path"], "suite": spec["suite"], "stage": spec["stage"],
            **{key: identity[key] for key in ("checkpoint_sha256", "evaluation_config_sha256", "benchmark_suite_id", "panel_fingerprint_sha256", "opponent", "opponent_java_sha256", "opponent_source_commit", "map", "map_sha256", "seed", "side")},
            **{key: runtime[key] for key in ("engine_version", "engine_jar_sha256", "engine_patch_sha256", "bridge_sha256")},
            "action_schema": config["policy_schema"]["action_version"],
            "observation_schema": config["policy_schema"]["feature_version"],
            "policy": "greedy-network-no-search", "search_mode": config["search_mode"],
            "status": "error", "score": None, "rounds": None, "started_at": datetime.now(timezone.utc).isoformat()}


def classify_error(error: Exception) -> str:
    message = str(error)
    if "Benchmark sandbox failure" in message or "unexpected_sandbox_exit" in message:
        return "opponent_runtime_error"
    if isinstance(error, TimeoutError):
        return "worker_timeout"
    if "engine stdout closed" in message or "engine exited" in message:
        return "engine_process_error"
    if "runtime identity changed" in message:
        return "runtime_changed"
    return "evaluation_error"


def run_cell_worker(spec: dict, attempt: int) -> dict:
    import torch
    from .env import EngineEnv
    torch.set_num_threads(1)
    torch.manual_seed(0)
    row = result_base(spec, attempt)
    started, env, decisions = time.monotonic(), None, 0
    identity = spec["identity"]
    config = identity["evaluation_config"]
    try:
        checkpoint = Path(spec["checkpoint_path"])
        if file_hash(checkpoint) != identity["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after scheduling")
        _, infer, _, schema = load_policy(checkpoint)
        if file_hash(checkpoint) != identity["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed while loading")
        if schema != config["policy_schema"]:
            raise ValueError("Checkpoint policy schema changed")
        if runtime_identity(schema) != config["runtime"]:
            raise RuntimeError("runtime identity changed after scheduling")
        if file_hash(Path(spec["maps_dir"]) / f"{identity['map']}.map26") != identity["map_sha256"]:
            raise ValueError("Map changed after scheduling")
        classes = prepare_benchmark(spec["bot"], config["runtime"])
        env = EngineEnv(opponent_package=spec["bot"]["package"], opponent_classes=classes,
                        learned_team=0 if identity["side"] == "A" else 1,
                        maps_dir=spec["maps_dir"], action_version=schema["action_version"],
                        response_timeout=min(120.0, config["timeout_seconds"]))
        state = env.reset(identity["map"], identity["seed"])
        if state.raw.get("action_version", "macro_v1") != schema["action_version"]:
            raise ValueError("Engine returned a different action schema than the checkpoint requested")
        while not state.terminal:
            if time.monotonic() - started > config["timeout_seconds"]:
                raise TimeoutError("Benchmark game exceeded its wall-clock limit")
            policy, _ = infer(state)
            if not state.legal_actions or any(not math.isfinite(float(policy[a])) for a in state.legal_actions):
                raise ValueError("Policy produced no finite legal action")
            action = max(state.legal_actions, key=lambda a: float(policy[a]))
            state = env.transition(state, int(action))
            env.release_except(state)
            decisions += 1
        if runtime_identity(schema) != config["runtime"]:
            raise RuntimeError("runtime identity changed during the game; result excluded")
        classes = prepare_benchmark(spec["bot"], config["runtime"])
        row["opponent_classes_sha256"] = fingerprint(json.loads((classes / ".class-hashes.json").read_text()))
        value = state.terminal_value_a * (1 if identity["side"] == "A" else -1)
        row.update(status="ok", score=(value + 1) / 2, rounds=state.raw.get("round"),
                   winner=state.raw.get("winner"), terminal=state.raw)
        if not valid_result(spec, row):
            row.update(status="error", score=None, error="Engine returned an invalid/inconsistent terminal game", error_kind="invalid_game")
    except Exception as error:
        row.update(error=str(error), error_kind=classify_error(error))
    finally:
        if env is not None:
            env.close()
    row.update(seconds=time.monotonic() - started, decisions=decisions)
    return row


def valid_result(spec: dict, row: dict) -> bool:
    identity = spec["identity"]
    terminal = row.get("terminal", {})
    if not isinstance(terminal, dict):
        return False
    score, rounds = row.get("score"), row.get("rounds")
    if (row.get("evaluation_cell_id") != spec["evaluation_cell_id"] or row.get("identity") != identity or
        row.get("status") != "ok" or type(score) not in (int, float) or score not in (0, 0.5, 1) or
        type(rounds) is not int or rounds <= 0 or terminal.get("terminal") is not True or
        terminal.get("valid_game") is not True or terminal.get("benchmark_failures") != [] or
        terminal.get("round") != rounds or terminal.get("map") != identity["map"] or
        terminal.get("seed") != identity["seed"] or
        terminal.get("action_version") != identity["evaluation_config"]["policy_schema"]["action_version"]):
        return False
    winner = terminal.get("winner")
    if type(winner) is not int or winner not in (-1, 0, 1) or row.get("winner") != winner:
        return False
    expected_score = 0.5 if winner == -1 else float(winner == (0 if identity["side"] == "A" else 1))
    return score == expected_score


def selected_result(spec: dict, rows: list[dict]) -> dict | None:
    relevant = [r for r in rows if r.get("evaluation_cell_id") == spec["evaluation_cell_id"] and r.get("identity") == spec["identity"]]
    # First successful completion is immutable, even if later manual attempts exist.
    for row in sorted(relevant, key=lambda r: int(r.get("attempt", 0))):
        if valid_result(spec, row):
            return row
    return max(relevant, key=lambda r: int(r.get("attempt", 0)), default=None)


def completion_report(specs: list[dict], rows: list[dict], *, max_attempts: int = 2,
                      required_opponents: int | None = None) -> dict:
    if not specs or len({s["evaluation_cell_id"] for s in specs}) != len(specs):
        raise ValueError("Completion requires a nonempty, unique expected-cell schedule")
    groups: dict[str, dict] = {}
    completed = errors = exhausted = 0
    pending_ids = []
    for spec in specs:
        identity = spec["identity"]
        opponent = identity["opponent"]
        stats = groups.setdefault(opponent, {"expected": 0, "completed": 0, "errors": 0,
                       "wins": 0, "draws": 0, "losses": 0, "score_sum": 0.0,
                       "sides": {side: {"expected": 0, "completed": 0, "score_sum": 0.0} for side in ("A", "B")}})
        stats["expected"] += 1
        side = stats["sides"][identity["side"]]
        side["expected"] += 1
        result = selected_result(spec, rows)
        if result is not None and valid_result(spec, result):
            completed += 1; stats["completed"] += 1; side["completed"] += 1
            score = result["score"]
            stats["score_sum"] += score; side["score_sum"] += score
            stats[{1: "wins", 0.5: "draws", 0: "losses"}[score]] += 1
        else:
            pending_ids.append(spec["evaluation_cell_id"])
            if result is not None:
                errors += 1; stats["errors"] += 1
                if result.get("error_kind") not in RETRYABLE_ERRORS or int(result.get("attempt", 0)) >= max_attempts:
                    exhausted += 1
    for stats in groups.values():
        stats["mean_score"] = stats["score_sum"] / stats["completed"] if stats["completed"] else None
        for side in stats["sides"].values():
            side["mean_score"] = side["score_sum"] / side["completed"] if side["completed"] else None
    full_panel = required_opponents is None or len(groups) == required_opponents
    return {"expected": len(specs), "completed": completed, "unresolved_errors": errors,
            "exhausted_cells": exhausted, "pending": len(pending_ids), "pending_cell_ids": pending_ids,
            "complete": completed == len(specs), "full_panel": full_panel,
            "full_panel_complete": full_panel and completed == len(specs), "opponents": groups}


def gate_stage_report(specs: list[dict], rows: list[dict], stage_config: dict, *, required_opponents: int) -> dict:
    report = completion_report(specs, rows, required_opponents=required_opponents)
    for stats in report["opponents"].values():
        stats["passed"] = (stats["completed"] == stats["expected"] and
                           stats["mean_score"] is not None and stats["mean_score"] >= stage_config["min_score"] and
                           all(s["completed"] == s["expected"] and s["mean_score"] is not None and
                               s["mean_score"] >= stage_config["min_score_each_side"] for s in stats["sides"].values()))
    report["passed"] = report["full_panel_complete"] and report["unresolved_errors"] == 0 and all(s["passed"] for s in report["opponents"].values())
    return report


def attempt_rows(directory: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(directory.glob("attempt-*.json"))]


def run_cell(spec: dict, store: Path, max_attempts: int) -> dict:
    directory = store / spec["evaluation_cell_id"]
    with locked(directory / ".lock"):
        rows = attempt_rows(directory)
        result = selected_result(spec, rows)
        if result is not None and valid_result(spec, result):
            return result
        if result is not None and result.get("error_kind") not in RETRYABLE_ERRORS:
            return result
        atomic_json(directory / "request.json", spec)
        attempt = max((int(r["attempt"]) for r in rows), default=0)
        while attempt < max_attempts:
            attempt += 1
            target = directory / f"attempt-{attempt:04d}.json"
            log = directory / f"attempt-{attempt:04d}.log"
            started = time.monotonic()
            with log.open("wb") as output:
                process = subprocess.Popen([sys.executable, "-m", "alpha0.evaluate", "--worker",
                            str(directory / "request.json"), "--worker-output", str(target), "--attempt", str(attempt)],
                            cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    process.wait(timeout=spec["identity"]["evaluation_config"]["timeout_seconds"] + 30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
                    row = result_base(spec, attempt)
                    row.update(error="Worker exceeded the process wall-clock bound", error_kind="worker_timeout",
                               seconds=time.monotonic() - started, worker_log=str(log))
                    atomic_json(target, row)
            if not target.exists():
                row = result_base(spec, attempt)
                row.update(error=f"Worker exited {process.returncode} without a result; see worker log",
                           error_kind="worker_process_error", seconds=time.monotonic() - started, worker_log=str(log))
                atomic_json(target, row)
            result = json.loads(target.read_text())
            if valid_result(spec, result) or result.get("error_kind") not in RETRYABLE_ERRORS:
                return result
        if result is None:
            raise RuntimeError("No worker attempt result")
        return result


def restore_known_attempts(specs: list[dict], progress: Path, store: Path) -> None:
    """A lost cell directory must not cause replay of an already published success."""
    known = read_jsonl(progress / "evaluations.jsonl") + read_jsonl(progress / "evaluation_errors.jsonl")
    expected = {spec["evaluation_cell_id"]: spec for spec in specs}
    for row in known:
        spec = expected.get(row.get("evaluation_cell_id"))
        if spec is None or row.get("identity") != spec["identity"] or int(row.get("attempt", 0)) < 1:
            continue
        directory = store / spec["evaluation_cell_id"]
        with locked(directory / ".lock"):
            target = directory / f"attempt-{int(row['attempt']):04d}.json"
            if not target.exists():
                atomic_json(directory / "request.json", spec)
                atomic_json(target, row)


def merge_progress(progress: Path, store: Path) -> None:
    with locked(progress / ".evaluation-merge.lock"):
        output = progress / "evaluations.jsonl"
        existing = read_jsonl(output)
        legacy = [r for r in existing if r.get("artifact_version") != ARTIFACT_VERSION or "evaluation_cell_id" not in r]
        current = {r["evaluation_cell_id"]: r for r in existing if r.get("artifact_version") == ARTIFACT_VERSION and "evaluation_cell_id" in r}
        errors = {}
        for error in read_jsonl(progress / "evaluation_errors.jsonl"):
            errors[(error["evaluation_cell_id"], error["attempt"])] = error
        for request in sorted(store.glob("*/request.json")):
            spec = json.loads(request.read_text()); attempts = attempt_rows(request.parent)
            result = selected_result(spec, attempts)
            if result is not None:
                row = dict(result)
                row["cell_artifact_dir"] = str(request.parent)
                if row.get("error"):
                    row["error"] = row["error"].splitlines()[0][:600]
                current[spec["evaluation_cell_id"]] = row
            for row in attempts:
                if row.get("status") == "error":
                    errors[(row["evaluation_cell_id"], row["attempt"])] = row
        rows = legacy + sorted(current.values(), key=lambda r: (r["iteration"], r["checkpoint"], r["opponent"], r["map"], r["seed"], r["side"]))
        atomic_text(output, "".join(canonical(row) + "\n" for row in rows))
        atomic_text(progress / "evaluation_errors.jsonl", "".join(canonical(row) + "\n" for _, row in sorted(errors.items())))


def collect_rows(progress: Path, store: Path, specs: list[dict]) -> list[dict]:
    rows = read_jsonl(progress / "evaluations.jsonl")
    for spec in specs:
        rows.extend(attempt_rows(store / spec["evaluation_cell_id"]))
    return rows


def run_schedule(specs: list[dict], store: Path, *, jobs: int, max_attempts: int,
                 on_result, deadline: float | None = None, stop_file: Path | None = None) -> str | None:
    """Keep at most jobs active; stop launching, then let active games finish."""
    def stop_reason():
        if stop_file is not None and Path(stop_file).exists():
            return "stop_file"
        if deadline is not None and time.monotonic() >= deadline:
            return "deadline"
        return None
    iterator = iter(specs)
    exhausted, stopped = False, None
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        active = {}
        while True:
            while not exhausted and len(active) < jobs:
                stopped = stop_reason()
                if stopped is not None:
                    break
                spec = next(iterator, None)
                if spec is None:
                    exhausted = True
                    break
                active[executor.submit(run_cell, spec, store, max_attempts)] = spec
            if not active:
                break
            ready, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in ready:
                del active[future]
                on_result(future.result())
    return stopped


def verify_gate(manifest: dict, gate: dict) -> None:
    if manifest["suite_id"] != gate["benchmark_suite_id"] or panel_fingerprint(manifest) != gate["panel_fingerprint_sha256"]:
        raise ValueError("Victory gate does not match the frozen benchmark panel")
    if len(manifest["bots"]) != gate["required_opponents"]:
        raise ValueError("Victory gate opponent count changed")


def verify_holdout_isolation(payload: dict, gate: dict) -> None:
    training_maps = payload.get("training_maps", payload.get("config", {}).get("maps"))
    if not isinstance(training_maps, list) or any(not isinstance(name, str) for name in training_maps):
        raise ValueError("Holdout requires checkpoint metadata declaring its complete training map list")
    overlap = set(training_maps) & set(gate["stages"]["holdout"]["maps"])
    if overlap:
        raise ValueError(f"Holdout maps were used for training: {sorted(overlap)}")


def lock_holdout(path: Path, gate: dict, context: dict, *, standard_passed: bool) -> dict:
    if not standard_passed:
        raise ValueError("Holdout requires a complete passing standard stage with identical checkpoint/configuration")
    candidate = {"gate_id": gate["gate_id"], "panel_fingerprint_sha256": gate["panel_fingerprint_sha256"],
                 "gate_config_sha256": fingerprint(gate), "checkpoint_sha256": context["checkpoint_sha256"],
                 "evaluation_config_sha256": context["evaluation_config_sha256"]}
    with locked(path.with_suffix(".lock")):
        if path.exists():
            ledger = json.loads(path.read_text())
            if any(ledger.get(key) != value for key, value in candidate.items()):
                raise ValueError("This holdout version is already locked/consumed by a different candidate or configuration")
            return ledger
        runtime = context["evaluation_config"]["runtime"]
        schema = context["evaluation_config"]["policy_schema"]
        ledger = {**candidate, "status": "in_progress", "started_at": datetime.now(timezone.utc).isoformat(),
                  "engine_patch_sha256": runtime["engine_patch_sha256"], "action_schema": schema["action_version"],
                  "observation_schema": schema["feature_version"]}
        atomic_json(path, ledger)
        return ledger


def evaluate_checkpoint(checkpoint: Path, *, iteration: int | None = None, generation: int | None = None,
                        suite: str = "smoke", progress_dir: Path = ROOT / "progress/campaign-v2", jobs: int = 2,
                        timeout_seconds: float = 1800, max_attempts: int = 2, search_mode: str = "greedy",
                        search_config: dict | None = None, opponents: list[str] | None = None,
                        manifest_path: Path = ROOT / "configs/benchmarks.json", maps_dir: Path = ROOT / "maps",
                        gate_path: Path = ROOT / "configs/victory_gate.json", gate_stage: str = "standard",
                        gate_ledger_dir: Path = ROOT / "runs/gate-ledgers",
                        deadline: float | None = None, stop_file: Path | None = None,
                        execute: bool = True) -> dict:
    if jobs < 1 or jobs > 32 or max_attempts < 1 or max_attempts > 5:
        raise ValueError("jobs must be1..32 and max_attempts1..5")
    if search_mode != "greedy" or search_config:
        raise ValueError("Only greedy evaluation is currently supported; no search fallback is allowed")
    checkpoint = Path(checkpoint).resolve(); progress = Path(progress_dir).resolve(); maps_dir = Path(maps_dir).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    checkpoint_hash = file_hash(checkpoint)
    _, _, payload, schema = load_policy(checkpoint)
    if file_hash(checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed while loading; use an immutable checkpoint file")
    iteration = int(generation if generation is not None else iteration if iteration is not None else payload.get("iteration", 0))
    context = make_context(checkpoint_hash, schema, runtime_identity(schema), timeout_seconds=timeout_seconds,
                           search_mode=search_mode, search_config=search_config)
    gate = None
    if suite == "victory_gate":
        gate = json.loads(Path(gate_path).read_text()); verify_gate(manifest, gate)
        if opponents is not None:
            raise ValueError("Victory gate cannot select a reduced panel")
        if gate_stage not in ("standard", "holdout"):
            raise ValueError("Victory gate stage must be standard or holdout, never an unguarded all-stages run")
        schedule, stage = gate["stages"][gate_stage], gate_stage
    else:
        if suite not in ("smoke", "standard"):
            raise ValueError("Unknown evaluation suite")
        schedule, stage = manifest["evaluation_protocol"][suite], suite
    specs = build_cell_specs(manifest, schedule, context, checkpoint=checkpoint, iteration=iteration,
                             suite=suite, stage=stage, maps_dir=maps_dir, opponents=opponents)
    store = progress / "evaluation_cells"
    progress.mkdir(parents=True, exist_ok=True)
    ledger_path, standard = None, None
    if gate is not None and gate_stage == "holdout":
        verify_holdout_isolation(payload, gate)
        standard_specs = build_cell_specs(manifest, gate["stages"]["standard"], context, checkpoint=checkpoint,
                    iteration=iteration, suite=suite, stage="standard", maps_dir=maps_dir)
        standard = gate_stage_report(standard_specs, collect_rows(progress, store, standard_specs),
                                     gate["stages"]["standard"], required_opponents=gate["required_opponents"])
        ledger_path = Path(gate_ledger_dir) / f"{gate['gate_id']}.json"
        if execute:
            lock_holdout(ledger_path, gate, context, standard_passed=standard["passed"])
    stopped = "report_only" if not execute else None
    if execute:
        restore_known_attempts(specs, progress, store)
        if ledger_path is not None:
            ledger = json.loads(ledger_path.read_text())
            if ledger["status"] in ("passed", "consumed_failed"):
                existing_report = completion_report(specs, collect_rows(progress, store, specs))
                if not existing_report["complete"]:
                    raise ValueError("Completed holdout evidence is missing; restore its archive rather than replay successful cells")
        def on_result(row):
            merge_progress(progress, store)
            print(canonical({key: row.get(key) for key in ("iteration", "opponent", "map", "seed", "side", "status", "score", "rounds", "error_kind")}), flush=True)
        # Compilation happens inside each scheduled worker under a per-build lock.
        # No unscheduled games or builds are launched after STOP/deadline.
        stopped = run_schedule(specs, store, jobs=jobs, max_attempts=max_attempts, on_result=on_result,
                               deadline=deadline, stop_file=stop_file)
        merge_progress(progress, store)
    rows = collect_rows(progress, store, specs)
    report = (gate_stage_report(specs, rows, schedule, required_opponents=gate["required_opponents"]) if gate
              else completion_report(specs, rows, max_attempts=max_attempts, required_opponents=len(manifest["bots"])))
    report.update(checkpoint_sha256=context["checkpoint_sha256"], evaluation_config_sha256=context["evaluation_config_sha256"],
                  benchmark_suite_id=manifest["suite_id"], suite=suite, stage=stage, iteration=iteration,
                  expected_cell_ids=[s["evaluation_cell_id"] for s in specs],
                  executed=execute, stop_reason=stopped, stopped_early=bool(stopped and not report["complete"]))
    if ledger_path is not None:
        report["standard_stage_passed"] = standard["passed"]
        report["victory_gate_passed"] = bool(standard["passed"] and report["passed"])
    if ledger_path is not None and execute:
        with locked(ledger_path.with_suffix(".lock")):
            ledger = json.loads(ledger_path.read_text())
            ledger["latest_report"] = report
            if report["complete"]:
                ledger["status"] = "passed" if report["passed"] else "consumed_failed"
                ledger["completed_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(ledger_path, ledger)
        report["holdout_ledger"] = str(ledger_path)
    run_id = fingerprint({"context": context, "cell_ids": report["expected_cell_ids"]})
    if execute:
        atomic_json(progress / "evaluation_reports" / f"{run_id}.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--iteration", "--generation", dest="iteration", type=int)
    parser.add_argument("--suite", choices=("smoke", "standard", "victory_gate"), default="smoke")
    parser.add_argument("--gate-stage", choices=("standard", "holdout"), default="standard")
    parser.add_argument("--progress-dir", type=Path, default=ROOT / "progress/campaign-v2")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--search-mode", default="greedy")
    parser.add_argument("--opponent", action="append")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--budget-seconds", type=float)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--attempt", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        atomic_json(args.worker_output, run_cell_worker(json.loads(args.worker.read_text()), args.attempt))
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required")
    report = evaluate_checkpoint(args.checkpoint, iteration=args.iteration, suite=args.suite,
              progress_dir=args.progress_dir, jobs=args.jobs, timeout_seconds=args.timeout_seconds,
              max_attempts=args.max_attempts, search_mode=args.search_mode, opponents=args.opponent, gate_stage=args.gate_stage,
              stop_file=args.stop_file, deadline=time.monotonic() + args.budget_seconds if args.budget_seconds is not None else None,
              execute=not args.report_only)
    print(json.dumps(report, indent=2))
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
