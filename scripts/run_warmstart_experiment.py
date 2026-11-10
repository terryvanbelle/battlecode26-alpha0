#!/usr/bin/env python3
"""One bounded, resumable MCTS iteration from the explicit own-teacher prior.

Separate run/progress/lock/STOP paths protect the neutral v2 control. Import only
model weights, evaluate cp0, complete one search-target iteration, then evaluate
cp1. This runner never promotes a policy or opens the reserved holdout.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from alpha0.evaluate import atomic_json
from alpha0.learn_v2 import Learner
from alpha0.network_v2 import NetworkConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def pid_token(pid: int) -> str | None:
    """Linux process start tick prevents signaling an unrelated reused PID."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19]  # starttime is field22; fields starts at field3.
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return None


def track_tree(pid: int, tracked: dict[int, str]) -> None:
    # This executor does not expose task/*/children. PPID scanning also finds
    # subprocesses started from evaluator threads or in a separate session.
    by_parent, tokens = {}, {}
    for stat_file in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat_file.read_text().rsplit(")", 1)[1].split()
            child, parent = int(stat_file.parent.name), int(fields[1])
            tokens[child] = fields[19]
            by_parent.setdefault(parent, []).append(child)
        except (FileNotFoundError, PermissionError, IndexError, ValueError):
            continue
    pending = [pid]
    while pending:
        current = pending.pop()
        if current in tokens:
            tracked[current] = tokens[current]
            pending.extend(by_parent.get(current, ()))


def stop_tree(process: subprocess.Popen, tracked: dict[int, str]) -> None:
    """Terminate only this evaluation's descendants, including separate sessions."""
    track_tree(process.pid, tracked)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, token in reversed(tuple(tracked.items())):
            if pid_token(pid) == token:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM:
            end = time.monotonic()+2
            while time.monotonic() < end and process.poll() is None:
                time.sleep(0.05)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def initialize_learner(root: Path, run: Path, progress: Path, config: dict) -> Learner:
    """Safely recover even if a crash followed Learner's initial random cp0."""
    initial = config["initialization"]
    artifact = root / initial["artifact"]
    if sha256(artifact) != initial["artifact_sha256"]:
        raise ValueError("initialization artifact checksum changed")
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    if (payload.get("model_version") != "dual_scale_v2" or payload.get("action_version") != "macro_v2" or
            payload.get("feature_version") != 2 or payload.get("experiment_id") != initial["source_experiment_id"] or
            payload.get("teacher_version") != initial["teacher_version"] or
            payload.get("source_dataset_sha256") != initial["source_dataset_sha256"] or
            payload.get("benchmark_data_used") is not False):
        raise ValueError("initialization lineage/schema mismatch")
    if payload["network_config"] != asdict(NetworkConfig(**config["model"])):
        raise ValueError("initialization network configuration differs")
    weights = payload["model_state_dict"]
    if weights["value.2.weight"].count_nonzero() or weights["value.2.bias"].count_nonzero():
        raise ValueError("warmstart artifact's final value head is not zero")
    expected = {"artifact_sha256": initial["artifact_sha256"], "config_sha256": fingerprint(config),
                "source_dataset_sha256": initial["source_dataset_sha256"]}
    marker = run / "initialization.json"
    if marker.exists():
        recorded = json.loads(marker.read_text())
        if any(recorded.get(key) != value for key, value in expected.items()):
            raise ValueError("warmstart initialization marker differs")
        # Learner treats missing latest as a new run and writes random cp0.
        # Validate recovery prerequisites before it can mutate either file.
        if not (run / "latest.pt").exists() or not (run / "checkpoint-0000.pt").exists():
            raise ValueError("published warmstart checkpoint missing; restore recovery files before resuming")
        if sha256(run / "checkpoint-0000.pt") != recorded["checkpoint_zero_sha256"]:
            raise ValueError("published warmstart baseline changed")
        learner = Learner(run, config, progress)
        if learner.iteration == 0 and sha256(run / "latest.pt") != recorded["checkpoint_zero_sha256"]:
            raise ValueError("latest generation-zero checkpoint differs from published warmstart baseline")
        return learner
    if list(run.glob("plan-*.json")) or (progress / "evaluation_cells").exists() or (progress / "evaluations.jsonl").exists():
        raise ValueError("missing initialization marker with existing plans/evaluations; restore provenance instead of rewriting cp0")
    if (run / "checkpoint-0000.pt").exists() and not (run / "latest.pt").exists():
        raise ValueError("interrupted unpublished checkpoint copy; restore latest from the verified cp0 before retrying initialization")
    learner = Learner(run, config, progress)
    if (learner.iteration != 0 or learner.games_total or learner.positions_total or len(learner.buffer) or
            learner.buffer.num_games or learner.optimizer.state):
        raise ValueError("cannot import a prior into an existing trained/replay-populated learner")
    # load_state_dict copies into the existing Parameters, preserving the fresh
    # optimizer's references. Neither teacher optimizer nor examples are loaded.
    learner.model.load_state_dict(weights)
    if not all(torch.equal(learner.model.state_dict()[key], value) for key, value in weights.items()):
        raise AssertionError("model-only import failed")
    learner.checkpoint()
    atomic_json(marker, {**expected, "initialized_at": utc_now(),
                        "checkpoint_zero_sha256": sha256(run / "checkpoint-0000.pt"),
                        "model_weights_only": True, "replay_games": 0, "replay_positions": 0,
                        "optimizer_state_entries": 0, "value_head": "zero"})
    return learner


def cpu_cores_used(interval: float = 1.0) -> float:
    path = Path("/sys/fs/cgroup/cpu.stat")
    if not path.exists():
        return 0.0
    def usage():
        return int(dict(line.split() for line in path.read_text().splitlines())["usage_usec"])
    started, before = time.monotonic(), usage()
    time.sleep(interval)
    return (usage()-before)/1e6/(time.monotonic()-started)


class Experiment:
    def __init__(self, root: Path, config: dict, hours: float, *, initialize_only: bool = False,
                 target_iteration: int = 1):
        self.root, self.config = root.resolve(), config
        self.run, self.progress = self.root / "runs/warmstart-v3", self.root / "progress/warmstart-v3"
        self.stop_file = self.run / "STOP"
        self.started, self.deadline = time.monotonic(), time.monotonic()+hours*3600
        self.initialize_only = initialize_only
        self.target_iteration = target_iteration
        gate = json.loads((self.root / "configs/victory_gate.json").read_text())
        self.opponent_ids = {b["id"] for b in gate["frozen_opponents"]}
        if len(self.opponent_ids) != 12 or set(config["maps"]) & set(gate["stages"]["holdout"]["maps"]):
            raise ValueError("expected full12 panel and training maps disjoint from holdout")
        if config["workers"] != 1 or config["evaluation_workers"] != 1 or config["torch_threads"] != 1 or target_iteration < 1:
            raise ValueError("this experiment requires a positive target, one actor, one evaluator and one Torch thread")
        self.learner = None
        self.state = {"experiment_id": config["experiment_id"], "pid": os.getpid(), "started_at": utc_now(),
                      "budget_hours": hours, "target_iteration": target_iteration, "initialization": config["initialization"],
                      "all_opponents_beaten": False, "production_promoted": False}

    def status(self, phase: str, **extra):
        self.state.update(phase=phase, updated_at=utc_now(), elapsed_seconds=time.monotonic()-self.started, **extra)
        if self.learner is not None:
            self.state.update(iteration=self.learner.iteration, games_total=self.learner.games_total,
                              checkpoint=str(self.learner.checkpoint_path.relative_to(self.root)))
        atomic_json(self.run / "status.json", self.state)
        atomic_json(self.progress / "status.json", self.state)
        print(json.dumps(self.state, allow_nan=False), flush=True)

    def stopped(self):
        return self.stop_file.exists() or time.monotonic() >= self.deadline

    def plots(self):
        subprocess.run([sys.executable, str(self.root / "scripts/plot_progress.py"),
                        "--progress-dir", str(self.progress)], cwd=self.root, check=True, timeout=60)

    def evaluate(self, iteration: int) -> dict | None:
        checkpoint = self.run / f"checkpoint-{iteration:04d}.pt"
        self.status("evaluating", evaluation_iteration=iteration)
        remaining = self.deadline-time.monotonic()
        if remaining <= 0 or self.stop_file.exists():
            return None
        command = [sys.executable, "-u", "-m", "alpha0.evaluate", "--checkpoint", str(checkpoint),
                   "--iteration", str(iteration), "--suite", "smoke", "--progress-dir", str(self.progress),
                   "--jobs", "1", "--timeout-seconds", str(self.config["evaluation_timeout_seconds"]),
                   "--max-attempts", str(self.config["evaluation_attempts"]),
                   "--budget-seconds", str(remaining), "--stop-file", str(self.stop_file)]
        process = subprocess.Popen(command, cwd=self.root, start_new_session=True)
        tracked: dict[int, str] = {}
        try:
            while process.poll() is None:
                track_tree(process.pid, tracked)
                if self.stopped():
                    stop_tree(process, tracked)
                    atomic_json(self.run / f"evaluation-{iteration:04d}-interruption.json",
                                {"at": utc_now(), "reason": "STOP" if self.stop_file.exists() else "deadline",
                                 "checkpoint_sha256": sha256(checkpoint),
                                 "note": "owned evaluation process tree terminated; successful cells remain immutable"})
                    return None
                time.sleep(0.2)
            if process.returncode != 0:
                if self.stopped():
                    return None
                raise RuntimeError(f"smoke evaluator exited {process.returncode}; inspect retained cell attempts")
        finally:
            if process.poll() is None:
                stop_tree(process, tracked)
        checkpoint_hash = sha256(checkpoint)
        reports = [json.loads(path.read_text()) for path in (self.progress / "evaluation_reports").glob("*.json")]
        reports = [r for r in reports if r.get("checkpoint_sha256") == checkpoint_hash and
                   r.get("iteration") == iteration and r.get("suite") == "smoke"]
        if len(reports) != 1:
            raise ValueError("expected one immutable runtime identity for this baseline/checkpoint evaluation")
        report = reports[0]
        if (not report.get("complete") or not report.get("full_panel_complete") or report.get("unresolved_errors") != 0 or
                set(report.get("opponents", {})) != self.opponent_ids or report.get("completed") != 24):
            raise ValueError("incomplete/invalid smoke panel cannot advance the warmstart experiment")
        atomic_json(self.progress / f"checkpoint-{iteration:04d}-smoke.json", report)
        self.plots()
        return report

    def execute(self):
        if self.stop_file.exists():
            self.status("paused", reason="STOP exists; intentional restart requires explicitly removing it")
            return
        self.learner = initialize_learner(self.root, self.run, self.progress, self.config)
        if self.learner.iteration > self.target_iteration:
            raise ValueError("requested target precedes the current checkpoint; choose an explicit later target")
        self.status("initialized", empty_initial_replay=True, fresh_initial_optimizer=True, neutral_initial_value=True)
        if self.initialize_only:
            return
        measured = cpu_cores_used()
        self.status("resource_check", measured_active_cpu_cores=measured)
        if measured >= 3.5:
            self.status("paused", reason="insufficient CPU headroom for one extra actor", measured_active_cpu_cores=measured)
            return
        if self.stopped():
            self.status("paused", reason="budget_or_stop")
            return
        baseline = self.evaluate(0)
        if baseline is None or self.stopped():
            self.status("paused", reason="baseline_budget_or_stop")
            return
        baseline_manifest_hash = sha256(self.root / "configs/benchmarks.json")
        trained = baseline
        if self.learner.iteration > 0:
            trained = self.evaluate(self.learner.iteration)
            if trained is None or self.stopped():
                self.status("paused", reason="current_checkpoint_evaluation_pending")
                return
            self.check_comparison(baseline, trained)
        while self.learner.iteration < self.target_iteration:
            if self.stopped():
                self.status("paused", reason="training_budget_or_stop")
                return
            self.status("self_play_and_training", evaluation_iteration=None)
            if self.learner.train_iteration(self.deadline, self.stop_file) is None:
                self.status("paused", reason="training_budget_or_stop")
                return
            self.plots()
            self.status("checkpoint_ready")
            if self.stopped():
                self.status("paused", reason="checkpoint_evaluation_pending")
                return
            trained = self.evaluate(self.learner.iteration)
            if trained is None:
                self.status("paused", reason="checkpoint_evaluation_budget_or_stop")
                return
            self.check_comparison(baseline, trained)
            if sha256(self.root / "configs/benchmarks.json") != baseline_manifest_hash:
                raise ValueError("benchmark manifest changed between baseline and trained evaluation")
        wins = lambda r: sum(x["wins"] for x in r["opponents"].values())
        self.status("complete_review_required", baseline_wins=wins(baseline), trained_wins=wins(trained),
                    evaluation_games_per_checkpoint=24, completed_mcts_iterations=self.learner.iteration,
                    reason="requested iteration target complete; no promotion or holdout attempted")

    @staticmethod
    def check_comparison(baseline: dict, trained: dict):
        if (baseline["evaluation_config_sha256"] != trained["evaluation_config_sha256"] or
                baseline["benchmark_suite_id"] != trained["benchmark_suite_id"]):
            raise ValueError("baseline/trained evaluation runtime, policy, watchdog or suite differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/warmstart-v3.json")
    parser.add_argument("--hours", type=float, default=2)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--target-iteration", type=int, default=1,
                        help="Operational target; later explicit targets resume the same immutable training configuration")
    args = parser.parse_args()
    if not 0 < args.hours <= 2:
        parser.error("this standalone cycle must be greater than zero and no longer than two hours")
    root = args.root.resolve()
    run = root / "runs/warmstart-v3"
    run.mkdir(parents=True, exist_ok=True)
    with (run / "experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another warmstart process owns the lock; nothing started")
        (run / "experiment.pid").write_text(str(os.getpid())+"\n")
        print(f"Warmstart PID {os.getpid()}; log {run / 'experiment.log'}", flush=True)
        with (run / "experiment.log").open("a", buffering=1) as log:
            os.dup2(log.fileno(), sys.stdout.fileno())
            os.dup2(log.fileno(), sys.stderr.fileno())
            experiment = None
            try:
                config = json.loads(args.config.read_text())
                experiment = Experiment(root, config, args.hours, initialize_only=args.initialize_only,
                                        target_iteration=args.target_iteration)
                experiment.execute()
            except BaseException as error:
                if experiment is not None:
                    experiment.status("failed", error=str(error), traceback=traceback.format_exc())
                raise


if __name__ == "__main__":
    main()
