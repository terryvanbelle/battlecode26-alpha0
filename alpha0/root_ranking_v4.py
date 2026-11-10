"""Preregistered, bounded terminal-rollout root-ranking diagnostic; never trains.

The supervisor owns all locks and kills the one worker's process group on STOP,
deadline or a stuck task. All essential evidence is immutable under progress/;
runs/ contains only replaceable supervision state and logs.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import torch

from alpha0.env import EngineEnv
from alpha0.features_v2 import make_evaluator
from alpha0.learn_v2 import _training_runtime_identity, load_checkpoint
from alpha0.mcts import SearchConfig, search

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/root-ranking-v4"
PROGRESS = ROOT / "progress/root-ranking-v4"
STOPS = (ROOT / "runs/campaign/STOP", ROOT / "runs/warmstart-v3/STOP", RUN / "STOP")
THRESHOLDS = {"mean_delta_min": .10, "improved_roots_min": 2, "worst_root_delta_min": -.25}


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"immutable evidence differs: {path}")
        return
    # Single writer is enforced by the supervisor's lock. No result is replaced.
    atomic_json(path, value)


def validate_config(config, reserved_seeds=()):
    expected = {"experiment_id": "terminal-root-ranking-v4", "schema_version": 1,
                "source_games": [f"iter-0001-game-{i:03d}" for i in range(4, 8)],
                "root_roles": ["BABY_RAT", "RAT_KING"], "required_legal_action": 37,
                "selection_seeds": [103003, 103013], "validation_seeds": [103021, 103043, 103067, 103091],
                "temperature": .35, "thresholds": THRESHOLDS, "workers": 1, "torch_threads": 1,
                "gradient_updates": 0, "benchmark_data_used": False}
    for key, value in expected.items():
        if config.get(key) != value or type(config.get(key)) is not type(value):
            raise ValueError(f"preregistered configuration differs: {key}")
    if config.get("search") != {"simulations": 32, "max_depth": 12, "cpuct": 1.5,
            "dirichlet_alpha": .3, "exploration_fraction": .25, "root_noise": True, "root_seed_base": 20261114}:
        raise ValueError("fixed PUCT comparator differs")
    seeds = config["selection_seeds"] + config["validation_seeds"]
    if len(set(seeds)) != len(seeds) or set(seeds) & set(reserved_seeds):
        raise ValueError("selection/validation/reserved seeds overlap")
    for key in ("wall_budget_seconds", "cell_timeout_seconds", "task_timeout_seconds", "response_timeout_seconds",
                "max_decisions", "max_error_attempts", "max_worker_restarts"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"positive integer required: {key}")
    if config["wall_budget_seconds"] > 7200 or config["response_timeout_seconds"] > config["cell_timeout_seconds"]:
        raise ValueError("invalid bounded runtime")


def successful_cell(attempts, expected_identity):
    successes = []
    for attempt in attempts:
        if attempt.get("identity") != expected_identity:
            raise ValueError("cell identity mismatch")
        if attempt.get("status") == "ok":
            if attempt.get("terminal") is not True or attempt.get("score") not in (0, .5, 1):
                raise ValueError("successful cell needs a genuine terminal score")
            if attempt.get("valid_game") is not True or attempt.get("benchmark_failures") != []:
                raise ValueError("failed engine game cannot be a successful cell")
            successes.append(attempt)
        elif attempt.get("score") is not None:
            raise ValueError("partial/error cell cannot have a score")
    if len(successes) > 1:
        # A successful cell is never scheduled again. Even equal repeat scores
        # are invalid evidence of this implementation's single-success contract.
        raise ValueError("duplicate successful cell attempts")
    return successes[0] if successes else None


def cell_map(cells):
    result = {}
    for cell in cells:
        if cell.get("status") != "ok":
            if cell.get("score") is not None:
                raise ValueError("partial cell cannot have a score")
            continue
        if cell.get("score") not in (0, .5, 1):
            raise ValueError("invalid terminal score")
        key = (cell.get("root_id"), cell["action"], cell["seed"])
        if key in result and result[key]["score"] != cell["score"]:
            raise ValueError("conflicting successful cell outcomes")
        result[key] = cell
    return result


def choose_action(legal_actions, priors, cells, seeds):
    actions = sorted(legal_actions)
    if not actions or len(actions) != len(set(actions)):
        raise ValueError("unique legal actions required")
    complete = cell_map(cells)
    # This pure helper receives one root; identity filtering occurs at load.
    scores = {(cell["action"], cell["seed"]): cell["score"] for cell in complete.values()}
    means = {str(action): float(np.mean([scores[action, seed] for seed in seeds]))
             for action in actions if all((action, seed) in scores for seed in seeds)}
    count = sum((action, seed) in scores for action in actions for seed in seeds)
    ready = count == len(actions) * len(seeds)
    selected = max(actions, key=lambda action: (means[str(action)], float(priors[action]), -action)) if ready else None
    return {"complete": ready, "selected_action": selected, "action_scores": means,
            "completed": count, "expected": len(actions) * len(seeds)}


def summarize_validation(root_specs, selections, comparators, cells, config):
    scored = cell_map([cell for cell in cells if cell.get("stage") == "validation"])
    deltas = []
    for root in root_specs:
        rid = root["root_id"]
        selected, comparator = selections.get(rid, {}), comparators.get(rid, {})
        if not selected.get("complete") or "selected_action" not in comparator:
            break
        a, b = selected["selected_action"], comparator["selected_action"]
        if any((rid, action, seed) not in scored for action in (a, b) for seed in config["validation_seeds"]):
            break
        deltas.append(float(np.mean([scored[rid, a, seed]["score"] - scored[rid, b, seed]["score"]
                                    for seed in config["validation_seeds"]])))
    complete = bool(root_specs) and len(deltas) == len(root_specs)
    mean, improved, worst = (float(np.mean(deltas)), sum(d > 0 for d in deltas), min(deltas)) if complete else (None, None, None)
    passes = complete and mean + 1e-12 >= THRESHOLDS["mean_delta_min"] and improved >= 2 and worst >= -.25
    return {"complete": complete, "passes": bool(passes), "mean_delta": mean, "improved_roots": improved,
            "worst_root_delta": worst, "thresholds": THRESHOLDS}


def root_state_hash(raw):
    return fingerprint({key: value for key, value in raw.items() if key != "snapshot"})


def stop_reason(deadline, stop_paths=STOPS):
    if any(Path(path).exists() for path in stop_paths):
        return "STOP"
    if time.monotonic() >= deadline:
        return "deadline"
    return None


class Interrupted(Exception):
    pass


def supervise_process(command, deadline, stop_paths, on_tick=lambda: None, task_path=None, *, cwd=None, stdout=None):
    """Run one isolated process group; no child can outlive the supervisor budget."""
    existing_stop = stop_reason(deadline, stop_paths)
    if existing_stop:
        return existing_stop
    previous_term = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise Interrupted("supervisor received SIGTERM")
    signal.signal(signal.SIGTERM, terminate)
    try:
        process = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=subprocess.STDOUT,
                                   start_new_session=True, env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    except BaseException:
        signal.signal(signal.SIGTERM, previous_term)
        raise
    reason = None
    try:
        while process.poll() is None:
            reason = stop_reason(deadline, stop_paths)
            if reason is None and task_path:
                try:
                    task = json.loads(Path(task_path).read_text())
                except FileNotFoundError:
                    task = None  # The worker can finish/remove its task between polls.
                if task and time.monotonic() >= task["deadline_monotonic"]:
                    reason = "task_timeout"
            if reason:
                break
            on_tick()
            time.sleep(.1)
        if reason is None and process.returncode != 0:
            reason = f"worker_exit_{process.returncode}"
    except BaseException:
        reason = "supervisor_interrupted"
        raise
    finally:
        # Java inherits this group. Kill the group even if the Python leader has
        # exited, so orphaned Java cannot survive a worker exception or timeout.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=.5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)
        signal.signal(signal.SIGTERM, previous_term)
    return reason


def identity(config, config_path):
    checkpoint = ROOT / config["checkpoint"]
    if sha256(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint changed")
    runtime = _training_runtime_identity()
    sources, maps = {}, {}
    for game_id in config["source_games"]:
        path = ROOT / config["source_run"] / "games" / f"{game_id}.json"
        row = json.loads(path.read_text())
        if (row.get("source") != "selfplay" or row.get("status") != "ok" or row.get("opponent_checkpoint") is not None
                or row.get("action_version") != "macro_v2" or row["game_id"] != game_id):
            raise ValueError("root source is not completed own current-policy self-play")
        if row["training_runtime"] != runtime:
            raise ValueError("source trajectory and current engine/inference runtime differ")
        map_path = ROOT / "maps" / f"{row['map']}.map26"
        if sha256(map_path) != row["identity"]["map_sha256"]:
            raise ValueError("source map changed")
        sources[str(path.relative_to(ROOT))] = sha256(path)
        maps[str(map_path.relative_to(ROOT))] = sha256(map_path)
    gate = json.loads((ROOT / "configs/victory_gate.json").read_text())
    validate_config(config, gate["stages"]["confirmation"]["seeds"])
    return {"schema_version": 1, "config": config, "config_sha256": fingerprint(config),
            "config_file_sha256": sha256(config_path), "checkpoint_sha256": sha256(checkpoint),
            "runtime": runtime, "source_games": sources, "maps": maps,
            "runner_source_sha256": sha256(__file__), "reserved_confirmation_seeds": gate["stages"]["confirmation"]["seeds"]}


class Experiment:
    def __init__(self, config, expected_identity, deadline):
        self.config, self.identity, self.deadline = config, expected_identity, deadline
        self.identity_hash = fingerprint(expected_identity)
        self.model, _ = load_checkpoint(ROOT / config["checkpoint"])
        self.model.eval()
        self.infer = make_evaluator(self.model)
        self.env = self.state = self.root = None

    def budget(self):
        reason = stop_reason(self.deadline)
        if reason:
            raise Interrupted(reason)
        if self.env:
            self.env.response_timeout = min(self.config["response_timeout_seconds"], max(.01, self.deadline - time.monotonic()))

    def task(self, kind, **metadata):
        limit = self.config["cell_timeout_seconds"] if kind == "cell" else self.config["task_timeout_seconds"]
        atomic_json(RUN / "task.json", {"kind": kind, "started_at": now(),
                    "deadline_monotonic": min(self.deadline, time.monotonic() + limit), **metadata})

    def release(self, *states):
        keep = {s.snapshot for s in states if s is not None and s.snapshot is not None}
        released = list(self.env._snapshots - keep)
        if released:
            self.env._request({"cmd": "release", "snapshots": released})
            self.env._snapshots.difference_update(released)

    def close(self):
        if self.env:
            self.env.close()
        self.env = self.state = self.root = None

    def source(self, game_id):
        return json.loads((ROOT / self.config["source_run"] / "games" / f"{game_id}.json").read_text())

    def discover(self):
        destination = PROGRESS / "roots.json"
        if destination.exists():
            result = json.loads(destination.read_text())
            if result["identity"] != self.identity_hash:
                raise ValueError("frozen root manifest identity differs")
            return result
        self.task("discovery")
        roots, availability = [], []
        for game_id in self.config["source_games"]:
            self.budget()
            source = self.source(game_id)
            found = set()
            self.close()
            self.env = EngineEnv(action_version="macro_v2", response_timeout=self.config["response_timeout_seconds"])
            state = self.env.reset(source["map"], source["seed"])
            for decision, action in enumerate(source["actions"]):
                self.budget()
                if state.terminal:
                    raise ValueError("source replay terminated before recorded actions")
                actor = next(r for r in state.raw["robots"] if r["id"] == state.raw["current_id"])
                role = actor["type"]
                if role in self.config["root_roles"] and role not in found and self.config["required_legal_action"] in state.legal_actions:
                    prefix = source["actions"][:decision]
                    roots.append({"root_id": f"root-{len(roots):02d}", "ordinal": len(roots), "game_id": game_id,
                        "map": source["map"], "seed": source["seed"], "decision_index": decision,
                        "prefix_sha256": fingerprint(prefix), "state_sha256": root_state_hash(state.raw),
                        "player": state.player, "role": role, "round": state.raw["round"],
                        "actor_id": actor["id"], "legal_actions": list(state.legal_actions)})
                    found.add(role)
                if len(found) == len(self.config["root_roles"]):
                    break
                state = self.env.transition(state, action)
                self.release(state)
            availability.append({"game_id": game_id, "found_roles": sorted(found),
                                 "unavailable_roles": sorted(set(self.config["root_roles"]) - found)})
        self.close()
        result = {"identity": self.identity_hash, "roots": roots, "source_availability": availability,
                  "selection_rule": "first role turn per source game where macro37 is legal; no outcome-based backfill",
                  "all_sources_scanned": True}
        immutable_json(destination, result)
        return result

    def restore(self, root):
        if self.root and self.root["root_id"] == root["root_id"]:
            return self.state
        self.task("root_reconstruction", root_id=root["root_id"])
        self.close()
        source = self.source(root["game_id"])
        if (source["map"] != root["map"] or source["seed"] != root["seed"]
                or not 0 <= root["decision_index"] < len(source["actions"])):
            raise ValueError("root source metadata differs")
        prefix = source["actions"][:root["decision_index"]]
        if fingerprint(prefix) != root["prefix_sha256"]:
            raise ValueError("root hidden-state action prefix changed")
        self.env = EngineEnv(action_version="macro_v2", response_timeout=self.config["response_timeout_seconds"])
        state = self.env.reset(root["map"], root["seed"])
        for action in prefix:
            self.budget()
            state = self.env.transition(state, action)
            self.release(state)
        if root_state_hash(state.raw) != root["state_sha256"] or list(state.legal_actions) != root["legal_actions"]:
            raise ValueError("root deterministic reconstruction mismatch")
        actor = next(r for r in state.raw["robots"] if r["id"] == state.raw["current_id"])
        if (state.player != root["player"] or actor["type"] != root["role"] or actor["id"] != root["actor_id"]
                or state.raw["round"] != root["round"]):
            raise ValueError("root actor/player metadata differs from reconstructed state")
        self.root, self.state = root, state
        return state

    def comparator(self, root, roots_hash):
        path = PROGRESS / "comparators" / f"{root['root_id']}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["identity"] != self.identity_hash or record["roots_sha256"] != roots_hash:
                raise ValueError("comparator provenance differs")
            return record
        state = self.restore(root)
        self.task("comparator", root_id=root["root_id"])
        priors, value = self.infer(state)
        def transition(parent, action):
            self.budget()
            return self.env.transition(parent, action)
        settings = {k: v for k, v in self.config["search"].items() if k not in ("root_noise", "root_seed_base")}
        result = search(state, self.infer, transition, 40, config=SearchConfig(**settings),
                        rng=np.random.default_rng(self.config["search"]["root_seed_base"] + root["ordinal"]), add_root_noise=True)
        self.release(state)
        # Matches the prior probe and the unchanged MCTS visit dictionary order.
        chosen = max(result.visits, key=result.visits.get)
        record = {"identity": self.identity_hash, "roots_sha256": roots_hash, "root_id": root["root_id"],
                  "selected_action": chosen, "policy_prior": priors.tolist(), "network_value": float(value),
                  "visits": result.visits, "visit_policy": result.policy.tolist(), "nodes": result.nodes,
                  "max_depth": result.max_depth, "search_seed": self.config["search"]["root_seed_base"] + root["ordinal"],
                  "search": self.config["search"]}
        immutable_json(path, record)
        return record

    def cell_spec(self, root, roots_hash, stage, action, seed):
        return {"experiment_identity": self.identity_hash, "roots_sha256": roots_hash,
                "root_id": root["root_id"], "root_state_sha256": root["state_sha256"],
                "root_prefix_sha256": root["prefix_sha256"], "root_player": root["player"], "stage": stage, "action": action,
                "seed": seed, "temperature": self.config["temperature"]}

    def cell(self, root, roots_hash, stage, action, seed):
        spec = self.cell_spec(root, roots_hash, stage, action, seed)
        key = fingerprint(spec)
        directory = PROGRESS / "cells" / f"{stage}-{root['root_id']}-a{action:02d}-s{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        attempts = [json.loads(p.read_text()) for p in sorted(directory.glob("attempt-[0-9][0-9][0-9].json"))]
        success = successful_cell(attempts, key)
        if success or sum(a["status"] == "error" for a in attempts) >= self.config["max_error_attempts"]:
            return success
        state = self.restore(root)
        number = len(list(directory.glob("attempt-*-start.json"))) + 1
        result_path = directory / f"attempt-{number:03d}.json"
        started = time.monotonic()
        record = {"identity": key, **spec, "attempt": number, "started_at": now(),
                  "status": "incomplete", "terminal": False, "score": None}
        immutable_json(directory / f"attempt-{number:03d}-start.json", record)
        self.task("cell", result_path=str(result_path.relative_to(ROOT)), start_record=record)
        rng = np.random.default_rng(seed)
        decisions, action_hash = 0, hashlib.sha256()
        try:
            while True:
                self.budget()
                if time.monotonic() - started >= self.config["cell_timeout_seconds"]:
                    raise TimeoutError("per-cell continuation deadline")
                if decisions >= self.config["max_decisions"]:
                    raise TimeoutError("per-cell decision cap")
                if decisions == 0:
                    selected = action
                else:
                    priors, _ = self.infer(state)
                    legal = np.asarray(state.legal_actions, dtype=int)
                    logits = np.log(np.maximum(priors[legal], 1e-12)) / self.config["temperature"]
                    weights = np.exp(logits - logits.max())
                    selected = int(rng.choice(legal, p=weights / weights.sum()))
                action_hash.update(bytes([selected]))
                state = self.env.transition(state, selected)
                decisions += 1
                self.release(self.state, state)
                if state.terminal:
                    if state.raw.get("valid_game", True) is not True or state.raw.get("benchmark_failures", []):
                        raise ValueError("engine reported an invalid terminal game")
                    value = state.terminal_value_a * (1 if root["player"] == 0 else -1)
                    record.update(status="ok", terminal=True, score=(value + 1) / 2,
                        valid_game=True, benchmark_failures=[],
                        winner=state.raw["winner"], terminal_round=state.raw["round"],
                        terminal_state_sha256=root_state_hash(state.raw),
                        terminal_summary={k: state.raw.get(k) for k in ("terminal", "round", "winner", "win_reason", "bank", "kings", "transferred", "cat_damage")})
                    break
        except Interrupted as exc:
            record["error"] = str(exc)
            raise
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            self.close()
        finally:
            record.update(decisions=decisions, actions_sha256=action_hash.hexdigest(), elapsed_seconds=time.monotonic()-started,
                          ended_at=now())
            immutable_json(result_path, record)
            if self.env:
                self.release(self.state)
            (RUN / "task.json").unlink(missing_ok=True)
        return record if record["status"] == "ok" else None

    def execute(self):
        manifest = self.discover()
        roots, roots_hash = manifest["roots"], fingerprint(manifest)
        if not roots:
            raise ValueError("no qualifying own-self-play roots")
        # All roots were fixed before inference/search and all comparators are
        # fixed before any rollout scoring.
        comparators = {root["root_id"]: self.comparator(root, roots_hash) for root in roots}
        for root in roots:
            for action in sorted(root["legal_actions"]):
                for seed in self.config["selection_seeds"]:
                    for _ in range(self.config["max_error_attempts"]):
                        self.budget()
                        if self.cell(root, roots_hash, "selection", action, seed):
                            break
            summary = build_summary(self.identity, "selecting")
            selected = next(r["selection"] for r in summary["roots"] if r["root_id"] == root["root_id"])
            if not selected["complete"]:
                continue
            immutable_json(PROGRESS / "selections" / f"{root['root_id']}.json",
                {"identity": self.identity_hash, "roots_sha256": roots_hash, "root_id": root["root_id"], **selected})
            for action in sorted({selected["selected_action"], comparators[root["root_id"]]["selected_action"]}):
                for seed in self.config["validation_seeds"]:
                    for _ in range(self.config["max_error_attempts"]):
                        self.budget()
                        if self.cell(root, roots_hash, "validation", action, seed):
                            break
            build_summary(self.identity, "validating")
        self.close()
        (RUN / "task.json").unlink(missing_ok=True)


def load_records(directory):
    return {p.stem: json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))} if directory.exists() else {}


def build_summary(expected_identity, phase, *, integrity_verified=False):
    config = expected_identity["config"]
    manifest_path = PROGRESS / "roots.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"roots": []}
    roots_hash = fingerprint(manifest)
    identity_hash = fingerprint(expected_identity)
    comparators, selections = load_records(PROGRESS / "comparators"), load_records(PROGRESS / "selections")
    if manifest.get("identity", identity_hash) != identity_hash:
        raise ValueError("root manifest identity mismatch")
    by_id = {root["root_id"]: root for root in manifest["roots"]}
    if len(by_id) != len(manifest["roots"]):
        raise ValueError("duplicate frozen root ID")
    for records in (comparators, selections):
        for rid, record in records.items():
            if (rid not in by_id or record.get("root_id") != rid or record.get("identity") != identity_hash
                    or record.get("roots_sha256") != roots_hash
                    or record.get("selected_action") not in by_id[rid]["legal_actions"]):
                raise ValueError("comparator/selection provenance mismatch")
    for rid, comparator in comparators.items():
        root = by_id[rid]
        prior = np.asarray(comparator.get("policy_prior"), dtype=float)
        visits = comparator.get("visits", {})
        if (prior.shape != (40,) or not np.isfinite(prior).all() or (prior < 0).any()
                or not np.isclose(prior.sum(), 1) or any(prior[a] != 0 for a in set(range(40)) - set(root["legal_actions"]))
                or set(visits) != {str(a) for a in root["legal_actions"]}
                or any(type(v) is not int or v < 0 for v in visits.values()) or sum(visits.values()) != 32
                or comparator["selected_action"] != max(root["legal_actions"], key=lambda a: visits[str(a)])
                or comparator.get("search") != config["search"]
                or comparator.get("search_seed") != config["search"]["root_seed_base"] + root["ordinal"]):
            raise ValueError("frozen comparator prior/search evidence differs")
    successes, errors = [], []
    for directory in sorted((PROGRESS / "cells").glob("*")):
        attempts = [json.loads(p.read_text()) for p in sorted(directory.glob("attempt-[0-9][0-9][0-9].json"))]
        if not attempts:
            continue
        spec = {k: attempts[0][k] for k in ("experiment_identity", "roots_sha256", "root_id", "root_state_sha256",
                "root_prefix_sha256", "root_player", "stage", "action", "seed", "temperature")}
        if spec["experiment_identity"] != identity_hash or spec["roots_sha256"] != roots_hash:
            raise ValueError("cell experiment/root identity mismatch")
        root = by_id.get(spec["root_id"])
        if (root is None or spec["stage"] not in ("selection", "validation")
                or spec["action"] not in root["legal_actions"]
                or spec["seed"] not in config[f"{spec['stage']}_seeds"]
                or spec["root_state_sha256"] != root["state_sha256"]
                or spec["root_prefix_sha256"] != root["prefix_sha256"]
                or spec["root_player"] != root["player"]
                or spec["temperature"] != config["temperature"]):
            raise ValueError("cell is outside the frozen experiment plan")
        if spec["stage"] == "validation":
            if (spec["root_id"] not in selections or spec["root_id"] not in comparators
                    or spec["action"] not in {selections[spec["root_id"]]["selected_action"],
                                               comparators[spec["root_id"]]["selected_action"]}):
                raise ValueError("validation cell predates or differs from frozen selection")
        result = successful_cell(attempts, fingerprint(spec))
        if any({k: attempt[k] for k in spec} != spec for attempt in attempts):
            raise ValueError("cell attempt metadata differs from its identity")
        if result:
            expected_score = .5 if result["winner"] == -1 else float(result["winner"] == root["player"])
            if result["winner"] not in (-1, 0, 1) or result["score"] != expected_score:
                raise ValueError("terminal winner and root-perspective score disagree")
            terminal = result.get("terminal_summary", {})
            if (terminal.get("terminal") is not True or terminal.get("winner") != result["winner"]
                    or type(result.get("terminal_round")) is not int or result["terminal_round"] < root["round"]
                    or terminal.get("round") != result["terminal_round"]
                    or len(result.get("terminal_state_sha256", "")) != 64):
                raise ValueError("terminal proof differs from the scored outcome")
            successes.append(result)
        errors.extend(a for a in attempts if a["status"] == "error")
    reported = []
    for root in manifest["roots"]:
        rid = root["root_id"]
        comparator = comparators.get(rid, {})
        selection_cells = [c for c in successes if c["root_id"] == rid and c["stage"] == "selection"]
        selected = choose_action(root["legal_actions"], comparator.get("policy_prior", [0]*40), selection_cells, config["selection_seeds"])
        selected["errors"] = len({(e["action"], e["seed"]) for e in errors if e["root_id"] == rid and e["stage"] == "selection"
                                  and not any(c["action"] == e["action"] and c["seed"] == e["seed"] for c in selection_cells)})
        if rid in selections and any(selections[rid].get(key) != value for key, value in selected.items()):
            raise ValueError("frozen selection differs from complete selection evidence")
        validation = [c for c in successes if c["root_id"] == rid and c["stage"] == "validation"]
        a, b = selected["selected_action"], comparator.get("selected_action")
        actions = {a, b} - {None} if selected["complete"] else set()
        expected = len(actions) * len(config["validation_seeds"])
        cells = {(c["action"], c["seed"]): c for c in validation}
        ready = bool(actions) and all((action, seed) in cells for action in actions for seed in config["validation_seeds"])
        scores_a = [cells[a, seed]["score"] for seed in config["validation_seeds"]] if ready else []
        scores_b = [cells[b, seed]["score"] for seed in config["validation_seeds"]] if ready else []
        reported.append({**root, "selection": selected, "comparator": comparator,
            "validation": {"complete": ready, "seeds": config["validation_seeds"], "expected_unique_cells": expected,
                "completed_unique_cells": len(validation), "errors": len({(e["action"], e["seed"]) for e in errors
                    if e["root_id"] == rid and e["stage"] == "validation" and (e["action"], e["seed"]) not in cells}),
                "rollout_action": a, "puct_action": b, "rollout_scores": scores_a, "puct_scores": scores_b,
                "rollout_mean": float(np.mean(scores_a)) if ready else None,
                "puct_mean": float(np.mean(scores_b)) if ready else None,
                "delta": float(np.mean(scores_a)-np.mean(scores_b)) if ready else None}})
    decision = summarize_validation(manifest["roots"], selections, comparators,
                                    [c for c in successes if c["stage"] == "validation"], config)
    decision["criteria_pass"] = decision["passes"]
    decision["integrity_verified"] = integrity_verified
    decision["passes"] = bool(decision["criteria_pass"] and integrity_verified)
    summary = {"schema_version": 1, "experiment_id": config["experiment_id"], "identity_sha256": identity_hash,
        "phase": phase, "complete": decision["complete"], "updated_at": now(), "roots_frozen": manifest_path.exists(),
        "gradient_updates": 0, "benchmark_data_used": False, "roots": reported,
        "source_availability": manifest.get("source_availability", []), "decision": decision,
        "totals": {"selection_expected": sum(r["selection"]["expected"] for r in reported),
            "selection_completed": sum(r["selection"]["completed"] for r in reported),
            "validation_expected_unique": sum(r["validation"]["expected_unique_cells"] for r in reported),
            "validation_completed_unique": sum(r["validation"]["completed_unique_cells"] for r in reported),
            "terminal_games_completed": len(successes), "attempt_errors": len(errors)}}
    atomic_json(PROGRESS / "summary.json", summary)
    return summary


def finish_interrupted_task(reason):
    task_path = RUN / "task.json"
    if not task_path.exists():
        return
    task = json.loads(task_path.read_text())
    if task.get("kind") == "cell":
        path = ROOT / task["result_path"]
        if not path.exists():
            record = {**task["start_record"], "status": "error" if reason == "task_timeout" else "incomplete",
                      "terminal": False, "score": None, "error": reason, "ended_at": now(), "supervisor_finalized": True}
            immutable_json(path, record)
    task_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/root-ranking-v4.json")
    parser.add_argument("--seconds", type=float, default=7200)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not 0 < args.seconds <= config["wall_budget_seconds"]:
        parser.error("bounded cycle must be positive and at most two hours")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.worker:
        expected = json.loads((PROGRESS / "identity.json").read_text())
        if identity(config, args.config) != expected:
            raise ValueError("worker runtime/input identity changed")
        experiment = Experiment(config, expected, args.deadline)
        try:
            experiment.execute()
        finally:
            experiment.close()
        return
    RUN.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in (RUN / "experiment.lock", ROOT / "runs/campaign/campaign.lock", ROOT / "runs/warmstart-v3/experiment.lock"):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(path.exists() for path in STOPS):
            raise ValueError("STOP exists; nothing started")
        started, deadline = time.monotonic(), time.monotonic() + args.seconds
        expected = identity(config, args.config)
        PROGRESS.mkdir(parents=True, exist_ok=True)
        immutable_json(PROGRESS / "identity.json", expected)
        immutable_json(PROGRESS / "config.json", config)
        # Exact executed own sources are portable; third-party dependencies are
        # referenced by the runtime identity rather than bundled.
        source = PROGRESS / "source"
        source.mkdir(exist_ok=True)
        for original, target in ((Path(__file__), source / "root_ranking_v4.py"), (args.config, source / "root-ranking-v4.json")):
            if target.exists() and target.read_bytes() != original.read_bytes():
                raise ValueError("executed source snapshot differs")
            if not target.exists():
                target.write_bytes(original.read_bytes())
        finish_interrupted_task("previous supervisor interrupted")
        invocation = {"started_at": now(), "pid": os.getpid(), "budget_seconds": args.seconds,
                      "identity": fingerprint(expected)}
        atomic_json(RUN / "status.json", {**invocation, "phase": "running"})
        (RUN / "experiment.pid").write_text(str(os.getpid()) + "\n")
        summary = build_summary(expected, "starting")
        reason = None
        try:
            with (RUN / "experiment.log").open("a") as log:
                for restart in range(config["max_worker_restarts"] + 1):
                    if summary["complete"] or stop_reason(deadline):
                        break
                    command = [sys.executable, "-u", "-m", "alpha0.root_ranking_v4", "--worker", "--config", str(args.config),
                               "--deadline", str(deadline), "--seconds", str(args.seconds)]
                    last_report = [0.0]
                    def update():
                        if time.monotonic() - last_report[0] > 10:
                            build_summary(expected, "running")
                            last_report[0] = time.monotonic()
                    reason = supervise_process(command, deadline, STOPS, update, RUN / "task.json", cwd=ROOT, stdout=log)
                    reason = stop_reason(deadline) or reason
                    finish_interrupted_task(reason or "worker completed")
                    summary = build_summary(expected, "complete" if reason is None else "paused")
                    if reason is None or reason in ("STOP", "deadline"):
                        break
            if identity(config, args.config) != expected:
                raise ValueError("runtime/input identity changed during diagnostic")
            phase = "complete_review_required" if summary["complete"] else "paused"
            summary = build_summary(expected, phase, integrity_verified=True)
            result = {**invocation, "ended_at": now(), "phase": phase, "reason": reason,
                      "elapsed_seconds": time.monotonic()-started, "decision": summary["decision"],
                      "runtime_and_checkpoint_unchanged": True, "gradient_updates": 0}
            atomic_json(RUN / "status.json", result)
            number = len(list((PROGRESS / "invocations").glob("*.json"))) + 1
            immutable_json(PROGRESS / "invocations" / f"{number:03d}.json", result)
            print(json.dumps(result), flush=True)
        except BaseException as error:
            finish_interrupted_task(f"{type(error).__name__}: {error}")
            try:
                build_summary(expected, "invalidated", integrity_verified=False)
            except Exception:
                pass  # Preserve raw evidence even when malformed evidence caused failure.
            failure = {**invocation, "ended_at": now(), "phase": "invalidated",
                       "error": f"{type(error).__name__}: {error}", "integrity_verified": False,
                       "elapsed_seconds": time.monotonic()-started, "gradient_updates": 0}
            atomic_json(RUN / "status.json", failure)
            number = len(list((PROGRESS / "invocations").glob("*.json"))) + 1
            immutable_json(PROGRESS / "invocations" / f"{number:03d}.json", failure)
            raise



if __name__ == "__main__":
    main()
