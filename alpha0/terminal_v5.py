"""Bounded, resumable, supervised terminal-target-head-v5 experiment."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from . import terminal_v5_common as C


class ReviewRequired(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def relative(path):
    return str(Path(path).resolve().relative_to(C.ROOT))


def file_path(path):
    path = Path(path)
    return path if path.is_absolute() else C.ROOT / path


def source_path(job):
    return C.RUN / "sources" / f"{job['source_id']}.json"


def active_actors():
    found = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            command = process.joinpath("cmdline").read_bytes().decode(errors="replace").split("\0")[:-1]
            if not command or process.joinpath("cwd").resolve() != C.ROOT:
                continue
            executable = Path(command[0]).name
            python_actor = "python" in executable and any(
                value in ("alpha0.terminal_v5", "alpha0.root_ranking_v4", "alpha0.campaign", "alpha0.learn_v2")
                or Path(value).name in ("run_warmstart_experiment.py", "run_campaign.py") for value in command[1:])
            java_actor = executable == "java" and "alpha0.EngineBridge" in command
            if python_actor or java_actor:
                found.append({"pid": int(process.name), "command": command})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return found


def verified_source(job, expected):
    path = source_path(job)
    if not path.exists():
        return None
    row = read(path)
    if row.get("status") != "ok":
        return None
    if row.get("experiment_identity") != C.fingerprint(expected) or row.get("source") != "selfplay":
        raise ValueError("source ownership differs")
    for key in ("source_id", "split", "map_index", "j", "map", "seed", "policy_seeds"):
        if row.get(key) != job[key]:
            raise ValueError(f"source schedule differs: {key}")
    if C.sha256(file_path(row["cache_npz"])) != row["cache_sha256"]:
        raise ValueError("source cache changed")
    if row.get("exclusion_map_sha256") != C.fingerprint(job.get("exclusion_map", {})):
        raise ValueError("source input-exclusion set changed")
    from .terminal_v5_collect import _verify_source, source_spec
    return _verify_source(row, source_spec(job, expected["config"], expected))


def finish_task(path, reason):
    if not Path(path).exists():
        return
    task = read(path)
    if task.get("kind") == "benchmark" and "benchmark_spec" in task:
        from .terminal_v5_report import benchmark_error_record
        spec, attempt = task["benchmark_spec"], task["attempt"]
        output = C.PROGRESS / "benchmarks/cells" / spec["cell_id"] / f"attempt-{attempt:03d}.json"
        if not output.exists():
            C.immutable_json(output, benchmark_error_record(spec, attempt, "worker_timeout", reason))
    elif "result_path" in task and "start_record" in task:
        output = file_path(task["result_path"])
        if not output.exists():
            C.immutable_json(output, {**task["start_record"], "status": "incomplete", "terminal": False,
                "score": None, "error": reason, "ended_at": C.now(), "supervisor_finalized": True})
    Path(path).unlink(missing_ok=True)


def kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=3)


def process_group_resources(groups):
    ticks = os.sysconf("SC_CLK_TCK")
    pages = os.sysconf("SC_PAGE_SIZE")
    cpu, rss, processes = 0., 0, 0
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            raw = process.joinpath("stat").read_text()
            fields = raw[raw.rfind(")")+2:].split()
            if int(fields[2]) not in groups:
                continue
            cpu += (int(fields[11]) + int(fields[12])) / ticks
            rss += int(fields[21]) * pages
            processes += 1
        except (FileNotFoundError, PermissionError, ValueError, IndexError, ProcessLookupError):
            continue
    return {"live_process_group_cpu_seconds": cpu, "aggregate_rss_bytes": rss, "processes": processes}


class Supervisor:
    def __init__(self, config, expected, deadline):
        self.config, self.expected, self.deadline = config, expected, deadline
        self.identity_hash = C.fingerprint(expected)
        self.active = {}
        self.phase = "starting"
        self.job_count = 0
        self.resources = read(C.RUN / "resource-samples.json") if (C.RUN / "resource-samples.json").exists() else []

    def status(self, **extra):
        resources = process_group_resources({item["process"].pid for item in self.active.values()})
        self.resources.append({"at": C.now(), "phase": self.phase, "supervisor_pid": os.getpid(), **resources})
        C.atomic_json(C.RUN / "resource-samples.json", self.resources)
        C.atomic_json(C.RUN / "status.json", {"experiment_id": self.config["experiment_id"],
            "identity_sha256": self.identity_hash, "pid": os.getpid(), "phase": self.phase,
            "updated_at": C.now(), "active_jobs": list(self.active), "completed_jobs_this_cycle": self.job_count,
            "resources": resources, **extra})

    def run_jobs(self, jobs, workers=None):
        workers = workers or self.config["workers"]
        queue = list(jobs)
        last_status = 0.
        started, started_at, parent_cpu = time.monotonic(), C.now(), time.process_time()
        child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        completed_before = self.job_count
        failure = None
        try:
            while queue or self.active:
                C.check_stop(self.deadline)
                while queue and len(self.active) < workers:
                    task = queue.pop(0)
                    job_hash = C.fingerprint({"identity_sha256": self.identity_hash, "job": task})
                    directory = C.RUN / "jobs" / job_hash
                    directory.mkdir(parents=True, exist_ok=True)
                    number = len(list(directory.glob("attempt-*-start.json"))) + 1
                    task_path = directory / "task.json"
                    finish_task(task_path, "previous supervisor interrupted")
                    payload = {**task, "worker_task_path": relative(task_path)}
                    descriptor = {"identity_sha256": self.identity_hash, "job": payload}
                    descriptor_path = directory / "job.json"
                    C.immutable_json(descriptor_path, descriptor)
                    start = {"started_at": C.now(), "job": task, "job_hash": job_hash,
                             "identity_sha256": self.identity_hash, "attempt": number}
                    C.immutable_json(directory / f"attempt-{number:03d}-start.json", start)
                    log = (directory / f"attempt-{number:03d}.log").open("a")
                    command = [sys.executable, "-u", "-m", "alpha0.terminal_v5", "--worker", str(descriptor_path),
                               "--deadline", str(self.deadline)]
                    blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGALRM}
                    previous = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
                    try:
                        process = subprocess.Popen(command, cwd=C.ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        self.active[job_hash] = {"process": process, "log": log, "directory": directory,
                            "task_path": task_path, "number": number, "start": start, "started": time.monotonic(),
                            "idle_since": time.monotonic()}
                    except BaseException:
                        log.close()
                        raise
                    finally:
                        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
                for key, item in list(self.active.items()):
                    process = item["process"]
                    timeout = False
                    try:
                        current_task = read(item["task_path"])
                    except FileNotFoundError:
                        current_task = None
                    if current_task is not None:
                        item["idle_since"] = None
                        timeout = time.monotonic() >= current_task.get("deadline_monotonic", float("inf"))
                    # A worker must publish its first task promptly, even if imports fail/hang.
                    else:
                        if item["idle_since"] is None:
                            item["idle_since"] = time.monotonic()
                        timeout = time.monotonic() - item["idle_since"] > 120
                    if timeout or process.poll() is not None:
                        returncode = process.poll()
                        kill_group(process)
                        item["log"].close()
                        reason = "task_timeout" if timeout else None if returncode == 0 else f"worker_exit_{returncode}"
                        finish_task(item["task_path"], reason or "worker returned without final task record")
                        result = {**item["start"], "ended_at": C.now(), "returncode": returncode,
                                  "reason": reason, "elapsed_seconds": time.monotonic() - item["started"]}
                        C.immutable_json(item["directory"] / f"attempt-{item['number']:03d}.json", result)
                        del self.active[key]
                        if reason:
                            raise ReviewRequired(f"{reason}; inspect {relative(item['directory'])}")
                        self.job_count += 1
                if time.monotonic() - last_status >= 10:
                    self.status()
                    last_status = time.monotonic()
                time.sleep(.1)
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            for item in self.active.values():
                kill_group(item["process"])
                item["log"].close()
                finish_task(item["task_path"], C.stop_reason(self.deadline) or "supervisor interrupted")
                result = {**item["start"], "ended_at": C.now(), "returncode": item["process"].returncode,
                          "reason": C.stop_reason(self.deadline) or "supervisor interrupted"}
                C.immutable_json(item["directory"] / f"attempt-{item['number']:03d}.json", result)
            self.active.clear()
            child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
            timing = {"identity_sha256": self.identity_hash, "phase": self.phase, "started_at": started_at,
                "ended_at": C.now(), "wall_seconds": time.monotonic()-started,
                "supervisor_cpu_seconds": time.process_time()-parent_cpu,
                "reaped_descendant_cpu_seconds": child_after.ru_utime + child_after.ru_stime - child_before.ru_utime - child_before.ru_stime,
                "cpu_scope": "waited Python workers plus their reaped Java descendants; interrupted/unreaped descendants may be omitted",
                "workers_limit": workers, "jobs_scheduled": len(jobs), "jobs_completed": self.job_count-completed_before,
                "interruption_or_error": failure}
            name = f"{time.time_ns()}-{os.getpid()}.json"
            C.immutable_json(C.PROGRESS / "stage-timing" / name, timing)

    def source(self, job):
        row = verified_source(job, self.expected)
        if row is None:
            self.run_jobs([{"kind": "source", "payload": job}], workers=1)
            row = verified_source(job, self.expected)
        if row is None:
            raise ReviewRequired(f"source incomplete: {job['source_id']}")
        return row

    def discover(self, split, exclusions=None):
        count = 4 if split == "training" else 2
        maximum = 8 if split == "training" else 6
        initial = [C.source_job(split, m, j, self.config) for m in range(4) for j in range(1, count+1)]
        if split == "validation":
            for job in initial:
                job["exclusion_map"] = exclusions
        self.run_jobs([{"kind": "source", "payload": job} for job in initial if verified_source(job, self.expected) is None])
        roots, assignments, unavailable, sources = [], [], [], {}
        for m in range(4):
            used_reserves = set()
            for j in range(1, count+1):
                roles = [("BABY_RAT", 0 if j % 2 else 1), ("RAT_KING", 1 if j % 2 else 0)]
                missing = list(roles)
                attempts = [j] + [n for n in range(count+1, maximum+1) if n % 2 == j % 2]
                chosen = {}
                for n in attempts:
                    if n != j and n in used_reserves:
                        continue
                    if n != j:
                        used_reserves.add(n)
                    job = C.source_job(split, m, n, self.config)
                    if split == "validation":
                        job["exclusion_map"] = exclusions
                    row = self.source(job)
                    sources[job["source_id"]] = {"source_json": relative(source_path(job)),
                        "source_sha256": C.sha256(source_path(job)), "map": job["map"], "seed": job["seed"]}
                    filled = []
                    for role, player in list(missing):
                        candidate = row["root_candidates"].get(f"{role}:{player}")
                        if candidate is not None:
                            chosen[(role, player)] = {**candidate, **sources[job["source_id"]],
                                "cache_npz": row["cache_npz"], "cache_sha256": row["cache_sha256"],
                                "source_id": job["source_id"], "map_index": m, "slot_group_j": j,
                                "split": split}
                            missing.remove((role, player))
                            filled.append(f"{role}:{player}")
                    assignments.append({"map_index": m, "slot_group_j": j, "source_j": n,
                                        "source_id": job["source_id"], "filled": filled})
                    if not missing:
                        break
                for role, player in roles:
                    if (role, player) in chosen:
                        ordinal = len(roots)
                        roots.append({**chosen[(role, player)], "root_id": f"{split}-{ordinal:03d}", "root_ordinal": ordinal})
                    else:
                        unavailable.append({"map_index": m, "j": j, "role": role, "player": player})
        # Include every initial completed game, even if a source has no selected root.
        for job in initial:
            path = source_path(job)
            sources[job["source_id"]] = {"source_json": relative(path), "source_sha256": C.sha256(path),
                                       "map": job["map"], "seed": job["seed"]}
        result = {"identity_sha256": self.identity_hash, "split": split, "roots": roots,
            "sources": sources, "assignments": assignments, "unavailable_slots": unavailable,
            "complete": not unavailable and len(roots) == self.config[f"{split}_roots"],
            "distinct_root_source_games": len({r["source_id"] for r in roots}),
            "gradient_input_hashes_sha256": C.fingerprint(sorted(exclusions)) if exclusions is not None else None}
        if len({(r["source_id"], r["decision_index"]) for r in roots}) != len(roots):
            raise ValueError("physical root reused")
        if split == "validation" and any(r["input_hash"] in exclusions for r in roots):
            raise ValueError("validation overlaps gradient inputs")
        C.immutable_json(C.PROGRESS / f"roots-{split}.json", result)
        if not result["complete"]:
            raise ReviewRequired(f"{split} root coverage incomplete; no SGD")
        return result


@lru_cache(maxsize=8)
def load_cache(path, expected_hash):
    path = Path(path)
    if C.sha256(path) != expected_hash:
        raise ValueError("cached inputs changed")
    with np.load(path, allow_pickle=False) as cache:
        return {name: cache[name].copy() for name in ("hidden", "prior", "mask", "local", "global_board", "scalars")}


def cached_record(ref):
    cache = load_cache(str(file_path(ref["cache_npz"])), ref["cache_sha256"])
    i = ref["feature_index"]
    arrays = {name: values[i].copy() for name, values in cache.items()}
    if C.input_hash((arrays["local"], arrays["global_board"], arrays["scalars"]), arrays["mask"]) != ref["input_hash"]:
        raise ValueError("cached gradient input does not match hash")
    return arrays


def build_anchors(training, config, expected, deadline=float("inf")):
    rng = np.random.default_rng(config["anchor_seed"])
    strata = [(source_id, player) for source_id in sorted(training["sources"]) for player in (0, 1)]
    quotient, remainder = divmod(config["anchor_count"], len(strata))
    anchors, quotas = [], []
    for ordinal, (source_id, player) in enumerate(strata):
        C.check_stop(deadline)
        source_ref = training["sources"][source_id]
        row = read(file_path(source_ref["source_json"]))
        if C.sha256(file_path(source_ref["source_json"])) != source_ref["source_sha256"]:
            raise ValueError("source metadata changed before anchors")
        candidates = sorted(row["anchor_candidates"][str(player)], key=lambda a: a["decision_index"])
        quota = quotient + (ordinal < remainder)
        if len(candidates) < quota:
            raise ReviewRequired(f"anchor stratum shortfall {source_id}/{player}: {len(candidates)} < {quota}")
        chosen = sorted(rng.choice(len(candidates), size=quota, replace=False).tolist())
        for index in chosen:
            anchors.append({**candidates[index], **source_ref, "source_id": source_id,
                "cache_npz": row["cache_npz"], "cache_sha256": row["cache_sha256"],
                "anchor_id": f"anchor-{len(anchors):03d}"})
        quotas.append({"source_id": source_id, "player": player, "quota": quota})
    exclusions = {}
    for kind, rows in (("root", training["roots"]), ("anchor", anchors)):
        for row in rows:
            C.check_stop(deadline)
            cached_record(row)
            exclusions.setdefault(row["input_hash"], []).append({"kind": kind, "id": row.get("root_id", row.get("anchor_id"))})
    result = {"identity_sha256": C.fingerprint(expected), "anchors": anchors, "quotas": quotas,
              "count": len(anchors), "complete": len(anchors) == 512, "exclusion_map": exclusions,
              "gradient_input_hashes": sorted(exclusions), "gradient_input_hashes_sha256": C.fingerprint(sorted(exclusions))}
    C.immutable_json(C.PROGRESS / "anchors.json", result)
    return result


def make_dataset(training, validation, anchors, config, expected, deadline=float("inf")):
    from .terminal_v5_collect import cell_spec, successful_cell
    root_arrays, terminal_actions, puct_actions = [], [], []
    complete_cells, error_attempts = 0, 0
    for root in training["roots"]:
        C.check_stop(deadline)
        comparator = read(C.PROGRESS / "comparators" / f"{root['root_id']}.json")
        table = {}
        for action in sorted(root["legal_actions"]):
            scores = []
            for seed in config["selection_seeds"]:
                spec = cell_spec(root, "selection", action, seed, config, expected)
                directory = C.PROGRESS / "cells" / f"selection-{root['root_id']}-a{action:02d}-s{seed}"
                attempts = [read(p) for p in sorted(directory.glob("attempt-[0-9][0-9][0-9].json"))]
                result = successful_cell(attempts, spec)
                if result is None:
                    raise ReviewRequired("selection incomplete; no SGD")
                complete_cells += 1
                error_attempts += sum(row.get("status") == "error" for row in attempts)
                scores.append(result["score"])
            table[action] = float(np.mean(scores))
        arrays = cached_record(root)
        visits = {int(key): value for key, value in comparator["visits"].items()}
        if (set(visits) != set(root["legal_actions"]) or
                any(type(value) is not int or value < 0 for value in visits.values()) or sum(visits.values()) != 32 or
                comparator["selected_action"] != max(root["legal_actions"], key=lambda action: visits[action]) or
                not np.array_equal(np.asarray(comparator["policy_prior"], dtype=np.float32), arrays["prior"])):
            raise ValueError("PUCT selected-action/prior/visit evidence differs from frozen root")
        chosen = min(table, key=lambda a: (-table[a], -float(arrays["prior"][a]), a))
        terminal_actions.append(chosen)
        puct_actions.append(comparator["selected_action"])
        root_arrays.append({key: arrays[key] for key in ("hidden", "prior", "mask")})
    anchor_arrays = []
    for row in anchors["anchors"]:
        C.check_stop(deadline)
        arrays = cached_record(row)
        anchor_arrays.append({key: arrays[key] for key in ("hidden", "prior", "mask")})
    games = sorted(training["sources"])
    dataset = {**{f"root_{key}": np.stack([r[key] for r in root_arrays]) for key in ("hidden", "prior", "mask")},
               **{f"anchor_{key}": np.stack([r[key] for r in anchor_arrays]) for key in ("hidden", "prior", "mask")},
        "terminal_actions": np.asarray(terminal_actions, dtype=np.int64),
        "puct_actions": np.asarray(puct_actions, dtype=np.int64),
        "root_strata": np.asarray([[r["map_index"], int(r["role"] == "RAT_KING"), r["player"]] for r in training["roots"]], dtype=np.int64),
        "anchor_game": np.asarray([games.index(r["source_id"]) for r in anchors["anchors"]], dtype=np.int64),
        "anchor_player": np.asarray([r["player"] for r in anchors["anchors"]], dtype=np.int64)}
    path = C.RUN / "dataset/training.npz"
    C.immutable_npz(path, dataset)
    metadata = {"format_version": 1, "identity_sha256": C.fingerprint(expected), "npz_sha256": C.sha256(path),
        "complete": True, "source": "own_selfplay", "benchmark_data_used": False, "validation_data_used": False,
        "root_count": 32, "anchor_count": 512, "training_maps": config["maps"],
        "training_seeds": sorted(training["sources"][g]["seed"] for g in games),
        "selection_counts": {"expected": 2 * sum(len(r["legal_actions"]) for r in training["roots"]),
                             "completed": complete_cells, "errors": 0, "attempt_errors": error_attempts},
        "training_games": [{"game_id": g, "map": training["sources"][g]["map"], "seed": training["sources"][g]["seed"]} for g in games],
        "root_ids": [r["root_id"] for r in training["roots"]], "anchor_refs": anchors["anchors"],
        "root_input_hashes": [r["input_hash"] for r in training["roots"]],
        "anchor_input_hashes": [r["input_hash"] for r in anchors["anchors"]],
        "validation_input_hashes": [r["input_hash"] for r in validation["roots"]],
        "gradient_input_hashes": anchors["gradient_input_hashes"],
        "provenance": {"training_roots_sha256": C.fingerprint(training), "validation_roots_sha256": C.fingerprint(validation),
                       "anchors_sha256": C.fingerprint(anchors), "checkpoint_sha256": config["checkpoint_sha256"],
                       "comparators": {r["root_id"]: C.sha256(C.PROGRESS / "comparators" / f"{r['root_id']}.json")
                                       for r in training["roots"]}}}
    C.immutable_json(path.with_suffix(".json"), metadata)
    return path


def freeze_actions(validation, fit, config, expected):
    from .learn_v2 import load_checkpoint
    import torch.nn.functional as F
    checkpoints = {"cp2": {"path": config["checkpoint"], "sha256": config["checkpoint_sha256"]}, **fit["checkpoints"]}
    heads = {}
    for arm, checkpoint in checkpoints.items():
        path = file_path(checkpoint["path"])
        if C.sha256(path) != checkpoint["sha256"]:
            raise ValueError("candidate checkpoint changed")
        model, _ = load_checkpoint(path)
        model.eval()
        heads[arm] = model.policy
    actions = {}
    with torch.inference_mode():
        for root in validation["roots"]:
            data = cached_record(root)
            hidden = torch.from_numpy(data["hidden"]).unsqueeze(0)
            mask = torch.from_numpy(data["mask"].astype(bool)).unsqueeze(0)
            actions[root["root_id"]] = {arm: int(head(hidden).masked_fill(~mask, -torch.inf).argmax(-1).item()) for arm, head in heads.items()}
    result = {"identity_sha256": C.fingerprint(expected), "roots_sha256": C.fingerprint(validation),
              "checkpoint_sha256_by_arm": {arm: row["sha256"] for arm, row in checkpoints.items()}, "actions": actions}
    C.immutable_json(C.PROGRESS / "validation-actions.json", result)
    return result, checkpoints


def worker(descriptor, deadline, config, expected):
    if descriptor["identity_sha256"] != C.fingerprint(expected):
        raise ValueError("worker descriptor identity differs")
    job = descriptor["job"]
    task_path = job["worker_task_path"]
    C.set_task(task_path, "worker_start", 120)
    if job["kind"] == "source":
        from .terminal_v5_collect import source_game
        source_game({**job["payload"], "worker_task_path": task_path}, config, expected, deadline)
    elif job["kind"] in ("comparator", "selection", "validation"):
        from .terminal_v5_collect import root_job
        result = root_job({**job["payload"], "worker_task_path": task_path}, job["kind"], config, expected, deadline)
        if job["kind"] != "comparator" and not result["complete"]:
            raise ReviewRequired("root job exhausted attempts with incomplete coverage")
    elif job["kind"] == "fit":
        from .terminal_v5_fit import fit_arms
        C.set_task(task_path, "fit", 600)
        fit_arms(file_path(job["payload"]["dataset"]), config, expected, C.RUN, C.PROGRESS)
    elif job["kind"] == "benchmark":
        from .terminal_v5_report import record_benchmark_attempt, select_benchmark_result, valid_benchmark_result
        from .evaluate import RETRYABLE_ERRORS
        spec = job["payload"]
        directory = C.PROGRESS / "benchmarks/cells" / spec["cell_id"]
        for _ in range(config["evaluation_attempts"]):
            C.check_stop(deadline)
            rows = [read(path) for path in sorted(directory.glob("attempt-[0-9][0-9][0-9].json"))]
            selected = select_benchmark_result(spec, rows)
            if selected is not None and valid_benchmark_result(spec, selected):
                break
            if selected is not None and selected.get("error_kind") not in RETRYABLE_ERRORS:
                raise ReviewRequired("genuine benchmark failure requires review")
            attempt = max((row["attempt"] for row in rows), default=0) + 1
            if attempt > config["evaluation_attempts"]:
                raise ReviewRequired("benchmark exhausted retries")
            C.set_task(task_path, "benchmark", min(config["evaluation_timeout_seconds"] + 30, max(.01, deadline-time.monotonic())),
                       benchmark_spec=spec, attempt=attempt)
            result = record_benchmark_attempt(spec, directory, attempt=attempt, max_attempts=config["evaluation_attempts"])
            if valid_benchmark_result(spec, result):
                break
        else:
            raise ReviewRequired("benchmark incomplete after bounded attempts")
    else:
        raise ValueError("unknown worker kind")
    C.clear_task(task_path)


def execute(supervisor):
    config, expected = supervisor.config, supervisor.expected
    supervisor.phase = "discover_training"
    training = supervisor.discover("training")
    total = 2 * sum(len(r["legal_actions"]) for r in training["roots"])
    budget = {"identity_sha256": C.fingerprint(expected), "selection_cells": total,
              "ceiling": config["selection_cell_ceiling"], "within_ceiling": total <= config["selection_cell_ceiling"]}
    C.immutable_json(C.PROGRESS / "target-budget.json", budget)
    if not budget["within_ceiling"]:
        raise ReviewRequired(f"{total} planned selection cells exceed preregistered ceiling; no outcomes scored")
    anchors = build_anchors(training, config, expected, supervisor.deadline)
    supervisor.phase = "discover_validation"
    validation = supervisor.discover("validation", anchors["exclusion_map"])
    supervisor.phase = "freeze_comparators"
    supervisor.run_jobs([{"kind": "comparator", "payload": root} for root in training["roots"]])
    supervisor.phase = "collect_selection"
    supervisor.run_jobs([{"kind": "selection", "payload": root} for root in training["roots"]])
    dataset = make_dataset(training, validation, anchors, config, expected, supervisor.deadline)
    supervisor.phase = "fit_heads"
    supervisor.run_jobs([{"kind": "fit", "payload": {"dataset": relative(dataset)}}], workers=1)
    fit = read(C.RUN / "fit/manifest.json")
    actions, checkpoints = freeze_actions(validation, fit, config, expected)
    supervisor.phase = "collect_validation"
    supervisor.run_jobs([{"kind": "validation", "payload": {**root, "actions": actions["actions"][root["root_id"]]}} for root in validation["roots"]])
    from .terminal_v5_report import build_benchmark_specs, report
    # First-action validation must complete before starting any benchmark games.
    current = report(C.RUN, C.PROGRESS, config, expected, integrity_verified=False)
    if not current.get("validation", {}).get("complete"):
        raise ReviewRequired("independent validation incomplete; no benchmark stage")
    specs = build_benchmark_specs({arm: file_path(row["path"]) for arm, row in checkpoints.items()}, config, expected)
    C.immutable_json(C.PROGRESS / "benchmarks/specs.json", {"identity_sha256": C.fingerprint(expected), "specs": specs})
    supervisor.phase = "benchmark"
    supervisor.run_jobs([{"kind": "benchmark", "payload": spec} for spec in specs], workers=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10800)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    config = C.load_config()
    if not 0 < args.seconds <= config["wall_budget_seconds"]:
        parser.error("cycle must be positive and at most three hours")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(C.Interrupted("SIGTERM")))
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(C.Interrupted("SIGINT")))
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGALRM})
    if args.worker:
        expected = read(C.PROGRESS / "identity.json")
        if C.identity(config) != expected:
            raise ValueError("worker runtime identity changed")
        worker(read(args.worker), args.deadline, config, expected)
        return
    C.RUN.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in C.LOCKS:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open("a+"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(path.exists() for path in C.STOPS):
            raise C.Interrupted("STOP exists; no work started")
        actors = active_actors()
        if actors:
            raise ReviewRequired(f"existing actors require inspection; no duplicate launched: {actors}")
        expected = C.identity(config)
        C.immutable_json(C.PROGRESS / "identity.json", expected)
        C.immutable_json(C.PROGRESS / "config.json", config)
        for name in expected["sources"]:
            destination = C.PROGRESS / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            content = (C.ROOT / name).read_bytes()
            C.immutable_bytes(destination, content)
        started = time.monotonic()
        supervisor = Supervisor(config, expected, started + args.seconds)
        invocation = {"started_at": C.now(), "pid": os.getpid(), "budget_seconds": args.seconds,
                      "identity_sha256": C.fingerprint(expected)}
        outcome, reason = "complete_review_required", None
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(C.Interrupted("deadline")))
        signal.setitimer(signal.ITIMER_REAL, args.seconds)
        try:
            execute(supervisor)
        except C.Interrupted as exc:
            outcome, reason = "paused", str(exc)
        except ReviewRequired as exc:
            outcome, reason = "review_required", str(exc)
        except BaseException as exc:
            outcome, reason = "invalidated", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                unchanged = C.identity(config) == expected
            except Exception as exc:
                unchanged = False
                reason = f"identity recheck failed: {type(exc).__name__}: {exc}"
            if not unchanged:
                outcome, reason = "invalidated", reason or "runtime or inputs changed"
            from .terminal_v5_report import report
            try:
                summary = report(C.RUN, C.PROGRESS, config, expected, integrity_verified=unchanged)
            except Exception as exc:
                outcome, reason = "invalidated", f"final evidence check failed: {type(exc).__name__}: {exc}"
                summary = {"experiment_id": config["experiment_id"], "phase": "invalidated", "error": reason,
                           "decision": {"complete": False, "passes": False, "integrity_verified": False}}
            C.atomic_json(C.PROGRESS / "summary.json", summary)
            result = {**invocation, "ended_at": C.now(), "phase": outcome, "reason": reason,
                      "elapsed_seconds": time.monotonic() - started, "runtime_and_checkpoint_unchanged": unchanged,
                      "decision": summary.get("decision"), "active_jobs": []}
            C.atomic_json(C.RUN / "status.json", result)
            number = len(list((C.PROGRESS / "invocations").glob("*.json"))) + 1
            C.immutable_json(C.PROGRESS / "invocations" / f"{number:03d}.json", result)
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
