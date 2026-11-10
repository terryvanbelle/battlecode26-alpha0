#!/usr/bin/env python3
"""Offline, immutable recovery exports for terminal-target-head-v5.

Use the project Python with ``plan|check|export --root REPO`` or
``verify|restore --archive FILE --root REPO``. Export only after the runner has
released all three experiment locks. STOP files permit export and are never
archived. Partial collections are explicitly labelled; they are not completed
experiments. This helper neither publishes artifacts nor launches actors.

Restore copies only missing run/progress/dependency files, never current code.
A relocated copy preserves identity-bound bytes and receives a v5 STOP marker;
resume requires the original canonical repository path and pinned runtime.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUN = "runs/terminal-target-head-v5"
PROGRESS = "progress/terminal-target-head-v5"
CP2 = "runs/warmstart-v3/checkpoint-0002.pt"
CP2_SHA256 = "20d9e813349010630940e1cdfbaee5206490c7d323997688d26f47508bbeee7c"
CP2_INLINE_LIMIT = 2 * 1024 * 1024
SCHEMA = "terminal-target-head-v5-recovery-v1"
MANIFEST = "V5-RECOVERY-MANIFEST.json"
REFERENCE_FILES = ("configs/terminal-target-head-v5.json", "docs/terminal-target-head-v5.md")
LOCKS = ("runs/campaign/campaign.lock", "runs/warmstart-v3/experiment.lock", f"{RUN}/experiment.lock")
SUFFIXES = {".json", ".jsonl", ".npz", ".npy", ".pt", ".csv", ".md", ".png", ".svg", ".txt"}
EXCLUDED_NAMES = {"STOP", "experiment.pid", "campaign.pid"}
CANDIDATES = tuple(f"{RUN}/fit/{arm}_head.pt" for arm in ("terminal", "puct"))
STATE_KEYS = ("model_state_dict", "optimizer_state_dict", "torch_rng", "numpy_rng", "batch_indices")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def checked_path(root, name):
    path = PurePosixPath(name)
    if (not name or not path.parts or path.is_absolute() or path.as_posix() != name or ".." in path.parts or
            any(part.startswith(".") for part in path.parts)):
        raise ValueError(f"Unsafe recovery path: {name}")
    current = Path(root)
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Symlink is not permitted: {name}")
    return current


def category(name):
    path = PurePosixPath(name)
    checked_path(Path("/nonexistent-v5-recovery-root"), name)
    if name == CP2:
        return "baseline_dependency"
    if name in REFERENCE_FILES:
        return "historical_reference"
    if path.name in EXCLUDED_NAMES or path.suffix in {".lock", ".log", ".pid", ".tmp"}:
        return None
    if name.startswith(f"{PROGRESS}/source/") and path.suffix in {".py", ".json"}:
        return "own_source_snapshot"
    if not name.startswith((RUN + "/", PROGRESS + "/")) or path.suffix not in SUFFIXES:
        return None
    relative = path.relative_to(RUN if name.startswith(RUN + "/") else PROGRESS)
    if any(part in {"vendor", "build", "venv", "benchmarks-bots", "foreign-source", "__pycache__"} for part in relative.parts):
        raise ValueError(f"Foreign/runtime data is outside the recovery allowlist: {name}")
    if name.startswith(PROGRESS + "/benchmarks/") or name.startswith(RUN + "/evaluation/"):
        return "benchmark_evidence"
    if name.startswith(PROGRESS + "/evaluation_") or path.name in {"evaluations.jsonl", "evaluation_errors.jsonl"}:
        return "benchmark_evidence"
    return "owned_run" if name.startswith(RUN + "/") else "owned_progress"


def file_identity(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def active_actor_pids(root):
    """Detect repository actors/orphan engines without exposing their command lines."""
    actors = []
    proc = Path("/proc")
    if not proc.is_dir():
        raise RuntimeError("Actor liveness verification requires Linux /proc")
    for process in proc.iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            arguments = (process / "cmdline").read_bytes().split(b"\0")
            args = [arg.decode(errors="replace") for arg in arguments if arg]
            kinds = [arg for arg in args if arg in {"alpha0.EngineBridge", "alpha0.campaign", "alpha0.evaluate"}
                     or arg.startswith("alpha0.terminal_v5") or
                     (Path(arg).name.startswith("run_") and Path(arg).name.endswith(".py") and
                      ("warmstart" in Path(arg).name or ("terminal" in Path(arg).name and "v5" in Path(arg).name)))]
            if kinds and (process / "cwd").resolve() == root:
                actors.append(int(process.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return sorted(actors)


@contextmanager
def experiment_locks(root):
    """The runner uses the same order. A present STOP never prevents export."""
    with ExitStack() as stack:
        for name in LOCKS:
            path = checked_path(root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context(path.open("a"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Actor is active; cannot acquire {name}") from error
        actors = active_actor_pids(root)
        if actors:
            raise RuntimeError(f"Actor/orphan engine is still active (PIDs {actors}); export requires quiescence")
        yield


def _state_digest(value):
    """Stable byte-level tensor/RNG/container identity, without pickle loading."""
    import numpy as np
    import torch
    digest = hashlib.sha256()
    def put(data):
        digest.update(len(data).to_bytes(8, "little")); digest.update(data)
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            put(b"torch"); put(str(tensor.dtype).encode()); put(canonical(list(tensor.shape)).encode())
            put(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise ValueError("Object arrays are not permitted in recovery state")
            put(b"numpy"); put(item.dtype.str.encode()); put(canonical(list(item.shape)).encode())
            put(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            put(b"dict")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            put(type(item).__name__.encode())
            for child in item:
                visit(child)
        elif isinstance(item, bytes):
            put(b"bytes"); put(item)
        elif isinstance(item, (str, bool, int, float)) or item is None:
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("Nonfinite recovery metadata")
            put(type(item).__name__.encode()); put(repr(item).encode())
        else:
            raise ValueError(f"Unsupported recovery state type: {type(item).__name__}")
    visit(value)
    return digest.hexdigest()


def checkpoint_record(path, *, candidate=False, baseline=None):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model_key = "model_state_dict" if "model_state_dict" in payload else "policy_state_dict"
    if not isinstance(payload, dict) or not isinstance(payload.get(model_key), dict):
        raise ValueError(f"Checkpoint has no model state: {path.name}")
    model = payload[model_key]
    if not model or any(not isinstance(tensor, torch.Tensor) or
                        not torch.isfinite(tensor).all().item() for tensor in model.values()):
        raise ValueError("Checkpoint contains invalid/nonfinite model tensors")
    if candidate:
        if (any(key not in payload for key in STATE_KEYS) or payload.get("policy_head_only") is not True or
                payload.get("parent_checkpoint_sha256") != CP2_SHA256 or
                not isinstance(payload["optimizer_state_dict"], dict) or
                not isinstance(payload["batch_indices"], dict) or
                set(payload["batch_indices"]) != {"root", "anchor"}):
            raise ValueError("Candidate lacks complete model/optimizer/RNG/batch state or frozen baseline identity")
        if (payload.get("updates") != 128 or payload.get("arm") != path.stem or
            any(not isinstance(payload.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", payload[key])
                for key in ("fit_identity_sha256", "dataset_sha256", "dataset_metadata_sha256", "batch_sha256"))):
            raise ValueError("Final candidate lacks 128 completed updates or final fit provenance")
        for value in payload["batch_indices"].values():
            if not isinstance(value, torch.Tensor) or value.dtype != torch.int64 or tuple(value.shape) != (128, 32):
                raise ValueError("Final candidate has incomplete matched batch state")
        states = payload["optimizer_state_dict"].get("state", {})
        if len(states) != 2 or any(float(state.get("step", -1)) != 128 for state in states.values()):
            raise ValueError("Final candidate optimizer has not completed 128 updates")
        if baseline is not None:
            if set(model) != set(baseline):
                raise ValueError("Candidate model layout differs from frozen cp2")
            for key, value in baseline.items():
                if key not in {"policy.weight", "policy.bias"} and _state_digest(model[key]) != _state_digest(value):
                    raise ValueError(f"Candidate changed frozen tensor: {key}")
    elif model_key == "policy_state_dict":
        if (any(key not in payload for key in ("optimizer_state_dict", "torch_rng", "numpy_rng", "batch_sha256", "history")) or
            type(payload.get("step")) is not int or not 0 <= payload["step"] <= 128 or
            len(payload["history"]) != payload["step"]):
            raise ValueError("Partial policy fit lacks resumable optimizer/RNG/history state")
    return {"payload_sha256": _state_digest(payload), "state_sha256": {
                key: _state_digest(payload[key]) for key in (*STATE_KEYS, "policy_state_dict") if key in payload},
            "model_state_key": model_key, "model_tensor_count": len(model),
            "arm": payload.get("arm"), "updates": payload.get("updates"), "step": payload.get("step")}, payload


def collect_files(root):
    paths = []
    for directory in (RUN, PROGRESS):
        base = checked_path(root, directory)
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            name = path.relative_to(root).as_posix()
            if any(part.startswith(".") for part in path.relative_to(base).parts):
                continue
            checked_path(root, name)
            if path.is_file() and category(name):
                paths.append(path)
            elif not path.is_file() and not path.is_dir():
                raise ValueError(f"Non-regular recovery input: {name}")
    for name in REFERENCE_FILES:
        path = checked_path(root, name)
        if not path.is_file():
            raise FileNotFoundError(f"Required v5 reference missing: {name}")
        paths.append(path)
    baseline = checked_path(root, CP2)
    if sha256(baseline) != CP2_SHA256:
        raise ValueError("Frozen cp2 dependency checksum mismatch")
    if baseline.stat().st_size <= CP2_INLINE_LIMIT:
        paths.append(baseline)
    return sorted(set(paths))


def verify_npz(path):
    import numpy as np
    if path.suffix == ".npy":
        arrays = [(path.name, np.load(path, allow_pickle=False))]
    else:
        with np.load(path, allow_pickle=False) as bundle:
            arrays = [(name, bundle[name]) for name in bundle.files]
    for name, array in arrays:
        if array.dtype.hasobject or (np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all()):
            raise ValueError(f"Invalid array in {path.name}:{name}")


def _read_json_rows(path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [json.loads(path.read_text())]


def _contains_benchmark(value):
    if isinstance(value, dict):
        if value.get("source") in ("benchmark", "evaluation") or value.get("benchmark_data_used") is True:
            return True
        if "evaluation_cell_id" in value or "opponent_java_sha256" in value:
            return True
        return any(_contains_benchmark(child) for child in value.values())
    return isinstance(value, list) and any(_contains_benchmark(child) for child in value)


def verify_terminal_record(record, *, source=False):
    """Independent verification of a completed collector transcript, not its score selection."""
    raw = record.get("terminal_state", {})
    winner = record.get("winner")
    state = {key: value for key, value in raw.items() if key != "snapshot"} if isinstance(raw, dict) else {}
    if (record.get("terminal") is not True or record.get("valid_game") is not True or
        record.get("benchmark_failures") != [] or type(winner) is not int or winner not in (-1, 0, 1) or
        state.get("terminal") is not True or state.get("winner") != winner or
        type(record.get("terminal_round")) is not int or record["terminal_round"] <= 0 or
        state.get("round") != record["terminal_round"] or
        hashlib.sha256(canonical(state).encode()).hexdigest() != record.get("terminal_state_sha256") or
        not isinstance(record.get("actions"), list) or
        hashlib.sha256(canonical(record["actions"]).encode()).hexdigest() != record.get("actions_sha256")):
        raise ValueError("Completed collector transcript lacks consistent terminal/action proof")
    if source and (record.get("complete") is not True or record.get("source") != "selfplay"):
        raise ValueError("Completed source must be complete own self-play")
    if not source:
        identity = record.get("identity")
        player = record.get("root_player", identity.get("root_player") if isinstance(identity, dict) else None)
        if type(player) is not int or player not in (0, 1) or record.get("score") != (.5 if winner == -1 else float(winner == player)):
            raise ValueError("Completed continuation score differs from root-team terminal outcome")


def verify_recorded_files(root, paths):
    """Check declared data/checkpoint file hashes, keeping semantic hashes opaque."""
    names = {path.relative_to(root).as_posix() for path in paths}
    pairs = (("cache_npz", "cache_sha256"), ("data_npz", "data_sha256"),
             ("features_npz", "features_sha256"), ("path", "sha256"),
             ("file", "sha256"), ("artifact", "artifact_sha256"),
             ("source_json", "source_sha256"), ("checkpoint", "checkpoint_sha256"))
    def visit(value, path):
        if isinstance(value, dict):
            for key, hash_key in pairs:
                name, expected = value.get(key), value.get(hash_key)
                if not isinstance(name, str) or not isinstance(expected, str):
                    continue
                raw = Path(name)
                if raw.is_absolute():
                    if not raw.is_relative_to(root):
                        raise ValueError(f"External data reference in {path.relative_to(root)}")
                    candidates = [raw.relative_to(root).as_posix()]
                else:
                    candidates = [name, (path.parent / raw).relative_to(root).as_posix()]
                matched = [entry for entry in candidates if entry in names]
                if matched:
                    if sha256(checked_path(root, matched[0])) != expected:
                        raise ValueError(f"Recorded file checksum mismatch: {matched[0]}")
                elif key in {"cache_npz", "data_npz", "features_npz", "source_json"}:
                    raise ValueError(f"Recorded owned data is missing from recovery: {name}")
            if "npz_sha256" in value:
                data = path.with_suffix(".npz")
                if not data.is_file() or sha256(data) != value["npz_sha256"]:
                    raise ValueError(f"Dataset NPZ checksum mismatch: {path.relative_to(root)}")
            if path.is_relative_to(root / RUN) and isinstance(value.get("sha256"), str) and path.with_suffix(".pt").is_file():
                if sha256(path.with_suffix(".pt")) != value["sha256"]:
                    raise ValueError(f"Saved fit generation checksum mismatch: {path.relative_to(root)}")
            if path.name == "manifest.json" and path.parent.name in {"fit", "fit.pending"}:
                for filename, expected in value.get("files", {}).items():
                    if PurePosixPath(filename).name != filename:
                        raise ValueError("Fit manifest references a non-local file")
                    target = path.parent / filename
                    if target.relative_to(root).as_posix() not in names or sha256(target) != expected:
                        raise ValueError(f"Fit manifest checksum mismatch: {filename}")
            for child in value.values():
                visit(child, path)
        elif isinstance(value, list):
            for child in value:
                visit(child, path)
    for path in paths:
        name = path.relative_to(root).as_posix()
        if path.suffix in {".npz", ".npy"}:
            verify_npz(path)
        elif path.suffix in {".json", ".jsonl"}:
            for row in _read_json_rows(path):
                if name.startswith((RUN + "/dataset/", RUN + "/sources/", RUN + "/fit/")) and _contains_benchmark(row):
                    raise ValueError(f"Benchmark evidence is mixed into gradient data: {name}")
                if isinstance(row, dict) and row.get("status") == "ok":
                    if name.startswith(RUN + "/sources/"):
                        verify_terminal_record(row, source=True)
                    elif name.startswith(PROGRESS + "/cells/"):
                        verify_terminal_record(row)
                visit(row, path)


def plan_unlocked(root):
    paths = collect_files(root)
    verify_recorded_files(root, paths)
    baseline_record, baseline = checkpoint_record(checked_path(root, CP2))
    checkpoints = {}
    for path in paths:
        if path.suffix == ".pt":
            name = path.relative_to(root).as_posix()
            checkpoints[name], payload = checkpoint_record(path, candidate=name in CANDIDATES,
                                                          baseline=baseline["model_state_dict"])
            if name in CANDIDATES:
                for data_name, key in (("training.npz", "dataset_sha256"), ("training.json", "dataset_metadata_sha256")):
                    if sha256(checked_path(root, f"{RUN}/dataset/{data_name}")) != payload[key]:
                        raise ValueError("Candidate dataset provenance differs from retained gradient data")
    if any(name in checkpoints for name in CANDIDATES):
        fit_manifest_path = checked_path(root, f"{RUN}/fit/manifest.json")
        if not fit_manifest_path.is_file():
            raise ValueError("Final candidate pair lacks its completed fit manifest")
        fit_manifest = json.loads(fit_manifest_path.read_text())
        if (not all(name in checkpoints for name in CANDIDATES) or fit_manifest.get("complete") is not True or
            set(fit_manifest.get("checkpoints", {})) != {"terminal_head", "puct_head"}):
            raise ValueError("Final candidate pair is incomplete")
    status_path = checked_path(root, f"{RUN}/status.json")
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    progress_status = checked_path(root, f"{PROGRESS}/status.json")
    if not status and progress_status.exists():
        status = json.loads(progress_status.read_text())
    coverage = {"completed_training_sources": 0, "completed_validation_sources": 0,
                "successful_selection_cells": 0, "successful_validation_cells": 0}
    for path in paths:
        if path.suffix != ".json":
            continue
        name = path.relative_to(root).as_posix()
        if path.parent == root / RUN / "sources":
            row = json.loads(path.read_text())
            split = row.get("split")
            if row.get("status") == "ok" and split in ("training", "validation"):
                coverage[f"completed_{split}_sources"] += 1
        elif name.startswith(PROGRESS + "/cells/"):
            row = json.loads(path.read_text())
            stage = row.get("stage")
            if row.get("status") == "ok" and stage in ("selection", "validation"):
                coverage[f"successful_{stage}_cells"] += 1
    for split in ("training", "validation"):
        path = checked_path(root, f"{PROGRESS}/roots-{split}.json")
        if path.exists():
            value = json.loads(path.read_text())
            roots = value.get("roots", []) if isinstance(value, dict) else value
            coverage[f"frozen_{split}_roots"] = len(roots)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"schema": SCHEMA, "source_root": str(root), "git_commit": commit,
        "phase": status.get("phase", "unknown_partial"), "status": status, "coverage": coverage,
        "completed_candidate_count": sum(name in checkpoints for name in CANDIDATES),
        "both_candidates_present": all(name in checkpoints for name in CANDIDATES),
        "files": [{"path": path.relative_to(root).as_posix(), "size": path.stat().st_size,
                   "sha256": sha256(path), "category": category(path.relative_to(root).as_posix())} for path in paths],
        "checkpoints": checkpoints,
        "dependency": {"cp2_path": CP2, "cp2_sha256": CP2_SHA256, "cp2_included": checked_path(root, CP2) in paths,
                       "cp2_payload_sha256": baseline_record["payload_sha256"],
                       "initialization_ancestor": baseline.get("config", {}).get("initialization"),
                       "verified_warmstart_export": "artifacts/warmstart-v3/iteration-002/resume.tar.gz"},
        "recovery_policy": {"benchmark_data_separate": True, "no_data_omitted": True,
            "external_runtime": "Provision pinned Python/JDK/engine/bridge/benchmark dependencies from main code; compare recorded experiment runtime identities before resuming. No downloaded source, classes, vendor, or virtual environment is included.",
            "source_snapshots": "Owned source snapshots are historical evidence; never replace current main code automatically.",
            "restore": "Absent run only; existing progress must be byte-identical. Identity-bound absolute paths are never rewritten. Relocated restoration is stopped for review."}}


def plan(root=ROOT):
    root = Path(root).resolve()
    with experiment_locks(root):
        return plan_unlocked(root)


@contextmanager
def verified_staging(archive_path, *, expected_sha256=None):
    archive_path = Path(archive_path).resolve()
    sidecar = archive_path.with_name(archive_path.name + ".manifest.json")
    if expected_sha256 is None:
        if not sidecar.is_file():
            raise ValueError("Verification requires a sidecar manifest or explicit --sha256")
        expected_sha256 = json.loads(sidecar.read_text())["archive_sha256"]
    if sha256(archive_path) != expected_sha256:
        raise ValueError("Recovery archive SHA256 mismatch")
    with tempfile.TemporaryDirectory(prefix="terminal-v5-verify-") as temporary:
        stage = Path(temporary)
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or MANIFEST not in names:
                raise ValueError("Recovery archive has duplicate members or no manifest")
            for member in members:
                checked_path(stage, member.name)
                if not member.isfile() or member.size < 0:
                    raise ValueError("Recovery archive must contain regular files only")
            manifest = json.load(archive.extractfile(MANIFEST))
            if manifest.get("schema") != SCHEMA:
                raise ValueError("Unsupported recovery schema")
            entries = {entry["path"]: entry for entry in manifest["files"]}
            if len(entries) != len(manifest["files"]) or set(names) != set(entries) | {MANIFEST}:
                raise ValueError("Recovery archive file coverage differs from manifest")
            for member in members:
                if member.name == MANIFEST:
                    continue
                entry = entries[member.name]
                if category(member.name) != entry["category"] or not entry["category"] or member.size != entry["size"]:
                    raise ValueError("Recovery archive contains an unapproved or mis-sized payload")
                target = checked_path(stage, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                if sha256(target) != entry["sha256"]:
                    raise ValueError(f"Staged recovery checksum mismatch: {member.name}")
        for name, expected in manifest["checkpoints"].items():
            record, _ = checkpoint_record(checked_path(stage, name), candidate=name in CANDIDATES)
            if record != expected:
                raise ValueError(f"Staged model/optimizer/RNG/batch payload differs: {name}")
        yield stage, manifest


def verify(archive, *, expected_sha256=None):
    with verified_staging(archive, expected_sha256=expected_sha256) as (_, manifest):
        return {"schema": SCHEMA, "verified": True, "archive_sha256": sha256(archive),
                "files": len(manifest["files"]), "completed_candidate_count": manifest["completed_candidate_count"],
                "checkpoint_state_verified": sorted(manifest["checkpoints"]), "phase": manifest["phase"]}


def create_export(root=ROOT):
    root = Path(root).resolve()
    with experiment_locks(root):
        manifest = plan_unlocked(root)
        manifest["created_at"] = datetime.now(timezone.utc).isoformat()
        files = [checked_path(root, entry["path"]) for entry in manifest["files"]]
        identities = {path: file_identity(path) for path in files}
        output = checked_path(root, "artifacts/terminal-target-head-v5")
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".v5-export-", dir=output) as temporary:
            temporary = Path(temporary)
            archive = temporary / "recovery.tar.gz"
            with tarfile.open(archive, "w:gz", compresslevel=3) as bundle:
                for entry, path in zip(manifest["files"], files):
                    info = tarfile.TarInfo(entry["path"])
                    info.size, info.mode, info.mtime = entry["size"], 0o644, 0
                    with path.open("rb") as stream:
                        bundle.addfile(info, stream)
                    if file_identity(path) != identities[path] or sha256(path) != entry["sha256"]:
                        raise RuntimeError(f"Source changed during recovery export: {entry['path']}")
                content = (canonical(manifest) + "\n").encode()
                info = tarfile.TarInfo(MANIFEST); info.size, info.mode = len(content), 0o644
                bundle.addfile(info, io.BytesIO(content))
            if collect_files(root) != files or any(file_identity(path) != identities[path] for path in files):
                raise RuntimeError("Recovery inputs changed during export")
            digest = sha256(archive)
            verify(archive, expected_sha256=digest)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            name = f"recovery-{stamp}-{digest[:12]}.tar.gz"
            target = output / name
            sidecar = {"schema": SCHEMA, "archive": name, "archive_sha256": digest,
                       "archive_size": archive.stat().st_size, "manifest": manifest}
            with archive.open("rb") as stream:
                os.fsync(stream.fileno())
            staged_sidecar = temporary / "sidecar.json"
            staged_sidecar.write_text(canonical(sidecar) + "\n")
            with staged_sidecar.open("rb") as stream:
                os.fsync(stream.fileno())
            os.link(archive, target)
            os.link(staged_sidecar, target.with_name(name + ".manifest.json"))
            return {"schema": SCHEMA, "archive": str(target), "manifest": str(target) + ".manifest.json",
                    "sha256": digest, "size": sidecar["archive_size"], "files": len(files),
                    "completed_candidate_count": manifest["completed_candidate_count"], "verified": True}


def restore(archive, root=ROOT, *, expected_sha256=None):
    root = Path(root).resolve()
    with verified_staging(archive, expected_sha256=expected_sha256) as (stage, manifest):
        with experiment_locks(root):
            run = checked_path(root, RUN)
            if any(path.relative_to(run).as_posix() != "experiment.lock" for path in run.rglob("*")):
                raise FileExistsError("Existing v5 run must never be overwritten by recovery")
            selected = [entry for entry in manifest["files"] if entry["category"] != "historical_reference"]
            for entry in selected:
                target = checked_path(root, entry["path"])
                if target.exists() and (not target.is_file() or sha256(target) != entry["sha256"]):
                    raise FileExistsError(f"Recovery conflicts with existing evidence: {entry['path']}")
            restored = []
            for entry in selected:
                target = checked_path(root, entry["path"])
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with checked_path(stage, entry["path"]).open("rb") as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
                        output.flush(); os.fsync(output.fileno())
                    restored.append(entry["path"])
                if sha256(target) != entry["sha256"]:
                    raise ValueError("Restored bytes differ from verified staging")
            for name, expected in manifest["checkpoints"].items():
                if checkpoint_record(checked_path(root, name), candidate=name in CANDIDATES)[0] != expected:
                    raise ValueError("Restored model/optimizer/RNG/batch state differs")
            relocated = str(root) != manifest["source_root"]
            requirements = ["original_canonical_repository_path"] if relocated else []
            required = {entry["path"]: entry["sha256"] for entry in manifest["files"]
                        if entry["category"] == "historical_reference"}
            required[CP2] = manifest["dependency"]["cp2_sha256"]
            ancestor = manifest["dependency"].get("initialization_ancestor")
            if isinstance(ancestor, dict) and "artifact" in ancestor:
                required[ancestor["artifact"]] = ancestor["artifact_sha256"]
            identity_file = checked_path(root, f"{PROGRESS}/identity.json")
            if identity_file.exists():
                pinned = json.loads(identity_file.read_text())
                required.update(pinned.get("sources", {}))
                runtime = pinned.get("runtime", {})
                required.update(runtime.get("files", {}))
                required.update(runtime.get("compiled_classes", {}))
                if "jdk_release_sha256" in runtime:
                    required["vendor/jdk/release"] = runtime["jdk_release_sha256"]
            for name, expected in required.items():
                dependency = checked_path(root, name)
                if not dependency.is_file() or sha256(dependency) != expected:
                    requirements.append(name)
            if requirements:
                checked_path(root, f"{RUN}/STOP").write_text(
                    "Recovered state is verified, but resume requires the original canonical repository path "
                    f"{manifest['source_root']} and matching pinned dependencies. Current main code was preserved.\n"
                    + "\n".join(requirements) + "\n")
            return {"schema": SCHEMA, "restored": True, "files_written": len(restored),
                    "completed_candidate_count": manifest["completed_candidate_count"],
                    "relocated": relocated, "resume_blocked": bool(requirements), "resume_requirements": requirements,
                    "source_root": manifest["source_root"], "checkpoint_state_verified": sorted(manifest["checkpoints"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "check", "export", "verify", "restore"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--sha256", dest="expected_sha256")
    args = parser.parse_args()
    if args.command in {"verify", "restore"} and args.archive is None:
        parser.error("--archive is required for verify/restore")
    if args.command == "export":
        result = create_export(args.root)
    elif args.command in {"plan", "check"}:
        result = plan(args.root)
        if args.command == "check":
            result = {"schema": SCHEMA, "checked": True, "files": len(result["files"]),
                      "completed_candidate_count": result["completed_candidate_count"], "phase": result["phase"]}
    elif args.command == "verify":
        result = verify(args.archive, expected_sha256=args.expected_sha256)
    else:
        result = restore(args.archive, args.root, expected_sha256=args.expected_sha256)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
