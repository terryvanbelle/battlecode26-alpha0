#!/usr/bin/env python3
"""Offline recovery for the training-only cached-head-constraint-v7 diagnostic.

Commands: plan/check/export --root REPO; verify/restore --archive FILE --root REPO.
Run export only after actors, audits and reports are quiescent. All five locks
and conservative process checks apply. STOP files permit read-only export and
are preserved. This helper never fits, runs games, publishes, or changes runtime.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile

# Reuse pure checksum/tensor/path utilities, without modifying the v6 module or
# importing any actor/fitter. No v6 collection, verification or fit is executed.
_spec = importlib.util.spec_from_file_location("_v7_archive_utilities", Path(__file__).with_name("export_terminal_v6.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
sha256, checked_path, canonical, _state_digest = A.sha256, A.checked_path, A.canonical, A._state_digest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "cached-head-constraint-v7"
RUN, PROGRESS = "runs/" + EXPERIMENT_ID, "progress/" + EXPERIMENT_ID
SCHEMA, MANIFEST = EXPERIMENT_ID + "-recovery-v1", "V7-RECOVERY-MANIFEST.json"
CONFIG, PROTOCOL = "configs/" + EXPERIMENT_ID + ".json", "docs/" + EXPERIMENT_ID + ".md"
ARMS, SNAPSHOTS = {"kl_1": 1.0, "kl_0": 0.0}, (128, 512, 2048)
PINS = {
    "checkpoint": ("runs/warmstart-v3/checkpoint-0002.pt", "20d9e813349010630940e1cdfbaee5206490c7d323997688d26f47508bbeee7c"),
    "dataset": ("runs/terminal-target-head-v6/dataset/training.npz", "c6d0043b1de3c5d228598d749637cd73efa297da111669dd1431363baf9e5f26"),
    "dataset_metadata": ("runs/terminal-target-head-v6/dataset/training.json", "020b5615a430b986802f2ab2ebfbeb6eca532488d1ca1179a863987a8a876d7b"),
    "reference_terminal_checkpoint": ("runs/terminal-target-head-v6/fit/terminal_head.pt", "1479fe9474c3f149497fb9994ff53ad6e7fcdf79021fb12a444346b862530718"),
    "reference_batches": ("runs/terminal-target-head-v6/fit/batches.npz", "609a31a83326b44c5cf642a5d1804ad0d643e45ac860c3aea4b42ad6924cbd6a"),
    "v6_identity": ("progress/terminal-target-head-v6/identity.json", "1cd46fdc649c072e7ab0d736afd8960522d6db30c194375edef513feb4db608c"),
    "v6_final_audit": ("progress/terminal-target-head-v6/independent-final-audit.json", "ab8f824a645432916417e8142d61c7cd13ec5e69561b96869115ff95288dd564"),
    "training_diagnosis": ("progress/terminal-target-head-v6/training-underfit.json", "4ff88c4f8fe68b623d35941415a92ac3741907d840da4e7aea405bf9b0f4da68"),
}
V6_ARCHIVE = "artifacts/terminal-target-head-v6/recovery-20261110T092723489573Z-b5f724bb879e.tar.gz"
V6_ARCHIVE_SHA256 = "b5f724bb879e9a36b816c29c2ac306070d1657b418f2fd25960f2478c2fe3c38"
LOCKS = ("runs/campaign/campaign.lock", "runs/warmstart-v3/experiment.lock",
         "runs/terminal-target-head-v5/experiment.lock", "runs/terminal-target-head-v6/experiment.lock", RUN + "/experiment.lock")
STOPS = tuple(str(PurePosixPath(name).parent / "STOP") for name in LOCKS)
PROC = Path("/proc")
SUFFIXES = {".json", ".jsonl", ".npz", ".npy", ".pt", ".csv", ".md", ".png", ".svg", ".txt"}


def category(name):
    checked_path(Path("/nonexistent-v7-root"), name)
    if name in (CONFIG, PROTOCOL):
        return "historical_reference"
    if name in {value[0] for value in PINS.values()}:
        return "pinned_input"
    path = PurePosixPath(name)
    if path.name == "STOP" or path.suffix in {".lock", ".pid", ".log", ".tmp"}:
        return None
    if name.startswith(PROGRESS + "/source/") and path.suffix in {".py", ".json", ".md"}:
        return "source_snapshot"
    if not name.startswith((RUN + "/", PROGRESS + "/")) or path.suffix not in SUFFIXES:
        return None
    if any(part in {"vendor", "build", "venv", "benchmarks", "__pycache__"} for part in path.parts):
        raise ValueError("Downloaded/runtime/benchmark files are outside the v7 archive allowlist")
    return "owned_run" if name.startswith(RUN + "/") else "owned_progress"


def active_actor_pids(root):
    result = []
    if not PROC.is_dir():
        raise RuntimeError("Actor inspection requires /proc")
    for process in PROC.iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            args = [part.decode(errors="replace") for part in (process / "cmdline").read_bytes().split(b"\0") if part]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        matching = any(arg in {"alpha0.EngineBridge", "alpha0.campaign", "alpha0.evaluate"} or
                       arg.startswith(("alpha0.terminal_v", "alpha0.constraint_v")) or
                       (Path(arg).name.startswith("run_") and Path(arg).name.endswith(".py") and
                        any(word in Path(arg).name for word in ("warmstart", "terminal", "constraint"))) for arg in args)
        if not matching:
            continue
        try:
            cwd = (process / "cwd").resolve()
        except PermissionError:
            result.append(int(process.name)); continue
        except (FileNotFoundError, ProcessLookupError):
            continue
        if cwd == root:
            result.append(int(process.name))
    return sorted(result)


@contextmanager
def experiment_locks(root):
    with ExitStack() as stack:
        for name in LOCKS:
            path = checked_path(root, name); path.parent.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context(path.open("a"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Actor is active; cannot acquire {name}") from error
        actors = active_actor_pids(root)
        if actors:
            raise RuntimeError(f"Actor/orphan engine may be active: {actors}")
        yield


def checkpoint_record(path, config=None, baseline=None, fit_context=None):
    """Load weights-only and fingerprint every model/optimizer/RNG/history byte."""
    import torch
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary")
    generation = path.parent.name == "states" and path.parent.parent.name in ARMS
    snapshot = path.parent.name in ARMS and re.fullmatch(r"checkpoint-[0-9]{4}\.pt", path.name) is not None
    model_key = "policy_state_dict" if generation else "model_state_dict"
    model = payload.get(model_key)
    if not isinstance(model, dict) or not model or any(not isinstance(t, torch.Tensor) or not torch.isfinite(t).all().item() for t in model.values()):
        raise ValueError("Checkpoint model tensors are absent or invalid")
    if generation or snapshot:
        arm = path.parent.parent.name if generation else path.parent.name
        step = int(path.stem.split("-")[1]); field = "step" if generation else "updates"
        if (payload.get("arm") != arm or payload.get(field) != step or
            (generation and (step not in range(16, 2049, 16))) or (snapshot and step not in SNAPSHOTS) or
            payload.get("coefficient" if generation else "anchor_kl_weight") != ARMS[arm]):
            raise ValueError("Recovery state arm/update/coefficient differs from its filename")
        required = {"optimizer_state_dict", "torch_rng", "numpy_rng", "batch_sha256", "frozen_state_sha256"}
        if not required <= payload.keys():
            raise ValueError("Recovery state lacks optimizer/RNG/batch provenance")
        history = payload.get("history" if generation else "fit_history", [])
        if len(history) != step or [row.get("step") for row in history] != list(range(1, step + 1)):
            raise ValueError("Recovery loss history is incomplete or out of order")
        policy = model if generation else {name: model["policy." + name] for name in ("weight", "bias")}
        if set(policy) != {"weight", "bias"}:
            raise ValueError("Recovery policy tensor inventory differs")
        if baseline is not None:
            frozen = hashlib.sha256()
            for key, tensor in sorted(baseline.items()):
                if key in {"policy.weight", "policy.bias"}: continue
                header = json.dumps([key, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
                frozen.update(len(header).to_bytes(8, "little")); frozen.update(header)
                frozen.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            if payload["frozen_state_sha256"] != frozen.hexdigest():
                raise ValueError("Recovery frozen-state identity differs from cp2")
            for name, tensor in policy.items():
                original = baseline["policy." + name]
                if tensor.shape != original.shape or tensor.dtype != original.dtype:
                    raise ValueError("Recovery policy tensor shape/dtype differs from cp2")
        states = payload["optimizer_state_dict"].get("state", {})
        if set(states) != {0, 1}:
            raise ValueError("Recovery optimizer must preserve both policy parameters")
        for index, name in enumerate(("weight", "bias")):
            state = states[index]
            if set(state) != {"step", "exp_avg", "exp_avg_sq"} or float(state["step"]) != step:
                raise ValueError("Recovery optimizer update count differs")
            for key in ("exp_avg", "exp_avg_sq"):
                tensor = state[key]
                if tensor.shape != policy[name].shape or tensor.dtype != policy[name].dtype or not torch.isfinite(tensor).all().item():
                    raise ValueError("Recovery optimizer moments are invalid")
            if not (state["exp_avg_sq"] >= 0).all().item():
                raise ValueError("Recovery optimizer second moment is negative")
        if fit_context is not None:
            if payload["batch_sha256"] != fit_context["batch_sha256"] or _state_digest(payload["numpy_rng"]) != _state_digest(fit_context["rng"]):
                raise ValueError("Recovery batch/RNG provenance differs")
            if not torch.equal(payload["torch_rng"], torch.Generator().manual_seed(config["fit"]["seed"]).get_state()):
                raise ValueError("Recovery Torch RNG differs from the fixed seed")
            if generation and payload.get("fit_config_sha256") != fit_context["settings_sha256"]:
                raise ValueError("Generation fit configuration differs")
            if snapshot and payload.get("fit_identity_sha256") != fit_context["identity_sha256"]:
                raise ValueError("Snapshot fit identity differs")
        if snapshot:
            if (payload.get("experiment_id") != EXPERIMENT_ID or payload.get("policy_head_only") is not True or
                any(payload.get(key) is not False for key in ("automatic_promotion", "benchmark_data_used", "validation_data_used"))):
                raise ValueError("Snapshot is not the training-only v7 policy-head diagnostic")
            batches = payload.get("batch_indices", {})
            if set(batches) != {"root", "anchor"} or any(t.dtype != torch.int64 or tuple(t.shape) != (2048, 32) for t in batches.values()):
                raise ValueError("Snapshot is missing the complete extended batch schedule")
            if fit_context is not None:
                for name, tensor in batches.items():
                    if not torch.equal(tensor, torch.from_numpy(fit_context["batches"][name])):
                        raise ValueError("Snapshot batch arrays differ from the preserved schedule")
            if config:
                for key, input_name in (("parent_checkpoint_sha256", "checkpoint"), ("dataset_sha256", "dataset"),
                                        ("dataset_metadata_sha256", "dataset_metadata"), ("reference_v6_checkpoint_sha256", "reference_terminal_checkpoint")):
                    if payload.get(key) != config[input_name]["sha256"]:
                        raise ValueError("Snapshot input dependency hash differs")
            if baseline is not None:
                if set(model) != set(baseline):
                    raise ValueError("Snapshot model inventory differs from cp2")
                for key, tensor in baseline.items():
                    if key not in {"policy.weight", "policy.bias"} and _state_digest(model[key]) != _state_digest(tensor):
                        raise ValueError("Snapshot changed frozen cp2 tensors")
    keys = ("model_state_dict", "policy_state_dict", "optimizer_state_dict", "torch_rng", "numpy_rng", "batch_indices", "history", "fit_history")
    return {"payload_sha256": _state_digest(payload), "state_sha256": {key: _state_digest(payload[key]) for key in keys if key in payload},
            "kind": "generation" if generation else "snapshot" if snapshot else "dependency", "arm": payload.get("arm"),
            "step": payload.get("step", payload.get("updates"))}, payload


def fit_context(root):
    import numpy as np
    directory = checked_path(root, RUN + "/fit")
    if not directory.exists() or not (directory / "identity.json").exists():
        return None
    identity = json.loads((directory / "identity.json").read_text())
    rng = json.loads((directory / "batch-rng.json").read_text())
    with np.load(directory / "batches.npz", allow_pickle=False) as saved:
        if set(saved.files) != {"root", "anchor"}:
            raise ValueError("Extended batch array inventory differs")
        arrays = {name: saved[name] for name in saved.files}
    digest = hashlib.sha256()
    for name in ("root", "anchor"):
        array = arrays[name]
        if array.dtype != np.int64 or array.shape != (2048, 32) or (array < 0).any() or (array >= (32 if name == "root" else 512)).any():
            raise ValueError("Extended batch indices are invalid")
        digest.update(json.dumps([name, list(array.shape)], separators=(",", ":")).encode())
        digest.update(np.ascontiguousarray(array, dtype="<i8").tobytes())
    digest.update(hashlib.sha256(canonical(rng).encode()).hexdigest().encode())
    if identity.get("batch_sha256") != digest.hexdigest():
        raise ValueError("Extended batch schedule/RNG digest differs")
    return {"identity_sha256": hashlib.sha256(canonical(identity).encode()).hexdigest(),
            "settings_sha256": hashlib.sha256(canonical(identity["settings"]).encode()).hexdigest(),
            "batch_sha256": digest.hexdigest(), "rng": rng, "batches": arrays}


def collect_files(root):
    result = set()
    for name in (RUN, PROGRESS):
        base = checked_path(root, name)
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if any(part.startswith(".") for part in path.relative_to(base).parts):
                continue
            relative = path.relative_to(root).as_posix(); checked_path(root, relative)
            if path.is_file() and category(relative):
                result.add(path)
            elif not path.is_file() and not path.is_dir():
                raise ValueError("Non-regular recovery input")
    for name in (CONFIG, PROTOCOL, *(value[0] for value in PINS.values())):
        path = checked_path(root, name)
        if not path.is_file():
            raise FileNotFoundError(f"Required recovery input missing: {name}")
        result.add(path)
    return sorted(result)


def _fit_inventory(root, paths, checkpoints):
    """Retain all generations; reject gaps and incomplete claimed completion."""
    fit = checked_path(root, RUN + "/fit")
    context = fit_context(root)
    for path in paths:
        if (path.suffix == ".json" and path.is_relative_to(fit) and
            (re.fullmatch(r"step-[0-9]{4}\.json", path.name) or re.fullmatch(r"checkpoint-[0-9]{4}\.json", path.name)) and
            not path.with_suffix(".pt").is_file()):
            raise ValueError("Checkpoint/generation sidecar has no matching PT payload")
    snapshots, generations = [], {arm: [] for arm in ARMS}
    for name, record in checkpoints.items():
        if record["kind"] == "snapshot": snapshots.append(name)
        if record["kind"] == "generation": generations[record["arm"]].append(record["step"])
        if record["kind"] in {"snapshot", "generation"}:
            path = checked_path(root, name); sidecar = path.with_suffix(".json")
            row = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
            if (context is None or row.get("sha256") != sha256(path) or
                row.get("fit_identity_sha256") != context["identity_sha256"] or
                row.get("step" if record["kind"] == "generation" else "updates") != record["step"] or
                (record["kind"] == "snapshot" and row.get("arm") != record["arm"])):
                raise ValueError("Checkpoint/generation sidecar is missing or corrupt")
    for arm, steps in generations.items():
        if steps and sorted(steps) != list(range(16, max(steps) + 1, 16)):
            raise ValueError(f"Recovery generation sequence has gaps: {arm}")
    manifest_path = fit / "manifest.json"
    complete = manifest_path.is_file()
    if complete:
        manifest = json.loads(manifest_path.read_text())
        required = {"identity.json", "batches.npz", "batch-rng.json", "parent-metadata.json", "reproduction-128.json"}
        for arm in ARMS:
            required.update(f"{arm}/states/step-{step:04d}.{suffix}" for step in range(16, 2049, 16) for suffix in ("pt", "json"))
            required.update(f"{arm}/checkpoint-{step:04d}.{suffix}" for step in SNAPSHOTS for suffix in ("pt", "json"))
        actual = {p.relative_to(fit).as_posix() for p in paths if p.is_relative_to(fit) and p.name != "manifest.json"}
        if manifest.get("complete") is not True or set(manifest.get("files", {})) != actual or actual != required:
            raise ValueError("Completed v7 inventory must contain six snapshots and all 256 generation pairs")
        if len(snapshots) != 6 or any(len(steps) != 128 for steps in generations.values()):
            raise ValueError("Completed snapshot/generation coverage differs")
        for name, digest in manifest["files"].items():
            if sha256(checked_path(root, RUN + "/fit/" + name)) != digest:
                raise ValueError("Completed fit manifest checksum mismatch")
        if json.loads((fit / "reproduction-128.json").read_text()).get("passed") is not True:
            raise ValueError("Completed fit lacks its exact 128-step reproduction proof")
    return {"fit_complete": complete, "fixed_snapshots": len(snapshots),
            "generation_pairs": sum(len(values) for values in generations.values()),
            "latest_generation": {arm: max(steps, default=0) for arm, steps in generations.items()}}


def plan_unlocked(root):
    config = json.loads(checked_path(root, CONFIG).read_text())
    if (config.get("experiment_id") != EXPERIMENT_ID or config.get("updates") != 2048 or
        config.get("snapshots") != list(SNAPSHOTS) or config.get("checkpoint_interval") != 16):
        raise ValueError("V7 configuration does not match the fixed diagnostic schedule")
    paths = collect_files(root)
    for name, (relative, digest) in PINS.items():
        if config.get(name) != {"path": relative, "sha256": digest} or sha256(root / relative) != digest:
            raise ValueError(f"Pinned input checksum/config differs: {name}")
    baseline_record, baseline = checkpoint_record(root / PINS["checkpoint"][0])
    context = fit_context(root)
    checkpoints = {}
    for path in paths:
        if path.suffix in {".npz", ".npy"}: A.verify_npz(path)
        if path.suffix in {".json", ".jsonl"}: A._read_json_rows(path)
        if path.suffix == ".pt":
            name = path.relative_to(root).as_posix()
            checkpoints[name], _ = checkpoint_record(path, config=config, baseline=baseline["model_state_dict"], fit_context=context)
    coverage = _fit_inventory(root, paths, checkpoints)
    identity_path = checked_path(root, PROGRESS + "/identity.json")
    if identity_path.exists():
        identity = json.loads(identity_path.read_text())
        if identity.get("config") != config or identity.get("inputs") != {name: digest for name, digest in PINS.values()}:
            raise ValueError("Recorded experiment input identity differs")
        for name, digest in identity.get("sources", {}).items():
            snapshot = checked_path(root, PROGRESS + "/source/" + name)
            if not snapshot.is_file() or sha256(snapshot) != digest:
                raise ValueError("Pinned owned source snapshot is missing or differs")
    status_path = checked_path(root, RUN + "/status.json")
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"schema": SCHEMA, "source_root": str(root), "source_git_commit": commit,
        "phase": status.get("phase", "partial"), "status": status, "coverage": coverage,
        "files": [{"path": p.relative_to(root).as_posix(), "size": p.stat().st_size, "sha256": sha256(p),
                   "category": category(p.relative_to(root).as_posix())} for p in paths], "checkpoints": checkpoints,
        "inputs": {key: {"path": name, "sha256": digest} for key, (name, digest) in PINS.items()},
        "initialization_ancestor": baseline.get("config", {}).get("initialization"),
        "external_rollout_evidence": {"archive": V6_ARCHIVE, "sha256": V6_ARCHIVE_SHA256,
            "policy": "V6 raw source/continuation references remain in the verified v6 archive; only the pinned training numerical inputs and evidence identities are packaged here."},
        "recovery_policy": {"all_generations_retained": True, "training_only": True, "automatic_promotion": False,
            "restore": "Absent v7 run only; preserve current source and conflicting/newer evidence. No identity rebasing. Relocation or mismatched dependencies creates STOP.",
            "runtime": "Restore main-code/runtime versions matching preserved identity before resuming. No fitting, validation, games or publication runs during export/restore."}}


def plan(root=ROOT):
    root = Path(root).resolve()
    with experiment_locks(root): return plan_unlocked(root)


@contextmanager
def verified_staging(archive, *, expected_sha256=None):
    archive = Path(archive).resolve()
    if expected_sha256 is None:
        sidecar = archive.with_name(archive.name + ".manifest.json")
        if not sidecar.is_file(): raise ValueError("Archive sidecar or explicit SHA256 is required")
        expected_sha256 = json.loads(sidecar.read_text())["archive_sha256"]
    if sha256(archive) != expected_sha256: raise ValueError("Archive SHA256 mismatch")
    with tempfile.TemporaryDirectory(prefix="constraint-v7-verify-") as directory:
        stage = Path(directory)
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers(); names = [m.name for m in members]
            if len(names) != len(set(names)) or MANIFEST not in names: raise ValueError("Duplicate archive members or missing manifest")
            for member in members:
                checked_path(stage, member.name)
                if not member.isfile(): raise ValueError("Only regular archive files are permitted")
            manifest = json.load(bundle.extractfile(MANIFEST))
            if manifest.get("schema") != SCHEMA: raise ValueError("Unsupported v7 recovery schema")
            entries = {entry["path"]: entry for entry in manifest["files"]}
            if len(entries) != len(manifest["files"]) or set(names) != set(entries) | {MANIFEST}:
                raise ValueError("Archive coverage differs from manifest")
            for member in members:
                if member.name == MANIFEST: continue
                entry = entries[member.name]
                if not category(member.name) or category(member.name) != entry["category"] or member.size != entry["size"]:
                    raise ValueError("Unapproved/mis-sized archive member")
                target = checked_path(stage, member.name); target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("xb") as output: shutil.copyfileobj(source, output)
                if sha256(target) != entry["sha256"]: raise ValueError("Staged file checksum mismatch")
        config = json.loads((stage / CONFIG).read_text())
        expected_inputs = {key: {"path": name, "sha256": digest} for key, (name, digest) in PINS.items()}
        if manifest.get("inputs") != expected_inputs or any(config.get(key) != value for key, value in expected_inputs.items()):
            raise ValueError("Staged input dependency inventory differs")
        for value in expected_inputs.values():
            if sha256(checked_path(stage, value["path"])) != value["sha256"]:
                raise ValueError("Staged input dependency checksum differs")
        _, baseline = checkpoint_record(stage / PINS["checkpoint"][0])
        context = fit_context(stage)
        for name, expected in manifest["checkpoints"].items():
            if checkpoint_record(checked_path(stage, name), config=config, baseline=baseline["model_state_dict"], fit_context=context)[0] != expected:
                raise ValueError("Staged model/optimizer/RNG/history/batch state differs")
        if _fit_inventory(stage, [stage / name for name in entries], manifest["checkpoints"]) != manifest["coverage"]:
            raise ValueError("Staged snapshot/generation coverage differs")
        yield stage, manifest


def verify(archive, *, expected_sha256=None):
    with verified_staging(archive, expected_sha256=expected_sha256) as (_, manifest):
        return {"schema": SCHEMA, "verified": True, "archive_sha256": sha256(archive), "files": len(manifest["files"]),
                "coverage": manifest["coverage"], "checkpoint_payloads_verified": len(manifest["checkpoints"])}


def create_export(root=ROOT):
    root = Path(root).resolve()
    with experiment_locks(root):
        manifest = plan_unlocked(root); manifest["created_at"] = datetime.now(timezone.utc).isoformat()
        paths = [root / entry["path"] for entry in manifest["files"]]
        identities = {path: A.file_identity(path) for path in paths}
        output = checked_path(root, "artifacts/" + EXPERIMENT_ID); output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".v7-export-", dir=output) as temporary:
            archive = Path(temporary) / "recovery.tar.gz"
            with tarfile.open(archive, "w:gz", compresslevel=3) as bundle:
                for entry, path in zip(manifest["files"], paths):
                    member = tarfile.TarInfo(entry["path"]); member.size, member.mode, member.mtime = entry["size"], 0o644, 0
                    with path.open("rb") as stream: bundle.addfile(member, stream)
                    if A.file_identity(path) != identities[path] or sha256(path) != entry["sha256"]:
                        raise RuntimeError("Source changed during export")
                data = (canonical(manifest) + "\n").encode(); member = tarfile.TarInfo(MANIFEST)
                member.size, member.mode = len(data), 0o644; bundle.addfile(member, io.BytesIO(data))
            if collect_files(root) != paths or any(A.file_identity(p) != identities[p] for p in paths):
                raise RuntimeError("Recovery inventory changed during export")
            digest = sha256(archive); verify(archive, expected_sha256=digest)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            target = output / f"recovery-{stamp}-{digest[:12]}.tar.gz"
            sidecar = {"schema": SCHEMA, "archive": target.name, "archive_sha256": digest,
                       "archive_size": archive.stat().st_size, "manifest": manifest}
            staged = Path(temporary) / "sidecar.json"; staged.write_text(canonical(sidecar) + "\n")
            for path in (archive, staged):
                with path.open("rb") as stream: os.fsync(stream.fileno())
            os.link(archive, target); os.link(staged, target.with_name(target.name + ".manifest.json"))
            return {"archive": str(target), "manifest": str(target) + ".manifest.json", "sha256": digest,
                    "size": sidecar["archive_size"], "files": len(paths), "coverage": manifest["coverage"], "verified": True}


def restore(archive, root=ROOT, *, expected_sha256=None):
    root = Path(root).resolve()
    with verified_staging(archive, expected_sha256=expected_sha256) as (stage, manifest):
        with experiment_locks(root):
            run = checked_path(root, RUN)
            if any(p.relative_to(run).as_posix() != "experiment.lock" for p in run.rglob("*")):
                raise FileExistsError("Existing v7 run must never be overwritten")
            selected = [e for e in manifest["files"] if e["category"] != "historical_reference"]
            for entry in selected:
                target = checked_path(root, entry["path"])
                if target.exists() and (not target.is_file() or sha256(target) != entry["sha256"]):
                    raise FileExistsError("Recovery conflicts with existing evidence: " + entry["path"])
            written = 0
            for entry in selected:
                target = checked_path(root, entry["path"])
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with (stage / entry["path"]).open("rb") as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output); output.flush(); os.fsync(output.fileno())
                    written += 1
                if sha256(target) != entry["sha256"]: raise ValueError("Restored bytes differ")
            config = json.loads((stage / CONFIG).read_text()); _, baseline = checkpoint_record(root / PINS["checkpoint"][0])
            context = fit_context(root)
            for name, record in manifest["checkpoints"].items():
                if checkpoint_record(root / name, config=config, baseline=baseline["model_state_dict"], fit_context=context)[0] != record:
                    raise ValueError("Restored model/optimizer/RNG/batch payload differs")
            requirements = ["original_canonical_repository_path"] if str(root) != manifest["source_root"] else []
            required = {entry["path"]: entry["sha256"] for entry in manifest["files"] if entry["category"] == "historical_reference"}
            identity_path = checked_path(root, PROGRESS + "/identity.json")
            if identity_path.exists():
                pinned = json.loads(identity_path.read_text()); required.update(pinned.get("sources", {}))
                import platform, numpy, torch
                current = {"python": platform.python_version(), "numpy": str(numpy.__version__), "torch": str(torch.__version__)}
                if any(pinned.get("runtime", {}).get(key) != value for key, value in current.items()): requirements.append("pinned_numerical_runtime")
            ancestor = manifest.get("initialization_ancestor")
            if isinstance(ancestor, dict) and "artifact" in ancestor: required[ancestor["artifact"]] = ancestor["artifact_sha256"]
            for name, digest in required.items():
                path = checked_path(root, name)
                if not path.is_file() or sha256(path) != digest: requirements.append(name)
            if requirements:
                (run / "STOP").write_text("Recovery verified; preserve current main. Resolve these pinned resume requirements before execution:\n" + "\n".join(requirements) + "\n")
            return {"restored": True, "files_written": written, "coverage": manifest["coverage"],
                    "checkpoint_payloads_verified": len(manifest["checkpoints"]), "resume_blocked": bool(requirements), "resume_requirements": requirements}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "check", "export", "verify", "restore"))
    parser.add_argument("--root", type=Path, default=ROOT); parser.add_argument("--archive", type=Path)
    parser.add_argument("--sha256", dest="expected_sha256")
    args = parser.parse_args()
    if args.command in {"verify", "restore"} and args.archive is None: parser.error("--archive is required")
    if args.command == "export": result = create_export(args.root)
    elif args.command == "verify": result = verify(args.archive, expected_sha256=args.expected_sha256)
    elif args.command == "restore": result = restore(args.archive, args.root, expected_sha256=args.expected_sha256)
    else:
        result = plan(args.root)
        if args.command == "check": result = {"checked": True, "files": len(result["files"]), "phase": result["phase"], "coverage": result["coverage"]}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__": main()
