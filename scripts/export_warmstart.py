#!/usr/bin/env python3
"""Export a finished warmstart checkpoint without touching its live runner.

Run with the project Python: scripts/export_warmstart.py --iteration 1.
The fixed output is immutable; this tool performs no network operations.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUN = "runs/warmstart-v3"
PROGRESS = "progress/warmstart-v3"
MANIFEST = "SNAPSHOT-MANIFEST.json"
LAUNCH_RUNNER_SHA256 = "ceb84913cdc5e6a323d6ae7b1b5d07f015624cf478c9203a1ad5f46fba8e3762"
CURRENT_RUNNER_SHA256 = "926960741d2a99a43bd9252e6376c7508d507812e220d7d853ddd3cdcf1c2428"
RUNNER_OPERATIONAL_AMENDMENTS = [{
    "id": "all-map-victory-gate-v2",
    "previous_sha256": LAUNCH_RUNNER_SHA256,
    "runner_sha256": CURRENT_RUNNER_SHA256,
    "reason": "User expanded evaluation to all frozen maps; gradient exclusions now come from training_excluded_maps, with legacy gate-v1 fallback.",
    "scope": "Operational gate parsing and status terminology only; learner, engine, inference, model weights, teacher lineage and immutable training configuration are unchanged.",
}]
SOURCE_FILES = ("scripts/run_warmstart_experiment.py", "scripts/export_warmstart.py",
                "scripts/setup_engine.py", "scripts/bootstrap.sh", "scripts/fetch_benchmarks.py",
                "scripts/plot_progress.py", "configs/warmstart-v3.json", "configs/benchmarks.json",
                "configs/victory_gate.json", "configs/archive/victory_gate_v1.json",
                "pyproject.toml", "LICENSE", "THIRD_PARTY.md")
RUN_SUFFIXES = {".json", ".jsonl", ".pt", ".npz", ".log"}
PROGRESS_SUFFIXES = {".json", ".jsonl", ".csv", ".png", ".svg", ".md", ".txt", ".log"}


def payload_allowed(name, prior_directory):
    """Publish learned state and project sources, never downloaded dependencies."""
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or path.as_posix() != name or
            any(part.startswith(".") for part in path.parts)):
        return False
    if name.startswith(("benchmarks/", "build/", "vendor/")):
        return False
    if name in SOURCE_FILES or name == "patches/local-research.patch":
        return True
    for prefix, suffixes in ((RUN, RUN_SUFFIXES), (PROGRESS, PROGRESS_SUFFIXES),
                            ("alpha0", {".py"}), ("java/alpha0", {".java"}),
                            ("maps", {".map26"}),
                            (prior_directory, {".pt", ".json", ".jsonl", ".py", ".md"})):
        if name.startswith(prefix+"/") and path.suffix in suffixes:
            return True
    return False


def dependency_requirements(resume):
    return {
        "payload_policy": "No downloaded benchmark source, compiled classes, engine JARs or JDK files are bundled.",
        "bootstrap_command": "bash scripts/bootstrap.sh",
        "python_environment": "Recreate pinned packages from scripts/bootstrap.sh and recorded Torch/NumPy versions.",
        "engine_and_jdk": "Run scripts/setup_engine.py; its archived source pins URLs and SHA-256 hashes for official engine 1.2.6 and JDK 21.0.13+9, applies the archived patch, and rebuilds project bridge classes.",
        "benchmarks": "Run scripts/fetch_benchmarks.py and python -m alpha0.train compile-benchmarks; configs/benchmarks.json pins each public repository/commit/source hash. Downloaded bot code remains local and is not republished.",
        "verification": "Compare rebuilt engine/class/JDK identities against resume.training_runtime and benchmark identities against preserved evaluation evidence before resuming an existing game plan.",
        "runtime_hashes": {
            "downloaded_files": {name: digest for name, digest in resume["training_runtime"]["files"].items()
                                 if name.startswith(("vendor/", "build/", "benchmarks/"))},
            "compiled_classes": resume["training_runtime"]["compiled_classes"],
            "jdk_release_sha256": resume["training_runtime"]["jdk_release_sha256"],
        },
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def checked_path(root, relative):
    """Require a canonical repo-relative path and reject symlink components."""
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative or not path.parts:
        raise ValueError(f"unsafe archive path: {relative}")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink is not permitted: {relative}")
    return current


def read_json(path):
    return json.loads(path.read_text())


def identity(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


@contextmanager
def runner_lock(root):
    lock = checked_path(root, f"{RUN}/experiment.lock")
    if not lock.parent.is_dir():
        raise ValueError("warmstart run does not exist")
    with lock.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("warmstart runner is active; export only after it finishes") from error
        yield


def validated_resume(root, iteration, runner_sha256):
    import torch

    run = root / RUN
    status = read_json(checked_path(root, f"{RUN}/status.json"))
    if status.get("phase") != "complete_review_required" or status.get("iteration") != iteration:
        raise ValueError("export requires the requested completed, reviewed-later iteration boundary")
    checkpoint = checked_path(root, f"{RUN}/checkpoint-{iteration:04d}.pt")
    latest = checked_path(root, f"{RUN}/latest.pt")
    checkpoint_hash = sha256(checkpoint)
    if sha256(latest) != checkpoint_hash:
        raise ValueError("latest.pt does not match the requested checkpoint")
    payload = torch.load(latest, map_location="cpu", weights_only=True)
    required = {"model_state_dict", "network_config", "optimizer_state_dict", "numpy_rng", "torch_rng", "config",
                "games_total", "positions_total", "wall_seconds", "replay_path"}
    if (payload.get("model_version") != "dual_scale_v2" or payload.get("action_version") != "macro_v2"
            or payload.get("feature_version") != 2 or payload.get("iteration") != iteration or not required <= payload.keys()):
        raise ValueError("checkpoint is missing the complete v2 training/resume state")
    config = read_json(checked_path(root, f"{RUN}/config.json"))
    if config != payload["config"] or config != read_json(checked_path(root, "configs/warmstart-v3.json")):
        raise ValueError("checkpoint/run/launch configurations differ")
    marker = read_json(checked_path(root, f"{RUN}/initialization.json"))
    initial = config["initialization"]
    prior = checked_path(root, initial["artifact"])
    if (sha256(prior) != initial["artifact_sha256"] or marker.get("artifact_sha256") != initial["artifact_sha256"]
            or marker.get("config_sha256") != fingerprint(config)
            or marker.get("source_dataset_sha256") != initial["source_dataset_sha256"]
            or sha256(checked_path(root, f"{RUN}/checkpoint-0000.pt")) != marker.get("checkpoint_zero_sha256")):
        raise ValueError("initial prior/configuration/checkpoint-zero provenance mismatch")
    reports = []
    for number, expected_hash in ((0, marker["checkpoint_zero_sha256"]), (iteration, checkpoint_hash)):
        report = read_json(checked_path(root, f"{PROGRESS}/checkpoint-{number:04d}-smoke.json"))
        if (report.get("checkpoint_sha256") != expected_hash or report.get("iteration") != number
                or report.get("suite") != "smoke" or not report.get("complete")
                or not report.get("full_panel_complete") or report.get("completed") != 24
                or report.get("unresolved_errors") != 0 or len(report.get("opponents", {})) != 12):
            raise ValueError("completed smoke report does not match the published checkpoint")
        reports.append(report)
    if any(reports[0].get(key) != reports[1].get(key) or not reports[0].get(key)
           for key in ("evaluation_config_sha256", "benchmark_suite_id")):
        raise ValueError("baseline/trained smoke report identities differ")
    runner = checked_path(root, "scripts/run_warmstart_experiment.py")
    if sha256(runner) != runner_sha256:
        raise ValueError("runner source differs from the approved operational runner source")
    replay_path = payload["replay_path"]
    if replay_path != f"replay/iteration-{iteration:04d}":
        raise ValueError("checkpoint replay generation differs from its iteration")
    index_path = checked_path(root, f"{RUN}/{replay_path}/index.json")
    index = read_json(index_path)
    if not index.get("games") or "rng_state" not in index:
        raise ValueError("trained checkpoint must contain nonempty replay and replay RNG state")
    seen = set()
    for entry in index["games"]:
        name = entry["file"]
        if not re.fullmatch(r"game-[0-9a-f]{32}\.npz", name) or name in seen:
            raise ValueError("invalid or duplicate active replay filename")
        seen.add(name)
        if sha256(checked_path(root, f"{RUN}/{replay_path}/{name}")) != entry["sha256"]:
            raise ValueError(f"active replay checksum mismatch: {name}")
    if index.get("positions") != sum(entry["positions"] for entry in index["games"]):
        raise ValueError("active replay position counts differ")
    plan_path = checked_path(root, f"{RUN}/plan-{iteration:04d}.json")
    plan = read_json(plan_path)
    if plan.get("config_sha256") != fingerprint(config):
        raise ValueError("completed game plan configuration differs")
    if (plan.get("checkpoint_sha256") != sha256(checked_path(root, f"{RUN}/checkpoint-{iteration-1:04d}.pt"))
            or len(plan.get("jobs", [])) != config["games_per_iteration"]):
        raise ValueError("completed game plan checkpoint or game count differs")
    runtime = plan["training_runtime"]
    for name, expected in {**runtime["files"], **runtime["compiled_classes"]}.items():
        if sha256(checked_path(root, name)) != expected:
            raise ValueError(f"training runtime source/binary changed since launch: {name}")
    release = checked_path(root, "vendor/jdk/release")
    if sha256(release) != runtime["jdk_release_sha256"]:
        raise ValueError("local JDK release identity changed")
    for job in plan["jobs"]:
        game_id = job["game_id"]
        if not re.fullmatch(rf"iter-{iteration:04d}-game-[0-9]+", game_id):
            raise ValueError("completed plan contains an invalid game ID")
        output = checked_path(root, f"{RUN}/games/{game_id}.json")
        if Path(job["output"]) != output:
            raise ValueError("completed plan output is outside the expected run path")
        row = read_json(output)
        if (row.get("status") != "ok" or row.get("game_id") != game_id
                or row.get("identity") != job["identity"] or not re.fullmatch(r"[0-9a-f]{64}", row.get("data_sha256", ""))):
            raise ValueError("completed game transcript does not match its plan")
        name = f"maps/{job['map']}.map26"
        if sha256(checked_path(root, name)) != job["identity"]["map_sha256"]:
            raise ValueError(f"training map changed: {name}")
    return {"iteration": iteration, "checkpoint_sha256": checkpoint_hash, "replay_path": replay_path,
            "replay_index_sha256": sha256(index_path), "replay_games": len(index["games"]),
            "replay_positions": index["positions"], "config_sha256": fingerprint(config),
            "initialization": initial, "original_runner_sha256": LAUNCH_RUNNER_SHA256,
            "exported_runner_sha256": runner_sha256,
            "runner_operational_amendments": RUNNER_OPERATIONAL_AMENDMENTS if runner_sha256 == CURRENT_RUNNER_SHA256 else [],
            "training_runtime": runtime}


def source_files(root, resume):
    files = set()
    prior_directory = str(PurePosixPath(resume["initialization"]["artifact"]).parent)

    def add(path):
        relative = path.relative_to(root).as_posix()
        checked_path(root, relative)
        if not payload_allowed(relative, prior_directory):
            raise ValueError(f"file is outside the publication allowlist: {relative}")
        if not path.is_file():
            raise ValueError(f"required archive input missing: {relative}")
        files.add(path)

    def tree(relative, suffixes):
        directory = checked_path(root, relative)
        for path in sorted(directory.rglob("*")):
            name = path.relative_to(directory)
            if any(part.startswith(".") for part in name.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"archive input contains symlink: {path.relative_to(root)}")
            if path.is_file() and path.suffix in suffixes:
                add(path)

    tree(RUN, RUN_SUFFIXES)
    tree(PROGRESS, PROGRESS_SUFFIXES)
    for name in SOURCE_FILES:
        add(root / name)
    tree("alpha0", {".py"})
    tree("maps", {".map26"})
    for name in resume["training_runtime"]["files"]:
        if not name.startswith(("vendor/", "build/", "benchmarks/")):
            add(root / name)
    tree(prior_directory, {".pt", ".json", ".jsonl", ".py", ".md"})
    return sorted(files)


def compact_selection(root, paths, resume):
    retained, omitted = [], []
    run = root / RUN
    active = run / resume["replay_path"]
    for path in paths:
        reason = None
        if path.is_relative_to(run / "replay") and path.parent != active:
            reason = "obsolete_replay_generation_not_required_to_resume_latest"
        elif path.parent == run / "games" and path.suffix == ".npz":
            match = re.fullmatch(r"iter-([0-9]+)-game-[0-9]+", path.stem)
            if match and int(match[1]) <= resume["iteration"] and path.with_suffix(".json").is_file():
                row = read_json(path.with_suffix(".json"))
                if row.get("status") == "ok" and row.get("game_id") == path.stem:
                    if sha256(path) != row.get("data_sha256"):
                        raise ValueError("consumed game shard checksum differs from its retained transcript")
                    reason = "completed_game_npz_already_consumed_by_published_checkpoint"
        if reason:
            omitted.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size,
                            "sha256": sha256(path), "reason": reason})
        else:
            retained.append(path)
    return retained, omitted


def git_identity(root):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root, text=True)
    return {"commit": commit, "worktree_dirty": bool(dirty.strip())}


def create_export(root=ROOT, iteration=1, *, runner_sha256=CURRENT_RUNNER_SHA256):
    root = Path(root).resolve()
    if type(iteration) is not int or iteration < 1:
        raise ValueError("export iteration must be a positive integer")
    with runner_lock(root):
        resume = validated_resume(root, iteration, runner_sha256)
        output = checked_path(root, f"artifacts/warmstart-v3/iteration-{iteration:03d}")
        archive_path = output / "resume.tar.gz"
        sidecar_path = output / "resume.tar.gz.manifest.json"
        if output.exists():
            raise FileExistsError(f"immutable export destination already exists: {output}")
        all_paths = source_files(root, resume)
        identities = {path: identity(path) for path in all_paths}
        paths, omitted = compact_selection(root, all_paths, resume)
        manifest = {"format_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                    "source_root": str(root), "git": git_identity(root), "resume": resume,
                    "boundary": "finished-warmstart-runner-with-exclusive-lock", "files": [],
                    "omitted_files": omitted,
                    "omission_policy": {"local_files_deleted": False, "resume_latest_complete": True,
                        "historical_models_retained": True, "historical_replay_rollback_available": False},
                    "external_dependencies": dependency_requirements(resume)}
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".warmstart-export-", dir=output.parent))
        try:
            temporary_archive = temporary / archive_path.name
            with tarfile.open(temporary_archive, "w:gz", compresslevel=3) as archive:
                for path in paths:
                    name = path.relative_to(root).as_posix()
                    digest = sha256(path)
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = identities[path][2], path.stat().st_mode & 0o777, 0
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
                    if identity(path) != identities[path] or sha256(path) != digest:
                        raise RuntimeError(f"source changed during export: {name}")
                    manifest["files"].append({"path": name, "size": info.size, "sha256": digest})
                if source_files(root, resume) != all_paths or any(identity(p) != identities[p] for p in all_paths):
                    raise RuntimeError("warmstart files changed during export")
                for item in omitted:
                    if sha256(root / item["path"]) != item["sha256"]:
                        raise RuntimeError("omitted source changed during export")
                # Revalidate identity-bound inputs after collecting the complete archive.
                if validated_resume(root, iteration, runner_sha256) != resume:
                    raise RuntimeError("published resume identity changed during export")
                readme = (f"# Warmstart iteration {iteration} resume\n\n"
                    "Verify resume.tar.gz against its sidecar SHA-256 and verify extracted file hashes against SNAPSHOT-MANIFEST.json.\n\n"
                    f"Restore into a clean checkout at the original source root `{root}`. Archive paths are repository-relative. "
                    "Plans and evidence retain absolute original paths; this exporter does not rebase them. "
                    "Run the archived hash-pinned bootstrap/setup/fetch scripts to provision Python/JDK/engine and rebuild bridge/benchmark classes; verify recorded runtime hashes. "
                    "Downloaded benchmark source, compiled classes, engine JARs and JDK files are not bundled. "
                    "The archive contains the prior dependency, cp0, all model checkpoints, latest optimizer/RNG state, "
                    "active replay including its RNG, plans, raw game JSON, logs and evaluation evidence. "
                    "Older replay generations and consumed game NPZ files are omitted with hashes; historical model evaluation remains available, "
                    "but historical replay rollback is not included. No local files were pruned.\n\n"
                    f"Resume explicitly with `.venv/bin/python scripts/run_warmstart_experiment.py --target-iteration {iteration+1} --hours 2`. "
                    "No STOP, PID, lock or temporary file is restored. Resuming training does not promote the policy.\n")
                for name, encoded in (("RESUME.md", readme.encode()),
                        (MANIFEST, (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+"\n").encode())):
                    info = tarfile.TarInfo(name)
                    info.size, info.mode = len(encoded), 0o644
                    archive.addfile(info, io.BytesIO(encoded))
            with temporary_archive.open("rb") as stream:
                os.fsync(stream.fileno())
            sidecar = {**manifest, "archive": archive_path.name, "archive_sha256": sha256(temporary_archive),
                       "archive_size": temporary_archive.stat().st_size}
            with (temporary / sidecar_path.name).open("w") as stream:
                json.dump(sidecar, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.rename(output)
            return {"archive": str(archive_path), "manifest": str(sidecar_path),
                    "sha256": sidecar["archive_sha256"], "size": sidecar["archive_size"],
                    "files": len(paths), "omitted_files": len(omitted), "iteration": iteration}
        finally:
            if temporary.exists():
                for path in temporary.iterdir():
                    path.unlink()
                temporary.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--iteration", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(create_export(args.root, args.iteration), indent=2))


if __name__ == "__main__":
    main()
