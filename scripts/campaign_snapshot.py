#!/usr/bin/env python3
"""Create, publish, download and safely restore durable campaign checkpoints.

No network operation occurs without the explicit publish/download subcommands.
Call create --checkpoint-boundary only while the campaign's workers are quiescent.
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
import shutil
import subprocess
import tarfile
import tempfile

REPO = "terryvanbelle/battlecode26-alpha0"
TAG = "campaign-state"
GIT_BRANCH = "campaign-checkpoints"
MAX_GIT_ASSET_BYTES = 90 * 1024**2
RUN = "runs/campaign/v2"
PROGRESS = "progress/campaign-v2"
STATUS = "runs/campaign/status.json"
POINTER = f"{PROGRESS}/snapshot.json"
MANIFEST = "SNAPSHOT-MANIFEST.json"
FORMAT_VERSION = 1
MAX_EXPANDED_BYTES = 12 * 1024**3
MAX_ASSET_BYTES = 1900 * 1024**2  # Below GitHub's 2 GiB release asset ceiling.
ASSET_RE = re.compile(r"campaign-state-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{12}\.tar\.gz\Z")
SECRET_RE = re.compile(r"(^|[._-])(secret|secrets|password|credentials?|token|tokens|private[-_]key)([._-]|$)", re.I)
RUN_SUFFIXES = {".json", ".jsonl", ".npz", ".npy", ".pt"}
PROGRESS_SUFFIXES = {".json", ".jsonl", ".csv", ".png", ".svg", ".md", ".txt"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".snapshot-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def allowed_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        return False
    if any(part.startswith(".") or SECRET_RE.search(part) for part in path.parts):
        return False
    if name == POINTER:
        return False  # Pointer describes this archive; including it would be circular.
    if name == STATUS:
        return True
    if name.startswith(f"{RUN}/"):
        return path.suffix in RUN_SUFFIXES
    if name.startswith(f"{PROGRESS}/"):
        return path.suffix in PROGRESS_SUFFIXES
    return len(path.parts) == 3 and path.parts[:2] == ("runs", "gate-ledgers") and path.suffix == ".json"


def assert_no_symlink(root, path):
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink is not permitted in snapshot paths: {relative}")


@contextmanager
def checkpoint_lock(root, at_boundary=False):
    directory = root / "runs/campaign"
    directory.mkdir(parents=True, exist_ok=True)
    assert_no_symlink(root, directory)
    assert_no_symlink(root, directory / "snapshot.lock")
    assert_no_symlink(root, directory / "campaign.lock")
    with (directory / "snapshot.lock").open("a") as snapshot_lock:
        fcntl.flock(snapshot_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (directory / "campaign.lock").open("a") as campaign_lock:
            if not at_boundary:
                try:
                    fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError("campaign is active; only its quiescent checkpoint boundary may create a snapshot") from exc
            yield


def source_files(root):
    paths = []
    for relative in (RUN, PROGRESS, "runs/gate-ledgers"):
        directory = root / relative
        if directory.exists():
            assert_no_symlink(root, directory)
            for path in sorted(directory.rglob("*")):
                if path.is_symlink():
                    raise ValueError(f"snapshot source contains symlink: {path.relative_to(root)}")
                if path.is_file() and allowed_path(path.relative_to(root).as_posix()):
                    paths.append(path)
    status = root / STATUS
    if status.exists():
        assert_no_symlink(root, status)
        paths.append(status)
    return sorted(paths)


def file_identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def git_identity(root):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root,
                           capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": commit, "worktree_dirty": bool(dirty)}


def resume_reference(root):
    """Resolve the published checkpoint's replay generation, without unpickling code."""
    import torch

    root = Path(root)
    run = root / RUN
    checkpoint = run / "latest.pt"
    assert_no_symlink(root, checkpoint)
    if not checkpoint.is_file():
        raise ValueError("a complete v2 latest.pt checkpoint is required")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("model_version") != "dual_scale_v2":
        raise ValueError("snapshot requires a dual_scale_v2 checkpoint")
    relative = payload.get("replay_path", "replay")
    if not isinstance(relative, str):
        raise ValueError("checkpoint replay_path must be a relative path")
    replay = PurePosixPath(relative)
    if (replay.is_absolute() or replay.as_posix() != relative or ".." in replay.parts
            or not replay.parts or replay.parts[0] != "replay"):
        raise ValueError("checkpoint replay_path must stay inside the run's replay directory")
    index_path = run / relative / "index.json"
    assert_no_symlink(root, index_path)
    if not index_path.is_file():
        raise ValueError(f"checkpoint references missing replay index: {relative}/index.json")
    index = json.loads(index_path.read_text())
    for entry in index["games"]:
        filename = entry["file"]
        if not isinstance(filename, str) or not re.fullmatch(r"game-[0-9a-f]{32}\.npz", filename):
            raise ValueError("invalid replay game filename")
        game_path = index_path.parent / filename
        assert_no_symlink(root, game_path)
        if not game_path.is_file() or sha256(game_path) != entry["sha256"]:
            raise ValueError(f"checkpoint references missing or corrupt replay game: {filename}")
    return {"checkpoint_sha256": sha256(checkpoint), "iteration": payload["iteration"],
            "replay_path": relative, "replay_index_sha256": sha256(index_path)}


def create_snapshot(root, output_dir=None, at_boundary=False):
    root = Path(root).resolve()
    output = Path(output_dir).resolve() if output_dir else root / "runs/campaign/snapshots"
    with checkpoint_lock(root, at_boundary):
        resume = resume_reference(root)
        output.mkdir(parents=True, exist_ok=True)
        paths = source_files(root)
        identities = {p: file_identity(p) for p in paths}
        if sum(identity[2] for identity in identities.values()) > MAX_EXPANDED_BYTES:
            raise ValueError("snapshot exceeds expanded-size budget")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        manifest = {"format_version": FORMAT_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
                    "source_root": str(root), "repository": REPO, "git": git_identity(root),
                    "boundary": "quiescent-campaign-checkpoint", "resume": resume, "files": []}
        fd, temporary = tempfile.mkstemp(prefix=".campaign-snapshot-", suffix=".tar.gz", dir=output)
        os.close(fd)
        try:
            with tarfile.open(temporary, "w:gz", compresslevel=3) as archive:
                for path in paths:
                    name = path.relative_to(root).as_posix()
                    # Hash before and after archiving to reject a changing source.
                    digest = sha256(path)
                    info = tarfile.TarInfo(name)
                    info.size = identities[path][2]; info.mode = 0o600; info.mtime = 0
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
                    if file_identity(path) != identities[path] or sha256(path) != digest:
                        raise RuntimeError(f"source changed during snapshot: {name}")
                    manifest["files"].append({"path": name, "size": info.size, "sha256": digest})
                if source_files(root) != paths or any(file_identity(p) != identities[p] for p in paths):
                    raise RuntimeError("campaign artifacts changed during snapshot; retry at a checkpoint boundary")
                encoded = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
                info = tarfile.TarInfo(MANIFEST); info.size = len(encoded); info.mode = 0o600
                archive.addfile(info, io.BytesIO(encoded))
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            size = Path(temporary).stat().st_size
            if size > MAX_ASSET_BYTES:
                raise ValueError("compressed snapshot exceeds release asset budget")
            digest = sha256(temporary)
            destination = output / f"campaign-state-{timestamp}-{digest[:12]}.tar.gz"
            os.replace(temporary, destination)
            sidecar = {"format_version": FORMAT_VERSION, "archive": destination.name, "sha256": digest,
                       "size": size, "created_at": manifest["created_at"], "git": manifest["git"],
                       "file_count": len(manifest["files"]), "repository": REPO, "release": TAG}
            atomic_json(destination.with_name(destination.name + ".manifest.json"), sidecar)
            return {**sidecar, "archive": str(destination), "archive_name": destination.name,
                    "manifest": str(destination) + ".manifest.json"}
        finally:
            Path(temporary).unlink(missing_ok=True)


def gh(*args, capture=True):
    result = subprocess.run(["gh", *map(str, args)], check=True, text=True,
                            stdout=subprocess.PIPE if capture else None, stderr=subprocess.PIPE if capture else None)
    return result.stdout if capture else ""


def release_info(repo, tag):
    return json.loads(gh("api", f"repos/{repo}/releases/tags/{tag}"))


def validate_sidecar(archive, sidecar):
    if (sidecar.get("format_version") != FORMAT_VERSION or sidecar.get("archive") != archive.name
            or not ASSET_RE.fullmatch(archive.name) or archive.stat().st_size != sidecar.get("size")
            or sha256(archive) != sidecar.get("sha256")
            or not archive.name.endswith(str(sidecar.get("sha256", ""))[:12] + ".tar.gz")):
        raise ValueError("snapshot archive checksum/size/identity does not match its manifest")


def git_command(directory, *args, check=True, stdout=None):
    return subprocess.run(["git", *map(str, args)], cwd=directory, check=check, text=stdout is None,
                          stdout=subprocess.PIPE if stdout is None else stdout, stderr=subprocess.PIPE)


def remote_url(repo):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Git snapshot storage requires an owner/repository name")
    return f"https://github.com/{repo}.git"


@contextmanager
def isolated_git(repo, commit=None):
    with tempfile.TemporaryDirectory(prefix="bc26-state-git-") as temporary:
        directory = Path(temporary)
        git_command(directory, "init", "-q")
        git_command(directory, "remote", "add", "origin", remote_url(repo))
        if commit:
            git_command(directory, "fetch", "--quiet", "--filter=blob:none", "--depth=1", "--no-tags", "origin", commit)
        yield directory


def git_latest_pointer(repo):
    with isolated_git(repo) as directory:
        result = git_command(directory, "ls-remote", "--heads", "origin", f"refs/heads/{GIT_BRANCH}")
        if not result.stdout.strip():
            return None
        commit = result.stdout.split()[0]
        git_command(directory, "fetch", "--quiet", "--filter=blob:none", "--depth=1", "--no-tags", "origin", commit)
        pointer = json.loads(git_command(directory, "show", f"{commit}:latest.json").stdout)
        if pointer.get("storage") != "git" or pointer.get("branch") != GIT_BRANCH:
            raise ValueError("invalid Git snapshot latest pointer")
        return {**pointer, "commit": commit}


def publish_git_snapshot(root, archive, sidecar, repo, tag, auth_error):
    """Fast-forward-only isolated Git fallback for unavailable release upload auth."""
    if archive.stat().st_size > MAX_GIT_ASSET_BYTES:
        raise RuntimeError("Release upload authentication failed and archive exceeds the 90 MiB Git fallback limit; "
                           "durability requires chunked storage or corrected release-upload credentials")
    with isolated_git(repo) as directory:
        heads = git_command(directory, "ls-remote", "--heads", "origin", f"refs/heads/{GIT_BRANCH}").stdout.strip()
        previous = None
        git_command(directory, "symbolic-ref", "HEAD", f"refs/heads/{GIT_BRANCH}")
        if heads:
            parent = heads.split()[0]
            git_command(directory, "fetch", "--quiet", "--filter=blob:none", "--depth=1", "--no-tags", "origin", parent)
            git_command(directory, "update-ref", f"refs/heads/{GIT_BRANCH}", parent)
            git_command(directory, "read-tree", parent)
            previous = json.loads(git_command(directory, "show", f"{parent}:latest.json").stdout)
        archive_path = f"snapshots/{archive.name}"
        manifest_path = archive_path + ".manifest.json"
        changed = []
        for source, relative in ((archive, archive_path), (Path(str(archive)+".manifest.json"), manifest_path)):
            destination = directory/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            existing = git_command(directory, "cat-file", "-e", f"HEAD:{relative}", check=False) if heads else None
            if existing is not None and existing.returncode == 0:
                with destination.open("wb") as stream:
                    git_command(directory, "show", f"HEAD:{relative}", stdout=stream)
                if sha256(destination) != sha256(source):
                    raise ValueError(f"immutable Git snapshot asset differs: {relative}")
            else:
                shutil.copyfile(source, destination)
                git_command(directory, "add", "--", relative)
                changed.append(relative)
        pointer = {**sidecar, "repository": repo, "release": tag, "storage": "git", "branch": GIT_BRANCH,
                   "path": archive_path, "manifest_path": manifest_path,
                   "published_at": datetime.now(timezone.utc).isoformat(),
                   "release_upload_auth_error": auth_error}
        # Republishing an older snapshot must not move latest backwards.
        if previous and previous["created_at"] > pointer["created_at"]:
            pointer = previous
        atomic_json(directory/"latest.json", pointer)
        git_command(directory, "add", "--", "latest.json")
        tree_changed = not heads or git_command(directory, "diff", "--cached", "--quiet", check=False).returncode != 0
        if tree_changed:
            git_command(directory, "-c", "user.name=Battlecode Alpha0", "-c", "user.email=alpha0@users.noreply.github.com",
                        "commit", "--quiet", "-m", f"Preserve campaign snapshot {sidecar['sha256'][:12]}")
            # No force flag: concurrent publication fails rather than rewriting history.
            git_command(directory, "push", "--quiet", "origin", f"HEAD:refs/heads/{GIT_BRANCH}")
        commit = git_command(directory, "rev-parse", "HEAD").stdout.strip()
        pointer = {**pointer, "commit": commit,
                   "branch_url": f"https://github.com/{repo}/tree/{GIT_BRANCH}"}
        atomic_json(Path(root)/POINTER, pointer)
        return pointer


def download_git_snapshot(output, pointer):
    repo, commit = pointer["repository"], pointer["commit"]
    name = pointer["archive"]
    if (not re.fullmatch(r"[0-9a-f]{40}", commit) or not ASSET_RE.fullmatch(name)
            or pointer.get("branch") != GIT_BRANCH or pointer.get("path") != f"snapshots/{name}"
            or pointer.get("manifest_path") != f"snapshots/{name}.manifest.json"):
        raise ValueError("invalid exact Git snapshot reference")
    with isolated_git(repo, commit) as directory:
        for field, filename in (("manifest_path", name+".manifest.json"), ("path", name)):
            with (directory/filename).open("wb") as stream:
                git_command(directory, "show", f"{commit}:{pointer[field]}", stdout=stream)
        sidecar = json.loads((directory/(name+".manifest.json")).read_text())
        validate_sidecar(directory/name, sidecar)
        if sidecar["sha256"] != pointer["sha256"]:
            raise ValueError("Git snapshot differs from its committed pointer")
        for filename in (name, name+".manifest.json"):
            destination = output/filename
            if destination.exists():
                if sha256(destination) != sha256(directory/filename):
                    raise ValueError("existing local Git download differs; refusing to overwrite")
            else:
                # Different temp filesystem locations may not support rename.
                fd, temporary = tempfile.mkstemp(prefix=".git-download-", dir=output)
                os.close(fd)
                try:
                    shutil.copyfile(directory/filename, temporary)
                    os.replace(temporary, destination)
                finally:
                    Path(temporary).unlink(missing_ok=True)
    return {**pointer, "archive": str(output/name), "archive_name": name,
            "manifest": str(output/(name+".manifest.json"))}


def publish_snapshot(root, archive, repo=REPO, tag=TAG):
    root, archive = Path(root).resolve(), Path(archive).resolve()
    sidecar_path = archive.with_name(archive.name + ".manifest.json")
    sidecar = json.loads(sidecar_path.read_text())
    validate_sidecar(archive, sidecar)
    try:
        release = release_info(repo, tag)
    except subprocess.CalledProcessError as error:
        # Only create on an actual 404; authentication/network failures must not
        # be confused with an absent release.
        if "404" not in (error.stderr or ""):
            raise
        notes = "Resumable local training state. Immutable archives include per-file checksums and no credentials."
        gh("release", "create", tag, "--repo", repo, "--title", "Campaign training state", "--notes", notes,
           "--target", sidecar["git"]["commit"], "--prerelease")
        release = release_info(repo, tag)
    assets = {asset["name"]: asset for asset in release["assets"] if asset.get("state") == "uploaded"}
    # Publish the archive first, manifest last. Downloaders consider only complete pairs.
    for path in (archive, sidecar_path):
        if path.name in assets:
            with tempfile.TemporaryDirectory(prefix="bc26-existing-asset-") as directory:
                gh("release", "download", tag, "--repo", repo, "--pattern", path.name, "--dir", directory)
                if sha256(Path(directory)/path.name) != sha256(path):
                    raise ValueError(f"immutable remote asset differs: {path.name}")
        else:
            try:
                gh("release", "upload", tag, path, "--repo", repo)
            except subprocess.CalledProcessError as error:
                details = error.stderr or ""
                if not re.search(r"HTTP (?:401|403)|Bad credentials|Resource not accessible by integration", details, re.I):
                    raise
                # This fallback changes storage, never credentials or the source branch.
                return publish_git_snapshot(root, archive, sidecar, repo, tag,
                                            "GitHub release upload rejected configured credentials")
    verified = release_info(repo, tag)
    names = {asset["name"] for asset in verified["assets"] if asset.get("state") == "uploaded"}
    if not {archive.name, sidecar_path.name}.issubset(names):
        raise RuntimeError("release upload did not publish the complete snapshot pair")
    pointer = {**sidecar, "repository": repo, "release": tag, "storage": "release",
               "release_url": verified["html_url"], "published_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(root / POINTER, pointer)
    return pointer


def download_snapshot(root, output_dir=None, repo=REPO, tag=TAG, latest=False):
    root = Path(root).resolve()
    output = Path(output_dir).resolve() if output_dir else root / "runs/campaign/downloads"
    output.mkdir(parents=True, exist_ok=True)
    pointer = root / POINTER
    if pointer.exists() and not latest:
        expected = json.loads(pointer.read_text())
        if expected.get("repository") != repo or expected.get("release") != tag:
            raise ValueError("committed snapshot pointer selects another repository/release")
        if expected.get("storage") == "git":
            return download_git_snapshot(output, expected)
        name = expected["archive"]
        if not ASSET_RE.fullmatch(name):
            raise ValueError("invalid archive name in snapshot pointer")
    else:
        try:
            release = release_info(repo, tag)
            assets = {a["name"] for a in release["assets"] if a.get("state") == "uploaded"}
        except subprocess.CalledProcessError as error:
            if "404" not in (error.stderr or ""):
                raise
            assets = set()
        complete = sorted(name for name in assets if ASSET_RE.fullmatch(name) and name + ".manifest.json" in assets)
        git_pointer = git_latest_pointer(repo)
        # Archive timestamps share one UTC lexicographic format across storage backends.
        if git_pointer and (not complete or git_pointer["archive"] > complete[-1]):
            return download_git_snapshot(output, git_pointer)
        if not complete:
            raise ValueError("no complete published release or Git snapshot exists")
        name, expected = complete[-1], None
    with tempfile.TemporaryDirectory(prefix=".download-", dir=output) as temporary:
        directory = Path(temporary)
        for asset in (name + ".manifest.json", name):
            gh("release", "download", tag, "--repo", repo, "--pattern", asset, "--dir", directory)
        sidecar = json.loads((directory/(name+".manifest.json")).read_text())
        validate_sidecar(directory/name, sidecar)
        if expected and expected["sha256"] != sidecar["sha256"]:
            raise ValueError("downloaded snapshot differs from committed Git pointer")
        for asset in (name, name+".manifest.json"):
            destination = output/asset
            if destination.exists():
                if sha256(destination) != sha256(directory/asset):
                    raise ValueError("existing local download differs; refusing to overwrite")
            else:
                os.replace(directory/asset, destination)
    return {**sidecar, "archive": str(output/name), "archive_name": name,
            "manifest": str(output/(name+".manifest.json"))}


_PATH_KEYS = {"checkpoint", "opponent", "output", "stop_file", "checkpoint_path",
              "maps_dir", "opponent_checkpoint", "cell_artifact_dir", "holdout_ledger"}
_PROTECTED_KEYS = {"identity", "config", "evaluation_config", "runtime", "network_config"}


def rebase_json(value, old_root, new_root):
    """Rebase operational locations, never hash-bound identities/configuration.

    Learner job identities bind checkpoint bytes and config hashes rather than
    operational filenames. Evaluation identities also exclude checkpoint_path
    and maps_dir. Keep their complete identity/config subtrees byte-equivalent.
    """
    if isinstance(value, list):
        return [rebase_json(item, old_root, new_root) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in _PROTECTED_KEYS:
                result[key] = item
            elif key in _PATH_KEYS and isinstance(item, str) and item.startswith(old_root + "/"):
                result[key] = new_root + item[len(old_root):]
            else:
                result[key] = rebase_json(item, old_root, new_root)
        return result
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical(row) + "\n" for row in rows))


def merge_training(existing, incoming):
    by_iteration = {}
    for row in [*existing, *incoming]:
        iteration = row.get("iteration")
        if type(iteration) is not int or iteration < 0:
            raise ValueError("training evidence has no valid iteration")
        if iteration in by_iteration and by_iteration[iteration] != row:
            raise ValueError(f"conflicting training evidence for iteration {iteration}")
        by_iteration[iteration] = row
    return [by_iteration[key] for key in sorted(by_iteration)]


def merge_report(existing, incoming):
    """Choose a same-identity report only when its completed-cell set dominates."""
    if existing == incoming:
        return existing
    identity_keys = ("checkpoint_sha256", "evaluation_config_sha256", "benchmark_suite_id",
                     "suite", "stage", "iteration", "expected", "expected_cell_ids")
    if any(key not in existing or key not in incoming or existing[key] != incoming[key] for key in identity_keys):
        raise ValueError("conflicting evaluation report identities/schedules")
    expected = set(existing["expected_cell_ids"])
    def completed(report):
        pending = set(report["pending_cell_ids"])
        if (not pending <= expected or report["completed"] != len(expected - pending)
                or report["expected"] != len(expected)):
            raise ValueError("evaluation report completion counts are inconsistent")
        return expected - pending
    before, after = completed(existing), completed(incoming)
    if not (before <= after or after <= before):
        raise ValueError("evaluation reports have incomparable completion sets")
    older, newer = (existing, incoming) if before <= after else (incoming, existing)
    for opponent, stats in older.get("opponents", {}).items():
        later = newer.get("opponents", {}).get(opponent)
        if later is None:
            raise ValueError("evaluation report lost an opponent")
        for key in ("completed", "wins", "draws", "losses"):
            if key in stats and (key not in later or later[key] < stats[key]):
                raise ValueError("evaluation report changed previously completed outcomes")
        if before == after and ({k: v for k, v in stats.items() if k != "errors"}
                                    != {k: v for k, v in later.items() if k != "errors"}):
            raise ValueError("evaluation reports disagree on the same completed cells")
    if before == after:
        for key in ("complete", "full_panel_complete", "passed", "victory_gate_passed", "standard_stage_passed"):
            if older.get(key) != newer.get(key):
                raise ValueError("evaluation reports disagree on completion/pass result")
    return newer


def merge_evaluation_rows(progress, source_rows, destination_progress):
    """Rebuild summary ledgers from all known immutable attempts and summaries."""
    # Reuse the evaluator's result-selection semantics without scheduling games.
    import sys
    code_root = str(Path(__file__).resolve().parents[1])
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    from alpha0.evaluate import selected_result

    attempts, legacy = {}, {}
    def add(row):
        cell = row.get("evaluation_cell_id")
        if cell is None:
            legacy[canonical(row)] = row
            return
        identity, attempt = row.get("identity"), row.get("attempt")
        if (not isinstance(identity, dict) or type(attempt) is not int or attempt < 1
                or hashlib.sha256(canonical(identity).encode()).hexdigest() != cell):
            raise ValueError("evaluation evidence has an invalid cell identity/attempt")
        key = cell, attempt
        row = dict(row)
        row.pop("cell_artifact_dir", None)  # Derived location, not game evidence.
        previous = attempts.get(key)
        if previous is not None and previous != row:
            a, b = dict(previous), dict(row)
            error_a, error_b = a.pop("error", None), b.pop("error", None)
            # evaluations.jsonl deliberately truncates error summaries; preserve
            # the original full error from evaluation_errors/cell artifacts.
            errors_agree = error_a == error_b or (isinstance(error_a, str) and isinstance(error_b, str) and
                (error_a == error_b.splitlines()[0][:600] or error_b == error_a.splitlines()[0][:600]))
            if a != b or not errors_agree:
                raise ValueError(f"conflicting evaluation evidence for cell {cell}, attempt {attempt}")
            if len(str(error_a or "")) > len(str(error_b or "")):
                row = previous
        attempts[key] = row
    for row in source_rows:
        add(row)
    requests = {}
    for request_path in sorted((progress/"evaluation_cells").glob("*/request.json")):
        request = json.loads(request_path.read_text())
        cell = request.get("evaluation_cell_id")
        if cell != request_path.parent.name or hashlib.sha256(canonical(request["identity"]).encode()).hexdigest() != cell:
            raise ValueError("evaluation request path disagrees with its immutable identity")
        requests[cell] = request
    for attempt_path in sorted((progress/"evaluation_cells").glob("*/attempt-*.json")):
        row = json.loads(attempt_path.read_text())
        if (row.get("evaluation_cell_id") != attempt_path.parent.name or
                attempt_path.name != f"attempt-{int(row.get('attempt', 0)):04d}.json"):
            raise ValueError("evaluation attempt path disagrees with its immutable identity")
        if row["evaluation_cell_id"] in requests and row.get("identity") != requests[row["evaluation_cell_id"]]["identity"]:
            raise ValueError("evaluation attempt differs from its request identity")
        add(row)
    groups = {}
    for (cell, _), row in sorted(attempts.items()):
        groups.setdefault(cell, []).append(row)
    rows = list(legacy.values())
    for cell, candidates in sorted(groups.items()):
        chosen = selected_result({"evaluation_cell_id": cell, "identity": candidates[0]["identity"]}, candidates)
        row = dict(chosen)
        if (progress/"evaluation_cells"/cell).exists():
            row["cell_artifact_dir"] = str(destination_progress/"evaluation_cells"/cell)
        if row.get("error"):
            row["error"] = row["error"].splitlines()[0][:600]
        rows.append(row)
    write_jsonl(progress/"evaluations.jsonl", rows)
    write_jsonl(progress/"evaluation_errors.jsonl", [row for _, row in sorted(attempts.items()) if row.get("status") == "error"])
    return len(attempts)


def reconcile_progress(root, staging, old_root):
    """Merge Git and snapshot evidence in staging before modifying the checkout."""
    existing_dir, merged_dir = root/PROGRESS, staging/PROGRESS
    merged_dir.mkdir(parents=True, exist_ok=True)
    eval_names = {"evaluations.jsonl", "evaluation_errors.jsonl"}
    eval_rows = []
    for name in eval_names:
        eval_rows.extend(read_jsonl(merged_dir/name))
    changed, existing_files = [], []
    if existing_dir.exists():
        for path in sorted(existing_dir.rglob("*")):
            if path.is_symlink():
                raise ValueError("existing progress contains a symlink")
            if not path.is_file() or not allowed_path(path.relative_to(root).as_posix()):
                continue
            existing_files.append(path)
            relative = path.relative_to(existing_dir)
            destination = merged_dir/relative
            old = None
            if path.suffix == ".json":
                old = rebase_json(json.loads(path.read_text()), old_root, str(root))
            elif path.suffix == ".jsonl":
                old = rebase_json(read_jsonl(path), old_root, str(root))
            if relative.as_posix() in eval_names:
                eval_rows.extend(old)
                continue
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                if old is None:
                    shutil.copyfile(path, destination)
                elif path.suffix == ".jsonl":
                    write_jsonl(destination, old)
                else:
                    destination.write_text(json.dumps(old, sort_keys=True, indent=2, allow_nan=False)+"\n")
                changed.append(str(relative))
                continue
            cosmetic = (len(relative.parts) == 1 and (relative.name in ("README.md", "status.json")
                        or relative.name.startswith("latest-") and path.suffix == ".json")) or path.suffix in (".png", ".svg")
            if cosmetic:
                continue  # Historical presentation may use archive version; raw evidence is merged below.
            if relative.as_posix() == "training.jsonl":
                write_jsonl(destination, merge_training(old, read_jsonl(destination)))
                changed.append(str(relative))
            elif relative.parts[0] == "evaluation_reports" and path.suffix == ".json":
                merged = merge_report(old, json.loads(destination.read_text()))
                destination.write_text(json.dumps(merged, sort_keys=True, indent=2, allow_nan=False)+"\n")
                changed.append(str(relative))
            elif old is not None:
                incoming = json.loads(destination.read_text()) if path.suffix == ".json" else read_jsonl(destination)
                if old != incoming:
                    raise ValueError(f"conflicting immutable progress evidence: {relative}")
            elif path.read_bytes() != destination.read_bytes():
                raise ValueError(f"conflicting progress evidence: {relative}")
    # Normalize/validate even a training ledger that only existed on one side.
    training = merged_dir/"training.jsonl"
    if training.exists():
        write_jsonl(training, merge_training([], read_jsonl(training)))
    attempt_count = merge_evaluation_rows(merged_dir, eval_rows, root/PROGRESS)
    return {"merged_paths": changed, "evaluation_attempts": attempt_count,
            "max_training_iteration": max((row["iteration"] for row in read_jsonl(training)), default=-1)}


def restore_snapshot(root, archive):
    root, archive = Path(root).resolve(), Path(archive).resolve()
    sidecar = json.loads(archive.with_name(archive.name + ".manifest.json").read_text())
    validate_sidecar(archive, sidecar)
    with checkpoint_lock(root):
        run = root/RUN
        assert_no_symlink(root, run)
        if run.exists() and any(run.rglob("*")):
            raise ValueError("local v2 run already contains work; refusing to overwrite newer or active state")
        with tempfile.TemporaryDirectory(prefix=".restore-", dir=root/"runs/campaign") as temporary:
            staging = Path(temporary)
            with tarfile.open(archive, "r:gz") as source:
                members = source.getmembers()
                if len({member.name for member in members}) != len(members):
                    raise ValueError("archive contains duplicate paths")
                if sum(member.size for member in members) > MAX_EXPANDED_BYTES:
                    raise ValueError("archive exceeds expanded-size budget")
                manifest_member = next((m for m in members if m.name == MANIFEST), None)
                if manifest_member is None or not manifest_member.isfile() or manifest_member.size > 16*1024**2:
                    raise ValueError("snapshot has no valid checksum manifest")
                manifest = json.load(source.extractfile(manifest_member))
                if manifest.get("format_version") != FORMAT_VERSION:
                    raise ValueError("unsupported snapshot format")
                entries = manifest["files"]
                expected = {entry["path"]: entry for entry in entries}
                if len(expected) != len(entries) or set(expected) != {m.name for m in members if m.name != MANIFEST}:
                    raise ValueError("archive membership does not match checksum manifest")
                for member in members:
                    if member.name == MANIFEST:
                        continue
                    if not member.isfile() or not allowed_path(member.name):
                        raise ValueError(f"unsafe or unapproved archive path: {member.name}")
                    entry = expected[member.name]
                    if member.size != entry["size"]:
                        raise ValueError("member size differs from manifest")
                    destination = staging/member.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as reader, destination.open("xb") as writer:
                        shutil.copyfileobj(reader, writer, length=1024*1024)
                    if sha256(destination) != entry["sha256"]:
                        raise ValueError(f"member checksum mismatch: {member.name}")
            resume = resume_reference(staging)
            if manifest.get("resume") is not None and resume != manifest["resume"]:
                raise ValueError("published checkpoint/replay reference differs from snapshot manifest")
            old_root = manifest["source_root"]
            if not isinstance(old_root, str) or not Path(old_root).is_absolute():
                raise ValueError("invalid original repository root")
            rebased = []
            if old_root != str(root):
                for name in expected:
                    path = staging/name
                    if path.suffix == ".json":
                        original = json.loads(path.read_text())
                        updated = rebase_json(original, old_root, str(root))
                        if updated != original:
                            path.write_text(json.dumps(updated, sort_keys=True, indent=2, allow_nan=False)+"\n")
                            rebased.append(name)
                    elif path.suffix == ".jsonl":
                        original = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                        updated = rebase_json(original, old_root, str(root))
                        if updated != original:
                            path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False)+"\n" for row in updated))
                            rebased.append(name)
            reconciliation = reconcile_progress(root, staging, old_root)
            # Progress is reconciled, presentation/status is historical, and gate
            # ledgers remain strict. Capture every destination's current hash so
            # a concurrent writer cannot be silently overwritten during publish.
            publish_paths = sorted(p for p in staging.rglob("*") if p.is_file())
            observed = {}
            for source_path in publish_paths:
                name = source_path.relative_to(staging).as_posix()
                destination = root/name
                assert_no_symlink(root, destination)
                if destination.exists() and not destination.is_file():
                    raise ValueError(f"existing artifact is not a regular file: {name}")
                observed[name] = sha256(destination) if destination.exists() else None
                if name.startswith("runs/gate-ledgers/") and destination.exists():
                    if json.loads(destination.read_text()) != json.loads(source_path.read_text()):
                        raise ValueError(f"existing artifact differs; refusing to overwrite: {name}")
            # Publish merged evidence first. An interruption leaves a reusable
            # union; the complete run directory is published in one rename last.
            for source_path in publish_paths:
                name = source_path.relative_to(staging).as_posix()
                if name.startswith(RUN + "/"):
                    continue
                destination = root/name
                destination.parent.mkdir(parents=True, exist_ok=True)
                actual = sha256(destination) if destination.exists() else None
                if actual != observed[name]:
                    raise RuntimeError(f"destination changed during restore: {name}")
                if actual == sha256(source_path):
                    continue
                if actual is None:
                    os.link(source_path, destination)  # Fail rather than overwrite a new concurrent file.
                else:
                    os.replace(source_path, destination)
            run.parent.mkdir(parents=True, exist_ok=True)
            os.rename(staging/RUN, run)
            progress_ahead = reconciliation["max_training_iteration"] > resume["iteration"]
            if progress_ahead:
                stop_path = root/"runs/campaign/STOP"
                try:
                    with stop_path.open("x") as stream:
                        stream.write("Recovery review required: training metrics are newer than restored model weights. "
                                     "Do not resume the same iteration history until this discrepancy is resolved.\n")
                except FileExistsError:
                    pass  # Preserve any previous user stop request.
            report = {"status": "restored", "archive": archive.name, "sha256": sidecar["sha256"],
                      "source_commit": manifest["git"]["commit"], "restored_at": datetime.now(timezone.utc).isoformat(),
                      "files": len(expected), "rebased_json_files": rebased,
                      "progress_reconciliation": reconciliation,
                      "checkpoint_iteration": resume["iteration"],
                      "progress_ahead_of_checkpoint": progress_ahead, "review_required": progress_ahead,
                      "review_reason": "training metrics newer than restored weights; STOP created/preserved" if progress_ahead else None,
                      "note": "Restored process status is historical; no process was started."}
            atomic_json(root/"runs/campaign/restore.json", report)
            return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("create", "publish", "download", "restore"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
        if command in ("publish", "restore"):
            subparser.add_argument("--archive", type=Path, required=True)
        if command in ("publish", "download"):
            subparser.add_argument("--repo", default=REPO)
            subparser.add_argument("--release", default=TAG)
        if command in ("create", "download"):
            subparser.add_argument("--output-dir", type=Path)
        if command == "create":
            subparser.add_argument("--checkpoint-boundary", action="store_true",
                                   help="Campaign caller guarantees all writers are quiescent while it holds campaign.lock")
        if command == "download":
            subparser.add_argument("--latest", action="store_true", help="Discover newest complete asset pair instead of using Git pointer")
    args = parser.parse_args()
    if args.command == "create":
        result = create_snapshot(args.root, args.output_dir, args.checkpoint_boundary)
    elif args.command == "publish":
        result = publish_snapshot(args.root, args.archive, args.repo, args.release)
    elif args.command == "download":
        result = download_snapshot(args.root, args.output_dir, args.repo, args.release, args.latest)
    else:
        result = restore_snapshot(args.root, args.archive)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
