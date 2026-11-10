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
            gh("release", "upload", tag, path, "--repo", repo)
    verified = release_info(repo, tag)
    names = {asset["name"] for asset in verified["assets"] if asset.get("state") == "uploaded"}
    if not {archive.name, sidecar_path.name}.issubset(names):
        raise RuntimeError("release upload did not publish the complete snapshot pair")
    pointer = {**sidecar, "repository": repo, "release": tag,
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
        name = expected["archive"]
        if not ASSET_RE.fullmatch(name):
            raise ValueError("invalid archive name in snapshot pointer")
    else:
        assets = {a["name"] for a in release_info(repo, tag)["assets"] if a.get("state") == "uploaded"}
        complete = sorted(name for name in assets if ASSET_RE.fullmatch(name) and name + ".manifest.json" in assets)
        if not complete:
            raise ValueError("no complete published snapshot pair exists")
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
              "maps_dir", "opponent_checkpoint"}
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
            # Validate every destination before publishing anything. Existing
            # progress/gate ledgers may be equal but are never overwritten.
            for name in expected:
                destination = root/name
                assert_no_symlink(root, destination)
                if destination.exists() and (not destination.is_file() or sha256(destination) != sha256(staging/name)):
                    raise ValueError(f"existing artifact differs; refusing to overwrite: {name}")
            # Publish evidence first. If interrupted, identical files are reusable
            # on retry. Publish the complete run directory in one rename last.
            for name in expected:
                if name.startswith(RUN + "/"):
                    continue
                destination = root/name
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(staging/name, destination)  # O_EXCL semantics: never overwrite a concurrent writer.
                except FileExistsError:
                    if sha256(destination) != sha256(staging/name):
                        raise RuntimeError("destination changed during restore")
            run.parent.mkdir(parents=True, exist_ok=True)
            os.rename(staging/RUN, run)
            report = {"status": "restored", "archive": archive.name, "sha256": sidecar["sha256"],
                      "source_commit": manifest["git"]["commit"], "restored_at": datetime.now(timezone.utc).isoformat(),
                      "files": len(expected), "rebased_json_files": rebased,
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
