"""Training-only pins and immutable evidence for cached-head-constraint-v7."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/cached-head-constraint-v7"
PROGRESS = ROOT / "progress/cached-head-constraint-v7"
CONFIG = ROOT / "configs/cached-head-constraint-v7.json"
LOCKS = (ROOT / "runs/campaign/campaign.lock", ROOT / "runs/warmstart-v3/experiment.lock",
         ROOT / "runs/terminal-target-head-v5/experiment.lock", ROOT / "runs/terminal-target-head-v6/experiment.lock",
         RUN / "experiment.lock")
STOPS = (ROOT / "runs/campaign/STOP", ROOT / "runs/warmstart-v3/STOP",
         ROOT / "runs/terminal-target-head-v5/STOP", ROOT / "runs/terminal-target-head-v6/STOP", RUN / "STOP")
PINS = {
    "checkpoint": ("runs/warmstart-v3/checkpoint-0002.pt", "20d9e813349010630940e1cdfbaee5206490c7d323997688d26f47508bbeee7c"),
    "dataset": ("runs/terminal-target-head-v6/dataset/training.npz", "c6d0043b1de3c5d228598d749637cd73efa297da111669dd1431363baf9e5f26"),
    "dataset_metadata": ("runs/terminal-target-head-v6/dataset/training.json", "020b5615a430b986802f2ab2ebfbeb6eca532488d1ca1179a863987a8a876d7b"),
    "reference_terminal_checkpoint": ("runs/terminal-target-head-v6/fit/terminal_head.pt", "1479fe9474c3f149497fb9994ff53ad6e7fcdf79021fb12a444346b862530718"),
    "reference_batches": ("runs/terminal-target-head-v6/fit/batches.npz", "609a31a83326b44c5cf642a5d1804ad0d643e45ac860c3aea4b42ad6924cbd6a"),
    "v6_identity": ("progress/terminal-target-head-v6/identity.json", "1cd46fdc649c072e7ab0d736afd8960522d6db30c194375edef513feb4db608c"),
}
AUDIT_PINS = {
    "progress/terminal-target-head-v6/identity.json": "1cd46fdc649c072e7ab0d736afd8960522d6db30c194375edef513feb4db608c",
    "progress/terminal-target-head-v6/independent-final-audit.json": "ab8f824a645432916417e8142d61c7cd13ec5e69561b96869115ff95288dd564",
    "progress/terminal-target-head-v6/training-underfit.json": "4ff88c4f8fe68b623d35941415a92ac3741907d840da4e7aea405bf9b0f4da68",
}
SOURCES = ("alpha0/constraint_v7_common.py", "alpha0/constraint_v7.py", "alpha0/constraint_v7_fit.py",
           "alpha0/constraint_v7_report.py", "scripts/plot_constraint_v7.py",
           "alpha0/terminal_v6_fit.py", "alpha0/terminal_v6_common.py", "alpha0/network_v2.py",
           "alpha0/features_v2.py", "alpha0/learn_v2.py", "alpha0/network.py", "alpha0/env.py",
           "alpha0/mcts.py", "alpha0/train.py", "alpha0/replay.py", "alpha0/replay_v2.py",
           "alpha0/__init__.py", "docs/cached-head-constraint-v7.md")


class Interrupted(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _publish(path, data, immutable):
    path = Path(path)
    if immutable and path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".publishing-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            try:
                os.link(name, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError(f"concurrent immutable artifact differs: {path}")
        else:
            os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_json(path, value):
    _publish(path, _json_bytes(value), False)


def immutable_json(path, value):
    _publish(path, _json_bytes(value), True)


def immutable_bytes(path, value):
    _publish(path, value, True)


def immutable_npz(path, arrays):
    import numpy as np
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(old[k].dtype != np.asarray(v).dtype or
                    not np.array_equal(old[k], np.asarray(v)) for k, v in arrays.items()):
                raise ValueError(f"immutable arrays differ: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".arrays-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def stop_reason(deadline):
    if any(path.exists() for path in STOPS):
        return "STOP"
    if time.monotonic() >= deadline:
        return "deadline"
    return None


def check_stop(deadline):
    reason = stop_reason(deadline)
    if reason:
        raise Interrupted(reason)


def configure_threads():
    """Apply the same one-thread contract to parent verification and worker SGD."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import torch
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise ValueError("one-thread numerical runtime could not be configured")


def load_config(path=None):
    config = json.loads(Path(path or CONFIG).read_text())
    exact = {"experiment_id": "cached-head-constraint-v7", "schema_version": 1,
             "training_roots": 32, "anchor_count": 512, "changed_targets": 9, "retained_targets": 23,
             "trainable_parameters": 5160, "workers": 1, "torch_threads": 1,
             "total_wall_budget_seconds": 900, "updates": 2048, "snapshots": [128, 512, 2048],
             "checkpoint_interval": 16,
             "arms": [{"id": "kl_1", "anchor_kl_weight": 1.0}, {"id": "kl_0", "anchor_kl_weight": 0.0}],
             "validation_data_used_for_fitting": False, "benchmark_data_used_for_training": False,
             "automatic_promotion": False,
             "classification": {"required_greedy_targets": 32, "fixed_snapshots_only": True,
                "budget": "kl_1 reaches all32 at512or2048 after exact128 reproduction",
                "anchoring": "kl_0 reaches all32 at any fixed snapshot and kl_1 never does through2048",
                "neither": "neither arm reaches all32 at a fixed snapshot; review optimization/conditioning"},
             "fit": {"batch_size": 32, "anchor_batch_size": 32, "learning_rate": .0003,
                "weight_decay": .0001, "betas": [.9, .999], "eps": 1e-8, "seed": 203101, "target_weight": .75}}
    exact.update({key: {"path": name, "sha256": digest} for key, (name, digest) in PINS.items()})
    exact["v6_final_audit"] = {"path": "progress/terminal-target-head-v6/independent-final-audit.json",
                             "sha256": AUDIT_PINS["progress/terminal-target-head-v6/independent-final-audit.json"]}
    exact["training_diagnosis"] = {"path": "progress/terminal-target-head-v6/training-underfit.json",
                                 "sha256": AUDIT_PINS["progress/terminal-target-head-v6/training-underfit.json"]}
    if fingerprint(config) != fingerprint(exact):
        changed = sorted(key for key in set(config) | set(exact) if fingerprint(config.get(key)) != fingerprint(exact.get(key)))
        raise ValueError(f"v7 preregistration differs: {changed}")
    return config


def identity(config, config_path=None):
    """Hash prerequisites only; never parse held-out results or run an engine."""
    configure_threads()
    import numpy as np
    import torch
    inputs = {}
    for key, (name, expected) in PINS.items():
        if config.get(key) != {"path": name, "sha256": expected} or sha256(ROOT / name) != expected:
            raise ValueError(f"frozen training input changed: {key}")
        inputs[name] = expected
    for name, expected in AUDIT_PINS.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"frozen prerequisite changed: {name}")
        inputs[name] = expected
    metadata = json.loads((ROOT / config["dataset_metadata"]["path"]).read_text())
    if (metadata.get("complete") is not True or metadata.get("source") != "own_selfplay"
            or metadata.get("root_count") != 32 or metadata.get("anchor_count") != 512
            or metadata.get("npz_sha256") != config["dataset"]["sha256"]):
        raise ValueError("training dataset is incomplete or differs from 32 roots/512 anchors")
    with np.load(ROOT / config["dataset"]["path"], allow_pickle=False) as dataset:
        if dataset["root_hidden"].shape != (32, 128) or dataset["anchor_hidden"].shape != (512, 128):
            raise ValueError("training feature dimensions differ")
    return {"schema_version": 1, "experiment_id": config["experiment_id"], "config": config,
            "config_file_sha256": sha256(config_path or CONFIG), "inputs": inputs,
            "sources": {name: sha256(ROOT / name) for name in SOURCES},
            "runtime": {"python": platform.python_version(), "numpy": str(np.__version__), "torch": str(torch.__version__),
                        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "device": "cpu"},
            "training_only": True, "validation_data_read": False, "benchmark_data_read": False,
            "engine_games": 0, "promotion_allowed": False}
