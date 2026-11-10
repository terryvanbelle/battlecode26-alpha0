"""Pinned protocol, identities and lossless input hashing for terminal-head v8."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/terminal-target-head-v8"
PROGRESS = ROOT / "progress/terminal-target-head-v8"
CONFIG = ROOT / "configs/terminal-target-head-v8.json"
LOCKS = (ROOT / "runs/campaign/campaign.lock", ROOT / "runs/warmstart-v3/experiment.lock", ROOT / "runs/terminal-target-head-v5/experiment.lock", ROOT / "runs/terminal-target-head-v6/experiment.lock", ROOT / "runs/cached-head-constraint-v7/experiment.lock", RUN / "experiment.lock", ROOT / "runs/development-v3/experiment.lock")
STOPS = (ROOT / "runs/campaign/STOP", ROOT / "runs/warmstart-v3/STOP", ROOT / "runs/terminal-target-head-v5/STOP", ROOT / "runs/terminal-target-head-v6/STOP", ROOT / "runs/cached-head-constraint-v7/STOP", RUN / "STOP", ROOT / "runs/development-v3/STOP")
CHECKPOINT_SHA = "20d9e813349010630940e1cdfbaee5206490c7d323997688d26f47508bbeee7c"
V4_AUDIT_SHA = "b69616abb0bb802e386c7f5b00d5b7a061f188f4dccd8e7b5d1c09970bce1394"


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


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
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
            raise ValueError(f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".immutable-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(name, path)
        except FileExistsError:
            if json.loads(path.read_text()) != value:
                raise ValueError(f"concurrent immutable artifact differs: {path}")
    finally:
        Path(name).unlink(missing_ok=True)


def immutable_npz(path, arrays):
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


def immutable_bytes(path, value):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"immutable byte artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".bytes-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def input_hash(observation, legal_mask):
    """Canonical actual float32 model inputs, including the legal mask."""
    from .features_v2 import FEATURE_VERSION
    names = ("local", "global_board", "scalars", "legal_mask")
    arrays = []
    for value in observation:
        array = np.array(value, dtype="<f4", order="C", copy=True)
        if not np.isfinite(array).all():
            raise ValueError("nonfinite gradient input")
        array[array == 0] = 0  # Canonical positive zero, including converted -0.0.
        arrays.append(array)
    if len(arrays) != 3:
        raise ValueError("expected local/global/scalar inputs")
    mask = np.asarray(legal_mask)
    if mask.shape != (40,) or not np.isin(mask, [0, 1]).all() or not mask.any():
        raise ValueError("invalid legal mask")
    arrays.append(np.ascontiguousarray(mask, dtype="u1"))
    fields = [{"name": name, "shape": list(array.shape), "dtype": "|u1" if i == 3 else "<f4"}
              for i, (name, array) in enumerate(zip(names, arrays))]
    header = {"schema": "gradient-input-v1", "feature_version": FEATURE_VERSION,
              "action_version": "macro_v2", "fields": fields}
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    h = hashlib.sha256(len(encoded).to_bytes(8, "little") + encoded)
    for array in arrays:
        h.update(array.tobytes(order="C"))
    return h.hexdigest()


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


def set_task(path, kind, limit, **metadata):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    atomic_json(path, {"kind": kind, "started_at": now(),
                      "deadline_monotonic": time.monotonic() + float(limit), **metadata})


def clear_task(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path.unlink(missing_ok=True)


def release(env, *states):
    keep = {s.snapshot for s in states if s is not None and s.snapshot is not None}
    freed = list(env._snapshots - keep)
    if freed:
        env._request({"cmd": "release", "snapshots": freed})
        env._snapshots.difference_update(freed)


def source_job(split, map_index, j, config):
    if split not in ("training", "validation") or not 0 <= map_index < 4:
        raise ValueError("unknown source split/map")
    if not 1 <= j <= (8 if split == "training" else 6):
        raise ValueError("source outside preregistered seeds")
    engine_base = 20263200 if split == "training" else 20264200
    policy_base = 330000 if split == "training" else 340000
    return {"source_id": f"{split}-m{map_index}-j{j}", "split": split,
            "map_index": map_index, "j": j, "map": config["maps"][map_index],
            "seed": engine_base + 100 * map_index + j,
            "policy_seeds": [2 * (policy_base + 100 * map_index + j) + team for team in (0, 1)]}


# Historical inputs are used only for exclusion, never as v8 fit examples.
HISTORICAL_INPUT_PINS = {
    "runs/terminal-target-head-v6/dataset/training.json": "020b5615a430b986802f2ab2ebfbeb6eca532488d1ca1179a863987a8a876d7b",
    "progress/terminal-target-head-v6/roots-validation.json": "14552424f106bf83ffac3f98265c5f5f859e111a854c573e34b3d283f3e4be1d",
}


def historical_exclusions():
    for name, digest in HISTORICAL_INPUT_PINS.items():
        if sha256(ROOT / name) != digest:
            raise ValueError("historical exclusion input changed")
    training = json.loads((ROOT / "runs/terminal-target-head-v6/dataset/training.json").read_text())
    validation = json.loads((ROOT / "progress/terminal-target-head-v6/roots-validation.json").read_text())
    result = {}
    for kind, values in (("v6_training_root", training["root_input_hashes"]),
                         ("v6_anchor", training["anchor_input_hashes"]),
                         ("v6_validation_root", [r["input_hash"] for r in validation["roots"]])):
        for index, digest in enumerate(values):
            result.setdefault(digest, []).append({"kind": kind, "id": index})
    if len(result) != 560:
        raise ValueError("historical exclusion inventory must retain all 560 inputs")
    return result


def load_config(path=CONFIG):
    config = json.loads(Path(path).read_text())
    exact = {"experiment_id": "terminal-target-head-v8", "schema_version": 1,
        "checkpoint": "runs/warmstart-v3/checkpoint-0002.pt", "checkpoint_sha256": CHECKPOINT_SHA,
        "benchmark_manifest": "configs/benchmarks-development-v4.json", "improvement_config": "configs/improvement-v4.json",
        "maps": ["tiny", "DefaultSmall", "closeup", "knifefight"], "temperature": .35, "source_temperature": 1.0,
        "selection_seeds": [205003, 205013], "validation_seeds": [205021, 205043, 205067, 205091],
        "training_roots": 32, "validation_roots": 16, "anchor_count": 512, "anchor_seed": 205151,
        "selection_cell_ceiling": 1600, "workers": 3, "torch_threads": 1,
        "wall_budget_seconds": 10800, "benchmark_data_used_for_training": False,
        "v7_audit": {"path": "progress/cached-head-constraint-v7/independent-final-audit.json",
                     "sha256": "e3776f8fbde2d0ff65117c2efcf029a54265a734d00b1b069ada06dc70b6ebd2"}}
    for key, value in exact.items():
        if config.get(key) != value:
            raise ValueError(f"v8 preregistration differs: {key}")
    fit = {"updates": 2048, "batch_size": 32, "anchor_batch_size": 32, "learning_rate": .0003,
           "weight_decay": .0001, "betas": [.9, .999], "eps": 1e-8, "seed": 205101,
           "target_weight": .75, "anchor_kl_weight": 1}
    search = {"simulations": 32, "max_depth": 12, "cpuct": 1.5, "dirichlet_alpha": .3,
              "exploration_fraction": .25, "root_noise": True, "root_seed_base": 205201}
    thresholds = {"mean_delta_min": .1, "improved_roots_min": 2, "worst_root_delta_min": -.25,
                  "mean_vs_cp2_min": 0}
    if config.get("fit") != fit or config.get("search") != search or config.get("thresholds") != thresholds:
        raise ValueError("v8 fitting/search/decision protocol differs")
    for key in ("cell_timeout_seconds", "source_timeout_seconds", "response_timeout_seconds",
                "max_decisions", "max_error_attempts", "evaluation_timeout_seconds", "evaluation_attempts"):
        if not isinstance(config.get(key), (float, int)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"invalid operational bound: {key}")
    return config


def identity(config, config_path=CONFIG):
    from .learn_v2 import _training_runtime_identity
    from .evaluate import panel_fingerprint, runtime_identity
    if sha256(ROOT / config["checkpoint"]) != CHECKPOINT_SHA:
        raise ValueError("frozen cp2 changed")
    audit = ROOT / "progress/root-ranking-v4/independent-final-audit.json"
    if sha256(audit) != V4_AUDIT_SHA or not json.loads(audit.read_text())["independent_decision"]["passes_mechanism_rule"]:
        raise ValueError("verified v4 prerequisite missing or changed")
    coverage_audit = ROOT / "progress/terminal-target-head-v5/independent-coverage-audit.json"
    if sha256(coverage_audit) != "48b586e3bf2f1befe6c49a89de5d6d153b9506303e7c350a8e8baa4f559b4d1a":
        raise ValueError("verified v5 coverage prerequisite missing or changed")
    v7_audit = ROOT / "progress/cached-head-constraint-v7/independent-final-audit.json"
    if sha256(v7_audit) != "e3776f8fbde2d0ff65117c2efcf029a54265a734d00b1b069ada06dc70b6ebd2":
        raise ValueError("verified v7 budget prerequisite missing or changed")
    gate = json.loads((ROOT / "configs/victory_gate.json").read_text())
    panel = json.loads((ROOT / "configs/benchmarks.json").read_text())
    if gate["schema_version"] != 2 or panel_fingerprint(panel) != gate["panel_fingerprint_sha256"] or len(panel["bots"]) != 12:
        raise ValueError("frozen benchmark gate/panel differs")
    panel = json.loads((ROOT / config["benchmark_manifest"]).read_text())
    improvement = json.loads((ROOT / config["improvement_config"]).read_text())
    if len(panel["bots"]) != 8 or improvement["manifest"] != config["benchmark_manifest"]:
        raise ValueError("wrong development panel")
    inventory = [{"name": p.stem, "sha256": sha256(p)} for p in sorted((ROOT / "maps").glob("*.map26"))]
    if inventory != sorted(gate["frozen_maps"], key=lambda row: row["name"]):
        raise ValueError("all-map inventory changed; explicit gate update required")
    excluded = set(gate["training_excluded_maps"])
    if excluded & set(config["maps"]):
        raise ValueError("training-excluded map in pilot")
    historical = historical_exclusions()
    reserved = set(gate["stages"]["confirmation"]["seeds"])
    # Every v5/v6 source, target, anchor, optimizer and PUCT seed is historical.
    old = json.loads((ROOT / "configs/terminal-target-head-v6.json").read_text())
    reserved.update(old["selection_seeds"] + old["validation_seeds"] + [old["anchor_seed"], old["fit"]["seed"]])
    reserved.update(old["search"]["root_seed_base"] + i for i in range(32))
    for engine_base, policy_base, maximum in ((20261200, 310000, 8), (20262200, 320000, 6)):
        for m in range(4):
            for j in range(1, maximum + 1):
                reserved.add(engine_base + 100*m + j)
                reserved.update(2*(policy_base + 100*m + j)+t for t in (0, 1))
    v4 = json.loads((ROOT / "progress/root-ranking-v4/config.json").read_text())
    reserved.update(v4["selection_seeds"] + v4["validation_seeds"])
    old_roots = json.loads((ROOT / "progress/root-ranking-v4/roots.json").read_text())["roots"]
    reserved.update(r["seed"] for r in old_roots)
    reserved.update(v4["search"]["root_seed_base"] + r["ordinal"] for r in old_roots)
    used = set(config["selection_seeds"] + config["validation_seeds"] + [config["anchor_seed"], config["fit"]["seed"]])
    used.update(config["search"]["root_seed_base"] + n for n in range(32))
    for split, maximum in (("training", 8), ("validation", 6)):
        for m in range(4):
            for j in range(1, maximum + 1):
                job = source_job(split, m, j, config)
                used.update([job["seed"], *job["policy_seeds"]])
    if reserved & used:
        raise ValueError("pilot reuses a reserved/v4 seed")
    sources = ["alpha0/terminal_v8_common.py", "alpha0/terminal_v8_collect.py", "alpha0/terminal_v8_fit.py",
               "alpha0/terminal_v8_report.py", "alpha0/terminal_v8_development.py", "scripts/compare_development.py", "alpha0/terminal_v8.py", "alpha0/evaluate.py",
               "docs/terminal-target-head-v8.md", "scripts/plot_terminal_v8.py", "scripts/plot_terminal_v8_development.py",
               "docs/terminal-target-head-v5.md", "progress/terminal-target-head-v5/independent-coverage-audit.json",
               "docs/terminal-target-head-v6.md", "configs/terminal-target-head-v6.json",
               "progress/cached-head-constraint-v7/independent-final-audit.json"]
    return {"schema_version": 1, "experiment_id": config["experiment_id"], "config": config,
        "config_file_sha256": sha256(config_path), "checkpoint_sha256": CHECKPOINT_SHA,
        "historical_input_pins": HISTORICAL_INPUT_PINS, "historical_exclusion_hashes_sha256": fingerprint(sorted(historical)),
        "v7_audit_sha256": sha256(v7_audit), "v4_audit_sha256": V4_AUDIT_SHA, "v5_coverage_audit_sha256": sha256(coverage_audit), "runtime": _training_runtime_identity(),
        "evaluation_runtime": runtime_identity({"action_version": "macro_v2"}),
        "sources": {name: sha256(ROOT / name) for name in sources},
        "gate_sha256": sha256(ROOT / "configs/victory_gate.json"),
        "panel_sha256": sha256(ROOT / config["benchmark_manifest"]),
        "improvement_config": improvement, "improvement_config_sha256": sha256(ROOT / config["improvement_config"]), "maps": inventory,
        "training_excluded_maps": sorted(excluded), "reserved_confirmation_seeds": sorted(gate["stages"]["confirmation"]["seeds"])}
