"""Frozen all-map qualification and fresh-seed confirmation.

Every opponent/map pair must pass. Atomic game artifacts and subprocess workers
come from evaluate.py; its older opponent-pooled victory gate is never used here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time

from . import evaluate as E
from .gate_config import training_excluded_maps

ROOT = E.ROOT
STAGES = ("standard", "confirmation")


def map_inventory(maps_dir: Path) -> list[dict]:
    paths = sorted(Path(maps_dir).glob("*.map26"), key=lambda path: path.stem)
    if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("Map inventory requires ordinary non-symlink .map26 files")
    if len({path.stem.casefold() for path in paths}) != len(paths):
        raise ValueError("Map inventory contains case-insensitive duplicate names")
    return [{"name": path.stem, "sha256": E.file_hash(path)} for path in paths]


def verify_gate(manifest: dict, gate: dict, maps_dir: Path) -> None:
    if type(gate.get("schema_version")) is not int or gate["schema_version"] != 2 or set(gate.get("stages", {})) != set(STAGES):
        raise ValueError("All-map victory requires schema 2 with standard and confirmation stages")
    E.verify_gate(manifest, gate)
    ids = [bot["id"] for bot in manifest["bots"]]
    frozen = sorted(gate["frozen_opponents"], key=lambda bot: bot["id"])
    if len(set(ids)) != len(ids) or E.fingerprint(frozen) != gate["panel_fingerprint_sha256"]:
        raise ValueError("Frozen opponent inventory has duplicates or does not match its fingerprint")
    maps = gate["frozen_maps"]
    if (maps != sorted(maps, key=lambda item: item["name"]) or
        len({item["name"] for item in maps}) != len(maps) or
        gate["required_maps"] != len(maps) or
        E.fingerprint(maps) != gate["map_inventory_sha256"] or maps != map_inventory(maps_dir)):
        raise ValueError("Frozen map inventory changed: added, missing, duplicate or modified maps require a new gate")
    names = [item["name"] for item in maps]
    if not training_excluded_maps(gate).issubset(names):
        raise ValueError("Training exclusions must belong to the complete frozen map inventory")
    for stage in STAGES:
        schedule = gate["stages"][stage]
        seeds = schedule["seeds"]
        if (schedule["maps"] != names or schedule["sides"] != ["A", "B"] or
            len(seeds) != len(set(seeds)) or len(seeds) < (2 if stage == "standard" else 4) or
            any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)):
            raise ValueError("Every stage must cover the complete frozen map set, unique seeds and both seats")
        for key, minimum in (("min_score_per_map", 0.6), ("min_score_each_side_per_map", 0.5)):
            value = schedule[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= 1:
                raise ValueError("All-map score thresholds cannot weaken the declared victory criterion")
    if set(gate["stages"]["standard"]["seeds"]) & set(gate["stages"]["confirmation"]["seeds"]):
        raise ValueError("Confirmation seeds must be fresh relative to standard qualification")


def gate_context(checkpoint_hash: str, schema: dict, runtime: dict, gate: dict, *,
                 timeout_seconds: float, search_mode: str = "greedy", search_config: dict | None = None) -> dict:
    context = E.make_context(checkpoint_hash, schema, runtime, timeout_seconds=timeout_seconds,
                             search_mode=search_mode, search_config=search_config)
    context["evaluation_config"]["all_map_gate"] = {
        "gate_id": gate["gate_id"], "gate_config_sha256": E.fingerprint(gate),
        "map_inventory_sha256": gate["map_inventory_sha256"],
        "victory_source_sha256": E.file_hash(Path(__file__)),
        "gate_config_source_sha256": E.file_hash(Path(__file__).with_name("gate_config.py"))}
    context["evaluation_config_sha256"] = E.fingerprint(context["evaluation_config"])
    return context


def build_specs(manifest: dict, gate: dict, stage: str, context: dict, *, checkpoint: Path,
                iteration: int, maps_dir: Path) -> list[dict]:
    if stage not in STAGES:
        raise ValueError("Stage must be standard or confirmation")
    return E.build_cell_specs(manifest, gate["stages"][stage], context, checkpoint=checkpoint,
            iteration=iteration, suite="all_maps_victory", stage=f"{gate['gate_id']}:{stage}", maps_dir=maps_dir)


def indexed_rows(rows: list[dict]) -> dict[str, list[dict]]:
    result = defaultdict(list)
    for row in rows:
        result[row.get("evaluation_cell_id")].append(row)
    return result


def assert_consistent_successes(spec: dict, rows: list[dict]) -> None:
    outcomes = {E.fingerprint({key: row[key] for key in ("score", "rounds", "winner", "terminal")})
                for row in rows if E.valid_result(spec, row)}
    if len(outcomes) > 1:
        raise ValueError("Conflicting successful outcomes for an immutable evaluation cell")


def merge_progress(progress: Path, store: Path) -> None:
    """Preserve published successes and check raw evidence before any replacement.

    This uses the evaluator's merge lock, but deliberately does not call its
    older merger: that merger can overwrite a published success before comparing
    it with a changed raw attempt. No reader can select a favorable rerun here.
    """
    with E.locked(progress / ".evaluation-merge.lock"):
        output = progress / "evaluations.jsonl"
        existing = E.read_jsonl(output)
        legacy = [row for row in existing if row.get("artifact_version") != E.ARTIFACT_VERSION or "evaluation_cell_id" not in row]
        current = {row["evaluation_cell_id"]: row for row in existing
                   if row.get("artifact_version") == E.ARTIFACT_VERSION and "evaluation_cell_id" in row}
        errors = {(row["evaluation_cell_id"], row["attempt"]): row
                  for row in E.read_jsonl(progress / "evaluation_errors.jsonl")}
        published = indexed_rows(existing)
        for request in sorted(store.glob("*/request.json")):
            spec = json.loads(request.read_text())
            cell_id = spec["evaluation_cell_id"]
            if cell_id != request.parent.name or cell_id != E.fingerprint(spec["identity"]):
                raise ValueError("Evaluation cell request identity changed")
            known = published[cell_id]
            if any(row.get("identity") != spec["identity"] for row in known):
                raise ValueError("Published evaluation cell identity changed")
            attempts = E.attempt_rows(request.parent)
            assert_consistent_successes(spec, known + attempts)
            # The already published valid outcome is authoritative even if its
            # raw artifact vanished or a later attempt carries the same score.
            result = next((row for row in known if E.valid_result(spec, row)), None)
            if result is None:
                result = E.selected_result(spec, attempts)
            if result is not None:
                row = dict(result)
                row["cell_artifact_dir"] = str(request.parent)
                if row.get("error"):
                    row["error"] = row["error"].splitlines()[0][:600]
                current[cell_id] = row
            for row in attempts:
                if row.get("status") == "error":
                    errors[(row["evaluation_cell_id"], row["attempt"])] = row
        rows = legacy + sorted(current.values(), key=lambda row: (row["iteration"], row["checkpoint"],
                       row["opponent"], row["map"], row["seed"], row["side"]))
        E.atomic_text(output, "".join(E.canonical(row) + "\n" for row in rows))
        E.atomic_text(progress / "evaluation_errors.jsonl", "".join(E.canonical(row) + "\n" for _, row in sorted(errors.items())))


def stage_report(specs: list[dict], rows: list[dict], gate: dict, stage: str, *, max_attempts: int = 2) -> dict:
    """Linear-time reporting; missing/extra expected cells are configuration errors."""
    schedule = gate["stages"][stage]
    names = [item["name"] for item in gate["frozen_maps"]]
    ids = [bot["id"] for bot in gate["frozen_opponents"]]
    expected = {(bot, name, seed, side) for bot in ids for name in names
                for seed in schedule["seeds"] for side in schedule["sides"]}
    actual = [(s["identity"]["opponent"], s["identity"]["map"], s["identity"]["seed"], s["identity"]["side"]) for s in specs]
    if (not expected or len(specs) != len(expected) or set(actual) != expected or
        len({s["evaluation_cell_id"] for s in specs}) != len(specs)):
        raise ValueError("Victory reporting requires the exact full opponent/map/seed/seat schedule")
    context_ids = {(s["identity"]["checkpoint_sha256"], s["identity"]["evaluation_config_sha256"]) for s in specs}
    map_hashes = {item["name"]: item["sha256"] for item in gate["frozen_maps"]}
    bots = {bot["id"]: bot for bot in gate["frozen_opponents"]}
    gate_hash = E.fingerprint(gate)
    for spec in specs:
        identity = spec["identity"]
        bot = bots[identity["opponent"]]
        contract = identity["evaluation_config"].get("all_map_gate", {})
        if (len(context_ids) != 1 or identity["stage"] != f"{gate['gate_id']}:{stage}" or
            identity["map_sha256"] != map_hashes[identity["map"]] or
            identity["panel_fingerprint_sha256"] != gate["panel_fingerprint_sha256"] or
            identity["opponent_java_sha256"] != bot["java_tree_sha256"] or
            identity["opponent_source_commit"] != bot["source_commit"] or
            contract.get("gate_config_sha256") != gate_hash or
            spec["evaluation_cell_id"] != E.fingerprint(identity) or
            identity["evaluation_config_sha256"] != E.fingerprint(identity["evaluation_config"])):
            raise ValueError("Mixed or stale candidate, runtime, map, opponent or gate identities")
    row_index = indexed_rows(rows)
    groups = defaultdict(list)
    for spec in specs:
        groups[(spec["identity"]["opponent"], spec["identity"]["map"])].append(spec)
    opponents = {}
    pending_ids = []
    complete_pairs = passed_pairs = completed = errors = exhausted = 0
    for bot in ids:
        stats = {"expected": 0, "completed": 0, "errors": 0, "wins": 0, "draws": 0,
                 "losses": 0, "score_sum": 0.0, "maps": {},
                 "sides": {side: {"expected": 0, "completed": 0, "score_sum": 0.0} for side in ("A", "B")}}
        for name in names:
            group = groups[(bot, name)]
            group_rows = [row for spec in group for row in row_index[spec["evaluation_cell_id"]]]
            for spec in group:
                assert_consistent_successes(spec, row_index[spec["evaluation_cell_id"]])
            cell_report = E.completion_report(group, group_rows, max_attempts=max_attempts, required_opponents=1)
            cell = cell_report["opponents"][bot]
            cell["passed"] = bool(cell_report["complete"] and cell_report["unresolved_errors"] == 0 and
                cell["mean_score"] >= schedule["min_score_per_map"] and
                all(side["mean_score"] >= schedule["min_score_each_side_per_map"] for side in cell["sides"].values()))
            cell["complete"] = cell_report["complete"]
            stats["maps"][name] = cell
            for key in ("expected", "completed", "errors", "wins", "draws", "losses", "score_sum"):
                stats[key] += cell[key]
            for side in ("A", "B"):
                for key in ("expected", "completed", "score_sum"):
                    stats["sides"][side][key] += cell["sides"][side][key]
            pending_ids.extend(cell_report["pending_cell_ids"])
            complete_pairs += cell_report["complete"]
            passed_pairs += cell["passed"]
            completed += cell_report["completed"]
            errors += cell_report["unresolved_errors"]
            exhausted += cell_report["exhausted_cells"]
        stats["mean_score"] = stats["score_sum"] / stats["completed"] if stats["completed"] else None
        for side in stats["sides"].values():
            side["mean_score"] = side["score_sum"] / side["completed"] if side["completed"] else None
        stats["passed"] = all(item["passed"] for item in stats["maps"].values())
        opponents[bot] = stats
    complete = completed == len(specs)
    return {"expected": len(specs), "completed": completed, "pending": len(pending_ids),
            "pending_cell_ids": pending_ids, "unresolved_errors": errors, "exhausted_cells": exhausted,
            "complete": complete, "full_panel": True, "full_panel_complete": complete,
            "all_maps_complete": complete, "expected_opponent_map_pairs": len(groups),
            "completed_opponent_map_pairs": complete_pairs, "passed_opponent_map_pairs": passed_pairs,
            "required_maps": len(names), "opponents": opponents,
            "passed": complete and errors == 0 and passed_pairs == len(groups)}


def candidate_identity(gate: dict, context: dict) -> dict:
    return {"gate_id": gate["gate_id"], "gate_config_sha256": E.fingerprint(gate),
            "panel_fingerprint_sha256": gate["panel_fingerprint_sha256"],
            "map_inventory_sha256": gate["map_inventory_sha256"],
            "checkpoint_sha256": context["checkpoint_sha256"],
            "evaluation_config_sha256": context["evaluation_config_sha256"]}


def matching_ledger(path: Path, gate: dict, context: dict) -> dict | None:
    if not path.exists():
        return None
    ledger = json.loads(path.read_text())
    if any(ledger.get(key) != value for key, value in candidate_identity(gate, context).items()):
        raise ValueError("Confirmation is already locked or consumed by another candidate/configuration")
    if ledger.get("status") not in ("in_progress", "passed", "consumed_failed"):
        raise ValueError("Invalid confirmation ledger status")
    return ledger


def results_fingerprint(specs: list[dict], rows: list[dict]) -> str:
    by_id = indexed_rows(rows)
    outcomes = []
    for spec in sorted(specs, key=lambda item: item["evaluation_cell_id"]):
        row = E.selected_result(spec, by_id[spec["evaluation_cell_id"]])
        outcomes.append({"evaluation_cell_id": spec["evaluation_cell_id"], "result": None if row is None else
                         {key: row.get(key) for key in ("attempt", "score", "rounds", "winner", "terminal")}})
    return E.fingerprint(outcomes)


def training_lineage(root: Path, checkpoint: Path, payload: dict, gate: dict) -> dict:
    """Follow checksum-pinned initialization weights without rewriting checkpoints."""
    root = root.resolve()
    excluded = training_excluded_maps(gate)
    maps, seeds, sources, seen = set(), set(), {}, set()
    def visit(path, data):
        path = path.resolve()
        if path in seen:
            raise ValueError("Cyclic initialization lineage")
        seen.add(path)
        config = data.get("config", {})
        bootstrap = data.get("bootstrap_config", {})
        declared_maps = data.get("training_maps", config.get("maps",
                              bootstrap.get("source_curriculum", bootstrap).get("maps")))
        if not isinstance(declared_maps, (list, tuple)) or any(not isinstance(name, str) for name in declared_maps):
            raise ValueError("Checkpoint ancestry is missing training-map provenance")
        if set(declared_maps) & excluded:
            raise ValueError("Checkpoint ancestry violates reserved training-map exclusions")
        maps.update(declared_maps)
        declared_seeds = data.get("training_seeds", [])
        if not isinstance(declared_seeds, list) or any(type(seed) is not int for seed in declared_seeds):
            raise ValueError("Invalid ancestor training-seed provenance")
        seeds.update(declared_seeds)
        if data.get("source_dataset_sha256"):
            dataset = path.parent / "source_dataset.json"
            if E.file_hash(dataset) != data["source_dataset_sha256"] or data.get("benchmark_data_used") is not False:
                raise ValueError("Initialization dataset provenance changed or contains benchmark examples")
            sources[str(dataset)] = E.file_hash(dataset)
            for game in json.loads(dataset.read_text())["games"]:
                if game.get("map") not in declared_maps or game.get("map") in excluded or type(game.get("seed")) is not int:
                    raise ValueError("Initialization dataset map/seed violates its provenance")
                seeds.add(game["seed"])
        initialization = config.get("initialization")
        if initialization:
            ancestor = (root / initialization["artifact"]).resolve()
            if not ancestor.is_relative_to(root) or E.file_hash(ancestor) != initialization["artifact_sha256"]:
                raise ValueError("Pinned initialization artifact changed or is outside the repository")
            import torch
            ancestor_data = torch.load(ancestor, map_location="cpu", weights_only=True)
            if E.file_hash(ancestor) != initialization["artifact_sha256"]:
                raise ValueError("Initialization artifact changed while loading")
            if initialization.get("source_dataset_sha256") and initialization["source_dataset_sha256"] != ancestor_data.get("source_dataset_sha256"):
                raise ValueError("Initialization artifact and dataset lineage disagree")
            sources[str(ancestor)] = initialization["artifact_sha256"]
            visit(ancestor, ancestor_data)
    visit(checkpoint, payload)
    return {"maps": sorted(maps), "training_seeds": sorted(seeds),
            "source_inventory_sha256": E.fingerprint(sources), "ancestors": len(seen)-1}


def audit_fresh_seeds(root: Path, checkpoint: Path, payload: dict, gate: dict, context: dict,
                      *, resume: bool, progress: Path) -> dict:
    """Reject prior recorded exposure, including failed games and other gate versions."""
    reserved = set(gate["stages"]["confirmation"]["seeds"])
    lineage = training_lineage(root, checkpoint, payload, gate)
    if reserved & set(lineage["training_seeds"]):
        raise ValueError("Confirmation seed appears in initialization ancestry")
    sources, consumed_ids = {}, set()
    checkpoint_iteration, games_total = payload.get("iteration", 0), payload.get("games_total", 0)
    if type(checkpoint_iteration) is not int or checkpoint_iteration < 0 or type(games_total) is not int or games_total < 0:
        raise ValueError("Invalid checkpoint iteration/game-count provenance")
    runs = root / "runs"
    for path in sorted(set(runs.rglob("plan-*.json")) | set(checkpoint.parent.glob("plan-*.json"))):
        plan = json.loads(path.read_text())
        sources[str(path)] = E.file_hash(path)
        match = re.fullmatch(r"plan-([0-9]+)\.json", path.name)
        consumed_plan = path.parent == checkpoint.parent and match is not None and 1 <= int(match[1]) <= checkpoint_iteration
        for job in plan.get("jobs", []):
            if type(job.get("seed")) is not int or not 0 <= job["seed"] < 2**31:
                raise ValueError(f"Invalid seed in a recorded training plan: {path}")
            if job.get("seed") in reserved:
                raise ValueError(f"Confirmation seed was used in a training plan: {path}")
            if consumed_plan:
                game_id = job.get("game_id")
                if (not isinstance(game_id, str) or
                    not re.fullmatch(fr"iter-{int(match[1]):04d}-game-[0-9]{{3}}", game_id) or
                    game_id in consumed_ids):
                    raise ValueError("Consumed training plan has invalid or duplicate game identities")
                completed_path = checkpoint.parent / "games" / f"{game_id}.json"
                if not completed_path.exists():
                    continue
                completed = json.loads(completed_path.read_text())
                if completed.get("status") != "ok":
                    continue
                if (completed.get("game_id") != game_id or completed.get("seed") != job["seed"] or
                    completed.get("map") != job.get("map") or
                    not isinstance(job.get("identity"), dict) or completed.get("identity") != job["identity"] or
                    any(job["identity"].get(key) != job.get(key) for key in ("game_id", "map", "seed"))):
                    raise ValueError("Completed training game does not match its consumed plan")
                sources[str(completed_path)] = E.file_hash(completed_path)
                consumed_ids.add(game_id)
    declared = payload.get("training_seeds")
    if declared is not None and (not isinstance(declared, list) or any(type(seed) is not int for seed in declared)):
        raise ValueError("Invalid training seed provenance")
    if reserved & set(declared or []):
        raise ValueError("Confirmation seed appears in checkpoint training provenance")
    if games_total != len(consumed_ids) and games_total != len(declared or []):
        raise ValueError("Complete training seed history is required before fresh-seed confirmation")
    game_files = set(runs.rglob("games/*.json")) | set((root / "artifacts").rglob("games/*.json"))
    for path in sorted(game_files):
        row = json.loads(path.read_text())
        sources[str(path)] = E.file_hash(path)
        if row.get("seed") in reserved or row.get("identity", {}).get("seed") in reserved:
            raise ValueError(f"Confirmation seed was used for self-play: {path}")
    for path in sorted((root / "artifacts").rglob("source_dataset.json")):
        sources[str(path)] = E.file_hash(path)
        if any(game.get("seed") in reserved for game in json.loads(path.read_text()).get("games", [])):
            raise ValueError(f"Confirmation seed was used in an initialization dataset: {path}")
    for path in (runs / "gate-ledgers").glob("*.json"):
        prior = json.loads(path.read_text())
        prior_seeds = set(prior.get("seed_audit", {}).get("confirmation_seeds", []))
        if prior_seeds & reserved and not (resume and all(prior.get(key) == value
                for key, value in candidate_identity(gate, context).items())):
            raise ValueError(f"Confirmation seeds already belong to another locked/consumed gate: {path}")
    paths = set((root / "progress").rglob("*.jsonl")) | set(progress.glob("*.jsonl"))
    for path in sorted(paths):
        sources[str(path)] = E.file_hash(path)
        for row in E.read_jsonl(path):
            if row.get("seed") not in reserved:
                continue
            identity = row.get("identity", {})
            authorized_resume = (resume and identity.get("checkpoint_sha256") == context["checkpoint_sha256"] and
                identity.get("evaluation_config_sha256") == context["evaluation_config_sha256"] and
                identity.get("stage") == f"{gate['gate_id']}:confirmation")
            if not authorized_resume:
                raise ValueError(f"Confirmation seed was previously exposed in evaluation: {path}")
    # Atomic artifacts can survive a crash before the JSONL merger runs.
    stores = set((root / "progress").rglob("evaluation_cells")) | {progress / "evaluation_cells"}
    for store in stores:
        for path in store.glob("*/request.json"):
            identity = json.loads(path.read_text()).get("identity", {})
            if identity.get("seed") in reserved and not (resume and
                    identity.get("checkpoint_sha256") == context["checkpoint_sha256"] and
                    identity.get("evaluation_config_sha256") == context["evaluation_config_sha256"] and
                    identity.get("stage") == f"{gate['gate_id']}:confirmation"):
                raise ValueError(f"Confirmation seed was previously scheduled: {path}")
    return {"audited_files": len(sources), "source_inventory_sha256": E.fingerprint(sources),
            "confirmation_seeds": sorted(reserved), "training_lineage": lineage,
            "consumed_training_games": len(consumed_ids), "checkpoint_games_total": games_total}


def lock_confirmation(path: Path, gate: dict, context: dict, *, standard_passed: bool, audit,
                      qualification_results_sha256: str | None = None) -> dict:
    if not standard_passed:
        raise ValueError("Confirmation requires every opponent/map pair to pass standard with the identical candidate")
    with E.locked(path.with_suffix(".lock")):
        ledger = matching_ledger(path, gate, context)
        if ledger is not None and ledger.get("qualification_results_sha256") != qualification_results_sha256:
            raise ValueError("Locked qualification evidence changed; restore its original archive")
        seed_audit = audit(ledger is not None)
        if ledger is None:
            ledger = {**candidate_identity(gate, context), "status": "in_progress",
                      "started_at": datetime.now(timezone.utc).isoformat(), "seed_audit": seed_audit,
                      "qualification_results_sha256": qualification_results_sha256}
            E.atomic_json(path, ledger)
        return ledger


def write_stage_report(report: dict, progress: Path, gate: dict) -> None:
    E.atomic_json(progress / f"latest-all-maps-{report['stage']}.json", report)
    from scripts.plot_victory import generate
    generate(report, progress, gate=gate)


def evaluate_victory(checkpoint: Path, *, stage: str = "standard", iteration: int | None = None,
                     generation: int | None = None, progress_dir: Path = ROOT / "progress/campaign-v2",
                     jobs: int = 2, timeout_seconds: float = 1800, max_attempts: int = 2,
                     search_mode: str = "greedy", search_config: dict | None = None,
                     manifest_path: Path = ROOT / "configs/benchmarks.json", maps_dir: Path = ROOT / "maps",
                     gate_path: Path = ROOT / "configs/victory_gate.json",
                     gate_ledger_dir: Path = ROOT / "runs/gate-ledgers",
                     deadline: float | None = None, stop_file: Path | None = None, execute: bool = True,
                     root: Path = ROOT) -> dict:
    if not 1 <= jobs <= 32 or not 1 <= max_attempts <= 5 or stage not in STAGES:
        raise ValueError("Require stage standard/confirmation, jobs 1..32 and max_attempts 1..5")
    if search_mode != "greedy" or search_config:
        raise ValueError("Only greedy evaluation is implemented; unsupported search policies cannot be relabeled as evidence")
    checkpoint, progress, maps_dir = Path(checkpoint).resolve(), Path(progress_dir).resolve(), Path(maps_dir).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    gate = json.loads(Path(gate_path).read_text())
    verify_gate(manifest, gate, maps_dir)
    checkpoint_hash = E.file_hash(checkpoint)
    _, _, payload, schema = E.load_policy(checkpoint)
    if E.file_hash(checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed while loading")
    lineage = training_lineage(Path(root), checkpoint, payload, gate)
    iteration = int(generation if generation is not None else iteration if iteration is not None else payload.get("iteration", 0))
    context = gate_context(checkpoint_hash, schema, E.runtime_identity(schema), gate,
                          timeout_seconds=timeout_seconds, search_mode=search_mode, search_config=search_config)
    specs = build_specs(manifest, gate, stage, context, checkpoint=checkpoint, iteration=iteration, maps_dir=maps_dir)
    store = progress / "evaluation_cells"
    ledger_path = Path(gate_ledger_dir) / f"{gate['gate_id']}.json"
    pre_stop = ("stop_file" if stop_file is not None and Path(stop_file).exists() else
                "deadline" if deadline is not None and time.monotonic() >= deadline else None)
    execute_now = execute and pre_stop is None
    standard, ledger = None, None
    if stage == "confirmation":
        standard_specs = build_specs(manifest, gate, "standard", context, checkpoint=checkpoint,
                                     iteration=iteration, maps_dir=maps_dir)
        standard_rows = E.collect_rows(progress, store, standard_specs)
        standard = stage_report(standard_specs, standard_rows, gate, "standard")
        ledger = matching_ledger(ledger_path, gate, context)
        if execute_now:
            ledger = lock_confirmation(ledger_path, gate, context, standard_passed=standard["passed"],
                qualification_results_sha256=results_fingerprint(standard_specs, standard_rows),
                audit=lambda resume: audit_fresh_seeds(Path(root), checkpoint, payload, gate, context,
                                                       resume=resume, progress=progress))
    stopped = "report_only" if not execute else pre_stop
    if execute_now:
        E.restore_known_attempts(specs, progress, store)
        existing = E.collect_rows(progress, store, specs)
        # Compare published and raw successful evidence before any merger can
        # replace the public row or any worker can launch another attempt.
        existing_report = stage_report(specs, existing, gate, stage, max_attempts=max_attempts)
        if ledger is not None and ledger["status"] in ("passed", "consumed_failed"):
            if not existing_report["complete"]:
                raise ValueError("Consumed confirmation evidence is missing; restore it rather than replay successful cells")
        by_id = indexed_rows(existing)
        pending = []
        for spec in specs:
            row = E.selected_result(spec, by_id[spec["evaluation_cell_id"]])
            if row is None or (not E.valid_result(spec, row) and row.get("error_kind") in E.RETRYABLE_ERRORS and
                               int(row.get("attempt", 0)) < max_attempts):
                pending.append(spec)
        unmerged, last_merge = 0, time.monotonic()
        def on_result(row):
            nonlocal unmerged, last_merge
            unmerged += 1
            if unmerged >= 64 or time.monotonic() - last_merge >= 60:
                merge_progress(progress, store)
                unmerged, last_merge = 0, time.monotonic()
                partial = stage_report(specs, E.collect_rows(progress, store, specs), gate, stage, max_attempts=max_attempts)
                partial["score_criteria_passed"] = partial["passed"]
                partial["passed"] = False  # Final inventory/ledger validation has not happened yet.
                partial.update(**candidate_identity(gate, context), stage=stage, iteration=iteration,
                               suite="all_maps_victory", victory_gate_passed=False, report_kind="in_progress")
                write_stage_report(partial, progress, gate)
            print(E.canonical({key: row.get(key) for key in ("iteration", "opponent", "map", "seed", "side", "status", "score", "rounds", "error_kind")}), flush=True)
        stopped = E.run_schedule(pending, store, jobs=jobs, max_attempts=max_attempts,
                                on_result=on_result, deadline=deadline, stop_file=stop_file)
        merge_progress(progress, store)
    # Fail closed if a gate, source, map or panel changes while games are running.
    if E.fingerprint(json.loads(Path(gate_path).read_text())) != E.fingerprint(gate):
        raise ValueError("Gate configuration changed during evaluation")
    verify_gate(json.loads(Path(manifest_path).read_text()), gate, maps_dir)
    if E.file_hash(Path(__file__)) != context["evaluation_config"]["all_map_gate"]["victory_source_sha256"]:
        raise ValueError("Victory implementation changed during evaluation")
    if E.file_hash(Path(__file__).with_name("gate_config.py")) != context["evaluation_config"]["all_map_gate"]["gate_config_source_sha256"]:
        raise ValueError("Gate validation implementation changed during evaluation")
    if E.file_hash(checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed during evaluation; original results cannot certify its replacement")
    if E.runtime_identity(schema) != context["evaluation_config"]["runtime"]:
        raise ValueError("Evaluation runtime changed during qualification")
    if training_lineage(Path(root), checkpoint, payload, gate) != lineage:
        raise ValueError("Pinned training lineage changed during evaluation")
    rows = E.collect_rows(progress, store, specs)
    report = stage_report(specs, rows, gate, stage, max_attempts=max_attempts)
    if stage == "confirmation":
        standard_rows = E.collect_rows(progress, store, standard_specs)
        standard = stage_report(standard_specs, standard_rows, gate, "standard")
        if ledger is not None and (not standard["passed"] or
                results_fingerprint(standard_specs, standard_rows) != ledger.get("qualification_results_sha256")):
            raise ValueError("Locked qualification evidence changed or is missing; restore its original archive")
    # Bind a completed/consumed attempt to its immutable selected game results.
    evidence_hash = results_fingerprint(specs, rows) if report["complete"] else None
    if ledger is not None and ledger["status"] in ("passed", "consumed_failed"):
        if not report["complete"] or evidence_hash != ledger.get("completed_results_sha256"):
            raise ValueError("Completed confirmation evidence changed or is missing; restore its original archive")
    score_criteria_passed = report["passed"]
    if stage == "confirmation":
        report["passed"] = bool(score_criteria_passed and ledger and ledger["status"] != "consumed_failed" and standard["passed"])
    report.update(**candidate_identity(gate, context), benchmark_suite_id=manifest["suite_id"],
                  suite="all_maps_victory", stage=stage, iteration=iteration,
                  expected_cell_ids=[spec["evaluation_cell_id"] for spec in specs], executed=execute_now,
                  stop_reason=stopped, stopped_early=bool(stopped and not report["complete"]),
                  standard_stage_passed=bool(standard and standard["passed"]),
                  confirmation_locked=ledger is not None,
                  score_criteria_passed=score_criteria_passed,
                  training_lineage=lineage,
                  victory_gate_passed=bool(stage == "confirmation" and ledger and standard["passed"] and report["passed"]))
    if stage == "confirmation":
        report["confirmation_ledger"] = str(ledger_path)
        if execute_now:
            with E.locked(ledger_path.with_suffix(".lock")):
                ledger = matching_ledger(ledger_path, gate, context)
                ledger["latest_report"] = report
                if report["complete"]:
                    ledger["status"] = "passed" if report["passed"] else "consumed_failed"
                    ledger["completed_at"] = datetime.now(timezone.utc).isoformat()
                    ledger["completed_results_sha256"] = evidence_hash
                E.atomic_json(ledger_path, ledger)
    if execute:
        run_id = E.fingerprint({"context": context, "cell_ids": report["expected_cell_ids"]})
        E.atomic_json(progress / "evaluation_reports" / f"{run_id}.json", report)
        write_stage_report(report, progress, gate)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--stage", choices=STAGES, default="standard")
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--progress-dir", type=Path, default=ROOT / "progress/campaign-v2")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--hours", type=float)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    report = evaluate_victory(args.checkpoint, stage=args.stage, iteration=args.iteration,
        progress_dir=args.progress_dir, jobs=args.jobs, timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts, deadline=None if args.hours is None else time.monotonic()+3600*args.hours,
        stop_file=args.stop_file, execute=not args.report_only)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
