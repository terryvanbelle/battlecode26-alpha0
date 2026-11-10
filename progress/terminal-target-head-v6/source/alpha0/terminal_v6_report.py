"""V6 evidence summaries and isolated benchmark worker adapters.

Reporting never schedules games or fits a policy. Benchmark attempts are invoked
explicitly by the process-group-owning supervisor, not by report().
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from . import evaluate as E
from . import terminal_v6_common as C

ROOT = E.ROOT
ARMS = ("cp2", "terminal_head", "puct_head")
VALIDATION_SEEDS = [203021, 203043, 203067, 203091]
THRESHOLDS = {"mean_delta_vs_puct_min": 0.10, "improved_roots_min": 2,
              "worst_root_delta_vs_puct_min": -0.25, "mean_delta_vs_cp2_min": 0.0}
BASELINE_SHA256 = "20d9e813349010630940e1cdfbaee5206490c7d323997688d26f47508bbeee7c"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def immutable_json(path: Path, value: Any) -> None:
    C.immutable_json(path, value)


def build_benchmark_specs(checkpoints: dict[str, Path], config: dict, identity: dict) -> list[dict]:
    """Build three fresh, matched full-policy smoke panels without running them."""
    if set(checkpoints) != set(ARMS):
        raise ValueError("V6 benchmark requires cp2, terminal_head and puct_head checkpoints")
    manifest_path = Path(config.get("benchmark_manifest", ROOT / "configs/benchmarks.json"))
    maps_dir = Path(config.get("maps_dir", ROOT / "maps"))
    manifest = json.loads(manifest_path.read_text())
    if identity.get("panel_sha256") and E.file_hash(manifest_path) != identity["panel_sha256"]:
        raise ValueError("Benchmark manifest differs from the frozen experiment dependency")
    if len(manifest["bots"]) != 12 or len({bot["id"] for bot in manifest["bots"]}) != 12:
        raise ValueError("V6 smoke requires the entire twelve-opponent panel")
    schedule = manifest["evaluation_protocol"]["smoke"]
    if schedule["maps"] != ["tiny"] or schedule["seeds"] != [0] or schedule.get("sides", ["A", "B"]) != ["A", "B"]:
        raise ValueError("V6 smoke schedule must remain tiny/seed0/both seats")
    timeout = config.get("evaluation_timeout_seconds", 1800)
    if timeout != 1800:
        raise ValueError("V6 benchmark watchdog is precommitted to 1800 seconds")
    experiment_hash = E.fingerprint(identity)
    specs, common_config = [], None
    for arm in ARMS:
        checkpoint = Path(checkpoints[arm]).resolve()
        checkpoint_hash = E.file_hash(checkpoint)
        if arm == "cp2" and checkpoint_hash != config.get("checkpoint_sha256", BASELINE_SHA256):
            raise ValueError("V6 baseline checkpoint differs from frozen cp2")
        _, _, payload, schema = E.load_policy(checkpoint)
        if E.file_hash(checkpoint) != checkpoint_hash or schema["action_version"] != "macro_v2":
            raise ValueError("Benchmark checkpoint changed or uses another action schema")
        runtime = E.runtime_identity(schema)
        if identity.get("evaluation_runtime") != runtime:
            raise ValueError("Benchmark runtime differs from the frozen experiment runtime")
        context = E.make_context(checkpoint_hash, schema, runtime, timeout_seconds=timeout)
        if common_config is None:
            common_config = context["evaluation_config"]
        elif common_config != context["evaluation_config"]:
            raise ValueError("Benchmark arms must share the exact greedy policy/runtime configuration")
        for evaluation in E.build_cell_specs(manifest, schedule, context, checkpoint=checkpoint,
                iteration=int(payload.get("iteration", 0)), suite="smoke", stage="smoke", maps_dir=maps_dir):
            key = {"experiment_identity": experiment_hash, "arm": arm,
                   "evaluation_cell_id": evaluation["evaluation_cell_id"]}
            specs.append({"stage": "benchmark", **key, "cell_id": E.fingerprint(key),
                          "evaluation_spec": evaluation})
    if len(specs) != 72 or len({spec["cell_id"] for spec in specs}) != 72:
        raise ValueError("V6 benchmark did not produce exactly 72 fresh arm/cell identities")
    return specs


def validate_benchmark_spec(spec: dict, experiment_hash: str | None = None) -> None:
    evaluation = spec["evaluation_spec"]
    config = evaluation["identity"]["evaluation_config"]
    key = {name: spec[name] for name in ("experiment_identity", "arm", "evaluation_cell_id")}
    if (spec.get("stage") != "benchmark" or spec["arm"] not in ARMS or spec["cell_id"] != E.fingerprint(key) or
        evaluation["evaluation_cell_id"] != spec["evaluation_cell_id"] or
        evaluation["evaluation_cell_id"] != E.fingerprint(evaluation["identity"]) or
        evaluation["identity"]["evaluation_config_sha256"] != E.fingerprint(config) or
        config["search_mode"] != "greedy" or config["timeout_seconds"] != 1800 or
        config["policy_schema"]["action_version"] != "macro_v2" or evaluation["identity"]["stage"] != "smoke" or
        (experiment_hash is not None and spec["experiment_identity"] != experiment_hash)):
        raise ValueError("Benchmark wrapper or experiment identity changed")


def wrap_benchmark_result(spec: dict, evaluation: dict, attempt: int) -> dict:
    return {"schema_version": 1, "stage": "benchmark", "experiment_identity": spec["experiment_identity"],
            "cell_id": spec["cell_id"], "arm": spec["arm"], "attempt": attempt,
            "status": evaluation["status"], "error_kind": evaluation.get("error_kind"),
            "evaluation": evaluation, "recorded_at": now()}


def benchmark_error_record(spec: dict, attempt: int, error_kind: str, error: str) -> dict:
    """Supervisor watchdog failures remain unscored, identity-bound attempts."""
    validate_benchmark_spec(spec)
    evaluation = E.result_base(spec["evaluation_spec"], attempt)
    evaluation.update(error_kind=error_kind, error=error)
    return wrap_benchmark_result(spec, evaluation, attempt)


def benchmark_worker(spec: dict, attempt: int = 1) -> dict:
    """Run exactly one game; caller owns process-group supervision and deadline."""
    validate_benchmark_spec(spec)
    if type(attempt) is not int or not 1 <= attempt <= 2:
        raise ValueError("V6 benchmark permits at most two recorded attempts per cell")
    evaluation = E.run_cell_worker(spec["evaluation_spec"], attempt)
    return wrap_benchmark_result(spec, evaluation, attempt)


def valid_benchmark_result(spec: dict, row: dict) -> bool:
    evaluation = row.get("evaluation", {})
    return (row.get("experiment_identity") == spec["experiment_identity"] and row.get("cell_id") == spec["cell_id"] and
            row.get("stage") == "benchmark" and row.get("arm") == spec["arm"] and
            type(row.get("attempt")) is int and 1 <= row["attempt"] <= 2 and
            evaluation.get("attempt") == row["attempt"] and row.get("status") == "ok" and
            E.valid_result(spec["evaluation_spec"], evaluation))


def select_benchmark_result(spec: dict, rows: list[dict]) -> dict | None:
    relevant = []
    seen_attempts = {}
    for row in rows:
        if row.get("cell_id") != spec["cell_id"]:
            continue
        if row.get("experiment_identity") != spec["experiment_identity"] or row.get("arm") != spec["arm"]:
            raise ValueError("Benchmark attempt belongs to another experiment or arm")
        attempt = row.get("attempt")
        if type(attempt) is not int or not 1 <= attempt <= 2:
            raise ValueError("Invalid benchmark attempt number")
        if attempt in seen_attempts:
            if seen_attempts[attempt] != row:
                raise ValueError("Conflicting records for one immutable benchmark attempt")
            continue
        seen_attempts[attempt] = row
        relevant.append(row)
    successes = [row for row in relevant if valid_benchmark_result(spec, row)]
    if len(successes) > 1:
        raise ValueError("Successful benchmark cell was replayed; refusing to select a favorable attempt")
    return successes[0] if successes else max(relevant, key=lambda row: row["attempt"], default=None)


def record_benchmark_attempt(spec: dict, directory: Path, *, attempt: int, max_attempts: int = 2) -> dict:
    """Atomic per-attempt recording; no retries or subprocess launches hidden here."""
    validate_benchmark_spec(spec)
    if max_attempts != 2 or type(attempt) is not int or not 1 <= attempt <= max_attempts:
        raise ValueError("V6 benchmark caps total attempts at two")
    directory = Path(directory)
    with E.locked(directory / ".lock"):
        immutable_json(directory / "request.json", spec)
        rows = [json.loads(path.read_text()) for path in sorted(directory.glob("attempt-[0-9][0-9][0-9].json"))]
        selected = select_benchmark_result(spec, rows)
        if selected is not None and valid_benchmark_result(spec, selected):
            return selected
        if any(row["attempt"] == attempt for row in rows):
            return next(row for row in rows if row["attempt"] == attempt)
        if attempt != max((row["attempt"] for row in rows), default=0) + 1:
            raise ValueError("Benchmark attempts must be recorded in order")
        if selected is not None and selected.get("error_kind") not in E.RETRYABLE_ERRORS:
            raise ValueError("Only failed infrastructure benchmark cells may retry")
        result = benchmark_worker(spec, attempt)
        immutable_json(directory / f"attempt-{attempt:03d}.json", result)
        return result


def summarize_benchmarks(specs: list[dict], attempts: list[dict], identity: dict,
                         *, required_opponents: int = 12, expected_panel: dict | None = None,
                         expected_checkpoints: dict[str, str] | None = None) -> dict:
    """No historical result, missing seat or pooled mean can satisfy this panel."""
    expected_total = required_opponents * 2 * len(ARMS)
    if not specs:
        if attempts:
            raise ValueError("Benchmark attempts exist without the frozen cell manifest")
        return {"expected": expected_total, "completed": 0, "unresolved_errors": 0,
                "attempt_errors": 0, "complete": False, "identity_verified": False, "panels": {}, "opponent_comparisons": {},
                "per_opponent_nonregression": False, "historical_cp2_score": {"wins": 20, "games": 24, "fresh": False}}
    if len(specs) != expected_total or len({spec["cell_id"] for spec in specs}) != len(specs):
        raise ValueError("Benchmark manifest must contain all three complete panels")
    experiment_hash = E.fingerprint(identity)
    cells = set(); configs = set(); panels_seen = set(); map_hashes = set(); opponent_identities = defaultdict(set)
    opponent_sets = defaultdict(set); arm_hashes = defaultdict(set)
    rows_by_id = defaultdict(list)
    expected_ids = {spec["cell_id"] for spec in specs}
    for row in attempts:
        if row.get("cell_id") not in expected_ids:
            raise ValueError("Unexpected benchmark attempt outside the frozen 72-cell manifest")
        rows_by_id[row["cell_id"]].append(row)
    panels = {arm: {"expected": required_opponents*2, "completed": 0, "unresolved_errors": 0,
                   "wins": 0, "draws": 0, "losses": 0, "score_sum": 0.0, "opponents": {}}
              for arm in ARMS}
    attempt_errors = 0
    for spec in specs:
        validate_benchmark_spec(spec, experiment_hash)
        arm = spec["arm"]; evaluation = spec["evaluation_spec"]; cell = evaluation["identity"]
        if cell["evaluation_config"]["runtime"] != identity.get("evaluation_runtime"):
            raise ValueError("Benchmark runtime differs from the frozen experiment runtime")
        key = (arm, cell["opponent"], cell["map"], cell["seed"], cell["side"])
        if key in cells or cell["map"] != "tiny" or cell["seed"] != 0 or cell["side"] not in ("A", "B"):
            raise ValueError("Benchmark arm/opponent/seat schedule changed")
        cells.add(key); configs.add(cell["evaluation_config_sha256"])
        map_hashes.add(cell["map_sha256"])
        panels_seen.add((cell["benchmark_suite_id"], cell["panel_fingerprint_sha256"]))
        opponent_identities[cell["opponent"]].add((cell["opponent_java_sha256"], cell["opponent_source_commit"]))
        opponent_sets[arm].add(cell["opponent"]); arm_hashes[arm].add(cell["checkpoint_sha256"])
        panel = panels[arm]
        opponent = panel["opponents"].setdefault(cell["opponent"], {"expected": 2, "completed": 0,
                    "unresolved_errors": 0, "score_sum": 0.0, "sides": {"A": None, "B": None}})
        history = rows_by_id[spec["cell_id"]]
        attempt_errors += len({row["attempt"] for row in history if not valid_benchmark_result(spec, row)})
        row = select_benchmark_result(spec, history)
        if row is not None and valid_benchmark_result(spec, row):
            score = row["evaluation"]["score"]
            panel["completed"] += 1; panel["score_sum"] += score
            panel[{1: "wins", .5: "draws", 0: "losses"}[score]] += 1
            opponent["completed"] += 1; opponent["score_sum"] += score
            opponent["sides"][cell["side"]] = {"score": score, "rounds": row["evaluation"]["rounds"],
                                                   "cell_id": spec["cell_id"], "attempt": row["attempt"]}
        elif row is not None:
            panel["unresolved_errors"] += 1; opponent["unresolved_errors"] += 1
    if (len(configs) != 1 or len(panels_seen) != 1 or len(map_hashes) != 1 or any(len(values) != 1 for values in opponent_identities.values()) or
        any(len(arm_hashes[arm]) != 1 or len(opponent_sets[arm]) != required_opponents for arm in ARMS)
        or any(opponent_sets[arm] != opponent_sets["cp2"] for arm in ARMS)):
        raise ValueError("Benchmark panels have mixed runtime, candidate or opponent identities")
    if "maps" in identity:
        frozen_tiny = [item["sha256"] for item in identity["maps"] if item["name"] == "tiny"]
        if len(frozen_tiny) != 1 or map_hashes != set(frozen_tiny):
            raise ValueError("Benchmark map differs from the frozen tiny map")
    baseline_hash = identity.get("checkpoint_sha256", identity.get("config", {}).get("checkpoint_sha256"))
    if baseline_hash and arm_hashes["cp2"] != {baseline_hash}:
        raise ValueError("Fresh cp2 panel used a substituted baseline checkpoint")
    if expected_checkpoints is not None:
        if set(expected_checkpoints) != set(ARMS) or any(arm_hashes[arm] != {expected_checkpoints[arm]} for arm in ARMS):
            raise ValueError("Benchmark candidate differs from the frozen fit checkpoint")
    if expected_panel is not None:
        pinned = {bot["id"]: (bot["java_tree_sha256"], bot["source_commit"]) for bot in expected_panel["bots"]}
        if (len(pinned) != required_opponents or set(pinned) != opponent_sets["cp2"] or
            panels_seen != {(expected_panel["suite_id"], E.panel_fingerprint(expected_panel))} or
            any(opponent_identities[name] != {pin} for name, pin in pinned.items())):
            raise ValueError("Benchmark opponent identities differ from the frozen panel")
    for arm, panel in panels.items():
        panel["checkpoint_sha256"] = next(iter(arm_hashes[arm]))
        panel["complete"] = panel["completed"] == panel["expected"] and panel["unresolved_errors"] == 0
        panel["mean_score"] = panel["score_sum"]/panel["completed"] if panel["complete"] else None
        for opponent in panel["opponents"].values():
            opponent["complete"] = opponent["completed"] == 2 and opponent["unresolved_errors"] == 0
            opponent["mean_score"] = opponent["score_sum"]/2 if opponent["complete"] else None
    comparisons = {}
    for name in sorted(opponent_sets["cp2"]):
        baseline = panels["cp2"]["opponents"][name]; candidate = panels["terminal_head"]["opponents"][name]
        complete = baseline["complete"] and candidate["complete"]
        delta = candidate["mean_score"]-baseline["mean_score"] if complete else None
        comparisons[name] = {"complete": complete, "cp2_mean": baseline["mean_score"],
                             "terminal_head_mean": candidate["mean_score"], "delta": delta,
                             "nonregression": complete and delta >= 0}
    complete = all(panel["complete"] for panel in panels.values())
    return {"expected": expected_total, "completed": sum(panel["completed"] for panel in panels.values()),
            "unresolved_errors": sum(panel["unresolved_errors"] for panel in panels.values()),
            "attempt_errors": attempt_errors, "complete": complete, "panels": panels,
            "identity_verified": bool(expected_panel is not None and expected_checkpoints is not None and baseline_hash
                                      and identity.get("evaluation_runtime")),
            "evaluation_config_sha256": next(iter(configs)), "opponent_comparisons": comparisons,
            "per_opponent_nonregression": complete and all(item["nonregression"] for item in comparisons.values()),
            "historical_cp2_score": {"wins": 20, "games": 24, "fresh": False}}


def continuation_spec(root: dict, stage: str, action: int, seed: int, config: dict, identity: dict) -> dict:
    from .terminal_v6_collect import cell_spec
    return cell_spec(root, stage, action, seed, config, identity)


def selected_continuation(rows: list[dict], expected: dict) -> dict | None:
    from .terminal_v6_collect import successful_cell
    # Exact duplicates of the same physical artifact do not add evidence. Two
    # separate successful attempts are forbidden even when their scores agree.
    unique = {}
    for row in rows:
        key = row.get("attempt")
        if key in unique and unique[key] != row:
            raise ValueError("Conflicting immutable continuation attempt")
        unique[key] = row
    ordered = list(unique.values())
    if sum(row.get("status") == "ok" for row in ordered) > 1:
        raise ValueError("Successful continuation was replayed")
    return successful_cell(ordered, expected)


def summarize_validation(roots: list[dict], attempts: list[dict], config: dict, identity: dict,
                         *, required_roots: int | None = None) -> dict:
    """Equal-root paired first-action scores, separate from full-policy games."""
    required = config.get("validation_roots", 16) if required_roots is None else required_roots
    seeds = config.get("validation_seeds", VALIDATION_SEEDS)
    if seeds != VALIDATION_SEEDS or not required or len({root["root_id"] for root in roots}) != len(roots):
        raise ValueError("Validation requires unique roots and the four preregistered paired seeds")
    if len(roots) > required:
        raise ValueError("More validation roots than the frozen allocation")
    root_ids = {root["root_id"] for root in roots}
    by_cell = defaultdict(list)
    for row in attempts:
        if row.get("stage") != "validation" or row.get("root_id") not in root_ids:
            raise ValueError("Validation evidence belongs to an unknown root or stage")
        by_cell[(row["root_id"], row.get("action"), row.get("seed"))].append(row)
    views = []; seen_cells = set(); completed = errors = attempt_errors = expected = 0
    for root in roots:
        actions = root.get("actions")
        legal = root["legal_actions"]
        if not actions:
            if any(key[0] == root["root_id"] for key in by_cell):
                raise ValueError("Validation outcomes exist before frozen candidate actions")
            views.append({"root_id": root["root_id"], "map": root["map"], "role": root["role"],
                          "player": root["player"], "input_hash": root["input_hash"],
                          "source_json": root["source_json"], "actions": None, "complete": False,
                          "expected_unique_cells": None, "completed_unique_cells": 0, "errors": 0,
                          "scores": {arm: None for arm in ARMS}, "means": {arm: None for arm in ARMS},
                          "delta_vs_puct": None, "delta_vs_cp2": None})
            continue
        if set(actions) != set(ARMS) or any(type(action) is not int or action not in legal for action in actions.values()):
            raise ValueError("Validation needs exactly three fixed legal arm actions")
        physical = {}; root_errors = 0
        for action in sorted(set(actions.values())):
            for seed in seeds:
                key = (root["root_id"], action, seed); seen_cells.add(key); expected += 1
                history = by_cell[key]
                spec = continuation_spec(root, "validation", action, seed, config, identity)
                outcome = selected_continuation(history, spec)
                attempt_errors += len({row["attempt"] for row in history if row.get("status") != "ok"})
                if outcome is not None:
                    physical[(action, seed)] = outcome["score"]; completed += 1
                elif history:
                    errors += 1; root_errors += 1
        complete = len(physical) == len(set(actions.values()))*len(seeds) and root_errors == 0
        scores = {arm: [physical[(actions[arm], seed)] for seed in seeds] if complete else None for arm in ARMS}
        means = {arm: sum(scores[arm])/len(seeds) if complete else None for arm in ARMS}
        views.append({"root_id": root["root_id"], "map": root["map"], "role": root["role"],
                      "player": root["player"], "input_hash": root["input_hash"], "source_json": root["source_json"],
                      "actions": actions, "complete": complete, "expected_unique_cells": len(set(actions.values()))*len(seeds),
                      "completed_unique_cells": len(physical), "errors": root_errors, "scores": scores, "means": means,
                      "delta_vs_puct": means["terminal_head"]-means["puct_head"] if complete else None,
                      "delta_vs_cp2": means["terminal_head"]-means["cp2"] if complete else None})
    if set(by_cell) - seen_cells:
        raise ValueError("Validation evidence uses an unselected action or another seed")
    complete = len(roots) == required and all(view["complete"] for view in views) and errors == 0
    def grouped(field):
        groups = {}
        for value in sorted({str(view[field]) for view in views}):
            subset = [view for view in views if str(view[field]) == value]
            ready = bool(subset) and all(view["complete"] for view in subset)
            groups[value] = {"roots": len(subset), "complete": ready,
                "completed_roots": sum(view["complete"] for view in subset),
                "mean_delta_vs_puct": sum(view["delta_vs_puct"] for view in subset)/len(subset) if ready else None,
                "mean_delta_vs_cp2": sum(view["delta_vs_cp2"] for view in subset)/len(subset) if ready else None}
        return groups
    deltas = [view["delta_vs_puct"] for view in views] if complete else []
    mean_puct = sum(deltas)/required if complete else None
    mean_cp2 = sum(view["delta_vs_cp2"] for view in views)/required if complete else None
    improved = sum(delta > 0 for delta in deltas) if complete else None
    worst = min(deltas) if complete else None
    criteria = bool(complete and mean_puct >= .10 and improved >= 2 and worst >= -.25 and mean_cp2 >= 0)
    return {"expected_roots": required, "frozen_roots": len(roots), "completed_roots": sum(view["complete"] for view in views),
            "distinct_source_games_represented": len({root["source_json"] for root in roots}),
            "distinct_validation_inputs": len({root["input_hash"] for root in roots}),
            "expected_unique_cells": expected if all(root.get("actions") for root in roots) and len(roots)==required else None,
            "maximum_unique_cells": required*len(ARMS)*len(seeds), "completed_unique_cells": completed,
            "unresolved_errors": errors, "attempt_errors": attempt_errors, "complete": complete,
            "roots": views, "groups": {"map": grouped("map"), "role": grouped("role"), "team": grouped("player")},
            "mean_delta_vs_puct": mean_puct, "mean_delta_vs_cp2": mean_cp2,
            "improved_roots": improved, "worst_root_delta_vs_puct": worst, "criteria_pass": criteria,
            "thresholds": THRESHOLDS, "seeds": seeds,
            "scope": "One candidate first action followed by frozen cp2 on both teams; not full-policy deployment."}


def summarize_selection(roots: list[dict], attempts: list[dict], config: dict, identity: dict) -> dict:
    required = config["training_roots"]
    if len(roots) > required or len({root["root_id"] for root in roots}) != len(roots):
        raise ValueError("Selection root inventory changed")
    by_cell = defaultdict(list)
    for row in attempts:
        by_cell[(row.get("root_id"), row.get("action"), row.get("seed"))].append(row)
    expected = completed = errors = attempt_errors = 0
    seen = set(); root_views = []
    for root in roots:
        count = done = unresolved = 0
        for action in root["legal_actions"]:
            for seed in config["selection_seeds"]:
                key = (root["root_id"], action, seed); seen.add(key)
                count += 1
                spec = continuation_spec(root, "selection", action, seed, config, identity)
                history = by_cell[key]
                outcome = selected_continuation(history, spec)
                if outcome is not None:
                    done += 1
                elif history:
                    unresolved += 1
                attempt_errors += len({row["attempt"] for row in history if row.get("status") != "ok"})
        expected += count; completed += done; errors += unresolved
        root_views.append({"root_id": root["root_id"], "expected": count, "completed": done,
                           "unresolved_errors": unresolved, "complete": count == done and unresolved == 0})
    if set(by_cell)-seen:
        raise ValueError("Selection evidence lies outside frozen roots/legal actions/seeds")
    within_budget = expected <= config["selection_cell_ceiling"]
    return {"planned_roots": required, "frozen_roots": len(roots), "completed_roots": sum(r["complete"] for r in root_views),
            "expected_cells": expected if len(roots)==required else None, "completed_cells": completed,
            "unresolved_errors": errors, "attempt_errors": attempt_errors, "cell_ceiling": config["selection_cell_ceiling"],
            "within_collection_ceiling": within_budget,
            "complete": len(roots)==required and within_budget and completed==expected and errors==0,
            "roots": root_views}


def artifact_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_manifest(path: Path, identity: dict, split: str, config: dict) -> dict:
    manifest = read_json(path)
    if manifest is None:
        return {"identity_sha256": E.fingerprint(identity), "split": split, "roots": [],
                "sources": {}, "assignments": [], "unavailable_slots": [], "complete": False}
    if manifest.get("identity_sha256") != E.fingerprint(identity) or manifest.get("split") != split:
        raise ValueError("Root manifest belongs to another experiment or split")
    roots = manifest["roots"]
    physical = [(root["source_id"], root["decision_index"]) for root in roots]
    if len(set(physical)) != len(roots) or len({root["root_id"] for root in roots}) != len(roots):
        raise ValueError("Duplicate physical roots or identifiers in frozen manifest")
    for root in roots:
        if (root["map"] not in config["maps"] or root["role"] not in ("BABY_RAT", "RAT_KING") or
            type(root["player"]) is not int or root["player"] not in (0, 1) or
            not root["legal_actions"] or len(set(root["legal_actions"])) != len(root["legal_actions"]) or
            37 not in root["legal_actions"] or any(type(a) is not int or not 0 <= a < 40 for a in root["legal_actions"])):
            raise ValueError("Frozen root does not satisfy preregistered role/team/map/legality")
    if manifest.get("complete") is True:
        if len(roots) != config[f"{split}_roots"] or manifest.get("unavailable_slots"):
            raise ValueError("Complete root manifest has missing planned slots")
        expected_per_stratum = 2 if split == "training" else 1
        for map_name in config["maps"]:
            for role in ("BABY_RAT", "RAT_KING"):
                for player in (0, 1):
                    if sum(root["map"]==map_name and root["role"]==role and root["player"]==player for root in roots) != expected_per_stratum:
                        raise ValueError("Root map/role/team allocation differs from preregistration")
    return manifest


def source_coverage(run_dir: Path, manifests: dict[str, dict], identity: dict) -> dict:
    from .terminal_v6_collect import verify_terminal
    result = {}
    for split, manifest in manifests.items():
        rows = [read_json(path) for path in sorted((run_dir/"sources").glob(f"{split}-*.json"))]
        attempts = [read_json(path) for path in sorted((run_dir/"sources").glob(f"{split}-*.attempts/attempt-[0-9][0-9][0-9].json"))]
        starts = [read_json(path) for path in sorted((run_dir/"sources").glob(f"{split}-*.attempts/attempt-[0-9][0-9][0-9]-start.json"))]
        for row in rows + attempts + starts:
            if row.get("experiment_identity") != E.fingerprint(identity) or row.get("split") != split:
                raise ValueError("Source-game evidence belongs to another experiment/split")
            if (row.get("source") != "selfplay" or row.get("temperature") != 1.0 or
                row.get("source_temperature") != 1.0):
                raise ValueError("V6 source games must be own self-play at temperature 1.0")
            if row.get("status") == "ok":
                verify_terminal(row)
        sources = {row["source_id"]: row for row in rows if row.get("status") == "ok"}
        if len(sources) != len(rows):
            raise ValueError("Final source files must contain one valid completed game each")
        for source in sources.values():
            cache = artifact_path(source["cache_npz"])
            if not cache.resolve().is_relative_to(run_dir.resolve()):
                raise ValueError("V6 source cache must belong to this freshly generated run")
            if not cache.is_file() or E.file_hash(cache) != source["cache_sha256"]:
                raise ValueError("Completed source input cache changed or disappeared")
        if manifest.get("complete") is True and set(manifest.get("sources", {})) != set(sources):
            raise ValueError("Complete root manifest omits completed discovery source games")
        for source_id, ref in manifest.get("sources", {}).items():
            path = artifact_path(ref["source_json"])
            if (path.resolve() != (run_dir/"sources"/f"{source_id}.json").resolve() or
                source_id not in sources or E.file_hash(path) != ref["source_sha256"]):
                raise ValueError("Frozen source-game reference changed or disappeared")
        finished_keys = {(row["source_id"], row["attempt"]) for row in attempts}
        start_only = [row for row in starts if (row["source_id"], row["attempt"]) not in finished_keys]
        for source_id in {row["source_id"] for row in attempts}:
            if sum(row.get("status") == "ok" and row["source_id"] == source_id for row in attempts) > 1:
                raise ValueError("Successful source game was replayed")
        attempted = {row["source_id"] for row in attempts + starts} | set(sources)
        result[split] = {"attempted_source_games": len(attempted), "completed_source_games": len(sources),
            "source_attempts": len(finished_keys | {(row["source_id"], row["attempt"]) for row in starts}),
            "start_only_attempts": len(start_only),
            "attempt_errors_or_interruptions": sum(row.get("status") != "ok" for row in attempts) + len(start_only),
            "unresolved_source_games": len(attempted-set(sources)),
            "source_games_represented_by_roots": len({root["source_id"] for root in manifest["roots"]}),
            "planned_roots": identity.get("config", {}).get(f"{split}_roots", 32 if split=="training" else 16),
            "frozen_roots": len(manifest["roots"]), "manifest_complete": manifest.get("complete") is True,
            "unavailable_slots": manifest.get("unavailable_slots", []),
            "reserve_assignments": [entry for entry in manifest.get("assignments", []) if entry["source_j"] != entry["slot_group_j"]],
            "skipped_exact_input_duplicates": sum(len(row.get("skipped_input_duplicates", [])) for row in sources.values()),
            "worker_cpu_seconds": sum(row.get("worker_cpu_seconds", 0) for row in attempts),
            "task_elapsed_seconds_sum": sum(row.get("elapsed_seconds", 0) for row in attempts)}
    return result


def verify_fit(run_dir: Path, progress_dir: Path, config: dict, identity: dict,
               training: dict, validation: dict, anchors: dict | None) -> dict:
    manifest = read_json(run_dir/"fit/manifest.json")
    if manifest is None:
        return {"complete": False, "checkpoints": {}, "reason": "Both completed fixed candidate heads are not yet published"}
    if anchors is None or training.get("complete") is not True or validation.get("complete") is not True:
        raise ValueError("Completed fit lacks frozen complete roots and anchors")
    if (manifest.get("complete") is not True or manifest.get("identity_sha256") != E.fingerprint(identity) or
        manifest.get("fit_config") != config["fit"] or set(manifest.get("checkpoints", {})) != {"terminal_head", "puct_head"} or
        manifest.get("fit_identity_sha256") != E.fingerprint(manifest.get("fit_identity"))):
        raise ValueError("Completed fit manifest identity/protocol changed")
    public = read_json(progress_dir/"fit-summary.json")
    if public != manifest:
        raise ValueError("Published fit summary differs from the complete candidate pair")
    for filename, digest in manifest["files"].items():
        if Path(filename).name != filename or E.file_hash(run_dir/"fit"/filename) != digest:
            raise ValueError("Completed fit artifact checksum mismatch")
    dataset = manifest["dataset"]
    dataset_path = artifact_path(dataset["path"])
    metadata_path = dataset_path.with_suffix(".json")
    if E.file_hash(dataset_path) != dataset["sha256"] or E.file_hash(metadata_path) != dataset["metadata_sha256"]:
        raise ValueError("Fit dataset provenance changed")
    metadata = read_json(metadata_path)
    if (dataset["root_count"] != 32 or dataset["anchor_count"] != 512 or
        metadata.get("identity_sha256") != E.fingerprint(identity) or metadata.get("source") != "own_selfplay" or
        metadata.get("benchmark_data_used") is not False or metadata.get("validation_data_used") is not False or
        metadata.get("provenance", {}).get("training_roots_sha256") != E.fingerprint(training) or
        metadata.get("provenance", {}).get("validation_roots_sha256") != E.fingerprint(validation) or
        metadata.get("provenance", {}).get("anchors_sha256") != E.fingerprint(anchors)):
        raise ValueError("Fit gradient inputs differ from frozen roots/anchors or include evaluation data")
    guard = manifest["gradient_input_overlap"]
    if (guard.get("checked") is not True or guard.get("intersection_count") != 0 or
        guard.get("validation_input_count") != 16 or guard.get("set_sha256") != anchors["gradient_input_hashes_sha256"]):
        raise ValueError("Fit lacks exact gradient/validation input isolation")
    schedule = manifest["batch_schedule"]
    if (schedule["root_shape"] != [128,32] or schedule["anchor_shape"] != [128,32] or schedule["seed"] != 203101 or
        E.file_hash(artifact_path(schedule["path"])) != schedule["sha256"]):
        raise ValueError("Fitting did not use the frozen matched batch schedule")
    for arm, item in manifest["checkpoints"].items():
        if (item.get("updates") != 128 or item.get("trainable_parameters") != 5160 or
            item.get("frozen_tensors_equal") is not True or item.get("value_logits_equal") is not True or
            E.file_hash(artifact_path(item["path"])) != item["sha256"]):
            raise ValueError("Candidate checkpoint or frozen-tensor proof differs from preregistration")
    return {"complete": True, "checkpoints": manifest["checkpoints"], "fit_identity_sha256": manifest["fit_identity_sha256"],
            "batch_schedule": schedule, "gradient_input_overlap": guard, "frozen_state_sha256": manifest["frozen_state_sha256"]}


def report(run_dir: Path, progress_dir: Path, config: dict, identity: dict, *, integrity_verified: bool = False) -> dict:
    """Read frozen evidence only; caller must separately certify final live inputs."""
    run_dir, progress_dir = Path(run_dir), Path(progress_dir)
    if identity.get("config") != config or config.get("experiment_id") != "terminal-target-head-v6":
        raise ValueError("Report config differs from the frozen v6 experiment")
    if config.get("source_temperature") != 1.0 or config.get("temperature") != .35:
        raise ValueError("V6 requires source temperature 1.0 and continuation temperature 0.35")
    training = load_manifest(progress_dir/"roots-training.json", identity, "training", config)
    validation_manifest = load_manifest(progress_dir/"roots-validation.json", identity, "validation", config)
    roots = [dict(root) for root in validation_manifest["roots"]]
    anchors = read_json(progress_dir/"anchors.json")
    anchor_complete = False
    if anchors is not None:
        gradient = sorted(set(root["input_hash"] for root in training["roots"]) |
                          {anchor["input_hash"] for anchor in anchors["anchors"]})
        if (anchors.get("identity_sha256") != E.fingerprint(identity) or anchors["gradient_input_hashes"] != gradient or
            anchors["gradient_input_hashes_sha256"] != E.fingerprint(gradient) or
            any(root["input_hash"] in gradient for root in roots)):
            raise ValueError("Anchor/gradient-input inventory differs or overlaps validation")
        if validation_manifest.get("complete") and validation_manifest["gradient_input_hashes_sha256"] != E.fingerprint(gradient):
            raise ValueError("Validation was selected against another gradient-input set")
        anchor_complete = anchors.get("complete") is True and anchors.get("count") == 512 and len(anchors["anchors"]) == 512
    collection = source_coverage(run_dir, {"training": training, "validation": validation_manifest}, identity)
    fit = verify_fit(run_dir, progress_dir, config, identity, training, validation_manifest, anchors)
    expected_checkpoints = {"cp2": config["checkpoint_sha256"], **{arm:item["sha256"] for arm,item in fit["checkpoints"].items()}}
    actions = read_json(progress_dir/"validation-actions.json")
    if actions is not None:
        if (not fit["complete"] or actions.get("identity_sha256") != E.fingerprint(identity) or
            actions.get("roots_sha256") != E.fingerprint(validation_manifest) or
            actions.get("checkpoint_sha256_by_arm") != expected_checkpoints or
            set(actions.get("actions", {})) != {root["root_id"] for root in roots}):
            raise ValueError("Validation actions predate or differ from the frozen candidate heads/roots")
        for root in roots:
            root["actions"] = actions["actions"][root["root_id"]]
    cell_rows = [read_json(path) for path in sorted((progress_dir/"cells").glob("*/attempt-[0-9][0-9][0-9].json"))]
    if any(row.get("stage") not in ("selection", "validation") for row in cell_rows):
        raise ValueError("Unknown continuation stage")
    selection = summarize_selection(training["roots"], [row for row in cell_rows if row["stage"]=="selection"], config, identity)
    validation = summarize_validation(roots, [row for row in cell_rows if row["stage"]=="validation"], config, identity)
    benchmark_manifest = read_json(progress_dir/"benchmarks/specs.json")
    specs = []
    if benchmark_manifest is not None:
        if not fit["complete"] or benchmark_manifest.get("identity_sha256") != E.fingerprint(identity):
            raise ValueError("Benchmark manifest lacks frozen fitted candidates")
        specs = benchmark_manifest["specs"]
    benchmark_rows = [read_json(path) for path in sorted((progress_dir/"benchmarks/cells").glob("*/attempt-[0-9][0-9][0-9].json"))]
    panel_path = Path(config.get("benchmark_manifest", ROOT/"configs/benchmarks.json"))
    if E.file_hash(panel_path) != identity["panel_sha256"]:
        raise ValueError("Frozen benchmark dependency changed")
    benchmarks = summarize_benchmarks(specs, benchmark_rows, identity, expected_panel=read_json(panel_path),
                                      expected_checkpoints=expected_checkpoints if fit["complete"] else None)
    protocol_ready = (training.get("complete") is True and validation_manifest.get("complete") is True and
                      anchor_complete and selection["complete"] and fit["complete"])
    complete = bool(protocol_ready and validation["complete"] and benchmarks["complete"] and benchmarks["identity_verified"])
    criteria_pass = bool(complete and validation["criteria_pass"] and benchmarks["per_opponent_nonregression"])
    decision = {"complete": complete, "criteria_pass": criteria_pass, "integrity_verified": integrity_verified is True,
                "review_for_expansion": criteria_pass and integrity_verified is True,
                "automatic_promotion": False, "all_map_victory": False,
                "scope": "Precommitted research review rule; no model promotion or all-map victory claim."}
    return {"schema_version": 1, "experiment_id": config["experiment_id"], "identity_sha256": E.fingerprint(identity),
            "created_at": now(), "benchmark_data_used_for_training": False, "complete": complete,
            "sampling": {"source_temperature": config["source_temperature"], "continuation_temperature": config["temperature"],
                "source_policy": "frozen_cp2_both_teams", "v5_data_reused": False,
                "seed_inventory": "paired_v5_same_seeds_fresh_generation"},
            "collection": collection, "anchors": {"complete": anchor_complete, "planned": 512,
                "completed": len(anchors["anchors"]) if anchors else 0,
                "unique_gradient_inputs": len(anchors["gradient_input_hashes"]) if anchors else 0},
            "selection": selection, "fit": fit, "validation": validation, "benchmarks": benchmarks, "decision": decision}


def write_report(summary: dict, progress_dir: Path) -> dict:
    """Publish a measured report and figures; never invoke collection or fitting."""
    from scripts.plot_terminal_v6 import generate
    progress_dir = Path(progress_dir)
    E.atomic_json(progress_dir/"summary.json", summary)
    return generate(summary, progress_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=C.RUN)
    parser.add_argument("--progress-dir", type=Path, default=C.PROGRESS)
    args = parser.parse_args()
    config = C.load_config()
    identity = read_json(args.progress_dir/"identity.json")
    if identity is None:
        parser.error("Frozen v6 identity.json is required")
    summary = report(args.run_dir, args.progress_dir, config, identity)
    write_report(summary, args.progress_dir)
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
