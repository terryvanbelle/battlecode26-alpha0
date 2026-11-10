"""V8 collection workers: complete fresh source games and terminal action cells.

No benchmark policies, fitting, allocation decisions or process supervision live
here. Each worker owns at most one Java engine and uses one Torch thread. The
supervisor freezes allocations and enforces process-group deadlines separately.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from alpha0 import terminal_v8_common as C
from alpha0.env import EngineEnv
from alpha0.features_v2 import Observation, make_evaluator
from alpha0.learn_v2 import load_checkpoint
from alpha0.mcts import SearchConfig, search
from alpha0.terminal_v8_fit import encode_state

ROLES = ("BABY_RAT", "RAT_KING")
SOURCE_FIELDS = ("source_id", "split", "map_index", "j", "map", "seed", "policy_seeds")
CACHE_FIELDS = ("hidden", "prior", "mask", "local", "global_board", "scalars")
RESERVOIR_CAPACITY = 16
CACHE_SHAPES = {"hidden": (128,), "prior": (40,), "mask": (40,),
                "local": (33, 17, 17), "global_board": (33, 60, 60), "scalars": (64,)}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def relative(path):
    path = Path(path)
    try:
        return str(path.relative_to(C.ROOT))
    except ValueError:
        return str(path)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else C.ROOT / path


def state_hash(raw):
    return C.fingerprint({key: value for key, value in raw.items() if key != "snapshot"})


def actor_info(state):
    actor = next(robot for robot in state.raw["robots"] if robot["id"] == state.raw["current_id"])
    if actor["type"] not in ROLES or state.player not in (0, 1):
        raise ValueError("unexpected source actor role/team")
    return actor


def source_spec(job, config, identity):
    if identity.get("config", config) != config:
        raise ValueError("collector configuration differs from frozen experiment identity")
    split, m, j = job["split"], job["map_index"], job["j"]
    if split not in ("training", "validation") or type(m) is not int or not 0 <= m < 4:
        raise ValueError("invalid source split/map index")
    if type(j) is not int or not 1 <= j <= (8 if split == "training" else 6):
        raise ValueError("source j is outside the predeclared initial/reserve inventory")
    seed_base, policy_base = (20263200, 330000) if split == "training" else (20264200, 340000)
    expected = {"source_id": f"{split}-m{m}-j{j}", "split": split, "map_index": m, "j": j,
                "map": config["maps"][m], "seed": seed_base + 100*m + j,
                "policy_seeds": [2*(policy_base + 100*m + j) + team for team in (0, 1)]}
    if any(job.get(key) != value for key, value in expected.items()):
        raise ValueError("source job differs from preregistered seed/slot inventory")
    exclusions = job.get("exclusion_map", {})
    if not isinstance(exclusions, dict):
        raise ValueError("gradient-input exclusion map must be a dictionary")
    if split == "training" and exclusions:
        raise ValueError("training discovery must not filter gradient inputs")
    if split == "validation" and (not isinstance(exclusions, dict) or not exclusions):
        raise ValueError("validation requires the frozen gradient-input exclusion map")
    return {"experiment_identity": C.fingerprint(identity), **expected,
            "checkpoint_sha256": config["checkpoint_sha256"], "temperature": config["source_temperature"],
            "source_temperature": config["source_temperature"],
            "exclusion_map_sha256": C.fingerprint(exclusions), "anchor_seed": config["anchor_seed"]}


def reservoir_seed(anchor_seed, source_id, player):
    """Independent, stable source/team stream, unrelated to policy RNG consumption."""
    return int(C.fingerprint({"anchor_seed": anchor_seed, "source_id": source_id, "player": player})[:16], 16)


def reservoir_index(rng, seen, capacity=RESERVOIR_CAPACITY):
    """Algorithm R: return a replacement slot or None for the 1-indexed item."""
    if type(seen) is not int or seen <= 0:
        raise ValueError("reservoir seen count must be positive")
    index = seen - 1 if seen <= capacity else int(rng.integers(seen))
    return index if index < capacity else None


def choose_policy_action(prior, legal_actions, rng, temperature):
    legal = np.asarray(legal_actions, dtype=np.int64)
    prior = np.asarray(prior, dtype=np.float64)
    if prior.shape != (40,) or not len(legal) or len(set(legal.tolist())) != len(legal):
        raise ValueError("invalid policy/action shape")
    if not np.isfinite(prior).all() or np.any(prior < 0) or prior[legal].sum() <= 0:
        raise ValueError("invalid frozen policy probabilities")
    if temperature not in (.35, 1.0):
        raise ValueError("v8 temperatures are fixed at source=1.0 and continuation=0.35")
    logits = np.log(np.maximum(prior[legal], 1e-12)) / temperature
    weights = np.exp(logits - logits.max())
    return int(rng.choice(legal, p=weights / weights.sum()))


def checked_features(encoded, state):
    result = {}
    for key, shape in CACHE_SHAPES.items():
        value = np.asarray(encoded[key])
        dtype = np.dtype("uint8") if key == "mask" else np.dtype("float32")
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"invalid encoded {key} shape/dtype/values")
        result[key] = np.ascontiguousarray(value).copy()
    expected_mask = np.zeros(40, dtype=np.uint8)
    expected_mask[list(state.legal_actions)] = 1
    prior, mask = result["prior"], result["mask"]
    if (not np.array_equal(mask, expected_mask) or np.any(prior < 0) or np.any(prior[mask == 0] != 0)
            or not np.isclose(prior.sum(dtype=np.float64), 1, atol=1e-6)):
        raise ValueError("cached prior/mask must be normalized over the recorded legal actions")
    actual_hash = C.input_hash(Observation(result["local"], result["global_board"], result["scalars"]), mask)
    if actual_hash != encoded["input_hash"]:
        raise ValueError("feature input hash differs from exact retained float32 arrays")
    result["input_hash"] = actual_hash
    return result


def position_metadata(state, actions, encoded):
    actor = actor_info(state)
    return {"decision_index": len(actions), "player": state.player, "role": actor["type"],
            "round": state.raw["round"], "actor_id": actor["id"],
            "legal_actions": list(state.legal_actions), "prefix_sha256": C.fingerprint(actions),
            "state_sha256": state_hash(state.raw), "input_hash": encoded["input_hash"]}


def terminal_proof(state, root_player=None):
    if not state.terminal or state.raw.get("valid_game", True) is not True or state.raw.get("benchmark_failures", []):
        raise ValueError("no valid official terminal game")
    winner = state.raw.get("winner")
    if type(winner) is not int or winner not in (-1, 0, 1) or type(state.raw.get("round")) is not int or state.raw["round"] <= 0:
        raise ValueError("invalid official terminal winner/round")
    raw = {key: value for key, value in state.raw.items() if key != "snapshot"}
    result = {"terminal": True, "valid_game": True, "benchmark_failures": [],
              "winner": winner, "terminal_round": state.raw["round"],
              "terminal_state_sha256": state_hash(raw), "terminal_state": raw,
              "terminal_summary": {key: raw.get(key) for key in
                  ("terminal", "round", "winner", "win_reason", "bank", "kings", "transferred", "cat_damage")}}
    if root_player is not None:
        if root_player not in (0, 1):
            raise ValueError("invalid root player")
        result["score"] = .5 if winner == -1 else float(winner == root_player)
    return result


def verify_terminal(record, root_player=None):
    if (record.get("status") != "ok" or record.get("terminal") is not True
            or record.get("valid_game") is not True or record.get("benchmark_failures") != []):
        raise ValueError("successful record lacks valid terminal proof")
    raw = record.get("terminal_state", {})
    if (raw.get("terminal") is not True or raw.get("winner") != record.get("winner")
            or raw.get("round") != record.get("terminal_round") or record.get("terminal_round", 0) <= 0
            or raw.get("valid_game", True) is not True or raw.get("benchmark_failures", []) != []
            or state_hash(raw) != record.get("terminal_state_sha256")):
        raise ValueError("terminal proof hash/winner/round mismatch")
    winner = record["winner"]
    if (type(winner) is not int or type(raw.get("winner")) is not int or winner not in (-1, 0, 1)
            or type(record.get("terminal_round")) is not int or type(raw.get("round")) is not int):
        raise ValueError("invalid terminal winner")
    if root_player is not None:
        expected = .5 if winner == -1 else float(winner == root_player)
        if type(record.get("score")) not in (int, float) or record.get("score") not in (0, .5, 1) or record.get("score") != expected:
            raise ValueError("terminal score does not match root-team winner")
    expected_summary = {key: raw.get(key) for key in
                        ("terminal", "round", "winner", "win_reason", "bank", "kings", "transferred", "cat_damage")}
    if record.get("terminal_summary") != expected_summary:
        raise ValueError("terminal summary differs from raw proof")
    actions = record.get("actions")
    if (not isinstance(actions, list) or any(type(a) is not int or not 0 <= a < 40 for a in actions)
            or record.get("decisions") != len(actions) or type(record.get("decisions")) is not int):
        raise ValueError("terminal action transcript count/type mismatch")
    if C.fingerprint(record.get("actions", [])) != record.get("actions_sha256"):
        raise ValueError("completed action transcript changed")


def cell_spec(root, stage, action, seed, config, identity):
    if stage not in ("selection", "validation"):
        raise ValueError("invalid terminal-cell stage")
    seeds = config[f"{stage}_seeds"]
    if seed not in seeds or action not in root["legal_actions"]:
        raise ValueError("cell is outside frozen legal-action/seed inventory")
    return {"experiment_identity": C.fingerprint(identity), "root_id": root["root_id"],
            "root_player": root["player"], "root_state_sha256": root["state_sha256"],
            "root_prefix_sha256": root["prefix_sha256"], "root_input_hash": root["input_hash"],
            "source_sha256": root["source_sha256"], "stage": stage, "action": int(action),
            "seed": int(seed), "temperature": config["temperature"]}


def successful_cell(attempts, expected_spec):
    expected_identity = C.fingerprint(expected_spec)
    successes = []
    for attempt in attempts:
        if attempt.get("identity") != expected_identity or any(attempt.get(k) != v for k, v in expected_spec.items()):
            raise ValueError("terminal cell identity mismatch")
        if attempt.get("status") == "ok":
            verify_terminal(attempt, expected_spec["root_player"])
            if not attempt["actions"] or attempt["actions"][0] != expected_spec["action"]:
                raise ValueError("cell transcript did not force its declared first action")
            successes.append(attempt)
        elif attempt.get("score") is not None or attempt.get("terminal") is True:
            raise ValueError("failed/incomplete cell cannot carry a terminal score")
    if len(successes) > 1:
        raise ValueError("duplicate successful terminal cell")
    return successes[0] if successes else None


def _attempts(directory):
    return [json.loads(path.read_text()) for path in sorted(directory.glob("attempt-[0-9][0-9][0-9].json"))]


def _next_attempt(directory):
    return max((int(path.name.split("-")[1].split(".")[0]) for path in directory.glob("attempt-*")), default=0) + 1


def _task(job, kind, limit, deadline, **metadata):
    C.check_stop(deadline)
    task_path = resolve(job["worker_task_path"])
    C.set_task(task_path, kind, min(limit, max(.01, deadline-time.monotonic())), **metadata)
    return task_path


def _budget(env, config, deadline, task_deadline):
    C.check_stop(deadline)
    if time.monotonic() >= task_deadline:
        raise TimeoutError("collector task timeout")
    env.response_timeout = min(config["response_timeout_seconds"], max(.01, min(deadline, task_deadline)-time.monotonic()))


def _interrupted(error, deadline):
    return (isinstance(error, (KeyboardInterrupt, SystemExit)) or error.__class__.__name__ in ("Interrupted", "StopRequested")
            or time.monotonic() >= deadline)


def _model(config):
    torch.set_num_threads(1)
    if C.sha256(C.ROOT / config["checkpoint"]) != config["checkpoint_sha256"]:
        raise ValueError("frozen cp2 changed")
    model, _ = load_checkpoint(C.ROOT / config["checkpoint"])
    model.eval()
    return model


def _write_npz(path, arrays):
    # Atomic exclusive publication prevents killed writers leaving a final
    # truncated cache that would poison a deterministic retry.
    C.immutable_npz(path, arrays)


def cache_arrays(ordered):
    return {field: np.stack([item["features"][field] for item in ordered]) if ordered
            else np.empty((0, *shape), dtype=np.uint8 if field == "mask" else np.float32)
            for field, shape in CACHE_SHAPES.items()}


def _verify_source(record, spec):
    if record.get("identity") != C.fingerprint(spec) or any(record.get(k) != v for k, v in spec.items()):
        raise ValueError("source record identity mismatch")
    if record.get("complete") is not True or record.get("source") != "selfplay":
        raise ValueError("only complete own-self-play source games can be reused")
    verify_terminal(record)
    cache = resolve(record["cache_npz"])
    if C.sha256(cache) != record["cache_sha256"]:
        raise ValueError("source cache changed")
    return record


def source_game(job, config, identity, deadline):
    """Generate/reuse one complete frozen-cp2 game; never choose reserve slots."""
    C.check_stop(deadline)
    spec = source_spec(job, config, identity)
    destination = C.RUN / "sources" / f"{job['source_id']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return _verify_source(json.loads(destination.read_text()), spec)
    directory = destination.parent / f"{job['source_id']}.attempts"
    directory.mkdir(exist_ok=True)
    previous = _attempts(directory)
    successful_attempts = [row for row in previous if row.get("status") == "ok"]
    if len(successful_attempts) > 1:
        raise ValueError("duplicate completed source attempts")
    if successful_attempts:
        record = _verify_source(successful_attempts[0], spec)
        C.immutable_json(destination, record)
        return record
    if any(row.get("identity") != C.fingerprint(spec) for row in previous):
        raise ValueError("prior source attempt belongs to different inputs")
    if sum(row.get("status") == "error" for row in previous) >= config["max_error_attempts"]:
        raise RuntimeError("source exhausted bounded error attempts; no reserve substitution allowed")
    number = _next_attempt(directory)
    result_path = directory / f"attempt-{number:03d}.json"
    record = {"identity": C.fingerprint(spec), **spec, "source": "selfplay", "status": "incomplete",
              "complete": False, "terminal": False, "score": None, "attempt": number, "started_at": now()}
    C.immutable_json(directory / f"attempt-{number:03d}-start.json", record)
    limit = config["source_timeout_seconds"]
    task = _task(job, "source", limit, deadline, source_id=job["source_id"], result_path=relative(result_path), start_record=record)
    started, cpu_started = time.monotonic(), time.process_time()
    task_deadline = min(deadline, started+limit)
    actions, candidates, skipped = [], {}, []
    anchors, seen = {0: [], 1: []}, {0: 0, 1: 0}
    reservoir_seeds = {t: reservoir_seed(config["anchor_seed"], job["source_id"], t) for t in (0, 1)}
    reservoir_rng = {t: np.random.default_rng(reservoir_seeds[t]) for t in (0, 1)}
    policy_rng = {t: np.random.default_rng(job["policy_seeds"][t]) for t in (0, 1)}
    exclusions = job.get("exclusion_map", {})
    env = None
    try:
        model = _model(config)
        env = EngineEnv(action_version="macro_v2", response_timeout=config["response_timeout_seconds"])
        _budget(env, config, deadline, task_deadline)
        state = env.reset(job["map"], job["seed"])
        while not state.terminal:
            _budget(env, config, deadline, task_deadline)
            if len(actions) >= config["max_decisions"]:
                raise TimeoutError("source decision cap")
            actor = actor_info(state)
            encoded = encode_state(model, state)
            key = f"{actor['type']}:{state.player}"
            candidate = key not in candidates and 37 in state.legal_actions
            if candidate and encoded["input_hash"] in exclusions:
                skipped.append({"source_id": job["source_id"], "decision_index": len(actions), "player": state.player,
                                "role": actor["type"], "input_hash": encoded["input_hash"],
                                "matches": exclusions[encoded["input_hash"]]})
                candidate = False
            seen[state.player] += 1
            slot = (reservoir_index(reservoir_rng[state.player], seen[state.player])
                    if job["split"] == "training" else None)
            if candidate or slot is not None:
                features = checked_features(encoded, state)
                saved = {"metadata": position_metadata(state, actions, features), "features": features}
                if candidate:
                    candidates[key] = saved
                if slot is not None:
                    if slot == len(anchors[state.player]):
                        anchors[state.player].append(saved)
                    else:
                        anchors[state.player][slot] = saved
            action = choose_policy_action(encoded["prior"], state.legal_actions, policy_rng[state.player], config["source_temperature"])
            actions.append(action)
            state = env.transition(state, action)
            C.release(env, state)
        proof = terminal_proof(state)
        positions = {item["metadata"]["decision_index"]: item for item in
                     [*candidates.values(), *anchors[0], *anchors[1]]}
        ordered = [positions[index] for index in sorted(positions)]
        indexes = {item["metadata"]["decision_index"]: i for i, item in enumerate(ordered)}
        def metadata(item):
            return {**item["metadata"], "source_id": job["source_id"],
                    "feature_index": indexes[item["metadata"]["decision_index"]]}
        arrays = cache_arrays(ordered)
        cache = destination.with_suffix(".npz")
        _write_npz(cache, arrays)
        record.update(status="ok", complete=True, **proof, actions=actions, actions_sha256=C.fingerprint(actions),
                      root_candidates={key: metadata(item) for key, item in sorted(candidates.items())},
                      unavailable_slots=[f"{role}:{player}" for role in ROLES for player in (0, 1)
                                         if f"{role}:{player}" not in candidates],
                      anchor_candidates={str(t): [metadata(item) for item in sorted(anchors[t], key=lambda x: x["metadata"]["decision_index"])] for t in (0, 1)},
                      skipped_input_duplicates=skipped, actor_positions_seen={str(t): seen[t] for t in (0, 1)},
                      anchor_reservoir={"capacity_per_team": RESERVOIR_CAPACITY, "rng_seeds": {str(t): reservoir_seeds[t] for t in (0, 1)},
                          "derivation": "uint64(first16hex(SHA256(compact_sorted_JSON({anchor_seed,source_id,player})))); independent Algorithm-R streams"},
                      cache_npz=relative(cache), cache_sha256=C.sha256(cache), cache_positions=[metadata(item) for item in ordered],
                      source_json=relative(destination), decisions=len(actions), ended_at=now(),
                      elapsed_seconds=time.monotonic()-started, worker_cpu_seconds=time.process_time()-cpu_started)
        C.immutable_json(result_path, record)
        C.immutable_json(destination, record)
        return record
    except BaseException as error:
        record.update(status="incomplete" if _interrupted(error, deadline) else "error", complete=False, terminal=False,
                      score=None, error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc(), actions=actions, actions_sha256=C.fingerprint(actions), decisions=len(actions),
                      ended_at=now(), elapsed_seconds=time.monotonic()-started, worker_cpu_seconds=time.process_time()-cpu_started)
        if not result_path.exists():
            C.immutable_json(result_path, record)
        raise
    finally:
        if env:
            env.close()
        C.clear_task(task)


class RootWorker:
    def __init__(self, root, config, identity, deadline):
        self.root, self.config, self.identity, self.deadline = root, config, identity, deadline
        self.model = _model(config)
        self.infer = make_evaluator(self.model)
        self.env, self.state, self.encoded = None, None, None

    def close(self):
        if self.env:
            self.env.close()
        self.env, self.state, self.encoded = None, None, None

    def restore(self):
        if self.state is not None:
            return self.state
        root, config = self.root, self.config
        task = _task(root, "root_reconstruction", config["source_timeout_seconds"], self.deadline, root_id=root["root_id"])
        task_deadline = min(self.deadline, time.monotonic()+config["source_timeout_seconds"])
        try:
            source_path = resolve(root["source_json"])
            if C.sha256(source_path) != root["source_sha256"]:
                raise ValueError("root source transcript changed")
            source = json.loads(source_path.read_text())
            if source.get("complete") is not True or source.get("experiment_identity") != C.fingerprint(self.identity):
                raise ValueError("root is not from a complete source of this experiment")
            verify_terminal(source)
            if (source["map"] != root["map"] or source["seed"] != root["seed"]
                    or not 0 <= root["decision_index"] < len(source["actions"])):
                raise ValueError("root/source map, seed or prefix differs")
            prefix = source["actions"][:root["decision_index"]]
            if C.fingerprint(prefix) != root["prefix_sha256"]:
                raise ValueError("root hidden-state action prefix changed")
            self.env = EngineEnv(action_version="macro_v2", response_timeout=config["response_timeout_seconds"])
            _budget(self.env, config, self.deadline, task_deadline)
            state = self.env.reset(root["map"], root["seed"])
            for action in prefix:
                _budget(self.env, config, self.deadline, task_deadline)
                state = self.env.transition(state, action)
                C.release(self.env, state)
            actor = actor_info(state)
            if (state_hash(state.raw) != root["state_sha256"] or list(state.legal_actions) != root["legal_actions"]
                    or state.player != root["player"] or actor["type"] != root["role"]
                    or actor["id"] != root["actor_id"] or state.raw["round"] != root["round"]):
                raise ValueError("root deterministic physical/actor reconstruction mismatch")
            encoded = checked_features(encode_state(self.model, state), state)
            if encoded["input_hash"] != root["input_hash"]:
                raise ValueError("root canonical model input changed")
            cache_path = resolve(root["cache_npz"])
            if C.sha256(cache_path) != root["cache_sha256"] or root["cache_sha256"] != source["cache_sha256"]:
                raise ValueError("root feature-cache provenance differs")
            with np.load(cache_path, allow_pickle=False) as cache:
                for field in CACHE_FIELDS:
                    if not np.array_equal(cache[field][root["feature_index"]], encoded[field]):
                        raise ValueError(f"root cached {field} differs from exact reconstruction")
            self.state = state
            self.encoded = encoded
            return state
        finally:
            C.clear_task(task)

    def comparator(self):
        root, config = self.root, self.config
        path = C.PROGRESS / "comparators" / f"{root['root_id']}.json"
        spec = {"experiment_identity": C.fingerprint(self.identity), "root_id": root["root_id"],
                "root_state_sha256": root["state_sha256"], "root_input_hash": root["input_hash"],
                "source_sha256": root["source_sha256"], "search": config["search"],
                "search_seed": 205201 + root["root_ordinal"]}
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("identity") != C.fingerprint(spec) or any(record.get(k) != v for k, v in spec.items()):
                raise ValueError("frozen comparator identity changed")
            return record
        state = self.restore()
        limit = config["source_timeout_seconds"]
        task = _task(root, "comparator", limit, self.deadline, root_id=root["root_id"])
        task_deadline = min(self.deadline, time.monotonic()+limit)
        started = time.monotonic()
        try:
            def evaluate(parent):
                _budget(self.env, config, self.deadline, task_deadline)
                return self.infer(parent)
            def transition(parent, action):
                _budget(self.env, config, self.deadline, task_deadline)
                return self.env.transition(parent, action)
            settings = {k: v for k, v in config["search"].items() if k not in ("root_noise", "root_seed_base")}
            result = search(state, evaluate, transition, 40, config=SearchConfig(**settings),
                            rng=np.random.default_rng(spec["search_seed"]), add_root_noise=True)
            C.release(self.env, state)
            _, value = self.infer(state)
            prior = self.encoded["prior"]
            record = {"identity": C.fingerprint(spec), **spec, "selected_action": int(max(result.visits, key=result.visits.get)),
                      "policy_prior": prior.tolist(), "network_value": float(value), "visits": result.visits,
                      "visit_policy": result.policy.tolist(), "nodes": result.nodes, "max_depth": result.max_depth,
                      "elapsed_seconds": time.monotonic()-started, "completed_at": now()}
            C.immutable_json(path, record)
            return record
        finally:
            C.clear_task(task)

    def cell(self, stage, action, seed):
        root, config = self.root, self.config
        spec = cell_spec(root, stage, action, seed, config, self.identity)
        directory = C.PROGRESS / "cells" / f"{stage}-{root['root_id']}-a{action:02d}-s{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        attempts = _attempts(directory)
        success = successful_cell(attempts, spec)
        if success:
            return success
        if sum(row["status"] == "error" for row in attempts) >= config["max_error_attempts"]:
            return None
        state = self.restore()
        number = _next_attempt(directory)
        result_path = directory / f"attempt-{number:03d}.json"
        record = {"identity": C.fingerprint(spec), **spec, "attempt": number, "started_at": now(),
                  "status": "incomplete", "terminal": False, "score": None}
        C.immutable_json(directory / f"attempt-{number:03d}-start.json", record)
        limit = config["cell_timeout_seconds"]
        task = _task(root, "cell", limit, self.deadline, root_id=root["root_id"],
                     result_path=relative(result_path), start_record=record)
        started, cpu_started = time.monotonic(), time.process_time()
        task_deadline = min(self.deadline, started+limit)
        rng, actions = np.random.default_rng(seed), []
        try:
            while not state.terminal:
                _budget(self.env, config, self.deadline, task_deadline)
                if len(actions) >= config["max_decisions"]:
                    raise TimeoutError("terminal continuation decision cap")
                if not actions:
                    selected = action
                else:
                    prior, _ = self.infer(state)
                    selected = choose_policy_action(prior, state.legal_actions, rng, config["temperature"])
                actions.append(selected)
                state = self.env.transition(state, selected)
                C.release(self.env, self.state, state)
            record.update(status="ok", **terminal_proof(state, root["player"]))
        except BaseException as error:
            record.update(status="incomplete" if _interrupted(error, self.deadline) else "error",
                          error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            interrupted = _interrupted(error, self.deadline)
            if interrupted:
                raise
        finally:
            record.update(actions=actions, actions_sha256=C.fingerprint(actions), decisions=len(actions), ended_at=now(),
                          elapsed_seconds=time.monotonic()-started, worker_cpu_seconds=time.process_time()-cpu_started)
            C.immutable_json(result_path, record)
            C.clear_task(task)
        if record["status"] == "error":
            # A timed-out Java may be unusable; rebuild the exact root before a retry.
            self.close()
        return record if record["status"] == "ok" else None


def root_job(root, stage, config, identity, deadline):
    """One isolated root/stage job; the parent owns global manifests/barriers."""
    if stage not in ("comparator", "selection", "validation"):
        raise ValueError("unknown root collection stage")
    C.check_stop(deadline)
    if identity.get("config", config) != config:
        raise ValueError("collector configuration differs from frozen experiment identity")
    worker = RootWorker(root, config, identity, deadline)
    started, cpu_started = time.monotonic(), time.process_time()
    try:
        if stage == "comparator":
            return worker.comparator()
        if stage == "selection":
            comparator_path = C.PROGRESS / "comparators" / f"{root['root_id']}.json"
            if not comparator_path.exists():
                raise ValueError("PUCT comparator must be frozen before terminal scoring")
            comparator = worker.comparator()  # Checks the immutable comparator identity.
            actions = sorted(root["legal_actions"])
        else:
            actions = sorted(set(root["actions"].values()))
            if set(root["actions"]) != {"cp2", "terminal_head", "puct_head"} or any(a not in root["legal_actions"] for a in actions):
                raise ValueError("validation needs all three frozen legal candidate actions")
        seeds = config[f"{stage}_seeds"]
        completed = []
        for action in actions:
            for seed in seeds:
                for _ in range(config["max_error_attempts"]):
                    C.check_stop(deadline)
                    result = worker.cell(stage, action, seed)
                    if result:
                        completed.append(result)
                        break
        scores = {(cell["action"], cell["seed"]): cell["score"] for cell in completed}
        expected = len(actions)*len(seeds)
        result = {"root_id": root["root_id"], "stage": stage, "expected": expected,
                  "completed": len(scores), "complete": len(scores) == expected,
                  "elapsed_seconds": time.monotonic()-started, "worker_cpu_seconds": time.process_time()-cpu_started}
        if stage == "selection":
            means = {str(action): float(np.mean([scores[action, seed] for seed in seeds]))
                     for action in actions if all((action, seed) in scores for seed in seeds)}
            selected = max(actions, key=lambda a: (means[str(a)], comparator["policy_prior"][a], -a)) if result["complete"] else None
            result.update(action_scores=means, selected_action=selected)
        return result
    finally:
        worker.close()
