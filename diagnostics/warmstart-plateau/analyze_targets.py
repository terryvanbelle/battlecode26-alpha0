#!/usr/bin/env python3
"""Read-only audit of own self-play targets and fixed checkpoint predictions.

Writes targets.json beside this script. Never runs an engine, trains a model,
or reads benchmark examples. Run from the repository with its project Python.
"""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alpha0.learn_v2 import load_checkpoint

ARRAY_KEYS = ("local", "global_board", "scalars", "policy", "legal_mask", "player")


def array_fingerprints(arrays):
    """Hash source-equivalent dtype, shape and bytes; replay compresses player to int8."""
    result = {}
    for key in ARRAY_KEYS:
        value = np.ascontiguousarray(arrays[key], dtype=np.int64 if key == "player" else None)
        result[key] = {"dtype": value.dtype.str, "shape": list(value.shape),
                       "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    return result


class AuditInputs:
    """Pinned complete game arrays from original shards or checkpoint-2 replay."""

    def __init__(self, run, pins_path=None):
        self.run = Path(run)
        self.pins_path = pins_path or Path(__file__).with_name("inputs.json")
        self.pins = json.loads(Path(self.pins_path).read_text())
        self.replay_index = None

    def load(self, path, *, replay_only=False):
        row = json.loads(path.read_text())
        pin = self.pins["games"][row["game_id"]]
        if sha256(path) != pin["json_sha256"] or row["data_sha256"] != pin["original_npz_sha256"]:
            raise ValueError("game transcript/source checksum mismatch")
        shard = path.with_suffix(".npz")
        provenance = {"game_id": row["game_id"], "json_sha256": sha256(path),
                      "original_npz_sha256": row["data_sha256"]}
        if shard.exists() and not replay_only:
            if sha256(shard) != row["data_sha256"]:
                raise ValueError("original game shard checksum mismatch")
            with np.load(shard, allow_pickle=False) as data:
                if set(data.files) != set(ARRAY_KEYS):
                    raise ValueError("unexpected original shard arrays")
                arrays = {key: data[key].copy() for key in ARRAY_KEYS}
            provenance.update(source="original_game_npz", loaded_path=str(shard.relative_to(self.run)),
                              loaded_sha256=row["data_sha256"])
        else:
            if self.replay_index is None:
                checkpoint = self.run / self.pins["checkpoint"]
                if sha256(checkpoint) != self.pins["checkpoint_sha256"]:
                    raise ValueError("pinned replay checkpoint checksum mismatch")
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if payload["replay_path"] != self.pins["replay_path"]:
                    raise ValueError("checkpoint replay path mismatch")
                index_path = self.run / payload["replay_path"] / "index.json"
                if sha256(index_path) != self.pins["replay_index_sha256"]:
                    raise ValueError("replay index checksum mismatch")
                self.replay_index = json.loads(index_path.read_text())
            matches = [entry for entry in self.replay_index["games"] if entry["game_id"] == row["game_id"]]
            if len(matches) != 1:
                raise ValueError("game absent or duplicated in pinned replay")
            entry = matches[0]
            expected_metadata = {"source": "selfplay", "map": row["map"], "seed": row["seed"]}
            if (entry["metadata"] != expected_metadata or entry["terminal_value_a"] != row["terminal_value_a"]
                    or entry["positions"] != row["retained_positions"]
                    or entry["positions_seen"] != row["retained_positions"]
                    or entry["file"] != pin["replay_file"] or entry["sha256"] != pin["replay_sha256"]):
                raise ValueError("replay metadata/count/source mismatch")
            shard = self.run / self.pins["replay_path"] / entry["file"]
            if sha256(shard) != pin["replay_sha256"]:
                raise ValueError("replay shard checksum mismatch")
            with np.load(shard, allow_pickle=False) as data:
                if set(data.files) != set(ARRAY_KEYS) | {"value"}:
                    raise ValueError("unexpected replay shard arrays")
                arrays = {key: data[key].copy() for key in ARRAY_KEYS}
                expected = row["terminal_value_a"] * np.where(arrays["player"] == 0, 1, -1)
                if not np.array_equal(data["value"], expected):
                    raise ValueError("replay terminal labels mismatch")
            arrays["player"] = arrays["player"].astype(np.int64)
            provenance.update(source="checkpoint_replay", loaded_path=str(shard.relative_to(self.run)),
                              loaded_sha256=pin["replay_sha256"],
                              replay_index_sha256=self.pins["replay_index_sha256"])
        if any(len(value) != row["retained_positions"] for value in arrays.values()):
            raise ValueError("array position count mismatch")
        if array_fingerprints(arrays) != pin["source_arrays"]:
            raise ValueError("arrays differ from verified original source")
        provenance["source_equivalent_arrays_verified"] = True
        return row, arrays, provenance

ACTION_NAMES = ["hold", "move_n", "move_ne", "move_e", "move_se", "move_s", "move_sw", "move_w", "move_nw",
    "build_n", "build_ne", "build_e", "build_se", "build_s", "build_sw", "build_w", "build_nw",
    "gather_cheese", "return_cheese", "claim_mine", "guard_king", "assault_king", "hunt_rat", "hunt_cat",
    "retreat", "king_relocate", "spawn_economy", "spawn_assault", "rat_trap", "cat_trap", "clear_own_trap",
    "form_king", "carry_enemy", "rescue_ally", "throw_damage", "throw_transport", "drop_safe", "powered_bite",
    "clear_route", "cooperate_economy"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values):
    total = float(np.sum(values))
    return [{"id": int(i), "name": ACTION_NAMES[i], "fraction": float(values[i] / total)}
            for i in np.argsort(-values) if values[i] > 0] if total else []


def choose_stratified(scalars, players, game_id, limit=32):
    # Deterministic without-replacement draws within king/baby × team strata.
    seed = int.from_bytes(hashlib.sha256(("target-audit-v1:" + game_id).encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero((scalars[:, 6] > 0.5) == king)
              for king in (False, True)]
    groups = [g[players[g] == player] for g in groups for player in (0, 1)]
    groups = [g for g in groups if len(g)]
    budget = min(limit, len(players))
    counts = np.ones(len(groups), dtype=int)
    sizes = np.array([len(g) for g in groups])
    while counts.sum() < budget:
        deficit = budget * sizes / sizes.sum() - counts
        deficit[counts >= sizes] = -np.inf
        counts[int(np.argmax(deficit))] += 1
    selected, weights = [], []
    for group, count in zip(groups, counts):
        selected.extend(rng.choice(group, count, replace=False).tolist())
        weights.extend([len(group) / count / len(players)] * count)
    order = np.argsort(selected)
    return np.asarray(selected)[order], np.asarray(weights)[order]


def mask_groups(metadata):
    n = len(metadata["iteration"])
    groups = {"all": np.ones(n, bool)}
    for iteration in (1, 2):
        base = metadata["iteration"] == iteration
        groups[f"iteration_{iteration}"] = base
        for role in ("baby", "king"):
            groups[f"iteration_{iteration}/{role}"] = base & (metadata["role"] == role)
        for match in ("current_only", "historical"):
            groups[f"iteration_{iteration}/{match}"] = base & (metadata["match"] == match)
    for field in ("role", "match", "map"):
        for value in np.unique(metadata[field]):
            groups[f"{field}/{value}"] = metadata[field] == value
    return {name: mask for name, mask in groups.items() if mask.any()}


def summarize_targets(targets, legal, mask):
    policy = targets[mask].astype(np.float64)
    entropy = -(policy * np.log(np.maximum(policy, 1e-30))).sum(-1)
    support = (policy > 0).sum(-1)
    counts = np.bincount(policy.argmax(-1), minlength=40)
    return {"positions": len(policy), "mean_legal_actions": float(legal[mask].sum(-1).mean()),
            "mean_support": float(support.mean()), "support_percentiles": np.percentile(support, [0, 10, 50, 90, 100]).tolist(),
            "mean_entropy_nats": float(entropy.mean()), "mean_effective_support": float(np.exp(entropy).mean()),
            "mean_top_mass": float(policy.max(-1).mean()), "one_hot_fraction": float((support == 1).mean()),
            "argmax_distribution": distribution(counts), "visit_mass_distribution": distribution(policy.sum(0))}


def inference_metrics(policy, probability_win, targets, outcome, weights, mask):
    p, t, q, y, w = policy[mask], targets[mask], probability_win[mask], outcome[mask], weights[mask]
    w = w / w.sum()
    mean = lambda value: float(np.sum(w * value))
    greedy = p.argmax(-1)
    target_greedy = t.argmax(-1)
    entropy_t = -(t * np.log(np.maximum(t, 1e-30))).sum(-1)
    ce = -(t * np.log(np.maximum(p, 1e-30))).sum(-1)
    hit = t[np.arange(len(t)), greedy]
    result = {"sample_positions": len(p), "ce_visit_target": mean(ce), "kl_target_to_policy": mean(ce - entropy_t),
              "argmax_target_agreement": mean(greedy == target_greedy),
              "argmax_in_target_maximizers": mean(np.isclose(hit, t.max(-1))),
              "target_mass_on_greedy": mean(hit), "policy_entropy_nats": mean(-(p * np.log(np.maximum(p, 1e-30))).sum(-1)),
              "policy_top_probability": mean(p.max(-1)),
              "greedy_action_distribution": distribution(np.bincount(greedy, weights=w, minlength=40)),
              "outcome_mean": mean(y), "predicted_win_mean": mean(q), "value_brier": mean((q - y) ** 2),
              "value_bce": mean(-(y * np.log(np.maximum(q, 1e-30)) + (1-y) * np.log(np.maximum(1-q, 1e-30)))),
              "value_mean": mean(2*q-1), "value_abs_mean": mean(np.abs(2*q-1)),
              "value_saturated_fraction": mean(np.abs(2*q-1) > .95)}
    for label, selected in (("winner", y == 1), ("loser", y == 0)):
        if selected.any():
            result[f"predicted_win_mean_on_{label}_positions"] = float(np.sum(w[selected]*q[selected]) / w[selected].sum())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-only", action="store_true", help="Verify/use pinned checkpoint replay even if original shards exist")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("targets.json"))
    options = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run = ROOT / "runs/warmstart-v3"
    inputs = AuditInputs(run)
    rows, provenance, all_policy, all_legal, all_meta = [], [], [], [], []
    sampled, sample_meta, selections, sample_weights = [], [], [], []
    for path in sorted((run / "games").glob("iter-*-game-[0-9][0-9][0-9].json")):
        row = json.loads(path.read_text())
        iteration = int(row["game_id"].split("-")[1])
        if iteration not in (1, 2):
            continue
        assert row["status"] == "ok" and row["source"] == "selfplay"
        row, data, source = inputs.load(path, replay_only=options.replay_only)
        provenance.append(source)
        policy, legal = data["policy"], data["legal_mask"]
        scalars, players = data["scalars"], data["player"]
        assert len(policy) == row["retained_positions"]
        assert np.allclose(policy.sum(-1), 1) and not np.any(policy[~legal])
        assert np.allclose(policy * 32, np.round(policy * 32))
        meta = {"iteration": np.full(len(policy), iteration), "role": np.where(scalars[:, 6] > .5, "king", "baby"),
                "match": np.full(len(policy), "historical" if row.get("opponent_checkpoint") else "current_only"),
                "map": np.full(len(policy), row["map"]), "game": np.full(len(policy), row["game_id"]),
                "outcome": (row["terminal_value_a"] * np.where(players == 0, 1, -1) + 1) / 2,
                "player": players,
                "roster_size": 5*(scalars[:, 20]+scalars[:, 21])+100*(scalars[:, 22]+scalars[:, 23]),
                "round": scalars[:, 0]*2000}
        all_policy.append(policy)
        all_legal.append(legal)
        all_meta.append(meta)
        indices, weights = choose_stratified(scalars, players, row["game_id"])
        sampled.append({key: data[key][indices].copy() for key in ("local", "global_board", "scalars", "policy", "legal_mask")})
        sample_meta.append({key: value[indices] for key, value in meta.items()})
        sample_weights.append(weights)
        selections.append({"game_id": row["game_id"], "indices": indices.tolist(), "population_weights": weights.tolist()})
        rows.append({key: row.get(key) for key in ("game_id", "map", "current_team", "opponent_checkpoint", "winner",
                                                  "rounds", "decisions", "searched_positions", "retained_positions", "search_nodes", "seconds")})
        rows[-1]["iteration"] = iteration
        rows[-1]["executed_action_counts"] = np.bincount(row["actions"], minlength=40).tolist()
        rows[-1]["search_fraction_all_decisions"] = row["searched_positions"] / row["decisions"]
        rows[-1]["current_team_won"] = row["winner"] == row["current_team"]
    targets, legal = np.concatenate(all_policy), np.concatenate(all_legal)
    meta = {key: np.concatenate([part[key] for part in all_meta]) for key in all_meta[0]}
    sample = {key: np.concatenate([part[key] for part in sampled]) for key in sampled[0]}
    smeta = {key: np.concatenate([part[key] for part in sample_meta]) for key in sample_meta[0]}
    weights = np.concatenate(sample_weights)
    exact = {name: summarize_targets(targets, legal, mask) for name, mask in mask_groups(meta).items()}
    horizon = {}
    for iteration in (1, 2):
        selected = meta["iteration"] == iteration
        horizon[str(iteration)] = {"mean_roster_size": float(meta["roster_size"][selected].mean()),
            "fraction_roster_larger_than_depth12": float((meta["roster_size"][selected] > 12.1).mean()),
            "median_recorded_round": float(np.median(meta["round"][selected])),
            "fraction_positions_at_round80_or_earlier": float((meta["round"][selected] <= 80.01).mean()),
            "interpretation": "Roster includes all babies/kings, including carried/thrown robots; it is a horizon proxy, not an exact count of eligible turns."}
    totals = {}
    for iteration in (1, 2):
        selected = [row for row in rows if row["iteration"] == iteration]
        result = {key: sum(row[key] for row in selected) for key in ("decisions", "searched_positions", "retained_positions", "search_nodes", "seconds")}
        result["search_fraction_all_decisions"] = result["searched_positions"] / result["decisions"]
        result["nodes_per_search"] = result["search_nodes"] / result["searched_positions"]
        result["executed_action_distribution"] = distribution(np.sum([row["executed_action_counts"] for row in selected], axis=0))
        result["historical_games"] = sum(bool(row["opponent_checkpoint"]) for row in selected)
        result["historical_current_team_wins"] = sum(row["current_team_won"] for row in selected if row["opponent_checkpoint"])
        totals[str(iteration)] = result
    predictions, checkpoint_provenance = {}, {}
    with torch.inference_mode():
        for number in range(3):
            checkpoint = run / f"checkpoint-{number:04d}.pt"
            model, payload = load_checkpoint(checkpoint)
            model.eval()
            ps, qs = [], []
            for start in range(0, len(weights), 16):
                end = start + 16
                args = [torch.as_tensor(sample[key][start:end], dtype=torch.float32) for key in ("local", "global_board", "scalars")]
                logits, values = model(*args, torch.as_tensor(sample["legal_mask"][start:end], dtype=torch.bool))
                ps.append(logits.softmax(-1).numpy().astype(np.float64))
                qs.append(values.sigmoid().numpy().astype(np.float64))
            predictions[str(number)] = (np.concatenate(ps), np.concatenate(qs))
            checkpoint_provenance[str(number)] = {"sha256": sha256(checkpoint), "iteration": payload["iteration"],
                                                  "training_games": payload["games_total"]}
    groups = mask_groups(smeta)
    metrics = {name: {number: inference_metrics(p, q, sample["policy"], smeta["outcome"], weights, mask)
                     for number, (p, q) in predictions.items()} for name, mask in groups.items()}
    pairs = {}
    for left, right in (("0", "1"), ("1", "2"), ("0", "2")):
        a, b = predictions[left][0], predictions[right][0]
        values = {}
        for name, mask in groups.items():
            w = weights[mask] / weights[mask].sum()
            mean = lambda x: float(np.sum(w * x))
            values[name] = {"argmax_agreement": mean(a[mask].argmax(-1) == b[mask].argmax(-1)),
                           "kl_left_to_right": mean((a[mask] * (np.log(np.maximum(a[mask], 1e-30)) - np.log(np.maximum(b[mask], 1e-30)))).sum(-1)),
                           "mean_total_variation": mean(np.abs(a[mask]-b[mask]).sum(-1) / 2)}
        pairs[f"{left}_to_{right}"] = values
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "scope": "Own completed self-play only; no benchmark examples/traces read",
              "sample_method": "Up to32/game, deterministic without-replacement proportional king/baby x team strata; inverse-stratum weights then equal game weight",
              "sample_positions": len(weights), "sampled_games": len(rows), "target_count_weighting": "Exact retained-position-weighted target counts; no reservoir truncation in these16 games",
              "inference_weighting": "Game-balanced, with exact population correction for sampled strata; group metrics condition these weights",
              "limitations": ["Positions within a game are correlated; independent outcome count is16 games, not the sample size.",
                  "cp1 on iteration1 and cp2 on both iterations are training-data diagnostics; cp1 on iteration2 is temporally out of sample.",
                  "Sparse coverage denominator includes historical-opponent turns that deliberately receive no search targets.",
                  "Recorded targets do not retain per-root depth, Q, or terminal-leaf counts; search horizon effects are mechanistic hypotheses."],
              "source_sha256": sha256(__file__), "input_manifest_sha256": sha256(inputs.pins_path),
              "checkpoint_provenance": checkpoint_provenance, "game_provenance": provenance,
              "iteration_totals": totals, "search_horizon_proxy": horizon, "games": rows, "exact_target_groups": exact,
              "sample_selection": selections, "inference_groups": metrics, "checkpoint_policy_pairs": pairs,
              "elapsed_seconds": time.monotonic() - started}
    output = options.output
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "sample_positions": len(weights), "iterations": totals,
                      "sample_policy_pairs": {k:v["all"] for k,v in pairs.items()},
                      "sample_metrics": metrics["all"], "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
