"""OPTIONAL own-rule curriculum initialization for a separate bootstrap-v3 run.

This is explicitly teacher-policy distillation, not tabula-rasa AlphaZero and
not benchmark imitation. Both sides are controlled by rules in this file. Only
completed official-engine games supply value labels. The resulting checkpoint
is an initialization artifact; this module never edits a production run or
adds teacher targets to the search-target replay used by learn_v2.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from .env import EngineEnv, EngineState
from .features_v2 import ACTION_DIM, ACTION_VERSION, encode
from .gate_config import training_excluded_maps
from .network_v2 import NetworkConfig, PolicyValueNetV2, policy_value_loss

ROOT = Path(__file__).resolve().parents[1]
TEACHER_VERSION = "own-rules-assault-forage-guard-v1"
STYLES = ("assault", "forage", "guard")
PAIRS = (("assault", "forage"), ("forage", "assault"),
         ("assault", "guard"), ("guard", "assault"),
         ("forage", "guard"), ("guard", "forage"))


@dataclass(frozen=True)
class BootstrapConfig:
    experiment_id: str = "macro40-bootstrap-v3"
    seed: int = 20261111
    maps: tuple[str, ...] = ("tiny", "closeup", "knifefight")
    games: int = 6
    positions_per_game: int = 256
    steps: int = 256
    batch_size: int = 32
    learning_rate: float = 0.0003
    teacher_epsilon: float = 0.02
    game_timeout_seconds: float = 900
    max_decisions: int = 100000
    value_weight: float = 0.25


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def teacher_policy(state: EngineState, style: str, epsilon: float = 0.02) -> np.ndarray:
    """Simple fixed rule priorities, authored solely from our macro interface.

An economic or defensive opening switches to assault after round250 so the
curriculum includes resource acquisition without requiring indefinite stalling.
The teacher gets exactly the same raw research state as the learned agent.
It has no access to opponent code, benchmark data, past outcomes or future RNG.
    """
    if style not in STYLES or not 0 <= epsilon < 1:
        raise ValueError("invalid teacher style or exploration probability")
    if state.terminal or not state.legal_actions:
        raise ValueError("teacher requires a nonterminal position with legal actions")
    if state.raw.get("action_version") != ACTION_VERSION:
        raise ValueError("teacher requires macro_v2")
    raw = state.raw
    actor = next(r for r in raw["robots"] if r["id"] == raw["current_id"])
    is_king = actor["type"] == "RAT_KING"
    phase_style = "assault" if raw["round"] >= 250 else style
    if is_king:
        priorities = ([27, 26, 21, 25, 39, 0] if phase_style == "assault" else
                      [26, 25, 39, 27, 0] if phase_style == "forage" else
                      [26, 27, 20, 23, 25, 0])
    elif float(actor.get("cheese", 0)) >= 10:
        priorities = [18, 17, 39, 21, 20, 0]
    elif phase_style == "assault":
        priorities = [21, 22, 23, 17, 19, 0]
    elif phase_style == "forage":
        priorities = [17, 19, 39, 23, 21, 0]
    else:
        priorities = [20, 23, 17, 21, 0]
    # Retreat is a common survival rule, not a benchmark-specific response.
    if not is_king and actor["hp"] <= 20:
        priorities = [24, *priorities]
    legal = tuple(state.legal_actions)
    if any(a < 0 or a >= ACTION_DIM for a in legal):
        raise ValueError("teacher action schema mismatch")
    chosen = next((action for action in priorities if action in legal), legal[0])
    policy = np.zeros(ACTION_DIM, dtype=np.float32)
    policy[list(legal)] = epsilon / len(legal)
    policy[chosen] += 1-epsilon
    return policy


def generate_game(config: BootstrapConfig, index: int, output: Path) -> dict:
    """Write teacher examples only after a genuine terminal game result."""
    # Two adjacent games exchange styles on the same map and seed.
    map_name = config.maps[(index//2) % len(config.maps)]
    seed = config.seed + index//2
    pair = PAIRS[index % len(PAIRS)]
    action_rng = np.random.default_rng(seed + index*100003)
    reservoir_rng = np.random.default_rng(seed + index*100003 + 1)
    retained, actions = [], []
    started = time.monotonic()
    row = {"source": "scripted_selfplay", "teacher_version": TEACHER_VERSION,
           "game_id": f"teacher-{index:04d}", "map": map_name, "seed": seed,
           "teacher_A": pair[0], "teacher_B": pair[1], "action_version": ACTION_VERSION,
           "policy_target_type": "own-rule-teacher-distribution", "benchmark": False}
    env = EngineEnv(action_version=ACTION_VERSION)
    try:
        state = env.reset(map_name, seed)
        while not state.terminal:
            remaining = config.game_timeout_seconds-(time.monotonic()-started)
            if remaining <= 0 or len(actions) >= config.max_decisions:
                raise TimeoutError("unfinished teacher game receives no value labels")
            env.response_timeout = min(120, remaining)
            policy = teacher_policy(state, pair[state.player], config.teacher_epsilon)
            position_index = len(actions)
            slot = position_index if position_index < config.positions_per_game else int(reservoir_rng.integers(position_index+1))
            if slot < config.positions_per_game:
                observation = encode(state)
                legal = np.zeros(ACTION_DIM, dtype=bool)
                legal[list(state.legal_actions)] = True
                position = {"local": observation.local.astype(np.float16),
                            "global_board": observation.global_board.astype(np.float16),
                            "scalars": observation.scalars, "policy": policy,
                            "legal_mask": legal, "player": np.int8(state.player)}
                if position_index < config.positions_per_game:
                    retained.append(position)
                else:
                    retained[slot] = position
            action = int(action_rng.choice(ACTION_DIM, p=policy.astype(np.float64)/policy.sum(dtype=np.float64)))
            state = env.transition(state, action)
            env.release_except(state)
            actions.append(action)
        if not retained:
            raise RuntimeError("terminal teacher game has no decisions")
        arrays = {key: np.stack([p[key] for p in retained]) for key in retained[0]}
        arrays["value"] = (state.terminal_value_a * np.where(arrays["player"] == 0, 1, -1)).astype(np.float32)
        data_path = output.with_suffix(".npz")
        temporary = data_path.with_suffix(".tmp")
        output.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(data_path)
        row.update(status="ok", terminal_value_a=state.terminal_value_a, winner=state.raw["winner"],
                   rounds=state.raw["round"], retained_positions=len(retained), terminal=state.raw,
                   data_sha256=_sha(data_path))
    except Exception as error:
        row.update(status="error", error=str(error))
    finally:
        env.close()
    row.update(seconds=time.monotonic()-started, decisions=len(actions), actions=actions)
    _json(output, row)
    return row


def _sample(games: list[dict[str, np.ndarray]], batch_size: int, rng) -> dict[str, torch.Tensor]:
    selections = [games[int(index)] for index in rng.integers(len(games), size=batch_size)]
    indices = [int(rng.integers(len(game["value"]))) for game in selections]
    return {key: torch.as_tensor(np.stack([game[key][i] for game, i in zip(selections, indices)]),
                                 dtype=torch.bool if key == "legal_mask" else torch.float32)
            for key in ("local", "global_board", "scalars", "policy", "legal_mask", "value")}


def _measure(model, batch, value_weight: float):
    with torch.inference_mode():
        logits, values = model(batch["local"], batch["global_board"], batch["scalars"], batch["legal_mask"])
        _, metrics = policy_value_loss(logits, values, batch["policy"], batch["value"], value_weight=value_weight)
        target = batch["policy"]
        metrics["teacher_argmax_agreement"] = float((logits.argmax(-1) == target.argmax(-1)).float().mean())
        metrics["teacher_selected_action_probability"] = float(
            logits.softmax(-1).gather(1, target.argmax(-1, keepdim=True)).mean())
        return metrics


def run(config: BootstrapConfig, output: Path) -> dict:
    """Create a fresh optional experiment; never overwrite any existing run."""
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("bootstrap destination must be new/empty; production runs cannot be overwritten")
    if config.games < 1 or config.positions_per_game < 1 or config.steps < 1 or config.batch_size < 1:
        raise ValueError("games, positions, steps and batch size must be positive")
    if config.game_timeout_seconds <= 0 or config.max_decisions < 1 or config.learning_rate <= 0 or config.value_weight < 0:
        raise ValueError("invalid optimization or execution budget")
    if not 0 <= config.teacher_epsilon < 1:
        raise ValueError("teacher epsilon must be in[0,1)")
    # Gate metadata only preserves gradient-data exclusions; evaluation may use all maps.
    gate = json.loads((ROOT / "configs/victory_gate.json").read_text())
    reserved = training_excluded_maps(gate)
    if not config.maps or reserved.intersection(config.maps):
        raise ValueError("bootstrap maps must be nonempty and exclude training-excluded maps (legacy holdout)")
    for name in config.maps:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not (ROOT / "maps" / f"{name}.map26").exists():
            raise ValueError(f"invalid or unavailable map {name!r}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    identity = {"teacher_source_sha256": _sha(Path(__file__)),
                "engine_jar_sha256": _sha(ROOT / "vendor/engine/battlecode26-java-1.2.6.jar"),
                "engine_patch_sha256": _sha(ROOT / "patches/local-research.patch"),
                "java_sources": {p.name: _sha(p) for p in sorted((ROOT / "java/alpha0").glob("*.java"))},
                "maps": {name: _sha(ROOT / "maps" / f"{name}.map26") for name in config.maps}}
    metadata = {"experiment_id": config.experiment_id, "training_stage": "own-rules-curriculum-initialization",
                "training_source": "scripted_selfplay", "teacher_version": TEACHER_VERSION,
                "initialization": "own-rule-policy-distillation; not tabula-rasa", "benchmark_data_used": False,
                "bootstrap_config": asdict(config), "bootstrap_identity": identity}
    _json(output / "config.json", {**asdict(config), "identity": identity})
    model = PolicyValueNetV2(NetworkConfig())
    neutral_metadata = {**metadata, "training_stage": "neutral-initialization-before-bootstrap",
                        "initialization": "random-policy-and-zero-value; no teacher updates yet"}
    torch.save(model.checkpoint(**neutral_metadata, bootstrap_steps=0), output / "neutral-initialization.pt")
    games, rows = [], []
    for index in range(config.games):
        game_path = output / "games" / f"teacher-{index:04d}.json"
        row = generate_game(config, index, game_path)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("actions", "terminal")}), flush=True)
        if row["status"] != "ok":
            _json(output / "summary.json", {"status": "error", "error": row["error"], "completed_games": len(games)})
            raise RuntimeError(f"teacher game failed; no student trained: {row['error']}")
        data_path = game_path.with_suffix(".npz")
        if _sha(data_path) != row["data_sha256"]:
            raise ValueError("teacher shard changed before training")
        with np.load(data_path, allow_pickle=False) as data:
            games.append({key: data[key] for key in data.files})
    # This fixed sample is a fit diagnostic, explicitly NOT held-out evaluation.
    diagnostic_batch = _sample(games, min(256, sum(len(g["value"]) for g in games)), np.random.default_rng(config.seed+1))
    before = _measure(model, diagnostic_batch, config.value_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    model.train()
    for step in range(1, config.steps+1):
        batch = _sample(games, config.batch_size, rng)
        logits, values = model(batch["local"], batch["global_board"], batch["scalars"], batch["legal_mask"])
        loss, metrics = policy_value_loss(logits, values, batch["policy"], batch["value"], value_weight=config.value_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite bootstrap objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step == config.steps or step % 32 == 0:
            with (output / "fit.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, **metrics}, allow_nan=False)+"\n")
    model.eval()
    after = _measure(model, diagnostic_batch, config.value_weight)
    checkpoint = output / "warmstart.pt"
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(model.checkpoint(**metadata, bootstrap_steps=config.steps,
                               bootstrap_games=config.games, bootstrap_positions=sum(len(g["value"]) for g in games),
                               bootstrap_game_hashes=[row["data_sha256"] for row in rows],
                               optimizer_state_dict=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                               numpy_rng=rng.bit_generator.state), temporary)
    temporary.replace(checkpoint)
    summary = {"status": "ok", "experiment_id": config.experiment_id, "teacher_version": TEACHER_VERSION,
               "games": len(games), "decisions": sum(row["decisions"] for row in rows),
               "positions": sum(len(game["value"]) for game in games), "steps": config.steps,
               "diagnostic_scope": "fixed training-data sample; not generalization or benchmark strength",
               "before": before, "after": after, "checkpoint_sha256": _sha(checkpoint),
               "checkpoint": str(checkpoint), "production_activated": False}
    _json(output / "summary.json", summary)
    print(json.dumps(summary, allow_nan=False), flush=True)
    return summary


def run_winner_only(source: Path, output: Path, *, steps: int = 256, batch_size: int = 32,
                    seed: int = 20261111) -> dict:
    """Fit a fresh policy on winners from checksum-verified OWN teacher games.

Filtering uses only terminal outcomes of the teacher curriculum. Benchmark
results and action histories are never opened. The value objective is disabled
and the final value layer is explicitly reset, yielding neutral values for a
later, separately configured MCTS experiment.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("winner-only destination must be new/empty")
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch size must be positive")
    source_config = json.loads((source / "config.json").read_text())
    reserved = training_excluded_maps(json.loads((ROOT / "configs/victory_gate.json").read_text()))
    if reserved.intersection(source_config["maps"]):
        raise ValueError("source teacher curriculum overlaps training-excluded maps (legacy holdout)")
    games, provenance = [], []
    row_paths = sorted((source / "games").glob("teacher-*.json"))
    if len(row_paths) != source_config["games"]:
        raise ValueError("source curriculum is incomplete")
    started = time.monotonic()
    for path in row_paths:
        row = json.loads(path.read_text())
        terminal = row.get("terminal", {})
        if (row.get("source") != "scripted_selfplay" or row.get("benchmark") is not False or
                row.get("teacher_version") != TEACHER_VERSION or row.get("status") != "ok" or
                terminal.get("terminal") is not True or terminal.get("winner") != row.get("winner") or
                row.get("action_version") != ACTION_VERSION or row.get("map") not in source_config["maps"]):
            raise ValueError("source must be completed own-rule self-play, never benchmark/evaluation data")
        if row["winner"] not in (0, 1):
            raise ValueError("a winner-only curriculum requires a decisive outcome in every source game")
        expected_terminal = 1 if row["winner"] == 0 else -1
        if row["terminal_value_a"] != expected_terminal:
            raise ValueError("source terminal outcome is inconsistent")
        data_path = path.with_suffix(".npz")
        if _sha(data_path) != row["data_sha256"]:
            raise ValueError("source teacher shard checksum mismatch")
        with np.load(data_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        expected_values = expected_terminal * np.where(arrays["player"] == 0, 1, -1)
        if (not np.isin(arrays["player"], (0, 1)).all() or
                not np.array_equal(arrays["value"], expected_values) or
                not np.isfinite(arrays["policy"]).all() or (arrays["policy"] < 0).any() or
                not np.allclose(arrays["policy"].sum(-1), 1, atol=1e-5) or
                np.any(arrays["policy"][~arrays["legal_mask"]])):
            raise ValueError("source labels/policy legality are invalid")
        selected = np.flatnonzero(arrays["player"] == row["winner"])
        if not len(selected):
            raise ValueError("source reservoir contains no winning-team examples")
        games.append({key: value[selected] for key, value in arrays.items()})
        provenance.append({"game_id": row["game_id"], "map": row["map"], "seed": row["seed"],
                           "winner": row["winner"], "source_json_sha256": _sha(path),
                           "source_data_sha256": row["data_sha256"],
                           "source_positions": len(arrays["player"]), "selected_positions": selected.tolist(),
                           "winning_teacher": row["teacher_A"] if row["winner"] == 0 else row["teacher_B"]})
    config = {"experiment_id": "macro40-bootstrap-v3-winners", "source_curriculum": source_config,
              "source_directory": str(source), "source_config_sha256": _sha(source / "config.json"),
              "policy_filter": "only winning-team positions from completed own-rule self-play",
              "value_objective_weight": 0.0, "reset_final_value_head": True,
              "steps": steps, "batch_size": batch_size, "seed": seed,
              "learning_rate": source_config["learning_rate"], "sampling": "uniform game then uniform winning position",
              "fitting_source_sha256": _sha(Path(__file__))}
    output.mkdir(parents=True, exist_ok=True)
    _json(output / "config.json", config)
    _json(output / "source_dataset.json", {"source": "scripted_selfplay", "games": provenance})
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PolicyValueNetV2(NetworkConfig())
    metadata = {"experiment_id": config["experiment_id"], "training_source": "scripted_selfplay-winning-team-only",
                "teacher_version": TEACHER_VERSION, "benchmark_data_used": False,
                "bootstrap_config": config, "policy_target_type": "winner-only own-rule behavior cloning",
                "value_initialization": "zero final layer; value objective disabled",
                "source_dataset_sha256": _sha(output / "source_dataset.json")}
    torch.save(model.checkpoint(**metadata, training_stage="neutral-before-winner-only-bootstrap", bootstrap_steps=0),
               output / "neutral-initialization.pt")
    diagnostic = _sample(games, min(256, sum(len(g["value"]) for g in games)), np.random.default_rng(seed+1))
    before = _measure(model, diagnostic, 0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    model.train()
    for step in range(1, steps+1):
        batch = _sample(games, batch_size, rng)
        logits, values = model(batch["local"], batch["global_board"], batch["scalars"], batch["legal_mask"])
        loss, metrics = policy_value_loss(logits, values, batch["policy"], batch["value"], value_weight=0.0)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite winner-only bootstrap objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step == steps or step % 32 == 0:
            with (output / "fit.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, **metrics}, allow_nan=False)+"\n")
    # The explicit reset is part of the artifact contract, even if a future
    # optimizer or auxiliary objective has touched the value head during fitting.
    torch.nn.init.zeros_(model.value[-1].weight)
    torch.nn.init.zeros_(model.value[-1].bias)
    model.eval()
    after = _measure(model, diagnostic, 0.0)
    if after["value_mean"] != 0 or after["value_saturated_fraction"] != 0:
        raise AssertionError("MCTS-ready value head must be neutral")
    checkpoint = output / "warmstart.pt"
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(model.checkpoint(**metadata, training_stage="winner-only-own-rule-bootstrap",
                               bootstrap_steps=steps, bootstrap_games=len(games),
                               bootstrap_positions=sum(len(g["value"]) for g in games),
                               optimizer_state_dict=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                               numpy_rng=rng.bit_generator.state), temporary)
    temporary.replace(checkpoint)
    summary = {"status": "ok", "experiment_id": config["experiment_id"],
               "method": "winner-only BC from own teacher games; not MCTS targets",
               "source_games": len(games), "source_positions": sum(p["source_positions"] for p in provenance),
               "selected_positions": sum(len(g["value"]) for g in games), "steps": steps,
               "winning_teachers": [p["winning_teacher"] for p in provenance],
               "value_head_reset_to_zero": True, "before": before, "after": after,
               "diagnostic_scope": "fixed training-data sample; not strength or calibration evaluation",
               "seconds": time.monotonic()-started, "checkpoint_sha256": _sha(checkpoint),
               "checkpoint": str(checkpoint), "production_activated": False}
    _json(output / "summary.json", summary)
    print(json.dumps(summary, allow_nan=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New directory, separate from every current training run")
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--positions-per-game", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--maps", default="tiny,closeup,knifefight")
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--teacher-epsilon", type=float, default=0.02)
    parser.add_argument("--game-timeout", type=float, default=900)
    parser.add_argument("--winner-only-from", type=Path,
                        help="Reuse a verified own-teacher curriculum; filter only its winning-team policy examples")
    args = parser.parse_args()
    if args.winner_only_from is not None:
        run_winner_only(args.winner_only_from, args.out, steps=args.steps,
                        batch_size=args.batch_size, seed=args.seed)
        return
    config = BootstrapConfig(seed=args.seed, maps=tuple(args.maps.split(",")), games=args.games,
                             steps=args.steps, positions_per_game=args.positions_per_game,
                             batch_size=args.batch_size, teacher_epsilon=args.teacher_epsilon,
                             game_timeout_seconds=args.game_timeout)
    run(config, args.out)


if __name__ == "__main__":
    main()
