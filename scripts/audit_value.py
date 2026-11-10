#!/usr/bin/env python3
"""Score each completed self-play game with its immutable pre-game checkpoint.

This diagnostic never writes training data or loads benchmark trajectories.
Games may subsequently train newer models; the forecast measured here always
comes from the generation checkpoint that existed before that game began.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GAME_NAME = re.compile(r"iter-(\d+)-game-(\d+)\Z")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def targets_from_players(players, terminal_value_a: float) -> np.ndarray:
    players = np.asarray(players)
    if players.ndim != 1 or not np.isin(players, (0, 1)).all():
        raise ValueError("players must be a vector containing only team 0/1")
    if not np.isfinite(terminal_value_a) or terminal_value_a not in (-1, 0, 1):
        raise ValueError("terminal value must be a real completed-game outcome -1/0/1")
    return ((terminal_value_a * np.where(players == 0, 1.0, -1.0)) + 1) / 2


def score_game_logits(logits, players, terminal_value_a: float, bins: int = 10) -> dict:
    logits = np.asarray(logits, dtype=np.float64)
    targets = targets_from_players(players, terminal_value_a)
    if logits.shape != targets.shape or not len(logits) or not np.isfinite(logits).all():
        raise ValueError("each retained position needs one finite pre-game win logit")
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be positive")
    # exp(-logaddexp(0,-x)) remains stable for very confident finite logits.
    probabilities = np.exp(-np.logaddexp(0.0, -logits))
    values, target_values = 2 * probabilities - 1, 2 * targets - 1
    bce = np.logaddexp(0.0, logits) - targets * logits
    indices = np.minimum((probabilities * bins).astype(int), bins - 1)
    reliability = []
    for index in range(bins):
        selected = indices == index
        count = int(selected.sum())
        reliability.append({"lower": index / bins, "upper": (index + 1) / bins, "positions": count,
                            "prediction_sum": float(probabilities[selected].sum()),
                            "outcome_sum": float(targets[selected].sum()),
                            "mean_prediction": float(probabilities[selected].mean()) if count else None,
                            "observed_win_rate": float(targets[selected].mean()) if count else None})
    return {"positions": len(logits), "brier": float(np.mean((probabilities - targets) ** 2)),
            "value_mse": float(np.mean((values - target_values) ** 2)), "bce": float(bce.mean()),
            "neutral_brier": float(np.mean((0.5 - targets) ** 2)),
            "neutral_value_mse": float(np.mean(target_values ** 2)), "neutral_bce": math.log(2),
            "mean_win_probability": float(probabilities.mean()), "mean_value": float(values.mean()),
            "saturated_fraction": float(np.mean(np.abs(values) > 0.95)),
            "positive_labels": int((targets == 1).sum()), "negative_labels": int((targets == 0).sum()),
            "draw_labels": int((targets == 0.5).sum()), "mean_target_win_rate": float(targets.mean()),
            "reliability_bins": reliability}


def equal_game_summary(games: list[dict]) -> dict:
    """Give each game weight 1/G and each position in it weight 1/(G*N_g)."""
    if not games:
        raise ValueError("cannot summarize no completed games")
    keys = ("brier", "value_mse", "bce", "neutral_brier", "neutral_value_mse", "neutral_bce",
            "mean_win_probability", "mean_value", "saturated_fraction", "mean_target_win_rate")
    result = {key: float(np.mean([game[key] for game in games])) for key in keys}
    result.update(games=len(games), positions=sum(game["positions"] for game in games),
                  weighting="uniform games, then uniform retained positions within each game")
    bins = len(games[0]["reliability_bins"])
    if any(len(game["reliability_bins"]) != bins for game in games):
        raise ValueError("cannot combine different reliability bins")
    reliability = []
    for index in range(bins):
        rows = [game["reliability_bins"][index] for game in games]
        weight = sum(row["positions"] / game["positions"] for row, game in zip(rows, games)) / len(games)
        predicted = sum(row["prediction_sum"] / game["positions"] for row, game in zip(rows, games)) / len(games)
        observed = sum(row["outcome_sum"] / game["positions"] for row, game in zip(rows, games)) / len(games)
        reliability.append({"lower": index / bins, "upper": (index + 1) / bins,
                            "positions": sum(row["positions"] for row in rows),
                            "represented_games": sum(row["positions"] > 0 for row in rows),
                            "equal_game_weight": weight,
                            "mean_prediction": predicted / weight if weight else None,
                            "observed_win_rate": observed / weight if weight else None})
    result["reliability_bins"] = reliability
    result["reliability_ece"] = sum(row["equal_game_weight"] * abs(row["mean_prediction"] - row["observed_win_rate"])
                                    for row in reliability if row["equal_game_weight"])
    result["brier_skill_vs_neutral"] = (1 - result["brier"] / result["neutral_brier"]
                                          if result["neutral_brier"] else None)
    return result


def checkpoint_path(row: dict, run: Path) -> Path:
    saved = Path(row["checkpoint"])
    # A copied run can retain the original absolute path in its provenance.
    local = run / saved.name
    if local.exists():
        return local
    if saved.exists():
        return saved
    raise FileNotFoundError(f"missing pre-game checkpoint {saved.name}")


def audit_game(path: Path, run: Path, cache: dict, *, batch_size: int, bins: int) -> dict:
    import torch
    from alpha0.network_v2 import NetworkConfig, PolicyValueNetV2

    raw = path.read_bytes()
    row = json.loads(raw)
    match = GAME_NAME.fullmatch(path.stem)
    if not match or row.get("game_id") != path.stem:
        raise ValueError("game identity does not match its canonical filename")
    if row.get("source") != "selfplay" or row.get("benchmark") or row.get("evaluation"):
        raise ValueError("only completed own self-play is permitted")
    if row.get("status") != "ok" or not row.get("terminal", {}).get("terminal"):
        raise ValueError("game is not a valid complete terminal trajectory")
    if row["terminal"].get("valid_game") is False:
        raise ValueError("engine marked this game invalid")
    winner = row["terminal"]["winner"]
    if winner not in (-1, 0, 1) or row["winner"] != winner:
        raise ValueError("terminal winner is missing or inconsistent")
    if row["terminal_value_a"] != {0: 1, 1: -1, -1: 0}[winner]:
        raise ValueError("terminal label disagrees with the engine winner")
    data_path = path.with_suffix(".npz")
    data_sha = file_hash(data_path)
    if data_sha != row["data_sha256"]:
        raise ValueError("retained game tensor checksum mismatch")
    model_path = checkpoint_path(row, run)
    model_sha = file_hash(model_path)
    if model_sha != row["identity"]["checkpoint_sha256"]:
        raise ValueError("pre-game checkpoint checksum mismatch")
    # Arrays are already encoded, but their producing feature schema and the
    # model forward implementation must retain their recorded meaning.
    recorded = row.get("training_runtime", {}).get("files", {})
    for source in ("alpha0/network_v2.py", "alpha0/network.py", "alpha0/features_v2.py"):
        if source not in recorded or file_hash(ROOT / source) != recorded[source]:
            raise ValueError(f"inference source differs from recorded game runtime: {source}")
    if model_sha not in cache:
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        if (payload.get("model_version") != "dual_scale_v2" or payload.get("feature_version") != 2 or
                payload.get("action_version") != "macro_v2"):
            raise ValueError("this audit supports versioned v2 self-play only")
        model = PolicyValueNetV2(NetworkConfig(**payload["network_config"]))
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        cache[model_sha] = model, payload
    model, payload = cache[model_sha]
    game_iteration = int(match[1])
    if payload["iteration"] != game_iteration - 1:
        raise ValueError("checkpoint is not the generation preceding this game; refusing training leakage")
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("local", "global_board", "scalars", "legal_mask", "player")}
    count = len(arrays["player"])
    if count != row["retained_positions"] or any(len(array) != count for array in arrays.values()):
        raise ValueError("retained-position count mismatch")
    if row.get("opponent_checkpoint") and not np.all(arrays["player"] == row["current_team"]):
        raise ValueError("historical-opponent positions unexpectedly entered current-policy training targets")
    logits = []
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            batch = {key: torch.as_tensor(value[start:start + batch_size],
                     dtype=torch.bool if key == "legal_mask" else torch.float32)
                     for key, value in arrays.items() if key != "player"}
            _, value_logits = model(batch["local"], batch["global_board"], batch["scalars"], batch["legal_mask"])
            logits.append(value_logits.cpu().numpy())
    metrics = score_game_logits(np.concatenate(logits), arrays["player"], row["terminal_value_a"], bins)
    return {"game_id": row["game_id"], "map": row["map"], "seed": row["seed"],
            "game_iteration": game_iteration, "checkpoint_iteration": int(payload["iteration"]),
            "checkpoint": model_path.name, "checkpoint_sha256": model_sha,
            "game_json_sha256": hashlib.sha256(raw).hexdigest(), "game_npz_sha256": data_sha,
            "training_runtime_sha256": row["identity"].get("runtime_sha256"),
            "terminal_value_a": row["terminal_value_a"], "winner": row["winner"],
            "opponent_checkpoint": row.get("opponent_checkpoint"), **metrics}


def run_audit(run: Path, *, batch_size: int = 32, bins: int = 10) -> dict:
    import torch

    run = Path(run).resolve()
    snapshot_started_at = datetime.now(timezone.utc).isoformat()
    paths = sorted(path for path in (run / "games").glob("*.json") if GAME_NAME.fullmatch(path.stem))
    games, excluded, cache = [], [], {}
    for path in paths:
        try:
            row = json.loads(path.read_text())
            if row.get("status") != "ok":
                excluded.append({"game_id": path.stem, "reason": "not_completed", "status": row.get("status")})
                continue
            measured = audit_game(path, run, cache, batch_size=batch_size, bins=bins)
            games.append(measured)
            print(json.dumps({key: measured[key] for key in
                             ("game_id", "checkpoint_iteration", "brier", "value_mse", "bce", "saturated_fraction")}), flush=True)
        except Exception as error:
            excluded.append({"game_id": path.stem, "reason": "invalid_input", "error": str(error)})
    planned = {}
    for path in sorted(run.glob("plan-*.json")):
        plan = json.loads(path.read_text())
        for job in plan["jobs"]:
            match = GAME_NAME.fullmatch(job["game_id"])
            if match:
                planned[job["game_id"]] = int(match[1]) - 1
    groups = defaultdict(list)
    for game in games:
        groups[(game["checkpoint_iteration"], game["checkpoint_sha256"])].append(game)
    summaries = []
    for (generation, checkpoint_sha), rows in sorted(groups.items()):
        summaries.append({"checkpoint_iteration": generation, "checkpoint_sha256": checkpoint_sha,
                          "historical_opponent_games": sum(bool(row["opponent_checkpoint"]) for row in rows),
                          "planned_games": sum(value == generation for value in planned.values()) or None,
                          **equal_game_summary(rows)})
    ids = {game["game_id"] for game in games}
    return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "snapshot_started_at": snapshot_started_at,
            "audit_source_sha256": file_hash(Path(__file__)),
            "inference_environment": {"torch_version": str(torch.__version__), "numpy_version": np.__version__},
            "run_dir": str(run), "method": "prequential own-self-play; each game scored only by its pre-game generation checkpoint",
            "interpretation": "Future games relative to the scored checkpoint; these games may train later checkpoints. Not a permanently held-out final-model test.",
            "metrics": {"brier": "mean squared error of win probability against player-relative terminal win label",
                        "value_mse": "mean squared error on [-1,1] values; four times Brier",
                        "bce": "stable binary cross entropy of raw win logits, in nats",
                        "saturated_fraction": "fraction with abs(predicted value)>0.95",
                        "reliability_bins": "uniform probability bins, equal-game weighted; final bin includes p=1"},
            "weighting": "uniform games, uniform retained searched positions within each game",
            "sample_scope": "retained searched positions only; not every robot turn",
            "bins": bins, "completed_games": len(games), "planned_games": len(planned),
            "pending_or_invalid_game_ids": sorted(set(planned) - ids),
            "excluded": excluded, "checkpoints": summaries, "games": games}


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(value)
    temporary.replace(path)


def plot(report: dict, path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "battlecode-alpha0-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["checkpoints"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle("Value forecasts on subsequent self-play games\nPre-game checkpoints only · equal weight per game", fontsize=15)
    if not rows:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No valid completed games", transform=ax.transAxes, ha="center")
    else:
        generations = [row["checkpoint_iteration"] for row in rows]
        axes[0, 0].plot(generations, [r["brier"] for r in rows], "o-", label="Measured Brier")
        axes[0, 0].plot(generations, [r["neutral_brier"] for r in rows], "--", color="gray", label="Neutral p=0.5")
        axes[0, 0].set(ylabel="Win-probability squared error", title="Brier score · lower is better")
        axes[0, 0].legend()
        axes[0, 1].plot(generations, [r["bce"] for r in rows], "o-", color="#b7791f")
        axes[0, 1].axhline(math.log(2), linestyle="--", color="gray", label="Neutral p=0.5")
        axes[0, 1].set(ylabel="Binary cross entropy (nats)", title="Log loss · lower is better")
        axes[0, 1].legend()
        for row in rows:
            bins = [b for b in row["reliability_bins"] if b["equal_game_weight"] > 0]
            label = f"Checkpoint {row['checkpoint_iteration']} · {row['games']} games"
            axes[1, 0].plot([b["mean_prediction"] for b in bins], [b["observed_win_rate"] for b in bins], "o-", label=label)
        axes[1, 0].plot([0, 1], [0, 1], "--", color="gray")
        axes[1, 0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted win probability",
                       ylabel="Observed terminal win label", title="Reliability · exploratory, correlated positions")
        axes[1, 0].legend(fontsize=8)
        axes[1, 1].plot(generations, [r["mean_value"] for r in rows], "o-", label="Mean value")
        axes[1, 1].plot(generations, [r["saturated_fraction"] for r in rows], "s--", label="Saturated fraction")
        axes[1, 1].axhline(0, color="gray", linewidth=0.7)
        axes[1, 1].set(ylim=(-1.05, 1.05), title="Bias and saturation")
        axes[1, 1].legend()
        for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
            ax.set_xlabel("Pre-game checkpoint iteration")
            ax.set_xticks(generations, [f"{g}\n(n={r['games']})" for g, r in zip(generations, rows)])
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        fig.savefig(temporary, dpi=160)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)


def markdown_report(report: dict) -> str:
    lines = ["# Value calibration on subsequent self-play games", "",
             f"Measured at `{report['generated_at']}` from **{report['completed_games']} completed unique self-play games** "
             f"identified when the input snapshot began at `{report['snapshot_started_at']}`.", "",
             "Each game is scored using the exact checkpoint that generated it, before that game was available for training. "
             "These are future games relative to that checkpoint; they subsequently train newer checkpoints, so this is **not** a permanently held-out test of the final model. "
             "No benchmark data is read. Only retained searched positions are measured, with equal weight per game.", "",
             "| Pre-game checkpoint | Completed / planned games | Positions | Brier | Value MSE | BCE (nats) | Mean value | Saturated |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["checkpoints"]:
        coverage = f"{row['games']} / {row['planned_games']}" if row["planned_games"] else str(row["games"])
        lines.append(f"| {row['checkpoint_iteration']} | {coverage} | {row['positions']:,} | {row['brier']:.4f} | {row['value_mse']:.4f} | {row['bce']:.4f} | {row['mean_value']:.4f} | {100*row['saturated_fraction']:.2f}% |")
    lines += ["", "Brier scores win probabilities against player-relative terminal outcomes; value MSE uses [-1,+1] and is exactly four times Brier. "
              "A neutral p=0.5 forecast has Brier 0.25, value MSE 1 and BCE ln(2)=0.6931 on decisive games. "
              "Saturation means |value|>0.95. Lower Brier/BCE is better. Reliability bins and plots are descriptive: positions from one game are correlated, and a few games cannot establish reliable generalization.", "",
              "| Checkpoint | Mean predicted win probability | Mean terminal win label | Games against older checkpoints |",
              "| ---: | ---: | ---: | ---: |"]
    for row in report["checkpoints"]:
        lines.append(f"| {row['checkpoint_iteration']} | {row['mean_win_probability']:.4f} | {row['mean_target_win_rate']:.4f} | {row['historical_opponent_games']} / {row['games']} |")
    learned = [row for row in report["checkpoints"] if row["checkpoint_iteration"] > 0]
    if learned:
        latest = learned[-1]
        relation = "higher" if latest["brier"] > latest["neutral_brier"] else "lower"
        lines += ["", f"The latest measured checkpoint ({latest['checkpoint_iteration']}) has {relation} Brier error than the neutral baseline "
                  f"on its {latest['games']} completed future games. Its mean predicted win probability is {latest['mean_win_probability']:.4f}, "
                  f"versus a terminal win-label mean of {latest['mean_target_win_rate']:.4f}; "
                  f"{100*latest['saturated_fraction']:.2f}% of predictions are saturated. "
                  "This measures value forecasting, not playing strength. Self-play policies and the mix of historical opponents change between generations, "
                  "so these scores do not isolate a causal effect of training. Historical-opponent games retain only current-policy positions, "
                  "which can shift the label balance; equal-game weighting prevents longer retained trajectories from dominating."]
    lines += ["", "## Per-game measurements", "",
              "| Game | Map | Checkpoint | Positions | Brier | Value MSE | BCE | Mean value | Saturated |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for game in report["games"]:
        lines.append(f"| `{game['game_id']}` | `{game['map']}` | {game['checkpoint_iteration']} | {game['positions']} | {game['brier']:.4f} | {game['value_mse']:.4f} | {game['bce']:.4f} | {game['mean_value']:.4f} | {100*game['saturated_fraction']:.2f}% |")
    lines += ["", f"Pending or invalid planned games: **{len(report['pending_or_invalid_game_ids'])}**. "
              f"Excluded canonical artifacts: **{len(report['excluded'])}**; their reasons are retained in the JSON report. "
              "Attempt journals are never counted as additional games.", "",
              "The JSON report records every game/NPZ/checkpoint SHA-256, generation identity, per-game scores and reliability bins. "
              "It rejects wrong terminal labels, changed inference sources, corrupt tensors, mismatched checkpoints and checkpoints from the same or later training generation.", "",
              "- [Machine-readable measurements](../progress/campaign-v2/value-calibration.json)",
              "- [Calibration plot](../progress/campaign-v2/value-calibration.png)", "",
              "Regenerate without changing training state:", "", "```bash",
              ".venv/bin/python scripts/audit_value.py --run-dir runs/campaign/v2", "```", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/campaign/v2")
    parser.add_argument("--output-json", type=Path, default=ROOT / "progress/campaign-v2/value-calibration.json")
    parser.add_argument("--output-png", type=Path, default=ROOT / "progress/campaign-v2/value-calibration.png")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/v2-value-calibration.md")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or not 1 <= args.bins <= 100:
        parser.error("batch-size/threads must be positive and bins1..100")
    import torch
    torch.set_num_threads(args.threads)
    report = run_audit(args.run_dir, batch_size=args.batch_size, bins=args.bins)
    atomic_text(args.output_json, json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot(report, args.output_png)
    atomic_text(args.report, markdown_report(report))
    print(json.dumps({"completed_games": report["completed_games"], "checkpoint_groups": len(report["checkpoints"]),
                      "excluded": report["excluded"], "output": str(args.output_json)}), flush=True)
    if any(row["reason"] == "invalid_input" for row in report["excluded"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
