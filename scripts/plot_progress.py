#!/usr/bin/env python3
"""Regenerate repository progress charts from observed training/evaluation JSONL.

No placeholder measurements are generated. Repeated evaluation cells are deduplicated
within the same checkpoint, stage and evaluation configuration.
Run from any directory: .venv/bin/python scripts/plot_progress.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
import random
import shlex
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    if not path.exists():
        return [], []
    text = path.read_text()
    lines = text.splitlines()
    rows, warnings = [], []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            # A live writer may be partway through its last append.
            if index == len(lines) - 1 and not text.endswith("\n"):
                warnings.append(f"{path.name}: ignored an incomplete final line while the run is writing.")
                continue
            raise ValueError(f"{path}:{index + 1}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index + 1}: expected a JSON object")
        rows.append(row)
    return rows, warnings


def wilson(score_sum: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson bounds for scores in [0,1]; fractional draws give a descriptive interval."""
    if n <= 0:
        return float("nan"), float("nan")
    p = score_sum / n
    if not 0 <= p <= 1:
        raise ValueError("score sum must lie in [0, n]")
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    # Exact endpoints prevent cancellation producing a tiny positive lower bound
    # at p=0 (or an upper bound below 1 at p=1), which breaks errorbar lengths.
    return (0.0 if p == 0 else max(0.0, center - radius),
            1.0 if p == 1 else min(1.0, center + radius))


def evaluation_series(row: dict) -> tuple[str, ...]:
    """Identity shared across checkpoints, without pooling distinct protocols.

    New all-map stages are gate-qualified; their evaluation-config hash also
    includes the frozen gate identity. Legacy rows lacking metadata stay in
    their own series rather than inheriting a newer protocol's identity.
    """
    fields = ("stage", "suite", "evaluation_config_sha256", "gate_id", "gate_config_sha256",
              "benchmark_suite_id", "panel_fingerprint_sha256", "engine_version", "engine_jar_sha256", "engine_patch_sha256",
              "bridge_sha256", "action_schema", "observation_schema", "policy", "search_mode")
    identity = row.get("identity") or {}
    configuration = identity.get("evaluation_config") or {}
    all_map_gate = configuration.get("all_map_gate")
    return tuple(json.dumps(value, sort_keys=True, separators=(",", ":"))
                 for value in (*[row.get(field) for field in fields], all_map_gate))


def series_label(row: dict) -> str:
    stage = str(row.get("stage") or "legacy/unlabelled")
    configuration = row.get("evaluation_config_sha256")
    return f"{stage} [{str(configuration)[:8]}]" if configuration else stage


def clean_rows(training: list[dict], evaluations: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    warnings = []
    by_iteration = {}
    for row in training:
        if "iteration" not in row:
            raise ValueError("training row lacks iteration")
        by_iteration[int(row["iteration"])] = row
    if len(by_iteration) < len(training):
        warnings.append(f"Training: kept the latest row for {len(training) - len(by_iteration)} repeated iterations.")
    cells = {}
    for original in evaluations:
        row = dict(original)
        required = ("iteration", "checkpoint", "opponent", "map", "seed", "side", "status")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"evaluation row lacks {', '.join(missing)}")
        if row["status"] == "ok" and row.get("score") not in (0, 0.5, 1):
            raise ValueError(f"completed evaluation has invalid score {row.get('score')!r}")
        if row["status"] not in ("ok", "error"):
            raise ValueError(f"unknown evaluation status {row['status']!r}")
        if row["side"] not in ("A", "B"):
            raise ValueError(f"unknown evaluation side {row['side']!r}")
        key = (*tuple(row[key] for key in required[:-1]), row.get("checkpoint_sha256"), evaluation_series(row),
               row.get("map_sha256"), row.get("opponent_source_commit"), row.get("opponent_java_sha256"))
        cells[key] = row
    if len(cells) < len(evaluations):
        warnings.append(f"Evaluation: kept the latest outcome for {len(evaluations) - len(cells)} repeated checkpoint cells; repeats are not extra evidence.")
    return [by_iteration[i] for i in sorted(by_iteration)], list(cells.values()), warnings


def paired_panel_interval(rows: list[dict], panel: list[str], repetitions: int = 2000) -> dict:
    """Bootstrap shared map-seed cells, preserving both seats and opponent correlation.

    A balanced complete intersection is used, so missing sides/errors never acquire
    implicit zero scores. Four cells is a minimal calculation threshold, not a claim
    that such a small benchmark establishes generalization.
    """
    paired = {name: defaultdict(dict) for name in panel}
    for row in rows:
        if row["status"] == "ok" and row["opponent"] in paired:
            paired[row["opponent"]][(row["map"], row["seed"])][row["side"]] = float(row["score"])
    complete = [{key for key, sides in cells.items() if set(sides) == {"A", "B"}} for cells in paired.values()]
    common = set.intersection(*complete) if complete else set()
    result = {"cells": len(common), "minimum_cells": 4, "mean": None, "lower": None, "upper": None}
    if not common:
        return result
    scores = [sum(sum(paired[name][key].values()) / 2 for name in panel) / len(panel)
              for key in sorted(common, key=str)]
    result["mean"] = sum(scores) / len(scores)
    if len(scores) < 4:
        return result
    rng = random.Random(2026)
    samples = sorted(sum(rng.choices(scores, k=len(scores))) / len(scores) for _ in range(repetitions))
    result["lower"] = samples[int(0.025 * (repetitions - 1))]
    result["upper"] = samples[int(0.975 * (repetitions - 1))]
    return result


def summarize(rows: list[dict], panel: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    # Distinct checkpoints, stages and evaluation protocols are separate evidence.
    last_indices = {}
    for index, row in enumerate(rows):
        key = (int(row["iteration"]), str(row["checkpoint"]),
               str(row.get("checkpoint_sha256") or ""), evaluation_series(row))
        grouped[key].append(row)
        last_indices[key] = index
    result = []
    for key, group in sorted(grouped.items()):
        iteration, checkpoint, checkpoint_sha256, series = key
        opponents = {}
        for name in panel + sorted({r["opponent"] for r in group} - set(panel)):
            selected = [r for r in group if r["opponent"] == name]
            ok = [r for r in selected if r["status"] == "ok"]
            score_sum = sum(float(r["score"]) for r in ok)
            n = len(ok)
            opponents[name] = {
                "n": n,
                "errors": len(selected) - n,
                "score": score_sum / n if n else None,
                "lower": wilson(score_sum, n)[0] if n else None,
                "upper": wilson(score_sum, n)[1] if n else None,
                "wins": sum(r["score"] == 1 for r in ok),
                "draws": sum(r["score"] == 0.5 for r in ok),
                "losses": sum(r["score"] == 0 for r in ok),
                "sides": {side: sum(r["side"] == side for r in ok) for side in ("A", "B")},
            }
        # Unknown opponents may be shown, but cannot silently change the fixed panel mean.
        known = [opponents[name]["score"] for name in panel if opponents[name]["n"]]
        result.append({
            "iteration": iteration, "checkpoint": checkpoint, "opponents": opponents,
            "checkpoint_sha256": checkpoint_sha256, "series_key": series,
            "series_label": series_label(group[-1]), "stage": group[-1].get("stage"),
            "evaluation_config_sha256": group[-1].get("evaluation_config_sha256"),
            "last_row_index": last_indices[key],
            "latest_started_at": max(str(row.get("started_at") or "") for row in group),
            "completed": sum(r["status"] == "ok" for r in group),
            "errors": sum(r["status"] == "error" for r in group),
            "observed_opponents": len(known), "panel_size": len(panel),
            "macro_score": sum(known) / len(known) if known else None,
            "full_panel": bool(panel) and len(known) == len(panel),
            "rows": group, "paired": paired_panel_interval(group, panel),
        })
    return result


def number(row: dict, key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def learning_health_series(training: list[dict]) -> dict | None:
    """Return recorded v2 diagnostics; missing values remain gaps, never zeros.

    A generic legacy value_loss is not necessarily BCE. Only an explicitly
    recorded value_bce or declared BCEWithLogits objective is labelled BCE.
    Search coverage is a per-iteration ratio, not a cumulative game metric.
    """
    triggers = ("value_bce", "value_mse", "value_mean", "value_saturated_fraction",
                "replay_games", "searched_positions", "actual_decisions")
    if not any(math.isfinite(number(row, key)) for row in training for key in triggers):
        return None
    series = {key: [number(row, key) for row in training] for key in
              (*triggers, "replay_positions")}
    series["iterations"] = [int(row["iteration"]) for row in training]
    series["value_bce"] = [number(row, "value_bce") if math.isfinite(number(row, "value_bce"))
                           else number(row, "value_loss") if row.get("value_loss_type") == "BCEWithLogits"
                           else float("nan") for row in training]
    series["search_fraction"] = [searched / actual if math.isfinite(searched) and math.isfinite(actual) and actual > 0
                                 else float("nan") for searched, actual in
                                 zip(series["searched_positions"], series["actual_decisions"])]
    return series


def generate(progress: Path, manifest_path: Path) -> dict:
    progress.mkdir(parents=True, exist_ok=True)
    # Matplotlib's default cache may be read-only in a managed executor.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bc26-alpha0-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    train, warnings = read_jsonl(progress / "training.jsonl")
    evals, eval_warnings = read_jsonl(progress / "evaluations.jsonl")
    train, evals, clean_warnings = clean_rows(train, evals)
    warnings += eval_warnings + clean_warnings
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    panel = [b["id"] for b in manifest.get("bots", [])]
    if not panel:
        panel = sorted({r["opponent"] for r in evals})
        warnings.append("No benchmark manifest found; displayed panel is inferred from observed opponents.")
    unknown = sorted({r["opponent"] for r in evals} - set(panel))
    if unknown:
        warnings.append("Opponents outside the frozen panel (excluded from panel means): " + ", ".join(unknown))
    summaries = summarize(evals, panel)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.2, "figure.facecolor": "white",
                         "axes.titleweight": "bold", "savefig.facecolor": "white"})
    colors = {"blue": "#2563eb", "green": "#059669", "orange": "#ea580c", "red": "#dc2626", "purple": "#7c3aed"}
    evidence_series = defaultdict(list)
    for group in summaries:
        evidence_series[group["series_key"]].append(group)
    multiple_series = len(evidence_series) > 1
    series_colors = {key: plt.get_cmap("tab10")(index % 10) for index, key in enumerate(evidence_series)}
    series_indices = {key: index for index, key in enumerate(evidence_series)}

    def group_color(group):
        return series_colors[group["series_key"]] if multiple_series else colors["blue"]

    def empty(ax, message):
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color="#64748b")

    def finish(fig, filename):
        with tempfile.NamedTemporaryFile(dir=progress, suffix=".png", delete=False) as f:
            temp = Path(f.name)
        try:
            fig.savefig(temp, dpi=150, bbox_inches="tight")
            os.replace(temp, progress / filename)
        finally:
            temp.unlink(missing_ok=True)
            plt.close(fig)

    def iteration_axis(ax, iterations=None):
        ax.set_xlabel("Training iteration")
        # min_n_ticks=1 prevents Matplotlib relaxing integer=True for one point.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=1))
        if iterations:
            ax.set_xlim(max(0, min(iterations) - 0.5), max(iterations) + 0.5)
        else:
            ax.set_xlim(0, 1)

    def count_axis(ax, values):
        finite = [value for value in values if math.isfinite(value)]
        ax.set_ylim(0, max(1, math.ceil(max(finite, default=0) * 1.08)))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("Self-play training · observed measurements", fontsize=17, fontweight="bold")
    loss_ax, entropy_ax, sample_ax, speed_ax = axes.flat
    loss_ax.set_title("Policy and value fitting loss"); loss_ax.set_ylabel("Loss")
    entropy_ax.set_title("Policy entropy"); entropy_ax.set_ylabel("Nats")
    sample_ax.set_title("Cumulative self-play data"); sample_ax.set_ylabel("Positions", color=colors["blue"])
    speed_ax.set_title("Self-play throughput"); speed_ax.set_ylabel("Games / hour")
    for ax in axes.flat:
        iteration_axis(ax)
    if train:
        xs = [int(r["iteration"]) for r in train]
        for ax in axes.flat:
            iteration_axis(ax, xs)
        for key, label, color in [("policy_loss", "Policy", colors["blue"]), ("value_loss", "Value", colors["orange"]), ("total_loss", "Total", colors["purple"])]:
            loss_ax.plot(xs, [number(r, key) for r in train], "o-", ms=4, label=label, color=color)
        loss_ax.legend()
        entropy_ax.plot(xs, [number(r, "entropy") for r in train], "o-", ms=4, color=colors["purple"])
        sample_ax.plot(xs, [number(r, "positions_total") for r in train], "o-", ms=4, color=colors["blue"])
        games_ax = sample_ax.twinx()
        games_ax.set_ylabel("Games", color=colors["green"])
        games_ax.plot(xs, [number(r, "games_total") for r in train], "s--", ms=4, color=colors["green"])
        games_ax.grid(False)
        count_axis(sample_ax, [number(r, "positions_total") for r in train])
        count_axis(games_ax, [number(r, "games_total") for r in train])
        speed_ax.plot(xs, [number(r, "games_per_hour") for r in train], "o-", ms=4, color=colors["green"])
    else:
        for ax in axes.flat:
            empty(ax, "No training measurements recorded yet")
        count_axis(sample_ax, [])
    finish(fig, "training.png")

    health = learning_health_series(train)
    if health is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        fig.suptitle("Learning health · training-batch diagnostics", fontsize=17, fontweight="bold")
        fig.supxlabel("Observed fitting batches, not held-out calibration. Missing measurements are gaps; search counts describe each iteration.", fontsize=10)
        value_ax, mean_ax, saturation_ax, replay_ax, decisions_ax, coverage_ax = axes.flat
        xs = health["iterations"]
        for ax in axes.flat:
            iteration_axis(ax, xs)

        def health_line(ax, key, label, color, style="o-"):
            values = health[key]
            if not any(math.isfinite(value) for value in values):
                return False
            ax.plot(xs, values, style, ms=5, label=label, color=color)
            return True

        value_ax.set_title("Value fitting: BCE and MSE")
        value_ax.set_ylabel("Training-batch loss")
        measured = health_line(value_ax, "value_bce", "Win-logit BCE", colors["blue"])
        measured |= health_line(value_ax, "value_mse", "Value MSE (−1 to +1 scale)", colors["orange"])
        if measured:
            value_ax.legend(fontsize=8)
            value_ax.set_ylim(bottom=0)
        else:
            empty(value_ax, "Value losses not recorded")

        mean_ax.set_title("Mean predicted value")
        mean_ax.set_ylabel("Training-batch mean · −1 to +1")
        mean_ax.set_ylim(-1.05, 1.05)
        mean_ax.axhline(0, color="#94a3b8", linewidth=0.8, linestyle="--")
        if not health_line(mean_ax, "value_mean", "Mean prediction", colors["purple"]):
            empty(mean_ax, "Mean value not recorded")

        saturation_ax.set_title("Saturated value predictions")
        saturation_ax.set_ylabel("Training-batch fraction with |value| > 0.95")
        saturation_ax.set_ylim(-0.03, 1.03)
        saturation_ax.yaxis.set_major_formatter(PercentFormatter(1))
        if not health_line(saturation_ax, "value_saturated_fraction", "Saturated fraction", colors["red"]):
            empty(saturation_ax, "Saturation not recorded")

        replay_ax.set_title("Current replay buffer")
        replay_ax.set_ylabel("Stored positions", color=colors["blue"])
        positions_recorded = health_line(replay_ax, "replay_positions", "Positions", colors["blue"])
        replay_games_ax = replay_ax.twinx()
        replay_games_ax.set_ylabel("Stored complete games", color=colors["green"])
        games_recorded = health_line(replay_games_ax, "replay_games", "Games", colors["green"], "s--")
        replay_games_ax.grid(False)
        count_axis(replay_ax, health["replay_positions"])
        count_axis(replay_games_ax, health["replay_games"])
        if not positions_recorded and not games_recorded:
            empty(replay_ax, "Replay counts not recorded")

        decisions_ax.set_title("Real decisions and searched positions")
        decisions_ax.set_ylabel("Count in this iteration")
        measured = health_line(decisions_ax, "actual_decisions", "Actual game decisions", colors["blue"])
        measured |= health_line(decisions_ax, "searched_positions", "Positions searched with MCTS", colors["orange"])
        if measured:
            decisions_ax.legend(fontsize=8)
        else:
            empty(decisions_ax, "Search/decision counts not recorded")
        count_axis(decisions_ax, health["actual_decisions"] + health["searched_positions"])

        coverage_ax.set_title("Search coverage")
        coverage_ax.set_ylabel("Searched positions / actual decisions")
        coverage_ax.set_ylim(-0.03, max(1.03, max((v for v in health["search_fraction"] if math.isfinite(v)), default=1) * 1.05))
        coverage_ax.yaxis.set_major_formatter(PercentFormatter(1))
        if not health_line(coverage_ax, "search_fraction", "Per-iteration search fraction", colors["green"]):
            empty(coverage_ax, "Coverage unavailable without both counts")
        finish(fig, "learning-health.png")

    fig, (score_ax, count_ax) = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True, sharex=True)
    fig.suptitle("Frozen benchmark evaluation", fontsize=17, fontweight="bold")
    score_ax.set_title("Equal-weight mean across opponents with completed games")
    score_ax.set_ylabel("Mean game score"); score_ax.set_ylim(-0.03, 1.12)
    score_ax.yaxis.set_major_formatter(PercentFormatter(1))
    count_ax.set_title("Recorded outcomes per checkpoint"); count_ax.set_ylabel("Games")
    for ax in (score_ax, count_ax):
        iteration_axis(ax)
    if summaries:
        for group in summaries:
            x, y = group["iteration"], group["macro_score"]
            if y is not None:
                score_ax.plot(x, y, "o" if group["full_panel"] else "D", ms=7,
                              color=group_color(group) if multiple_series else colors["blue"] if group["full_panel"] else colors["orange"])
                score_ax.annotate(f"n={group['completed']} · {group['observed_opponents']}/{group['panel_size']} bots",
                                  (x, y), xytext=(5 if x == 0 else 0, 10 + 13 * series_indices[group["series_key"]]), textcoords="offset points",
                                  ha="left" if x == 0 else "center", fontsize=8)
        xs = [g["iteration"] for g in summaries]
        for ax in (score_ax, count_ax):
            iteration_axis(ax, xs)
        # No line connects partial panels, which would suggest directly comparable strength.
        if multiple_series:
            for groups in evidence_series.values():
                score_ax.plot([], [], "o", color=group_color(groups[0]), label=groups[0]["series_label"])
            score_ax.plot([], [], "D", color="#64748b", label="Diamond = partial opponent panel")
        else:
            score_ax.plot([], [], "o", color=colors["blue"], label="All panel opponents represented")
            score_ax.plot([], [], "D", color=colors["orange"], label="Partial panel; composition may differ")
        for group in summaries:
            paired = group["paired"]
            if paired["lower"] is not None:
                score_ax.errorbar(group["iteration"], paired["mean"],
                                  yerr=[[paired["mean"] - paired["lower"]], [paired["upper"] - paired["mean"]]],
                                  fmt="s", color=colors["purple"], capsize=4, ms=5)
        if any(g["paired"]["lower"] is not None for g in summaries):
            score_ax.plot([], [], "s", color=colors["purple"], label="Balanced paired-cell mean + bootstrap 95% interval")
        score_ax.legend(loc="upper left" if multiple_series else "lower right", fontsize=8)
        for groups in evidence_series.values():
            label = f" · {groups[0]['series_label']}" if multiple_series else ""
            color = group_color(groups[0])
            count_ax.plot([g["iteration"] for g in groups], [g["completed"] for g in groups], "o-",
                          label="Completed" + label, color=color if multiple_series else colors["green"])
            count_ax.plot([g["iteration"] for g in groups], [g["errors"] for g in groups], "s--",
                          label="Errors / timeouts" + label, color=color if multiple_series else colors["red"])
        count_ax.legend(fontsize=8)
        count_axis(count_ax, [g[key] for g in summaries for key in ("completed", "errors")])
    else:
        empty(score_ax, "No benchmark games recorded yet")
        empty(count_ax, "Completed games and failures will appear here")
        count_axis(count_ax, [])
    finish(fig, "evaluation.png")

    names = panel + unknown
    nrows = max(1, math.ceil(len(names) / 4))
    fig, axes = plt.subplots(nrows, 4, figsize=(15, 3.3 * nrows), constrained_layout=True, squeeze=False)
    fig.suptitle("Per-opponent game score · descriptive 95% Wilson intervals", fontsize=16, fontweight="bold")
    for ax, name in zip(axes.flat, names):
        ax.set_title(name); ax.set_ylim(-0.04, 1.13)
        ax.yaxis.set_major_formatter(PercentFormatter(1)); iteration_axis(ax)
        measured = [g for g in summaries if g["opponents"].get(name, {}).get("n", 0)]
        if measured:
            xs = [g["iteration"] for g in measured]
            iteration_axis(ax, xs)
            for key in evidence_series:
                selected = [g for g in measured if g["series_key"] == key]
                if not selected:
                    continue
                series_xs = [g["iteration"] for g in selected]
                ys = [g["opponents"][name]["score"] for g in selected]
                lo = [max(0.0, y - g["opponents"][name]["lower"]) for y, g in zip(ys, selected)]
                hi = [max(0.0, g["opponents"][name]["upper"] - y) for y, g in zip(ys, selected)]
                ax.errorbar(series_xs, ys, yerr=[lo, hi], fmt="o-", color=group_color(selected[0]),
                            capsize=3, ms=4, label=selected[0]["series_label"])
                for x, y, group in zip(series_xs, ys, selected):
                    stats = group["opponents"][name]
                    ax.annotate(f"n={stats['n']}", (x, y), xytext=(4, 6 + 12 * series_indices[key]),
                                textcoords="offset points", fontsize=8, color=group_color(group))
        else:
            empty(ax, "No completed games")
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    if not names:
        axes.flat[0].set_visible(True); empty(axes.flat[0], "No benchmark panel or results yet")
    if multiple_series:
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=group_color(groups[0]), marker="o", label=groups[0]["series_label"])
                   for groups in evidence_series.values()]
        # Place below the explanatory footer; tight saving includes this artist.
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.005), ncol=2, fontsize=8)
    fig.supxlabel("Score = win 1 / draw ½ / loss 0. Errors excluded and reported separately. Paired sides/maps are correlated; intervals are descriptive.", fontsize=9)
    finish(fig, "opponents.png")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    regeneration = ["python", "scripts/plot_progress.py"]
    for flag, path, default in (("--progress-dir", progress, ROOT / "progress"),
                                ("--manifest", manifest_path, ROOT / "configs/benchmarks.json")):
        resolved = path.resolve()
        if resolved != default:
            shown = resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else resolved
            regeneration.extend((flag, str(shown)))
    lines = ["# Training progress", "", f"Generated from repository JSONL measurements at `{now}`.", "",
             f"Run `{shlex.join(regeneration)}` from the repository root to regenerate these figures and this report.", "",
             "## Latest training measurements", ""]
    if train:
        latest = train[-1]
        lines += ["| Measurement | Observed value |", "|---|---:|"]
        for key in ("iteration", "games_total", "positions_total", "policy_loss", "value_loss", "total_loss", "entropy", "games_per_hour", "wall_seconds"):
            value = latest.get(key)
            text = f"{value:.6g}" if isinstance(value, float) else str(value) if value is not None else "not recorded"
            lines.append(f"| `{key}` | {text} |")
    else:
        lines.append("No training measurements have been recorded. Empty axes are intentional; no synthetic baseline is plotted.")
    lines += ["", "![Training loss, entropy, data volume and throughput](training.png)"]
    if health is not None:
        lines += ["", "## Learning health", "",
                  "These are **training-batch diagnostics, not held-out calibration**. Value BCE fits win logits; value MSE and mean use predictions on the −1 to +1 scale. Saturation is the fraction with |value| > 0.95.", "",
                  "Replay counts describe the current buffer, not cumulative data. Searched positions and actual decisions describe each training iteration; their ratio shows search coverage. Missing measurements remain gaps and are not filled with zero.", "",
                  "![Training-batch value diagnostics, replay size and search coverage](learning-health.png)"]
    lines += ["", "## Latest benchmark checkpoint", ""]
    if summaries:
        latest = max(summaries, key=lambda group: (group["iteration"], group["latest_started_at"], group["last_row_index"]))
        lines += [f"Iteration **{latest['iteration']}**, checkpoint `{markdown_cell(latest['checkpoint'])}`.", "",
                  f"Evaluation series: `{markdown_cell(latest['series_label'])}`. Stages and evaluation configurations are reported separately.", "",
                  f"**{latest['completed']} completed games; {latest['errors']} errors; {latest['observed_opponents']}/{latest['panel_size']} panel opponents represented.**", ""]
        if latest["macro_score"] is not None:
            qualification = "all frozen panel opponents" if latest["full_panel"] else "the observed subset of opponents only"
            lines += [f"Equal-weight mean game score: **{latest['macro_score']:.1%}** across {qualification}. This is not an Elo estimate.", ""]
        paired = latest["paired"]
        if paired["lower"] is not None:
            lines += [f"Balanced paired-cell mean: **{paired['mean']:.1%}**, bootstrap 95% interval **{paired['lower']:.1%}–{paired['upper']:.1%}**, using {paired['cells']} map–seed cells with both sides completed against every panel opponent.", ""]
        else:
            lines += [f"Paired-map bootstrap interval unavailable: {paired['cells']} shared map–seed cells have both sides completed against every panel opponent; at least {paired['minimum_cells']} are required. A small smoke suite is insufficient for a strength claim.", ""]
        lines += ["| Opponent | Completed | W–D–L | Mean score | Descriptive 95% Wilson interval | Errors | Sides A/B |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name, stats in latest["opponents"].items():
            score = f"{stats['score']:.1%}" if stats["n"] else "—"
            interval = f"{stats['lower']:.1%}–{stats['upper']:.1%}" if stats["n"] else "—"
            lines.append(f"| `{markdown_cell(name)}` | {stats['n']} | {stats['wins']}–{stats['draws']}–{stats['losses']} | {score} | {interval} | {stats['errors']} | {stats['sides']['A']}/{stats['sides']['B']} |")
        errors = [r for r in latest["rows"] if r["status"] == "error"]
        if errors:
            lines += ["", "Recorded errors at the latest checkpoint:", ""]
            for row in errors:
                lines.append(f"- `{markdown_cell(row['opponent'])}`, `{markdown_cell(row['map'])}`, seed `{markdown_cell(row['seed'])}`, side {markdown_cell(row['side'])}: {markdown_cell(row.get('error', 'No error detail recorded'))}")
    else:
        lines.append("No benchmark games have been recorded. Historical opponent ratings are selection evidence, not results for this agent.")
    lines += ["", "![Benchmark scores and completion counts](evaluation.png)", "", "![Per-opponent results with uncertainty](opponents.png)", "",
              "## Interpretation", "",
              "- Game score is win = 1, draw = 0.5, loss = 0. Errors/timeouts are excluded from the completed-game score and listed separately; inspect completion counts before comparing checkpoints.",
              "- The panel mean weights opponents equally. A partial panel is labelled and is not comparable to a full panel. Different map/seed or side coverage can also change the score.",
              "- Wilson intervals are descriptive per-game intervals. With draws they use fractional successes. Mirrored sides and repeated maps are correlated, so these are not paired-map uncertainty estimates or evidence of a statistically significant improvement.",
              "- The panel bootstrap resamples complete shared map–seed cells, keeping both sides and every opponent together (2,000 resamples, fixed RNG seed). It uses the balanced intersection of completed cells; this can differ from the all-completed-game mean. Four cells is the calculation minimum, not a guarantee of adequate power. Bootstrap intervals can collapse when every observed cell has the same outcome.",
              "- Distinct checkpoints, stages, panels and evaluation configurations are never pooled. Missing legacy identity is not treated as the current gate. Duplicate cells within the same identity keep their final recorded outcome and do not increase sample size.",
              "- These descriptive panel averages do not establish the all-map victory gate. That gate separately requires every opponent–map pair and both seats to satisfy its standard and confirmation thresholds; averages across opponents or maps cannot substitute.",
              "- Loss reductions describe fitting to self-play targets; benchmark outcomes measure playing performance. Neither a small smoke suite nor self-play win balance establishes strength against the wider field.", ""]
    if warnings:
        lines += ["## Data notices", ""] + [f"- {warning}" for warning in warnings] + [""]
    atomic_text(progress / "README.md", "\n".join(lines))
    summary = {"training_rows": len(train), "evaluation_cells": len(evals),
               "checkpoints": len({(g["iteration"], g["checkpoint"], g["checkpoint_sha256"]) for g in summaries}),
               "evaluation_groups": len(summaries), "warnings": warnings,
               "outputs": ["training.png", "evaluation.png", "opponents.png", "README.md"]}
    if health is not None:
        summary["outputs"].insert(1, "learning-health.png")
    return summary


def atomic_text(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        temp = Path(f.name); f.write(text)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-dir", type=Path, default=ROOT / "progress")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/benchmarks.json")
    args = parser.parse_args()
    print(json.dumps(generate(args.progress_dir, args.manifest), indent=2))


if __name__ == "__main__":
    main()
