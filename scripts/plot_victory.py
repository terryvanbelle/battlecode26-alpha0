#!/usr/bin/env python3
"""Plot an all-map victory report without inventing scores for unresolved games.

The input is a report emitted by alpha0.victory. This script visualizes that report;
it does not independently establish gate eligibility or rewrite evaluation evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def finite_score(value: Any) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid score {value!r}")
    return float(value)


def pair_values(pair: dict | None) -> dict:
    """A pair's display values; partial results never become complete-pair scores."""
    if pair is None:
        return {"expected": None, "completed": 0, "errors": 0, "complete": False,
                "passed": False, "scores": [None, None, None], "side_counts": [None, None],
                "status": "Not recorded"}
    expected = count(pair.get("expected", 0), "pair expected")
    completed = count(pair.get("completed", 0), "pair completed")
    errors = count(pair.get("errors", 0), "pair errors")
    if completed > expected:
        raise ValueError("Pair completed count exceeds expected count")
    side_counts, side_scores, sides_complete = [], [], True
    for side in ("A", "B"):
        data = pair.get("sides", {}).get(side, {})
        side_expected = count(data.get("expected", 0), f"side {side} expected")
        side_completed = count(data.get("completed", 0), f"side {side} completed")
        if side_completed > side_expected:
            raise ValueError(f"Side {side} completed count exceeds expected count")
        side_counts.append((side_completed, side_expected))
        side_score = finite_score(data.get("mean_score"))
        side_scores.append(side_score)
        sides_complete &= side_expected > 0 and side_completed == side_expected and side_score is not None
    mean = finite_score(pair.get("mean_score"))
    complete = (expected > 0 and completed == expected and errors == 0 and
                sides_complete and mean is not None)
    if complete:
        if sum(x[0] for x in side_counts) != completed or sum(x[1] for x in side_counts) != expected:
            raise ValueError("Pair and seat completion counts disagree")
        weighted_mean = sum(s * c[0] for s, c in zip(side_scores, side_counts)) / completed
        if abs(mean - weighted_mean) > 1e-8:
            raise ValueError("Pair and seat mean scores disagree")
    passed = complete and pair.get("passed") is True
    status = "Pass" if passed else "Complete; fails threshold" if complete else "Unresolved error" if errors else "Incomplete"
    return {"expected": expected, "completed": completed, "errors": errors, "complete": complete,
            "passed": passed, "scores": [mean, *side_scores] if complete else [None, None, None],
            "side_counts": side_counts, "status": status}


def report_view(report: dict, gate: dict | None = None) -> dict:
    stage = report.get("stage")
    if stage not in ("standard", "confirmation"):
        raise ValueError("stage must be standard or confirmation")
    observed = report.get("opponents", {})
    if not isinstance(observed, dict):
        raise ValueError("opponents must be an object")
    if gate is not None:
        if gate.get("schema_version") != 2:
            raise ValueError("Expected all-map gate schema_version 2")
        if report.get("gate_id") != gate.get("gate_id"):
            raise ValueError("Report and gate identities differ")
        opponents = [item["id"] for item in gate["frozen_opponents"]]
        maps = gate["stages"][stage]["maps"]
        minimum = gate["stages"][stage]["min_score_per_map"]
        seat_minimum = gate["stages"][stage]["min_score_each_side_per_map"]
        unexpected = set(observed) - set(opponents)
        unexpected_maps = {m for op in observed.values() for m in op.get("maps", {})} - set(maps)
        if unexpected or unexpected_maps:
            raise ValueError("Report contains opponents or maps outside the frozen gate")
    else:
        opponents = sorted(observed)
        maps = sorted({name for opponent in observed.values() for name in opponent.get("maps", {})})
        minimum, seat_minimum = 0.6, 0.5
    if len(set(opponents)) != len(opponents) or len(set(maps)) != len(maps):
        raise ValueError("Duplicate opponent or map in plotting inventory")
    values = {(opponent, name): pair_values(observed.get(opponent, {}).get("maps", {}).get(name))
              for name in maps for opponent in opponents}
    for key, pair in values.items():
        if pair["passed"] and (pair["scores"][0] < minimum or any(score < seat_minimum for score in pair["scores"][1:])):
            raise ValueError(f"Passing pair below the stated thresholds: {key}")
    expected = count(report.get("expected", 0), "expected")
    completed = count(report.get("completed", 0), "completed")
    if completed > expected:
        raise ValueError("Report completed count exceeds expected")
    if report.get("passed") is True:
        if (not values or completed != expected or expected == 0 or
                report.get("unresolved_errors", 0) != 0 or
                not all(pair["passed"] for pair in values.values()) or
                len(maps) != report.get("required_maps", len(maps)) or
                len(values) != report.get("expected_opponent_map_pairs", len(values))):
            raise ValueError("Passing stage report lacks complete passing opponent-map evidence")
    if report.get("victory_gate_passed") is True and (stage != "confirmation" or report.get("passed") is not True):
        raise ValueError("Full victory requires a passing confirmation report")
    return {"stage": stage, "opponents": opponents, "maps": maps, "values": values,
            "expected": expected, "completed": completed,
            "minimum": minimum, "seat_minimum": seat_minimum,
            "complete_pairs_displayed": sum(p["complete"] for p in values.values()),
            "passed_pairs_displayed": sum(p["passed"] for p in values.values())}


def markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def atomic_text(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".victory-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generate(report: dict, output_dir: Path, *, gate: dict | None = None) -> dict:
    """Write atomic all-maps-{stage}.png and .md files from observed report data."""
    view = report_view(report, gate)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bc26-alpha0-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter
    import numpy as np

    stage, names, maps = view["stage"], view["opponents"], view["maps"]
    n_maps, n_opponents = max(1, len(maps)), max(1, len(names))
    matrix = np.full((3, n_maps, n_opponents), np.nan)
    failed_pairs = []
    for y, name in enumerate(maps):
        for x, opponent in enumerate(names):
            pair = view["values"][(opponent, name)]
            for side_index, score in enumerate(pair["scores"]):
                if score is not None:
                    matrix[side_index, y, x] = score
            if pair["complete"] and not pair["passed"]:
                failed_pairs.append((x, y))
    cmap = LinearSegmentedColormap.from_list("battlecode_scores", ["#b91c1c", "#fbbf24", "#14b8a6"])
    cmap.set_bad("#e2e8f0")
    height = max(8, len(maps) * 0.245 + 4.6)
    fig, axes = plt.subplots(1, 3, figsize=(22, height), sharey=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.16, right=0.965, bottom=0.145, top=0.89, wspace=0.12)
    state = "PASSED" if report.get("passed") is True else "INCOMPLETE" if view["completed"] < view["expected"] else "NOT PASSED"
    victory = "Full victory gate passed" if report.get("victory_gate_passed") is True else "Full victory gate not established"
    fig.suptitle(f"All-map benchmark gate · {stage} · {state}", fontsize=21, fontweight="bold", y=0.985)
    errors = count(report.get("unresolved_errors", 0), "unresolved_errors")
    pending = count(report.get("pending", view["expected"] - view["completed"]), "pending")
    pair_expected = report.get("expected_opponent_map_pairs", len(names) * len(maps))
    fig.text(0.5, 0.965,
             f"{view['completed']:,}/{view['expected']:,} valid games · {pending:,} unresolved cells · {errors:,} unresolved errors\n"
             f"{view['complete_pairs_displayed']:,}/{pair_expected:,} complete opponent–map pairs; {view['passed_pairs_displayed']:,} passed · {victory}",
             ha="center", va="top", fontsize=12, linespacing=1.6)
    fig.text(0.5, 0.925,
             f"Gate {report.get('gate_id', 'not recorded')}  |  checkpoint {str(report.get('checkpoint_sha256', 'not recorded'))[:16]}\n"
             f"Every pair requires ≥{view['minimum']:.0%} score overall and ≥{view['seat_minimum']:.0%} from each seat. Gray = incomplete/error; × = complete pair that fails.",
             ha="center", va="top", fontsize=10, linespacing=1.5)
    titles = ["Both seats · mean score", "Seat A · mean score", "Seat B · mean score"]
    for index, ax in enumerate(axes):
        image = ax.imshow(np.ma.masked_invalid(matrix[index]), vmin=0, vmax=1, cmap=cmap,
                          aspect="auto", interpolation="nearest")
        ax.set_title(titles[index], fontsize=13, pad=12, fontweight="bold")
        ax.set_xticks(range(len(names)), [f"{i + 1:02d}" for i in range(len(names))], fontsize=9)
        ax.xaxis.tick_top()
        ax.tick_params(axis="both", length=0)
        ax.set_yticks(range(len(maps)), maps, fontsize=8)
        ax.set_xticks(np.arange(-0.5, n_opponents, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_maps, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if not maps or not names:
            ax.text(0.5, 0.5, "No opponent–map observations", transform=ax.transAxes,
                    ha="center", va="center", color="#475569", fontsize=11)
    if failed_pairs:
        axes[0].scatter([p[0] for p in failed_pairs], [p[1] for p in failed_pairs], marker="x", s=12, c="#111827", linewidths=0.7)
    cax = fig.add_axes([0.36, 0.108, 0.38, 0.012])
    bar = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=[0, 0.25, 0.5, 0.6, 0.75, 1])
    bar.ax.xaxis.set_major_formatter(PercentFormatter(1))
    bar.set_label("Completed-game score: win = 1, draw = ½, loss = 0", fontsize=10)
    handles = [Patch(facecolor="none", edgecolor="none", label=f"{i + 1:02d}  {name}") for i, name in enumerate(names)]
    if handles:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.015), ncol=3,
                   frameon=False, fontsize=9, handlelength=0, handletextpad=0, columnspacing=3)
    stem = f"all-maps-{stage}"
    png = output_dir / f"{stem}.png"
    with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".victory-", suffix=".png", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        fig.savefig(temporary, dpi=140, facecolor="white")
        os.replace(temporary, png)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)

    lines = [f"# All-map gate: {stage}", "", f"**Stage: {state}. {victory}.**", "",
             f"Gate: `{markdown(report.get('gate_id', 'not recorded'))}`. Checkpoint: `{markdown(report.get('checkpoint_sha256', 'not recorded'))}`.", "",
             f"**{view['completed']:,}/{view['expected']:,} valid games; {pending:,} unresolved cells; {errors:,} unresolved errors.**", "",
             f"Displayed complete opponent–map pairs: **{view['complete_pairs_displayed']:,}/{pair_expected:,}**; passed pairs: **{view['passed_pairs_displayed']:,}**.", "",
             f"![All maps, every opponent, and both seats]({stem}.png)", "",
             "## How to read the chart", "",
             f"- Every opponent–map pair must independently reach **{view['minimum']:.0%}** overall score and **{view['seat_minimum']:.0%}** from **each** seat. Panel averages and averages across maps cannot substitute.",
             "- Only fully completed, error-free opponent–map pairs have colored score cells in any of the three matrices. Gray means incomplete, missing, or errored evidence; it is not a zero or a loss.",
             "- A colored zero is a measured score of zero from completed games. A × marks a completed pair that failed its threshold; the seat matrices show side-specific weaknesses.",
             "- Scores count wins as 1, draws as 0.5, and losses as 0. Game failures stay separate. The report records unresolved errors; earlier recovered infrastructure attempts do not become losses.",
             "- Standard and confirmation require the same checkpoint and evaluation configuration. A passing standard stage alone does not establish full victory. This plot reproduces the supplied report's verdict and is not an independent gate verifier.",
             "- Confirmation uses fresh seeds on the same maps; those maps are not claimed to be unseen. The small number of seed pairs per map does not establish simultaneous statistical superiority across all comparisons.", ""]
    if gate is None:
        lines += ["No gate inventory was supplied. Rows and columns show only names present in the report; absent inventory entries cannot be inferred. Supply `--gate configs/victory_gate.json` for the complete frozen layout.", ""]
    lines += ["## Opponent key and coverage", "", "| Key | Opponent | Complete pairs | Passed pairs | Recorded valid games | Unresolved errors |", "|---:|---|---:|---:|---:|---:|"]
    for index, opponent in enumerate(names):
        pairs = [view["values"][(opponent, name)] for name in maps]
        lines.append(f"| {index + 1:02d} | `{markdown(opponent)}` | {sum(p['complete'] for p in pairs)}/{len(maps)} | {sum(p['passed'] for p in pairs)}/{len(maps)} | {sum(p['completed'] for p in pairs)} | {sum(p['errors'] for p in pairs)} |")
    lines += ["", "## Every opponent–map pair", ""]
    for opponent in names:
        lines += [f"### {markdown(opponent)}", "", "| Map | Valid / expected | Score | Seat A score (valid/expected) | Seat B score (valid/expected) | Errors | Status |", "|---|---:|---:|---:|---:|---:|---|"]
        for name in maps:
            pair = view["values"][(opponent, name)]
            scores = [f"{x:.1%}" if x is not None else "—" for x in pair["scores"]]
            counts = [f"{x[0]}/{x[1]}" if x is not None else "not recorded" for x in pair["side_counts"]]
            expected_text = str(pair["expected"]) if pair["expected"] is not None else "not recorded"
            lines.append(f"| `{markdown(name)}` | {pair['completed']}/{expected_text} | {scores[0]} | {scores[1]} ({counts[0]}) | {scores[2]} ({counts[1]}) | {pair['errors']} | {pair['status']} |")
        lines.append("")
    atomic_text(output_dir / f"{stem}.md", "\n".join(lines))
    return {"stage": stage, "png": str(png), "markdown": str(output_dir / f"{stem}.md"),
            "maps": len(maps), "opponents": len(names),
            "complete_pairs_displayed": view["complete_pairs_displayed"],
            "passed_pairs_displayed": view["passed_pairs_displayed"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate", type=Path, default=ROOT / "configs/victory_gate.json")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    gate = json.loads(args.gate.read_text())
    print(json.dumps(generate(report, args.output_dir, gate=gate), indent=2))


if __name__ == "__main__":
    main()
