#!/usr/bin/env python3
"""Render measured terminal-root-ranking-v4 reports; never run games or SGD.

The exact input report is retained under report-snapshots/<SHA256>.json before
rendering. Incomplete/error cells are coverage, never fabricated score zeroes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SELECTION_SEEDS = [103003, 103013]
VALIDATION_SEEDS = [103021, 103043, 103067, 103091]
THRESHOLDS = {"mean_delta_min": 0.10, "improved_roots_min": 2, "worst_root_delta_min": -0.25}


def count(value, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def score(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite score in [0,1]")
    return float(value)


def close_recorded(record: dict, field: str, expected: float | None) -> None:
    value = record.get(field)
    if value is None and expected is None:
        return
    if (expected is None or type(value) not in (float, int) or not math.isfinite(value)
            or abs(value - expected) > 1e-9):
        raise ValueError(f"Recorded {field} disagrees with paired validation scores")


def report_view(report: dict) -> dict:
    """Validate the report and recompute its small research decision rule."""
    if report.get("experiment_id") != "terminal-root-ranking-v4":
        raise ValueError("Expected terminal-root-ranking-v4 report")
    if report.get("schema_version", 1) != 1:
        raise ValueError("Unsupported root-ranking report schema")
    if report.get("gradient_updates") != 0 or report.get("benchmark_data_used") is not False:
        raise ValueError("This report requires zero SGD updates and no benchmark data")
    roots = report.get("roots", [])
    if not isinstance(roots, list):
        raise ValueError("roots must be a list")
    root_ids = [root["root_id"] for root in roots]
    if len(root_ids) != len(set(root_ids)):
        raise ValueError("Duplicate root identifiers cannot count as distinct roots")
    decision = report.get("decision", {})
    if decision.get("thresholds") != THRESHOLDS:
        raise ValueError("Decision thresholds differ from the precommitted rule")
    views = []
    for root in roots:
        root_id = root["root_id"]
        legal = root.get("legal_actions", [])
        if not legal or any(type(a) is not int or a < 0 for a in legal) or len(set(legal)) != len(legal):
            raise ValueError(f"{root_id}: legal action inventory must be nonempty and unique")
        selection = root.get("selection", {})
        sel_expected = count(selection.get("expected", 0), "selection expected")
        sel_completed = count(selection.get("completed", 0), "selection completed")
        sel_errors = count(selection.get("errors", 0), "selection errors")
        if sel_expected != len(legal) * len(SELECTION_SEEDS) or sel_completed + sel_errors > sel_expected:
            raise ValueError(f"{root_id}: invalid selection coverage")
        means = {int(action): score(value, "selection action mean")
                 for action, value in selection.get("action_scores", {}).items()}
        if set(means) - set(legal) or len(means) * len(SELECTION_SEEDS) > sel_completed:
            raise ValueError(f"{root_id}: selection means lack complete legal-action evidence")
        sel_complete = sel_completed == sel_expected and sel_errors == 0
        if sel_complete and set(means) != set(legal):
            raise ValueError(f"{root_id}: complete selection lacks action scores")
        rollout_action = selection.get("selected_action")
        puct_action = root.get("comparator", {}).get("selected_action")
        if rollout_action is not None and (not sel_complete or rollout_action not in legal):
            raise ValueError(f"{root_id}: rollout choice exists before complete selection")
        if puct_action is not None and puct_action not in legal:
            raise ValueError(f"{root_id}: PUCT selected an illegal action")
        validation = root.get("validation", {})
        expected = count(validation.get("expected_unique_cells", 0), "validation expected")
        completed = count(validation.get("completed_unique_cells", 0), "validation completed")
        errors = count(validation.get("errors", 0), "validation errors")
        if completed + errors > expected:
            raise ValueError(f"{root_id}: invalid validation coverage")
        if validation.get("seeds", VALIDATION_SEEDS) != VALIDATION_SEEDS:
            raise ValueError(f"{root_id}: validation seeds differ from the separate precommitted set")
        if expected:
            if rollout_action is None or puct_action is None:
                raise ValueError(f"{root_id}: validation scheduled before both actions were locked")
            required = len(VALIDATION_SEEDS) * len({rollout_action, puct_action})
            if expected != required:
                raise ValueError(f"{root_id}: unique validation count does not match locked actions")
            if validation.get("rollout_action") != rollout_action or validation.get("puct_action") != puct_action:
                raise ValueError(f"{root_id}: validation actions differ from locked selections")
        rollout_scores = [score(x, "rollout validation score") for x in validation.get("rollout_scores", [])]
        puct_scores = [score(x, "PUCT validation score") for x in validation.get("puct_scores", [])]
        if any(value not in (0, .5, 1) for value in rollout_scores + puct_scores):
            raise ValueError(f"{root_id}: validation outcomes must be terminal loss/draw/win scores")
        if len(rollout_scores) > 4 or len(puct_scores) > 4:
            raise ValueError(f"{root_id}: too many validation scores")
        complete = (sel_complete and expected > 0 and completed == expected and errors == 0
                    and len(rollout_scores) == 4 and len(puct_scores) == 4)
        if validation.get("complete", complete) is not complete:
            raise ValueError(f"{root_id}: validation completion flag disagrees with evidence")
        if complete and rollout_action == puct_action and rollout_scores != puct_scores:
            raise ValueError(f"{root_id}: the same action has conflicting validation outcomes")
        rollout_mean = sum(rollout_scores) / 4 if complete else None
        puct_mean = sum(puct_scores) / 4 if complete else None
        delta = rollout_mean - puct_mean if complete else None
        for field, value in (("rollout_mean", rollout_mean), ("puct_mean", puct_mean), ("delta", delta)):
            close_recorded(validation, field, value)
        views.append(dict(root=root, root_id=root_id, legal=legal, action_scores=means,
                          selection_expected=sel_expected, selection_completed=sel_completed,
                          selection_errors=sel_errors, selection_complete=sel_complete,
                          expected=expected, completed=completed, errors=errors, complete=complete,
                          rollout_action=rollout_action, puct_action=puct_action,
                          rollout_scores=rollout_scores, puct_scores=puct_scores,
                          rollout_mean=rollout_mean, puct_mean=puct_mean, delta=delta))
    complete = bool(views) and report.get("roots_frozen") is True and all(v["complete"] for v in views)
    deltas = [v["delta"] for v in views if v["complete"]]
    mean_delta = sum(deltas) / len(deltas) if complete else None
    improved = sum(delta > 0 for delta in deltas) if complete else None
    worst = min(deltas) if complete else None
    criteria_pass = (complete and mean_delta >= THRESHOLDS["mean_delta_min"]
                     and improved >= THRESHOLDS["improved_roots_min"] and worst >= THRESHOLDS["worst_root_delta_min"])
    integrity_verified = decision.get("integrity_verified") is True
    passes = criteria_pass and integrity_verified
    if (decision.get("complete") is not complete or bool(decision.get("criteria_pass")) != criteria_pass
            or bool(decision.get("passes")) != passes):
        raise ValueError("Recorded decision disagrees with complete paired validation evidence")
    if report.get("complete", complete) is not complete:
        raise ValueError("Report completion flag disagrees with validation evidence")
    for field, value in (("mean_delta", mean_delta), ("improved_roots", improved), ("worst_root_delta", worst)):
        close_recorded(decision, field, value)
    return {"roots": views, "complete": complete, "passes": passes,
            "criteria_pass": criteria_pass, "integrity_verified": integrity_verified,
            "state": "INCOMPLETE" if not complete else "AWAITING VERIFICATION" if not integrity_verified else "PASS" if passes else "FAIL",
            "mean_delta": mean_delta, "improved_roots": improved, "worst_root_delta": worst,
            "complete_roots": len(deltas),
            "selection_expected": sum(v["selection_expected"] for v in views),
            "selection_completed": sum(v["selection_completed"] for v in views),
            "validation_expected_unique": sum(v["expected"] for v in views),
            "validation_completed_unique": sum(v["completed"] for v in views)}


def atomic_text(path: Path, value: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(value)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def markdown(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def generate(report: dict, output_dir: Path, *, source_bytes: bytes | None = None) -> dict:
    view = report_view(report)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_bytes = source_bytes if source_bytes is not None else (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    if json.loads(source_bytes) != report:
        raise ValueError("Source bytes do not match the rendered report")
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    snapshots = output_dir / "report-snapshots"
    snapshots.mkdir(exist_ok=True)
    snapshot = snapshots / f"{source_sha}.json"
    try:
        with snapshot.open("xb") as stream:
            stream.write(source_bytes)
    except FileExistsError:
        if snapshot.read_bytes() != source_bytes:
            raise ValueError("Existing immutable report snapshot changed")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bc26-alpha0-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    import numpy as np

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    roots = view["roots"]
    labels = [f"{v['root_id']} · {v['root'].get('map', 'map unrecorded')} · {v['root'].get('role', '?')}" for v in roots]
    blue, orange, gray, green, red = "#2563eb", "#d97706", "#e2e8f0", "#059669", "#b91c1c"

    def finish(fig, filename):
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".png", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            fig.savefig(temporary, dpi=145, bbox_inches="tight")
            os.replace(temporary, output_dir / filename)
        finally:
            temporary.unlink(missing_ok=True)
            plt.close(fig)

    def empty(ax, text):
        ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color="#64748b")

    fig = plt.figure(figsize=(15, max(8.5, len(roots) * 0.48 + 5)), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    selection_ax, score_ax, delta_ax = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    fig.suptitle(f"Root-ranking mechanism test · {view['state']}\nSelection and fresh-seed validation are separate", fontsize=17, fontweight="bold")
    selection_ax.set_title("Selection only · mean terminal score after both selection seeds complete", pad=12)
    actions = sorted({action for root in roots for action in root["legal"]})
    if roots and actions:
        values = np.full((len(roots), len(actions)), np.nan)
        for y, root in enumerate(roots):
            for x, action in enumerate(actions):
                if action in root["action_scores"]:
                    values[y, x] = root["action_scores"][action]
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(gray)
        heatmap = selection_ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap=cmap, aspect="auto")
        selection_ax.set_xticks(range(len(actions)), actions, fontsize=8)
        selection_ax.set_yticks(range(len(roots)), labels, fontsize=9)
        selection_ax.set_xlabel("Legal first-action IDs (columns absent from every root are omitted)")
        for y, root in enumerate(roots):
            for x, action in enumerate(actions):
                if action not in root["legal"]:
                    selection_ax.add_patch(Rectangle((x-.5, y-.5), 1, 1, facecolor="white", edgecolor="#cbd5e1", hatch="//", linewidth=.2))
            if root["rollout_action"] is not None:
                selection_ax.scatter(actions.index(root["rollout_action"]), y, marker="*", s=100, facecolor="white", edgecolor="black", linewidth=.6)
            if root["puct_action"] is not None:
                selection_ax.scatter(actions.index(root["puct_action"]), y, marker="s", s=125, facecolor="none", edgecolor="black", linewidth=1.5)
        fig.colorbar(heatmap, ax=selection_ax, shrink=.8, label="Selection mean score: loss 0 · draw ½ · win 1")
        selection_ax.text(0, -.29, "★ rollout-selected action    □ PUCT action    Gray: unresolved legal action    Hatched: illegal action", transform=selection_ax.transAxes, fontsize=9)
    else:
        empty(selection_ax, "No frozen roots recorded yet")
    score_ax.set_title("Validation only · paired root means (four fresh seeds)")
    delta_ax.set_title("Validation advantage · rollout-selected minus PUCT")
    for ax in (score_ax, delta_ax):
        ax.set_yticks(range(len(roots)), labels if ax is score_ax else [v["root_id"] for v in roots], fontsize=9)
        ax.set_ylim(max(1, len(roots))-.5, -.5)
        ax.grid(axis="x", alpha=.2)
    score_ax.set_xlim(-.04, 1.04)
    score_ax.set_xlabel("Mean score from complete validation pairs")
    delta_ax.set_xlim(-1.05, 1.05)
    delta_ax.set_xlabel("Per-root mean Δ; small dots show paired-seed differences")
    delta_ax.axvline(0, color="#64748b", linewidth=1)
    delta_ax.axvline(-.25, color=red, linestyle="--", linewidth=1, label="Worst-root floor −0.25")
    for y, root in enumerate(roots):
        if not root["complete"]:
            for ax in (score_ax, delta_ax):
                ax.axhspan(y-.4, y+.4, color=gray)
                ax.text(.5, y, "Unresolved — no score assigned", transform=ax.get_yaxis_transform(), ha="center", va="center", fontsize=8, color="#475569")
            continue
        score_ax.plot([root["puct_mean"], root["rollout_mean"]], [y, y], color="#64748b", linewidth=1.2)
        score_ax.scatter(root["puct_mean"], y-.08, color=blue, s=45, zorder=3)
        score_ax.scatter(root["rollout_mean"], y+.08, color=orange, s=45, zorder=3)
        seed_deltas = [a-b for a, b in zip(root["rollout_scores"], root["puct_scores"])]
        delta_ax.scatter(seed_deltas, [y + (i-1.5)*.09 for i in range(4)], color="#94a3b8", s=18, zorder=2)
        delta_ax.scatter(root["delta"], y, color=green if root["delta"] >= 0 else red, marker="D", s=42, zorder=3)
        delta_ax.annotate(f"{root['delta']:+.3f}", (root["delta"], y), xytext=(6, 7), textcoords="offset points", fontsize=8)
    score_ax.scatter([], [], color=blue, label="PUCT action")
    score_ax.scatter([], [], color=orange, label="Rollout-selected action")
    score_ax.legend(loc="lower left", bbox_to_anchor=(0, -0.24), ncol=2, fontsize=9)
    delta_ax.legend(loc="lower left", bbox_to_anchor=(0, -0.24), fontsize=9)
    fig.supxlabel("Research decision rule, not a strength claim. No SGD or benchmark data. Validation uses new continuation seeds on the same frozen self-play roots.", fontsize=10)
    finish(fig, "comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5.5, len(roots)*.48+2.5)), constrained_layout=True)
    fig.suptitle(f"Measured coverage · {view['complete_roots']}/{len(roots)} roots fully validated", fontsize=17, fontweight="bold")
    for ax, phase, expected_key, completed_key, errors_key in ((axes[0], "Selection", "selection_expected", "selection_completed", "selection_errors"),
                                                            (axes[1], "Validation", "expected", "completed", "errors")):
        ax.set_title(f"{phase} · unique terminal continuations")
        ax.set_yticks(range(len(roots)), labels if ax is axes[0] else [v["root_id"] for v in roots], fontsize=9)
        ax.set_ylim(max(1, len(roots))-.5, -.5)
        limit = max((v[expected_key] for v in roots), default=1)
        ax.set_xlim(0, max(1, limit)*1.2)
        ax.set_xlabel("Actual completed / scheduled unique cells")
        ax.grid(axis="x", alpha=.2)
        for y, root in enumerate(roots):
            expected, completed, errors = root[expected_key], root[completed_key], root[errors_key]
            if expected == 0:
                ax.text(0, y, "Not scheduled; actions not locked", va="center", color="#64748b", fontsize=8)
                continue
            ax.barh(y, expected, color=gray, height=.6)
            ax.barh(y, completed, color=green, height=.6)
            if errors:
                ax.barh(y, errors, left=completed, color=red, hatch="//", height=.6)
            ax.text(expected+limit*.02, y, f"{completed}/{expected}" + (f" · {errors} errors" if errors else ""), va="center", fontsize=8)
        if not roots:
            empty(ax, "No frozen roots recorded yet")
    axes[0].legend(handles=[Patch(facecolor=green, label="Valid terminal result"), Patch(facecolor=red, hatch="//", label="Unresolved error"), Patch(facecolor=gray, label="Pending / missing")],
                   loc="lower left", bbox_to_anchor=(0, -.24), ncol=3, fontsize=8)
    fig.supxlabel("Validation reuses the same cell when both methods choose the same action: four unique runs for one action, eight for two. Errors never become wins or losses.", fontsize=9)
    finish(fig, "coverage.png")

    def shown(value, *, signed=False):
        return "—" if value is None else f"{value:+.3f}" if signed else f"{value:.3f}"

    lines = ["# Root-ranking mechanism test", "", f"**Research decision: {view['state']}.** Phase: `{markdown(report.get('phase', 'unrecorded'))}`.", "",
             "This is a bounded test of first-action ranking on frozen own-self-play roots. **No SGD updates and no benchmark data are used. It does not establish benchmark strength or authorize an automatic training promotion.**", "",
             f"**{view['complete_roots']}/{len(roots)} roots have complete paired validation.** Selection: {view['selection_completed']}/{view['selection_expected']} valid cells. Validation: {view['validation_completed_unique']}/{view['validation_expected_unique']} valid scheduled unique cells.", "",
             "Selection uses seeds `103003, 103013` for every legal first action. The rollout action is chosen by mean terminal score, with the frozen original network prior breaking ties. The comparator is unchanged 32-simulation/depth-12 PUCT.", "",
             "Validation uses separate continuation seeds `103021, 103043, 103067, 103091`, with temperature 0.35 for both teams. These are fresh continuation seeds on the same roots, not new held-out roots. Selection scores never count as validation.", "",
             "![Separate selection and paired validation results](comparison.png)", "", "## Precommitted decision rule", "",
             "The conjunction below is evaluated only after every frozen root has complete valid paired validation. Each root has equal weight, regardless of its legal-action count. A delta is rollout-selected score minus PUCT-selected score.", "",
             "| Requirement | Observed | Outcome |", "|---|---:|---|",
             f"| Every root complete | {view['complete_roots']}/{len(roots)} | {'Met' if view['complete'] else 'Incomplete'} |",
             f"| Equal-root mean Δ ≥ +0.10 | {shown(view['mean_delta'], signed=True)} | {'Met' if view['complete'] and view['mean_delta'] >= .1 else 'Not met' if view['complete'] else 'Pending'} |",
             f"| At least two distinct roots improve (Δ > 0) | {view['improved_roots'] if view['complete'] else '—'} | {'Met' if view['complete'] and view['improved_roots'] >= 2 else 'Not met' if view['complete'] else 'Pending'} |",
             f"| No root has mean Δ < −0.25 | {shown(view['worst_root_delta'], signed=True)} | {'Met' if view['complete'] and view['worst_root_delta'] >= -.25 else 'Not met' if view['complete'] else 'Pending'} |", "",
             f"Final evidence-integrity verification: **{'verified' if view['integrity_verified'] else 'pending'}**. Complete score criteria alone cannot establish a pass; the runner must first stop its worker and verify the frozen experiment identity.", "",
             "The +0.10 threshold applies to the equal-root aggregate, not each individual root. The same four validation seeds are reused across roots, and source games are correlated. This is a small mechanism diagnostic, not a calibrated confidence interval or proof of general playing strength.", "",
             "## Validation by root", "", "| Root | Map / role / round | Rollout / PUCT action | Source cells | Unique valid / scheduled | Rollout mean | PUCT mean | Paired mean Δ | Status |",
             "|---|---|---|---|---:|---:|---:|---:|---|"]
    for root in roots:
        source = root["root"]
        metadata = f"{source.get('map', '?')} / {source.get('role', '?')} / {source.get('round', '?')}"
        status = "Complete" if root["complete"] else "Unresolved error" if root["errors"] else "Incomplete"
        source_cells = "Not scheduled" if not root["expected"] else "Shared four cells" if root["rollout_action"] == root["puct_action"] else "Two actions × four cells"
        lines.append(f"| `{markdown(root['root_id'])}` | {markdown(metadata)} | {root['rollout_action'] if root['rollout_action'] is not None else '—'} / {root['puct_action'] if root['puct_action'] is not None else '—'} | {source_cells} | {root['completed']}/{root['expected']} | {shown(root['rollout_mean'])} | {shown(root['puct_mean'])} | {shown(root['delta'], signed=True)} | {status} |")
    lines += ["", "## Paired validation outcomes", "", "Values below are root-player scores, in the fixed validation-seed order. The same-action case reuses identical results and adds no extra evidence.", "",
              "| Root | Seed | Rollout-selected score | PUCT-selected score | Δ |", "|---|---:|---:|---:|---:|"]
    for root in roots:
        if root["complete"]:
            for seed, rollout_score, puct_score in zip(VALIDATION_SEEDS, root["rollout_scores"], root["puct_scores"]):
                lines.append(f"| `{markdown(root['root_id'])}` | {seed} | {rollout_score:g} | {puct_score:g} | {rollout_score-puct_score:+g} |")
    if not any(root["complete"] for root in roots):
        lines.append("| — | — | — | — | — |")
    lines += ["", "![Measured selection and validation coverage](coverage.png)", "",
              "Gray denotes missing or incomplete evidence, never score zero. Measured zeroes from completed losses remain colored score zeroes. Red coverage bars mark unresolved errors; successful retries remain valid while earlier error attempts remain in the source evidence.", "",
              f"Cumulative recorded error attempts: **{count(report.get('totals', {}).get('attempt_errors', 0), 'attempt errors')}**.", "",
              "## Source snapshot", "", f"- Exact input report: [`{source_sha}.json`](report-snapshots/{source_sha}.json)",
              f"- Report SHA-256: `{source_sha}`", f"- Experiment identity SHA-256: `{markdown(report.get('identity_sha256', 'not recorded'))}`",
              f"- Generated at: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`", "",
              "Regenerate from the repository root with `.venv/bin/python scripts/plot_root_ranking.py`. The renderer performs no engine runs or parameter updates; it recomputes the reported research decision from complete paired validation scores.", ""]
    atomic_text(output_dir / "README.md", "\n".join(lines))
    return {"state": view["state"], "roots": len(roots), "complete_roots": view["complete_roots"], "report_sha256": source_sha,
            "outputs": ["README.md", "comparison.png", "coverage.png"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "progress/root-ranking-v4/summary.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "progress/root-ranking-v4")
    args = parser.parse_args()
    source = args.report.read_bytes()
    print(json.dumps(generate(json.loads(source), args.output_dir, source_bytes=source), indent=2))


if __name__ == "__main__":
    main()
