#!/usr/bin/env python3
"""Plot the recorded own-self-play plateau audit; no model or engine execution."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "battlecode-matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    directory = Path(__file__).resolve().parent
    report = json.loads((directory / "targets.json").read_text())
    exact = report["exact_target_groups"]
    inference = report["inference_groups"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c2cbd3", "axes.labelcolor": "#334155",
                         "text.color": "#172b42", "xtick.color": "#475569", "ytick.color": "#475569"})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), gridspec_kw={"width_ratios": [1, 1.15, 1.2]})
    fig.subplots_adjust(left=.055, right=.98, bottom=.25, top=.76, wspace=.36)
    fig.patch.set_facecolor("#fbfcfe")
    teal, orange, blue, pale = "#168f88", "#d97b38", "#4169a8", "#dbe4eb"
    for ax in axes:
        ax.set_facecolor("#fbfcfe")
        ax.set_axisbelow(True)

    ax = axes[0]
    x = np.arange(2)
    legal = [exact[f"iteration_{i}"]["mean_legal_actions"] for i in (1, 2)]
    support = [exact[f"iteration_{i}"]["mean_support"] for i in (1, 2)]
    ax.bar(x, legal, width=.52, color=pale, label="Legal actions")
    ax.bar(x, support, width=.52, color=teal, label="Actions visited by search")
    for xi, l, s in zip(x, legal, support):
        ax.text(xi, l+.5, f"{l:.1f}", ha="center", fontsize=11, color="#475569")
        ax.text(xi, s/2, f"{s:.2f}", ha="center", va="center", fontsize=12, color="white", weight="bold")
    ax.set(ylim=(0, 28), ylabel="Mean actions per searched position",
           xticks=x, xticklabels=[f"Iteration {i}\n{exact[f'iteration_{i}']['positions']:,} targets" for i in (1, 2)])
    ax.set_title("Search visits few alternatives", loc="left", fontsize=12.5, weight="bold", pad=20)
    ax.grid(axis="y", color="#e9edf2")
    ax.legend(loc="upper left", frameon=False, fontsize=9.3)

    ax = axes[1]
    agreement = [inference["iteration_1"]["0"]["argmax_target_agreement"],
                 inference["iteration_2"]["1"]["argmax_target_agreement"],
                 report["checkpoint_policy_pairs"]["0_to_2"]["all"]["argmax_agreement"]]
    labels = ["cp0 greedy = iteration 1 MCTS choice", "cp1 greedy = iteration 2 MCTS choice", "cp0 greedy = cp2 greedy"]
    y = np.array([2.5, 1.35, .2])
    ax.barh(y, [100]*3, height=.30, color=pale)
    ax.barh(y, np.asarray(agreement)*100, height=.30, color=[teal, teal, blue])
    for yi, value, label in zip(y, agreement, labels):
        ax.text(0, yi+.25, label, va="bottom", fontsize=9.8)
        ax.text(value*100+1.6, yi, f"{100*value:.2f}%", va="center", fontsize=10.5, weight="bold")
    ax.set(xlim=(0, 124), ylim=(-.35, 3.2), xlabel="Agreement (%)", yticks=[], xticks=[0, 25, 50, 75, 100])
    ax.spines["left"].set_visible(False)
    ax.set_title("Preferred actions barely change", loc="left", fontsize=12.5, weight="bold", pad=20)
    ax.grid(axis="x", color="#e9edf2")

    ax = axes[2]
    same = [inference["iteration_2/current_only"][str(cp)]["value_brier"] for cp in range(3)]
    historical = [inference["iteration_2/historical"][str(cp)]["value_brier"] for cp in range(3)]
    same_games = sum(row["iteration"] == 2 and not row["opponent_checkpoint"] for row in report["games"])
    prior_games = sum(row["iteration"] == 2 and bool(row["opponent_checkpoint"]) for row in report["games"])
    ax.axhspan(0, .30, xmin=.80, xmax=1, color="#f1f4f8", zorder=0)
    ax.axhline(.25, color="#94a3b8", linestyle="--", linewidth=1, zorder=1)
    ax.plot(range(3), same, color=blue, marker="o", linewidth=2.4, label=f"Current vs current · {same_games} games")
    ax.plot(range(3), historical, color=orange, marker="o", linewidth=2.4, label=f"Current vs prior · {prior_games} games")
    ax.text(0, .25+.012, ".250", ha="center", fontsize=10)
    for xi in (1, 2):
        ax.text(xi, same[xi]+.012, f"{same[xi]:.4f}", ha="center", color=blue, fontsize=10)
        ax.text(xi, historical[xi]-.021, f"{historical[xi]:.4f}", ha="center", color=orange, fontsize=10)
    ax.set(xlim=(-.15, 2.3), ylim=(0, .30), ylabel="Win-probability Brier score ↓",
           xticks=[0, 1, 2], xticklabels=["cp0\nneutral", "cp1\nfuture games", "cp2\ntraining games"])
    ax.set_title("Critic gain depends on matchup", loc="left", fontsize=12.5, weight="bold", pad=20)
    ax.grid(axis="y", color="#e9edf2")
    ax.legend(loc="lower left", frameon=False, fontsize=9.2)

    fig.text(.055, .94, "Warmstart plateau: broader probabilities, nearly unchanged choices", fontsize=18, weight="bold")
    fig.text(.055, .875, "Own self-play only · 32 PUCT simulations · depth 12 · sparse search · checkpoints 0 → 1 → 2", fontsize=11.5, color="#526579")
    fig.text(.055, .15,
        f"Left: all {exact['all']['positions']:,} retained targets. Inference pool: {report['sample_positions']} fixed stratified positions, game-balanced weights; positions within a game are correlated.",
        fontsize=9.5, color="#526579")
    fig.text(.055, .105,
        f"Right: {inference['iteration_2']['0']['sample_positions']} iteration 2 positions "
        f"({inference['iteration_2/current_only']['0']['sample_positions']} current vs current; "
        f"{inference['iteration_2/historical']['0']['sample_positions']} current vs prior). "
        "cp1 predates these games; cp2 trained on them. Dashed line = neutral forecast.",
        fontsize=9.5, color="#526579")
    fig.text(.055, .06,
        "Historical matchup confound: current player explores more (temperature 1 early + root noise); frozen prior uses temperature 0.3. Losses do not prove weaker greedy play.",
        fontsize=9.5, color="#526579")
    destination = directory / "targets.png"
    fig.savefig(destination, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
