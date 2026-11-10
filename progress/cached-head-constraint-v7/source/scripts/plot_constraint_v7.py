#!/usr/bin/env python3
"""Plot measured v7 training diagnostics; no numerical fitting or held-out reads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

ARMS=("kl_1","kl_0")
COLORS={"kl_1":"#0f766e","kl_0":"#b45309"}
LABELS={"kl_1":"Anchor KL coefficient 1","kl_0":"Anchor KL coefficient 0"}
STEPS=(128,512,2048)


def text_file(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w",dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name); stream.write(text)
    os.replace(temporary,path)


def figure_file(figure,path):
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix=".png",delete=False) as stream: temporary=Path(stream.name)
    figure.savefig(temporary,dpi=150,facecolor="white"); os.replace(temporary,path)


def generate(summary,output):
    if (summary.get("experiment_id")!="cached-head-constraint-v7" or summary.get("training_only") is not True or
        summary.get("validation_data_read") is not False or summary.get("benchmark_data_read") is not False or
        summary.get("automatic_promotion") is not False):
        raise ValueError("Expected training-only v7 evidence")
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(summary,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    digest=hashlib.sha256(encoded).hexdigest()
    text_file(output/"report-snapshots"/f"{digest}.json",json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+"\n")
    os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"bc26-alpha0-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    status=summary["decision"]["classification"] or "incomplete / no classification"
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    for ax,key,title in zip(axes,("target_ce","weighted_anchor_kl","loss"),
                            ("Mixed-target cross-entropy","Weighted sampled KL contribution","Minibatch objective: CE + weighted KL")):
        for arm in ARMS:
            rows=summary["arms"][arm]["history"]
            if rows: ax.plot([row["step"] for row in rows],[row[key] for row in rows],color=COLORS[arm],linewidth=.9,label=LABELS[arm])
        ax.set_xlim(1,2048); ax.set_xticks([128,512,1024,1536,2048]); ax.set_xlabel("Observed optimizer update")
        ax.set_ylabel("Nats"); ax.set_title(title,fontsize=10); ax.grid(alpha=.15)
        if not any(summary["arms"][arm]["history"] for arm in ARMS):
            ax.text(.5,.5,"No completed updates recorded",transform=ax.transAxes,ha="center",fontsize=9)
    handles,labels=axes[0].get_legend_handles_labels()
    if handles: fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.06),ncol=2,frameon=False)
    fig.suptitle("V7 cached-head diagnostic · actual unsmoothed training traces",fontweight="bold")
    fig.text(.5,.015,"Only 5,160 policy-head parameters train; encoders/value remain frozen. Curves use pre-update minibatches, not held-out performance.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.15,1,.92)); figure_file(fig,output/"learning-curves.png"); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    panels=[("root_selected_action_matches","All selected targets (32)",32),
            ("discordant_target_matches","Changed selected targets (9)",9),
            ("retained_target_matches","Retained selected targets (23)",23),
            ("expected_anchor_kl","Anchor KL, weighted by game/team",None),
            ("anchor_action_changes","Anchor greedy actions changed (512)",512),
            ("ce_kl_gradient_cosine","CE / weighted-anchor-KL gradient cosine",None)]
    for ax,(key,title,maximum) in zip(axes.flat,panels):
        for arm in ARMS:
            snapshots=summary["arms"][arm]["snapshots"]
            rows=[(step,snapshots[str(step)]["metrics"][key]) for step in STEPS if str(step) in snapshots]
            rows=[(step,value) for step,value in rows if value is not None]
            if rows: ax.plot([step for step,value in rows],[value for step,value in rows],marker="o",color=COLORS[arm],label=LABELS[arm])
        ax.set_xlim(0,2176); ax.set_xticks(STEPS); ax.set_xlabel("Fixed snapshot update"); ax.set_title(title,fontsize=10); ax.grid(alpha=.15)
        if maximum:
            ax.set_ylim(0,maximum*1.06); ax.yaxis.set_major_locator(MaxNLocator(integer=True)); ax.axhline(maximum,color="#94a3b8",linestyle=":",linewidth=1)
        elif key=="ce_kl_gradient_cosine": ax.set_ylim(-1.05,1.05); ax.axhline(0,color="#94a3b8",linewidth=.6)
    fig.suptitle(f"V7 fixed snapshots · {status}",fontweight="bold")
    handles,labels=axes[0,0].get_legend_handles_labels()
    if handles: fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.055),ncol=2,frameon=False)
    fig.text(.5,.015,"Missing snapshots are absent, never zero. Classification requires all six snapshots and exact 128 reproduction. No promotion or strength claim.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.13,1,.93)); figure_file(fig,output/"fixed-snapshots.png"); plt.close(fig)
    decision=summary["decision"]
    lines=["# Cached-head-constraint-v7", "", f"**Classification: {status}. Training-only diagnostic; no automatic promotion or victory claim.**", "",
        f"Exact plotted [report snapshot](report-snapshots/{digest}.json). Reproduction verified: **{decision['reproduction_verified']}**. Final integrity verified: **{decision['integrity_verified']}**.","",
        "![Actual optimizer traces](learning-curves.png)","","![Fixed snapshot diagnostics](fixed-snapshots.png)","",
        "Both arms retain all 32 terminal labels, including 9 changed and 23 retained labels, and all 512 anchors. Only 5,160 policy-head parameters train. The sole arm difference is the anchor-KL coefficient; sampled batch order, targets, optimizer and frozen features are shared.","",
        "Curves are actual pre-update minibatch observations. Fixed-snapshot KL and objective below use the declared uniform-game, uniform-team, within-stratum anchor weighting. The raw KL remains reported for KL=0 even though its objective contribution is zero. Different aggregation means a curve endpoint need not equal a full-set snapshot metric.","",
        "| Arm | Update | All 32 | Changed 9 | Retained 23 | CE | Game/team KL | Weighted KL contribution | Objective | Anchor changes/512 | Parameter L2 | Gradient cosine |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fmt=lambda value:"—" if value is None else f"{value:.6f}"
    for arm in ARMS:
        for step in STEPS:
            row=summary["arms"][arm]["snapshots"].get(str(step))
            if row is None:
                lines.append(f"| {arm} | {step} | Pending | Pending | Pending | — | — | — | — | — | — | — |")
                continue
            m=row["metrics"]
            lines.append(f"| {arm} | {step} | {m['root_selected_action_matches']}/32 | {m['discordant_target_matches']}/9 | {m['retained_target_matches']}/23 | {fmt(m['target_ce'])} | {fmt(m['expected_anchor_kl'])} | {fmt(m['weighted_anchor_kl_contribution'])} | {fmt(m['expected_objective'])} | {m['anchor_action_changes']}/512 | {fmt(m['parameter_displacement_l2'])} | {fmt(m['ce_kl_gradient_cosine'])} |")
    lines += ["", "The ordered rule first assigns budget evidence if KL=1 fits all 32 at 512 or 2048 after exact 128 reproduction, even if it later regresses. Otherwise it assigns anchoring evidence within the tested budget only when KL=0 fits all 32 and KL=1 never does. If neither succeeds, classification is neither. Incomplete evidence permits no completed classification.","",
        f"Regression after a full match: `{json.dumps(decision['regressed_after_full_match'],sort_keys=True)}`. First full-match snapshots: `{json.dumps(decision['first_full_match_step'],sort_keys=True)}`.","",
        "All 32 greedy actions, complete observed losses, displacement and gradient diagnostics are retained in the linked JSON. These results establish neither label correctness, generalization, playing strength nor a stationary regularized optimum. No old validation or benchmark data was read for this report.",""]
    text_file(output/"README.md","\n".join(lines))
    return {"report_sha256":digest,"curves":str(output/"learning-curves.png"),"snapshots":str(output/"fixed-snapshots.png")}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(); generate(json.loads(args.report.read_text()),args.output_dir)


if __name__=="__main__": main()
