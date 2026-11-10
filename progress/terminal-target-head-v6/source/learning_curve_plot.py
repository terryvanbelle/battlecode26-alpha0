#!/usr/bin/env python3
"""Plot only completed, hash-verified v6 fitting traces; never run an actor/model.

Run after both fixed candidates have been published. There is no partial-fit
fallback, invented step-zero observation, smoothing, or playing-strength claim.
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

ROOT = Path(__file__).resolve().parents[3]
ARMS = ("terminal_head", "puct_head")
LABELS = {"terminal_head":"Terminal-ranking targets", "puct_head":"Matched PUCT targets"}
COLORS = {"terminal_head":"#0f766e", "puct_head":"#b45309"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def finite_number(value, name, *, rate=False):
    if type(value) not in (int,float) or not math.isfinite(value) or (rate and not 0 <= value <= 1):
        raise ValueError(f"Invalid recorded metric: {name}")
    return float(value)


def verify_trace(history, final_metrics, kl_weight):
    if not isinstance(history,list) or len(history)!=128:
        raise ValueError("Both arms must have exactly 128 observed fitting updates")
    rows=[]
    for expected,row in enumerate(history,1):
        if type(row.get("step")) is not int or row["step"]!=expected:
            raise ValueError("Fitting history is missing, duplicated or reordered")
        measured={key:finite_number(row[key],key) for key in ("target_ce","anchor_kl","loss")}
        if not math.isclose(measured["loss"],measured["target_ce"]+kl_weight*measured["anchor_kl"],rel_tol=2e-6,abs_tol=2e-6):
            raise ValueError("Recorded objective does not equal target CE plus weighted anchor KL")
        rows.append({"step":expected,**measured})
    metrics={key:finite_number(final_metrics[key],key) for key in ("target_ce","anchor_kl","loss")}
    if not math.isclose(metrics["loss"],metrics["target_ce"]+kl_weight*metrics["anchor_kl"],rel_tol=2e-6,abs_tol=2e-6):
        raise ValueError("Final recorded full-dataset objective is inconsistent")
    for key in ("root_action_changed_fraction","anchor_action_changed_fraction","root_target_agreement"):
        metrics[key]=finite_number(final_metrics[key],key,rate=True) if key in final_metrics else None
    return rows,metrics


def load_completed(root, run_dir, progress_dir):
    root,run_dir,progress_dir=(Path(path).resolve() for path in (root,run_dir,progress_dir))
    manifest_path=run_dir/"fit/manifest.json"
    summary_path=progress_dir/"fit-summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise ValueError("Both completed fixed heads must be published before plotting")
    paths={manifest_path,summary_path,progress_dir/"identity.json"}
    initial_hashes={str(path):sha256(path) for path in paths}
    manifest=json.loads(manifest_path.read_text())
    identity=json.loads((progress_dir/"identity.json").read_text())
    if (manifest!=json.loads(summary_path.read_text()) or manifest.get("complete") is not True or
        manifest.get("experiment_id")!="terminal-target-head-v6" or identity.get("experiment_id")!="terminal-target-head-v6" or
        manifest.get("identity_sha256")!=fingerprint(identity) or
        manifest.get("fit_identity_sha256")!=fingerprint(manifest.get("fit_identity")) or
        manifest.get("fit_config")!=identity["config"]["fit"] or set(manifest.get("checkpoints",{}))!=set(ARMS)):
        raise ValueError("Completed fitting identity or public summary changed")
    if manifest["fit_config"]["updates"]!=128 or manifest["fit_config"]["anchor_kl_weight"]!=1:
        raise ValueError("Learning curve does not use the frozen 128-update CE+KL protocol")
    required={f"{arm}.pt" for arm in ARMS}|{"batches.npz","batch-rng.json","parent-metadata.json","identity.json"}
    if set(manifest["files"])!=required:
        raise ValueError("Completed fit artifact inventory is incomplete")
    for name,digest in manifest["files"].items():
        path=run_dir/"fit"/name
        if Path(name).name!=name or sha256(path)!=digest:
            raise ValueError("Completed fitting artifact checksum mismatch")
        initial_hashes[str(path)]=digest
    schedule=manifest["batch_schedule"]
    if (schedule["sha256"]!=manifest["files"]["batches.npz"] or schedule["root_shape"]!=[128,32] or
        schedule["anchor_shape"]!=[128,32] or schedule["seed"]!=203101):
        raise ValueError("Candidate arms lack the frozen shared batch schedule")
    import torch
    torch.set_num_threads(1)
    arms={}; batch_hashes=set()
    for arm in ARMS:
        record=manifest["checkpoints"][arm]
        path=Path(record["path"]); path=path if path.is_absolute() else root/path
        if (path.resolve()!=(run_dir/"fit"/f"{arm}.pt").resolve() or record["sha256"]!=sha256(path) or
            record["sha256"]!=manifest["files"][f"{arm}.pt"] or record.get("updates")!=128 or
            record.get("trainable_parameters")!=5160 or record.get("frozen_tensors_equal") is not True or
            record.get("value_logits_equal") is not True or record.get("fit_identity_sha256")!=manifest["fit_identity_sha256"]):
            raise ValueError("Candidate identity or policy-head-only fitting proof changed")
        payload=torch.load(path,map_location="cpu",weights_only=True)
        if (payload.get("arm")!=arm or payload.get("experiment_id")!="terminal-target-head-v6" or
            payload.get("fit_identity_sha256")!=manifest["fit_identity_sha256"] or payload.get("updates")!=128 or
            payload.get("trainable_parameters")!=5160 or payload.get("policy_head_only") is not True or
            payload.get("benchmark_data_used") is not False or payload.get("validation_data_used") is not False or
            payload.get("parent_checkpoint_sha256")!=identity["checkpoint_sha256"] or
            payload.get("frozen_state_sha256")!=manifest["frozen_state_sha256"] or
            payload.get("fit_metrics")!=record["metrics"]):
            raise ValueError("Checkpoint fitting provenance disagrees with the completed manifest")
        history,metrics=verify_trace(payload["fit_history"],record["metrics"],manifest["fit_config"]["anchor_kl_weight"])
        batch_hashes.add(payload["batch_sha256"])
        arms[arm]={"checkpoint_sha256":record["sha256"],"history":history,"final_full_dataset_metrics":metrics}
    if len(batch_hashes)!=1:
        raise ValueError("Candidate arms used different fitting minibatches")
    if any(sha256(path)!=digest for path,digest in initial_hashes.items()):
        raise ValueError("Completed fit evidence changed while reading curves")
    return {"schema_version":1,"analysis":"terminal-head-v6-observed-fit-curves-v1",
        "created_at":datetime.now(timezone.utc).isoformat(),"helper_sha256":sha256(__file__),
        "experiment_identity_sha256":fingerprint(identity),"fit_identity_sha256":manifest["fit_identity_sha256"],
        "source_file_hashes":initial_hashes,"fit_config":manifest["fit_config"],"batch_schedule":schedule,
        "trainable_parameters":5160,"frozen_encoders_and_value_head":True,"observed_updates_per_arm":128,
        "trace_timing":"Recorded minibatch losses immediately before each optimizer update, steps 1 through 128; no synthetic step zero.",
        "final_metrics_timing":"One full cached training/anchor evaluation after all 128 updates; changed-action rates are not per-step traces.",
        "scope":"Training objective and policy movement only; no playing-strength, transfer-success, promotion, or victory claim.",
        "arms":arms}


def atomic_text(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w",dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name); stream.write(text); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)


def plot(audit,output):
    os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"bc26-alpha0-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,key,title in zip(axes.flat,("target_ce","anchor_kl","loss"),
                            ("Selected-action target cross-entropy","Preservation KL: cp2 → fitted policy","Total objective: target CE + anchor KL")):
        for arm,record in audit["arms"].items():
            ax.plot([row["step"] for row in record["history"]],[row[key] for row in record["history"]],
                    color=COLORS[arm],label=LABELS[arm],linewidth=1.35,alpha=.9)
        ax.set_xlim(1,128); ax.set_xticks([1,32,64,96,128]); ax.set_xlabel("Optimizer update (recorded pre-update minibatch)")
        ax.set_ylabel("Nats"); ax.set_title(title,fontsize=11); ax.grid(alpha=.15)
    ax=axes[1,1]
    for offset,arm in enumerate(ARMS):
        metrics=audit["arms"][arm]["final_full_dataset_metrics"]
        for index,key in enumerate(("root_action_changed_fraction","anchor_action_changed_fraction")):
            x=index+(offset-.5)*.34; value=metrics[key]
            if value is None: ax.text(x,.025,"Not\nrecorded",ha="center",va="bottom",fontsize=9)
            else:
                ax.bar(x,value,width=.31,color=COLORS[arm],label=LABELS[arm] if index==0 else None)
                ax.text(x,value+.025,f"{value:.1%}",ha="center",fontsize=9)
    ax.set_ylim(0,1.12); ax.set_xlim(-.6,1.6); ax.set_xticks([0,1],["32 training roots","512 KL anchors"])
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_ylabel("Greedy actions changed from cp2")
    ax.set_title("Final movement after 128 updates (not a learning curve)",fontsize=11)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.065),ncol=2,frameon=False)
    fig.suptitle("V6 policy-head fitting · both fixed candidates",fontsize=15,fontweight="bold",y=.98)
    fig.text(.5,.925,"Only 5,160 policy-head parameters train; encoders and value head remain frozen.",ha="center",fontsize=10)
    fig.text(.5,.025,"Actual unsmoothed minibatch traces. Lower training loss or changed actions alone do not establish playing strength.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.90))
    with tempfile.NamedTemporaryFile(dir=output,suffix=".png",delete=False) as stream: temporary=Path(stream.name)
    fig.savefig(temporary,dpi=160,facecolor="white"); plt.close(fig)
    os.replace(temporary,output/"learning-curves.png")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--run-dir",type=Path,default=ROOT/"runs/terminal-target-head-v6")
    parser.add_argument("--progress-dir",type=Path,default=ROOT/"progress/terminal-target-head-v6")
    args=parser.parse_args()
    audit=load_completed(args.root,args.run_dir,args.progress_dir)
    output=args.progress_dir
    atomic_text(output/"learning-curves.json",json.dumps(audit,sort_keys=True,indent=2,allow_nan=False)+"\n")
    plot(audit,output)
    lines=["# Observed v6 policy-head fitting", "", "![Actual learning curves](learning-curves.png)", "",
        "Both fixed candidate heads completed exactly 128 AdamW updates with the same minibatches and optimizer settings. Only 5,160 policy-head parameters trained; encoders and the value head stayed frozen.","",
        "Curves show every recorded pre-update minibatch target cross-entropy, cp2-preservation KL, and their summed objective. There is no synthetic step-zero point, smoothing, or extrapolation. Target selectors differ between arms; losses measure each arm’s own target objective.","",
        "Changed greedy-action rates are measured once after fitting over all 32 training roots and 512 anchors. They are final policy-movement measurements, not invented per-update trajectories.","",
        "| Arm | Final full-set CE | Final full-set KL | Final full-set objective | Root actions changed | Anchor actions changed |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm,record in audit["arms"].items():
        m=record["final_full_dataset_metrics"]
        rate=lambda value:"Not recorded" if value is None else f"{value:.2%}"
        lines.append(f"| {LABELS[arm]} | {m['target_ce']:.6f} | {m['anchor_kl']:.6f} | {m['loss']:.6f} | {rate(m['root_action_changed_fraction'])} | {rate(m['anchor_action_changed_fraction'])} |")
    lines += ["", "The final full-set metrics use a different aggregation than the minibatch trace, so the last curve point need not equal the table. Neither objective reduction nor policy movement establishes validation improvement, benchmark strength, promotion eligibility, or all-map victory.","",
        "[Exact observed traces and checksum provenance](learning-curves.json). [Standalone helper](source/learning_curve_plot.py); it verifies complete manifests and checkpoint hashes and reads the two checkpoints with `weights_only=True`, without model inference or actors.",""]
    atomic_text(output/"learning-curves.md","\n".join(lines))
    print(json.dumps({"fit_identity_sha256":audit["fit_identity_sha256"],"updates_per_arm":128,"output":str(output/"learning-curves.png")},indent=2))


if __name__=="__main__": main()
