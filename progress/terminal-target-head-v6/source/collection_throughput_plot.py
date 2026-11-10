#!/usr/bin/env python3
"""Plot observed selection completions from immutable journals; no actors or SGD."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[3]
MAPS=["tiny","DefaultSmall","closeup","knifefight"]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def timestamp(value):
    result=datetime.fromisoformat(value.replace("Z","+00:00"))
    if result.tzinfo is None: raise ValueError("Journal timestamp has no timezone")
    return result


def read(path,files,root):
    content=path.read_bytes()
    files[path.relative_to(root).as_posix()]=hashlib.sha256(content).hexdigest()
    return json.loads(content)


def valid_terminal(row,spec):
    raw=row.get("terminal_state",{}); actions=row.get("actions",[])
    winner=row.get("winner"); score=row.get("score")
    if (row.get("terminal") is not True or row.get("valid_game") is not True or row.get("benchmark_failures")!=[] or
        type(winner) is not int or winner not in (-1,0,1) or type(score) not in (int,float) or
        score!=(.5 if winner==-1 else float(winner==spec["root_player"])) or
        type(row.get("terminal_round")) is not int or row["terminal_round"]<=0 or
        raw.get("terminal") is not True or raw.get("winner")!=winner or raw.get("round")!=row["terminal_round"] or
        raw.get("valid_game",True) is not True or raw.get("benchmark_failures",[])!=[] or
        fingerprint({k:v for k,v in raw.items() if k!="snapshot"})!=row.get("terminal_state_sha256") or
        not actions or actions[0]!=spec["action"] or any(type(a) is not int or not 0<=a<40 for a in actions) or
        type(row.get("decisions")) is not int or len(actions)!=row["decisions"] or fingerprint(actions)!=row.get("actions_sha256") or
        row.get("terminal_summary")!={k:raw.get(k) for k in ("terminal","round","winner","win_reason","bank","kings","transferred","cat_damage")}):
        raise ValueError("An alleged successful cell lacks valid immutable terminal proof")


def collect(root,progress):
    root,progress=Path(root).resolve(),Path(progress).resolve()
    scan_started=datetime.now(timezone.utc); files={}
    identity=read(progress/"identity.json",files,root)
    config=read(progress/"config.json",files,root)
    manifest=read(progress/"roots-training.json",files,root)
    exp=fingerprint(identity)
    if (identity.get("experiment_id")!="terminal-target-head-v6" or identity["config"]!=config or
        manifest["identity_sha256"]!=exp or manifest["complete"] is not True or len(manifest["roots"])!=32 or
        config["selection_seeds"]!=[203003,203013] or config["temperature"]!=.35):
        raise ValueError("Throughput requires the frozen complete 32-root v6 selection plan")
    specs={}; planned=Counter()
    for item in manifest["roots"]:
        for action in item["legal_actions"]:
            for seed in config["selection_seeds"]:
                spec={"experiment_identity":exp,"root_id":item["root_id"],"root_player":item["player"],
                    "root_state_sha256":item["state_sha256"],"root_prefix_sha256":item["prefix_sha256"],
                    "root_input_hash":item["input_hash"],"source_sha256":item["source_sha256"],
                    "stage":"selection","action":action,"seed":seed,"temperature":.35}
                key=f"selection-{item['root_id']}-a{action:02d}-s{seed}"
                if key in specs: raise ValueError("Duplicate planned selection cell")
                specs[key]=(spec,item["map"]); planned[item["map"]]+=1
    if len(specs)!=1458: raise ValueError("Frozen v6 selection plan no longer contains 1,458 cells")
    # Capture finite path inventories. Later actor publications are intentionally
    # excluded from this auditable snapshot, never inferred from wall time.
    ends=sorted((progress/"cells").glob("selection-*/attempt-[0-9][0-9][0-9].json"))
    starts=sorted((progress/"cells").glob("selection-*/attempt-[0-9][0-9][0-9]-start.json"))
    start_records={}; end_records={}; success={}; errors=[]; interrupted=[]
    for paths,is_start in ((starts,True),(ends,False)):
        for path in paths:
            cell=path.parent.name
            if cell not in specs: raise ValueError("Journal lies outside the frozen selection plan")
            spec,map_name=specs[cell]; row=read(path,files,root)
            if (row.get("identity")!=fingerprint(spec) or any(row.get(k)!=v for k,v in spec.items()) or
                type(row.get("attempt")) is not int or row["attempt"]<1):
                raise ValueError("Attempt identity differs from its frozen selection cell")
            key=(cell,row["attempt"]); started=timestamp(row["started_at"])
            if started>scan_started: continue
            if is_start:
                start_records[key]=row
                continue
            if key in end_records: raise ValueError("Duplicate immutable attempt")
            end_records[key]=row
            ended=timestamp(row["ended_at"]) if row.get("ended_at") else None
            if ended and ended<started: raise ValueError("Attempt ends before it starts")
            record={"cell":cell,"map":map_name,"attempt":row["attempt"],"status":row["status"],
                    "started_at":row["started_at"],"ended_at":row.get("ended_at"),
                    "artifact":path.relative_to(root).as_posix(),"sha256":files[path.relative_to(root).as_posix()]}
            if row["status"]=="ok":
                valid_terminal(row,spec)
                if not ended: raise ValueError("Valid completion has no observed immutable end time")
                if cell in success: raise ValueError("Successful selection cell was replayed")
                success[cell]=record
            else:
                if row.get("score") is not None or row.get("terminal") is True:
                    raise ValueError("Unfinished/failed attempt carries a scored terminal result")
                if row["status"]=="error": errors.append(record)
                elif row["status"]=="incomplete": interrupted.append(record)
                else: raise ValueError("Unknown unscored selection attempt status")
    for key,row in end_records.items():
        if key not in start_records or start_records[key]["started_at"]!=row["started_at"]:
            raise ValueError("Attempt result lacks its matching immutable start journal")
    if not start_records: raise ValueError("Selection has not started; no observed throughput curve")
    origin=min(timestamp(row["started_at"]) for row in start_records.values())
    events=sorted(success.values(),key=lambda row:(timestamp(row["ended_at"]),row["cell"]))
    counts=Counter()
    for count,row in enumerate(events,1):
        counts[row["map"]]+=1
        row["elapsed_wall_seconds"]=(timestamp(row["ended_at"])-origin).total_seconds()
        row["valid_cells_total"]=count; row["valid_cells_this_map"]=counts[row["map"]]
    start_only=sorted(set(start_records)-set(end_records))
    unresolved_error_cells={row["cell"] for row in errors+interrupted}-set(success)
    observed_elapsed=events[-1]["elapsed_wall_seconds"] if events else None
    return {"schema_version":1,"analysis":"v6-selection-throughput-v1","created_at":datetime.now(timezone.utc).isoformat(),
        "snapshot_scan_started_at":scan_started.isoformat(),"experiment_identity_sha256":exp,
        "helper_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"planned_valid_cells":len(specs),
        "planned_by_map":dict(planned),"valid_cells":len(events),"valid_by_map":dict(counts),
        "earliest_observed_selection_start":origin.isoformat(),"elapsed_to_latest_valid_end_seconds":observed_elapsed,
        "observed_valid_cells_per_hour":len(events)*3600/observed_elapsed if observed_elapsed else None,
        "error_attempts":errors,"interrupted_attempts":interrupted,"unresolved_error_or_interruption_cells":len(unresolved_error_cells),
        "start_only_attempts":[{"cell":cell,"attempt":attempt} for cell,attempt in start_only],
        "events":events,"source_file_hashes":files,"source_file_hashes_sha256":fingerprint(files),
        "scope":"Completed terminal selection cells, not independent training states, optimizer progress, or playing strength.",
        "snapshot_limit":"Only enumerated immutable attempt-end journals count. New completions after this read are absent. Start-only journals are pending/unresolved, not classified as errors or scored games."}


def atomic_text(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w",dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name); stream.write(text); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)


def render(audit,output):
    os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"bc26-alpha0-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig,(ax,detail)=plt.subplots(1,2,figsize=(13,5),gridspec_kw={"width_ratios":[3.3,1]})
    events=audit["events"]
    if events:
        ax.step([row["elapsed_wall_seconds"]/60 for row in events],[row["valid_cells_total"] for row in events],
                where="post",color="#172554",linewidth=2,label=f"All maps: {audit['valid_cells']}/1,458")
        for name,color in zip(MAPS,("#2563eb","#0f766e","#b45309","#7c3aed")):
            rows=[row for row in events if row["map"]==name]
            if rows: ax.step([row["elapsed_wall_seconds"]/60 for row in rows],[row["valid_cells_this_map"] for row in rows],
                        where="post",color=color,linewidth=1.3,label=f"{name}: {len(rows)}/{audit['planned_by_map'][name]}")
    ax.axhline(1458,color="#64748b",linestyle="--",linewidth=1,label="Fixed plan: 1,458 valid terminal cells")
    ax.set_ylim(0,1540); ax.set_xlim(left=0); ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Observed elapsed wall minutes from earliest selection start")
    ax.set_ylabel("Cumulative valid terminal cells"); ax.legend(loc="upper left",fontsize=8); ax.grid(alpha=.12)
    names=["Error attempts","Interrupted ends","Start-only journals"]
    values=[len(audit["error_attempts"]),len(audit["interrupted_attempts"]),len(audit["start_only_attempts"])]
    detail.barh(range(3),values,color=["#b91c1c","#d97706","#94a3b8"])
    for i,value in enumerate(values): detail.text(value+.04,i,str(value),va="center")
    detail.set_yticks(range(3),names,fontsize=9); detail.invert_yaxis(); detail.set_xlim(0,max(values+[1])*1.4)
    detail.xaxis.set_major_locator(MaxNLocator(integer=True)); detail.set_title("Unscored attempt accounting",fontsize=10)
    detail.set_xlabel("Recorded attempts")
    fig.suptitle(f"V6 terminal-target collection · {audit['valid_cells']:,}/1,458 valid cells",fontweight="bold",fontsize=15)
    fig.text(.5,.035,"Immutable terminal end times only; no forecast. Pending/error attempts carry no score.\n1,458 continuations are generated from 32 training roots; this chart makes no learning or strength claim.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.11,1,.93))
    with tempfile.NamedTemporaryFile(dir=output,suffix=".png",delete=False) as stream: temporary=Path(stream.name)
    fig.savefig(temporary,dpi=160,facecolor="white"); plt.close(fig)
    os.replace(temporary,output/"collection-throughput.png")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--progress-dir",type=Path,default=ROOT/"progress/terminal-target-head-v6")
    args=parser.parse_args(); audit=collect(args.root,args.progress_dir)
    atomic_text(args.progress_dir/"collection-throughput.json",json.dumps(audit,sort_keys=True,indent=2,allow_nan=False)+"\n")
    render(audit,args.progress_dir)
    elapsed=audit["elapsed_to_latest_valid_end_seconds"]
    text=["# Observed terminal-target collection", "", "![Valid selection cell collection](collection-throughput.png)", "",
        f"Snapshot read began **{audit['snapshot_scan_started_at']}**. It contains **{audit['valid_cells']:,}/1,458 valid terminal selection cells** from the fixed 32-root training plan.","",
        f"Elapsed wall time from the earliest immutable selection-start record to the latest included valid terminal end: **{elapsed/60:.2f} minutes**. Observed average: **{audit['observed_valid_cells_per_hour']:.1f} valid cells/hour**. This is neither an ETA nor a forecast.","",
        f"Error attempts: **{len(audit['error_attempts'])}**; interrupted end records: **{len(audit['interrupted_attempts'])}**; start-only journals: **{len(audit['start_only_attempts'])}**. Start-only records can be in flight or paused and receive no score. Errors remain counted even if a later valid retry resolves their cell.","",
        "Only immutable successful attempt-end records with valid identity, terminal state, root-relative result, and action-transcript checks count. Each physical cell counts once; repeated successful attempts are rejected. Every x-coordinate is an observed end time relative to the observed first start. New publications after the snapshot are intentionally absent.","",
        "This measures continuation collection, not independent training states, SGD updates, playing strength, or benchmark performance. [Exact event table and SHA-256 provenance](collection-throughput.json). [Standalone helper](source/collection_throughput_plot.py).",""]
    atomic_text(args.progress_dir/"collection-throughput.md","\n".join(text))
    print(json.dumps({key:audit[key] for key in ("valid_cells","planned_valid_cells","valid_by_map","elapsed_to_latest_valid_end_seconds","observed_valid_cells_per_hour")},indent=2))


if __name__=="__main__": main()
