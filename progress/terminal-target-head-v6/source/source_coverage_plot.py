#!/usr/bin/env python3
"""Audit source-slot coverage only; never import actors or use game outcomes.

The two initial training panels share all 16 engine/policy seed pairs. Reserve
contributions remain separate. This is a source sampling comparison, not a
measurement of playing strength, policy transfer, or the all-map victory gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

MAPS = ["tiny", "DefaultSmall", "closeup", "knifefight"]
STRATA = [("BABY_RAT", 0), ("BABY_RAT", 1), ("RAT_KING", 0), ("RAT_KING", 1)]
ROOT = Path(__file__).resolve().parents[3]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def slots(j):
    return [("BABY_RAT", 0 if j % 2 else 1), ("RAT_KING", 1 if j % 2 else 0)]


def read_artifact(root, path, evidence):
    path = Path(path)
    path = path if path.is_absolute() else root/path
    before = path.read_bytes()
    evidence[path.relative_to(root).as_posix()] = digest(before)
    return json.loads(before)


def audit_version(root, version, evidence):
    progress = root/f"progress/terminal-target-head-{version}"
    run = root/f"runs/terminal-target-head-{version}"
    config = read_artifact(root, progress/"config.json", evidence)
    identity = read_artifact(root, progress/"identity.json", evidence)
    expected_temperature = .35 if version == "v5" else 1.0
    if (config["maps"] != MAPS or identity["config"] != config or
        config["training_roots"] != 32 or config["validation_roots"] != 16 or
        config["temperature"] != .35 or config.get("source_temperature", config["temperature"]) != expected_temperature):
        raise ValueError("Source allocation/temperature differs from the declared paired experiment")
    source_index = {}
    splits = {}
    for split, initial_games, maximum_j, planned in (("training", 4, 8, 32), ("validation", 2, 6, 16)):
        manifest_path = progress/f"roots-{split}.json"
        manifest = read_artifact(root, manifest_path, evidence) if manifest_path.exists() else None
        if manifest is not None and manifest["identity_sha256"] != fingerprint(identity):
            raise ValueError("Root manifest belongs to another experiment")
        source_rows = {}
        for path in sorted((run/"sources").glob(f"{split}-m*-j*.json")):
            row = read_artifact(root, path, evidence)
            m, j = row["map_index"], row["j"]
            base, policy_base = (20261200, 310000) if split == "training" else (20262200, 320000)
            expected = {"source_id": f"{split}-m{m}-j{j}", "split": split,
                        "map": MAPS[m], "seed": base+100*m+j,
                        "policy_seeds": [2*(policy_base+100*m+j)+t for t in (0,1)]}
            if (type(m) is not int or m not in range(4) or type(j) is not int or not 1 <= j <= maximum_j or
                any(row.get(k) != v for k,v in expected.items()) or row["experiment_identity"] != fingerprint(identity) or
                row.get("source") != "selfplay" or row.get("status") != "ok" or row.get("complete") is not True or
                row["temperature"] != expected_temperature or row["checkpoint_sha256"] != config["checkpoint_sha256"]):
                raise ValueError(f"Source identity or completion mismatch: {path}")
            if row["source_id"] in source_rows:
                raise ValueError("Duplicate source game")
            for key, candidate in row["root_candidates"].items():
                role, player = key.rsplit(":",1)
                if ((role,int(player)) not in STRATA or candidate["role"] != role or candidate["player"] != int(player) or
                    37 not in candidate["legal_actions"]):
                    raise ValueError("Candidate lacks its declared role/team/qualifying action")
            source_rows[row["source_id"]] = row
            source_index[row["source_id"]] = {**expected, "map_index": m, "j": j,
                "temperature": row["temperature"], "source_json": path.relative_to(root).as_posix(),
                "source_sha256": evidence[path.relative_to(root).as_posix()],
                "cache_sha256_recorded": row["cache_sha256"],
                "candidate_slots": sorted(row["root_candidates"])}
        initial = []
        for m, name in enumerate(MAPS):
            for j in range(1, initial_games+1):
                source_id = f"{split}-m{m}-j{j}"
                row = source_rows.get(source_id)
                for role, player in slots(j):
                    key = f"{role}:{player}"
                    candidate = row["root_candidates"].get(key) if row else None
                    initial.append({"map": name, "map_index": m, "j": j, "source_id": source_id,
                        "role": role, "player": player, "observed": row is not None,
                        "qualified": candidate is not None if row else None,
                        "candidate_sha256": fingerprint(candidate) if candidate is not None else None})
        assignments = manifest["assignments"] if manifest else []
        assigned = set()
        reserve_entries = []
        for assignment in assignments:
            m, group_j, j = assignment["map_index"], assignment["slot_group_j"], assignment["source_j"]
            source_id = assignment["source_id"]
            if (source_id != f"{split}-m{m}-j{j}" or source_id not in source_rows or
                group_j not in range(1,initial_games+1) or j % 2 != group_j % 2):
                raise ValueError("Unexpected reserve assignment")
            row = source_rows[source_id]
            allowed = {f"{role}:{player}" for role,player in slots(group_j)}
            if set(assignment["filled"]) - allowed:
                raise ValueError("Reserve assigned a different role/team slot")
            for slot in assignment["filled"]:
                key = (m,group_j,slot)
                if key in assigned or slot not in row["root_candidates"]:
                    raise ValueError("Duplicate or unsupported accepted source slot")
                assigned.add(key)
            if j > initial_games:
                reserve_entries.append({**assignment, "map": MAPS[m], "source_sha256": source_index[source_id]["source_sha256"]})
            elif set(assignment["filled"]) != allowed & set(row["root_candidates"]):
                raise ValueError("Initial assignment omits a qualifying declared slot")
        if manifest is not None:
            if len(assigned) != len(manifest["roots"]):
                raise ValueError("Accepted manifest roots do not match its source assignments")
            if (set(manifest["sources"]) != set(source_rows) or
                sum(x["qualified"] is True for x in initial)+sum(len(x["filled"]) for x in reserve_entries) != len(assigned)):
                raise ValueError("Source inventory or initial/reserve coverage accounting differs from the manifest")
            for source_id, ref in manifest["sources"].items():
                if source_id not in source_index or ref["source_sha256"] != source_index[source_id]["source_sha256"]:
                    raise ValueError("Frozen manifest source reference changed")
            if manifest["complete"] and (len(assigned) != planned or manifest["unavailable_slots"]):
                raise ValueError("Complete source manifest has missing allocated slots")
        strata = []
        for name in MAPS:
            for role, player in STRATA:
                cells = [x for x in initial if (x["map"],x["role"],x["player"]) == (name,role,player)]
                strata.append({"map": name, "role": role, "player": player, "planned": len(cells),
                    "observed": sum(x["observed"] for x in cells), "qualified": sum(x["qualified"] is True for x in cells)})
        splits[split] = {"planned_slots": planned, "initial_planned_source_games": 4*initial_games,
            "initial_completed_source_games": sum(row["j"] <= initial_games for row in source_rows.values()),
            "initial_observed_slots": sum(x["observed"] for x in initial),
            "initial_qualified_slots": sum(x["qualified"] is True for x in initial),
            "reserve_completed_source_games": sum(row["j"] > initial_games for row in source_rows.values()),
            "reserve_slots_added": sum(len(x["filled"]) for x in reserve_entries),
            "final_accepted_slots": len(assigned), "manifest_present": manifest is not None,
            "manifest_complete": manifest is not None and manifest["complete"] is True,
            "initial_slots": initial, "strata": strata, "reserve_assignments": reserve_entries}
    return {"source_temperature": expected_temperature, "continuation_temperature": .35,
            "experiment_identity_sha256": fingerprint(identity), "checkpoint_sha256": config["checkpoint_sha256"],
            "engine_runtime_sha256": fingerprint(identity["runtime"]), "splits": splits, "sources": source_index}


def build_audit(root):
    root = Path(root).resolve()
    evidence = {}
    versions = {v:audit_version(root,v,evidence) for v in ("v5","v6")}
    if (versions["v5"]["checkpoint_sha256"] != versions["v6"]["checkpoint_sha256"] or
        versions["v5"]["engine_runtime_sha256"] != versions["v6"]["engine_runtime_sha256"]):
        raise ValueError("Paired coverage comparison changed baseline/runtime")
    paired = []
    for m in range(4):
        for j in range(1,5):
            source_id = f"training-m{m}-j{j}"
            a,b = (versions[v]["sources"].get(source_id) for v in ("v5","v6"))
            if a and b and any(a[k] != b[k] for k in ("map","seed","policy_seeds")):
                raise ValueError("Initial source games are not paired by identical engine/policy seeds")
            paired.append({"source_id": source_id, "both_observed": bool(a and b),
                           "seed": (a or b)["seed"] if a or b else None,
                           "policy_seeds": (a or b)["policy_seeds"] if a or b else None})
    # Re-read only immutable inputs, never a live summary or actor-owned log.
    if any(digest((root/path).read_bytes()) != sha for path,sha in evidence.items()):
        raise ValueError("Source evidence changed during coverage audit")
    return {"schema_version":1, "analysis":"paired-source-slot-coverage-v1",
        "created_at":datetime.now(timezone.utc).isoformat(), "helper_sha256":digest(Path(__file__).read_bytes()),
        "scope":"Source temperature coverage amendment; not playing strength, learning transfer, or victory.",
        "outcomes_used_for_selection_or_comparison":False, "benchmark_data_used":False,
        "paired_initial_training_games":paired, "paired_initial_training_games_observed":sum(x["both_observed"] for x in paired),
        "source_file_hashes":evidence, "source_file_hashes_sha256":fingerprint(evidence),
        "cache_hashes":"Recorded in source JSON; caches were not reopened for this read-only coverage comparison.",
        "reporting_note":"The pinned generic v6 coverage plot labels fully scored validation roots as 'Validation roots'. This separate figure counts collected source roots; no pinned file was changed during active collection.",
        "versions":versions}


def atomic_text(path, text):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w",dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name); stream.write(text); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)


def plot(audit, output):
    os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"bc26-alpha0-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig=plt.figure(figsize=(12,8.5))
    grid=fig.add_gridspec(2,2,height_ratios=[1.2,1],hspace=.60,wspace=.24,top=.83,bottom=.24)
    legend_handles = legend_labels = None
    for column,version in enumerate(("v5","v6")):
        ax=fig.add_subplot(grid[0,column]); stats=audit["versions"][version]["splits"]["training"]
        values=np.array([x["qualified"]/x["planned"] if x["observed"]==x["planned"] else np.nan
                         for x in stats["strata"]]).reshape(4,4)
        ax.imshow(values,vmin=0,vmax=1,cmap="YlGnBu",aspect="auto")
        for index,cell in enumerate(stats["strata"]):
            row,col=divmod(index,4)
            label=f"{cell['qualified']}/{cell['planned']}" if cell["observed"]==cell["planned"] else f"{cell['qualified']} known / {cell['planned']}\n{cell['planned']-cell['observed']} unobserved"
            ax.text(col,row,label,ha="center",va="center",color="white" if values[row,col]>.6 else "#172554",fontsize=12)
        ax.set_xticks(range(4),["Baby A","Baby B","King A","King B"])
        ax.set_yticks(range(4),MAPS); ax.tick_params(length=0)
        temp=audit["versions"][version]["source_temperature"]
        ax.set_title(f"{version.upper()} source T={temp:g}: {stats['initial_qualified_slots']}/32 initial slots",fontweight="bold",pad=12)
    for column,split in enumerate(("training","validation")):
        ax=fig.add_subplot(grid[1,column]); total=32 if split=="training" else 16
        for row,version in enumerate(("v5","v6")):
            stats=audit["versions"][version]["splits"][split]
            initial=stats["initial_qualified_slots"]; extra=stats["reserve_slots_added"]
            ax.barh(row,total,color="#e2e8f0",height=.52,label="Unfilled / not collected" if row==0 else None)
            ax.barh(row,initial,color="#2563eb",height=.52,label="Initial games" if row==0 else None)
            ax.barh(row,extra,left=initial,color="#0f766e",height=.52,label="Reserve additions" if row==0 else None)
            if initial: ax.text(initial/2,row,str(initial),ha="center",va="center",color="white",fontweight="bold")
            if extra: ax.text(initial+extra/2,row,f"+{extra}",ha="center",va="center",color="white",fontweight="bold")
            suffix="; not collected" if not stats["initial_completed_source_games"] else ""
            ax.text(total+.4,row,f"{stats['final_accepted_slots']}/{total}{suffix}",va="center",fontsize=10)
        ax.set_yticks([0,1],["V5","V6"]); ax.invert_yaxis(); ax.set_xlim(0,total*1.5)
        ax.set_xticks([0,total//4,total//2,3*total//4,total]); ax.set_xlabel("Assigned slots; planned denominator retained")
        ax.set_title(f"{split.capitalize()} coverage after fixed reserves",fontsize=11,pad=10)
        if split=="training": legend_handles,legend_labels=ax.get_legend_handles_labels()
    fig.suptitle("Source sampling coverage · same 16 initial training seed pairs",fontweight="bold",fontsize=15,y=.97)
    fig.text(.5,.92,"Eligibility: first declared role/team turn with macro 37 legal. Initial panels exclude reserves.",ha="center",fontsize=10)
    fig.legend(legend_handles,legend_labels,loc="lower center",bbox_to_anchor=(.5,.10),ncol=3,fontsize=10,frameon=False)
    fig.text(.5,.025,"Coverage change, not playing strength. Source T changed; continuation T remains 0.35.\nValidation adds the input-overlap guard; V5 validation was never collected. No outcomes enter this comparison.",ha="center",fontsize=10)
    with tempfile.NamedTemporaryFile(dir=output,suffix=".png",delete=False) as stream: temporary=Path(stream.name)
    fig.savefig(temporary,dpi=160,facecolor="white",bbox_inches="tight"); plt.close(fig)
    os.replace(temporary,output/"source-coverage-comparison.png")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"progress/terminal-target-head-v6")
    args=parser.parse_args(); output=args.output_dir; output.mkdir(parents=True,exist_ok=True)
    audit=build_audit(args.root)
    atomic_text(output/"source-coverage-comparison.json",json.dumps(audit,sort_keys=True,indent=2,allow_nan=False)+"\n")
    plot(audit,output)
    lines=["# Paired source-slot coverage", "", "![Source coverage comparison](source-coverage-comparison.png)", "",
        "This compares the same 16 initial training engine/policy seed pairs. V5 source temperature was 0.35; v6 source temperature is 1.0 on both teams and all four development maps. Every source game was regenerated. Continuation temperature remains 0.35.","",
        "| Version | Initial training slots | Reserve games / slots added | Final training slots | Initial validation slots | Reserve games / slots added | Final validation slots |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for version,view in audit["versions"].items():
        t,v=(view["splits"][split] for split in ("training","validation"))
        lines.append(f"| {version} | {t['initial_qualified_slots']}/32 | {t['reserve_completed_source_games']} / +{t['reserve_slots_added']} | {t['final_accepted_slots']}/32 | {v['initial_qualified_slots']}/16{' (not collected)' if not v['initial_completed_source_games'] else ''} | {v['reserve_completed_source_games']} / +{v['reserve_slots_added']} | {v['final_accepted_slots']}/16 |")
    lines += ["", "The initial-training comparison is paired; reserve schedules and contributions are reported separately. V5 validation was not attempted, so 0/16 denotes missing coverage, not 16 failed games. Validation eligibility additionally excludes exact training/anchor inputs.", "",
        "This is evidence about source-slot availability under the temperature amendment. It does not measure playing strength or establish policy-transfer or victory criteria. Source winners, terminal scores and benchmark outcomes are neither selected nor compared.","",
        "Reporting note: the pinned generic v6 coverage graph labels fully scored validation roots as ‘Validation roots’. This source-only figure instead shows all 16 collected validation roots. The generic label should be clarified after actors are quiescent; no pinned file was changed here.","",
        "[Machine-readable audit and source SHA-256 inventory](source-coverage-comparison.json). The standalone [helper](source/source_coverage_plot.py) reads frozen source records/manifests only; it does not import actors or modify their pinned runtime.",""]
    atomic_text(output/"source-coverage-comparison.md","\n".join(lines))
    print(json.dumps({"paired_initial_games":audit["paired_initial_training_games_observed"],
        "initial_slots":{v:x["splits"]["training"]["initial_qualified_slots"] for v,x in audit["versions"].items()},
        "output_dir":str(output)},indent=2))


if __name__=="__main__": main()
