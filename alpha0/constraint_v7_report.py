"""Read-only, training-only v7 evidence reporting and independent classification."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import constraint_v7_common as C

ARMS={"kl_1":1.0,"kl_0":0.0}
STEPS=(128,512,2048)
PROOF_FLAGS=("passed","model_bytes_equal","optimizer_state_equal","torch_rng_equal",
             "prefix_numpy_rng_equal","batch_prefix_equal","loss_history_equal")


def require(value,message):
    if not value: raise ValueError(message)


def numeric(value,name,minimum=None):
    require(type(value) in (int,float) and math.isfinite(value),f"Invalid {name}")
    require(minimum is None or value>=minimum,f"Invalid {name} range")
    return float(value)


def close(actual,expected,name):
    require(math.isclose(numeric(actual,name),expected,rel_tol=2e-6,abs_tol=2e-6),f"Recorded {name} disagrees with evidence")


def validate_metrics(metrics,targets,baseline,masks,coefficient):
    """Recount every action; never infer agreement from a reduced denominator."""
    require(len(targets)==len(baseline)==len(masks)==32,"All 32 training inputs are required")
    changed=[i for i,(a,b) in enumerate(zip(targets,baseline)) if a!=b]
    retained=[i for i in range(32) if i not in changed]
    require(len(changed)==9 and len(retained)==23,"Frozen changed9/retained23 split differs")
    actions=metrics.get("root_actions",[])
    require(len(actions)==32 and all(type(a) is int and 0<=a<40 and masks[i][a] for i,a in enumerate(actions)),
            "Snapshot must retain all 32 legal selected greedy actions")
    counts={"root_count":32,"root_selected_action_matches":sum(a==b for a,b in zip(actions,targets)),
            "discordant_target_count":9,"discordant_target_matches":sum(actions[i]==targets[i] for i in changed),
            "retained_target_count":23,"retained_target_matches":sum(actions[i]==targets[i] for i in retained)}
    require(all(type(metrics.get(k)) is int and metrics[k]==v for k,v in counts.items()),"Recorded snapshot agreement/count differs from all 32 actions")
    close(metrics["root_target_agreement"],counts["root_selected_action_matches"]/32,"root agreement")
    close(metrics["root_action_changed_fraction"],sum(a!=b for a,b in zip(actions,baseline))/32,"changed root actions")
    n=metrics.get("anchor_action_changes")
    require(type(n) is int and 0<=n<=512,"Invalid512-anchor change count")
    close(metrics["anchor_action_changed_fraction"],n/512,"anchor action changes")
    for key in ("target_ce","anchor_kl","expected_anchor_kl","loss","expected_objective",
                "parameter_displacement_l2","target_ce_gradient_norm","balanced_anchor_kl_gradient_norm","expected_total_gradient_norm"):
        numeric(metrics[key],key,-1e-6)
    close(metrics["loss"],metrics["target_ce"]+coefficient*metrics["anchor_kl"],"storage-mean objective")
    close(metrics["expected_objective"],metrics["target_ce"]+coefficient*metrics["expected_anchor_kl"],"game/team-weighted objective")
    cosine=metrics.get("ce_kl_gradient_cosine")
    require(cosine is None or abs(numeric(cosine,"CE/KL gradient cosine"))<=1+1e-6,"Invalid gradient cosine")
    return {**metrics,"weighted_anchor_kl_contribution":coefficient*metrics["expected_anchor_kl"]}


def validate_history(history,step,coefficient):
    require(len(history)==step and [row.get("step") for row in history]==list(range(1,step+1)),"Observed loss history has missing or duplicate updates")
    output=[]
    for row in history:
        for key in ("loss","target_ce","anchor_kl"): numeric(row[key],key,-1e-6)
        close(row["loss"],row["target_ce"]+coefficient*row["anchor_kl"],"minibatch objective")
        output.append({**row,"weighted_anchor_kl":coefficient*row["anchor_kl"]})
    return output


def classify(snapshots,reproduction,*,fit_complete,integrity_verified):
    """Implement protocol order independently of the fitter's classifier."""
    available=sum(len(snapshots.get(arm,{})) for arm in ARMS)
    all_six=(set(snapshots)==set(ARMS) and all(set(snapshots[arm])=={str(n) for n in STEPS} for arm in ARMS))
    proof_ok=bool(reproduction and reproduction.get("updates")==128 and all(reproduction.get(k) is True for k in PROOF_FLAGS))
    decision={"complete":False,"classification":None,"snapshots_observed":available,"snapshots_required":6,
              "reproduction_verified":proof_ok,"integrity_verified":integrity_verified is True,
              "automatic_promotion":False,"all_map_victory":False,"training_data_only":True,
              "first_full_match_step":{arm:None for arm in ARMS},"regressed_after_full_match":{arm:False for arm in ARMS},
              "budget_evidence":False,"anchoring_evidence":False}
    if not (all_six and proof_ok and fit_complete is True and integrity_verified is True):
        decision["reason"]="No completed classification without all six fixed snapshots, complete fit, exact 128 reproduction and final integrity."
        return decision
    for arm in ARMS:
        for step in STEPS:
            row=snapshots[arm][str(step)]
            require(row.get("root_count")==32 and type(row.get("root_selected_action_matches")) is int and
                    0<=row["root_selected_action_matches"]<=32,"Classification requires32 targets at every fixed snapshot")
        matches=[step for step in STEPS if snapshots[arm][str(step)]["root_selected_action_matches"]==32]
        first=min(matches) if matches else None
        decision["first_full_match_step"][arm]=first
        decision["regressed_after_full_match"][arm]=bool(first and any(step>first and snapshots[arm][str(step)]["root_selected_action_matches"]<32 for step in STEPS))
    require(decision["first_full_match_step"]["kl_1"]!=128,"Exact v6 reproduction cannot already match all 32 targets")
    budget=decision["first_full_match_step"]["kl_1"] in (512,2048)
    anchoring=not budget and decision["first_full_match_step"]["kl_1"] is None and decision["first_full_match_step"]["kl_0"] is not None
    decision.update(complete=True,classification="budget" if budget else "anchoring" if anchoring else "neither",
                    budget_evidence=budget,anchoring_evidence=anchoring)
    return decision


def resolve(value):
    path=Path(value)
    return path if path.is_absolute() else C.ROOT/path


def report(run_dir,progress_dir,config,identity,*,integrity_verified=False):
    """Read training artifacts only. Never call fitting, inference or an engine."""
    import numpy as np
    run_dir,progress_dir=Path(run_dir),Path(progress_dir)
    require(identity.get("config")==config and config.get("experiment_id")=="cached-head-constraint-v7","Report identity/config differs")
    require(config["updates"]==2048 and config["snapshots"]==list(STEPS) and config["training_roots"]==32 and config["anchor_count"]==512,
            "Diagnostic allocation differs")
    evidence={}
    def check(path,digest=None):
        path=Path(path); observed=C.sha256(path)
        require(digest is None or observed==digest,f"Artifact checksum changed: {path}")
        evidence[str(path.resolve())]=observed
        return path
    def read(path):
        return json.loads(check(path).read_text())
    # Prerequisite audits stay opaque: hash them, never parse held-out outcomes.
    for name,digest in {**identity.get("inputs",{}),**identity.get("sources",{})}.items(): check(resolve(name),digest)
    dataset=check(resolve(config["dataset"]["path"]),config["dataset"]["sha256"])
    with np.load(dataset,allow_pickle=False) as arrays:
        targets=arrays["terminal_actions"].tolist(); prior=arrays["root_prior"].copy(); masks=arrays["root_mask"].tolist()
    require(prior.shape==(32,40) and len(targets)==32 and len(masks)==32,"Frozen training label dimensions differ")
    baseline=prior.argmax(-1).tolist()
    require(sum(a!=b for a,b in zip(targets,baseline))==9,"Frozen nine changed labels differ")
    directory=run_dir/"fit"; fit_path=directory/"identity.json"
    fit_identity=read(fit_path) if fit_path.exists() else None
    fit_hash=C.fingerprint(fit_identity) if fit_identity else None
    if fit_identity:
        require(fit_identity.get("identity_sha256")==C.fingerprint(identity),"Fitting belongs to another experiment")
    proof_path=directory/"reproduction-128.json"
    proof=read(proof_path) if proof_path.exists() else None
    if proof:
        require(proof.get("reference_checkpoint_sha256")==config["reference_terminal_checkpoint"]["sha256"] and
                proof.get("fit_identity_sha256")==fit_hash,"Reproduction proof references another fit")
    arms={arm:{"coefficient":coefficient,"completed_updates":0,"history":[],"snapshots":{}} for arm,coefficient in ARMS.items()}
    records={arm:{} for arm in ARMS}
    for arm,coefficient in ARMS.items():
        histories=[]
        for step in STEPS:
            sidecar=directory/arm/f"checkpoint-{step:04d}.json"
            if not sidecar.exists(): continue
            record=read(sidecar); path=resolve(record["path"])
            require(path.resolve()==sidecar.with_suffix(".pt").resolve(),"Snapshot path lies outside this run")
            check(path,record["sha256"])
            require(record.get("arm")==arm and record.get("updates")==step and record.get("anchor_kl_weight")==coefficient and
                    record.get("fit_identity_sha256")==fit_hash and record.get("trainable_parameters")==5160 and
                    record.get("frozen_tensors_equal") is True and record.get("value_logits_equal") is True,"Snapshot metadata/proof differs")
            import torch
            payload=torch.load(path,map_location="cpu",weights_only=True)
            require(payload.get("identity_sha256")==C.fingerprint(identity) and payload.get("fit_identity_sha256")==fit_hash and
                    payload.get("arm")==arm and payload.get("updates")==step and payload.get("anchor_kl_weight")==coefficient and
                    payload.get("fit_metrics")==record["metrics"] and payload.get("parent_checkpoint_sha256")==config["checkpoint"]["sha256"] and
                    payload.get("dataset_sha256")==config["dataset"]["sha256"] and payload.get("benchmark_data_used") is False and
                    payload.get("validation_data_used") is False and payload.get("automatic_promotion") is False and
                    payload.get("policy_head_only") is True and payload.get("trainable_parameters")==5160 and
                    payload.get("gradient_updates")==step,"Snapshot payload provenance differs")
            metrics=validate_metrics(record["metrics"],targets,baseline,masks,coefficient)
            history=validate_history(payload["fit_history"],step,coefficient); histories.append(history)
            records[arm][str(step)]=record
            arms[arm]["snapshots"][str(step)]={"step":step,"checkpoint_sha256":record["sha256"],"metrics":metrics}
        # Recovery histories are observations, never additional classification snapshots.
        generations=sorted((directory/arm/"states").glob("step-*.json"))
        if generations:
            sidecar=generations[-1]; record=read(sidecar); step=record["step"]
            require(type(step) is int and step in range(16,2049,16) and sidecar.name==f"step-{step:04d}.json" and
                    record.get("fit_identity_sha256")==fit_hash,"Partial recovery identity differs")
            path=check(sidecar.with_suffix(".pt"),record["sha256"])
            import torch
            payload=torch.load(path,map_location="cpu",weights_only=True)
            require(payload.get("arm")==arm and payload.get("step")==step and payload.get("coefficient")==coefficient,
                    "Partial recovery arm/update differs")
            histories.append(validate_history(payload["history"],step,coefficient))
        if histories:
            history=max(histories,key=len)
            require(all(history[:len(prefix)]==prefix for prefix in histories),"Immutable snapshot loss histories conflict")
            arms[arm].update(completed_updates=len(history),history=history)
    manifest_path=directory/"manifest.json"
    manifest=read(manifest_path) if manifest_path.exists() else None
    if manifest:
        require(manifest.get("complete") is True and manifest.get("identity_sha256")==C.fingerprint(identity) and
                manifest.get("fit_identity")==fit_identity and manifest.get("fit_identity_sha256")==fit_hash and
                manifest.get("checkpoints")==records and manifest.get("reproduction")==proof and
                manifest.get("updates_per_arm")==2048 and manifest.get("snapshots")==list(STEPS) and
                manifest.get("training_data_only") is True and manifest.get("validation_data_used") is False and
                manifest.get("benchmark_data_used") is False,"Complete fit manifest disagrees with observed artifacts")
        require(read(progress_dir/"fit-summary.json")==manifest,"Public fit summary differs")
        for name,digest in manifest["files"].items():
            path=(directory/name).resolve(); require(path.is_relative_to(directory.resolve()),"Fit artifact escapes its directory")
            check(path,digest)
    metrics={arm:{step:row["metrics"] for step,row in view["snapshots"].items()} for arm,view in arms.items()}
    fit_complete=bool(manifest and all(view["completed_updates"]==2048 for view in arms.values()))
    raw=classify(metrics,proof,fit_complete=fit_complete,integrity_verified=True)
    if manifest:
        require(raw["complete"],"Completed manifest lacks six verified snapshots/reproduction")
        recorded=manifest["classification"]
        for key in ("classification","first_full_match_step","budget_evidence","anchoring_evidence","automatic_promotion"):
            require(recorded.get(key)==raw[key],"Fitter classification disagrees with independent protocol order")
    decision=classify(metrics,proof,fit_complete=fit_complete,integrity_verified=integrity_verified)
    require(all(C.sha256(path)==digest for path,digest in evidence.items()),"Evidence changed during report")
    return {"schema_version":1,"experiment_id":config["experiment_id"],"created_at":C.now(),
        "identity_sha256":C.fingerprint(identity),"fit_identity_sha256":fit_hash,"training_only":True,
        "validation_data_read":False,"benchmark_data_read":False,"engine_games":0,"automatic_promotion":False,
        "complete":decision["complete"],"decision":decision,"reproduction":proof,
        "allocation":{"roots":32,"changed_targets":9,"retained_targets":23,"anchors":512,"trainable_parameters":5160,
                      "updates_per_arm":2048,"fixed_snapshots":list(STEPS)},
        "root_targets":targets,"root_baseline_actions":baseline,"arms":arms,"source_file_hashes":evidence,
        "scope":"Training-only budget/anchor diagnostic; no held-out evaluation, deployment recommendation or victory claim."}


def write_report(summary,progress_dir):
    from scripts.plot_constraint_v7 import generate
    progress_dir=Path(progress_dir)
    C.atomic_json(progress_dir/"summary.json",summary)
    return generate(summary,progress_dir)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=C.RUN)
    parser.add_argument("--progress-dir",type=Path,default=C.PROGRESS)
    args=parser.parse_args(); config=C.load_config()
    identity=json.loads((args.progress_dir/"identity.json").read_text())
    write_report(report(args.run_dir,args.progress_dir,config,identity),args.progress_dir)


if __name__=="__main__": main()
