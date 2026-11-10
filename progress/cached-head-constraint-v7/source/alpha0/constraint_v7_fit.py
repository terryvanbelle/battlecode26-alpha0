"""Training-only v7 budget/anchor diagnostic on immutable v6 cached features.

Both arms begin at cp2. Numerical helpers are the pinned, unchanged v6 helpers;
only the update budget and the preregistered KL coefficient differ. No engine,
validation observation, benchmark result, or policy promotion is used here.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import constraint_v7_common as C
from .learn_v2 import load_checkpoint
from . import terminal_v6_fit as V6
from .terminal_v6_fit import (
    make_batch_schedule, batch_fingerprint, normalize_masked_prior, head_loss,
    freeze_policy_head, frozen_state_sha256, tensor_bytes, _tree_equal,
)

ARMS = {"kl_1": 1.0, "kl_0": 0.0}
SNAPSHOTS = (128, 512, 2048)
FIT_CONFIG = {**V6.FIT_CONFIG, "updates": 2048}
POLICY_KEYS = frozenset(("policy.weight", "policy.bias"))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _optimizer(parameters, settings):
    return torch.optim.AdamW(parameters, lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"], betas=tuple(settings["betas"]), eps=settings["eps"])


def _verify_optimizer(saved, parameters, settings, step):
    fresh = _optimizer(parameters, settings)
    _require(_tree_equal(saved["param_groups"], fresh.state_dict()["param_groups"]), "optimizer settings/flags differ")
    expected = {0, 1} if step else set()
    _require(set(saved["state"]) == expected, "optimizer parameter-state inventory differs")
    for index, parameter in enumerate(parameters):
        if not step:
            continue
        value = saved["state"][index]
        _require(set(value) == {"step", "exp_avg", "exp_avg_sq"}, "optimizer state keys differ")
        _require(value["step"].dtype == torch.float32 and value["step"].shape == torch.Size([])
                 and float(value["step"]) == step, "optimizer update count differs")
        for name in ("exp_avg", "exp_avg_sq"):
            _require(value[name].dtype == parameter.dtype and value[name].shape == parameter.shape
                     and bool(torch.isfinite(value[name]).all()), "invalid optimizer moment")
        _require(bool((value["exp_avg_sq"] >= 0).all()), "negative optimizer second moment")
    fresh.load_state_dict(deepcopy(saved))
    _require(_tree_equal(fresh.state_dict(), saved), "optimizer state does not restore exactly")


def anchor_weights(arrays):
    games, teams = arrays["anchor_game"], arrays["anchor_player"]
    ids = sorted(set(games.tolist()))
    result = np.zeros(len(games), np.float64)
    for game in ids:
        for team in (0, 1):
            selected = (games == game) & (teams == team)
            _require(selected.any(), "missing game/team anchor group")
            result[selected] = 1 / (len(ids) * 2 * int(selected.sum()))
    _require(abs(float(result.sum()) - 1) < 1e-12, "anchor weights do not sum to one")
    return torch.as_tensor(result, dtype=torch.float32)


def cached_metrics(model, arrays, coefficient, base_model=None):
    with torch.no_grad():
        rh = torch.as_tensor(arrays["root_hidden"])
        ah = torch.as_tensor(arrays["anchor_hidden"])
        logits, anchor_logits = model.policy(rh), model.policy(ah)
        _, result = head_loss(logits, anchor_logits, arrays["root_prior"], arrays["root_mask"],
            arrays["anchor_prior"], arrays["anchor_mask"], arrays["terminal_actions"], anchor_kl_weight=coefficient)
        rlog = V6._log_policy(logits, arrays["root_mask"])
        alog = V6._log_policy(anchor_logits, arrays["anchor_mask"])
        chosen, anchors = rlog.argmax(-1).numpy(), alog.argmax(-1).numpy()
        target = arrays["terminal_actions"]
        changed = target != arrays["root_prior"].argmax(-1)
        p0 = torch.as_tensor(normalize_masked_prior(arrays["anchor_prior"], arrays["anchor_mask"]))
        per_anchor = (torch.xlogy(p0, p0) - p0 * alog).sum(-1)
        expected_kl = float((per_anchor * anchor_weights(arrays)).sum())
        result.update(root_count=len(target), root_selected_action_matches=int((chosen == target).sum()),
            root_target_agreement=float(np.mean(chosen == target)),
            discordant_target_count=int(changed.sum()), discordant_target_matches=int(((chosen == target) & changed).sum()),
            retained_target_count=int((~changed).sum()), retained_target_matches=int(((chosen == target) & ~changed).sum()),
            root_action_changed_fraction=float(np.mean(chosen != arrays["root_prior"].argmax(-1))),
            anchor_action_changes=int((anchors != arrays["anchor_prior"].argmax(-1)).sum()),
            anchor_action_changed_fraction=float(np.mean(anchors != arrays["anchor_prior"].argmax(-1))),
            expected_anchor_kl=expected_kl, expected_objective=result["target_ce"] + coefficient * expected_kl,
            root_actions=chosen.tolist())
        result["parameter_displacement_l2"] = (float(torch.cat([
            (model.policy.weight-base_model.policy.weight).reshape(-1),
            (model.policy.bias-base_model.policy.bias).reshape(-1)]).norm()) if base_model is not None else None)
    # Diagnostic derivatives use detached copies; no model gradient/optimizer is changed.
    with torch.enable_grad():
        weight = model.policy.weight.detach().clone().requires_grad_(True)
        bias = model.policy.bias.detach().clone().requires_grad_(True)
        rlog = V6._log_policy(F.linear(rh, weight, bias), arrays["root_mask"])
        alog = V6._log_policy(F.linear(ah, weight, bias), arrays["anchor_mask"])
        rp = torch.as_tensor(normalize_masked_prior(arrays["root_prior"], arrays["root_mask"]))
        ap = torch.as_tensor(normalize_masked_prior(arrays["anchor_prior"], arrays["anchor_mask"]))
        target_q = .25*rp + .75*F.one_hot(torch.as_tensor(arrays["terminal_actions"]), 40).float()
        ce = -(target_q*rlog).sum(-1).mean()
        kl = ((torch.xlogy(ap, ap)-ap*alog).sum(-1)*anchor_weights(arrays)).sum()
        cg = torch.cat([g.reshape(-1) for g in torch.autograd.grad(ce, (weight, bias), retain_graph=True)])
        kg = torch.cat([g.reshape(-1) for g in torch.autograd.grad(kl, (weight, bias))])
        total = cg + coefficient*kg
        denominator = float(cg.norm()*kg.norm())
        result.update(target_ce_gradient_norm=float(cg.norm()), balanced_anchor_kl_gradient_norm=float(kg.norm()),
            expected_total_gradient_norm=float(total.norm()),
            ce_kl_gradient_cosine=float(torch.dot(cg, kg))/denominator if denominator else None)
    return result


def _state(model, optimizer, history, batches, settings, arm, step):
    return {"arm": arm, "coefficient": ARMS[arm], "step": step,
        "policy_state_dict": deepcopy(model.policy.state_dict()),
        "optimizer_state_dict": deepcopy(optimizer.state_dict()),
        "torch_rng": torch.get_rng_state().clone(), "numpy_rng": deepcopy(batches["rng_state"]),
        "batch_sha256": batch_fingerprint(batches), "frozen_state_sha256": frozen_state_sha256(model),
        "fit_config_sha256": C.fingerprint(settings), "history": deepcopy(history)}


def _verify_state(state, base_model, batches, settings, arm):
    step = state.get("step")
    _require(type(step) is int and 0 <= step <= settings["updates"], "invalid saved update index")
    _require(state.get("arm") == arm and state.get("coefficient") == ARMS[arm], "saved arm/coefficient differs")
    _require(state.get("batch_sha256") == batch_fingerprint(batches)
        and state.get("fit_config_sha256") == C.fingerprint(settings)
        and state.get("frozen_state_sha256") == frozen_state_sha256(base_model), "saved fit provenance differs")
    _require(_tree_equal(state.get("numpy_rng"), batches["rng_state"]), "saved NumPy schedule state differs")
    seeded = torch.Generator().manual_seed(settings["seed"]).get_state()
    _require(_tree_equal(state.get("torch_rng"), seeded), "saved Torch RNG differs")
    history = state.get("history", [])
    _require(len(history) == step and [row.get("step") for row in history] == list(range(1, step + 1)),
             "saved history/update index differs")
    _require(all(all(np.isfinite(row.get(key, np.nan)) for key in ("loss", "target_ce", "anchor_kl")) for row in history),
             "nonfinite saved loss history")
    policy = state.get("policy_state_dict", {})
    _require(set(policy) == {"weight", "bias"}, "saved policy inventory differs")
    for key, tensor in base_model.policy.state_dict().items():
        _require(policy[key].dtype == tensor.dtype and policy[key].shape == tensor.shape
                 and bool(torch.isfinite(policy[key]).all()), "invalid saved policy tensor")
    _verify_optimizer(state["optimizer_state_dict"], list(base_model.policy.parameters()), settings, step)


def fit_cached_head(base_model, arrays, batches, fit_config, arm, resume=None, on_update=None, check_stop=None):
    """One deterministic arm; callbacks are immutable recovery generations only."""
    _require(arm in ARMS, "unknown diagnostic arm")
    _require(batches["root"].shape == (fit_config["updates"], fit_config["batch_size"])
        and batches["anchor"].shape == (fit_config["updates"], fit_config["anchor_batch_size"]), "batch shape differs")
    model = deepcopy(base_model).cpu()
    parameters = freeze_policy_head(model)
    optimizer = _optimizer(parameters, fit_config)
    start, history = 0, []
    if resume is not None:
        _verify_state(resume, base_model, batches, fit_config, arm)
        model.policy.load_state_dict(resume["policy_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start, history = resume["step"], deepcopy(resume["history"])
    rh = torch.as_tensor(arrays["root_hidden"], dtype=torch.float32)
    ah = torch.as_tensor(arrays["anchor_hidden"], dtype=torch.float32)
    actions = arrays["terminal_actions"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(fit_config["seed"])
        if resume is not None:
            torch.set_rng_state(resume["torch_rng"])
        for update in range(start, fit_config["updates"]):
            if check_stop:
                check_stop()
            r, a = batches["root"][update], batches["anchor"][update]
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = head_loss(model.policy(rh[r]), model.policy(ah[a]), arrays["root_prior"][r],
                arrays["root_mask"][r], arrays["anchor_prior"][a], arrays["anchor_mask"][a], actions[r],
                target_weight=fit_config["target_weight"], anchor_kl_weight=ARMS[arm])
            loss.backward()
            _require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters), "invalid policy gradient")
            optimizer.step()
            history.append({"step": update + 1, **metrics})
            step = update + 1
            if step % 16 == 0 or step == fit_config["updates"]:
                state = _state(model, optimizer, history, batches, fit_config, arm, step)
                if on_update:
                    on_update(state)
        if start == fit_config["updates"]:
            state = deepcopy(resume)
    _require(frozen_state_sha256(model) == frozen_state_sha256(base_model), "fit changed a frozen tensor")
    with torch.no_grad():
        hidden = torch.cat((rh, ah))
        _require(torch.equal(model.value(hidden), base_model.value(hidden)), "fit changed frozen critic outputs")
    return model, optimizer, state, cached_metrics(model, arrays, ARMS[arm], base_model)


def prove_prefix_reproduction(model_state, optimizer_state, torch_rng, history, batches,
                              reference_payload, reference_batches, prefix_rng_state):
    """Compare the original 128-update prefix, never the extended schedule end."""
    _require(_tree_equal(model_state, reference_payload["model_state_dict"]), "128-update model differs from v6")
    _require(_tree_equal(optimizer_state, reference_payload["optimizer_state_dict"]), "128-update optimizer differs from v6")
    _require(_tree_equal(torch_rng, reference_payload["torch_rng"]), "128-update Torch RNG differs from v6")
    _require(_tree_equal(history, reference_payload["fit_history"]), "128-update loss history differs from v6")
    for key in ("root", "anchor"):
        expected = np.asarray(reference_batches[key])
        _require(expected.shape == (128, 32) and expected.dtype == np.int64
            and np.array_equal(batches[key][:128], expected)
            and np.array_equal(reference_payload["batch_indices"][key].numpy(), expected), "original batch prefix differs")
    _require(_tree_equal(prefix_rng_state, reference_payload["numpy_rng"]), "128-prefix sampling RNG differs from v6")
    _require(len(history) == 128, "reproduction proof must end at update128")
    return {"passed": True, "updates": 128, "model_bytes_equal": True, "optimizer_state_equal": True,
        "torch_rng_equal": True, "prefix_numpy_rng_equal": True, "batch_prefix_equal": True,
        "loss_history_equal": True, "extended_schedule_rng_is_distinct_metadata": True}


def classify_results(results):
    _require(set(results) == set(ARMS), "both diagnostic arms required")
    first = {}
    for arm in ARMS:
        _require(set(results[arm]) == {str(step) for step in SNAPSHOTS}, "all six fixed snapshots required")
        steps = []
        for step in SNAPSHOTS:
            row = results[arm][str(step)]
            count = row.get("root_selected_action_matches")
            _require(row.get("root_count") == 32 and type(count) is int and 0 <= count <= 32,
                     "invalid fixed-snapshot agreement")
            if count == 32:
                steps.append(step)
        first[arm] = min(steps) if steps else None
    _require(first["kl_1"] != 128, "v6 reproduction baseline unexpectedly fits all32")
    budget = first["kl_1"] in (512, 2048)
    anchoring = first["kl_1"] is None and first["kl_0"] is not None
    return {"complete": True, "classification": "budget" if budget else "anchoring" if anchoring else "neither",
        "first_full_match_step": first, "budget_evidence": budget, "anchoring_evidence": anchoring,
        "automatic_promotion": False, "training_data_only": True}


def _resolve(spec):
    path = Path(spec["path"])
    return path if path.is_absolute() else C.ROOT / path


def _path(path):
    path = Path(path).resolve()
    return str(path.relative_to(C.ROOT)) if path.is_relative_to(C.ROOT) else str(path)


def _load_inputs(config, identity):
    _require(config["experiment_id"] == "cached-head-constraint-v7" and identity["config"] == config, "v7 identity/config differs")
    for name, path in (("alpha0/constraint_v7_fit.py", __file__), ("alpha0/constraint_v7_common.py", C.__file__),
                       ("alpha0/terminal_v6_fit.py", V6.__file__), ("alpha0/terminal_v6_common.py", V6.C.__file__)):
        _require(identity.get("sources", {}).get(name) == C.sha256(path), "fit source differs from frozen identity: " + name)
    _require(config["updates"] == 2048 and config["snapshots"] == list(SNAPSHOTS)
        and config["checkpoint_interval"] == 16, "diagnostic update/snapshot protocol differs")
    _require(config["arms"] == [{"id": arm, "anchor_kl_weight": value} for arm, value in ARMS.items()], "diagnostic arms differ")
    settings = {**config["fit"], "updates": config["updates"], "anchor_kl_weight": 1}
    _require(settings == FIT_CONFIG and torch.get_num_threads() == 1, "fit settings/thread count differ")
    names = ("checkpoint", "dataset", "dataset_metadata", "reference_terminal_checkpoint", "reference_batches", "v6_identity")
    paths = {name: _resolve(config[name]) for name in names}
    for name, path in paths.items():
        _require(C.sha256(path) == config[name]["sha256"], "pinned input differs: " + name)
    _require(paths["dataset_metadata"] == paths["dataset"].with_suffix(".json"), "training sidecar path differs")
    v6_identity = json.loads(paths["v6_identity"].read_text())
    arrays, metadata, guard = V6.load_dataset(paths["dataset"], v6_identity["config"], v6_identity)
    base_model, base_payload = load_checkpoint(paths["checkpoint"])
    V6._verify_cached_priors(base_model, arrays)
    _, reference = load_checkpoint(paths["reference_terminal_checkpoint"])
    _require(reference["dataset_sha256"] == config["dataset"]["sha256"]
        and reference["parent_checkpoint_sha256"] == config["checkpoint"]["sha256"]
        and reference["arm"] == "terminal_head" and reference["updates"] == 128, "reference fit provenance differs")
    _require(np.array_equal(arrays["puct_actions"], arrays["root_prior"].argmax(-1)),
             "v6 PUCT actions must coincide with cp2 greedy for this diagnostic")
    _require(int((arrays["terminal_actions"] != arrays["root_prior"].argmax(-1)).sum()) == 9, "changed-label count differs")
    with np.load(paths["reference_batches"], allow_pickle=False) as saved:
        _require(set(saved.files) == {"root", "anchor"}, "reference batch inventory differs")
        reference_batches = {key: saved[key].copy() for key in saved.files}
    batches = make_batch_schedule(arrays["root_strata"], arrays["anchor_game"], arrays["anchor_player"], settings)
    prefix = make_batch_schedule(arrays["root_strata"], arrays["anchor_game"], arrays["anchor_player"], {**settings, "updates": 128})
    for key in ("root", "anchor"):
        _require(np.array_equal(batches[key][:128], reference_batches[key])
            and np.array_equal(prefix[key], reference_batches[key]), "extended schedule changed original prefix")
    fit_identity = {"identity_sha256": C.fingerprint(identity), "input_hashes": {name: config[name]["sha256"] for name in names},
        "batch_sha256": batch_fingerprint(batches), "settings": settings,
        "fitter_sha256": C.sha256(__file__), "common_sha256": C.sha256(C.__file__),
        "v6_fitter_sha256": C.sha256(V6.__file__), "torch_version": str(torch.__version__), "numpy_version": np.__version__}
    return base_model, base_payload, arrays, metadata, guard, batches, prefix, reference, reference_batches, fit_identity


def _model_from_state(base_model, state):
    model = deepcopy(base_model).cpu()
    freeze_policy_head(model)
    model.policy.load_state_dict(state["policy_state_dict"])
    return model


def _candidate(base_model, base_payload, arrays, metadata, guard, batches, settings, config, identity, fit_identity, state):
    model = _model_from_state(base_model, state)
    metrics = cached_metrics(model, arrays, state["coefficient"], base_model)
    parent = {key: deepcopy(value) for key, value in base_payload.items()
              if key not in ("model_state_dict", "optimizer_state_dict", "torch_rng", "numpy_rng")}
    lineage_config = deepcopy(config)
    lineage_config["initialization"] = {"artifact": config["checkpoint"]["path"],
        "artifact_sha256": config["checkpoint"]["sha256"], "method": "fresh cp2 frozen features; training-only KL/budget diagnostic"}
    payload = model.checkpoint(experiment_id=config["experiment_id"], arm=state["arm"],
        anchor_kl_weight=state["coefficient"], config=lineage_config, iteration=0, updates=state["step"],
        gradient_updates=state["step"], trainable_parameters=5160, policy_head_only=True,
        training_stage="training-only cached-head budget/constraint mechanism diagnostic",
        parent_checkpoint_sha256=config["checkpoint"]["sha256"], parent_checkpoint_metadata=parent,
        training_maps=sorted(set(metadata["training_maps"]) | set(base_payload.get("training_maps", []))),
        training_seeds=metadata["training_seeds"], games_total=len(metadata["training_seeds"]), positions_total=544,
        benchmark_data_used=False, validation_data_used=False, automatic_promotion=False,
        dataset_sha256=config["dataset"]["sha256"], dataset_metadata_sha256=config["dataset_metadata"]["sha256"],
        reference_v6_checkpoint_sha256=config["reference_terminal_checkpoint"]["sha256"],
        identity_sha256=C.fingerprint(identity), fit_identity_sha256=C.fingerprint(fit_identity),
        frozen_state_sha256=state["frozen_state_sha256"], batch_sha256=state["batch_sha256"],
        batch_indices={key: torch.from_numpy(batches[key].copy()) for key in ("root", "anchor")},
        optimizer_state_dict=state["optimizer_state_dict"], torch_rng=state["torch_rng"], numpy_rng=state["numpy_rng"],
        fit_history=state["history"], fit_metrics=metrics, gradient_input_overlap=guard,
        training_randomness={"fit_seed": settings["seed"], "source_games": metadata["training_games"],
                             "selection_seeds": [203003, 203013], "anchor_seed": 203151})
    return model, payload, metrics


def _verify_candidate(payload, base_model, arrays, batches, settings, config, fit_identity, arm, step):
    _require(payload.get("arm") == arm and payload.get("updates") == step and payload.get("gradient_updates") == step
        and payload.get("anchor_kl_weight") == ARMS[arm] and payload.get("fit_identity_sha256") == C.fingerprint(fit_identity),
        "candidate identity/arm/update differs")
    _require(payload.get("parent_checkpoint_sha256") == config["checkpoint"]["sha256"]
        and payload.get("dataset_sha256") == config["dataset"]["sha256"]
        and payload.get("dataset_metadata_sha256") == config["dataset_metadata"]["sha256"]
        and payload.get("reference_v6_checkpoint_sha256") == config["reference_terminal_checkpoint"]["sha256"],
        "candidate input lineage differs")
    _require(payload.get("trainable_parameters") == 5160 and payload.get("policy_head_only") is True
        and payload.get("benchmark_data_used") is False and payload.get("validation_data_used") is False
        and payload.get("automatic_promotion") is False, "candidate scope differs")
    expected_config = deepcopy(config)
    expected_config["initialization"] = {"artifact": config["checkpoint"]["path"],
        "artifact_sha256": config["checkpoint"]["sha256"], "method": "fresh cp2 frozen features; training-only KL/budget diagnostic"}
    _require(payload.get("identity_sha256") == fit_identity["identity_sha256"]
        and payload.get("config") == expected_config, "candidate config/initialization identity differs")
    source = base_model.state_dict(); saved = payload["model_state_dict"]
    _require(set(source) == set(saved), "candidate tensor inventory differs")
    for key, value in source.items():
        _require(saved[key].dtype == value.dtype and saved[key].shape == value.shape
                 and bool(torch.isfinite(saved[key]).all()), "candidate tensor is invalid")
        if key not in POLICY_KEYS:
            _require(tensor_bytes(saved[key]) == tensor_bytes(value), "candidate changed frozen tensor")
    state = {"arm": arm, "coefficient": ARMS[arm], "step": step,
        "policy_state_dict": {key: saved["policy." + key] for key in ("weight", "bias")},
        "optimizer_state_dict": payload["optimizer_state_dict"], "torch_rng": payload["torch_rng"],
        "numpy_rng": payload["numpy_rng"], "batch_sha256": payload["batch_sha256"],
        "frozen_state_sha256": payload["frozen_state_sha256"], "fit_config_sha256": C.fingerprint(settings),
        "history": payload["fit_history"]}
    _verify_state(state, base_model, batches, settings, arm)
    for key in ("root", "anchor"):
        _require(payload["batch_indices"][key].dtype == torch.int64
                 and np.array_equal(payload["batch_indices"][key].numpy(), batches[key]), "candidate schedule differs")
    model = _model_from_state(base_model, state)
    metrics = cached_metrics(model, arrays, ARMS[arm], base_model)
    _require(metrics == payload["fit_metrics"], "candidate recorded metrics differ")
    with torch.no_grad():
        hidden = torch.as_tensor(np.concatenate((arrays["root_hidden"], arrays["anchor_hidden"])))
        _require(torch.equal(model.value(hidden), base_model.value(hidden)), "candidate critic outputs differ")
    return model, state, metrics


def run_fit(config, identity, deadline):
    """Worker entrypoint. Supervisor owns process-group deadline and run locks."""
    C.check_stop(deadline)
    loaded = _load_inputs(config, identity)
    base_model, base_payload, arrays, metadata, guard, batches, prefix, reference, reference_batches, fit_identity = loaded
    settings = fit_identity["settings"]; fit_hash = C.fingerprint(fit_identity)
    directory = C.RUN / "fit"
    if (directory / "manifest.json").exists():
        result = verify_completed(config, identity, loaded=loaded)
        C.immutable_json(C.PROGRESS / "fit-summary.json", result)
        return result
    directory.mkdir(parents=True, exist_ok=True)
    C.immutable_json(directory / "identity.json", fit_identity)
    C.immutable_npz(directory / "batches.npz", {key: batches[key] for key in ("root", "anchor")})
    C.immutable_json(directory / "batch-rng.json", batches["rng_state"])
    parent = {key: deepcopy(value) for key, value in base_payload.items()
              if key not in ("model_state_dict", "optimizer_state_dict", "torch_rng", "numpy_rng")}
    C.immutable_json(directory / "parent-metadata.json", parent)
    results = {arm: {} for arm in ARMS}
    proof = None
    for arm in ARMS:
        C.check_stop(deadline)
        arm_dir = directory / arm; generations = arm_dir / "states"
        generations.mkdir(parents=True, exist_ok=True)
        states = {}
        for sidecar in sorted(generations.glob("step-*.json")):
            record = json.loads(sidecar.read_text()); path = sidecar.with_suffix(".pt")
            step = record.get("step")
            _require(type(step) is int and 16 <= step <= 2048 and step % 16 == 0
                and sidecar.name == f"step-{step:04d}.json" and record.get("fit_identity_sha256") == fit_hash
                and C.sha256(path) == record.get("sha256"), "saved recovery generation differs")
            state = torch.load(path, map_location="cpu", weights_only=True)
            _verify_state(state, base_model, batches, settings, arm)
            _require(state["step"] == step, "generation update differs")
            states[step] = state
        if states:
            _require(sorted(states) == list(range(16, max(states) + 1, 16)), "recovery generation sequence has gaps")

        def publish_snapshot(state):
            nonlocal proof
            step = state["step"]
            if step not in SNAPSHOTS:
                return
            C.check_stop(deadline)
            model, payload, metrics = _candidate(base_model, base_payload, arrays, metadata, guard, batches,
                settings, config, identity, fit_identity, state)
            if arm == "kl_1" and step == 128:
                proof = prove_prefix_reproduction(model.state_dict(), state["optimizer_state_dict"],
                    state["torch_rng"], state["history"], batches, reference, reference_batches, prefix["rng_state"])
                proof.update(reference_checkpoint_sha256=config["reference_terminal_checkpoint"]["sha256"],
                    fit_identity_sha256=fit_hash)
                C.immutable_json(directory / "reproduction-128.json", proof)
            path = arm_dir / f"checkpoint-{step:04d}.pt"
            digest = V6._immutable_torch(path, payload)
            record = {"path": _path(path), "sha256": digest, "fit_identity_sha256": fit_hash,
                "arm": arm, "updates": step, "anchor_kl_weight": ARMS[arm], "trainable_parameters": 5160,
                "frozen_tensors_equal": True, "value_logits_equal": True, "metrics": metrics}
            C.immutable_json(path.with_suffix(".json"), record)
            results[arm][str(step)] = record

        for step in SNAPSHOTS:
            if step in states:
                publish_snapshot(states[step])

        def persist(state):
            C.check_stop(deadline)
            _verify_state(state, base_model, batches, settings, arm)
            path = generations / f"step-{state['step']:04d}.pt"
            digest = V6._immutable_torch(path, state)
            C.immutable_json(path.with_suffix(".json"), {"fit_identity_sha256": fit_hash,
                "step": state["step"], "sha256": digest})
            publish_snapshot(state)

        resume = states[max(states)] if states else None
        fit_cached_head(base_model, arrays, batches, settings, arm, resume=resume, on_update=persist,
                        check_stop=lambda: C.check_stop(deadline))
    C.check_stop(deadline)
    _require(proof and proof["passed"], "reproduction guard missing")
    decision = classify_results({arm: {step: row["metrics"] for step, row in records.items()} for arm, records in results.items()})
    _require(_load_inputs(config, identity)[-1] == fit_identity, "fit inputs changed during execution")
    files = {str(path.relative_to(directory)): C.sha256(path) for path in sorted(directory.rglob("*"))
             if path.is_file() and not path.name.startswith(".") and path.name != "manifest.json"}
    manifest = {"schema_version": 1, "experiment_id": config["experiment_id"], "complete": True,
        "identity_sha256": C.fingerprint(identity), "fit_identity_sha256": fit_hash, "fit_identity": fit_identity,
        "updates_per_arm": 2048, "snapshots": list(SNAPSHOTS), "trainable_parameters": 5160,
        "checkpoints": results, "classification": decision, "reproduction": proof,
        "batch_schedule": {"path": _path(directory / "batches.npz"), "sha256": C.sha256(directory / "batches.npz"),
            "batch_sha256": batch_fingerprint(batches), "root_shape": [2048, 32], "anchor_shape": [2048, 32], "seed": 203101},
        "gradient_input_overlap": guard, "frozen_state_sha256": frozen_state_sha256(base_model),
        "training_data_only": True, "validation_data_used": False, "benchmark_data_used": False,
        "automatic_promotion": False, "files": files}
    # Validate the complete pair before publishing final completion evidence.
    _verify_records(manifest, config, identity, loaded)
    C.immutable_json(directory / "manifest.json", manifest)
    C.immutable_json(C.PROGRESS / "fit-summary.json", manifest)
    return manifest


def _verify_records(manifest, config, identity, loaded):
    base_model, base_payload, arrays, metadata, guard, batches, prefix, reference, reference_batches, expected_identity = loaded
    settings = expected_identity["settings"]; directory = C.RUN / "fit"
    _require(manifest.get("complete") is True and manifest.get("fit_identity") == expected_identity
        and manifest.get("fit_identity_sha256") == C.fingerprint(expected_identity)
        and manifest.get("identity_sha256") == C.fingerprint(identity), "completed fit identity differs")
    expected_top = {"schema_version": 1, "experiment_id": config["experiment_id"], "updates_per_arm": 2048,
        "snapshots": list(SNAPSHOTS), "trainable_parameters": 5160, "training_data_only": True,
        "validation_data_used": False, "benchmark_data_used": False, "automatic_promotion": False,
        "gradient_input_overlap": guard, "frozen_state_sha256": frozen_state_sha256(base_model)}
    _require(all(manifest.get(key) == value and type(manifest.get(key)) is type(value)
        for key, value in expected_top.items()), "completed protocol/scope declaration differs")
    files = manifest.get("files", {})
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()
              and not p.name.startswith(".") and p.name != "manifest.json"}
    required = {"identity.json", "batches.npz", "batch-rng.json", "parent-metadata.json", "reproduction-128.json"}
    for arm in ARMS:
        for step in range(16, 2049, 16):
            required.update(f"{arm}/states/step-{step:04d}.{suffix}" for suffix in ("pt", "json"))
        for step in SNAPSHOTS:
            required.update(f"{arm}/checkpoint-{step:04d}.{suffix}" for suffix in ("pt", "json"))
    _require(set(files) == actual == required, "completed artifact inventory differs")
    for name, digest in files.items():
        path = directory / name
        _require(path.resolve().is_relative_to(directory.resolve()) and C.sha256(path) == digest, "completed artifact hash differs")
    _require(json.loads((directory / "identity.json").read_text()) == expected_identity
        and json.loads((directory / "batch-rng.json").read_text()) == batches["rng_state"], "completed identity/RNG sidecar differs")
    with np.load(directory / "batches.npz", allow_pickle=False) as saved:
        _require(set(saved.files) == {"root", "anchor"}
            and all(saved[k].dtype == np.int64 and np.array_equal(saved[k], batches[k]) for k in saved.files),
            "completed batch arrays differ")
    parent = {key: value for key, value in base_payload.items()
              if key not in ("model_state_dict", "optimizer_state_dict", "torch_rng", "numpy_rng")}
    _require(json.loads((directory / "parent-metadata.json").read_text()) == parent, "parent metadata differs")
    _require(manifest.get("batch_schedule") == {"path": _path(directory / "batches.npz"),
        "sha256": C.sha256(directory / "batches.npz"), "batch_sha256": batch_fingerprint(batches),
        "root_shape": [2048, 32], "anchor_shape": [2048, 32], "seed": 203101}, "completed batch declaration differs")
    records = manifest.get("checkpoints", {})
    _require(set(records) == set(ARMS), "completed arms differ")
    metrics = {}; proof = None
    for arm in ARMS:
        _require(set(records[arm]) == {str(x) for x in SNAPSHOTS}, "completed fixed snapshots differ")
        metrics[arm] = {}
        for update in range(16, 2049, 16):
            generation = directory / arm / "states" / f"step-{update:04d}.pt"
            sidecar = json.loads(generation.with_suffix(".json").read_text())
            _require(sidecar == {"fit_identity_sha256": C.fingerprint(expected_identity), "step": update,
                "sha256": C.sha256(generation)}, "completed recovery sidecar differs")
            saved_state = torch.load(generation, map_location="cpu", weights_only=True)
            _verify_state(saved_state, base_model, batches, settings, arm)
            _require(saved_state["step"] == update, "completed recovery update differs")
        for step in SNAPSHOTS:
            row = records[arm][str(step)]; path = directory / arm / f"checkpoint-{step:04d}.pt"
            _require(row["path"] == _path(path) and row["sha256"] == C.sha256(path)
                and json.loads(path.with_suffix(".json").read_text()) == row, "snapshot record differs")
            expected_row = {"arm": arm, "updates": step, "anchor_kl_weight": ARMS[arm], "trainable_parameters": 5160,
                "fit_identity_sha256": C.fingerprint(expected_identity), "frozen_tensors_equal": True, "value_logits_equal": True}
            _require(all(row.get(key) == value and type(row.get(key)) is type(value)
                for key, value in expected_row.items()), "snapshot protocol/proof declaration differs")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            model, state, measured = _verify_candidate(payload, base_model, arrays, batches, settings,
                config, expected_identity, arm, step)
            _require(row["metrics"] == measured and payload["parent_checkpoint_metadata"] == parent
                and payload["gradient_input_overlap"] == guard
                and payload["training_maps"] == sorted(set(metadata["training_maps"]) | set(base_payload.get("training_maps", [])))
                and payload["training_seeds"] == metadata["training_seeds"]
                and payload["games_total"] == len(metadata["training_seeds"]) and payload["positions_total"] == 544,
                "snapshot metrics/ancestry/guard differs")
            metrics[arm][str(step)] = measured
            if arm == "kl_1" and step == 128:
                proof = prove_prefix_reproduction(model.state_dict(), state["optimizer_state_dict"], state["torch_rng"],
                    state["history"], batches, reference, reference_batches, prefix["rng_state"])
                proof.update(reference_checkpoint_sha256=config["reference_terminal_checkpoint"]["sha256"],
                    fit_identity_sha256=C.fingerprint(expected_identity))
    _require(manifest.get("classification") == classify_results(metrics)
        and manifest.get("reproduction") == proof
        and json.loads((directory / "reproduction-128.json").read_text()) == proof, "completed decision/proof differs")
    return manifest


def verify_completed(config, identity, loaded=None):
    """Read-only semantic/hash verification; never resumes or trains."""
    loaded = loaded or _load_inputs(config, identity)
    manifest = json.loads((C.RUN / "fit/manifest.json").read_text())
    return _verify_records(manifest, config, identity, loaded)
