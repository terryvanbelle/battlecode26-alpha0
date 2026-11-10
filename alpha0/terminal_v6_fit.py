"""Matched, policy-head-only v6 fits from frozen own-self-play features.

No engine, benchmark, validation observation or rollout is executed here. Both
arms share one immutable batch schedule and differ only in selected-action labels.
"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from . import terminal_v6_common as C
from .features_v2 import Observation, encode
from .learn_v2 import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("terminal_head", "puct_head")
POLICY_KEYS = frozenset(("policy.weight", "policy.bias"))
FIT_CONFIG = {"updates": 128, "batch_size": 32, "anchor_batch_size": 32,
    "learning_rate": .0003, "weight_decay": .0001, "betas": [.9, .999], "eps": 1e-8,
    "seed": 203101, "target_weight": .75, "anchor_kl_weight": 1}
DATA_KEYS = frozenset(("root_hidden", "root_prior", "root_mask", "terminal_actions", "puct_actions",
    "root_strata", "anchor_hidden", "anchor_prior", "anchor_mask", "anchor_game", "anchor_player"))


def normalize_masked_prior(prior, mask):
    prior, mask = np.asarray(prior), np.asarray(mask)
    if (prior.ndim not in (1, 2) or prior.shape[-1] != 40 or mask.shape != prior.shape
            or not np.isin(mask, (0, 1)).all() or not np.isfinite(prior).all() or (prior < 0).any()
            or not mask.any(axis=-1).all()):
        raise ValueError("invalid finite prior/legal mask")
    result = np.where(mask, prior, 0).astype(np.float64)
    mass = result.sum(axis=-1, keepdims=True)
    if (mass <= 0).any():
        raise ValueError("prior has zero legal mass")
    return np.ascontiguousarray(result / mass, dtype=np.float32)


@torch.inference_mode()
def encode_state(model, state):
    """Capture the exact frozen fusion vector without changing the model class."""
    if model.config.action_dim != 40 or model.config.hidden_dim != 128:
        raise ValueError("v6 requires the frozen 128-to-40 policy head")
    observation = Observation(*(np.ascontiguousarray(x, dtype=np.float32) for x in encode(state)))
    legal = tuple(state.legal_actions)
    if not legal or len(set(legal)) != len(legal) or any(type(a) is not int or not 0 <= a < 40 for a in legal):
        raise ValueError("invalid legal actions for cached features")
    mask = np.zeros(40, dtype=np.uint8)
    mask[list(legal)] = 1
    input_digest = C.input_hash(observation, mask)
    device = next(model.parameters()).device
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError("v6 feature cache requires float32 model parameters")
    captured = []
    hook = model.policy.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach().clone()))
    was_training = model.training
    model.eval()
    try:
        inputs = [torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0) for x in observation]
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        logits, value = model(*inputs, mask_t)
    finally:
        hook.remove()
        model.train(was_training)
    if len(captured) != 1 or captured[0].shape != (1, 128):
        raise ValueError("policy hook did not capture exactly one frozen fusion vector")
    hidden_t = captured[0]
    reconstructed = model.policy(hidden_t).masked_fill(~mask_t, torch.finfo(logits.dtype).min)
    reconstructed_value = model.value(hidden_t).squeeze(-1)
    if not torch.equal(reconstructed, logits) or not torch.equal(reconstructed_value, value):
        raise ValueError("cached fusion does not reproduce original logits/value")
    prior = normalize_masked_prior(logits.softmax(-1)[0].cpu().numpy(), mask)
    return {"input_hash": input_digest, "hidden": hidden_t[0].cpu().numpy().astype(np.float32, copy=True),
            "prior": prior, "mask": mask, "value_logit": float(value.item()),
            "local": observation.local, "global_board": observation.global_board, "scalars": observation.scalars}


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def frozen_state_sha256(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key in POLICY_KEYS:
            continue
        header = json.dumps([key, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        h.update(len(header).to_bytes(8, "little"))
        h.update(header)
        h.update(tensor_bytes(value))
    return h.hexdigest()


def freeze_policy_head(model):
    model.eval()
    selected = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in POLICY_KEYS)
        if parameter.requires_grad:
            selected.append(parameter)
    if len(selected) != 2 or sum(parameter.numel() for parameter in selected) != 5160:
        raise ValueError("v6 must train exactly the 5,160 policy weight/bias parameters")
    return selected


def _integer_array(value, ndim, name):
    value = np.asarray(value)
    if value.ndim != ndim or value.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer array with {ndim} dimensions")
    return value.astype(np.int64, copy=False)


def make_batch_schedule(root_strata, anchor_game, anchor_player, fit_config):
    """Uniform root stratum; independent uniform game→team→anchor sampling."""
    strata = _integer_array(root_strata, 2, "root_strata")
    games = _integer_array(anchor_game, 1, "anchor_game")
    players = _integer_array(anchor_player, 1, "anchor_player")
    if strata.shape[1:] != (3,) or len(games) != len(players) or not len(games):
        raise ValueError("invalid strata lengths")
    maps = sorted(set(strata[:, 0].tolist()))
    groups = [np.flatnonzero(np.all(strata == (m, role, team), axis=1))
              for m in maps for role in (0, 1) for team in (0, 1)]
    game_ids = sorted(set(games.tolist()))
    anchors = {game: [np.flatnonzero((games == game) & (players == team)) for team in (0, 1)] for game in game_ids}
    if (not groups or any(not len(group) for group in groups)
            or not np.isin(strata[:, 1:], (0, 1)).all() or not np.isin(players, (0, 1)).all()
            or any(not len(group) for pair in anchors.values() for group in pair)):
        raise ValueError("every required root stratum and anchor game/team must be represented")
    sizes = [fit_config[k] for k in ("updates", "batch_size", "anchor_batch_size")]
    if any(type(size) is not int or size < 1 for size in sizes):
        raise ValueError("positive batch schedule sizes required")
    updates, batch_size, anchor_size = sizes
    rng = np.random.default_rng(fit_config["seed"])
    roots = np.empty((updates, batch_size), dtype=np.int64)
    sampled_anchors = np.empty((updates, anchor_size), dtype=np.int64)
    for update in range(updates):
        for column in range(batch_size):
            group = groups[int(rng.integers(len(groups)))]
            roots[update, column] = group[int(rng.integers(len(group)))]
        for column in range(anchor_size):
            game = game_ids[int(rng.integers(len(game_ids)))]
            group = anchors[game][int(rng.integers(2))]
            sampled_anchors[update, column] = group[int(rng.integers(len(group)))]
    return {"root": roots, "anchor": sampled_anchors, "rng_state": deepcopy(rng.bit_generator.state)}


def batch_fingerprint(batches):
    h = hashlib.sha256()
    for key in ("root", "anchor"):
        array = np.ascontiguousarray(batches[key], dtype="<i8")
        h.update(json.dumps([key, list(array.shape)], separators=(",", ":")).encode())
        h.update(array.tobytes())
    h.update(C.fingerprint(batches["rng_state"]).encode())
    return h.hexdigest()


def _log_policy(logits, mask):
    mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    if logits.shape != mask.shape or not mask.any(-1).all() or not torch.isfinite(logits).all():
        raise ValueError("invalid head logits/legal mask")
    return F.log_softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=-1)


def head_loss(root_logits, anchor_logits, root_prior, root_mask, anchor_prior, anchor_mask,
              actions, *, target_weight=.75, anchor_kl_weight=1):
    device = root_logits.device
    p0 = torch.as_tensor(normalize_masked_prior(root_prior, root_mask), device=device)
    anchor_p0 = torch.as_tensor(normalize_masked_prior(anchor_prior, anchor_mask), device=device)
    action_ids = torch.as_tensor(actions, dtype=torch.long, device=device)
    mask = torch.as_tensor(root_mask, dtype=torch.bool, device=device)
    if (action_ids.shape != (len(root_logits),) or (action_ids < 0).any() or (action_ids >= 40).any()
            or not mask.gather(1, action_ids[:, None]).all()):
        raise ValueError("selected action is illegal")
    target = (1-target_weight)*p0 + target_weight*F.one_hot(action_ids, 40).to(p0.dtype)
    root_logp = _log_policy(root_logits, root_mask)
    anchor_logp = _log_policy(anchor_logits, anchor_mask)
    ce = -(target * root_logp).sum(-1).mean()
    kl = (torch.xlogy(anchor_p0, anchor_p0) - anchor_p0*anchor_logp).sum(-1).mean()
    loss = ce + anchor_kl_weight*kl
    if not torch.isfinite(loss):
        raise ValueError("nonfinite policy-head loss")
    return loss, {"loss": float(loss.detach()), "target_ce": float(ce.detach()), "anchor_kl": float(kl.detach())}


def _cached_metrics(model, arrays, actions):
    with torch.inference_mode():
        rh = torch.as_tensor(arrays["root_hidden"], dtype=torch.float32)
        ah = torch.as_tensor(arrays["anchor_hidden"], dtype=torch.float32)
        logits, anchor_logits = model.policy(rh), model.policy(ah)
        _, result = head_loss(logits, anchor_logits, arrays["root_prior"], arrays["root_mask"],
                             arrays["anchor_prior"], arrays["anchor_mask"], actions)
        chosen = _log_policy(logits, arrays["root_mask"]).argmax(-1).numpy()
        anchor_chosen = _log_policy(anchor_logits, arrays["anchor_mask"]).argmax(-1).numpy()
        result.update(root_target_agreement=float(np.mean(chosen == actions)),
            root_action_changed_fraction=float(np.mean(chosen != arrays["root_prior"].argmax(-1))),
            anchor_action_changed_fraction=float(np.mean(anchor_chosen != arrays["anchor_prior"].argmax(-1))))
        return result


def fit_cached_head(base_model, arrays, batches, fit_config, arm, resume=None, on_update=None):
    if arm not in ARMS:
        raise ValueError("unknown matched fit arm")
    model = deepcopy(base_model).cpu()
    parameters = freeze_policy_head(model)
    frozen_digest = frozen_state_sha256(base_model)
    optimizer = torch.optim.AdamW(parameters, lr=fit_config["learning_rate"],
        weight_decay=fit_config["weight_decay"], betas=tuple(fit_config["betas"]), eps=fit_config["eps"])
    schedule_hash = batch_fingerprint(batches)
    actions = arrays["terminal_actions" if arm == "terminal_head" else "puct_actions"]
    start, history = 0, []
    if resume is not None:
        if (resume.get("arm") != arm or resume.get("batch_sha256") != schedule_hash
                or resume.get("frozen_state_sha256") != frozen_digest
                or type(resume.get("step")) is not int or not 0 <= resume["step"] <= fit_config["updates"]
                or len(resume.get("history", [])) != resume["step"]):
            raise ValueError("saved fit state has incompatible provenance or update index")
        model.policy.load_state_dict(resume["policy_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start, history = resume["step"], deepcopy(resume["history"])
    rh = torch.as_tensor(arrays["root_hidden"], dtype=torch.float32)
    ah = torch.as_tensor(arrays["anchor_hidden"], dtype=torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(fit_config["seed"])
        if resume is not None:
            torch.set_rng_state(resume["torch_rng"])
        for update in range(start, fit_config["updates"]):
            r, a = batches["root"][update], batches["anchor"][update]
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = head_loss(model.policy(rh[r]), model.policy(ah[a]), arrays["root_prior"][r],
                arrays["root_mask"][r], arrays["anchor_prior"][a], arrays["anchor_mask"][a], actions[r],
                target_weight=fit_config["target_weight"], anchor_kl_weight=fit_config["anchor_kl_weight"])
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError("nonfinite or missing policy-head gradients")
            optimizer.step()
            history.append({"step": update+1, **metrics})
            state = {"arm": arm, "step": update+1,
                "policy_state_dict": deepcopy(model.policy.state_dict()),
                "optimizer_state_dict": deepcopy(optimizer.state_dict()), "torch_rng": torch.get_rng_state().clone(),
                "numpy_rng": deepcopy(batches["rng_state"]), "batch_sha256": schedule_hash,
                "frozen_state_sha256": frozen_digest, "history": deepcopy(history)}
            if on_update:
                on_update(state)
        if start == fit_config["updates"]:
            state = deepcopy(resume)
    if frozen_state_sha256(model) != frozen_digest:
        raise ValueError("policy fit changed a frozen model tensor")
    with torch.inference_mode():
        all_hidden = torch.cat((rh, ah))
        if not torch.equal(base_model.value(all_hidden), model.value(all_hidden)):
            raise ValueError("policy fit changed frozen value predictions")
    return model, optimizer, state, _cached_metrics(model, arrays, actions)


def _hashes(values, count, name):
    if (not isinstance(values, list) or len(values) != count or any(not isinstance(v, str)
            or len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in values)):
        raise ValueError(f"invalid {name} input hashes")
    return values


def load_dataset(dataset_path, config, identity):
    """Accept only complete, identity-pinned training features and guard hashes."""
    dataset_path = Path(dataset_path)
    metadata_path = dataset_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    expected = {"format_version": 1, "identity_sha256": C.fingerprint(identity),
        "complete": True, "source": "own_selfplay", "benchmark_data_used": False,
        "validation_data_used": False, "root_count": 32, "anchor_count": 512}
    for key, value in expected.items():
        if metadata.get(key) != value or type(metadata.get(key)) is not type(value):
            raise ValueError(f"dataset is not complete own training data: {key}")
    if C.sha256(dataset_path) != metadata.get("npz_sha256"):
        raise ValueError("dataset checksum mismatch")
    with np.load(dataset_path, allow_pickle=False) as archive:
        if set(archive.files) != DATA_KEYS:
            raise ValueError("unexpected/missing dataset arrays; validation and benchmark arrays are forbidden")
        arrays = {key: archive[key].copy() for key in DATA_KEYS}
    for prefix, count in (("root", 32), ("anchor", 512)):
        for suffix, width in (("hidden", 128), ("prior", 40), ("mask", 40)):
            array = arrays[f"{prefix}_{suffix}"]
            expected_dtype = np.uint8 if suffix == "mask" else np.float32
            if array.shape != (count, width) or array.dtype != expected_dtype or not np.isfinite(array).all():
                raise ValueError(f"invalid {prefix}_{suffix} cached shape/dtype/values")
        arrays[f"{prefix}_prior"] = normalize_masked_prior(arrays[f"{prefix}_prior"], arrays[f"{prefix}_mask"])
    for key in ("terminal_actions", "puct_actions"):
        action = arrays[key]
        if (action.dtype != np.int64 or action.shape != (32,) or (action < 0).any() or (action >= 40).any()
                or not arrays["root_mask"][np.arange(32), action].all()):
            raise ValueError("target actions must be legal int64 decisions")
    strata = arrays["root_strata"]
    if strata.dtype != np.int64 or strata.shape != (32, 3):
        raise ValueError("invalid root strata")
    required = {(m, role, team) for m in range(4) for role in (0, 1) for team in (0, 1)}
    if set(map(tuple, strata.tolist())) != required or any(np.all(strata == key, axis=1).sum() != 2 for key in required):
        raise ValueError("exactly two roots per map/role/team stratum are required")
    for key in ("anchor_game", "anchor_player"):
        if arrays[key].dtype != np.int64 or arrays[key].shape != (512,):
            raise ValueError(f"invalid {key}")
    roots = metadata.get("root_ids")
    if not isinstance(roots, list) or len(roots) != 32 or len(set(roots)) != 32 or any(not isinstance(r, str) for r in roots):
        raise ValueError("unique frozen root IDs required")
    rh = _hashes(metadata.get("root_input_hashes"), 32, "root")
    ah = _hashes(metadata.get("anchor_input_hashes"), 512, "anchor")
    vh = _hashes(metadata.get("validation_input_hashes"), 16, "validation guard")
    gradient_hashes = sorted(set(rh + ah))
    if metadata.get("gradient_input_hashes") != gradient_hashes or set(gradient_hashes) & set(vh):
        raise ValueError("gradient-input union differs or overlaps validation")
    refs = metadata.get("anchor_refs")
    if not isinstance(refs, list) or len(refs) != 512 or any(not isinstance(ref, dict) for ref in refs):
        raise ValueError("anchor reconstruction references required")
    games = metadata.get("training_games")
    if not isinstance(games, list) or not games:
        raise ValueError("complete training source-game history required")
    ids = [game.get("source_id", game.get("game_id")) for game in games]
    if any(not isinstance(game, str) for game in ids) or ids != sorted(set(ids)):
        raise ValueError("training games must be ordered by unique source ID")
    maps = sorted({game.get("map") for game in games})
    seeds = [game.get("seed") for game in games]
    declared_maps = metadata.get("training_maps")
    if (maps != sorted(config["maps"]) or not isinstance(declared_maps, list)
            or sorted(declared_maps) != maps
            or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)
            or len(set(seeds)) != len(seeds) or metadata.get("training_seeds") != sorted(seeds)
            or set(maps) & set(identity.get("training_excluded_maps", []))
            or set(seeds) & set(identity.get("reserved_confirmation_seeds", []))):
        raise ValueError("source-game map/seed provenance differs or violates exclusions")
    if set(arrays["anchor_game"].tolist()) != set(range(len(games))):
        raise ValueError("anchors must cover every completed training source game")
    for i, ref in enumerate(refs):
        if (ref.get("input_hash") != ah[i]
                or ref.get("source_id", ref.get("game_id")) != ids[arrays["anchor_game"][i]]
                or ref.get("player") != int(arrays["anchor_player"][i])):
            raise ValueError("anchor reconstruction reference differs from cached source/player/input")
    # The schedule validator additionally rejects an absent team within a game.
    make_batch_schedule(strata, arrays["anchor_game"], arrays["anchor_player"],
                        {**config["fit"], "updates": 1, "batch_size": 1, "anchor_batch_size": 1})
    counts = metadata.get("selection_counts", {})
    expected_cells = 2*int(arrays["root_mask"].sum())
    if (counts.get("expected") != expected_cells or counts.get("completed") != expected_cells
            or counts.get("errors") != 0 or expected_cells > config["selection_cell_ceiling"]):
        raise ValueError("every legal-action selection cell must be complete before fitting")
    if not isinstance(metadata.get("provenance"), dict) or not metadata["provenance"]:
        raise ValueError("frozen input/source provenance required")
    guard = {"checked": True, "gradient_input_count": len(gradient_hashes), "validation_input_count": 16,
             "intersection_count": 0, "set_sha256": C.fingerprint(gradient_hashes)}
    return arrays, metadata, guard


def _tree_equal(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and left.shape == right.shape and tensor_bytes(left) == tensor_bytes(right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _immutable_torch(path, payload):
    path = Path(path)
    if path.exists():
        if not _tree_equal(torch.load(path, map_location="cpu", weights_only=True), payload):
            raise ValueError(f"immutable checkpoint differs: {path}")
        return C.sha256(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not _tree_equal(torch.load(path, map_location="cpu", weights_only=True), payload):
                raise ValueError("concurrent immutable checkpoint differs")
    finally:
        Path(temporary).unlink(missing_ok=True)
    return C.sha256(path)


def _verify_cached_priors(model, arrays):
    with torch.inference_mode():
        for prefix in ("root", "anchor"):
            hidden = torch.as_tensor(arrays[f"{prefix}_hidden"])
            actual = _log_policy(model.policy(hidden), arrays[f"{prefix}_mask"]).exp().numpy()
            if not np.allclose(actual, arrays[f"{prefix}_prior"], atol=1e-6, rtol=1e-5):
                raise ValueError("cached prior does not match the frozen baseline head/hidden features")


def _path_string(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _verify_completed(directory, expected_fit_identity, base_model, arrays, batches, base_payload, guard):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("complete") is not True or manifest.get("fit_identity_sha256") != C.fingerprint(expected_fit_identity)
            or manifest.get("fit_identity") != expected_fit_identity or set(manifest.get("checkpoints", {})) != set(ARMS)
            or manifest.get("identity_sha256") != expected_fit_identity["identity_sha256"]
            or manifest.get("fit_config") != expected_fit_identity["fit_config"]
            or manifest.get("gradient_input_overlap") != guard):
        raise ValueError("completed fit identity differs")
    required_files = {f"{arm}.pt" for arm in ARMS} | {"batches.npz", "batch-rng.json", "parent-metadata.json", "identity.json"}
    if set(manifest.get("files", {})) != required_files:
        raise ValueError("completed fit artifact inventory is incomplete or unexpected")
    for filename, digest in manifest["files"].items():
        if Path(filename).name != filename or C.sha256(directory / filename) != digest:
            raise ValueError("completed fit artifact checksum mismatch")
    with np.load(directory / "batches.npz", allow_pickle=False) as saved:
        if set(saved.files) != {"root", "anchor"} or any(saved[key].dtype != np.int64
                or not np.array_equal(saved[key], batches[key]) for key in saved.files):
            raise ValueError("completed fit batch schedule differs")
    if (json.loads((directory / "identity.json").read_text()) != expected_fit_identity
            or json.loads((directory / "batch-rng.json").read_text()) != batches["rng_state"]):
        raise ValueError("completed fit identity/RNG sidecar differs")
    parent_metadata = {key: value for key, value in base_payload.items()
                      if key not in ("model_state_dict", "optimizer_state_dict", "torch_rng", "numpy_rng")}
    if json.loads((directory / "parent-metadata.json").read_text()) != parent_metadata:
        raise ValueError("preserved baseline ancestry metadata differs")
    digest = frozen_state_sha256(base_model)
    if manifest.get("frozen_state_sha256") != digest:
        raise ValueError("reported frozen-state digest differs")
    batch_info = manifest.get("batch_schedule", {})
    if (batch_info.get("sha256") != manifest["files"]["batches.npz"]
            or batch_info.get("path") != _path_string(directory.parent / "fit/batches.npz")
            or batch_info.get("root_shape") != [128, 32] or batch_info.get("anchor_shape") != [128, 32]
            or batch_info.get("seed") != 203101):
        raise ValueError("reported shared batch schedule differs")
    all_hidden = torch.as_tensor(np.concatenate((arrays["root_hidden"], arrays["anchor_hidden"])))
    for arm in ARMS:
        path = directory / f"{arm}.pt"
        record = manifest["checkpoints"][arm]
        final_path = directory.parent / "fit" / path.name
        if (record.get("path") != _path_string(final_path) or record.get("sha256") != C.sha256(path)
                or record.get("fit_identity_sha256") != manifest["fit_identity_sha256"]
                or record.get("updates") != 128 or record.get("trainable_parameters") != 5160
                or record.get("frozen_tensors_equal") is not True or record.get("value_logits_equal") is not True):
            raise ValueError("reported candidate path/checksum/proof differs")
        model, payload = load_checkpoint(path)
        if (payload.get("arm") != arm or payload.get("updates") != 128
                or payload.get("fit_identity_sha256") != manifest["fit_identity_sha256"]
                or payload.get("batch_sha256") != batch_fingerprint(batches)
                or payload.get("parent_checkpoint_sha256") != expected_fit_identity["checkpoint_sha256"]
                or frozen_state_sha256(model) != digest or payload.get("parent_checkpoint_metadata") != parent_metadata
                or payload.get("fit_metrics") != record.get("metrics") or payload.get("gradient_input_overlap") != guard):
            raise ValueError("completed fit checkpoint provenance/frozen tensors differ")
        if any(not np.array_equal(payload["batch_indices"][key].numpy(), batches[key]) for key in ("root", "anchor")):
            raise ValueError("candidate batch indices differ")
        if not _tree_equal(payload["numpy_rng"], batches["rng_state"]):
            raise ValueError("candidate sampling RNG differs")
        seeded_rng = torch.Generator(device="cpu").manual_seed(expected_fit_identity["fit_config"]["seed"]).get_state()
        if not torch.equal(payload["torch_rng"], seeded_rng):
            raise ValueError("candidate fixed Torch RNG differs")
        states = payload["optimizer_state_dict"]["state"]
        if len(states) != 2 or any(float(value["step"]) != 128 for value in states.values()):
            raise ValueError("candidate optimizer update count differs")
        groups = payload["optimizer_state_dict"]["param_groups"]
        settings = expected_fit_identity["fit_config"]
        if (len(groups) != 1 or groups[0]["params"] != [0, 1]
                or any(groups[0][key] != settings[source] for key, source in (("lr", "learning_rate"),
                    ("weight_decay", "weight_decay"), ("eps", "eps")))
                or tuple(groups[0]["betas"]) != tuple(settings["betas"])):
            raise ValueError("candidate optimizer settings differ")
        with torch.inference_mode():
            if not torch.equal(model.value(all_hidden), base_model.value(all_hidden)):
                raise ValueError("candidate frozen value outputs differ")
    return manifest


def fit_arms(dataset_path: Path, config: dict, identity: dict, run_dir: Path, progress_dir: Path) -> dict:
    """Fit/publish both arms once; reuse only fully verified immutable artifacts.

Interrupted fits resume the latest checksum-pinned 16-update generation.
Unpublished extra updates after that generation can be replayed deterministically.
Neither a partial arm nor a partially completed pair is a candidate artifact.
"""
    if (config.get("fit") != FIT_CONFIG or config.get("experiment_id") != "terminal-target-head-v6"
            or identity.get("config") != config or identity.get("checkpoint_sha256") != config.get("checkpoint_sha256")):
        raise ValueError("fit does not match the preregistered configuration/identity")
    if config.get("torch_threads") != 1:
        raise ValueError("v6 fits use one Torch thread")
    torch.set_num_threads(1)
    run_dir, progress_dir, dataset_path = Path(run_dir), Path(progress_dir), Path(dataset_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "fit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checkpoint = ROOT / config["checkpoint"]
        if C.sha256(checkpoint) != config["checkpoint_sha256"]:
            raise ValueError("initial cp2 checkpoint checksum differs")
        base_model, base_payload = load_checkpoint(checkpoint)
        base_model.eval()
        if base_model.config.hidden_dim != 128 or base_model.config.action_dim != 40:
            raise ValueError("unexpected baseline architecture")
        arrays, metadata, guard = load_dataset(dataset_path, config, identity)
        _verify_cached_priors(base_model, arrays)
        batches = make_batch_schedule(arrays["root_strata"], arrays["anchor_game"], arrays["anchor_player"], config["fit"])
        fitter_hash, common_hash = C.sha256(__file__), C.sha256(C.__file__)
        if (identity.get("sources", {}).get("alpha0/terminal_v6_fit.py") != fitter_hash
                or identity.get("sources", {}).get("alpha0/terminal_v6_common.py") != common_hash):
            raise ValueError("fit/common source differs from the frozen experiment identity")
        fit_identity = {"identity_sha256": C.fingerprint(identity), "dataset_sha256": C.sha256(dataset_path),
            "dataset_metadata_sha256": C.sha256(dataset_path.with_suffix(".json")),
            "checkpoint_sha256": C.sha256(checkpoint), "fit_config": config["fit"],
            "batch_sha256": batch_fingerprint(batches), "fitter_sha256": fitter_hash, "common_sha256": common_hash,
            "torch_version": str(torch.__version__), "numpy_version": str(np.__version__)}
        final, pending = run_dir / "fit", run_dir / "fit.pending"
        if final.exists():
            result = _verify_completed(final, fit_identity, base_model, arrays, batches, base_payload, guard)
            C.immutable_json(progress_dir / "fit-summary.json", result)
            return result
        pending.mkdir(exist_ok=True)
        C.immutable_json(pending / "identity.json", fit_identity)
        C.immutable_npz(pending / "batches.npz", {key: batches[key] for key in ("root", "anchor")})
        C.immutable_json(pending / "batch-rng.json", batches["rng_state"])
        fit_hash = C.fingerprint(fit_identity)
        parent_metadata = {key: deepcopy(value) for key, value in base_payload.items()
                           if key not in ("model_state_dict", "optimizer_state_dict", "torch_rng", "numpy_rng")}
        # Parent metadata is a snapshot, while config.initialization remains the
        # authoritative checksum-pinned ancestry edge understood by victory.py.
        C.immutable_json(pending / "parent-metadata.json", parent_metadata)
        results = {}
        frozen_digest = frozen_state_sha256(base_model)
        for arm in ARMS:
            destination = pending / f"{arm}.pt"
            arm_record = pending / f"{arm}.json"
            if arm_record.exists():
                record = json.loads(arm_record.read_text())
                if record.get("fit_identity_sha256") != fit_hash or C.sha256(destination) != record["sha256"]:
                    raise ValueError("completed pending arm checksum/identity differs")
                results[arm] = record
                continue
            generations = pending / f"states-{arm}"
            generations.mkdir(exist_ok=True)
            resume = None
            for path in sorted(generations.glob("step-*.json")):
                generation = json.loads(path.read_text())
                state_path = path.with_suffix(".pt")
                if generation.get("fit_identity_sha256") != fit_hash or C.sha256(state_path) != generation["sha256"]:
                    raise ValueError("saved fit generation checksum/identity differs")
                candidate = torch.load(state_path, map_location="cpu", weights_only=True)
                if candidate["step"] != generation["step"]:
                    raise ValueError("saved fit generation update index differs")
                resume = candidate
            def persist(state):
                if state["step"] % 16 and state["step"] != 128:
                    return
                path = generations / f"step-{state['step']:04d}.pt"
                digest = _immutable_torch(path, state)
                C.immutable_json(path.with_suffix(".json"), {"fit_identity_sha256": fit_hash,
                                                          "step": state["step"], "sha256": digest})
            model, optimizer, state, metrics = fit_cached_head(base_model, arrays, batches, config["fit"],
                                                              arm, resume=resume, on_update=persist)
            ancestry_config = deepcopy(config)
            ancestry_config["initialization"] = {"artifact": config["checkpoint"],
                "artifact_sha256": config["checkpoint_sha256"], "method": "model-weights-only frozen cp2 ancestor"}
            training_maps = sorted(set(metadata["training_maps"]) | set(base_payload.get("training_maps", base_payload.get("config", {}).get("maps", []))))
            payload = model.checkpoint(experiment_id=config["experiment_id"], arm=arm, fit_version="policy-head-v6-1",
                training_stage="own-terminal-or-PUCT-selected-action-policy-head-distillation",
                config=ancestry_config, parent_checkpoint_sha256=config["checkpoint_sha256"],
                parent_checkpoint_metadata=parent_metadata, training_maps=training_maps,
                training_seeds=metadata["training_seeds"], games_total=len(metadata["training_seeds"]),
                positions_total=544, iteration=0, updates=128, gradient_updates=128, policy_head_only=True,
                benchmark_data_used=False, validation_data_used=False, trainable_parameters=5160,
                terminal_target_continuations=metadata["selection_counts"]["completed"],
                training_randomness={"selection_seeds": config["selection_seeds"], "source_games": metadata["training_games"],
                                     "anchor_seed": config["anchor_seed"], "fit_seed": config["fit"]["seed"]},
                fit_identity_sha256=fit_hash, dataset_sha256=fit_identity["dataset_sha256"],
                dataset_metadata_sha256=fit_identity["dataset_metadata_sha256"], frozen_state_sha256=frozen_digest,
                batch_sha256=batch_fingerprint(batches), batch_indices={key: torch.from_numpy(batches[key].copy()) for key in ("root", "anchor")},
                optimizer_state_dict=optimizer.state_dict(), torch_rng=state["torch_rng"], numpy_rng=state["numpy_rng"],
                fit_history=state["history"], fit_metrics=metrics, gradient_input_overlap=guard)
            digest = _immutable_torch(destination, payload)
            record = {"path": _path_string(final / f"{arm}.pt"), "sha256": digest, "fit_identity_sha256": fit_hash,
                      "updates": 128, "trainable_parameters": 5160, "frozen_tensors_equal": True,
                      "value_logits_equal": True, "metrics": metrics}
            C.immutable_json(arm_record, record)
            results[arm] = record
        if (C.sha256(checkpoint) != fit_identity["checkpoint_sha256"] or C.sha256(__file__) != fitter_hash
                or C.sha256(C.__file__) != common_hash or C.sha256(dataset_path) != fit_identity["dataset_sha256"]
                or C.sha256(dataset_path.with_suffix(".json")) != fit_identity["dataset_metadata_sha256"]):
            raise ValueError("fit inputs changed during training")
        artifact_names = [f"{arm}.pt" for arm in ARMS] + ["batches.npz", "batch-rng.json", "parent-metadata.json", "identity.json"]
        manifest = {"format_version": 1, "experiment_id": config["experiment_id"], "complete": True,
            "identity_sha256": C.fingerprint(identity), "fit_identity_sha256": fit_hash, "fit_identity": fit_identity,
            "fit_config": config["fit"], "dataset": {"path": _path_string(dataset_path), "sha256": fit_identity["dataset_sha256"],
                "metadata_sha256": fit_identity["dataset_metadata_sha256"], "root_count": 32, "anchor_count": 512},
            "batch_schedule": {"path": _path_string(final / "batches.npz"), "sha256": C.sha256(pending / "batches.npz"),
                "root_shape": [128, 32], "anchor_shape": [128, 32], "seed": 203101},
            "checkpoints": results, "frozen_state_sha256": frozen_digest, "gradient_input_overlap": guard,
            "files": {name: C.sha256(pending / name) for name in artifact_names}}
        C.immutable_json(pending / "manifest.json", manifest)
        _verify_completed(pending, fit_identity, base_model, arrays, batches, base_payload, guard)
        # Publish only the compact complete pair. Pending generations remain
        # separate recovery evidence and are never mistaken for candidates.
        staging = run_dir / ".fit-publish"
        staging.mkdir(exist_ok=True)
        for name in artifact_names + ["manifest.json"]:
            target = staging / name
            if target.exists():
                if C.sha256(target) != C.sha256(pending / name):
                    raise ValueError("staged completed artifact differs")
            else:
                os.link(pending / name, target)
        os.rename(staging, final)
        result = _verify_completed(final, fit_identity, base_model, arrays, batches, base_payload, guard)
        C.immutable_json(progress_dir / "fit-summary.json", result)
        return result
