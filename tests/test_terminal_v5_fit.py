"""Synthetic, engine-free checks for the preregistered v5 policy-head fit."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from alpha0.env import EngineState
from alpha0.features_v2 import encode
from alpha0.network_v2 import NetworkConfig, PolicyValueNetV2
from alpha0 import terminal_v5_common as common
from alpha0 import terminal_v5_fit as fitter
from alpha0.terminal_v5_fit import (
    FIT_CONFIG,
    encode_state,
    fit_cached_head,
    fit_arms,
    freeze_policy_head,
    head_loss,
    make_batch_schedule,
    normalize_masked_prior,
    load_dataset,
)


def model_fixture():
    torch.manual_seed(419)
    # Keep the actual 128-to-40 head, with cheaper unchanged encoder topology.
    model = PolicyValueNetV2(NetworkConfig(local_width=8, global_width=8, blocks=0))
    # A nonzero critic makes accidentally resetting it detectable.
    with torch.no_grad():
        model.value[-1].weight.fill_(0.03125)
        model.value[-1].bias.fill_(0.125)
    return model


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def forty(values, dtype=np.float32):
    array = np.asarray(values, dtype=dtype)
    return np.pad(array, [(0, 0)] * (array.ndim - 1) + [(0, 40-array.shape[-1])])


def cached_fixture(model):
    rng = np.random.default_rng(707)
    root_hidden = rng.normal(0, .3, (16, 128)).astype(np.float32)
    anchor_hidden = rng.normal(0, .3, (7, 128)).astype(np.float32)
    root_mask = np.zeros((16, 40), bool)
    root_mask[:, [0, 21, 37]] = True
    anchor_mask = np.zeros((7, 40), bool)
    anchor_mask[:, [0, 17, 21, 27, 37]] = True
    with torch.no_grad():
        root_prior = model.policy(torch.from_numpy(root_hidden)).masked_fill(
            ~torch.from_numpy(root_mask), -torch.inf).softmax(-1).numpy()
        anchor_prior = model.policy(torch.from_numpy(anchor_hidden)).masked_fill(
            ~torch.from_numpy(anchor_mask), -torch.inf).softmax(-1).numpy()
    return dict(root_hidden=root_hidden, root_prior=root_prior, root_mask=root_mask,
                terminal_actions=np.full(16, 21, np.int64), puct_actions=np.full(16, 21, np.int64),
                anchor_hidden=anchor_hidden, anchor_prior=anchor_prior, anchor_mask=anchor_mask)


def small_schedule():
    return make_batch_schedule(
        [(m, role, team) for m in range(4) for role in (0, 1) for team in (0, 1)],
        np.array([0, 0, 0, 1, 1, 1, 1]),
        np.array([0, 0, 1, 0, 1, 1, 1]), copy.deepcopy(FIT_CONFIG))


def dataset_fixture(directory):
    """Complete synthetic data contract, without collecting any engine states."""
    directory = Path(directory)
    baseline = directory / "synthetic-baseline.pt"
    model = model_fixture()
    torch.save(model.checkpoint(training_maps=[]), baseline)
    config = json.loads((common.ROOT / "configs/terminal-target-head-v5.json").read_text())
    config.update(checkpoint=str(baseline), checkpoint_sha256=common.sha256(baseline))
    identity = {"config": config, "checkpoint_sha256": config["checkpoint_sha256"],
                "sources": {"alpha0/terminal_v5_fit.py": common.sha256(fitter.__file__),
                            "alpha0/terminal_v5_common.py": common.sha256(common.__file__)},
                "training_excluded_maps": [], "reserved_confirmation_seeds": []}
    rng = np.random.default_rng(201)
    arrays = {}
    for prefix, count in (("root", 32), ("anchor", 512)):
        hidden = rng.normal(0, .25, (count, 128)).astype(np.float32)
        mask = np.zeros((count, 40), np.uint8)
        mask[:, [0, 21, 37]] = 1
        with torch.no_grad():
            prior = model.policy(torch.from_numpy(hidden)).masked_fill(
                ~torch.from_numpy(mask).bool(), -torch.inf).softmax(-1).numpy()
        arrays.update({f"{prefix}_hidden": hidden, f"{prefix}_mask": mask, f"{prefix}_prior": prior})
    arrays.update(
        terminal_actions=np.full(32, 21, np.int64), puct_actions=np.full(32, 37, np.int64),
        root_strata=np.array([(m, role, team) for m in range(4) for role in (0, 1)
                             for team in (0, 1) for _ in range(2)], np.int64),
        anchor_game=np.arange(512, dtype=np.int64) % 4,
        anchor_player=(np.arange(512, dtype=np.int64) // 4) % 2,
    )
    path = directory / "dataset.npz"
    np.savez_compressed(path, **arrays)

    def digest(label):
        return hashlib.sha256(label.encode()).hexdigest()

    root_hashes = [digest(f"training-{i}") for i in range(32)]
    anchor_hashes = [digest(f"anchor-{i}") for i in range(512)]
    metadata = {"format_version": 1, "identity_sha256": common.fingerprint(identity), "complete": True,
        "source": "own_selfplay", "benchmark_data_used": False, "validation_data_used": False,
        "root_count": 32, "anchor_count": 512, "npz_sha256": common.sha256(path),
        "root_ids": [f"root-{i}" for i in range(32)], "root_input_hashes": root_hashes,
        "anchor_input_hashes": anchor_hashes, "validation_input_hashes": [digest(f"validation-{i}") for i in range(16)],
        "gradient_input_hashes": sorted(set(root_hashes + anchor_hashes)),
        "anchor_refs": [{"source_id": f"g{arrays['anchor_game'][i]}", "player": int(arrays["anchor_player"][i]),
                         "input_hash": anchor_hashes[i], "decision_index": i} for i in range(512)],
        "training_games": [{"source_id": f"g{i}", "map": map_name, "seed": i + 1}
                           for i, map_name in enumerate(config["maps"])],
        "training_maps": sorted(config["maps"]), "training_seeds": [1, 2, 3, 4],
        "selection_counts": {"expected": 192, "completed": 192, "errors": 0},
        "provenance": {"fixture": "synthetic cached features; no engine or actor"}}
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return path, config, identity, model, arrays, metadata


class TerminalV5NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_only_the_5160_policy_parameters_are_trainable(self):
        model = model_fixture()
        before = {name: tensor_bytes(tensor) for name, tensor in model.state_dict().items()}
        parameters = freeze_policy_head(model)
        self.assertEqual({name for name, p in model.named_parameters() if p.requires_grad},
                         {"policy.weight", "policy.bias"})
        self.assertEqual(sum(p.numel() for p in parameters), 5160)
        self.assertEqual({id(p) for p in parameters}, {id(model.policy.weight), id(model.policy.bias)})
        self.assertEqual(before, {name: tensor_bytes(tensor) for name, tensor in model.state_dict().items()})

    def test_legal_prior_normalization_preserves_zero_mass_and_inputs(self):
        prior = forty([[2.0, 90.0, 3.0, 0.0], [0.0, 4.0, 7.0, 0.0]], np.float64)
        mask = forty([[1, 0, 1, 1], [0, 1, 0, 1]], bool)
        untouched = prior.copy()
        normalized = normalize_masked_prior(prior, mask)
        self.assertEqual(normalized.dtype, np.float32)
        np.testing.assert_allclose(normalized, forty([[.4, 0, .6, 0], [0, 1, 0, 0]]), atol=1e-7)
        np.testing.assert_array_equal(normalized[~mask], 0)
        np.testing.assert_array_equal(prior, untouched)
        np.testing.assert_allclose(normalized.sum(-1), 1)

    def test_invalid_priors_and_masks_fail_closed(self):
        cases = [
            ([0, 0], [1, 1]),
            ([1, 1], [0, 0]),
            ([-1, 2], [1, 1]),
            ([1, float("nan")], [1, 0]),
            ([1, float("inf")], [1, 0]),
            ([1, 1], [1]),
        ]
        for prior, mask in cases:
            with self.subTest(prior=prior, mask=mask), self.assertRaises(ValueError):
                normalize_masked_prior(forty(prior), forty(mask, bool) if len(mask) == len(prior) else np.array(mask, bool))

    def test_loss_is_smoothed_target_ce_plus_forward_kl_on_legal_actions(self):
        # Enormous illegal logits must contribute neither mass nor gradient.
        root_logits = torch.tensor(forty([[math.log(.8), 1000., math.log(.2), -1000.]]), requires_grad=True)
        anchor_logits = torch.tensor(forty([[math.log(.6), math.log(.3), math.log(.1), 1000.]]), requires_grad=True)
        root_prior = torch.from_numpy(forty([[.4, 0., .6, 0.]]))
        root_mask = torch.from_numpy(forty([[True, False, True, False]], bool))
        anchor_prior = torch.from_numpy(forty([[.2, .8, 0., 0.]]))
        anchor_mask = torch.from_numpy(forty([[True, True, True, False]], bool))
        loss, metrics = head_loss(root_logits, anchor_logits, root_prior, root_mask,
                                  anchor_prior, anchor_mask, torch.tensor([2]))
        expected_ce = -(.1 * math.log(.8) + .9 * math.log(.2))
        expected_kl = .2 * math.log(.2 / .6) + .8 * math.log(.8 / .3)
        self.assertAlmostEqual(float(loss.detach()), expected_ce + expected_kl, places=6)
        self.assertTrue(all(math.isfinite(float(value)) for value in metrics.values()))
        loss.backward()
        self.assertTrue(torch.isfinite(root_logits.grad).all())
        self.assertTrue(torch.isfinite(anchor_logits.grad).all())
        self.assertEqual(float(root_logits.grad[0, 1]), 0)
        self.assertEqual(float(root_logits.grad[0, 3]), 0)
        self.assertEqual(float(anchor_logits.grad[0, 3]), 0)
        # Forward KL has p_theta-p0 gradient, including a legal zero-p0 row.
        torch.testing.assert_close(anchor_logits.grad[0, :3], torch.tensor([.4, -.5, .1]),
                                   rtol=1e-6, atol=1e-6)

    def test_loss_rejects_an_illegal_selected_action(self):
        logits = torch.zeros((1, 40))
        prior = torch.from_numpy(forty([[.5, .5, 0., 0.]]))
        mask = torch.from_numpy(forty([[True, True, False, False]], bool))
        with self.assertRaises(ValueError):
            head_loss(logits, logits, prior, mask, prior, mask, torch.tensor([2]))

    def test_batches_are_reproducible_and_balance_strata_not_stored_rows(self):
        # Every required map/role/team stratum is present, but one has many rows.
        keys = [(m, role, team) for m in range(4) for role in (0, 1) for team in (0, 1)]
        root_strata = [key for i, key in enumerate(keys) for _ in range(60 if i == 0 else 1)]
        # Unequal storage within game and team would severely bias flat sampling.
        anchor_game = np.array([0] * 61 + [1] * 8)
        anchor_player = np.array([0] * 60 + [1] + [0] * 3 + [1] * 5)
        first = make_batch_schedule(root_strata, anchor_game, anchor_player, copy.deepcopy(FIT_CONFIG))
        second = make_batch_schedule(root_strata, anchor_game, anchor_player, copy.deepcopy(FIT_CONFIG))
        for name in ("root", "anchor"):
            self.assertEqual(first[name].shape, (128, 32))
            self.assertEqual(first[name].dtype, np.int64)
            np.testing.assert_array_equal(first[name], second[name])
        self.assertEqual(first["rng_state"], second["rng_state"])
        root_counts = {key: 0 for key in keys}
        for index in first["root"].flat:
            root_counts[root_strata[index]] += 1
        # Wide deterministic bounds, but far from the ~3,200 samples flat
        # sampling would give the overrepresented first stratum.
        self.assertTrue(all(150 <= count <= 370 for count in root_counts.values()), root_counts)
        anchor_counts = {(game, team): 0 for game in (0, 1) for team in (0, 1)}
        for index in first["anchor"].flat:
            anchor_counts[(anchor_game[index], int(anchor_player[index]))] += 1
        self.assertTrue(all(850 <= count <= 1200 for count in anchor_counts.values()), anchor_counts)

    def test_anchor_sampling_rejects_a_missing_team_in_a_source_game(self):
        strata = [(m, role, team) for m in range(4) for role in (0, 1) for team in (0, 1)]
        with self.assertRaises(ValueError):
            make_batch_schedule(strata, np.array([0, 1, 1]),
                                np.array([0, 0, 1]), copy.deepcopy(FIT_CONFIG))

    def test_encoder_cache_matches_the_unchanged_network_forward(self):
        model = model_fixture().eval()
        state = EngineState({"terminal": False, "action_version": "macro_v2", "width": 12,
            "height": 12, "current_team": 0, "current_id": 1, "round": 41, "cooperation": 0,
            "legal_actions": [0, 21, 37], "robots": [
                {"id": 1, "team": 0, "type": "BABY_RAT", "x": 5, "y": 6, "hp": 73,
                 "direction": 2, "cheese": 9},
                {"id": 2, "team": 1, "type": "RAT_KING", "x": 9, "y": 9, "hp": 480,
                 "direction": 6, "cheese": 0}]})
        observed_hidden = []
        handle = model.fusion.register_forward_hook(lambda _m, _i, output: observed_hidden.append(output.detach().clone()))
        try:
            cached = encode_state(model, state)
            observation = encode(state)
            tensors = [torch.from_numpy(array).unsqueeze(0) for array in observation]
            with torch.no_grad():
                logits, value = model(*tensors, torch.from_numpy(cached["mask"]).unsqueeze(0))
        finally:
            handle.remove()
        np.testing.assert_array_equal(cached["hidden"], observed_hidden[-1][0].numpy())
        np.testing.assert_allclose(cached["prior"], logits.softmax(-1)[0].numpy(), atol=1e-7)
        self.assertEqual(float(cached["value_logit"]), float(value[0]))
        self.assertEqual(len(cached["input_hash"]), 64)
        for name, original in zip(("local", "global_board", "scalars"), observation):
            np.testing.assert_array_equal(cached[name], original)
            self.assertEqual(cached[name].dtype, np.float32)
        np.testing.assert_array_equal(cached["prior"][~cached["mask"].astype(bool)], 0)

    def test_full_128_update_arms_share_batches_and_preserve_every_frozen_byte(self):
        base = model_fixture()
        before = {name: tensor_bytes(tensor) for name, tensor in base.state_dict().items()}
        arrays, batches = cached_fixture(base), small_schedule()
        original_arrays = {name: array.copy() for name, array in arrays.items()}
        terminal, terminal_optimizer, terminal_state, terminal_metrics = fit_cached_head(
            base, arrays, batches, copy.deepcopy(FIT_CONFIG), "terminal_head")
        puct, puct_optimizer, puct_state, puct_metrics = fit_cached_head(
            base, arrays, batches, copy.deepcopy(FIT_CONFIG), "puct_head")
        self.assertEqual(terminal_state["step"], 128)
        self.assertEqual(puct_state["step"], 128)
        self.assertEqual(len(terminal_state["history"]), 128)
        self.assertEqual(terminal_state["batch_sha256"], puct_state["batch_sha256"])
        self.assertEqual(terminal_state["frozen_state_sha256"], puct_state["frozen_state_sha256"])
        self.assertEqual(terminal_state["history"], puct_state["history"])
        self.assertEqual(terminal_metrics, puct_metrics)
        self.assertTrue(all(math.isfinite(float(v)) for v in terminal_metrics.values()))
        for name, tensor in terminal.state_dict().items():
            self.assertEqual(tensor_bytes(tensor), tensor_bytes(puct.state_dict()[name]), name)
            if not name.startswith("policy."):
                self.assertEqual(tensor_bytes(tensor), before[name], name)
        self.assertNotEqual(tensor_bytes(terminal.policy.weight), before["policy.weight"])
        self.assertEqual(before, {name: tensor_bytes(tensor) for name, tensor in base.state_dict().items()})
        for name in arrays:
            np.testing.assert_array_equal(arrays[name], original_arrays[name])
        terminal_opt = terminal_optimizer.state_dict()
        puct_opt = puct_optimizer.state_dict()
        self.assertEqual(terminal_opt["param_groups"], puct_opt["param_groups"])
        self.assertEqual(len(terminal_opt["state"]), 2)
        for index, moments in terminal_opt["state"].items():
            for key, value in moments.items():
                self.assertEqual(tensor_bytes(value), tensor_bytes(puct_opt["state"][index][key]), key)

    def test_arm_targets_are_the_only_change_to_the_fit(self):
        base = model_fixture()
        arrays, batches = cached_fixture(base), small_schedule()
        arrays["puct_actions"][:] = 37
        terminal, _, _, _ = fit_cached_head(base, arrays, batches, copy.deepcopy(FIT_CONFIG), "terminal_head")
        puct, _, _, _ = fit_cached_head(base, arrays, batches, copy.deepcopy(FIT_CONFIG), "puct_head")
        self.assertFalse(torch.equal(terminal.policy.weight, puct.policy.weight))
        for name, tensor in terminal.state_dict().items():
            if not name.startswith("policy."):
                self.assertEqual(tensor_bytes(tensor), tensor_bytes(puct.state_dict()[name]), name)

    def test_interrupted_fit_resumes_to_the_exact_uninterrupted_head_and_optimizer(self):
        base = model_fixture()
        arrays, batches = cached_fixture(base), small_schedule()
        full, full_optimizer, full_state, full_metrics = fit_cached_head(
            base, arrays, batches, copy.deepcopy(FIT_CONFIG), "terminal_head")
        saved = []

        class PlannedInterruption(Exception):
            pass

        def stop_after_37(state):
            if state["step"] == 37:
                saved.append(copy.deepcopy(state))
                raise PlannedInterruption

        rng_before = torch.get_rng_state().clone()
        with self.assertRaises(PlannedInterruption):
            fit_cached_head(base, arrays, batches, copy.deepcopy(FIT_CONFIG), "terminal_head",
                            on_update=stop_after_37)
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
        resumed, optimizer, state, metrics = fit_cached_head(
            base, arrays, batches, copy.deepcopy(FIT_CONFIG), "terminal_head", resume=saved[0])
        self.assertEqual(metrics, full_metrics)
        self.assertEqual(state["history"], full_state["history"])
        self.assertEqual(state["numpy_rng"], full_state["numpy_rng"])
        torch.testing.assert_close(state["torch_rng"], full_state["torch_rng"], rtol=0, atol=0)
        for name, tensor in full.state_dict().items():
            self.assertEqual(tensor_bytes(tensor), tensor_bytes(resumed.state_dict()[name]), name)
        for index, moments in full_optimizer.state_dict()["state"].items():
            for key, tensor in moments.items():
                self.assertEqual(tensor_bytes(tensor), tensor_bytes(optimizer.state_dict()["state"][index][key]), key)
        # A complete resume is read-only with respect to optimization updates.
        calls = []
        repeated, _, repeated_state, repeated_metrics = fit_cached_head(
            base, arrays, batches, copy.deepcopy(FIT_CONFIG), "terminal_head", resume=state,
            on_update=lambda update: calls.append(update))
        self.assertFalse(calls)
        self.assertEqual(repeated_state["history"], full_state["history"])
        self.assertEqual(repeated_metrics, full_metrics)
        self.assertEqual(tensor_bytes(repeated.policy.weight), tensor_bytes(full.policy.weight))

    def test_fit_resume_rejects_mismatched_arm_schedule_frozen_state_and_step(self):
        base = model_fixture()
        arrays, batches = cached_fixture(base), small_schedule()
        short_config = {**copy.deepcopy(FIT_CONFIG), "updates": 1}
        _, _, original, _ = fit_cached_head(base, arrays, batches, short_config, "terminal_head")
        for field, value in (("arm", "puct_head"), ("batch_sha256", "0" * 64),
                             ("frozen_state_sha256", "0" * 64), ("step", -1), ("step", 129)):
            changed = copy.deepcopy(original)
            changed[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                fit_cached_head(base, arrays, batches, copy.deepcopy(FIT_CONFIG), "terminal_head", resume=changed)
        changed_batches = copy.deepcopy(batches)
        changed_batches["root"][0, 0] = (changed_batches["root"][0, 0] + 1) % 16
        with self.assertRaises(ValueError):
            fit_cached_head(base, arrays, changed_batches, copy.deepcopy(FIT_CONFIG), "terminal_head", resume=original)

    def test_complete_export_reuse_is_immutable_and_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, config, identity, baseline, _, _ = dataset_fixture(root)
            run, progress = root / "run", root / "progress"
            manifest = fit_arms(dataset, config, identity, run, progress)
            self.assertIs(manifest["complete"], True)
            self.assertEqual(set(manifest["checkpoints"]), {"terminal_head", "puct_head"})
            self.assertEqual(manifest["gradient_input_overlap"]["intersection_count"], 0)
            snapshots = {path.relative_to(root): (path.read_bytes(), path.stat().st_mtime_ns)
                         for parent in (run / "fit", progress) for path in parent.rglob("*") if path.is_file()}
            with patch.object(fitter, "fit_cached_head", side_effect=AssertionError("completed fit reran SGD")):
                reused = fit_arms(dataset, config, identity, run, progress)
            self.assertEqual(reused, manifest)
            self.assertEqual(snapshots, {path: ((root / path).read_bytes(), (root / path).stat().st_mtime_ns)
                                         for path in snapshots})
            for arm in ("terminal_head", "puct_head"):
                payload = torch.load(run / "fit" / f"{arm}.pt", map_location="cpu", weights_only=True)
                self.assertEqual(payload["trainable_parameters"], 5160)
                self.assertEqual(payload["updates"], 128)
                self.assertEqual(payload["parent_checkpoint_sha256"], config["checkpoint_sha256"])
                for name, tensor in baseline.state_dict().items():
                    if not name.startswith("policy."):
                        self.assertEqual(tensor_bytes(tensor), tensor_bytes(payload["model_state_dict"][name]), name)
                self.assertEqual(payload["batch_indices"]["root"].shape, (128, 32))
                self.assertEqual(payload["batch_indices"]["anchor"].shape, (128, 32))
            for candidate in (run / "fit/terminal_head.pt", run / "fit/batches.npz", dataset):
                original = candidate.read_bytes()
                candidate.write_bytes(original + b"CORRUPTED")
                try:
                    with self.subTest(file=candidate.name), self.assertRaises(ValueError):
                        fit_arms(dataset, config, identity, run, progress)
                finally:
                    candidate.write_bytes(original)
            manifest_path = run / "fit/manifest.json"
            original_manifest = manifest_path.read_bytes()
            for corruption in ("candidate_digest", "candidate_path", "missing_file_digest"):
                changed_manifest = json.loads(original_manifest)
                if corruption == "candidate_digest":
                    changed_manifest["checkpoints"]["terminal_head"]["sha256"] = "0" * 64
                elif corruption == "candidate_path":
                    changed_manifest["checkpoints"]["terminal_head"]["path"] = str(root / "wrong-candidate.pt")
                else:
                    del changed_manifest["files"]["terminal_head.pt"]
                manifest_path.write_text(json.dumps(changed_manifest))
                try:
                    with self.subTest(manifest=corruption), self.assertRaises(ValueError):
                        # A restored/exported fit must verify itself even when
                        # no old progress report is available to catch a mismatch.
                        fit_arms(dataset, config, identity, run, root / f"fresh-progress-{corruption}")
                finally:
                    manifest_path.write_bytes(original_manifest)
            # Even a valid metadata edit changes the immutable fit identity.
            metadata_path = dataset.with_suffix(".json")
            original = metadata_path.read_bytes()
            metadata = json.loads(original)
            metadata["provenance"]["changed"] = True
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                fit_arms(dataset, config, identity, run, progress)
            metadata_path.write_bytes(original)
            changed_config = copy.deepcopy(config)
            changed_config["fit"]["learning_rate"] *= 2
            with self.assertRaises(ValueError):
                fit_arms(dataset, changed_config, {**identity, "config": changed_config}, run, progress)
            self.assertEqual(set(manifest["files"]), {path.name for path in (run / "fit").iterdir()} - {"manifest.json"})

    def test_partial_pair_is_not_published_and_resumes_only_a_verified_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, config, identity, _, _, _ = dataset_fixture(root)
            run, progress = root / "run", root / "progress"
            real_fit = fit_cached_head

            class PlannedInterruption(Exception):
                pass

            def interrupted(*args, **kwargs):
                persist = kwargs["on_update"]

                def on_update(state):
                    persist(state)
                    if state["step"] == 37:
                        raise PlannedInterruption

                return real_fit(*args, **{**kwargs, "on_update": on_update})

            with patch.object(fitter, "fit_cached_head", side_effect=interrupted):
                with self.assertRaises(PlannedInterruption):
                    fit_arms(dataset, config, identity, run, progress)
            self.assertFalse((run / "fit").exists())
            self.assertFalse((progress / "fit-summary.json").exists())
            generations = sorted((run / "fit.pending/states-terminal_head").glob("step-*.pt"))
            self.assertEqual([path.name for path in generations], ["step-0016.pt", "step-0032.pt"])
            latest = generations[-1]
            intact = latest.read_bytes()
            latest.write_bytes(intact + b"CORRUPTED")
            with patch.object(fitter, "fit_cached_head", side_effect=AssertionError("corrupt resume reached SGD")):
                with self.assertRaises(ValueError):
                    fit_arms(dataset, config, identity, run, progress)
            latest.write_bytes(intact)
            starts = []

            def observe_resume(*args, **kwargs):
                state = kwargs.get("resume")
                starts.append((args[4], None if state is None else state["step"]))
                return real_fit(*args, **kwargs)

            with patch.object(fitter, "fit_cached_head", side_effect=observe_resume):
                manifest = fit_arms(dataset, config, identity, run, progress)
            self.assertEqual(starts, [("terminal_head", 32), ("puct_head", None)])
            self.assertIs(manifest["complete"], True)
            self.assertEqual(set(manifest["checkpoints"]), {"terminal_head", "puct_head"})

    def test_dataset_boundary_rejects_validation_overlap_missing_strata_and_incomplete_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, config, identity, _, arrays, metadata = dataset_fixture(root)
            loaded, _, guard = load_dataset(dataset, config, identity)
            self.assertEqual(loaded["root_hidden"].shape, (32, 128))
            self.assertEqual(guard["intersection_count"], 0)
            for change in ("validation_root_overlap", "validation_anchor_overlap", "incomplete_targets", "benchmark"):
                changed = copy.deepcopy(metadata)
                if change.startswith("validation"):
                    key = "root_input_hashes" if "root" in change else "anchor_input_hashes"
                    changed["validation_input_hashes"][0] = changed[key][0]
                elif change == "incomplete_targets":
                    changed["selection_counts"]["completed"] -= 1
                else:
                    changed["benchmark_data_used"] = True
                dataset.with_suffix(".json").write_text(json.dumps(changed))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    load_dataset(dataset, config, identity)
            for key, change in (("root_strata", lambda values: values.__setitem__((0, 0), 1)),
                                ("anchor_game", lambda values: values.__setitem__(values == 3, 2))):
                changed_arrays = {name: value.copy() for name, value in arrays.items()}
                change(changed_arrays[key])
                np.savez_compressed(dataset, **changed_arrays)
                changed = copy.deepcopy(metadata)
                changed["npz_sha256"] = common.sha256(dataset)
                dataset.with_suffix(".json").write_text(json.dumps(changed))
                with self.subTest(array=key), self.assertRaises(ValueError):
                    load_dataset(dataset, config, identity)


if __name__ == "__main__":
    unittest.main()
