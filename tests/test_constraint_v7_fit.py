"""Synthetic v7 fit checks: no engine, real dataset, or candidate publication."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from alpha0 import constraint_v7_fit as v7
from alpha0 import terminal_v6_fit as v6
from alpha0.network_v2 import NetworkConfig, PolicyValueNetV2


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def assert_tree_equal(test, left, right):
    if torch.is_tensor(left):
        test.assertTrue(torch.is_tensor(right))
        test.assertEqual(left.dtype, right.dtype)
        test.assertEqual(tuple(left.shape), tuple(right.shape))
        test.assertEqual(tensor_bytes(left), tensor_bytes(right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for name in left:
            assert_tree_equal(test, left[name], right[name])
    elif isinstance(left, (tuple, list)):
        test.assertIs(type(left), type(right))
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree_equal(test, a, b)
    else:
        test.assertEqual(left, right)


def fixture():
    torch.manual_seed(612)
    model = PolicyValueNetV2(NetworkConfig(local_width=8, global_width=8, blocks=0))
    with torch.no_grad():
        model.value[-1].weight.fill_(.0625)
        model.value[-1].bias.fill_(.25)
    rng = np.random.default_rng(43)
    hidden = rng.normal(0, .4, (32, 128)).astype(np.float32)
    arrays = {}
    for prefix, features in (("root", hidden), ("anchor", hidden[:18].copy())):
        mask = np.zeros((len(features), 40), np.uint8)
        mask[:, [0, 21, 37]] = 1
        with torch.no_grad():
            prior = model.policy(torch.from_numpy(features)).masked_fill(
                ~torch.from_numpy(mask).bool(), -torch.inf).softmax(-1).numpy()
        arrays.update({f"{prefix}_hidden": features, f"{prefix}_mask": mask, f"{prefix}_prior": prior})
    arrays.update(terminal_actions=np.array([21] * 16 + [37] * 16, np.int64),
                  puct_actions=arrays["root_prior"].argmax(-1).astype(np.int64),
                  root_strata=np.array([(m, role, team) for m in range(4)
                                       for role in (0, 1) for team in (0, 1) for _ in range(2)], np.int64),
                  anchor_game=np.array([0] * 6 + [1] * 12, np.int64),
                  anchor_player=np.array([0, 0, 0, 1, 1, 1] + [0] * 6 + [1] * 6, np.int64))
    return model, arrays


def schedule(arrays, updates):
    config = {**copy.deepcopy(v7.FIT_CONFIG), "updates": updates}
    return v7.make_batch_schedule(arrays["root_strata"], arrays["anchor_game"], arrays["anchor_player"], config)


def full_results(kl1=(23, 31, 31), kl0=(23, 31, 31)):
    return {arm: {str(step): {"root_selected_action_matches": count, "root_count": 32}
                  for step, count in zip((128, 512, 2048), counts)}
            for arm, counts in (("kl_1", kl1), ("kl_0", kl0))}


class ConstraintV7FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_protocol_changes_only_update_count_and_fixed_kl_arm(self):
        self.assertEqual(v7.FIT_CONFIG, {**v6.FIT_CONFIG, "updates": 2048})
        self.assertEqual(v7.ARMS, {"kl_1": 1.0, "kl_0": 0.0})
        self.assertEqual(v7.SNAPSHOTS, (128, 512, 2048))
        self.assertIs(v7.make_batch_schedule, v6.make_batch_schedule)
        self.assertIs(v7.head_loss, v6.head_loss)
        self.assertIs(v7.freeze_policy_head, v6.freeze_policy_head)
        self.assertIs(v7.frozen_state_sha256, v6.frozen_state_sha256)

    def test_extended_schedule_keeps_the_exact_128_prefix_but_not_end_rng(self):
        _, arrays = fixture()
        original, extended = schedule(arrays, 128), schedule(arrays, 2048)
        for key in ("root", "anchor"):
            self.assertEqual(extended[key].shape, (2048, 32))
            np.testing.assert_array_equal(extended[key][:128], original[key])
        self.assertNotEqual(original["rng_state"], extended["rng_state"])
        self.assertNotEqual(v6.batch_fingerprint(original), v6.batch_fingerprint(extended))

    def test_kl1_prefix_reproduces_v6_parameters_optimizer_rng_and_history(self):
        base, arrays = fixture()
        old_batches, extended = schedule(arrays, 128), schedule(arrays, 2048)
        old, old_opt, old_state, _ = v6.fit_cached_head(
            base, arrays, old_batches, copy.deepcopy(v6.FIT_CONFIG), "terminal_head")
        prefix = []

        class PrefixCaptured(Exception):
            pass

        def capture(state):
            if state["step"] == 128:
                prefix.append(copy.deepcopy(state))
                raise PrefixCaptured

        # Exercise the real 2,048-row recipe, stopping at its first snapshot.
        with self.assertRaises(PrefixCaptured):
            v7.fit_cached_head(base, arrays, extended, copy.deepcopy(v7.FIT_CONFIG), "kl_1", on_update=capture)
        state = prefix[0]
        new = copy.deepcopy(base)
        new.policy.load_state_dict(state["policy_state_dict"])
        new_opt = torch.optim.AdamW(v7.freeze_policy_head(new), lr=.0003, weight_decay=.0001,
                                   betas=(.9, .999), eps=1e-8)
        new_opt.load_state_dict(state["optimizer_state_dict"])
        assert_tree_equal(self, old.state_dict(), new.state_dict())
        assert_tree_equal(self, old_opt.state_dict(), new_opt.state_dict())
        assert_tree_equal(self, old_state["torch_rng"], state["torch_rng"])
        self.assertEqual(old_state["history"], state["history"])
        self.assertNotEqual(old_state["numpy_rng"], state["numpy_rng"])
        self.assertEqual(state["numpy_rng"], extended["rng_state"])
        reference = old.checkpoint(optimizer_state_dict=old_opt.state_dict(), torch_rng=old_state["torch_rng"],
            fit_history=old_state["history"], batch_indices={key: torch.from_numpy(old_batches[key].copy())
                for key in ("root", "anchor")}, numpy_rng=old_batches["rng_state"],
            batch_sha256=v6.batch_fingerprint(old_batches), updates=128)
        proof = v7.prove_prefix_reproduction(new.state_dict(), new_opt.state_dict(), state["torch_rng"],
            state["history"], extended, reference, old_batches, old_batches["rng_state"])
        self.assertIs(proof["passed"], True)
        self.assertEqual(proof["updates"], 128)
        for key in ("model_bytes_equal", "optimizer_state_equal", "torch_rng_equal", "prefix_numpy_rng_equal",
                    "batch_prefix_equal", "loss_history_equal", "extended_schedule_rng_is_distinct_metadata"):
            self.assertIs(proof[key], True, key)
        for corruption in ("policy", "frozen", "optimizer", "torch_rng", "history", "prefix", "prefix_rng"):
            model_state, optimizer, rng, history = (copy.deepcopy(new.state_dict()), copy.deepcopy(new_opt.state_dict()),
                                                   state["torch_rng"].clone(), copy.deepcopy(state["history"]))
            batches, prefix_rng = copy.deepcopy(extended), copy.deepcopy(old_batches["rng_state"])
            if corruption == "policy":
                model_state["policy.weight"][0, 0] += .125
            elif corruption == "frozen":
                model_state["value.2.bias"][0] += .125
            elif corruption == "optimizer":
                optimizer["state"][0]["exp_avg"][0, 0] += .125
            elif corruption == "torch_rng":
                rng[0] ^= 1
            elif corruption == "history":
                history[0]["loss"] += .125
            elif corruption == "prefix":
                batches["root"][0, 0] = (batches["root"][0, 0] + 1) % 32
            else:
                prefix_rng = copy.deepcopy(extended["rng_state"])
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                v7.prove_prefix_reproduction(model_state, optimizer, rng, history, batches,
                                            reference, old_batches, prefix_rng)

    def test_kl_ablation_changes_only_policy_and_preserves_the_original_baseline(self):
        base, arrays = fixture()
        before = copy.deepcopy(base.state_dict())
        batches = schedule(arrays, 64)
        config = {**v7.FIT_CONFIG, "updates": 64}
        models = {}
        for arm in ("kl_1", "kl_0"):
            callbacks = []
            model, optimizer, state, metrics = v7.fit_cached_head(base, arrays, batches, config, arm,
                                                                 on_update=lambda row: callbacks.append(row["step"]))
            self.assertEqual(callbacks, [16, 32, 48, 64])
            self.assertEqual(state["step"], 64)
            self.assertEqual(state["coefficient"], v7.ARMS[arm])
            self.assertEqual(len(state["history"]), 64)
            for row in state["history"]:
                self.assertAlmostEqual(row["loss"], row["target_ce"] + v7.ARMS[arm] * row["anchor_kl"], places=6)
            self.assertEqual(sum(p.numel() for group in optimizer.param_groups for p in group["params"]), 5160)
            self.assertEqual({name for name, p in model.named_parameters() if p.requires_grad},
                             {"policy.weight", "policy.bias"})
            for name, tensor in model.state_dict().items():
                if not name.startswith("policy."):
                    self.assertEqual(tensor_bytes(tensor), tensor_bytes(before[name]), name)
            self.assertFalse(torch.equal(model.policy.weight, before["policy.weight"]))
            self.assertIsInstance(metrics, dict)
            models[arm] = model
        self.assertFalse(torch.equal(models["kl_1"].policy.weight, models["kl_0"].policy.weight))
        assert_tree_equal(self, base.state_dict(), before)

    def test_stop_before_update37_resumes_saved32_to_exact_uninterrupted_fit(self):
        base, arrays = fixture()
        config, batches = {**v7.FIT_CONFIG, "updates": 64}, schedule(arrays, 64)
        full, optimizer, complete, metrics = v7.fit_cached_head(base, arrays, batches, config, "kl_1")
        checks, saved = [], []

        class StopRequested(Exception):
            pass

        def stop():
            checks.append(len(checks) + 1)
            if len(checks) == 37:
                raise StopRequested

        rng_before = torch.get_rng_state().clone()
        with self.assertRaises(StopRequested):
            v7.fit_cached_head(base, arrays, batches, config, "kl_1", check_stop=stop,
                               on_update=lambda row: saved.append(copy.deepcopy(row)))
        self.assertEqual([row["step"] for row in saved], [16, 32])
        self.assertEqual(len(checks), 37)
        assert_tree_equal(self, torch.get_rng_state(), rng_before)
        resumed, resumed_optimizer, resumed_state, resumed_metrics = v7.fit_cached_head(
            base, arrays, batches, config, "kl_1", resume=saved[-1])
        assert_tree_equal(self, resumed.state_dict(), full.state_dict())
        assert_tree_equal(self, resumed_optimizer.state_dict(), optimizer.state_dict())
        assert_tree_equal(self, resumed_state, complete)
        self.assertEqual(resumed_metrics, metrics)
        callbacks = []
        v7.fit_cached_head(base, arrays, batches, config, "kl_1", resume=complete,
                           on_update=lambda row: callbacks.append(row))
        self.assertFalse(callbacks)

    def test_preexisting_stop_prevents_any_update_or_saved_generation(self):
        base, arrays = fixture()
        config, batches = {**v7.FIT_CONFIG, "updates": 64}, schedule(arrays, 64)
        before = copy.deepcopy(base.state_dict())
        callbacks = []

        def stop():
            raise InterruptedError("synthetic STOP")

        with self.assertRaises(InterruptedError):
            v7.fit_cached_head(base, arrays, batches, config, "kl_0", check_stop=stop,
                               on_update=lambda state: callbacks.append(state))
        self.assertFalse(callbacks)
        assert_tree_equal(self, before, base.state_dict())

    def test_resume_rejects_other_arm_coefficient_recipe_schedule_and_frozen_state(self):
        base, arrays = fixture()
        config, batches = {**v7.FIT_CONFIG, "updates": 64}, schedule(arrays, 64)
        _, _, state, _ = v7.fit_cached_head(base, arrays, batches, config, "kl_1")
        changes = {"arm": "kl_0", "coefficient": 0., "fit_config_sha256": "0" * 64,
                   "batch_sha256": "0" * 64, "frozen_state_sha256": "0" * 64, "step": -1}
        for field, value in changes.items():
            altered = copy.deepcopy(state)
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                v7.fit_cached_head(base, arrays, batches, config, "kl_1", resume=altered)

    def test_classification_uses_all_fixed_snapshots_and_distinguishes_budget_from_anchoring(self):
        cases = [
            (full_results(kl1=(23, 32, 30)), "budget", 512, None, True, False),
            (full_results(kl1=(23, 30, 32), kl0=(32, 32, 32)), "budget", 2048, 128, True, False),
            (full_results(kl1=(23, 31, 31), kl0=(23, 32, 30)), "anchoring", None, 512, False, True),
            (full_results(), "neither", None, None, False, False),
        ]
        for results, classification, step1, step0, budget, anchoring in cases:
            with self.subTest(classification=classification, step=step1):
                result = v7.classify_results(results)
                self.assertIs(result["complete"], True)
                self.assertEqual(result["classification"], classification)
                self.assertEqual(result["first_full_match_step"], {"kl_1": step1, "kl_0": step0})
                self.assertEqual(result["budget_evidence"], budget)
                self.assertEqual(result["anchoring_evidence"], anchoring)

    def test_partial_or_malformed_evidence_cannot_receive_a_completed_classification(self):
        for corruption in ("missing_arm", "missing_snapshot", "wrong_count", "negative_matches",
                           "too_many_matches", "boolean_matches", "unexpected_initial_full_match"):
            results = full_results()
            if corruption == "missing_arm":
                del results["kl_0"]
            elif corruption == "missing_snapshot":
                del results["kl_1"]["2048"]
            else:
                row = results["kl_1"]["128"]
                if corruption == "wrong_count":
                    row["root_count"] = 31
                else:
                    row["root_selected_action_matches"] = {"negative_matches": -1, "too_many_matches": 33,
                        "boolean_matches": True, "unexpected_initial_full_match": 32}[corruption]
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                v7.classify_results(results)

    def test_run_fit_exports_six_snapshots_and_256_generations_and_reuses_verified_artifacts(self):
        # Reuse the established synthetic v6 provenance fixture, without
        # importing or reading any actual game, candidate, or evaluation data.
        helper_path = Path(__file__).with_name("test_terminal_v6_fit.py")
        spec = importlib.util.spec_from_file_location("synthetic_v6_fixture_for_v7", helper_path)
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dataset, old_config, old_identity, base, arrays, metadata = helpers.dataset_fixture(directory)
            baseline_path = Path(old_config["checkpoint"])
            with torch.no_grad():
                base.policy.weight.zero_()
                base.policy.bias.zero_()
                base.policy.bias[21] = 5
            torch.save(base.checkpoint(training_maps=[]), baseline_path)
            old_config["checkpoint_sha256"] = v7.C.sha256(baseline_path)
            old_identity["checkpoint_sha256"] = old_config["checkpoint_sha256"]
            old_identity["config"] = old_config
            for prefix in ("root", "anchor"):
                with torch.no_grad():
                    mask = torch.from_numpy(arrays[f"{prefix}_mask"]).bool()
                    arrays[f"{prefix}_prior"] = base.policy(torch.from_numpy(arrays[f"{prefix}_hidden"])).masked_fill(
                        ~mask, -torch.inf).softmax(-1).numpy()
            arrays["puct_actions"][:] = 21
            arrays["terminal_actions"][:] = 21
            arrays["terminal_actions"][:9] = 37
            np.savez_compressed(dataset, **arrays)
            metadata["identity_sha256"] = v7.C.fingerprint(old_identity)
            metadata["npz_sha256"] = v7.C.sha256(dataset)
            dataset.with_suffix(".json").write_text(json.dumps(metadata))
            old_identity_path = directory / "synthetic-v6-identity.json"
            old_identity_path.write_text(json.dumps(old_identity))
            loaded, _, _ = v6.load_dataset(dataset, old_config, old_identity)
            original_batches = v6.make_batch_schedule(loaded["root_strata"], loaded["anchor_game"],
                                                      loaded["anchor_player"], old_config["fit"])
            original, original_opt, original_state, _ = v6.fit_cached_head(
                base, loaded, original_batches, old_config["fit"], "terminal_head")
            reference_path = directory / "synthetic-v6-terminal.pt"
            torch.save(original.checkpoint(arm="terminal_head", updates=128,
                dataset_sha256=v7.C.sha256(dataset), parent_checkpoint_sha256=v7.C.sha256(baseline_path),
                optimizer_state_dict=original_opt.state_dict(), torch_rng=original_state["torch_rng"],
                numpy_rng=original_state["numpy_rng"], fit_history=original_state["history"],
                batch_sha256=v6.batch_fingerprint(original_batches),
                batch_indices={key: torch.from_numpy(original_batches[key].copy()) for key in ("root", "anchor")}),
                reference_path)
            reference_batches_path = directory / "synthetic-v6-batches.npz"
            np.savez_compressed(reference_batches_path, **{key: original_batches[key] for key in ("root", "anchor")})
            config = json.loads((v7.C.ROOT / "configs/cached-head-constraint-v7.json").read_text())
            paths = {"checkpoint": baseline_path, "dataset": dataset, "dataset_metadata": dataset.with_suffix(".json"),
                     "reference_terminal_checkpoint": reference_path, "reference_batches": reference_batches_path,
                     "v6_identity": old_identity_path}
            # These prerequisites belong to the supervisor, not run_fit. Keep
            # even their unused configuration references synthetic and local.
            for name in ("v6_final_audit", "training_diagnosis"):
                paths[name] = directory / f"synthetic-{name}.json"
                paths[name].write_text(json.dumps({"synthetic": True}))
            for name, path in paths.items():
                config[name] = {"path": str(path), "sha256": v7.C.sha256(path)}
            identity = {"config": config, "sources": {name: v7.C.sha256(v7.C.ROOT / name) for name in (
                "alpha0/constraint_v7_fit.py", "alpha0/constraint_v7_common.py",
                "alpha0/terminal_v6_fit.py", "alpha0/terminal_v6_common.py")}}
            run, progress = directory / "run", directory / "progress"
            with patch.object(v7.C, "RUN", run), patch.object(v7.C, "PROGRESS", progress), \
                 patch.object(v7.C, "check_stop", lambda _deadline: None):
                manifest = v7.run_fit(config, identity, deadline=float("inf"))
                self.assertIs(manifest["complete"], True)
                self.assertIs(manifest["automatic_promotion"], False)
                self.assertIs(manifest["reproduction"]["passed"], True)
                self.assertEqual(len(list((run / "fit").glob("kl_*/checkpoint-*.pt"))), 6)
                self.assertEqual(len(list((run / "fit").glob("kl_*/states/step-*.pt"))), 256)
                self.assertEqual(len(list((run / "fit").glob("kl_*/states/step-*.json"))), 256)
                self.assertEqual(len(manifest["files"]), 529)
                for arm in v7.ARMS:
                    self.assertEqual(set(manifest["checkpoints"][arm]), {"128", "512", "2048"})
                    for step in (128, 512, 2048):
                        payload = torch.load(run / "fit" / arm / f"checkpoint-{step:04d}.pt",
                                             map_location="cpu", weights_only=True)
                        self.assertEqual(payload["updates"], step)
                        self.assertEqual(payload["anchor_kl_weight"], v7.ARMS[arm])
                        self.assertEqual(payload["trainable_parameters"], 5160)
                        self.assertFalse(payload["automatic_promotion"])
                        for name, tensor in base.state_dict().items():
                            if not name.startswith("policy."):
                                self.assertEqual(tensor_bytes(tensor), tensor_bytes(payload["model_state_dict"][name]), name)
                snapshots = {path.relative_to(directory): (v7.C.sha256(path), path.stat().st_mtime_ns)
                             for parent in (run / "fit", progress) for path in parent.rglob("*") if path.is_file()}
                with patch.object(v7, "fit_cached_head", side_effect=AssertionError("completed fit reran SGD")):
                    reused = v7.run_fit(config, identity, deadline=float("inf"))
                self.assertEqual(reused, manifest)
                self.assertEqual(snapshots, {name: (v7.C.sha256(directory / name), (directory / name).stat().st_mtime_ns)
                                             for name in snapshots})
                manifest_path = run / "fit/manifest.json"
                manifest_bytes = manifest_path.read_bytes()
                sidecar = run / "fit/kl_1/checkpoint-0128.json"
                sidecar_bytes = sidecar.read_bytes()
                for corruption in ("reported_arm", "automatic_promotion"):
                    changed = json.loads(manifest_bytes)
                    if corruption == "reported_arm":
                        changed["checkpoints"]["kl_1"]["128"]["arm"] = "kl_0"
                        sidecar.write_text(json.dumps(changed["checkpoints"]["kl_1"]["128"]))
                        changed["files"]["kl_1/checkpoint-0128.json"] = v7.C.sha256(sidecar)
                    else:
                        changed["automatic_promotion"] = True
                    manifest_path.write_text(json.dumps(changed))
                    try:
                        # No old progress copy may conceal a missing semantic
                        # check in the restored fit's own completion evidence.
                        with self.subTest(corruption=corruption), \
                             patch.object(v7.C, "PROGRESS", directory / f"fresh-progress-{corruption}"), \
                             patch.object(v7, "fit_cached_head", side_effect=AssertionError("corrupt fit reached SGD")), \
                             self.assertRaises(ValueError):
                            v7.run_fit(config, identity, deadline=float("inf"))
                    finally:
                        sidecar.write_bytes(sidecar_bytes)
                        manifest_path.write_bytes(manifest_bytes)


if __name__ == "__main__":
    unittest.main()
