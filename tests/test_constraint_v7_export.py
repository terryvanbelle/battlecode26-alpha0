"""Offline v7 recovery checks: synthetic tensors only; never run SGD or actors."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("export_constraint_v7", REPO / "scripts/export_constraint_v7.py")
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)
ARMS = {"kl_1": 1.0, "kl_0": 0.0}
SNAPSHOTS = (128, 512, 2048)


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value.encode() if isinstance(value, str) else value)
    return path


def write_json(root, name, value):
    return write(root, name, json.dumps(value, sort_keys=True, allow_nan=False))


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
    return path


def save_arrays(root, name, /, **arrays):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def fingerprint(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def batch_fingerprint(batches, rng):
    digest = hashlib.sha256()
    for name in ("root", "anchor"):
        array = np.ascontiguousarray(batches[name], dtype="<i8")
        digest.update(json.dumps([name, list(array.shape)], separators=(",", ":")).encode())
        digest.update(array.tobytes())
    digest.update(fingerprint(rng).encode())
    return digest.hexdigest()


def frozen_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.items()):
        if name in {"policy.weight", "policy.bias"}:
            continue
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class SyntheticRun:
    """Write the persistence protocol without importing or executing a fitter."""

    def __init__(self, root):
        self.root = root
        self.config = json.loads((REPO / "configs/cached-head-constraint-v7.json").read_text())
        self.model = {"encoder.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                      "value.bias": torch.tensor([.25]),
                      "policy.weight": torch.zeros((40, 128)), "policy.bias": torch.zeros(40)}
        self.parent = {"config": {"initialization": {"artifact": "artifacts/prior/prior.pt",
                                                    "artifact_sha256": "a" * 64}},
                       "iteration": 2, "training_maps": ["synthetic"]}
        self.baseline = {"model_state_dict": self.model, **self.parent}
        self.batches = {"root": np.tile(np.arange(32, dtype=np.int64), (2048, 1)),
                        "anchor": np.tile(np.arange(32, dtype=np.int64) + 32, (2048, 1))}
        self.numpy_rng = np.random.default_rng(203101).bit_generator.state
        self.torch_rng = torch.Generator().manual_seed(203101).get_state()
        self.batch_hash = batch_fingerprint(self.batches, self.numpy_rng)
        self.frozen_hash = frozen_fingerprint(self.model)
        self.pins = {}
        for key, (name, _expected) in E.PINS.items():
            if key == "checkpoint":
                path = save(root, name, self.baseline)
            elif key == "reference_terminal_checkpoint":
                path = save(root, name, self.baseline)
            elif key == "dataset":
                path = save_arrays(root, name, root_hidden=np.ones((32, 128), np.float32),
                                   anchor_hidden=np.ones((512, 128), np.float32))
            elif key == "reference_batches":
                path = save_arrays(root, name, **{k: v[:128] for k, v in self.batches.items()})
            else:
                path = write_json(root, name, {"synthetic": key})
            self.pins[key] = (name, E.sha256(path))
            self.config[key] = {"path": name, "sha256": E.sha256(path)}
        config_path = write_json(root, "configs/cached-head-constraint-v7.json", self.config)
        write(root, "docs/cached-head-constraint-v7.md", "# Synthetic frozen v7 protocol\n")
        source_name = "alpha0/constraint_v7_fit.py"
        source = write(root, f"{E.PROGRESS}/source/{source_name}", "# Synthetic source evidence\n")
        self.identity = {"schema_version": 1, "experiment_id": self.config["experiment_id"],
                         "config": self.config, "config_file_sha256": E.sha256(config_path),
                         "inputs": {name: digest for name, digest in self.pins.values()},
                         "sources": {source_name: E.sha256(source)}, "training_only": True,
                         "validation_data_read": False, "benchmark_data_read": False,
                         "engine_games": 0, "promotion_allowed": False}
        write_json(root, f"{E.PROGRESS}/identity.json", self.identity)
        write_json(root, f"{E.RUN}/status.json", {"phase": "stopped", "synthetic": True})
        self.settings = {**self.config["fit"], "updates": 2048, "anchor_kl_weight": 1}
        self.fit_identity = {"identity_sha256": fingerprint(self.identity),
                             "input_hashes": {key: self.config[key]["sha256"] for key in (
                                 "checkpoint", "dataset", "dataset_metadata", "reference_terminal_checkpoint",
                                 "reference_batches", "v6_identity")},
                             "batch_sha256": self.batch_hash, "settings": self.settings}
        self.fit_hash = fingerprint(self.fit_identity)
        write_json(root, f"{E.RUN}/fit/identity.json", self.fit_identity)
        save_arrays(root, f"{E.RUN}/fit/batches.npz", **self.batches)
        write_json(root, f"{E.RUN}/fit/batch-rng.json", self.numpy_rng)
        write_json(root, f"{E.RUN}/fit/parent-metadata.json", self.parent)
        write_json(root, f"{E.PROGRESS}/diagnostics.json", {"training_data_only": True})
        self.generation("kl_1", 16)

    def state(self, arm, step):
        policy = {key: self.model["policy." + key].clone() + step / 16384 for key in ("weight", "bias")}
        # Explicit optimizer payloads preserve nontrivial moments without taking an update.
        optimizer = {"state": {
            index: {"step": torch.tensor(float(step)), "exp_avg": torch.full_like(value, .125),
                    "exp_avg_sq": torch.full_like(value, .0625)}
            for index, value in enumerate(policy.values())},
            "param_groups": [{"lr": .0003, "betas": (.9, .999), "eps": 1e-8,
                              "weight_decay": .0001, "amsgrad": False, "maximize": False,
                              "foreach": None, "capturable": False, "differentiable": False,
                              "fused": None, "params": [0, 1]}]}
        return {"arm": arm, "coefficient": ARMS[arm], "step": step,
                "policy_state_dict": policy, "optimizer_state_dict": optimizer,
                "torch_rng": self.torch_rng.clone(), "numpy_rng": copy.deepcopy(self.numpy_rng),
                "batch_sha256": self.batch_hash, "frozen_state_sha256": self.frozen_hash,
                "fit_config_sha256": fingerprint(self.settings),
                "history": [{"step": index, "loss": 1.0, "target_ce": 1.0, "anchor_kl": 0.0}
                            for index in range(1, step + 1)]}

    def generation(self, arm, step):
        path = save(self.root, f"{E.RUN}/fit/{arm}/states/step-{step:04d}.pt", self.state(arm, step))
        write_json(self.root, path.with_suffix(".json").relative_to(self.root),
                   {"step": step, "sha256": E.sha256(path), "fit_identity_sha256": self.fit_hash})
        return path

    def snapshot(self, arm, step):
        state = self.state(arm, step)
        model = copy.deepcopy(self.model)
        model.update({"policy." + key: value for key, value in state["policy_state_dict"].items()})
        config = copy.deepcopy(self.config)
        config["initialization"] = {"artifact": self.config["checkpoint"]["path"],
            "artifact_sha256": self.config["checkpoint"]["sha256"],
            "method": "fresh cp2 frozen features; training-only KL/budget diagnostic"}
        payload = {"model_state_dict": model, "config": config, "experiment_id": self.config["experiment_id"],
                   "arm": arm, "updates": step, "gradient_updates": step, "anchor_kl_weight": ARMS[arm],
                   "policy_head_only": True, "trainable_parameters": 5160, "automatic_promotion": False,
                   "benchmark_data_used": False, "validation_data_used": False,
                   "parent_checkpoint_sha256": self.config["checkpoint"]["sha256"],
                   "parent_checkpoint_metadata": self.parent,
                   "dataset_sha256": self.config["dataset"]["sha256"],
                   "dataset_metadata_sha256": self.config["dataset_metadata"]["sha256"],
                   "reference_v6_checkpoint_sha256": self.config["reference_terminal_checkpoint"]["sha256"],
                   "identity_sha256": fingerprint(self.identity), "fit_identity_sha256": self.fit_hash,
                   "frozen_state_sha256": self.frozen_hash, "batch_sha256": self.batch_hash,
                   "batch_indices": {key: torch.from_numpy(value.copy()) for key, value in self.batches.items()},
                   "optimizer_state_dict": state["optimizer_state_dict"], "torch_rng": state["torch_rng"],
                   "numpy_rng": state["numpy_rng"], "fit_history": state["history"],
                   "fit_metrics": {"root_count": 32, "root_selected_action_matches": 23}}
        path = save(self.root, f"{E.RUN}/fit/{arm}/checkpoint-{step:04d}.pt", payload)
        row = {"path": path.relative_to(self.root).as_posix(), "sha256": E.sha256(path),
               "fit_identity_sha256": self.fit_hash, "arm": arm, "updates": step,
               "anchor_kl_weight": ARMS[arm], "trainable_parameters": 5160,
               "frozen_tensors_equal": True, "value_logits_equal": True, "metrics": payload["fit_metrics"]}
        write_json(self.root, path.with_suffix(".json").relative_to(self.root), row)
        return row

    def complete(self):
        records = {arm: {} for arm in ARMS}
        for arm in ARMS:
            for step in range(16, 2049, 16):
                self.generation(arm, step)
            for step in SNAPSHOTS:
                records[arm][str(step)] = self.snapshot(arm, step)
        proof = {"passed": True, "updates": 128, "reference_checkpoint_sha256": self.config["reference_terminal_checkpoint"]["sha256"],
                 "fit_identity_sha256": self.fit_hash, **{key: True for key in (
                     "model_bytes_equal", "optimizer_state_equal", "torch_rng_equal", "prefix_numpy_rng_equal",
                     "batch_prefix_equal", "loss_history_equal", "extended_schedule_rng_is_distinct_metadata")}}
        write_json(self.root, f"{E.RUN}/fit/reproduction-128.json", proof)
        directory = self.root / E.RUN / "fit"
        self.manifest = {"schema_version": 1, "experiment_id": self.config["experiment_id"], "complete": True,
            "identity_sha256": fingerprint(self.identity), "fit_identity_sha256": self.fit_hash,
            "fit_identity": self.fit_identity, "updates_per_arm": 2048, "snapshots": list(SNAPSHOTS),
            "trainable_parameters": 5160, "checkpoints": records, "reproduction": proof,
            "batch_schedule": {"path": f"{E.RUN}/fit/batches.npz", "sha256": E.sha256(directory / "batches.npz"),
                "batch_sha256": self.batch_hash, "root_shape": [2048, 32], "anchor_shape": [2048, 32], "seed": 203101},
            "training_data_only": True, "validation_data_used": False, "benchmark_data_used": False,
            "automatic_promotion": False, "frozen_state_sha256": self.frozen_hash,
            "files": {path.relative_to(directory).as_posix(): E.sha256(path)
                      for path in directory.rglob("*") if path.is_file() and path.name != "manifest.json"}}
        self.publish_manifest()

    def publish_manifest(self):
        write_json(self.root, f"{E.RUN}/fit/manifest.json", self.manifest)


class ConstraintV7ExportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        self.root.mkdir()
        self.fake_proc = Path(self.temp.name) / "proc"
        self.fake_proc.mkdir()
        proc_patch = mock.patch.object(E, "PROC", self.fake_proc)
        proc_patch.start()
        self.addCleanup(proc_patch.stop)
        self.fixture = SyntheticRun(self.root)
        pins_patch = mock.patch.object(E, "PINS", self.fixture.pins)
        pins_patch.start()
        self.addCleanup(pins_patch.stop)

    def archive_manifest(self, path):
        with tarfile.open(path) as archive:
            return json.load(archive.extractfile(E.MANIFEST))

    def test_partial_generation_roundtrip_preserves_exact_state_and_leaves_restore_stopped(self):
        result = E.create_export(self.root)
        self.assertTrue(E.verify(result["archive"])["verified"])
        manifest = self.archive_manifest(result["archive"])
        self.assertEqual(manifest["coverage"], {"fit_complete": False, "fixed_snapshots": 0,
            "generation_pairs": 1, "latest_generation": {"kl_1": 16, "kl_0": 0}})
        generation = f"{E.RUN}/fit/kl_1/states/step-0016.pt"
        self.assertIn(generation, manifest["checkpoints"])
        target = Path(self.temp.name) / "restored"
        E.restore(result["archive"], target)
        original = torch.load(self.root / generation, weights_only=True)
        recovered = torch.load(target / generation, weights_only=True)
        self.assertEqual(E._state_digest(original), E._state_digest(recovered))
        self.assertEqual((self.root / generation).read_bytes(), (target / generation).read_bytes())
        for key in ("policy_state_dict", "optimizer_state_dict", "torch_rng", "numpy_rng", "history"):
            self.assertEqual(E._state_digest(original[key]), E._state_digest(recovered[key]), key)
        self.assertTrue((target / E.RUN / "STOP").is_file())
        with self.assertRaises(FileExistsError):
            E.restore(result["archive"], target)

    def test_each_of_five_local_actor_locks_blocks_export_without_changing_stops(self):
        self.assertEqual(len(E.LOCKS), 5)
        for name in E.STOPS:
            write(self.root, name, "retained stop\n")
        for name in E.LOCKS:
            path = write(self.root, name, b"")
            with self.subTest(lock=name), path.open("a") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(RuntimeError):
                    E.create_export(self.root)
        result = E.create_export(self.root)
        self.assertTrue(result["verified"])
        for name in E.STOPS:
            self.assertEqual((self.root / name).read_text(), "retained stop\n")
        names = {row["path"] for row in self.archive_manifest(result["archive"])["files"]}
        self.assertFalse(names.intersection(E.STOPS))
        self.assertFalse(names.intersection(E.LOCKS))

    def test_matching_actor_with_permission_denied_cwd_blocks_export_and_restore(self):
        result = E.create_export(self.root)
        for pid, arguments in ((900031, ["python", "-m", "alpha0.constraint_v7", "--worker", "job.json"]),
                               (900032, ["java", "alpha0.EngineBridge"]),
                               (900033, ["python", "-m", "pytest"])):
            directory = self.fake_proc / str(pid)
            directory.mkdir()
            (directory / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in arguments) + b"\0")
            (directory / "cwd").symlink_to(self.root, target_is_directory=True)
        resolve = Path.resolve

        def inaccessible_cwd(path, *args, **kwargs):
            if path.parent.parent == self.fake_proc and path.name == "cwd":
                raise PermissionError("synthetic cross-namespace cwd")
            return resolve(path, *args, **kwargs)

        with mock.patch.object(Path, "resolve", autospec=True, side_effect=inaccessible_cwd):
            self.assertEqual(E.active_actor_pids(self.root), [900031, 900032])
            with self.assertRaises(RuntimeError):
                E.create_export(self.root)
            with self.assertRaises(RuntimeError):
                E.restore(result["archive"], Path(self.temp.name) / "restored")

    def test_complete_archive_contains_all_six_snapshots_and_256_generation_pairs(self):
        self.fixture.complete()
        write(self.root, f"{E.RUN}/experiment.log", "transient")
        write(self.root, f"{E.RUN}/experiment.pid", "999999")
        write(self.root, f"{E.RUN}/.partial.json", "{}")
        write(self.root, "vendor/engine/foreign.jar", "foreign runtime")
        result = E.create_export(self.root)
        manifest = self.archive_manifest(result["archive"])
        self.assertEqual(manifest["coverage"], {"fit_complete": True, "fixed_snapshots": 6,
            "generation_pairs": 256, "latest_generation": {"kl_1": 2048, "kl_0": 2048}})
        names = {row["path"] for row in manifest["files"]}
        snapshots = {f"{E.RUN}/fit/{arm}/checkpoint-{step:04d}.pt" for arm in ARMS for step in SNAPSHOTS}
        generations = {f"{E.RUN}/fit/{arm}/states/step-{step:04d}.pt" for arm in ARMS for step in range(16, 2049, 16)}
        self.assertEqual(len(snapshots), 6)
        self.assertEqual(len(generations), 256)
        self.assertTrue((snapshots | generations).issubset(manifest["checkpoints"]))
        self.assertTrue({str(Path(name).with_suffix(".json")) for name in snapshots | generations}.issubset(names))
        for name in ("identity.json", "batches.npz", "batch-rng.json", "parent-metadata.json", "reproduction-128.json"):
            self.assertIn(f"{E.RUN}/fit/{name}", names)
        self.assertFalse(any(name.endswith((".lock", ".log", ".pid")) or "/.partial" in name or
                             name.startswith("vendor/") for name in names))
        target = Path(self.temp.name) / "complete-restored"
        E.restore(result["archive"], target, expected_sha256=result["sha256"])
        for row in manifest["files"]:
            # References document the historical code, and are deliberately not installed as current code.
            if row["path"].startswith(("configs/", "docs/")):
                continue
            self.assertEqual((self.root / row["path"]).read_bytes(), (target / row["path"]).read_bytes(), row["path"])
        for name in snapshots:
            before = torch.load(self.root / name, weights_only=True)
            after = torch.load(target / name, weights_only=True)
            self.assertEqual(E._state_digest(before), E._state_digest(after), name)
            for key in ("model_state_dict", "optimizer_state_dict", "torch_rng", "numpy_rng", "batch_indices", "fit_history"):
                self.assertEqual(E._state_digest(before[key]), E._state_digest(after[key]), f"{name}:{key}")

    def test_complete_claim_rejects_removed_coverage_even_after_manifest_hashes_are_updated(self):
        self.fixture.complete()
        cases = (("kl_0/checkpoint-2048.pt", "kl_0/checkpoint-2048.json"),
                 ("kl_1/states/step-0032.pt", "kl_1/states/step-0032.json"),
                 ("batch-rng.json",), ("parent-metadata.json",), ("reproduction-128.json",))
        for removed in cases:
            with self.subTest(removed=removed):
                backup = {name: (self.root / E.RUN / "fit" / name).read_bytes() for name in removed}
                for name in removed:
                    (self.root / E.RUN / "fit" / name).unlink()
                    self.fixture.manifest["files"].pop(name)
                self.fixture.publish_manifest()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    E.plan(self.root)
                for name, data in backup.items():
                    path = write(self.root, f"{E.RUN}/fit/{name}", data)
                    self.fixture.manifest["files"][name] = E.sha256(path)
                self.fixture.publish_manifest()

    def test_partial_generation_requires_contiguous_pairs_and_consistent_sidecars(self):
        generation = self.fixture.generation("kl_1", 48)
        with self.assertRaises(ValueError):
            E.plan(self.root)
        generation.unlink()
        generation.with_suffix(".json").unlink()
        generation = self.root / E.RUN / "fit/kl_1/states/step-0016.pt"
        sidecar = generation.with_suffix(".json")
        checkpoint_before = generation.read_bytes()
        before = sidecar.read_bytes()
        generation.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            E.plan(self.root)
        generation.write_bytes(checkpoint_before)
        sidecar.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            E.plan(self.root)
        sidecar.write_bytes(before)
        row = json.loads(before)
        row["step"] = 32
        sidecar.write_text(json.dumps(row))
        with self.assertRaises(ValueError):
            E.plan(self.root)

    def test_snapshot_rejects_missing_rng_wrong_updates_and_frozen_tensor_changes(self):
        row = self.fixture.snapshot("kl_1", 128)
        path = self.root / row["path"]
        baseline = torch.load(path, weights_only=True)
        E.checkpoint_record(path, config=self.fixture.config, baseline=self.fixture.model)
        for change in ("numpy_rng", "updates", "frozen", "optimizer", "batches"):
            payload = copy.deepcopy(baseline)
            if change == "numpy_rng":
                del payload["numpy_rng"]
            elif change == "updates":
                payload["updates"] = 16
            elif change == "frozen":
                payload["model_state_dict"]["value.bias"][0] += 1
            elif change == "optimizer":
                payload["optimizer_state_dict"]["state"][0]["step"] = torch.tensor(16.0)
            else:
                payload["batch_indices"]["root"] = payload["batch_indices"]["root"][:128]
            torch.save(payload, path)
            with self.subTest(change=change), self.assertRaises(ValueError):
                E.checkpoint_record(path, config=self.fixture.config, baseline=self.fixture.model)

    def test_partial_state_cannot_change_rng_or_fit_provenance_by_rehashing_its_sidecar(self):
        path = self.root / E.RUN / "fit/kl_1/states/step-0016.pt"
        baseline = torch.load(path, weights_only=True)
        E.plan(self.root)
        for change in ("numpy_rng", "torch_rng", "batch_sha256", "frozen_state_sha256", "fit_config_sha256"):
            payload = copy.deepcopy(baseline)
            if change == "numpy_rng":
                payload[change]["state"]["state"] ^= 1
            elif change == "torch_rng":
                payload[change][0] ^= 1
            else:
                payload[change] = "0" * 64
            torch.save(payload, path)
            write_json(self.root, path.with_suffix(".json").relative_to(self.root),
                       {"step": 16, "sha256": E.sha256(path), "fit_identity_sha256": self.fixture.fit_hash})
            with self.subTest(change=change), self.assertRaises(ValueError):
                E.plan(self.root)

    def test_snapshot_same_shape_batch_mutation_is_rejected_after_sidecar_rehash(self):
        for step in range(32, 129, 16):
            self.fixture.generation("kl_1", step)
        row = self.fixture.snapshot("kl_1", 128)
        E.plan(self.root)
        path = self.root / row["path"]
        payload = torch.load(path, weights_only=True)
        payload["batch_indices"]["root"][0, 0] = 1
        torch.save(payload, path)
        row["sha256"] = E.sha256(path)
        write_json(self.root, path.with_suffix(".json").relative_to(self.root), row)
        with self.assertRaises(ValueError):
            E.plan(self.root)

    def test_progress_conflict_aborts_restore_before_copying_run_or_dependencies(self):
        result = E.create_export(self.root)
        target = Path(self.temp.name) / "conflicting-target"
        name = f"{E.PROGRESS}/diagnostics.json"
        conflict = write(target, name, '{"newer":true}')
        with self.assertRaises(FileExistsError):
            E.restore(result["archive"], target)
        self.assertEqual(conflict.read_text(), '{"newer":true}')
        self.assertFalse((target / E.RUN / "fit").exists())
        self.assertFalse((target / self.fixture.config["checkpoint"]["path"]).exists())

    def test_identical_progress_is_preserved_and_current_reference_code_is_not_replaced(self):
        result = E.create_export(self.root)
        target = Path(self.temp.name) / "existing-progress"
        name = f"{E.PROGRESS}/diagnostics.json"
        original = (self.root / name).read_bytes()
        write(target, name, original)
        current = write(target, "configs/cached-head-constraint-v7.json", '{"current_main":true}')
        E.restore(result["archive"], target)
        self.assertEqual((target / name).read_bytes(), original)
        self.assertEqual(current.read_text(), '{"current_main":true}')

    def test_archive_checksum_covers_trailing_bytes_and_unsafe_members_are_rejected(self):
        result = E.create_export(self.root)
        archive = Path(result["archive"])
        archive.write_bytes(archive.read_bytes() + b"changed after gzip trailer")
        with self.assertRaises(ValueError):
            E.verify(archive)
        malicious = Path(self.temp.name) / "unsafe.tar.gz"
        with tarfile.open(malicious, "w:gz") as bundle:
            for name, data in ((E.MANIFEST, b"{}"), ("../escape", b"bad")):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                bundle.addfile(info, io.BytesIO(data))
        with self.assertRaises(ValueError):
            E.verify(malicious, expected_sha256=E.sha256(malicious))

    def test_repeated_exports_are_immutable_and_do_not_rewrite_checkpoint_inputs(self):
        generation = self.root / E.RUN / "fit/kl_1/states/step-0016.pt"
        before = generation.read_bytes()
        first = E.create_export(self.root)
        archive_before = Path(first["archive"]).read_bytes()
        manifest_before = Path(first["manifest"]).read_bytes()
        second = E.create_export(self.root)
        self.assertNotEqual(first["archive"], second["archive"])
        self.assertEqual(Path(first["archive"]).read_bytes(), archive_before)
        self.assertEqual(Path(first["manifest"]).read_bytes(), manifest_before)
        self.assertEqual(generation.read_bytes(), before)
        self.assertEqual(E.sha256(first["archive"]), first["sha256"])
        self.assertTrue(E.verify(first["archive"])["verified"])
        self.assertTrue(E.verify(second["archive"])["verified"])

    def test_pinned_dependency_and_frozen_source_changes_fail_closed(self):
        name = self.fixture.config["dataset_metadata"]["path"]
        dependency = self.root / name
        before = dependency.read_bytes()
        dependency.write_text('{"synthetic":"changed"}')
        with self.assertRaises(ValueError):
            E.plan(self.root)
        dependency.write_bytes(before)
        source = self.root / E.PROGRESS / "source/alpha0/constraint_v7_fit.py"
        source.write_text("# Changed evidence\n")
        with self.assertRaises(ValueError):
            E.plan(self.root)

    def test_symlink_in_owned_recovery_files_is_rejected(self):
        link = self.root / E.PROGRESS / "linked.json"
        link.symlink_to(self.root / E.PROGRESS / "diagnostics.json")
        with self.assertRaises(ValueError):
            E.plan(self.root)

    def test_repacked_payload_cannot_bypass_checkpoint_state_or_sidecar_verification(self):
        result = E.create_export(self.root)
        with tarfile.open(result["archive"]) as bundle:
            data = {member.name: bundle.extractfile(member).read() for member in bundle.getmembers()}
        name = f"{E.RUN}/fit/kl_1/states/step-0016.pt"
        checkpoint = torch.load(io.BytesIO(data[name]), weights_only=True)
        checkpoint["optimizer_state_dict"]["state"][0]["exp_avg"][0, 0] += .25
        modified = io.BytesIO()
        torch.save(checkpoint, modified)
        data[name] = modified.getvalue()
        manifest = json.loads(data[E.MANIFEST])
        for row in manifest["files"]:
            if row["path"] == name:
                row["size"] = len(data[name])
                row["sha256"] = hashlib.sha256(data[name]).hexdigest()
        data[E.MANIFEST] = json.dumps(manifest).encode()
        repacked = Path(self.temp.name) / "repacked.tar.gz"
        with tarfile.open(repacked, "w:gz") as bundle:
            for member_name, content in data.items():
                info = tarfile.TarInfo(member_name)
                info.size = len(content)
                bundle.addfile(info, io.BytesIO(content))
        with self.assertRaises(ValueError):
            E.verify(repacked, expected_sha256=E.sha256(repacked))


if __name__ == "__main__":
    unittest.main()
