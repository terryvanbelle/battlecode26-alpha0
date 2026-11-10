"""Recovery verifies complete bytes and training state without actors or network."""
import copy
import fcntl
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location("export_terminal_v8", Path(__file__).resolve().parents[1] / "scripts/export_terminal_v8.py")
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value.encode() if isinstance(value, str) else value)
    return path


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
    return path


def proof(winner=0):
    state = {"terminal": True, "winner": winner, "round": 42}
    actions = [0, 1]
    return {"status": "ok", "terminal": True, "valid_game": True, "benchmark_failures": [],
            "winner": winner, "terminal_round": 42, "terminal_state": state, "actions": actions,
            "terminal_state_sha256": E.hashlib.sha256(E.canonical(state).encode()).hexdigest(),
            "actions_sha256": E.hashlib.sha256(E.canonical(actions).encode()).hexdigest()}


def fixture(root):
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    baseline_model = {"encoder.weight": model[0].weight.detach().clone(),
                      "encoder.bias": model[0].bias.detach().clone(),
                      "policy.weight": model[1].weight.detach().clone(),
                      "policy.bias": model[1].bias.detach().clone()}
    baseline = {"model_state_dict": baseline_model,
                "config": {"initialization": {"artifact": "artifacts/prior/prior.pt", "artifact_sha256": "a" * 64}}}
    cp2 = save(root, E.CP2, baseline)
    cp2_sha = E.sha256(cp2)
    for name in E.REFERENCE_FILES:
        write(root, name, "{}" if name.endswith(".json") else "# Frozen protocol\n")
    write(root, f"{E.RUN}/identity.json", json.dumps({"baseline_sha256": cp2_sha}))
    write(root, f"{E.RUN}/status.json", '{"phase":"complete_review_required"}')
    write(root, f"{E.RUN}/plan.json", '{"immutable":true}')
    shard = root / E.RUN / "sources/training-m0-j1.npz"
    shard.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(shard, hidden=np.ones((2, 4), np.float32), mask=np.ones((2, 2), np.uint8))
    write(root, f"{E.RUN}/sources/training-m0-j1.json", json.dumps({**proof(), "complete": True,
          "source": "selfplay", "cache_npz": shard.name, "cache_sha256": E.sha256(shard)}))
    write(root, f"{E.PROGRESS}/benchmarks/cells/game/attempt-0001.json", '{"evaluation_cell_id":"example","score":0}')
    write(root, f"{E.PROGRESS}/cells/selection-root-a00-s1/attempt-0001.json", json.dumps({**proof(), "identity": "collector-spec-hash", "score": 0, "root_player": 1}))
    write(root, f"{E.PROGRESS}/cells/validation-root-a00-s2/attempt-0001.json", json.dumps({**proof(), "score": 1, "root_player": 0}))
    write(root, f"{E.PROGRESS}/source/collector.py", "# owned source snapshot\n")
    write(root, f"{E.PROGRESS}/roots-validation.json", '{"roots":[]}')
    write(root, f"{E.PROGRESS}/anchors.json", '{"anchors":[]}')
    data = root / E.RUN / "dataset/training.npz"
    data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data, hidden=np.ones((2, 4), np.float32))
    metadata = write(root, f"{E.RUN}/dataset/training.json", json.dumps({"benchmark_data_used": False, "npz_sha256": E.sha256(data)}))
    batches = {"root": torch.zeros((2048, 32), dtype=torch.int64), "anchor": torch.ones((2048, 32), dtype=torch.int64)}
    for arm, name in zip(("terminal_head", "puct_head"), E.CANDIDATES):
        policy = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(policy.parameters())
        policy(torch.ones((1, 4))).sum().backward(); optimizer.step()
        for state in optimizer.state.values():
            state["step"] = torch.tensor(2048.0)
        payload = {"model_state_dict": copy.deepcopy(baseline_model), "optimizer_state_dict": optimizer.state_dict(),
                   "experiment_id": "terminal-target-head-v8",
                   "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.default_rng(7).bit_generator.state,
                   "batch_indices": batches, "parent_checkpoint_sha256": cp2_sha, "policy_head_only": True,
                   "arm": arm, "updates": 2048, "fit_identity_sha256": "b" * 64,
                   "dataset_sha256": E.sha256(data), "dataset_metadata_sha256": E.sha256(metadata), "batch_sha256": "e" * 64}
        payload["model_state_dict"]["policy.bias"] += 0.1
        save(root, name, payload)
    write(root, f"{E.RUN}/fit/manifest.json", json.dumps({"complete": True,
          "checkpoints": {"terminal_head": {}, "puct_head": {}},
          "files": {Path(name).name: E.sha256(root / name) for name in E.CANDIDATES}}))
    for name in (f"{E.RUN}/STOP", f"{E.RUN}/experiment.log", f"{E.RUN}/experiment.pid", f"{E.RUN}/.partial.json"):
        write(root, name, "not exported")
    write(root, "vendor/engine/foreign.jar", b"not exported")
    write(root, "benchmarks/bots/example/RobotPlayer.java", b"not exported")
    return cp2_sha


class TerminalV8ExportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "source"
        self.root.mkdir()
        self.baseline_hash = fixture(self.root)
        self.pin = mock.patch.object(E, "CP2_SHA256", self.baseline_hash)
        self.pin.start()
        # Offline fixture exports must not inspect or interfere with real actors.
        self.fake_proc = Path(self.temp.name) / "proc"
        self.fake_proc.mkdir()
        self.proc_patch = mock.patch.object(E, "PROC", self.fake_proc)
        self.proc_patch.start()

    def tearDown(self):
        self.proc_patch.stop()
        self.pin.stop()
        self.temp.cleanup()

    def test_full_roundtrip_preserves_models_optimizer_rng_batches_and_every_cell(self):
        result = E.create_export(self.root)
        verified = E.verify(result["archive"])
        self.assertTrue(verified["verified"])
        self.assertEqual(verified["completed_candidate_count"], 2)
        with tarfile.open(result["archive"]) as archive:
            manifest = json.load(archive.extractfile(E.MANIFEST))
            names = {entry["path"] for entry in manifest["files"]}
        self.assertIn(f"{E.RUN}/sources/training-m0-j1.npz", names)
        self.assertIn(f"{E.PROGRESS}/cells/selection-root-a00-s1/attempt-0001.json", names)
        self.assertIn(f"{E.PROGRESS}/cells/validation-root-a00-s2/attempt-0001.json", names)
        self.assertIn(f"{E.PROGRESS}/benchmarks/cells/game/attempt-0001.json", names)
        self.assertIn(f"{E.PROGRESS}/source/collector.py", names)
        self.assertFalse(any(name.endswith((".lock", ".log", ".pid")) or name.endswith("/STOP") for name in names))
        self.assertFalse(any(name.startswith(("vendor/", "benchmarks/", "build/")) for name in names))
        target = Path(self.temp.name) / "restored"
        restored = E.restore(result["archive"], target)
        self.assertTrue(restored["restored"])
        self.assertTrue(restored["resume_blocked"])
        self.assertTrue((target / E.RUN / "STOP").exists())
        self.assertFalse((target / E.REFERENCE_FILES[0]).exists())
        for name in E.CANDIDATES:
            original = torch.load(self.root / name, weights_only=True)
            actual = torch.load(target / name, weights_only=True)
            self.assertEqual(E._state_digest(original), E._state_digest(actual))
            self.assertTrue(actual["optimizer_state_dict"]["state"])
        with self.assertRaisesRegex(FileExistsError, "Existing v8 run"):
            E.restore(result["archive"], target)

    def test_partial_export_keeps_complete_sources_without_claiming_candidates(self):
        shutil.rmtree(self.root / E.RUN / "fit")
        result = E.create_export(self.root)
        self.assertEqual(result["completed_candidate_count"], 0)
        self.assertEqual(E.verify(result["archive"])["completed_candidate_count"], 0)

    def test_each_actor_lock_blocks_export_and_stop_allows_it(self):
        for name in E.LOCKS:
            path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "Actor is active"):
                    E.create_export(self.root)
        self.assertTrue(E.create_export(self.root)["verified"])

    def test_orphan_actor_without_locks_blocks_export_and_restore(self):
        result = E.create_export(self.root)
        with mock.patch.object(E, "active_actor_pids", return_value=[12345]):
            with self.assertRaisesRegex(RuntimeError, "orphan engine"):
                E.create_export(self.root)
            with self.assertRaisesRegex(RuntimeError, "orphan engine"):
                E.restore(result["archive"], Path(self.temp.name) / "target")

    def test_v5_and_v6_orphan_commands_are_both_detected(self):
        fake_proc = Path(self.temp.name) / "proc"
        for number, arguments in enumerate((
                ["python", "-m", "alpha0.terminal_v5", "--worker", "job.json"],
                ["python", "-m", "alpha0.terminal_v8", "--worker", "job.json"],
                ["java", "alpha0.EngineBridge"],
                ["python", "-m", "pytest"]), start=900001):
            process = fake_proc / str(number); process.mkdir(parents=True)
            (process / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in arguments) + b"\0")
            (process / "cwd").symlink_to(self.root, target_is_directory=True)
        with mock.patch.object(E, "PROC", fake_proc):
            self.assertEqual(E.active_actor_pids(self.root), [900001, 900002, 900003])

    def test_matching_actor_with_unreadable_cwd_blocks_even_without_locks(self):
        for number, arguments in enumerate((
                ["python", "-m", "alpha0.terminal_v8", "--worker", "job.json"],
                ["java", "alpha0.EngineBridge"],
                ["python", "-m", "pytest"]), start=900011):
            process = self.fake_proc / str(number); process.mkdir()
            (process / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in arguments) + b"\0")
            (process / "cwd").symlink_to(self.root, target_is_directory=True)
        real_resolve = Path.resolve
        def resolve(path, *args, **kwargs):
            if path.parent.parent == self.fake_proc and path.name == "cwd":
                raise PermissionError("sandbox namespace hides cwd")
            return real_resolve(path, *args, **kwargs)
        with mock.patch.object(Path, "resolve", autospec=True, side_effect=resolve):
            self.assertEqual(E.active_actor_pids(self.root), [900011, 900012])
            with self.assertRaisesRegex(RuntimeError, "orphan engine"):
                with E.experiment_locks(self.root):
                    self.fail("an unattributable matching actor must block export")

    def test_all_seven_stops_are_preserved_by_quiescent_export(self):
        self.assertEqual(len(E.LOCKS), 7)
        self.assertEqual(len(E.STOPS), 7)
        for name in E.STOPS:
            write(self.root, name, "user stop remains\n")
        result = E.create_export(self.root)
        self.assertTrue(result["verified"])
        for name in E.STOPS:
            self.assertEqual((self.root / name).read_text(), "user stop remains\n")

    def test_partial_fit_is_kept_but_final_filename_cannot_claim_partial_completion(self):
        original = torch.load(self.root / E.CANDIDATES[0], weights_only=True)
        partial = {key: original[key] for key in ("optimizer_state_dict", "torch_rng", "numpy_rng", "batch_sha256", "arm")}
        partial.update(policy_state_dict={"weight": original["model_state_dict"]["policy.weight"],
                                         "bias": original["model_state_dict"]["policy.bias"]}, step=1, history=[{"step": 1}])
        pending = f"{E.RUN}/fit-state/terminal_head.pt"
        save(self.root, pending, partial)
        result = E.create_export(self.root)
        self.assertIn(pending, E.verify(result["archive"])["checkpoint_state_verified"])
        original["updates"] = 1
        save(self.root, E.CANDIDATES[0], original)
        manifest = self.root / E.RUN / "fit/manifest.json"
        value = json.loads(manifest.read_text())
        value["files"][Path(E.CANDIDATES[0]).name] = E.sha256(self.root / E.CANDIDATES[0])
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "2048 completed updates"):
            E.create_export(self.root)

    def test_source_cache_hash_and_frozen_candidate_tensors_are_checked(self):
        path = self.root / E.RUN / "sources/training-m0-j1.npz"
        before = path.read_bytes(); np.savez(path, hidden=np.zeros((2, 4)))
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            E.plan(self.root)
        path.write_bytes(before)
        checkpoint = self.root / E.CANDIDATES[0]
        payload = torch.load(checkpoint, weights_only=True)
        payload["model_state_dict"]["encoder.bias"] += 1
        torch.save(payload, checkpoint)
        manifest = self.root / E.RUN / "fit/manifest.json"
        value = json.loads(manifest.read_text()); value["files"][checkpoint.name] = E.sha256(checkpoint)
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "changed frozen tensor"):
            E.plan(self.root)

    def test_missing_optimizer_or_rng_is_not_a_complete_candidate(self):
        checkpoint = self.root / E.CANDIDATES[0]
        payload = torch.load(checkpoint, weights_only=True)
        del payload["numpy_rng"]
        torch.save(payload, checkpoint)
        manifest = self.root / E.RUN / "fit/manifest.json"
        value = json.loads(manifest.read_text()); value["files"][checkpoint.name] = E.sha256(checkpoint)
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "complete model/optimizer/RNG"):
            E.create_export(self.root)

    def test_v5_candidate_cannot_be_published_as_v6(self):
        checkpoint = self.root / E.CANDIDATES[0]
        payload = torch.load(checkpoint, weights_only=True)
        payload["experiment_id"] = "terminal-target-head-v5"
        torch.save(payload, checkpoint)
        manifest = self.root / E.RUN / "fit/manifest.json"
        value = json.loads(manifest.read_text()); value["files"][checkpoint.name] = E.sha256(checkpoint)
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "complete model/optimizer/RNG"):
            E.create_export(self.root)

    def test_completed_transcript_hash_or_outcome_cannot_be_silently_exported(self):
        source = self.root / E.RUN / "sources/training-m0-j1.json"
        row = json.loads(source.read_text()); row["actions"].append(2)
        source.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "terminal/action proof"):
            E.create_export(self.root)

    def test_gradient_benchmark_mixing_and_symlinks_fail_closed(self):
        path = write(self.root, f"{E.RUN}/dataset/bad.json", '{"source":"benchmark"}')
        with self.assertRaisesRegex(ValueError, "mixed into gradient"):
            E.plan(self.root)
        path.unlink()
        path.symlink_to(self.root / E.RUN / "identity.json")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            E.plan(self.root)

    def test_progress_conflicts_abort_before_any_state_is_copied(self):
        result = E.create_export(self.root)
        target = Path(self.temp.name) / "target"
        conflict = write(target, f"{E.PROGRESS}/roots-validation.json", '{"newer":true}')
        with self.assertRaisesRegex(FileExistsError, "conflicts with existing evidence"):
            E.restore(result["archive"], target)
        self.assertEqual(conflict.read_text(), '{"newer":true}')
        self.assertFalse((target / E.CANDIDATES[0]).exists())

    def test_identical_existing_progress_is_preserved_and_new_main_code_is_untouched(self):
        result = E.create_export(self.root)
        target = Path(self.temp.name) / "target"
        name = f"{E.PROGRESS}/roots-validation.json"
        write(target, name, (self.root / name).read_bytes())
        current = write(target, E.REFERENCE_FILES[0], '{"current_main":true}')
        E.restore(result["archive"], target)
        self.assertEqual(current.read_text(), '{"current_main":true}')

    def test_archive_checksum_and_unsafe_tar_members_are_rejected(self):
        result = E.create_export(self.root)
        archive = Path(result["archive"])
        archive.write_bytes(archive.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "archive SHA256"):
            E.verify(archive)
        malicious = Path(self.temp.name) / "unsafe.tar.gz"
        with tarfile.open(malicious, "w:gz") as bundle:
            for name, data in ((E.MANIFEST, b'{}'), ("../escape", b"bad")):
                info = tarfile.TarInfo(name); info.size = len(data)
                bundle.addfile(info, io.BytesIO(data))
        with self.assertRaisesRegex(ValueError, "Unsafe recovery path"):
            E.verify(malicious, expected_sha256=E.sha256(malicious))

    def test_original_prior_is_a_pinned_external_reference_not_foreign_payload(self):
        result = E.plan(self.root)
        self.assertEqual(result["dependency"]["initialization_ancestor"]["artifact_sha256"], "a" * 64)
        self.assertTrue(result["dependency"]["cp2_included"])

    def test_historical_prerequisite_snapshot_does_not_import_v5_source_games(self):
        name = f"{E.PROGRESS}/source/progress/terminal-target-head-v5/independent-coverage-audit.json"
        reference = {"source_json": "runs/terminal-target-head-v5/sources/training-m0-j1.json",
                     "source_sha256": "f" * 64,
                     "cache_npz": "runs/terminal-target-head-v5/sources/training-m0-j1.npz",
                     "cache_sha256": "e" * 64}
        write(self.root, name, json.dumps(reference))
        comparison = f"{E.PROGRESS}/source-coverage-comparison.json"
        write(self.root, comparison, json.dumps({"v5": reference}))
        result = E.create_export(self.root)
        with tarfile.open(result["archive"]) as archive:
            self.assertIn(name, archive.getnames())
            self.assertIn(comparison, archive.getnames())
            self.assertFalse(any(path.startswith("runs/terminal-target-head-v5/") for path in archive.getnames()))


if __name__ == "__main__":
    unittest.main()
