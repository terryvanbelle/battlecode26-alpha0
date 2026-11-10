"""Offline durable-export checks; no actual run, engine or network is used."""
import fcntl
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from alpha0.learn_v2 import Learner

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("export_warmstart", ROOT / "scripts/export_warmstart.py")
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


def write(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode() if isinstance(content, str) else content)
    return path


def fixture(root):
    config = json.loads((ROOT / "configs/warmstart-v3.json").read_text())
    config.update(replay_capacity=4, replay_max_games=2, game_position_capacity=2, batch_size=1, games_per_iteration=1)
    config["model"].update(local_width=8, global_width=8, blocks=0, hidden_dim=16)
    prior = write(root, "artifacts/prior/warmstart.pt", b"synthetic-prior")
    config["initialization"]["artifact"] = prior.relative_to(root).as_posix()
    config["initialization"]["artifact_sha256"] = E.sha256(prior)
    for name in E.SOURCE_FILES:
        write(root, name, "source snapshot\n")
    write(root, "configs/warmstart-v3.json", json.dumps(config))
    source = write(root, "alpha0/env.py", "runtime source\n")
    bridge = write(root, "java/alpha0/EngineBridge.java", "project-owned bridge source\n")
    classes = write(root, "build/java/alpha0/EngineBridge.class", b"compiled class")
    engine = write(root, "vendor/engine/battlecode26-java-1.2.6.jar", b"downloaded engine jar")
    release = write(root, "vendor/jdk/release", "JAVA_VERSION=21\n")
    write(root, "vendor/jdk/bin/java", b"downloaded runtime executable")
    benchmark = write(root, "benchmarks/bots/example/RobotPlayer.java", "third-party bot source\n")
    write(root, "build/benchmarks/example/RobotPlayer.class", b"third-party bot class")
    write(root, "configs/benchmarks.json", json.dumps({"bots": [{"id": "example",
        "source_repository": "https://example.invalid/frozen-bot", "source_commit": "b" * 40,
        "files": [{"path": benchmark.relative_to(root).as_posix(), "sha256": E.sha256(benchmark)}]}]}))
    map_file = write(root, "maps/tiny.map26", b"synthetic-map")
    runtime = {"files": {"alpha0/env.py": E.sha256(source), "java/alpha0/EngineBridge.java": E.sha256(bridge),
                         "vendor/engine/battlecode26-java-1.2.6.jar": E.sha256(engine)},
               "compiled_classes": {"build/java/alpha0/EngineBridge.class": E.sha256(classes)},
               "jdk_release_sha256": E.sha256(release), "numpy_version": np.__version__,
               "torch_version": str(torch.__version__)}
    run, progress = root / E.RUN, root / E.PROGRESS
    learner = Learner(run, config, progress)
    marker = {"artifact_sha256": E.sha256(prior), "config_sha256": E.fingerprint(config),
              "checkpoint_zero_sha256": E.sha256(run / "checkpoint-0000.pt"),
              "source_dataset_sha256": config["initialization"]["source_dataset_sha256"]}
    write(root, f"{E.RUN}/initialization.json", json.dumps(marker))
    policy = np.zeros(40, np.float32)
    policy[0] = 1
    game_id = "iter-0001-game-000"
    learner.buffer.add_game(game_id, [{"local": np.zeros((33, 17, 17), np.float16),
        "global_board": np.zeros((33, 60, 60), np.float16), "scalars": np.zeros(64, np.float32),
        "policy": policy, "legal_mask": np.ones(40, bool), "player": 0}], 1, {"source": "selfplay"})
    learner.model.policy.weight.sum().backward()
    learner.optimizer.step()
    learner.rng.integers(1000)
    torch.rand(2)
    learner.iteration, learner.games_total, learner.positions_total = 1, 1, 1
    learner.checkpoint()
    replay_index = json.loads((run / "replay/iteration-0001/index.json").read_text())
    replay_file = run / "replay/iteration-0001" / replay_index["games"][0]["file"]
    consumed = write(root, f"{E.RUN}/games/{game_id}.npz", replay_file.read_bytes())
    job_identity = {"map_sha256": E.sha256(map_file)}
    row = {"game_id": game_id, "status": "ok", "data_sha256": E.sha256(consumed), "identity": job_identity}
    write(root, f"{E.RUN}/games/{game_id}.json", json.dumps(row))
    write(root, f"{E.RUN}/games/{game_id}.attempt-0001.json", json.dumps(row))
    plan = {"config_sha256": E.fingerprint(config), "training_runtime": runtime,
            "checkpoint_sha256": E.sha256(run / "checkpoint-0000.pt"),
            "jobs": [{"map": "tiny", "identity": job_identity, "game_id": game_id,
                      "output": str(run / "games" / f"{game_id}.json")}]}
    write(root, f"{E.RUN}/plan-0001.json", json.dumps(plan))
    write(root, f"{E.RUN}/status.json", json.dumps({"phase": "complete_review_required", "iteration": 1}))
    write(root, f"{E.RUN}/experiment.log", "launch and training transcript\n")
    for name in ("experiment.pid", "STOP", "latest.tmp", ".partial.json"):
        write(root, f"{E.RUN}/{name}", "do not restore")
    write(root, f"{E.PROGRESS}/evaluation_reports/report.json", '{"complete": true}')
    for iteration in (0, 1):
        report = {"checkpoint_sha256": E.sha256(run / f"checkpoint-{iteration:04d}.pt"),
                  "iteration": iteration, "suite": "smoke", "complete": True, "full_panel_complete": True,
                  "completed": 24, "unresolved_errors": 0, "opponents": {str(n): {} for n in range(12)},
                  "evaluation_config_sha256": "eval-runtime", "benchmark_suite_id": "frozen-panel"}
        write(root, f"{E.PROGRESS}/checkpoint-{iteration:04d}-smoke.json", json.dumps(report))
    write(root, f"{E.PROGRESS}/evaluation_cells/cell/.lock", "excluded")
    write(root, f"{E.PROGRESS}/training.jsonl", '{"iteration": 1}\n')
    write(root, f"{E.PROGRESS}/evaluation.png", b"plot")
    return config, replay_file


def export(root):
    with patch.object(E, "git_identity", return_value={"commit": "a" * 40, "worktree_dirty": True}):
        return E.create_export(root, runner_sha256=E.sha256(root / "scripts/run_warmstart_experiment.py"))


class WarmstartExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "source"
        self.root.mkdir()
        self.config, self.replay_file = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_archive_restores_optimizer_rng_replay_and_provenance(self):
        result = export(self.root)
        archive_path = Path(result["archive"])
        sidecar = json.loads(Path(result["manifest"]).read_text())
        self.assertEqual(E.sha256(archive_path), sidecar["archive_sha256"])
        original = torch.load(self.root / E.RUN / "latest.pt", weights_only=True)
        target = Path(self.temp.name) / "restored"
        with tarfile.open(archive_path) as archive:
            names = set(archive.getnames())
            manifest = json.load(archive.extractfile(E.MANIFEST))
            for entry in manifest["files"]:
                content = archive.extractfile(entry["path"]).read()
                self.assertEqual(len(content), entry["size"])
                self.assertEqual(E.hashlib.sha256(content).hexdigest(), entry["sha256"])
            archive.extractall(target, filter="data")
        for name in ("experiment.pid", "STOP", "latest.tmp", "experiment.lock", ".partial.json"):
            self.assertNotIn(f"{E.RUN}/{name}", names)
        self.assertNotIn(f"{E.PROGRESS}/evaluation_cells/cell/.lock", names)
        self.assertIn(f"{E.RUN}/experiment.log", names)
        self.assertIn("scripts/run_warmstart_experiment.py", names)
        self.assertEqual(manifest["resume"]["original_runner_sha256"], E.LAUNCH_RUNNER_SHA256)
        self.assertEqual(manifest["resume"]["exported_runner_sha256"],
                         E.sha256(self.root / "scripts/run_warmstart_experiment.py"))
        self.assertEqual(E.RUNNER_OPERATIONAL_AMENDMENTS[-1]["previous_sha256"], E.LAUNCH_RUNNER_SHA256)
        self.assertEqual(E.RUNNER_OPERATIONAL_AMENDMENTS[-1]["runner_sha256"], E.CURRENT_RUNNER_SHA256)
        self.assertIn("artifacts/prior/warmstart.pt", names)
        self.assertIn(self.replay_file.relative_to(self.root).as_posix(), names)
        self.assertNotIn(f"{E.RUN}/replay/iteration-0000/index.json", names)
        omitted = {entry["path"]: entry for entry in manifest["omitted_files"]}
        old_shard = f"{E.RUN}/games/iter-0001-game-000.npz"
        self.assertNotIn(old_shard, names)
        self.assertEqual(omitted[old_shard]["sha256"], E.sha256(self.root / old_shard))
        self.assertIn(f"{E.RUN}/games/iter-0001-game-000.json", names)
        self.assertEqual(E.sha256(target / E.RUN / "latest.pt"), E.sha256(self.root / E.RUN / "latest.pt"))
        resumed = Learner(target / E.RUN, self.config, target / E.PROGRESS)
        self.assertEqual(resumed.iteration, 1)
        self.assertEqual(resumed.buffer.game_ids, ("iter-0001-game-000",))
        self.assertEqual(resumed.rng.bit_generator.state, original["numpy_rng"])
        self.assertTrue(torch.equal(torch.get_rng_state(), original["torch_rng"]))
        restored_optimizer = resumed.optimizer.state_dict()
        self.assertEqual(restored_optimizer["param_groups"], original["optimizer_state_dict"]["param_groups"])
        for key, state in original["optimizer_state_dict"]["state"].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(restored_optimizer["state"][key][name], value))
        original_index = json.loads(self.replay_file.with_name("index.json").read_text())
        self.assertEqual(resumed.buffer._rng.bit_generator.state, original_index["rng_state"])
        self.assertFalse(manifest["omission_policy"]["historical_replay_rollback_available"])
        with self.assertRaises(FileExistsError):
            export(self.root)

    def test_external_runtime_payloads_are_excluded_but_pins_remain(self):
        result = export(self.root)
        with tarfile.open(result["archive"]) as archive:
            names = set(archive.getnames())
            manifest = json.load(archive.extractfile(E.MANIFEST))
            self.assertFalse(any(name.startswith(("benchmarks/bots/", "build/", "vendor/")) for name in names))
            self.assertFalse(any(entry["path"].startswith(("benchmarks/bots/", "build/", "vendor/"))
                                 for entry in manifest["files"]))
            pinned_benchmarks = json.load(archive.extractfile("configs/benchmarks.json"))
        self.assertEqual(pinned_benchmarks, json.loads((self.root / "configs/benchmarks.json").read_text()))
        runtime = manifest["resume"]["training_runtime"]
        for name in ("vendor/engine/battlecode26-java-1.2.6.jar",):
            self.assertEqual(runtime["files"][name], E.sha256(self.root / name))
        for name in ("build/java/alpha0/EngineBridge.class",):
            self.assertEqual(runtime["compiled_classes"][name], E.sha256(self.root / name))
        self.assertEqual(runtime["jdk_release_sha256"], E.sha256(self.root / "vendor/jdk/release"))
        self.assertIn("java/alpha0/EngineBridge.java", names)
        self.assertIn("scripts/setup_engine.py", names)
        self.assertIn("scripts/fetch_benchmarks.py", names)
        self.assertIn("external_dependencies", manifest)

    def test_active_runner_blocks_export_without_output(self):
        with (self.root / E.RUN / "experiment.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "runner is active"):
                export(self.root)
        self.assertFalse((self.root / "artifacts/warmstart-v3").exists())

    def test_corrupt_replay_and_checkpoint_mismatch_are_rejected(self):
        self.replay_file.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "replay checksum"):
            export(self.root)
        (self.root / E.RUN / "latest.pt").write_bytes(b"other checkpoint")
        with self.assertRaisesRegex(ValueError, "latest.pt"):
            export(self.root)

    def test_missing_transcript_and_stale_evaluation_are_rejected(self):
        transcript = self.root / E.RUN / "games/iter-0001-game-000.json"
        before = transcript.read_bytes()
        transcript.unlink()
        with self.assertRaises(FileNotFoundError):
            export(self.root)
        transcript.write_bytes(before)
        report = self.root / E.PROGRESS / "checkpoint-0001-smoke.json"
        data = json.loads(report.read_text())
        data["checkpoint_sha256"] = "0" * 64
        report.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "smoke report"):
            export(self.root)

    def test_runner_drift_symlink_and_unfinished_boundary_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "approved operational runner source"):
            E.create_export(self.root, runner_sha256="0" * 64)
        write(self.root, f"{E.RUN}/status.json", '{"phase":"paused","iteration":1}')
        with self.assertRaisesRegex(ValueError, "completed"):
            export(self.root)
        write(self.root, f"{E.RUN}/status.json", '{"phase":"complete_review_required","iteration":1}')
        (self.root / E.PROGRESS / "linked.json").symlink_to(self.root / E.RUN / "config.json")
        with self.assertRaisesRegex(ValueError, "symlink"):
            export(self.root)

    def test_changing_source_rejects_archive_and_cleans_temporary_output(self):
        original = E.sha256
        target = self.root / E.PROGRESS / "training.jsonl"
        called = False

        def mutate(path):
            nonlocal called
            digest = original(path)
            if Path(path) == target and not called:
                called = True
                target.write_text("changed during read\n")
            return digest

        with patch.object(E, "sha256", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                export(self.root)
        parent = self.root / "artifacts/warmstart-v3"
        self.assertFalse((parent / "iteration-001").exists())
        self.assertEqual(list(parent.glob(".warmstart-export-*")), [])


if __name__ == "__main__":
    unittest.main()
