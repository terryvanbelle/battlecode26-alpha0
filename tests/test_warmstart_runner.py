"""Protect model-only initialization, recoverable identity, and owned cleanup."""
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("warmstart_runner", ROOT/"scripts/run_warmstart_experiment.py")
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)


class InitializationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.run, self.progress = self.directory/"run", self.directory/"progress"
        self.config = json.loads((ROOT/"configs/warmstart-v3.json").read_text())

    def tearDown(self):
        self.temp.cleanup()

    def initialize(self):
        return W.initialize_learner(ROOT, self.run, self.progress, self.config)

    def test_fresh_import_and_resume_preserve_exact_baseline(self):
        learner = self.initialize()
        prior = torch.load(ROOT/self.config["initialization"]["artifact"], map_location="cpu", weights_only=True)
        self.assertTrue(all(torch.equal(value, prior["model_state_dict"][key]) for key, value in learner.model.state_dict().items()))
        self.assertEqual(len(learner.buffer), 0)
        self.assertEqual(learner.buffer.num_games, 0)
        self.assertEqual(len(learner.optimizer.state), 0)
        checkpoint = (self.run/"checkpoint-0000.pt").read_bytes()
        resumed = self.initialize()
        self.assertEqual((self.run/"checkpoint-0000.pt").read_bytes(), checkpoint)
        self.assertEqual(resumed.iteration, 0)
        self.assertEqual(len(resumed.buffer), 0)

    def test_recovers_unpublished_random_checkpoint_before_marker(self):
        W.Learner(self.run, self.config, self.progress)
        self.assertFalse((self.run/"initialization.json").exists())
        learner = self.initialize()
        prior = torch.load(ROOT/self.config["initialization"]["artifact"], map_location="cpu", weights_only=True)
        self.assertTrue(torch.equal(learner.model.policy.weight, prior["model_state_dict"]["policy.weight"]))
        self.assertTrue((self.run/"initialization.json").exists())

    def test_missing_latest_cannot_destroy_published_cp0(self):
        self.initialize()
        expected = (self.run/"checkpoint-0000.pt").read_bytes()
        (self.run/"latest.pt").unlink()
        with self.assertRaisesRegex(ValueError, "published warmstart checkpoint missing"):
            self.initialize()
        self.assertEqual((self.run/"checkpoint-0000.pt").read_bytes(), expected)

    def test_changed_artifact_and_comparison_identity_fail(self):
        self.config["initialization"]["artifact_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.initialize()
        self.assertFalse((self.run/"checkpoint-0000.pt").exists())
        with self.assertRaisesRegex(ValueError, "differs"):
            W.Experiment.check_comparison({"evaluation_config_sha256":"a", "benchmark_suite_id":"x"},
                                         {"evaluation_config_sha256":"b", "benchmark_suite_id":"x"})

    def test_operational_target_does_not_change_training_config(self):
        self.assertNotIn("target_iteration", self.config)
        first = W.Experiment(ROOT, copy.deepcopy(self.config), 2, target_iteration=1)
        later = W.Experiment(ROOT, copy.deepcopy(self.config), 2, target_iteration=2)
        self.assertEqual(first.config, later.config)
        self.assertEqual(later.target_iteration, 2)

    def test_second_process_cannot_acquire_run_lock(self):
        run = self.directory/"runs/warmstart-v3"
        run.mkdir(parents=True)
        with (run/"experiment.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = subprocess.run([sys.executable, str(ROOT/"scripts/run_warmstart_experiment.py"),
                                     "--root", str(self.directory), "--initialize-only"],
                                     capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("owns the lock", result.stderr)
        self.assertFalse((run/"latest.pt").exists())


class ProcessOwnershipTests(unittest.TestCase):
    def test_thread_spawned_new_session_descendant_is_cleaned(self):
        code = "import subprocess,threading,time,sys; t=threading.Thread(target=lambda: subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)); t.start(); t.join(); time.sleep(30)"
        process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
        tracked = {}
        try:
            end = time.monotonic()+5
            while len(tracked) < 2 and time.monotonic() < end:
                W.track_tree(process.pid, tracked)
                time.sleep(0.05)
            self.assertGreaterEqual(len(tracked), 2)
            W.stop_tree(process, tracked)
            self.assertIsNotNone(process.poll())
            # Orphan zombies can briefly remain until the executor reaps them.
            for pid, token in tracked.items():
                if W.pid_token(pid) == token:
                    state = Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()[0]
                    self.assertEqual(state, "Z")
        finally:
            if process.poll() is None:
                W.stop_tree(process, tracked)


if __name__ == "__main__":
    unittest.main()
