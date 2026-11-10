"""Synthetic protocol checks; no engine, real checkpoints or SGD are run."""
from contextlib import ExitStack
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from alpha0 import constraint_v7_common as C
from alpha0 import constraint_v7 as P


class CommonTests(unittest.TestCase):
    def test_exact_configuration_rejects_extra_fields_and_changed_constraints(self):
        config = C.load_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for key, value in (("updates", 2049), ("total_wall_budget_seconds", 901),
                               ("torch_threads", True), ("automatic_promotion", True), ("unregistered", True)):
                changed = {**config, key: value}
                path.write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "preregistration"):
                    C.load_config(path)
            changed = deepcopy(config)
            changed["dataset"]["path"] = "runs/terminal-target-head-v6/dataset/validation.npz"
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "preregistration"):
                C.load_config(path)

    def test_immutable_json_cannot_be_overwritten_and_nan_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            C.immutable_json(path, {"complete": False})
            C.immutable_json(path, {"complete": False})
            with self.assertRaisesRegex(ValueError, "immutable"):
                C.immutable_json(path, {"complete": True})
            with self.assertRaises(ValueError):
                C.atomic_json(path, {"value": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"complete": False})

    def test_each_of_five_stops_blocks_and_deadline_is_checked(self):
        self.assertEqual(len(C.STOPS), 5)
        with tempfile.TemporaryDirectory() as directory:
            stops = tuple(Path(directory)/str(i) for i in range(5))
            with patch.object(C, "STOPS", stops):
                for path in stops:
                    path.touch()
                    with self.subTest(path=path), self.assertRaisesRegex(C.Interrupted, "STOP"):
                        C.check_stop(time.monotonic()+30)
                    path.unlink()
                with self.assertRaisesRegex(C.Interrupted, "deadline"):
                    C.check_stop(time.monotonic()-1)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(C, "ROOT", self.root))
        self.stack.enter_context(patch.object(C, "RUN", self.root/"runs/v7"))
        self.stack.enter_context(patch.object(C, "PROGRESS", self.root/"progress/v7"))

    def process(self, pid, command, cwd):
        path = self.root / "proc" / str(pid)
        path.mkdir(parents=True)
        (path/"cmdline").write_bytes("\0".join(command).encode()+b"\0")
        if cwd is not None:
            (path/"cwd").symlink_to(cwd, target_is_directory=True)
        return path

    def test_actor_scan_keeps_matching_processes_with_missing_or_unreadable_cwd(self):
        self.process(123001, ["python", "-m", "alpha0.constraint_v7"], self.root)
        self.process(123002, ["java", "-cp", "fixture", "alpha0.EngineBridge"], None)
        other = self.root / "other"
        other.mkdir()
        self.process(123003, ["python", "-m", "alpha0.terminal_v6"], other)
        self.process(123004, ["python", "harmless.py"], None)
        actors = P.active_actors(self.root/"proc")
        self.assertEqual({a["pid"] for a in actors}, {123001, 123002})
        self.assertEqual(next(a for a in actors if a["pid"] == 123002)["reason"], "unreadable_cwd")
        original = Path.resolve
        def permission(path, *args, **kwargs):
            if str(path).endswith("123001/cwd"):
                raise PermissionError("fixture")
            return original(path, *args, **kwargs)
        with patch.object(Path, "resolve", permission):
            actors = P.active_actors(self.root/"proc")
        self.assertEqual(next(a for a in actors if a["pid"] == 123001)["reason"], "unreadable_cwd")

    def test_all_five_locks_are_held_and_released(self):
        locks = tuple(self.root / f"lock-{i}" for i in range(5))
        with patch.object(C, "LOCKS", locks):
            with P.hold_locks():
                for path in locks:
                    with path.open("a+") as other, self.assertRaises(BlockingIOError):
                        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for path in locks:
                with path.open("a+") as other:
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_partial_lock_failure_releases_preceding_locks(self):
        locks = tuple(self.root / f"lock-{i}" for i in range(5))
        with locks[2].open("a+") as blocker, patch.object(C, "LOCKS", locks):
            fcntl.flock(blocker, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                with P.hold_locks():
                    self.fail("all five locks must be acquired before entering")
            with locks[0].open("a+") as other:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_resumed_total_budget_charges_actual_pauses_and_orphan_reservations(self):
        directory = C.PROGRESS / "invocations"
        base = {"identity_sha256": "identity", "started_at": "2026-11-10T10:00:00+00:00", "reserved_wall_seconds": 900, "invocation": 1}
        C.immutable_json(directory/"001-start.json", base)
        C.immutable_json(directory/"001.json", {**base, "charged_wall_seconds": 125, "ended_at": "2026-11-10T10:02:05+00:00"})
        self.assertEqual(P.consumed_budget("identity"), 125)
        C.immutable_json(directory/"002-start.json", {**base, "reserved_wall_seconds": 775, "invocation": 2})
        self.assertEqual(P.consumed_budget("identity"), 900)
        with self.assertRaisesRegex(ValueError, "identity"):
            P.consumed_budget("changed")

    def test_budget_rejects_negative_or_nonfinite_charge(self):
        directory = C.PROGRESS / "invocations"
        base = {"identity_sha256": "id", "started_at": "2026-11-10T10:00:00+00:00", "reserved_wall_seconds": 900, "invocation": 1}
        C.immutable_json(directory/"001-start.json", base)
        C.immutable_json(directory/"001.json", {**base, "charged_wall_seconds": -1, "ended_at": "2026-11-10T10:00:00+00:00"})
        with self.assertRaisesRegex(ValueError, "budget accounting"):
            P.consumed_budget("id")

    def test_worker_environment_limits_all_numeric_thread_pools(self):
        env = P.worker_environment()
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertEqual(env[name], "1")

    def test_hard_deadline_kills_process_group_immediately(self):
        process = Mock(pid=12345)
        with patch.object(os, "killpg") as kill:
            P.kill_group(process, immediate=True)
        kill.assert_called_once_with(12345, signal.SIGKILL)
        process.wait.assert_called_once_with(timeout=3)

    def test_stop_before_worker_spawn_launches_nothing(self):
        with patch.object(C, "check_stop", side_effect=C.Interrupted("STOP")), patch.object(P.subprocess, "Popen") as spawn:
            with self.assertRaises(C.Interrupted):
                P.run_worker(self.root/"worker.json", time.monotonic()+30, lambda **kwargs: None)
            spawn.assert_not_called()

    def execution_patches(self, identity_sequence=None):
        config = C.load_config()
        identity = {"sources": {}, "config": config, "fixture": True}
        self.stack.enter_context(patch.object(C, "STOPS", ()))
        self.stack.enter_context(patch.object(C, "LOCKS", tuple(self.root/f"lock-{i}" for i in range(5))))
        self.stack.enter_context(patch.object(C, "identity", side_effect=identity_sequence, return_value=identity))
        self.stack.enter_context(patch.object(P, "active_actors", return_value=[]))
        self.stack.enter_context(patch.object(P.signal, "setitimer"))
        return identity

    def test_paused_invocation_retains_only_actual_charge_and_can_resume(self):
        identity = self.execution_patches()
        with patch.object(P, "run_worker", side_effect=C.Interrupted("STOP")):
            first = P.execute(10)
            second = P.execute(10)
        self.assertEqual((first["phase"], second["phase"]), ("paused", "paused"))
        self.assertEqual(second["invocation"], 2)
        self.assertFalse((C.PROGRESS/"final-verification.json").exists())
        self.assertAlmostEqual(P.consumed_budget(C.fingerprint(identity)),
                               first["charged_wall_seconds"]+second["charged_wall_seconds"])

    def test_success_is_published_only_after_completed_verification(self):
        self.execution_patches()
        with patch.object(P, "run_worker"), patch("alpha0.constraint_v7_fit.verify_completed", return_value={"complete": True}) as verify:
            result = P.execute(10)
        verify.assert_called_once()
        self.assertEqual(result["phase"], "complete_review_required")
        self.assertTrue(result["integrity_verified"])
        self.assertTrue(json.loads((C.PROGRESS/"final-verification.json").read_text())["complete"])
        self.assertFalse(result["promotion_allowed"])

    def test_identity_change_cannot_publish_completed_verification(self):
        config = C.load_config()
        original = {"sources": {}, "config": config, "fixture": True}
        changed = {**original, "fixture": "changed"}
        self.execution_patches([original, changed, changed])
        with patch.object(P, "run_worker"):
            result = P.execute(10)
        self.assertEqual(result["phase"], "invalidated")
        self.assertFalse(result["integrity_verified"])
        self.assertFalse((C.PROGRESS/"final-verification.json").exists())


if __name__ == "__main__":
    unittest.main()
