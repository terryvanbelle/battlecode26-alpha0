"""Independent v6 protocol proofs; no engine or model fitting is launched."""
import hashlib
from contextlib import ExitStack
import json
import os
import signal
import subprocess
import sys
import time
import struct
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from alpha0.terminal_v6_common import input_hash, load_config, source_job


class InputIdentityProof(unittest.TestCase):
    @staticmethod
    def reference(observation, mask):
        # Deliberately serialize independently of input_hash and fingerprint.
        fields = []
        chunks = []
        for name, values in zip(("local", "global_board", "scalars"), observation):
            values = np.asarray(values, dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError("invalid fixture")
            fields.append({"name": name, "shape": list(values.shape), "dtype": "<f4"})
            chunks.append(b"".join(struct.pack("<f", float(x) if x != 0 else 0.0)
                                   for x in values.flat))
        fields.append({"name": "legal_mask", "shape": [40], "dtype": "|u1"})
        chunks.append(bytes(int(x) for x in mask))
        header = json.dumps({"schema": "gradient-input-v1", "feature_version": 2,
                             "action_version": "macro_v2", "fields": fields},
                            sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(struct.pack("<Q", len(header)) + header + b"".join(chunks)).hexdigest()

    @staticmethod
    def fixture():
        local = np.zeros((33, 17, 17), dtype=np.float32)
        global_board = np.zeros((33, 60, 60), dtype=np.float32)
        scalars = np.arange(64, dtype=np.float32) / 17
        local[0, 1, 2], local[32, 16, 16] = -0.0, 1.25
        global_board[0, 17, 16], global_board[32, 59, 59] = -2.5, 5
        mask = np.zeros(40, dtype=np.uint8)
        mask[[0, 21, 37]] = 1
        return (local, global_board, scalars), mask

    def test_independent_byte_oracle_and_portable_storage(self):
        observation, mask = self.fixture()
        expected = self.reference(observation, mask)
        self.assertEqual(input_hash(observation, mask), expected)
        alternative = tuple(np.asfortranarray(x.astype(">f4")) for x in observation)
        alternative[0][0, 1, 2] = 0.0
        self.assertEqual(input_hash(alternative, mask.astype(bool)), expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.npz"
            np.savez_compressed(path, local=alternative[0], global_board=alternative[1],
                                scalars=alternative[2], legal_mask=mask)
            with np.load(path, allow_pickle=False) as stored:
                self.assertEqual(input_hash(tuple(stored[k] for k in ("local", "global_board", "scalars")),
                                            stored["legal_mask"]), expected)

    def test_actual_float32_precision_and_mask_are_part_of_identity(self):
        observation, mask = self.fixture()
        base = input_hash(observation, mask)
        converted = tuple(x.astype(np.float64) for x in observation)
        converted[0][32, 16, 16] = np.nextafter(1.25, np.inf)
        self.assertEqual(input_hash(converted, mask), base)
        converted[0][32, 16, 16] = np.nextafter(np.float32(1.25), np.float32(np.inf))
        self.assertNotEqual(input_hash(converted, mask), base)
        different_mask = mask.copy()
        different_mask[1] = 1
        self.assertNotEqual(input_hash(observation, different_mask), base)

    def test_invalid_feature_or_mask_cannot_enter_overlap_guard(self):
        observation, mask = self.fixture()
        for invalid in (float("nan"), float("inf"), -float("inf")):
            observation[0][0, 0, 0] = invalid
            with self.assertRaises(ValueError):
                input_hash(observation, mask)
        observation[0][0, 0, 0] = 0
        for invalid_mask in (np.zeros(40), np.full(40, .5), np.ones(39)):
            with self.assertRaises(ValueError):
                input_hash(observation, invalid_mask)


class SeedAllocationProof(unittest.TestCase):
    def test_all_source_and_control_streams_are_disjoint_and_balanced(self):
        config = load_config()
        groups = {}
        for split, maximum, original in (("training", 8, 4), ("validation", 6, 2)):
            sources = []
            policy = []
            counts = {}
            for map_index in range(4):
                for j in range(1, maximum + 1):
                    job = source_job(split, map_index, j, config)
                    sources.append(job["seed"])
                    policy.extend(job["policy_seeds"])
                    if j <= original:
                        for role, player in (("BABY_RAT", 0 if j % 2 else 1),
                                             ("RAT_KING", 1 if j % 2 else 0)):
                            key = (map_index, role, player)
                            counts[key] = counts.get(key, 0) + 1
            self.assertEqual(set(counts.values()), {2 if split == "training" else 1})
            self.assertEqual(sum(counts.values()), 32 if split == "training" else 16)
            groups[split + "_engine"] = sources
            groups[split + "_policy"] = policy
        groups.update(selection=config["selection_seeds"], validation=config["validation_seeds"],
                      anchor=[config["anchor_seed"]], optimizer=[config["fit"]["seed"]],
                      search=[config["search"]["root_seed_base"] + i for i in range(32)])
        root = Path(__file__).resolve().parents[1]
        gate = json.loads((root / "configs/victory_gate.json").read_text())
        v4 = json.loads((root / "progress/root-ranking-v4/config.json").read_text())
        roots = json.loads((root / "progress/root-ranking-v4/roots.json").read_text())
        old = set(v4["selection_seeds"] + v4["validation_seeds"])
        old.update(v4["search"]["root_seed_base"] + i for i in range(8))
        old.update(row["seed"] for row in roots["roots"])
        old.update(gate["stages"]["confirmation"]["seeds"])
        seen = set()
        for name, seeds in groups.items():
            with self.subTest(group=name):
                self.assertEqual(len(seeds), len(set(seeds)))
                self.assertFalse(set(seeds) & seen)
                self.assertFalse(set(seeds) & old)
                seen.update(seeds)


class TerminalEvidenceProof(unittest.TestCase):
    @staticmethod
    def record(player=0, winner=1):
        from alpha0 import terminal_v6_common as common
        spec = {"experiment_identity": "a" * 64, "root_id": "training-root-00",
                "root_player": player, "root_state_sha256": "b" * 64,
                "root_prefix_sha256": "c" * 64, "root_input_hash": "d" * 64,
                "source_sha256": "e" * 64, "stage": "selection", "action": 21,
                "seed": 203003, "temperature": .35}
        raw = {"terminal": True, "winner": winner, "round": 30, "valid_game": True,
               "benchmark_failures": [], "win_reason": "RAT_KING_DEATH", "bank": [0, 0],
               "kings": [0, 1], "transferred": [0, 0], "cat_damage": [0, 0]}
        actions = [21, 0]
        record = {**spec, "identity": common.fingerprint(spec), "status": "ok", "terminal": True,
                  "valid_game": True, "benchmark_failures": [], "winner": winner, "terminal_round": 30,
                  "terminal_state": raw, "terminal_state_sha256": common.fingerprint(raw),
                  "terminal_summary": {key: raw[key] for key in
                      ("terminal", "round", "winner", "win_reason", "bank", "kings", "transferred", "cat_damage")},
                  "actions": actions, "actions_sha256": common.fingerprint(actions), "decisions": len(actions),
                  "score": .5 if winner == -1 else float(player == winner)}
        return spec, record

    def test_player_relative_score_is_recomputed(self):
        from alpha0.terminal_v6_collect import successful_cell
        for player in (0, 1):
            for winner in (-1, 0, 1):
                spec, record = self.record(player, winner)
                self.assertIs(successful_cell([record], spec), record)
                record["score"] = 1 if record["score"] != 1 else 0
                with self.assertRaises(ValueError):
                    successful_cell([record], spec)

    def test_hash_consistent_wrong_action_or_invalid_world_is_rejected(self):
        from alpha0 import terminal_v6_common as common
        from alpha0.terminal_v6_collect import successful_cell
        spec, record = self.record()
        record["actions"][0] = 0
        record["actions_sha256"] = common.fingerprint(record["actions"])
        with self.assertRaises(ValueError):
            successful_cell([record], spec)
        spec, record = self.record()
        record["terminal_state"]["valid_game"] = False
        record["terminal_state_sha256"] = common.fingerprint(record["terminal_state"])
        with self.assertRaises(ValueError):
            successful_cell([record], spec)
        spec, record = self.record()
        record["terminal_summary"]["winner"] = 0
        with self.assertRaises(ValueError):
            successful_cell([record], spec)


class ReserveAssignmentProof(unittest.TestCase):
    def discover_fixture(self, exhaust=False):
        from alpha0 import terminal_v6 as runner
        config = load_config()
        def record(job):
            candidates = {}
            for index, (role, team) in enumerate((("BABY_RAT", 0), ("BABY_RAT", 1),
                                                  ("RAT_KING", 0), ("RAT_KING", 1))):
                candidates[f"{role}:{team}"] = {"decision_index": index, "player": team,
                    "role": role, "input_hash": f"{job['source_id']}-{index}"}
            if job["map_index"] == 0:
                if job["j"] == 1:
                    candidates.pop("BABY_RAT:0")
                    candidates.pop("RAT_KING:1")
                if job["j"] == 5:
                    candidates.pop("RAT_KING:1")
                if exhaust and job["j"] == 3:
                    candidates.pop("BABY_RAT:0")
            return {"root_candidates": candidates, "cache_npz": "fake.npz", "cache_sha256": "f" * 64}
        class MockSupervisor(runner.Supervisor):
            def run_jobs(self, jobs, workers=None):
                pass
            def source(self, job):
                return record(job)
        with tempfile.TemporaryDirectory() as directory:
            progress = Path(directory)
            supervisor = MockSupervisor(config, {}, float("inf"))
            with patch.object(runner.C, "PROGRESS", progress), patch.object(runner.C, "sha256", return_value="f"*64), \
                 patch.object(runner, "verified_source", side_effect=lambda job, _expected: record(job)):
                if exhaust:
                    with self.assertRaises(runner.ReviewRequired):
                        supervisor.discover("training")
                else:
                    supervisor.discover("training")
            return json.loads((progress / "roots-training.json").read_text())

    def test_reserves_only_fill_missing_slots_in_declared_order(self):
        result = self.discover_fixture()
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["roots"]), 32)
        self.assertEqual(len(result["sources"]), 18)
        self.assertEqual(result["distinct_root_source_games"], 17)
        assigned = [row for row in result["assignments"] if row["map_index"] == 0 and row["slot_group_j"] == 1]
        self.assertEqual([(row["source_j"], row["filled"]) for row in assigned],
                         [(1, []), (5, ["BABY_RAT:0"]), (7, ["RAT_KING:1"])])
        root_slots = [(row["source_id"], row["role"], row["player"]) for row in result["roots"][:2]]
        self.assertEqual(root_slots, [("training-m0-j5", "BABY_RAT", 0), ("training-m0-j7", "RAT_KING", 1)])

    def test_consumed_reserve_is_not_reassigned_and_denominator_does_not_shrink(self):
        result = self.discover_fixture(exhaust=True)
        self.assertFalse(result["complete"])
        self.assertEqual(len(result["roots"]), 31)
        self.assertEqual(result["unavailable_slots"], [{"map_index": 0, "j": 3, "role": "BABY_RAT", "player": 0}])
        group3 = [row["source_j"] for row in result["assignments"] if row["map_index"] == 0 and row["slot_group_j"] == 3]
        self.assertEqual(group3, [3])


class TargetBudgetBarrierProof(unittest.TestCase):
    def test_excessive_complete_root_plan_cannot_score_or_fit(self):
        from alpha0 import terminal_v6 as runner
        config = load_config()
        class MockSupervisor(runner.Supervisor):
            def discover(self, split, exclusions=None):
                if split != "training":
                    raise AssertionError("validation discovery passed the failed budget barrier")
                return {"roots": [{"legal_actions": list(range(40))} for _ in range(32)]}
            def run_jobs(self, jobs, workers=None):
                raise AssertionError("target scoring or fitting passed the failed budget barrier")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner.C, "PROGRESS", Path(directory)):
                with self.assertRaises(runner.ReviewRequired):
                    runner.execute(MockSupervisor(config, {}, float("inf")))
            frozen = json.loads((Path(directory) / "target-budget.json").read_text())
            self.assertEqual(frozen["selection_cells"], 2560)
            self.assertEqual(frozen["ceiling"], 1600)
            self.assertIs(frozen["within_ceiling"], False)


class SupervisorProcessProof(unittest.TestCase):
    # These subprocesses only sleep. No engine, game or candidate is launched.
    @staticmethod
    def live_group(group):
        result = []
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                fields = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
                if int(fields[2]) == group and fields[0] != "Z":
                    result.append(int(proc.name))
            except (OSError, IndexError, ValueError):
                continue
        return result

    def context(self, root, runner):
        stack = ExitStack()
        stack.enter_context(patch.object(runner.C, "ROOT", root))
        stack.enter_context(patch.object(runner.C, "RUN", root / "run"))
        stack.enter_context(patch.object(runner.C, "PROGRESS", root / "progress"))
        stack.enter_context(patch.object(runner.C, "STOPS", ()))
        return stack

    def assert_quiescent(self, spawned):
        for process in spawned:
            self.assertIsNotNone(process.poll())
            self.assertEqual(self.live_group(process.pid), [])

    @staticmethod
    def cleanup(spawned):
        for process in spawned:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)

    def test_global_deadline_cleans_all_three_real_child_process_groups(self):
        from alpha0 import terminal_v6 as runner
        real_popen = subprocess.Popen
        spawned = []
        child = ("import signal,subprocess,sys,time; "
                 "signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT}); "
                 "subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); time.sleep(60)")
        def launch(_command, **kwargs):
            process = real_popen([sys.executable, "-c", child], **kwargs)
            spawned.append(process)
            return process
        with tempfile.TemporaryDirectory() as directory:
            with self.context(Path(directory), runner), patch.object(runner.subprocess, "Popen", side_effect=launch):
                supervisor = runner.Supervisor(load_config(), {}, time.monotonic()+.4)
                try:
                    with self.assertRaises(runner.C.Interrupted):
                        supervisor.run_jobs([{"kind": "fake", "payload": n} for n in range(3)])
                    self.assertEqual(len(spawned), 3)
                    self.assertEqual(supervisor.active, {})
                    self.assert_quiescent(spawned)
                    timing = list((Path(directory) / "progress/stage-timing").glob("*.json"))
                    self.assertEqual(len(timing), 1)
                    self.assertIn("deadline", json.loads(timing[0].read_text())["interruption_or_error"])
                finally:
                    self.cleanup(spawned)

    def test_signal_during_spawn_registration_is_deferred_then_cleans_child(self):
        from alpha0 import terminal_v6 as runner
        real_popen = subprocess.Popen
        spawned = []
        inherited_block = []
        previous_handler = signal.getsignal(signal.SIGTERM)
        previous_mask = signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
        def interrupted(_number, _frame):
            raise runner.C.Interrupted("injected SIGTERM")
        def launch(_command, **kwargs):
            process = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
            spawned.append(process)
            inherited_block.append(signal.SIGTERM in signal.pthread_sigmask(signal.SIG_BLOCK, set()))
            os.kill(os.getpid(), signal.SIGTERM)
            return process
        signal.signal(signal.SIGTERM, interrupted)
        try:
            with tempfile.TemporaryDirectory() as directory:
                with self.context(Path(directory), runner), patch.object(runner.subprocess, "Popen", side_effect=launch):
                    supervisor = runner.Supervisor(load_config(), {}, time.monotonic()+30)
                    with self.assertRaises(runner.C.Interrupted):
                        supervisor.run_jobs([{"kind": "fake", "payload": 0}], workers=1)
                    self.assertEqual(inherited_block, [True])
                    self.assertEqual(supervisor.active, {})
                    self.assert_quiescent(spawned)
        finally:
            self.cleanup(spawned)
            signal.signal(signal.SIGTERM, previous_handler)
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    def test_new_missing_task_gap_does_not_use_old_process_start_time(self):
        from alpha0 import terminal_v6 as runner
        real_popen, original_read = subprocess.Popen, runner.read
        spawned = []
        reads = 0
        supervisor = None
        child = ("import signal,time; signal.pthread_sigmask(signal.SIG_UNBLOCK,"
                 "{signal.SIGTERM,signal.SIGINT}); time.sleep(.35)")
        def launch(_command, **kwargs):
            process = real_popen([sys.executable, "-c", child], **kwargs)
            spawned.append(process)
            return process
        def read_task(path):
            nonlocal reads
            if Path(path).name != "task.json":
                return original_read(path)
            reads += 1
            if reads == 1:
                next(iter(supervisor.active.values()))["started"] = time.monotonic()-1000
                return {"deadline_monotonic": time.monotonic()+30}
            raise FileNotFoundError(path)
        with tempfile.TemporaryDirectory() as directory:
            with self.context(Path(directory), runner), patch.object(runner.subprocess, "Popen", side_effect=launch), \
                 patch.object(runner, "read", side_effect=read_task):
                supervisor = runner.Supervisor(load_config(), {}, time.monotonic()+10)
                try:
                    supervisor.run_jobs([{"kind": "fake", "payload": 0}], workers=1)
                    self.assertGreaterEqual(reads, 2)
                    self.assertEqual(supervisor.job_count, 1)
                    self.assert_quiescent(spawned)
                finally:
                    self.cleanup(spawned)


class VersionedCoverageAmendmentProof(unittest.TestCase):
    def test_entire_source_reserve_inventory_is_paired_with_v5(self):
        from alpha0 import terminal_v5_common as previous
        current = load_config()
        before = json.loads((Path(__file__).resolve().parents[1] /
                             "progress/terminal-target-head-v5/config.json").read_text())
        self.assertEqual(current["source_temperature"], 1.0)
        self.assertEqual(current["temperature"], before["temperature"])
        for split, maximum in (("training", 8), ("validation", 6)):
            for map_index in range(4):
                for j in range(1, maximum+1):
                    self.assertEqual(source_job(split, map_index, j, current),
                                     previous.source_job(split, map_index, j, before))
        # Scientific hyperparameters other than the source behavior must match.
        for name, value in before.items():
            if name != "experiment_id":
                self.assertEqual(current[name], value, name)

    def test_source_and_continuation_specs_use_separate_fixed_temperatures(self):
        from alpha0.terminal_v6_collect import source_spec, cell_spec
        config = load_config()
        root = {"root_id": "training-000", "player": 0, "legal_actions": [0, 21, 37],
                "state_sha256": "b"*64, "prefix_sha256": "c"*64,
                "input_hash": "d"*64, "source_sha256": "e"*64}
        for split in ("training", "validation"):
            for map_index in range(4):
                job = source_job(split, map_index, 1, config)
                if split == "validation":
                    job["exclusion_map"] = {"a" * 64: [{"kind": "root", "id": "training-000"}]}
                self.assertEqual(source_spec(job, config, {})["temperature"], 1.0)
        for stage in ("selection", "validation"):
            for seed in config[stage + "_seeds"]:
                self.assertEqual(cell_spec(root, stage, 21, seed, config, {})["temperature"], .35)

    def test_all_four_predecessor_locks_and_stops_are_respected(self):
        from alpha0 import terminal_v6_common as common
        relative_locks = [str(p.relative_to(common.ROOT)) for p in common.LOCKS]
        relative_stops = [str(p.relative_to(common.ROOT)) for p in common.STOPS]
        self.assertEqual(relative_locks, ["runs/campaign/campaign.lock", "runs/warmstart-v3/experiment.lock",
                         "runs/terminal-target-head-v5/experiment.lock", "runs/terminal-target-head-v6/experiment.lock"])
        self.assertEqual(relative_stops, ["runs/campaign/STOP", "runs/warmstart-v3/STOP",
                         "runs/terminal-target-head-v5/STOP", "runs/terminal-target-head-v6/STOP"])
        with tempfile.TemporaryDirectory() as directory:
            stops = tuple(Path(directory) / f"stop-{i}" for i in range(4))
            with patch.object(common, "STOPS", stops):
                self.assertIsNone(common.stop_reason(float("inf")))
                for stopped in stops:
                    stopped.touch()
                    self.assertEqual(common.stop_reason(float("inf")), "STOP")
                    with self.assertRaises(common.Interrupted):
                        common.check_stop(float("inf"))
                    stopped.unlink()

    def test_v5_runtime_protocol_and_source_evidence_remain_byte_identical(self):
        root = Path(__file__).resolve().parents[1]
        previous = json.loads((root / "progress/terminal-target-head-v5/identity.json").read_text())
        audit = json.loads((root / "progress/terminal-target-head-v5/independent-coverage-audit.json").read_text())
        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()
        for name, expected in previous["sources"].items():
            self.assertEqual(sha(root / name), expected, name)
        self.assertEqual(sha(root / "configs/terminal-target-head-v5.json"), previous["config_file_sha256"])
        for name, expected in audit["input_sha256"].items():
            if (name.startswith("runs/terminal-target-head-v5/sources/") or
                    name in ("progress/terminal-target-head-v5/roots-training.json",
                             "progress/terminal-target-head-v5/identity.json",
                             "progress/terminal-target-head-v5/config.json")):
                self.assertEqual(sha(root / name), expected, name)
        self.assertEqual(sha(root / "progress/terminal-target-head-v5/independent-coverage-audit.json"),
                         "48b586e3bf2f1befe6c49a89de5d6d153b9506303e7c350a8e8baa4f559b4d1a")


class FinalizationAndLockProof(unittest.TestCase):
    def mock_main_context(self, stack, root, runner):
        import alpha0.terminal_v6_report as reporter
        config = load_config()
        locks = tuple(root / name for name in ("campaign.lock", "warmstart.lock", "v5.lock", "v6.lock"))
        for name, value in (("ROOT", root), ("RUN", root / "run"), ("PROGRESS", root / "progress"),
                            ("LOCKS", locks), ("STOPS", ())):
            stack.enter_context(patch.object(runner.C, name, value))
        stack.enter_context(patch.object(runner.C, "identity", return_value={"sources": {}, "config": config}))
        stack.enter_context(patch.object(runner, "active_actors", return_value=[]))
        stack.enter_context(patch.object(runner.torch, "set_num_threads"))
        stack.enter_context(patch.object(runner.torch, "set_num_interop_threads"))
        stack.enter_context(patch.object(sys, "argv", ["terminal_v6", "--seconds", "10"]))
        stack.enter_context(patch("builtins.print"))
        summary = {"experiment_id": "terminal-target-head-v6", "selection": {},
                   "decision": {"complete": False, "review_for_expansion": False}}
        stack.enter_context(patch.object(reporter, "report", return_value=summary))
        execution = stack.enter_context(patch.object(runner, "execute"))
        return locks, execution, reporter

    @staticmethod
    def signal_restore(stack):
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        timer = signal.getitimer(signal.ITIMER_REAL)
        def restore():
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            signal.setitimer(signal.ITIMER_REAL, *timer)
        stack.callback(restore)

    def test_each_predecessor_lock_prevents_main_from_starting_work(self):
        import fcntl
        from alpha0 import terminal_v6 as runner
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self.signal_restore(stack)
            locks, execution, _ = self.mock_main_context(stack, Path(directory), runner)
            for path in locks:
                with path.open("a+") as stream:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with self.assertRaises(BlockingIOError):
                        runner.main()
            execution.assert_not_called()
            self.assertFalse((Path(directory) / "progress/identity.json").exists())

    def test_plot_failure_cannot_hide_durable_final_status_or_invocation(self):
        from alpha0 import terminal_v6 as runner
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self.signal_restore(stack)
            _, execution, reporter = self.mock_main_context(stack, Path(directory), runner)
            stack.enter_context(patch.object(reporter, "write_report", side_effect=RuntimeError("simulated plot failure")))
            runner.main()
            execution.assert_called_once()
            status = json.loads((Path(directory) / "run/status.json").read_text())
            invocation = json.loads((Path(directory) / "progress/invocations/001.json").read_text())
            self.assertEqual(status, invocation)
            self.assertEqual(status["active_jobs"], [])
            self.assertEqual(status["plot_error"], "RuntimeError: simulated plot failure")
            self.assertTrue((Path(directory) / "progress/summary.json").exists())


if __name__ == "__main__":
    unittest.main()
