"""Offline tests for preregistered terminal-root ranking and owned cleanup.

Pure fixtures and short Python subprocesses only: no engine, model inference,
benchmark games, gradient updates, or production artifact writes.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from alpha0 import root_ranking_v4 as R

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return json.loads((ROOT / "configs/root-ranking-v4.json").read_text())


def selection_cells(scores, seeds):
    return [{"action": action, "seed": seed, "status": "ok", "score": score}
            for action, values in scores.items() for seed, score in zip(seeds, values)]


def gate_fixture(deltas):
    """Construct four-seed validation differences in increments of 1/8."""
    config = configuration()
    roots, selected, compared, cells = [], {}, {}, []
    for index, delta in enumerate(deltas):
        root_id = f"root-{index:02d}"
        roots.append({"root_id": root_id, "legal_actions": [4, 21]})
        selected[root_id] = {"complete": True, "selected_action": 4}
        compared[root_id] = {"selected_action": 21}
        # Baseline score .5 leaves room for every tested positive/negative delta.
        steps = int(round(abs(delta) * 8))
        assert abs(steps/8-abs(delta)) < 1e-12 and steps <= 4
        challenger = [.5] * 4
        for i in range(steps):
            challenger[i] = 1 if delta > 0 else 0
        for action, scores in ((4, challenger), (21, [.5]*4)):
            for seed, score in zip(config["validation_seeds"], scores):
                cells.append({"root_id": root_id, "stage": "validation", "action": action,
                              "seed": seed, "status": "ok", "score": score})
    return roots, selected, compared, cells, config


class ConfigTests(unittest.TestCase):
    def test_default_and_disjoint_reserved_seeds(self):
        R.validate_config(configuration(), reserved_seeds=(90101, 90107, 90121, 90133))

    def test_seed_sets_are_complete_unique_and_disjoint(self):
        base = configuration()
        bad = []
        for key, values in (("selection_seeds", [103003]), ("selection_seeds", [103003, 103003]),
                            ("validation_seeds", [103021, 103043, 103067]),
                            ("validation_seeds", [103003, 103043, 103067, 103091]),
                            ("validation_seeds", [103021, 103021, 103067, 103091])):
            candidate = copy.deepcopy(base)
            candidate[key] = values
            bad.append(candidate)
        for candidate in bad:
            with self.subTest(config=candidate), self.assertRaises(ValueError):
                R.validate_config(candidate)
        for reserved in (base["selection_seeds"][:1], base["validation_seeds"][:1]):
            with self.assertRaises(ValueError):
                R.validate_config(base, reserved_seeds=reserved)

    def test_execution_and_no_training_contract(self):
        for key, value in (("wall_budget_seconds", 7201), ("wall_budget_seconds", 0),
                           ("workers", 2), ("torch_threads", 2), ("gradient_updates", 1),
                           ("benchmark_data_used", True), ("temperature", float("nan"))):
            config = configuration()
            config[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                R.validate_config(config)


class SelectionTests(unittest.TestCase):
    def test_all_legal_actions_and_seeds_required_before_ranking(self):
        seeds = configuration()["selection_seeds"]
        legal, priors = [4, 21, 27], {4: .7, 21: .2, 27: .1}
        cells = selection_cells({4: [1, 1], 21: [0, 0], 27: [0, 0]}, seeds)
        partial = R.choose_action(legal, priors, cells[:-1], seeds)
        self.assertFalse(partial["complete"])
        self.assertIsNone(partial["selected_action"])
        self.assertEqual(partial["expected"], 6)
        self.assertEqual(partial["completed"], 5)
        for status in ("incomplete", "error"):
            failed = cells[:-1] + [{**cells[-1], "status": status, "score": None}]
            self.assertFalse(R.choose_action(legal, priors, failed, seeds)["complete"])
        complete = R.choose_action(legal, priors, cells, seeds)
        self.assertTrue(complete["complete"])
        self.assertEqual(complete["selected_action"], 4)
        self.assertEqual((complete["completed"], complete["expected"]), (6, 6))

    def test_mean_score_then_prior_then_lowest_action_break_ties(self):
        seeds = configuration()["selection_seeds"]
        legal = [27, 21, 4]  # Legal enumeration order must not alter prior ties.
        priors = {4: .8, 21: .1, 27: .1}
        cells = selection_cells({4: [.5, .5], 21: [1, .5], 27: [0, 1]}, seeds)
        self.assertEqual(R.choose_action(legal, priors, cells, seeds)["selected_action"], 21)
        tied = selection_cells({4: [1, 0], 21: [.5, .5], 27: [0, 1]}, seeds)
        self.assertEqual(R.choose_action(legal, priors, tied, seeds)["selected_action"], 4)
        equal_priors = {4: 1/3, 21: 1/3, 27: 1/3}
        self.assertEqual(R.choose_action(legal, equal_priors, tied, seeds)["selected_action"], 4)

    def test_duplicate_success_is_not_extra_evidence_and_conflict_fails(self):
        seeds = configuration()["selection_seeds"]
        cells = selection_cells({4: [0, 1], 21: [.5, .5]}, seeds)
        result = R.choose_action([4, 21], {4: .9, 21: .1}, cells + [dict(cells[1])], seeds)
        self.assertEqual(result["completed"], 4)
        self.assertEqual(result["selected_action"], 4)
        with self.assertRaises(ValueError):
            R.choose_action([4, 21], {4: .9, 21: .1}, cells + [{**cells[0], "score": 1}], seeds)


class DecisionTests(unittest.TestCase):
    def test_exact_mean_boundary_and_two_positive_roots_pass(self):
        result = R.summarize_validation(*gate_fixture([.25, .25, 0, 0, 0]))
        self.assertTrue(result["complete"])
        self.assertTrue(result["passes"])
        self.assertAlmostEqual(result["mean_delta"], .10)
        self.assertEqual(result["improved_roots"], 2)
        self.assertEqual(result["worst_root_delta"], 0)

    def test_positive_root_count_and_worst_root_floor_are_independent(self):
        one_positive = R.summarize_validation(*gate_fixture([.5, 0, 0, 0, 0]))
        self.assertGreaterEqual(one_positive["mean_delta"], .10-1e-12)
        self.assertFalse(one_positive["passes"])
        at_floor = R.summarize_validation(*gate_fixture([-.25, .5, .5]))
        self.assertTrue(at_floor["passes"])
        below_floor = R.summarize_validation(*gate_fixture([-.375, .5, .5]))
        self.assertGreater(below_floor["mean_delta"], .10)
        self.assertEqual(below_floor["improved_roots"], 2)
        self.assertFalse(below_floor["passes"])

    def test_incomplete_roots_and_selection_cells_cannot_pass(self):
        roots, selections, comparators, cells, config = gate_fixture([.5, .5])
        for incomplete in (cells[:-1], [{**cell, "stage": "selection"} for cell in cells]):
            result = R.summarize_validation(roots, selections, comparators, incomplete, config)
            self.assertFalse(result["complete"])
            self.assertFalse(result["passes"])
            self.assertIsNone(result["mean_delta"])
        selections[roots[0]["root_id"]] = {"complete": False, "selected_action": None}
        result = R.summarize_validation(roots, selections, comparators, cells, config)
        self.assertFalse(result["complete"])
        self.assertFalse(result["passes"])

    def test_equal_selected_actions_share_validation_cells_and_have_zero_delta(self):
        roots, selections, comparators, cells, config = gate_fixture([.5, .5])
        for root in roots:
            comparators[root["root_id"]]["selected_action"] = 4
        shared = [cell for cell in cells if cell["action"] == 4]
        result = R.summarize_validation(roots, selections, comparators, shared, config)
        self.assertTrue(result["complete"])
        self.assertFalse(result["passes"])
        self.assertEqual(result["mean_delta"], 0)
        self.assertEqual(result["improved_roots"], 0)


class PersistenceTests(unittest.TestCase):
    def test_immutable_json_allows_only_identical_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cell.json"
            R.immutable_json(path, {"status": "ok", "score": 1})
            original = path.read_bytes()
            R.immutable_json(path, {"score": 1, "status": "ok"})
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(ValueError):
                R.immutable_json(path, {"status": "ok", "score": 0})
            self.assertEqual(path.read_bytes(), original)

    def test_verified_success_survives_later_failure_and_cannot_be_reselected(self):
        identity = "a" * 64
        success = {"identity": identity, "status": "ok", "score": .5, "terminal": True,
                   "valid_game": True, "benchmark_failures": [], "attempt": 1}
        failure = {"identity": identity, "status": "error", "score": None, "attempt": 2}
        chosen = R.successful_cell([success, failure], identity)
        self.assertEqual(chosen["score"], .5)
        self.assertTrue(chosen["terminal"])
        # Attempt logs enforce an even stronger contract than aggregate score
        # rows: a completed cell must never have a second successful attempt.
        with self.assertRaises(ValueError):
            R.successful_cell([success, dict(success)], identity)
        self.assertIsNone(R.successful_cell([failure], identity))
        with self.assertRaises(ValueError):
            R.successful_cell([success, {**success, "score": 1, "attempt": 2}], identity)

    def test_identity_invalid_scores_and_nonterminal_labels_fail_closed(self):
        identity = "a" * 64
        success = {"identity": identity, "status": "ok", "score": 1, "terminal": True,
                   "valid_game": True, "benchmark_failures": []}
        invalid = [{**success, "identity": "b" * 64}, {**success, "score": .3},
                   {**success, "score": None}, {**success, "score": float("nan")},
                   {**success, "terminal": False},
                   {"identity": identity, "status": "incomplete", "score": 1}]
        for record in invalid:
            with self.subTest(record=record), self.assertRaises(ValueError):
                R.successful_cell([record], identity)
        self.assertIsNone(R.successful_cell([{"identity": identity, "status": "incomplete", "score": None}], identity))

    def test_failed_engine_games_cannot_produce_terminal_labels(self):
        identity = "a" * 64
        success = {"identity": identity, "status": "ok", "score": 1, "terminal": True,
                   "valid_game": True, "benchmark_failures": []}
        invalid = [{**success, "valid_game": False},
                   {**success, "benchmark_failures": ["worker failure"]},
                   {key: value for key, value in success.items() if key != "valid_game"},
                   {key: value for key, value in success.items() if key != "benchmark_failures"}]
        for record in invalid:
            with self.subTest(record=record), self.assertRaisesRegex(ValueError, "failed engine game"):
                R.successful_cell([record], identity)


class SummaryProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.progress = self.directory / "progress"
        self.patch_root = patch.object(R, "ROOT", self.directory)
        self.patch_progress = patch.object(R, "PROGRESS", self.progress)
        self.patch_root.start()
        self.patch_progress.start()
        self.config = configuration()
        self.expected = {"config": self.config}
        self.identity = R.fingerprint(self.expected)
        self.roots = [{"root_id": f"root-{i:02d}", "ordinal": i, "round": 3, "legal_actions": [4, 21],
                       "state_sha256": f"state-{i}", "prefix_sha256": f"prefix-{i}", "player": i}
                      for i in (0, 1)]
        manifest = {"identity": self.identity, "roots": self.roots, "all_sources_scanned": True}
        R.immutable_json(self.progress / "roots.json", manifest)
        self.roots_hash = R.fingerprint(manifest)
        for root in self.roots:
            rid = root["root_id"]
            priors = [0.0] * 40
            priors[4], priors[21] = .8, .2
            common = {"identity": self.identity, "roots_sha256": self.roots_hash, "root_id": rid}
            R.immutable_json(self.progress / "comparators" / f"{rid}.json",
                             {**common, "selected_action": 21, "policy_prior": priors,
                              "visits": {"4": 0, "21": 32}, "search": self.config["search"],
                              "search_seed": self.config["search"]["root_seed_base"] + root["ordinal"]})
            selection = []
            for stage in ("selection", "validation"):
                for action in (4, 21):
                    for seed in self.config[f"{stage}_seeds"]:
                        spec = {"experiment_identity": self.identity, "roots_sha256": self.roots_hash,
                                "root_id": rid, "root_state_sha256": root["state_sha256"],
                                "root_prefix_sha256": root["prefix_sha256"], "root_player": root["player"], "stage": stage,
                                "action": action, "seed": seed, "temperature": self.config["temperature"]}
                        score = int(action == 4)
                        winner = root["player"] if score else 1-root["player"]
                        terminal = {"terminal": True, "round": 10, "winner": winner}
                        record = {**spec, "identity": R.fingerprint(spec), "attempt": 1, "status": "ok",
                                  "terminal": True, "valid_game": True, "benchmark_failures": [], "score": score,
                                  "winner": winner, "terminal_round": 10, "terminal_summary": terminal,
                                  "terminal_state_sha256": R.fingerprint(terminal)}
                        R.immutable_json(self.progress / "cells" / f"{stage}-{rid}-a{action:02d}-s{seed}" / "attempt-001.json", record)
                        if stage == "selection":
                            selection.append(record)
            chosen = R.choose_action(root["legal_actions"], priors, selection, self.config["selection_seeds"])
            R.immutable_json(self.progress / "selections" / f"{rid}.json", {**common, **chosen, "errors": 0})

    def tearDown(self):
        self.patch_progress.stop()
        self.patch_root.stop()
        self.temp.cleanup()

    def test_criteria_cannot_pass_until_final_integrity_is_verified(self):
        provisional = R.build_summary(self.expected, "running")
        self.assertTrue(provisional["decision"]["complete"])
        self.assertTrue(provisional["decision"]["criteria_pass"])
        self.assertFalse(provisional["decision"]["integrity_verified"])
        self.assertFalse(provisional["decision"]["passes"])
        final = R.build_summary(self.expected, "complete_review_required", integrity_verified=True)
        self.assertTrue(final["decision"]["passes"])
        self.assertEqual(final["decision"]["mean_delta"], 1)
        self.assertEqual(final["decision"]["improved_roots"], 2)

    def test_terminal_winner_must_match_root_player_score(self):
        seed = self.config["validation_seeds"][0]
        path = self.progress / "cells" / f"validation-root-01-a04-s{seed}" / "attempt-001.json"
        record = json.loads(path.read_text())
        record["winner"] = 0  # Root1 belongs to teamB: winnerA cannot score1.
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "winner.*score"):
            R.build_summary(self.expected, "running")

    def test_comparator_and_selection_must_match_frozen_provenance(self):
        for kind in ("comparators", "selections"):
            path = self.progress / kind / "root-00.json"
            original = path.read_bytes()
            record = json.loads(original)
            record["roots_sha256"] = "wrong-root-manifest"
            path.write_text(json.dumps(record))
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "provenance mismatch"):
                R.build_summary(self.expected, "running")
            path.write_bytes(original)


def process_state(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return None


class WatchdogTests(unittest.TestCase):
    def run_stoppable_tree(self, by_stop):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pid_file, stop = directory / "pids.json", directory / "STOP"
            code = ("import json,os,subprocess,sys,time; from pathlib import Path; "
                    "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                    f"Path({str(pid_file)!r}).write_text(json.dumps([os.getpid(),child.pid])); time.sleep(60)")
            started = time.monotonic()
            def tick():
                if by_stop and pid_file.exists():
                    stop.touch()
            try:
                reason = R.supervise_process([sys.executable, "-c", code],
                    deadline=started+(10 if by_stop else 1), stop_paths=[stop], on_tick=tick)
                self.assertIsInstance(reason, str)
                self.assertIn("stop" if by_stop else "deadline", reason.lower())
                self.assertLess(time.monotonic()-started, 7)
                self.assertTrue(pid_file.exists(), "test worker must start before its deadline")
                pids = json.loads(pid_file.read_text())
                for pid in pids:
                    self.assertIn(process_state(pid), (None, "Z"), "owned parent and child must be stopped")
            finally:
                if pid_file.exists():
                    parent, child = json.loads(pid_file.read_text())
                    try:
                        if process_state(child) not in (None, "Z") and os.getpgid(child) == parent:
                            os.killpg(parent, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_hard_deadline_stops_owned_child_process_group(self):
        self.run_stoppable_tree(by_stop=False)

    def test_stop_file_stops_owned_child_process_group(self):
        self.run_stoppable_tree(by_stop=True)

    def test_successful_worker_exits_without_stop_reason(self):
        result = R.supervise_process([sys.executable, "-c", "pass"],
                                    deadline=time.monotonic()+5, stop_paths=[])
        self.assertIsNone(result)

    def test_existing_stop_prevents_new_work(self):
        with tempfile.TemporaryDirectory() as directory:
            stop, marker = Path(directory) / "STOP", Path(directory) / "started"
            stop.touch()
            code = f"from pathlib import Path; Path({str(marker)!r}).touch()"
            with patch.object(R.subprocess, "Popen", side_effect=AssertionError("STOP must be checked before spawning")):
                result = R.supervise_process([sys.executable, "-c", code],
                                            deadline=time.monotonic()+5, stop_paths=[stop])
            self.assertIn("stop", result.lower())
            self.assertFalse(marker.exists())

    def test_sigterm_to_supervisor_cleans_worker_group_and_restores_handler(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "worker-pids.json"
            handler_file = Path(directory) / "handler.json"
            worker = ("import json,os,subprocess,sys,time; from pathlib import Path; "
                      "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                      f"Path({str(pid_file)!r}).write_text(json.dumps([os.getpid(),child.pid])); time.sleep(60)")
            supervisor_code = f"""
import json,signal,sys,time
from pathlib import Path
from alpha0 import root_ranking_v4 as R
before=signal.getsignal(signal.SIGTERM)
try:
    R.supervise_process([sys.executable,'-c',{worker!r}], deadline=time.monotonic()+30, stop_paths=[])
except R.Interrupted:
    Path({str(handler_file)!r}).write_text(json.dumps({{'restored': signal.getsignal(signal.SIGTERM)==before}}))
"""
            supervisor = subprocess.Popen([sys.executable, "-c", supervisor_code], cwd=ROOT,
                                          start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            try:
                ready_by = time.monotonic()+10
                while not pid_file.exists() and supervisor.poll() is None and time.monotonic() < ready_by:
                    time.sleep(.05)
                self.assertTrue(pid_file.exists(), "supervisor must start its isolated worker tree")
                supervisor.send_signal(signal.SIGTERM)
                _, stderr = supervisor.communicate(timeout=5)
                self.assertEqual(supervisor.returncode, 0, stderr)
                self.assertTrue(json.loads(handler_file.read_text())["restored"])
                for pid in json.loads(pid_file.read_text()):
                    self.assertIn(process_state(pid), (None, "Z"), "SIGTERM must leave no running worker child")
            finally:
                if supervisor.poll() is None:
                    supervisor.kill()
                    supervisor.wait(timeout=3)
                if supervisor.stderr:
                    supervisor.stderr.close()
                if pid_file.exists():
                    parent, child = json.loads(pid_file.read_text())
                    try:
                        if process_state(child) not in (None, "Z") and os.getpgid(child) == parent:
                            os.killpg(parent, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
