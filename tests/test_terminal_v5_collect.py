"""Collector protocol tests with a deterministic fake engine; never launch actors."""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from alpha0 import terminal_v5_collect as P
from alpha0 import terminal_v5_common as C
from alpha0.env import EngineState
from alpha0.features_v2 import Observation


def fake_state(index, length=40, *, eligible=True, duplicate_tags=False):
    if index == length:
        return EngineState({"terminal": True, "winner": 1, "round": length+1, "current_team": 0,
                            "legal_actions": [], "index": index, "win_reason": "fixture"}, (), None, 1)
    role, player = [("BABY_RAT", 0), ("RAT_KING", 1), ("BABY_RAT", 1), ("RAT_KING", 0)][index % 4]
    return EngineState({"terminal": False, "winner": -1, "round": index+1, "current_team": player,
                        "current_id": index+10, "robots": [{"id": index+10, "type": role}],
                        "legal_actions": [0, 37] if eligible else [0], "index": index,
                        "input_tag": 777 if duplicate_tags and index in (0, 4) else index}, (), index+1, 1)


def fake_encode(model, state):
    arrays = {name: np.zeros(shape, dtype=np.uint8 if name == "mask" else np.float32)
              for name, shape in P.CACHE_SHAPES.items()}
    arrays["scalars"][0] = state.raw["input_tag"]
    arrays["hidden"][0] = state.raw["input_tag"]
    arrays["mask"][list(state.legal_actions)] = 1
    arrays["prior"][0] = .75 if 37 in state.legal_actions else 1
    if 37 in state.legal_actions:
        arrays["prior"][37] = .25
    arrays["input_hash"] = C.input_hash(Observation(arrays["local"], arrays["global_board"], arrays["scalars"]), arrays["mask"])
    return arrays


class FakeEnv:
    instances = []
    length, eligible, duplicate_tags, fail_transitions = 40, True, False, False

    def __init__(self, **kwargs):
        self._snapshots = set()
        self.closed = False
        self.response_timeout = kwargs["response_timeout"]
        self.instances.append(self)

    def state(self, index):
        state = fake_state(index, self.length, eligible=self.eligible, duplicate_tags=self.duplicate_tags)
        if state.snapshot is not None:
            self._snapshots.add(state.snapshot)
        return state

    def reset(self, map_name, seed):
        return self.state(0)

    def transition(self, state, action):
        if self.fail_transitions:
            raise RuntimeError("fixture engine failure")
        if action not in state.legal_actions:
            raise ValueError("fixture illegal action")
        return self.state(state.raw["index"]+1)

    def _request(self, request):
        assert request["cmd"] == "release"
        return {"ok": True}

    def close(self):
        self.closed = True


class CollectorCase(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.config = C.load_config()
        self.identity = {"test_fixture": True}
        for name, value in (("ROOT", self.directory), ("RUN", self.directory/"runs/v5"),
                            ("PROGRESS", self.directory/"progress/v5"), ("STOPS", ())):
            self.stack.enter_context(patch.object(C, name, value))
        self.stack.enter_context(patch.object(P, "_model", return_value=object()))
        self.stack.enter_context(patch.object(P, "encode_state", side_effect=fake_encode))
        self.stack.enter_context(patch.object(P, "make_evaluator", side_effect=lambda model: lambda state: (fake_encode(model, state)["prior"], 0.0)))
        self.stack.enter_context(patch.object(P, "EngineEnv", FakeEnv))
        for key, value in (("instances", []), ("length", 40), ("eligible", True),
                           ("duplicate_tags", False), ("fail_transitions", False)):
            self.stack.enter_context(patch.object(FakeEnv, key, value))

    def job(self, split="training", **changes):
        result = C.source_job(split, 0, 1, self.config)
        result["worker_task_path"] = "runs/v5/workers/0/task.json"
        if split == "validation":
            result["exclusion_map"] = {"unrelated-hash": ["training-root"]}
        return {**result, **changes}

    def source(self, job=None):
        return P.source_game(job or self.job(), self.config, self.identity, time.monotonic()+60)

    def root(self, source, key="BABY_RAT:0"):
        return {**source["root_candidates"][key], "root_id": "training-root-00", "root_ordinal": 0,
                "map": source["map"], "seed": source["seed"], "source_json": source["source_json"],
                "source_sha256": C.sha256(C.ROOT/source["source_json"]),
                "cache_npz": source["cache_npz"], "cache_sha256": source["cache_sha256"],
                "worker_task_path": self.job()["worker_task_path"]}

    def test_source_seed_inventory_and_rng_derivation_are_pinned(self):
        spec = P.source_spec(self.job(), self.config, self.identity)
        self.assertEqual(spec["seed"], 20261201)
        self.assertEqual(spec["policy_seeds"], [620002, 620003])
        val = P.source_spec(self.job("validation"), self.config, self.identity)
        self.assertEqual(val["seed"], 20262201)
        self.assertEqual(val["policy_seeds"], [640002, 640003])
        expected = int(C.fingerprint({"anchor_seed": 203151, "source_id": "training-m0-j1", "player": 0})[:16], 16)
        self.assertEqual(P.reservoir_seed(203151, "training-m0-j1", 0), expected)
        self.assertNotEqual(expected, P.reservoir_seed(203151, "training-m0-j1", 1))
        with self.assertRaisesRegex(ValueError, "seed/slot"):
            P.source_spec(self.job(seed=1), self.config, self.identity)
        with self.assertRaisesRegex(ValueError, "exclusion"):
            P.source_spec(self.job("validation", exclusion_map={}), self.config, self.identity)

    def test_reservoir_can_keep_early_and_late_items_with_one_draw_per_new_item(self):
        class FixedRng:
            def __init__(self): self.bounds = []
            def integers(self, n): self.bounds.append(n); return 0 if n == 17 else n-1
        rng = FixedRng()
        self.assertEqual([P.reservoir_index(rng, i) for i in range(1, 17)], list(range(16)))
        self.assertEqual(P.reservoir_index(rng, 17), 0)
        self.assertIsNone(P.reservoir_index(rng, 18))
        self.assertEqual(rng.bounds, [17, 18])

    def test_complete_source_cache_first_roots_team_reservoir_and_resume(self):
        source = self.source()
        self.assertTrue(source["complete"])
        self.assertEqual(source["source"], "selfplay")
        self.assertEqual(len(source["actions"]), 40)
        self.assertEqual(source["root_candidates"]["BABY_RAT:0"]["decision_index"], 0)
        self.assertEqual(source["root_candidates"]["RAT_KING:1"]["decision_index"], 1)
        self.assertEqual(source["unavailable_slots"], [])
        for team in (0, 1):
            anchors = source["anchor_candidates"][str(team)]
            self.assertEqual(len(anchors), 16)
            self.assertTrue(all(anchor["player"] == team for anchor in anchors))
            self.assertEqual(len({a["decision_index"] for a in anchors}), 16)
        cache_path = C.ROOT/source["cache_npz"]
        with np.load(cache_path, allow_pickle=False) as cache:
            for row in source["cache_positions"]:
                i = row["feature_index"]
                actual = C.input_hash(Observation(cache["local"][i], cache["global_board"][i], cache["scalars"][i]), cache["mask"][i])
                self.assertEqual(actual, row["input_hash"])
        before = len(FakeEnv.instances)
        self.assertEqual(self.source(), source)
        self.assertEqual(len(FakeEnv.instances), before, "verified complete source reuse cannot launch another engine")
        self.assertTrue(FakeEnv.instances[0].closed)

    def test_validation_skips_exact_duplicates_then_continues_same_slot(self):
        FakeEnv.duplicate_tags = True
        duplicate = fake_encode(None, fake_state(0, duplicate_tags=True))["input_hash"]
        source = self.source(self.job("validation", exclusion_map={duplicate: ["root:training-root-00", "anchor:7"]}))
        self.assertEqual(source["root_candidates"]["BABY_RAT:0"]["decision_index"], 8)
        self.assertEqual([r["decision_index"] for r in source["skipped_input_duplicates"]], [0, 4])
        self.assertEqual(source["anchor_candidates"], {"0": [], "1": []})
        self.assertTrue(source["complete"])

    def test_completed_validation_with_no_qualifying_root_is_unavailable_not_error(self):
        FakeEnv.eligible = False
        source = self.source(self.job("validation"))
        self.assertEqual(source["root_candidates"], {})
        self.assertEqual(len(source["unavailable_slots"]), 4)
        self.assertTrue(source["complete"])
        with np.load(C.ROOT/source["cache_npz"], allow_pickle=False) as cache:
            for key, shape in P.CACHE_SHAPES.items():
                self.assertEqual(cache[key].shape, (0, *shape))

    def test_terminal_source_attempt_resumes_after_final_index_publish_interruption(self):
        source = self.source()
        (C.ROOT/source["source_json"]).unlink()
        before = len(FakeEnv.instances)
        self.assertEqual(self.source(), source)
        self.assertEqual(len(FakeEnv.instances), before)

    def test_failed_source_is_retained_unscored_and_not_reserve_unavailability(self):
        FakeEnv.fail_transitions = True
        with self.assertRaisesRegex(RuntimeError, "engine failure"):
            self.source()
        self.assertFalse((C.RUN/"sources/training-m0-j1.json").exists())
        result = json.loads(next((C.RUN/"sources/training-m0-j1.attempts").glob("attempt-[0-9][0-9][0-9].json")).read_text())
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["complete"])
        self.assertFalse(result["terminal"])
        self.assertIsNone(result["score"])

    def test_root_reconstruction_checks_prefix_state_full_input_and_cache(self):
        source = self.source()
        root = self.root(source, "RAT_KING:1")
        worker = P.RootWorker(root, self.config, self.identity, time.monotonic()+60)
        self.addCleanup(worker.close)
        state = worker.restore()
        self.assertEqual(state.player, 1)
        self.assertEqual(state.raw["index"], 1)
        worker.close()
        root["input_hash"] = "wrong"
        with self.assertRaisesRegex(ValueError, "canonical model input"):
            worker.restore()

    def test_cell_proof_scores_root_team_and_deduplicates_success(self):
        source = self.source()
        root = self.root(source, "RAT_KING:1")
        worker = P.RootWorker(root, self.config, self.identity, time.monotonic()+60)
        self.addCleanup(worker.close)
        result = worker.cell("selection", 37, 203003)
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["actions"][0], 37)
        spec = P.cell_spec(root, "selection", 37, 203003, self.config, self.identity)
        self.assertEqual(P.successful_cell([result], spec), result)
        before = len(FakeEnv.instances)
        self.assertEqual(worker.cell("selection", 37, 203003), result)
        self.assertEqual(len(FakeEnv.instances), before)
        with self.assertRaisesRegex(ValueError, "duplicate successful"):
            P.successful_cell([result, result], spec)
        wrong = deepcopy(result)
        wrong["actions"][0] = 0
        wrong["actions_sha256"] = C.fingerprint(wrong["actions"])
        with self.assertRaisesRegex(ValueError, "declared first action"):
            P.successful_cell([wrong], spec)

    def test_validation_deduplicates_three_identical_actions_physically(self):
        source = self.source()
        root = self.root(source)
        root["actions"] = {"cp2": 37, "terminal_head": 37, "puct_head": 37}
        result = P.root_job(root, "validation", self.config, self.identity, time.monotonic()+60)
        self.assertTrue(result["complete"])
        self.assertEqual(result["expected"], 4)
        files = list((C.PROGRESS/"cells").glob("*/attempt-[0-9][0-9][0-9].json"))
        self.assertEqual(len(files), 4)
        self.assertTrue(all(json.loads(p.read_text())["score"] == 0 for p in files))

    def test_source_cache_corruption_cannot_be_reused(self):
        source = self.source()
        (C.ROOT/source["cache_npz"]).write_bytes(b"damaged fixture")
        with self.assertRaisesRegex(ValueError, "cache changed"):
            self.source()

    def test_expired_deadline_starts_no_engine_and_creates_no_success(self):
        with self.assertRaises(C.Interrupted):
            P.source_game(self.job(), self.config, self.identity, time.monotonic()-1)
        self.assertEqual(FakeEnv.instances, [])
        self.assertFalse((C.RUN/"sources/training-m0-j1.json").exists())

    def test_error_cell_attempts_are_bounded_and_never_receive_a_score(self):
        source = self.source()
        root = self.root(source)
        worker = P.RootWorker(root, self.config, self.identity, time.monotonic()+60)
        self.addCleanup(worker.close)
        FakeEnv.fail_transitions = True
        for _ in range(4):
            self.assertIsNone(worker.cell("selection", 37, 203003))
        files = list((C.PROGRESS/"cells").glob("*/attempt-[0-9][0-9][0-9].json"))
        self.assertEqual(len(files), 2)
        for path in files:
            row = json.loads(path.read_text())
            self.assertEqual(row["status"], "error")
            self.assertFalse(row["terminal"])
            self.assertIsNone(row["score"])


if __name__ == "__main__":
    unittest.main()
