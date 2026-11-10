"""Checks for the optional own-rule bootstrap's data and activation boundaries."""
import json
import hashlib
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from alpha0.bootstrap_v2 import BootstrapConfig, ROOT, run, run_winner_only, teacher_policy, training_excluded_maps
from alpha0.env import EngineState
from alpha0.replay_v2 import GameReplayBuffer


def position(kind="BABY_RAT", cheese=0, hp=100, round_number=1, legal=None):
    return EngineState({"current_team": 0, "current_id": 1, "terminal": False,
        "action_version": "macro_v2", "round": round_number,
        "legal_actions": legal if legal is not None else [0, 17, 18, 19, 20, 21, 24, 26, 27, 39],
        "robots": [{"id": 1, "team": 0, "type": kind, "hp": hp, "cheese": cheese}]})


class BootstrapTests(unittest.TestCase):
    def test_teacher_styles_roles_and_legal_distribution(self):
        for style, baby_action, king_action in (("assault", 21, 27), ("forage", 17, 26), ("guard", 20, 26)):
            for state, expected in ((position(), baby_action), (position("RAT_KING", hp=600), king_action)):
                policy = teacher_policy(state, style)
                self.assertEqual(int(policy.argmax()), expected)
                self.assertAlmostEqual(float(policy.sum()), 1, places=6)
                self.assertTrue(np.isfinite(policy).all())
                illegal = np.ones(40, bool)
                illegal[list(state.legal_actions)] = False
                self.assertTrue(np.all(policy[illegal] == 0))

    def test_resource_survival_and_phase_priorities(self):
        self.assertEqual(int(teacher_policy(position(cheese=10), "assault").argmax()), 18)
        self.assertEqual(int(teacher_policy(position(hp=20), "assault").argmax()), 24)
        self.assertEqual(int(teacher_policy(position(round_number=250), "guard").argmax()), 21)

    def test_blocked_primary_falls_back_legally(self):
        policy = teacher_policy(position(legal=[0, 19]), "assault", epsilon=0)
        self.assertEqual(int(policy.argmax()), 19)
        np.testing.assert_array_equal(policy[[0, 19]], [0, 1])

    def test_existing_run_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            marker = output / "production-checkpoint"
            marker.write_bytes(b"preserve")
            with self.assertRaisesRegex(ValueError, "new/empty"):
                run(BootstrapConfig(), output)
            self.assertEqual(marker.read_bytes(), b"preserve")
            self.assertEqual(list(output.iterdir()), [marker])

    def test_reserved_holdout_rejected_before_generation(self):
        reserved = sorted(training_excluded_maps(json.loads((ROOT / "configs/victory_gate.json").read_text())))[0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "untouched"
            with self.assertRaisesRegex(ValueError, "holdout"):
                run(BootstrapConfig(maps=(reserved,)), output)
            self.assertFalse(output.exists())

    def test_versioned_training_exclusions_and_fail_closed_schema(self):
        old = {"schema_version": 1, "stages": {"holdout": {"maps": ["Stash", "Hike"]}}}
        new = {"schema_version": 2, "training_excluded_maps": ["Stash", "Hike"],
               "stages": {"standard": {"maps": ["tiny", "Stash", "Hike"]}, "confirmation": {}}}
        self.assertEqual(training_excluded_maps(old), training_excluded_maps(new))
        self.assertEqual(training_excluded_maps({**old, "training_excluded_maps": ["Hike", "Stash"]}),
                         training_excluded_maps(old))
        for invalid in ({}, {**new, "schema_version": 3}, {**new, "schema_version": True},
                        {**old, "training_excluded_maps": ["tiny"]},
                        {"schema_version": 1, "training_excluded_maps": ["Stash"]},
                        {"schema_version": 2, "stages": old["stages"]},
                        {**new, "training_excluded_maps": []}, {**new, "training_excluded_maps": "Stash"},
                        {**new, "training_excluded_maps": ["Stash", "Stash"]},
                        {**new, "training_excluded_maps": ["../Stash"]}):
            with self.subTest(gate=invalid), self.assertRaises(ValueError):
                training_excluded_maps(invalid)

    def test_teacher_targets_cannot_enter_search_replay(self):
        with self.assertRaisesRegex(ValueError, "source"):
            GameReplayBuffer().add_game("teacher", [], 1, {"source": "scripted_selfplay"})

    def test_winner_only_preserves_source_and_resets_value(self):
        # Two examples from one complete own-teacher game, one per team. The
        # losing policy target must never reach this policy-fitting variant.
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/"source", Path(directory)/"output"
            (source/"games").mkdir(parents=True)
            (source/"config.json").write_text(json.dumps({"maps": ["tiny"], "games": 1, "learning_rate": 0.0003}))
            local = np.zeros((2, 33, 17, 17), np.float16)
            global_board = np.zeros((2, 33, 60, 60), np.float16)
            local[:, 0] = 1
            global_board[:, 0, :20, :20] = 1
            local[:, 9, 8, 8] = 1
            global_board[:, 9, 10, 10] = 1
            policy = np.zeros((2, 40), np.float32)
            policy[0, 21], policy[1, 17] = 1, 1
            data_path = source/"games/teacher-0000.npz"
            np.savez_compressed(data_path, local=local, global_board=global_board,
                scalars=np.zeros((2, 64), np.float32), policy=policy,
                legal_mask=np.ones((2, 40), bool), player=np.array([0, 1], np.int8),
                value=np.array([1, -1], np.float32))
            digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
            (source/"games/teacher-0000.json").write_text(json.dumps({
                "source": "scripted_selfplay", "benchmark": False,
                "teacher_version": "own-rules-assault-forage-guard-v1", "status": "ok",
                "terminal": {"terminal": True, "winner": 0}, "winner": 0,
                "terminal_value_a": 1, "action_version": "macro_v2", "map": "tiny",
                "seed": 7, "game_id": "teacher-0000", "teacher_A": "assault", "teacher_B": "forage",
                "data_sha256": digest}))
            with redirect_stdout(io.StringIO()):
                result = run_winner_only(source, output, steps=1, batch_size=1, seed=7)
            self.assertEqual(result["source_positions"], 2)
            self.assertEqual(result["selected_positions"], 1)
            self.assertEqual(json.loads((output/"source_dataset.json").read_text())["games"][0]["selected_positions"], [0])
            checkpoint = torch.load(output/"warmstart.pt", map_location="cpu", weights_only=True)
            self.assertEqual(int(checkpoint["model_state_dict"]["value.2.weight"].count_nonzero()), 0)
            self.assertEqual(int(checkpoint["model_state_dict"]["value.2.bias"].count_nonzero()), 0)
            self.assertEqual(hashlib.sha256(data_path.read_bytes()).hexdigest(), digest)
            with data_path.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                run_winner_only(source, Path(directory)/"must-fail", steps=1, batch_size=1)


if __name__ == "__main__":
    unittest.main()
