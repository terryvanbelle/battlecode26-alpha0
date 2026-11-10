"""Replay semantics: equal game weight, finite capacity and durable publication."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from alpha0.replay_v2 import GameReplayBuffer


def position(tag=0, player=0):
    scalars = np.zeros(64, dtype=np.float32)
    scalars[0] = tag
    scalars[1] = player
    policy = np.zeros(40, dtype=np.float32)
    policy[2] = 1
    legal = np.zeros(40, dtype=bool)
    legal[2] = True
    return {"local": np.zeros((33, 17, 17), dtype=np.float32),
            "global_board": np.zeros((33, 60, 60), dtype=np.float32),
            "scalars": scalars, "policy": policy, "legal_mask": legal, "player": player}


class GameReplayTests(unittest.TestCase):
    def test_games_have_equal_sampling_weight_despite_length(self):
        replay = GameReplayBuffer()
        replay.add_game("short", [position(0)], 1, {"source": "selfplay"})
        replay.add_game("long", [position(1) for _ in range(19)], -1, {"source": "selfplay"})
        rng = np.random.default_rng(448)
        long_count = sum(int(replay.sample(1, rng)["scalars"][0, 0]) for _ in range(1000))
        self.assertGreater(long_count, 440)
        self.assertLess(long_count, 560)  # Position-weighted replay would give ~950.

    def test_whole_game_eviction_and_max_game_count(self):
        replay = GameReplayBuffer(capacity=7, max_positions_per_game=4)
        for game_id in ("a", "b"):
            replay.add_game(game_id, [position(), position()], 0, {"source": "selfplay"})
        replay.add_game("c", (position(i) for i in range(100)), 1, {"source": "selfplay"})
        self.assertEqual(replay.game_ids, ("b", "c"))
        self.assertEqual(len(replay), 6)
        replay = GameReplayBuffer(max_games=2)
        for game_id in ("a", "b", "c"):
            replay.add_game(game_id, [position()], 0, {"source": "selfplay"})
        self.assertEqual(replay.game_ids, ("b", "c"))
        self.assertEqual(replay.num_games, 2)

    def test_reservoir_uses_positions_from_entire_game(self):
        replay = GameReplayBuffer(max_positions_per_game=8, seed=123)
        retained = replay.add_game("long", (position(i) for i in range(100)), 1, {"source": "selfplay"})
        self.assertEqual(retained, 8)
        tags = replay.sample(128, np.random.default_rng(4))["scalars"][:, 0]
        self.assertGreater(tags.max(), 70)
        self.assertLess(tags.min(), 30)
        self.assertEqual(len(np.unique(tags)), 8)

    def test_value_sign_roundtrip_dtypes_and_rng_continuation(self):
        replay = GameReplayBuffer(capacity=8, max_positions_per_game=4, seed=2)
        inputs = [position(12, 0), position(13, 1)]
        replay.add_game("game", inputs, -1, {"source": "selfplay", "map": "tiny"})
        inputs[0]["scalars"][0] = 99  # Replay owns its arrays.
        with tempfile.TemporaryDirectory() as folder:
            replay.save(folder)
            restored = GameReplayBuffer.load(folder)
            self.assertEqual(restored.game_ids, replay.game_ids)
            batch = restored.sample(20, np.random.default_rng(2))
            self.assertEqual(set(batch), {"local", "global_board", "scalars", "policy", "legal_mask", "value"})
            self.assertEqual(batch["local"].shape, (20, 33, 17, 17))
            self.assertEqual(batch["global_board"].shape, (20, 33, 60, 60))
            self.assertEqual(batch["global_board"].dtype, np.float32)
            self.assertEqual(batch["legal_mask"].dtype, np.bool_)
            np.testing.assert_array_equal(batch["value"], np.where(batch["scalars"][:, 1] == 0, -1, 1))
            self.assertNotIn(99, batch["scalars"][:, 0])
            index = json.loads((Path(folder)/"index.json").read_text())
            self.assertEqual(index["games"][0]["metadata"]["map"], "tiny")
            with np.load(Path(folder)/index["games"][0]["file"], allow_pickle=False) as data:
                self.assertEqual(data["local"].dtype, np.float16)
                self.assertEqual(data["global_board"].dtype, np.float16)
                self.assertEqual(data["scalars"].dtype, np.float32)
            for target in (replay, restored):
                target.add_game("next", (position(i) for i in range(20)), 0, {"source": "selfplay"})
            np.testing.assert_array_equal(replay.sample(32, np.random.default_rng(5))["scalars"],
                                          restored.sample(32, np.random.default_rng(5))["scalars"])

    def test_partial_benchmark_duplicate_and_illegal_targets_rejected(self):
        replay = GameReplayBuffer()
        with self.assertRaises(ValueError):
            replay.add_game("partial", [position()], 1, {"source": "selfplay"}, completed=False)
        with self.assertRaises(ValueError):
            replay.add_game("benchmark", [position()], 1, {"source": "benchmark"})
        with self.assertRaises(ValueError):
            replay.add_game("ambiguous", [position()], 1, {})
        bad = position()
        bad["legal_mask"][2] = False
        with self.assertRaises(ValueError):
            replay.add_game("bad", [bad], 1, {"source": "selfplay"})
        self.assertEqual(len(replay), 0)
        replay.add_game("one", [position()], 1, {"source": "selfplay"})
        with self.assertRaises(ValueError):
            replay.add_game("one", [position()], -1, {"source": "selfplay"})
        self.assertEqual(len(replay), 1)

    def test_index_published_before_garbage_collection(self):
        replay = GameReplayBuffer(max_games=1)
        replay.add_game("old", [position()], 1, {"source": "selfplay"})
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            replay.save(directory)
            old_file = json.loads((directory/"index.json").read_text())["games"][0]["file"]
            replay.add_game("new", [position()], -1, {"source": "selfplay"})
            import os
            real_replace = os.replace
            def interrupted_replace(source, destination):
                if Path(destination).name == "index.json":
                    raise OSError("simulated interrupted index publication")
                real_replace(source, destination)
            with mock.patch("alpha0.replay_v2.os.replace", side_effect=interrupted_replace):
                with self.assertRaises(OSError):
                    replay.save(directory)
            self.assertTrue((directory/old_file).exists())
            self.assertEqual(GameReplayBuffer.load(directory).game_ids, ("old",))
            replay.save(directory)
            self.assertEqual(GameReplayBuffer.load(directory).game_ids, ("new",))
            self.assertFalse((directory/old_file).exists())
            self.assertEqual(len(list(directory.glob("game-*.npz"))), 1)

    def test_empty_checkpoint_and_checksum_validation(self):
        replay = GameReplayBuffer()
        with tempfile.TemporaryDirectory() as folder:
            replay.save(folder)
            self.assertEqual(len(GameReplayBuffer.load(folder)), 0)
            with self.assertRaises(ValueError):
                replay.sample(1, np.random.default_rng())
            replay.add_game("one", [position()], 1, {"source": "selfplay"})
            replay.save(folder)
            index = json.loads((Path(folder)/"index.json").read_text())
            with (Path(folder)/index["games"][0]["file"]).open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                GameReplayBuffer.load(folder)


if __name__ == "__main__":
    unittest.main()
