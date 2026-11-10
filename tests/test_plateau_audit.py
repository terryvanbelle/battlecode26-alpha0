"""Compact-recovery audit reads identical data and rejects damaged provenance."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


SPEC = importlib.util.spec_from_file_location("plateau_audit", Path(__file__).resolve().parents[1]
    / "diagnostics/warmstart-plateau/analyze_targets.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class PlateauAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        (self.run / "games").mkdir()
        replay = self.run / "replay/iteration-0002"
        replay.mkdir(parents=True)
        self.arrays = {key: np.arange(6, dtype=np.float32).reshape(2, 3)
                       for key in AUDIT.ARRAY_KEYS}
        self.arrays["player"] = np.array([0, 1], dtype=np.int64)
        self.original = self.run / "games/one.npz"
        np.savez_compressed(self.original, **self.arrays)
        packed = dict(self.arrays, player=self.arrays["player"].astype(np.int8),
                      value=np.array([1, -1], dtype=np.float32))
        self.replay = replay / "game-one.npz"
        np.savez_compressed(self.replay, **packed)
        row = dict(game_id="one", data_sha256=AUDIT.sha256(self.original), source="selfplay",
                   map="tiny", seed=42, terminal_value_a=1, retained_positions=2)
        self.path = self.original.with_suffix(".json")
        self.path.write_text(json.dumps(row))
        entry = dict(game_id="one", file=self.replay.name, sha256=AUDIT.sha256(self.replay),
                     metadata=dict(source="selfplay", map="tiny", seed=42),
                     terminal_value_a=1, positions=2, positions_seen=2)
        index = replay / "index.json"
        index.write_text(json.dumps(dict(games=[entry])))
        checkpoint = self.run / "checkpoint-0002.pt"
        torch.save(dict(replay_path="replay/iteration-0002"), checkpoint)
        self.pins = self.run / "inputs.json"
        self.pins.write_text(json.dumps(dict(checkpoint=checkpoint.name,
            checkpoint_sha256=AUDIT.sha256(checkpoint), replay_path="replay/iteration-0002",
            replay_index_sha256=AUDIT.sha256(index), games={"one": dict(
                json_sha256=AUDIT.sha256(self.path), original_npz_sha256=AUDIT.sha256(self.original),
                replay_file=self.replay.name, replay_sha256=AUDIT.sha256(self.replay),
                source_arrays=AUDIT.array_fingerprints(self.arrays))})))

    def test_missing_original_uses_exact_source_equivalent_replay(self):
        loader = AUDIT.AuditInputs(self.run, self.pins)
        _, original, original_provenance = loader.load(self.path)
        self.original.unlink()
        _, restored, provenance = loader.load(self.path)
        self.assertEqual(original_provenance["source"], "original_game_npz")
        self.assertEqual(provenance["source"], "checkpoint_replay")
        for key in AUDIT.ARRAY_KEYS:
            self.assertEqual(original[key].dtype, restored[key].dtype)
            np.testing.assert_array_equal(original[key], restored[key])

    def test_corrupt_present_original_does_not_silently_fallback(self):
        self.original.write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError, "original game shard checksum"):
            AUDIT.AuditInputs(self.run, self.pins).load(self.path)

    def test_corrupt_replay_rejected(self):
        self.original.unlink()
        self.replay.write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError, "replay shard checksum"):
            AUDIT.AuditInputs(self.run, self.pins).load(self.path)


if __name__ == "__main__":
    unittest.main()
