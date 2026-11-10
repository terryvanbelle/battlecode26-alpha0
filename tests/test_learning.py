"""Semantic checks for PUCT perspective, legality and terminal training labels."""
import importlib.util
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alpha0.mcts import SearchConfig, search
from alpha0.replay import Position, ReplayBuffer, label_game


@dataclass(frozen=True)
class State:
    key: str
    player: int = 0
    legal_actions: tuple = (0, 1)
    terminal: bool = False
    terminal_value_a: float = 0.0


class SearchTests(unittest.TestCase):
    def test_same_team_turn_does_not_negate_value(self):
        # A chooses either another A turn or a B turn. Both successor nodes can
        # choose an A win or B win, so only the consecutive A turn is winning.
        def transition(s, action):
            if s.key == "root":
                return State("choice", player=action)
            return State("terminal", terminal=True, legal_actions=(),
                         terminal_value_a=1.0 if action == 0 else -1.0)

        result = search(State("root"), lambda s: (np.array([0.5, 0.5]), 0.0),
                        transition, 2, config=SearchConfig(simulations=100))
        self.assertGreater(result.policy[0], 0.85)
        self.assertGreater(result.value, 0.7)
        self.assertEqual(sum(result.visits.values()), 100)
        self.assertAlmostEqual(float(result.policy.sum()), 1.0)

    def test_b_selects_b_win(self):
        result = search(State("root", player=1),
                        lambda s: (np.array([0.5, 0.5]), 0.0),
                        lambda s, a: State("end", terminal=True, legal_actions=(),
                                            terminal_value_a=1.0 if a == 0 else -1.0),
                        2, config=SearchConfig(simulations=64))
        self.assertGreater(result.policy[1], 0.9)
        self.assertGreater(result.value, 0.8)

    def test_leaf_value_is_converted_from_leaf_player(self):
        result = search(State("root", legal_actions=(0,)),
                        lambda s: (np.ones(1), 0.7),
                        lambda s, a: State("child", player=1, legal_actions=(0,)),
                        1, config=SearchConfig(simulations=1))
        self.assertAlmostEqual(result.value, -0.7)

    def test_mask_and_noise_preserve_only_legal_actions(self):
        result = search(State("root", legal_actions=(1, 3)),
                        lambda s: (np.ones(5), 0.0),
                        lambda s, a: State("end", terminal=True, legal_actions=()), 5,
                        config=SearchConfig(simulations=20),
                        rng=np.random.default_rng(17), add_root_noise=True)
        self.assertEqual(set(result.visits), {1, 3})
        self.assertTrue(np.all(result.policy[[0, 2, 4]] == 0))
        self.assertIn(result.select_action(np.random.default_rng(4), 0.01), (1, 3))

    def test_terminal_root_and_malformed_values_fail(self):
        with self.assertRaises(ValueError):
            search(State("end", terminal=True), lambda s: (np.ones(2), 0), lambda s, a: s, 2)
        with self.assertRaises(ValueError):
            search(State("root"), lambda s: (np.ones(2), float("nan")), lambda s, a: s, 2)


class ReplayTests(unittest.TestCase):
    def test_terminal_label_perspective_and_checkpoint(self):
        board = np.ones((3, 5, 5), dtype=np.float32)
        positions = [Position(board, np.zeros(2), np.array([0.2, 0.8]), np.ones(2, bool), p)
                     for p in (0, 0, 1)]
        examples = label_game(positions, -1.0)
        self.assertEqual([e.value for e in examples], [-1, -1, 1])
        board[:] = 0
        self.assertTrue(examples[0].board.any())
        replay = ReplayBuffer(capacity=2)
        replay.extend(examples)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.npz"
            replay.save(path)
            restored = ReplayBuffer.load(path)
            self.assertEqual(len(restored), 2)
            batch = restored.sample(3, np.random.default_rng(7))
            self.assertEqual(batch["board"].shape, (3, 3, 5, 5))
            self.assertTrue(np.isin(batch["value"], [-1, 1]).all())


class EncodingTests(unittest.TestCase):
    def test_relative_features_and_king_footprint(self):
        from alpha0.env import EngineState, encode
        raw = {"width": 20, "height": 20, "current_id": 1, "current_team": 1,
               "terminal": False, "round": 10, "cooperation": True,
               "legal_actions": [0, 1], "bank": [500, 1000],
               "robots": [{"id": 1, "team": 1, "type": "RAT_KING", "hp": 300,
                           "x": 10, "y": 10, "direction": 0},
                          {"id": 2, "team": 0, "type": "BABY_RAT", "hp": 80,
                           "x": 12, "y": 10, "direction": "WEST"}]}
        board, scalars = encode(EngineState(raw))
        self.assertEqual(board.shape, (24, 17, 17))
        self.assertEqual(scalars.shape, (32,))
        self.assertEqual(float(board[7].sum()), 9)  # actor is a 3x3 king
        self.assertEqual(float(board[10].sum()), 9)  # ally king despite team B
        self.assertEqual(float(board[9, 8, 10]), 1)  # team A is enemy baby
        self.assertAlmostEqual(float(board[15, 8, 8]), 0.5)
        self.assertAlmostEqual(float(scalars[14]), 0.4)  # own bank first
        self.assertAlmostEqual(float(scalars[15]), 0.2)

    def test_carried_robot_cannot_overwrite_carrier_features(self):
        from alpha0.env import EngineState, encode
        raw = {"width": 20, "height": 20, "current_id": 1, "current_team": 0,
               "terminal": False, "round": 10, "cooperation": False, "legal_actions": [0],
               "robots": [{"id": 1, "team": 0, "type": "BABY_RAT", "hp": 80,
                           "x": 10, "y": 10, "direction": 0, "cheese": 30, "parts": [[10, 10]]},
                          {"id": 2, "team": 1, "type": "BABY_RAT", "hp": 20,
                           "x": 10, "y": 10, "direction": 4, "cheese": 90,
                           "carried": True, "parts": [[10, 10]]}]}
        board, scalars = encode(EngineState(raw))
        self.assertEqual(float(board[8, 8, 8]), 1)
        self.assertEqual(float(board[9, 8, 8]), 0)
        self.assertAlmostEqual(float(board[13, 8, 8]), 0.8)
        self.assertAlmostEqual(float(board[18, 8, 8]), 0.3)
        self.assertEqual(float(board[20, 8, 8]), 1)
        self.assertAlmostEqual(float(scalars[23]), 0.01)  # carried enemy remains alive
        raw["robots"][1]["carried"] = False
        raw["robots"][1]["thrown"] = True
        board, _ = encode(EngineState(raw))
        self.assertEqual(float(board[9, 8, 8]), 0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
class NetworkTests(unittest.TestCase):
    def test_masked_training_updates_weights(self):
        import torch
        from alpha0.network import NetworkConfig, PolicyValueNet, policy_value_loss
        torch.set_num_threads(1)
        torch.manual_seed(8)
        net = PolicyValueNet(NetworkConfig(4, 3, 5, channels=8, blocks=1))
        board, scalars = torch.randn(2, 4, 17, 17), torch.randn(2, 3)
        mask = torch.tensor([[True, False, True, False, False]] * 2)
        logits, values = net(board, scalars, mask)
        self.assertTrue(torch.all(logits.softmax(-1)[:, [1, 3, 4]] == 0))
        self.assertTrue(torch.all(values.abs() <= 1))
        target_policy = torch.tensor([[0.3, 0, 0.7, 0, 0]] * 2)
        loss, metrics = policy_value_loss(logits, values, target_policy, torch.tensor([1.0, -1.0]))
        self.assertTrue(np.isfinite(metrics["loss"]))
        before = net.policy.weight.detach().clone()
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, net.policy.weight))
        with self.assertRaises(ValueError):
            net(board, scalars, torch.zeros_like(mask))


if __name__ == "__main__":
    unittest.main()
