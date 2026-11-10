"""Meaningful v2 observation, value-gradient, and checkpoint compatibility tests."""
import copy
import unittest

import numpy as np
import torch

from alpha0.env import EngineState
from alpha0.features_v2 import encode, CHANNEL_NAMES, SCALAR_NAMES, FEATURE_VERSION
from alpha0.network_v2 import NetworkConfig, PolicyValueNetV2, policy_value_loss


def fixture():
    raw = {"width": 60, "height": 30, "current_id": 1, "current_team": 0,
           "action_version": "macro_v2", "action_dim": 40,
           "terminal": False, "round": 10, "cooperation": True,
           "legal_actions": [0, 1, 17, 19], "bank": [500, 1000],
           "transferred": [200, 300], "cat_damage": [100, 50],
           "robots": [{"id": 1, "team": 0, "type": "BABY_RAT", "hp": 80,
                       "x": 4, "y": 5, "direction": 0, "cheese": 20},
                      {"id": 2, "team": 1, "type": "RAT_KING", "hp": 600,
                       "x": 50, "y": 20, "direction": 4}],
           "mines": [0] * 1800, "cheese": [0] * 1800,
           "trapsA": [0] * 1800, "trapsB": [0] * 1800}
    raw["mines"][15*60+40] = 1
    raw["cheese"][10*60+20] = 100
    raw["trapsA"][6*60+4] = 1
    raw["trapsB"][6*60+5] = 2
    return raw


class FeatureTests(unittest.TestCase):
    def test_distant_goals_global_context_and_boundary_padding(self):
        observation = encode(EngineState(fixture()))
        self.assertEqual(observation.local.shape, (33, 17, 17))
        self.assertEqual(observation.global_board.shape, (33, 60, 60))
        self.assertEqual(observation.scalars.shape, (64,))
        self.assertEqual(len(CHANNEL_NAMES), 33)
        self.assertEqual(len(SCALAR_NAMES), 64)
        self.assertEqual(observation.local[3].sum(), 0)  # mine beyond local radius
        self.assertEqual(observation.global_board[3, 15, 40], 1)
        self.assertEqual(observation.global_board[13, 20, 50], 1)  # distant enemy king
        self.assertEqual(observation.global_board[0, 30:].sum(), 0)
        self.assertEqual(observation.local[0, :, :4].sum(), 0)  # no wraparound at west edge
        mine = SCALAR_NAMES.index("nearest_mine_present")
        np.testing.assert_allclose(observation.scalars[mine:mine+3], [1, 36/60, 10/60])

    def test_swapping_team_labels_preserves_side_relative_observation(self):
        raw = fixture()
        swapped = copy.deepcopy(raw)
        swapped["current_team"] = 1
        for robot in swapped["robots"]:
            if robot["team"] in (0, 1):
                robot["team"] = 1-robot["team"]
        for field in ("bank", "transferred", "cat_damage"):
            swapped[field].reverse()
        swapped["trapsA"], swapped["trapsB"] = swapped["trapsB"], swapped["trapsA"]
        for original, reverse in zip(encode(EngineState(raw)), encode(EngineState(swapped))):
            np.testing.assert_array_equal(original, reverse)

    def test_carry_airborne_planes_do_not_corrupt_ground_robot(self):
        raw = fixture()
        raw["robots"].extend([
            {"id": 3, "team": 1, "type": "BABY_RAT", "hp": 20, "x": 4, "y": 5,
             "direction": 4, "cheese": 90, "carried": True},
            {"id": 4, "team": 0, "type": "BABY_RAT", "hp": 30, "x": 4, "y": 5,
             "direction": 2, "thrown": True},
        ])
        o = encode(EngineState(raw))
        self.assertEqual(o.global_board[28, 5, 4], 1)
        self.assertEqual(o.global_board[29, 5, 4], 1)
        self.assertEqual(o.global_board[11, 5, 4], 0)  # carried enemy not ground blocker
        self.assertAlmostEqual(float(o.global_board[15, 5, 4]), 0.8)
        self.assertAlmostEqual(float(o.global_board[20, 5, 4]), 0.2)
        self.assertEqual(o.global_board[23, 5, 4], 1)  # carrier's direction intact


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_neutral_value_legal_mask_and_versioned_checkpoint(self):
        torch.manual_seed(23)
        config = NetworkConfig(local_width=8, global_width=8, blocks=1)
        model = PolicyValueNetV2(config)
        observation = encode(EngineState(fixture()))
        probabilities, value = model.predict(observation, (0, 17, 19))
        self.assertEqual(value, 0)
        self.assertAlmostEqual(float(probabilities.sum()), 1, places=6)
        self.assertEqual(int(np.count_nonzero(probabilities)), 3)
        checkpoint = model.checkpoint()
        self.assertEqual(checkpoint["model_version"], "dual_scale_v2")
        self.assertEqual(checkpoint["feature_version"], FEATURE_VERSION)
        self.assertEqual(checkpoint["action_version"], "macro_v2")
        restored = PolicyValueNetV2(NetworkConfig(**checkpoint["network_config"]))
        restored.load_state_dict(checkpoint["model_state_dict"])
        np.testing.assert_array_equal(probabilities, restored.predict(observation, (0, 17, 19))[0])
        batch, values = model.predict_batch([observation, observation], [(0, 17, 19), (0, 17, 19)])
        np.testing.assert_allclose(batch[0], probabilities, atol=1e-7)
        np.testing.assert_array_equal(values, [0, 0])

    def test_wrong_extreme_value_logits_retain_corrective_gradient(self):
        # v1's tanh+MSE has essentially zero gradient when confidently wrong.
        wrong_logits = torch.tensor([-30.0, 30.0], requires_grad=True)
        labels = torch.tensor([1.0, -1.0])
        logits = torch.zeros((2, 40), requires_grad=True)
        policy = torch.full((2, 40), 1/40)
        loss, metrics = policy_value_loss(logits, wrong_logits, policy, labels)
        loss.backward()
        self.assertTrue(np.isfinite(metrics["loss"]))
        np.testing.assert_allclose(wrong_logits.grad.numpy(), [-0.5, 0.5], atol=1e-6)
        self.assertAlmostEqual(metrics["value_mse"], 4.0)
        saturated = torch.tensor([-30.0, 30.0], requires_grad=True)
        torch.nn.functional.mse_loss(saturated.tanh(), labels).backward()
        self.assertTrue(torch.all(saturated.grad.abs() < 1e-6))

    def test_forward_is_logit_and_optimizer_learns_terminal_direction(self):
        torch.manual_seed(23)
        model = PolicyValueNetV2(NetworkConfig(local_width=8, global_width=8, blocks=1))
        observation = encode(EngineState(fixture()))
        inputs = [torch.from_numpy(x).unsqueeze(0) for x in observation]
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        mask = torch.zeros((1, 40), dtype=torch.bool)
        mask[:, [0, 17, 19]] = True
        target = torch.zeros((1, 40))
        target[:, 19] = 1
        logits, value_logits = model(*inputs, mask)
        self.assertEqual(float(value_logits.item()), 0)
        loss, metrics = policy_value_loss(logits, value_logits, target, torch.tensor([1.0]))
        loss.backward()
        optimizer.step()
        self.assertGreater(model.predict(observation, (0, 17, 19))[1], 0)
        self.assertAlmostEqual(metrics["value_bce"], np.log(2), places=5)


if __name__ == "__main__":
    unittest.main()
