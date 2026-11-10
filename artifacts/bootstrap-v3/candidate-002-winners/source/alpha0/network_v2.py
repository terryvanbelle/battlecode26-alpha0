"""Small dual-scale residual model, versioned separately from v1 checkpoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .network import ResidualBlock, _groups
from .features_v2 import Observation, LOCAL_CHANNELS, GLOBAL_CHANNELS, SCALAR_DIM, FEATURE_VERSION

MODEL_VERSION = "dual_scale_v2"


@dataclass(frozen=True)
class NetworkConfig:
    action_dim: int = 40
    local_width: int = 24
    global_width: int = 16
    blocks: int = 2
    hidden_dim: int = 128
    local_channels: int = LOCAL_CHANNELS
    global_channels: int = GLOBAL_CHANNELS
    scalar_dim: int = SCALAR_DIM


def _conv(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                         nn.GroupNorm(_groups(out_channels), out_channels), nn.SiLU())


class PolicyValueNetV2(nn.Module):
    """Local17x17 residual trunk and global60x60 ->15x15 residual trunk.

Local actor-center/mean/max features preserve immediate tactics. Global
actor/mean/max and a coarse4x4 projection retain geography and distant goals.
The value output starts at zero to avoid an arbitrary saturated initial value
becoming the target of low-budget search on the first self-play games.
    """
    def __init__(self, config: NetworkConfig | None = None):
        super().__init__()
        self.config = config or NetworkConfig()
        cfg = self.config
        if cfg.local_channels <= 9 or cfg.global_channels <= 9:
            raise ValueError("v2 requires the documented actor channel9")
        l, g = cfg.local_width, cfg.global_width
        self.local_trunk = nn.Sequential(_conv(cfg.local_channels, l),
                                        *(ResidualBlock(l) for _ in range(cfg.blocks)))
        self.global_trunk = nn.Sequential(_conv(cfg.global_channels, g, stride=2),
                                         _conv(g, g, stride=2),
                                         *(ResidualBlock(g) for _ in range(cfg.blocks)))
        self.global_coarse = nn.Sequential(nn.Conv2d(g, 4, 1), nn.SiLU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.scalar_trunk = nn.Sequential(nn.Linear(cfg.scalar_dim, 32), nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(3*l+3*g+64+32, cfg.hidden_dim), nn.SiLU())
        self.policy = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        self.value = nn.Sequential(nn.Linear(cfg.hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.normal_(self.policy.weight, std=0.01)
        nn.init.zeros_(self.policy.bias)
        nn.init.zeros_(self.value[-1].weight)
        nn.init.zeros_(self.value[-1].bias)

    @staticmethod
    def _masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (features * mask).sum((-2, -1)) / mask.sum((-2, -1)).clamp_min(1)

    @staticmethod
    def _masked_max(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return features.masked_fill(mask <= 0, torch.finfo(features.dtype).min).amax((-2, -1))

    def forward(self, local: torch.Tensor, global_board: torch.Tensor, scalars: torch.Tensor,
                legal_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy logits and an unbounded scalar win logit, never tanh.

Training uses BCEWithLogits against(z+1)/2. Predict converts the win logit to
2*sigmoid(logit)-1 for the PUCT evaluator's[-1,1] value contract.
        """
        local_features, global_features = self.local_trunk(local), self.global_trunk(global_board)
        local_valid = local[:, 0:1]
        global_valid = F.adaptive_avg_pool2d(global_board[:, 0:1], global_features.shape[-2:])
        actor = F.adaptive_max_pool2d(global_board[:, 9:10], global_features.shape[-2:])
        cy, cx = local_features.shape[-2]//2, local_features.shape[-1]//2
        combined = torch.cat((local_features[:, :, cy, cx],
                              self._masked_mean(local_features, local_valid), self._masked_max(local_features, local_valid),
                              self._masked_mean(global_features, global_valid), self._masked_max(global_features, global_valid),
                              self._masked_mean(global_features, actor), self.global_coarse(global_features).flatten(1),
                              self.scalar_trunk(scalars)), dim=1)
        hidden = self.fusion(combined)
        logits = self.policy(hidden)
        if legal_mask is not None:
            mask = legal_mask.bool()
            if mask.shape != logits.shape or not torch.all(mask.any(dim=-1)):
                raise ValueError("each evaluated example needs legal actions")
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits, self.value(hidden).squeeze(-1)

    @torch.inference_mode()
    def predict(self, observation: Observation, legal_actions) -> tuple[np.ndarray, float]:
        device = next(self.parameters()).device
        inputs = [torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0) for x in observation]
        mask = torch.zeros((1, self.config.action_dim), dtype=torch.bool, device=device)
        mask[0, list(legal_actions)] = True
        was_training = self.training
        self.eval()
        try:
            logits, value = self(*inputs, mask)
            return logits.softmax(-1)[0].cpu().numpy(), float((2*value.sigmoid()-1).item())
        finally:
            self.train(was_training)

    @torch.inference_mode()
    def predict_batch(self, observations: list[Observation], legal_actions: list[tuple[int, ...]]):
        """Batch neural leaves when callers collect independent search frontiers."""
        if not observations or len(observations) != len(legal_actions):
            raise ValueError("observations and action lists must have equal nonzero length")
        device = next(self.parameters()).device
        inputs = [torch.as_tensor(np.stack([o[i] for o in observations]), dtype=torch.float32, device=device)
                  for i in range(3)]
        mask = torch.zeros((len(observations), self.config.action_dim), dtype=torch.bool, device=device)
        for row, actions in enumerate(legal_actions):
            mask[row, list(actions)] = True
        was_training = self.training
        self.eval()
        try:
            logits, value = self(*inputs, mask)
            return logits.softmax(-1).cpu().numpy(), (2*value.sigmoid()-1).cpu().numpy()
        finally:
            self.train(was_training)

    def checkpoint(self, **metadata: Any) -> dict[str, Any]:
        return {"format_version": 2, "model_version": MODEL_VERSION, "feature_version": FEATURE_VERSION,
                "action_version": "macro_v2", "network_config": asdict(self.config),
                "model_state_dict": self.state_dict(), **metadata}


def policy_value_loss(logits: torch.Tensor, value_logits: torch.Tensor,
                      target_policy: torch.Tensor, target_value: torch.Tensor,
                      *, value_weight: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
    """Visit-distribution CE plus win-logit BCE, with terminal labels in[-1,1].

Unlike tanh+MSE, BCEWithLogits retains a corrective gradient for a confidently
wrong value. Values used by MCTS remain in[-1,1]; that conversion is inference
only. MSE is reported separately for comparison with the v1 runs.
    """
    if not torch.isfinite(target_policy).all() or not torch.isfinite(target_value).all():
        raise ValueError("nonfinite training target")
    if (target_policy < 0).any() or not torch.allclose(
            target_policy.sum(-1), torch.ones_like(target_value), atol=1e-4):
        raise ValueError("policy targets must be normalized distributions")
    if (target_value.abs() > 1).any():
        raise ValueError("terminal values must be in[-1,1]")
    policy_loss = -(target_policy * F.log_softmax(logits, dim=-1)).sum(-1).mean()
    value_loss = F.binary_cross_entropy_with_logits(value_logits, (target_value+1)/2)
    loss = policy_loss + value_weight * value_loss
    probabilities = logits.detach().softmax(-1)
    predictions = 2*value_logits.detach().sigmoid()-1
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean()
    return loss, {"loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
                  "value_loss": float(value_loss.detach()), "value_bce": float(value_loss.detach()),
                  "value_mse": float(F.mse_loss(predictions, target_value)),
                  "value_mean": float(predictions.mean()),
                  "value_saturated_fraction": float((predictions.abs() > 0.95).float().mean()),
                  "policy_entropy": float(entropy)}
