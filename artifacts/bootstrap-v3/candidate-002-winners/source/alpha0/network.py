"""Compact residual policy/value model for CPU-first Battlecode research."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NetworkConfig:
    in_channels: int
    scalar_dim: int
    action_dim: int
    channels: int = 32
    blocks: int = 3
    hidden_dim: int = 128


def _groups(channels: int) -> int:
    return next(g for g in (8, 4, 2, 1) if channels % g == 0)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.layers(x))


class PolicyValueNet(nn.Module):
    """Spatial residual trunk + geometry/scalar fusion + separate policy/value.

The 4x4 pooled map preserves coarse geometry while permitting arbitrary map
sizes. Inputs must be side-relative (ally/enemy channels) and include the acting
robot plane. GroupNorm supports batch-one inference without moving statistics.
    """
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        c = config.channels
        self.trunk = nn.Sequential(
            nn.Conv2d(config.in_channels, c, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(c), c), nn.SiLU(),
            *(ResidualBlock(c) for _ in range(config.blocks)),
        )
        self.spatial = nn.Sequential(nn.Conv2d(c, 8, 1), nn.SiLU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.scalars = nn.Sequential(nn.Linear(config.scalar_dim, 32), nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(8 * 4 * 4 + 2 * c + 32, config.hidden_dim), nn.SiLU())
        self.policy = nn.Linear(config.hidden_dim, config.action_dim)
        self.value = nn.Sequential(nn.Linear(config.hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, board: torch.Tensor, scalars: torch.Tensor,
                legal_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(board)
        fused = torch.cat((self.spatial(x).flatten(1), x.mean((-2, -1)),
                           x.amax((-2, -1)), self.scalars(scalars)), dim=1)
        hidden = self.fusion(fused)
        logits = self.policy(hidden)
        if legal_mask is not None:
            mask = legal_mask.bool()
            if mask.shape != logits.shape or not torch.all(mask.any(dim=-1)):
                raise ValueError("each nonterminal example must have at least one legal action")
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits, self.value(hidden).squeeze(-1)

    @torch.inference_mode()
    def predict(self, board: np.ndarray, scalars: np.ndarray,
                legal_actions: list[int] | tuple[int, ...]) -> tuple[np.ndarray, float]:
        device = next(self.parameters()).device
        mask = torch.zeros((1, self.config.action_dim), dtype=torch.bool, device=device)
        mask[0, list(legal_actions)] = True
        was_training = self.training
        self.eval()
        try:
            logits, value = self(torch.as_tensor(board, dtype=torch.float32, device=device).unsqueeze(0),
                                 torch.as_tensor(scalars, dtype=torch.float32, device=device).unsqueeze(0), mask)
            return logits.softmax(dim=-1)[0].cpu().numpy(), float(value.item())
        finally:
            self.train(was_training)

    def checkpoint(self, **metadata: Any) -> dict[str, Any]:
        return {"format_version": 1, "network_config": asdict(self.config),
                "model_state_dict": self.state_dict(), **metadata}


def policy_value_loss(logits: torch.Tensor, values: torch.Tensor,
                      target_policy: torch.Tensor, target_value: torch.Tensor,
                      *, value_weight: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
    """Visit-distribution cross entropy + terminal-outcome MSE.

Use AdamW weight_decay for regularization. Value labels should be the actual
terminal game outcome from each example's player's perspective, without a
discount factor or heuristic shaping.
    """
    if not torch.isfinite(target_policy).all() or not torch.isfinite(target_value).all():
        raise ValueError("nonfinite training target")
    if (target_policy < 0).any() or not torch.allclose(
            target_policy.sum(-1), torch.ones_like(target_value), atol=1e-4):
        raise ValueError("policy targets must be probability distributions")
    if (target_value.abs() > 1).any():
        raise ValueError("value targets must be in [-1, 1]")
    policy_loss = -(target_policy * F.log_softmax(logits, dim=-1)).sum(-1).mean()
    value_loss = F.mse_loss(values, target_value)
    loss = policy_loss + value_weight * value_loss
    probabilities = logits.detach().softmax(-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean()
    return loss, {"loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
                  "value_loss": float(value_loss.detach()), "policy_entropy": float(entropy)}
