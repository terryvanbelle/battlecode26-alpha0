"""Bounded self-play replay storage with explicit terminal-result labeling."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Position:
    board: np.ndarray
    scalars: np.ndarray
    policy: np.ndarray
    legal_mask: np.ndarray
    player: int


@dataclass(frozen=True)
class TrainingExample:
    board: np.ndarray
    scalars: np.ndarray
    policy: np.ndarray
    legal_mask: np.ndarray
    value: float


def label_game(positions: Iterable[Position], terminal_value_a: float) -> list[TrainingExample]:
    """Call only after a real terminal result; aborted games receive no labels."""
    if not np.isfinite(terminal_value_a) or not -1 <= terminal_value_a <= 1:
        raise ValueError("terminal outcome must be in [-1, 1]")
    result = []
    for p in positions:
        if p.player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        result.append(TrainingExample(np.asarray(p.board, dtype=np.float16).copy(),
                                      np.asarray(p.scalars, dtype=np.float32).copy(),
                                      np.asarray(p.policy, dtype=np.float32).copy(),
                                      np.asarray(p.legal_mask, dtype=bool).copy(),
                                      terminal_value_a * (1 if p.player == 0 else -1)))
    return result


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.examples: deque[TrainingExample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.examples)

    def extend(self, examples: Iterable[TrainingExample]) -> None:
        self.examples.extend(examples)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if not self.examples or batch_size < 1:
            raise ValueError("cannot sample an empty replay or nonpositive batch")
        indices = rng.choice(len(self.examples), size=batch_size, replace=len(self.examples) < batch_size)
        batch = [self.examples[int(i)] for i in indices]
        return {"board": np.stack([e.board for e in batch]).astype(np.float32),
                "scalars": np.stack([e.scalars for e in batch]),
                "policy": np.stack([e.policy for e in batch]),
                "legal_mask": np.stack([e.legal_mask for e in batch]),
                "value": np.asarray([e.value for e in batch], dtype=np.float32)}

    def save(self, path: str | Path) -> None:
        """Atomic checkpoint; no pickle and no silently partial replay files."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        batch = list(self.examples)
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, capacity=self.capacity,
                board=np.stack([e.board for e in batch]) if batch else np.empty((0,)),
                scalars=np.stack([e.scalars for e in batch]) if batch else np.empty((0,)),
                policy=np.stack([e.policy for e in batch]) if batch else np.empty((0,)),
                legal_mask=np.stack([e.legal_mask for e in batch]) if batch else np.empty((0,)),
                value=np.asarray([e.value for e in batch], dtype=np.float32))
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        with np.load(path, allow_pickle=False) as data:
            result = cls(int(data["capacity"]))
            for i, value in enumerate(data["value"]):
                result.examples.append(TrainingExample(data["board"][i], data["scalars"][i],
                    data["policy"][i], data["legal_mask"][i], float(value)))
            return result
