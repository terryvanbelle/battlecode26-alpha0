"""Game-balanced, bounded replay for completed v2 self-play games.

Each game contributes equally when sampling, irrespective of its length. Within
one game, reservoir sampling retains a uniform subset of searched positions.
Files contain numeric arrays only; loading never enables NumPy pickle support.
A directory is a single-writer checkpoint (the campaign launcher owns the lock).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Mapping
import uuid

import numpy as np


_SHAPES = {"local": (33, 17, 17), "global_board": (33, 60, 60),
           "scalars": (64,), "policy": (40,), "legal_mask": (40,)}
_DTYPES = {"local": np.float16, "global_board": np.float16,
           "scalars": np.float32, "policy": np.float32, "legal_mask": np.bool_,
           "value": np.float32, "player": np.int8}
_ARRAY_KEYS = frozenset(_DTYPES)
_FILE_RE = re.compile(r"game-[0-9a-f]{32}\.npz\Z")
_FORMAT_VERSION = 1


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _metadata(value: Mapping) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a JSON object identifying self-play")
    try:
        result = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain finite JSON values") from exc
    if result.get("source") not in ("selfplay", "self_play"):
        raise ValueError("replay accepts only metadata source='selfplay', never benchmark/evaluation data")
    if result.get("benchmark") or result.get("evaluation"):
        raise ValueError("benchmark/evaluation positions are not training data")
    return result


def _position(position: Mapping) -> dict[str, np.ndarray]:
    if not isinstance(position, Mapping):
        raise ValueError("positions must be dictionaries")
    player = position.get("player")
    if isinstance(player, (bool, np.bool_)) or not isinstance(player, (int, np.integer)) or player not in (0, 1):
        raise ValueError("player must be integer 0 or 1")
    result = {}
    for key, shape in _SHAPES.items():
        if key not in position:
            raise ValueError(f"position is missing {key}")
        raw = np.asarray(position[key])
        if raw.shape != shape or raw.dtype.kind not in "biuf":
            raise ValueError(f"{key} must be a numeric array with shape {shape}")
        if not np.isfinite(raw).all():
            raise ValueError(f"{key} must be finite")
        if key == "legal_mask" and not np.isin(raw, (0, 1)).all():
            raise ValueError("legal_mask must contain only booleans or 0/1")
        with np.errstate(over="ignore"):
            result[key] = raw.astype(_DTYPES[key], copy=True)
        if not np.isfinite(result[key]).all():
            raise ValueError(f"{key} cannot be represented in its storage dtype")
    policy, legal = result["policy"], result["legal_mask"]
    if not legal.any() or (policy < 0).any() or not np.isclose(policy.sum(), 1.0, atol=1e-5):
        raise ValueError("policy must be a probability distribution with at least one legal action")
    if np.any(policy[~legal] != 0):
        raise ValueError("policy assigns probability to an illegal action")
    result["player"] = np.int8(player)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass
class _Game:
    game_id: str
    arrays: dict[str, np.ndarray]
    terminal_value_a: float
    metadata: dict
    positions_seen: int
    filename: str
    sha256: str | None = None

    def __len__(self) -> int:
        return len(self.arrays["value"])


class GameReplayBuffer:
    """Uniform-over-games replay with whole-game FIFO eviction.

    ``add_game`` accepts only complete self-play and refuses duplicate IDs. It
    labels values from each position's player's perspective. ``capacity`` and
    ``max_games`` bound total storage; a new game retains at most
    ``min(max_positions_per_game, capacity)`` positions. Samples are with
    replacement: choose a game uniformly, then a retained position uniformly.
    """

    def __init__(self, capacity: int = 6000, max_games: int = 32,
                 max_positions_per_game: int = 512, seed: int = 0):
        self.capacity = _positive_integer("capacity", capacity)
        self.max_games = _positive_integer("max_games", max_games)
        self.max_positions_per_game = _positive_integer("max_positions_per_game", max_positions_per_game)
        self._rng = np.random.default_rng(seed)
        self._games: OrderedDict[str, _Game] = OrderedDict()
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __contains__(self, game_id: str) -> bool:
        return game_id in self._games

    @property
    def num_games(self) -> int:
        return len(self._games)

    @property
    def game_ids(self) -> tuple[str, ...]:
        return tuple(self._games)

    def add_game(self, game_id: str, positions: Iterable[Mapping], terminal_value_a: float,
                 metadata: Mapping, completed: bool = True) -> int:
        """Add one terminal game; return its retained position count.

        Failed validation leaves existing games and reservoir RNG unchanged.
        The source marker is mandatory: ``metadata={"source": "selfplay", ...}``.
        Callers should feed searched positions only, never greedy rollout-only
        positions, partial games, or frozen benchmark observations.
        """
        if completed is not True:
            raise ValueError("incomplete games cannot receive terminal training labels")
        if not isinstance(game_id, str) or not game_id or game_id in self._games:
            raise ValueError("game_id must be nonempty and unique among retained games")
        if not np.isscalar(terminal_value_a) or not np.isfinite(terminal_value_a) or not -1 <= terminal_value_a <= 1:
            raise ValueError("terminal_value_a must be finite and in [-1, 1]")
        game_metadata = _metadata(metadata)
        # A local generator makes insertion transactional even for a malformed
        # item late in a streaming iterable.
        rng = np.random.default_rng()
        rng.bit_generator.state = self._rng.bit_generator.state
        cap = min(self.max_positions_per_game, self.capacity)
        retained: list[dict] = []
        seen = 0
        for seen, position in enumerate(positions, start=1):
            normalized = _position(position)
            if seen <= cap:
                retained.append(normalized)
            else:
                slot = int(rng.integers(seen))
                if slot < cap:
                    retained[slot] = normalized
        if not retained:
            raise ValueError("a complete game must contain at least one searched position")
        arrays = {key: np.stack([p[key] for p in retained]) for key in (*_SHAPES, "player")}
        arrays["value"] = (float(terminal_value_a) * np.where(arrays["player"] == 0, 1, -1)).astype(np.float32)
        for array in arrays.values():
            array.flags.writeable = False
        game = _Game(game_id, arrays, float(terminal_value_a), game_metadata, seen,
                     f"game-{uuid.uuid4().hex}.npz")
        while self._games and (self._size + len(game) > self.capacity or len(self._games) >= self.max_games):
            _, old = self._games.popitem(last=False)
            self._size -= len(old)
        self._games[game_id] = game
        self._size += len(game)
        self._rng.bit_generator.state = rng.bit_generator.state
        return len(game)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        batch_size = _positive_integer("batch_size", batch_size)
        if not self._games:
            raise ValueError("cannot sample an empty replay")
        games = tuple(self._games.values())
        selected = [games[int(index)] for index in rng.integers(len(games), size=batch_size)]
        indices = [int(rng.integers(len(game))) for game in selected]
        result = {}
        for key in (*_SHAPES, "value"):
            values = np.stack([game.arrays[key][index] for game, index in zip(selected, indices)])
            result[key] = values.astype(np.bool_ if key == "legal_mask" else np.float32, copy=False)
        return result

    def save(self, directory: str | Path) -> None:
        """Publish immutable compressed game files, then atomically replace index.

        Old game files are collected only after the new index is durable. A
        failed pre-publication save leaves the prior checkpoint loadable.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        entries = []
        for game in self._games.values():
            destination = directory / game.filename
            if destination.exists():
                if game.sha256 is None or _sha256(destination) != game.sha256:
                    raise ValueError(f"existing game checkpoint is inconsistent: {game.filename}")
            else:
                fd, temporary_name = tempfile.mkstemp(prefix=".game-", suffix=".tmp", dir=directory)
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        np.savez_compressed(stream, **game.arrays)
                        stream.flush()
                        os.fsync(stream.fileno())
                    digest = _sha256(temporary)
                    os.replace(temporary, destination)
                    game.sha256 = digest
                finally:
                    temporary.unlink(missing_ok=True)
            entries.append({"game_id": game.game_id, "file": game.filename, "sha256": game.sha256,
                            "positions": len(game), "positions_seen": game.positions_seen,
                            "terminal_value_a": game.terminal_value_a, "metadata": game.metadata})
        index = {"format_version": _FORMAT_VERSION, "capacity": self.capacity, "max_games": self.max_games,
                 "max_positions_per_game": self.max_positions_per_game, "positions": len(self),
                 "rng_state": self._rng.bit_generator.state, "games": entries}
        fd, temporary_name = tempfile.mkstemp(prefix=".index-", suffix=".tmp", dir=directory)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(index, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            _sync_directory(directory)  # Make game filenames durable before referencing them.
            os.replace(temporary, directory / "index.json")
            _sync_directory(directory)
        finally:
            temporary.unlink(missing_ok=True)
        live_files = {game.filename for game in self._games.values()}
        for path in directory.glob("game-*.npz"):
            if _FILE_RE.fullmatch(path.name) and path.name not in live_files:
                path.unlink()
        _sync_directory(directory)

    @classmethod
    def load(cls, directory: str | Path) -> "GameReplayBuffer":
        """Load a validated checkpoint without allowing object-array pickles."""
        directory = Path(directory)
        index = json.loads((directory / "index.json").read_text())
        if index.get("format_version") != _FORMAT_VERSION:
            raise ValueError("unsupported replay format")
        result = cls(index["capacity"], index["max_games"], index["max_positions_per_game"])
        for entry in index["games"]:
            game_id, filename = entry["game_id"], entry["file"]
            if not isinstance(game_id, str) or not game_id or game_id in result:
                raise ValueError("invalid or duplicate checkpoint game ID")
            if not isinstance(filename, str) or not _FILE_RE.fullmatch(filename):
                raise ValueError("invalid game checkpoint filename")
            path = directory / filename
            if path.is_symlink() or _sha256(path) != entry["sha256"]:
                raise ValueError(f"game checkpoint checksum mismatch: {filename}")
            metadata = _metadata(entry["metadata"])
            count = _positive_integer("positions", entry["positions"])
            seen = _positive_integer("positions_seen", entry["positions_seen"])
            terminal = entry["terminal_value_a"]
            if seen < count or count > min(result.capacity, result.max_positions_per_game):
                raise ValueError("invalid game reservoir bounds")
            if not np.isscalar(terminal) or not np.isfinite(terminal) or not -1 <= terminal <= 1:
                raise ValueError("invalid checkpoint terminal outcome")
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != _ARRAY_KEYS:
                    raise ValueError("game checkpoint array keys do not match schema")
                arrays = {key: data[key] for key in _ARRAY_KEYS}
            for key, array in arrays.items():
                expected_shape = (count, *_SHAPES[key]) if key in _SHAPES else (count,)
                if array.shape != expected_shape or array.dtype != _DTYPES[key] or not np.isfinite(array).all():
                    raise ValueError(f"invalid checkpoint array {key}")
            if not np.isin(arrays["player"], (0, 1)).all():
                raise ValueError("invalid checkpoint player")
            policy, legal = arrays["policy"], arrays["legal_mask"]
            if ((policy < 0).any() or not np.isclose(policy.sum(axis=1), 1, atol=1e-5).all()
                    or not legal.any(axis=1).all() or np.any(policy[~legal] != 0)):
                raise ValueError("invalid checkpoint policy distribution")
            expected_values = (float(terminal) * np.where(arrays["player"] == 0, 1, -1)).astype(np.float32)
            if not np.array_equal(arrays["value"], expected_values):
                raise ValueError("checkpoint value labels have wrong player perspective")
            for array in arrays.values():
                array.flags.writeable = False
            game = _Game(game_id, arrays, float(terminal), metadata, seen, filename, entry["sha256"])
            result._games[game_id] = game
            result._size += count
        if (result._size != index["positions"] or result._size > result.capacity
                or result.num_games > result.max_games):
            raise ValueError("checkpoint violates replay capacity")
        result._rng.bit_generator.state = index["rng_state"]
        return result
