"""JSONL adapter and side-relative features for the official Java game engine.

The research agent receives omniscient local state and global aggregates. This
is deliberately not a tournament-observation-compatible policy. All actions and
terminal outcomes still come from the real Battlecode engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING
import json
import os
import select
import subprocess
import tempfile
import time

import numpy as np

if TYPE_CHECKING:
    from alpha0.network import PolicyValueNet

ROOT = Path(__file__).resolve().parents[1]
ACTION_DIM, IN_CHANNELS, SCALAR_DIM, LOCAL_SIZE = 17, 24, 32, 17
DIRECTIONS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 0))
DIRECTION_NAMES = ("NORTH", "NORTHEAST", "EAST", "SOUTHEAST", "SOUTH", "SOUTHWEST", "WEST", "NORTHWEST", "CENTER")
JAVA_OPENS = [f"--add-opens=java.base/{package}=ALL-UNNAMED" for package in
              ("jdk.internal.misc", "jdk.internal.math", "jdk.internal.util", "jdk.internal.access", "sun.security.action")]
CHANNEL_NAMES = (
    "on_map", "wall", "dirt", "cheese_mine", "ground_cheese/100",
    "ally_trap", "enemy_trap", "acting_robot",
    "ally_baby", "enemy_baby", "ally_king", "enemy_king", "cat",
    "ally_baby_hp/100", "enemy_baby_hp/100", "ally_king_hp/600", "enemy_king_hp/600", "cat_hp/4000",
    "robot_cheese/100", "direction_dx", "direction_dy",
    "action_cooldown/100", "movement_cooldown/100", "turning_cooldown/100",
)
SCALAR_NAMES = (
    "round/2000", "cooperation", "width/60", "height/60", "actor_x_fraction", "actor_y_fraction",
    "actor_is_king", "actor_health_fraction", "actor_cheese/100", "actor_action_cooldown/100",
    "actor_movement_cooldown/100", "actor_turning_cooldown/100", "actor_direction_dx", "actor_direction_dy",
    "ally_bank/2500", "enemy_bank/2500", "ally_transferred/5000", "enemy_transferred/5000",
    "ally_cat_damage/10000", "enemy_cat_damage/10000", "ally_kings/5", "enemy_kings/5",
    "ally_babies/100", "enemy_babies/100", "ally_king_hp/3000", "enemy_king_hp/3000",
    "ally_baby_hp/10000", "enemy_baby_hp/10000", "cat_count/20", "cat_hp/80000",
    "ground_cheese_total/10000", "legal_actions/17",
)


@dataclass(frozen=True)
class EngineState:
    raw: dict[str, Any]
    history: tuple[int, ...] = ()
    snapshot: int | str | None = None
    game_id: int = 0

    @property
    def player(self) -> int:
        return int(self.raw.get("current_team", 0))

    @property
    def legal_actions(self) -> tuple[int, ...]:
        return tuple(int(a) for a in self.raw.get("legal_actions", ()))

    @property
    def terminal(self) -> bool:
        return bool(self.raw["terminal"])

    @property
    def terminal_value_a(self) -> float:
        if not self.terminal:
            raise ValueError("a nonterminal position has no terminal outcome")
        winner = int(self.raw["winner"])
        if winner not in (-1, 0, 1):
            raise ValueError(f"invalid engine winner {winner}")
        return 0.0 if winner == -1 else (1.0 if winner == 0 else -1.0)


class EngineEnv:
    """Own one Java process; snapshots are scoped to its current game.

Call release_except(actual_state) after committing a chosen action. This frees
all speculative MCTS snapshots while retaining the real trajectory's position.
Frozen Java opponents support sequential evaluation but not snapshot search.
    """
    def __init__(self, command: list[str] | None = None, maps_dir: str | Path | None = None,
                 opponent_package: str | None = None, opponent_classes: str | Path | None = None,
                 learned_team: int = 0, response_timeout: float = 120.0,
                 action_version: str = "macro_v1"):
        java = Path(os.environ["JAVA_HOME"]) / "bin/java" if os.environ.get("JAVA_HOME") else ROOT / "vendor/jdk/bin/java"
        command = command or [str(java) if java.exists() else "java", *JAVA_OPENS, "-cp",
            f"{ROOT / 'build/java'}:{ROOT / 'vendor/engine/battlecode26-java-1.2.6.jar'}", "alpha0.EngineBridge"]
        self.maps_dir = str(Path(maps_dir or ROOT / "maps").resolve())
        self.opponent_package = opponent_package
        self.opponent_classes = str(Path(opponent_classes).resolve()) if opponent_classes else None
        self.learned_team = learned_team
        self.response_timeout = response_timeout
        if action_version not in ("macro_v1", "macro_v2"):
            raise ValueError(f"unknown action version: {action_version}")
        self.action_version = action_version
        # A file avoids an unread stderr pipe stalling the Java process.
        self._stderr = tempfile.TemporaryFile(mode="w+b")
        self.process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self._stderr, bufsize=0)
        self._buffer = b""
        self._snapshots: set[int | str] = set()
        self._active_snapshot: int | str | None = None
        self._active_history: tuple[int, ...] = ()
        self._game_id = 0
        self._reset_args: dict[str, Any] | None = None

    def _error_tail(self) -> str:
        self._stderr.seek(0, 2)
        self._stderr.seek(max(0, self._stderr.tell() - 4000))
        return self._stderr.read().decode("utf-8", errors="replace")

    def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise RuntimeError(f"engine exited {self.process.returncode}: {self._error_tail()}")
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write((json.dumps(message, separators=(",", ":")) + "\n").encode())
        self.process.stdin.flush()
        deadline = time.monotonic() + self.response_timeout
        while True:
            if b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                    self.process.kill()
                    self.process.wait(timeout=10)
                    raise TimeoutError(f"engine response exceeded {self.response_timeout}s: {self._error_tail()}")
                data = self.process.stdout.read(65536)
                if not data:
                    raise RuntimeError(f"engine stdout closed: {self._error_tail()}")
                self._buffer += data
                continue
            line, self._buffer = self._buffer.split(b"\n", 1)
            try:
                response = json.loads(line)
            except (ValueError, UnicodeError):
                continue  # Official engine diagnostics can precede a JSON response.
            if not isinstance(response, dict):
                continue
            if "error" in response:
                raise RuntimeError(f"engine {message['cmd']} failed: {response['error']}")
            return response

    def _state(self, raw: dict[str, Any], history: tuple[int, ...]) -> EngineState:
        snapshot = raw.get("snapshot")
        if self.opponent_package is None and not raw["terminal"] and snapshot is None:
            snapshot = self._request({"cmd": "snapshot"})["snapshot"]
        if snapshot is not None:
            self._snapshots.add(snapshot)
        self._active_snapshot, self._active_history = snapshot, history
        return EngineState(raw, history, snapshot, self._game_id)

    def reset(self, map_name: str, seed: int) -> EngineState:
        if self._snapshots:
            self.release_except(None)
        self._game_id += 1
        self._reset_args = {"cmd": "reset", "map": map_name, "seed": int(seed), "maps_dir": self.maps_dir,
                            "snapshot": self.opponent_package is None, "action_version": self.action_version}
        if self.opponent_package is not None:
            self._reset_args.update(opponent_package=self.opponent_package,
                                    opponent_classes=self.opponent_classes, learned_team=self.learned_team)
        return self._state(self._request(self._reset_args), ())

    def transition(self, state: EngineState, action: int) -> EngineState:
        if state.game_id != self._game_id:
            raise ValueError("state belongs to an earlier game")
        if state.terminal or action not in state.legal_actions:
            raise ValueError(f"illegal action {action}")
        if state.snapshot is not None:
            if state.snapshot not in self._snapshots:
                raise ValueError("state snapshot has already been released")
            if self._active_snapshot != state.snapshot:
                self._request({"cmd": "restore", "snapshot": state.snapshot})
                self._active_snapshot = state.snapshot
        elif self._active_history != state.history:
            # Replay is also the reference fallback for a backend without snapshots.
            assert self._reset_args is not None
            self._request(self._reset_args)
            for past_action in state.history:
                self._request({"cmd": "step", "action": past_action})
        return self._state(self._request({"cmd": "step", "action": int(action),
                                        "snapshot": self.opponent_package is None}), state.history + (int(action),))

    def release_except(self, state: EngineState | None) -> None:
        keep = state.snapshot if state is not None else None
        released = [snapshot for snapshot in self._snapshots if snapshot != keep]
        if released:
            self._request({"cmd": "release", "snapshots": released})
            self._snapshots.difference_update(released)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()
        self._stderr.close()

    def __enter__(self) -> "EngineEnv":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def _direction(robot: dict[str, Any]) -> tuple[int, int]:
    value = robot.get("direction", 8)
    index = DIRECTION_NAMES.index(value) if isinstance(value, str) else int(value)
    return DIRECTIONS[index]


def _type(robot: dict[str, Any]) -> str:
    return str(robot["type"]).upper()


def encode(state: EngineState) -> tuple[np.ndarray, np.ndarray]:
    """Actor-centered 17x17 tensor plus 32 global/actor scalars, all side-relative.

    Arrays use [channel, local_y, local_x], with y increasing toward NORTH. The
    engine's exact ground footprints are filled. Carried and airborne robots are
    excluded from ground occupancy to avoid overwriting their carrier; they are
    still included in global counts/HP. This 24-plane v1 schema has no dedicated
    carried/airborne planes (self-play macros do not initiate these actions).
    Channels are clipped to [-5, 5] after normalization.
    """
    raw, player = state.raw, state.player
    if state.terminal:
        raise ValueError("terminal states do not need network observations")
    width, height = int(raw["width"]), int(raw["height"])
    robots = raw["robots"]
    actor = next(r for r in robots if r["id"] == raw["current_id"])
    cx, cy = int(actor["x"]), int(actor["y"])
    radius = LOCAL_SIZE // 2
    board = np.zeros((IN_CHANNELS, LOCAL_SIZE, LOCAL_SIZE), dtype=np.float32)
    fields = [(1, "walls", 1), (2, "dirt", 1), (3, "mines", 1), (4, "cheese", 100),
              (5, "trapsA" if player == 0 else "trapsB", 1),
              (6, "trapsB" if player == 0 else "trapsA", 1)]
    x0, x1, y0, y1 = max(0, cx-radius), min(width, cx+radius+1), max(0, cy-radius), min(height, cy+radius+1)
    tx, ty = slice(x0-cx+radius, x1-cx+radius), slice(y0-cy+radius, y1-cy+radius)
    board[0, ty, tx] = 1
    for channel, field, divisor in fields:
        values = np.asarray(raw.get(field, [0] * (width*height)), dtype=np.float32).reshape(height, width)
        if "trap" in field.lower():
            values = (values > 0).astype(np.float32)
        board[channel, ty, tx] = values[y0:y1, x0:x1] / divisor

    def footprint(robot: dict[str, Any]):
        kind, x, y = _type(robot), int(robot["x"]), int(robot["y"])
        parts = robot.get("parts")
        if parts is None:
            offsets = range(-1, 2) if kind == "RAT_KING" else (range(2) if kind == "CAT" else range(1))
            parts = [(x+dx, y+dy) for dy in offsets for dx in offsets]
        for px, py in parts:
            xx, yy = px-cx+radius, py-cy+radius
            if 0 <= xx < LOCAL_SIZE and 0 <= yy < LOCAL_SIZE:
                yield yy, xx

    for robot in robots:
        if robot.get("carried", False) or robot.get("thrown", False):
            continue
        kind = _type(robot)
        ally = int(robot["team"]) == player
        category = 4 if kind == "CAT" else (2 if kind == "RAT_KING" else 0) + (0 if ally else 1)
        max_hp = 4000 if kind == "CAT" else (600 if kind == "RAT_KING" else 100)
        dx, dy = _direction(robot)
        for yy, xx in footprint(robot):
            board[8+category, yy, xx] = 1
            board[13+category, yy, xx] = float(robot["hp"]) / max_hp
            board[18, yy, xx] = float(robot.get("cheese", 0)) / 100
            board[19, yy, xx], board[20, yy, xx] = dx, dy
            for channel, key in ((21, "action_cooldown"), (22, "movement_cooldown"), (23, "turning_cooldown")):
                board[channel, yy, xx] = float(robot.get(key, 0)) / 100
            if robot["id"] == actor["id"]:
                board[7, yy, xx] = 1

    def pair(key, denominator):
        values = raw.get(key, [0, 0])
        return [float(values[player]) / denominator, float(values[1-player]) / denominator]

    def group(kind: str, team: int):
        return [r for r in robots if _type(r) == kind and int(r["team"]) == team]

    own_babies, other_babies = group("BABY_RAT", player), group("BABY_RAT", 1-player)
    own_kings, other_kings = group("RAT_KING", player), group("RAT_KING", 1-player)
    cats = [r for r in robots if _type(r) == "CAT"]
    is_king = _type(actor) == "RAT_KING"
    dx, dy = _direction(actor)
    scalars = [float(raw["round"])/2000, float(raw["cooperation"]), width/60, height/60,
               cx/max(1, width-1), cy/max(1, height-1), float(is_king), float(actor["hp"])/(600 if is_king else 100),
               float(actor.get("cheese", 0))/100,
               *(float(actor.get(k, 0))/100 for k in ("action_cooldown", "movement_cooldown", "turning_cooldown")),
               dx, dy, *pair("bank", 2500), *pair("transferred", 5000), *pair("cat_damage", 10000),
               len(own_kings)/5, len(other_kings)/5, len(own_babies)/100, len(other_babies)/100,
               sum(r["hp"] for r in own_kings)/3000, sum(r["hp"] for r in other_kings)/3000,
               sum(r["hp"] for r in own_babies)/10000, sum(r["hp"] for r in other_babies)/10000,
               len(cats)/20, sum(r["hp"] for r in cats)/80000,
               sum(raw.get("cheese", []))/10000, len(state.legal_actions)/ACTION_DIM]
    return np.clip(board, -5, 5), np.clip(np.asarray(scalars, dtype=np.float32), -5, 5)


def make_evaluator(network: "PolicyValueNet"):
    def evaluate(state: EngineState) -> tuple[np.ndarray, float]:
        board, scalars = encode(state)
        return network.predict(board, scalars, state.legal_actions)
    return evaluate
