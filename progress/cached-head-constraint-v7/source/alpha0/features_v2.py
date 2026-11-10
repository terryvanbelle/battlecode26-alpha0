"""Versioned dual-scale, side-relative observations for the 40-action policy.

These are research observations: full terrain, robot state, and opposing traps
are visible. They are not limited to tournament sensing. No benchmark source or
trajectory is used to construct features; every feature is current engine state.
"""
from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING
import numpy as np

from .env import EngineState, DIRECTIONS, DIRECTION_NAMES

if TYPE_CHECKING:
    from .network_v2 import PolicyValueNetV2

FEATURE_VERSION = 2
ACTION_VERSION = "macro_v2"
ACTION_DIM = 40
LOCAL_SIZE, GLOBAL_SIZE = 17, 60
LOCAL_CHANNELS = GLOBAL_CHANNELS = 33
SCALAR_DIM = 64

CHANNEL_NAMES = (
    "on_map", "wall", "dirt", "mine", "log_ground_cheese",
    "ally_rat_trap", "ally_cat_trap", "enemy_rat_trap", "enemy_cat_trap", "acting_robot",
    "ally_baby", "enemy_baby", "ally_king", "enemy_king", "cat",
    "ally_baby_hp/100", "enemy_baby_hp/100", "ally_king_hp/600", "enemy_king_hp/600", "cat_hp/4000",
    "ally_robot_cheese/100", "enemy_robot_cheese/100", "direction_dx", "direction_dy",
    "action_cooldown/100", "movement_cooldown/100", "turning_cooldown/100",
    "carried_ally", "carried_enemy", "airborne_ally", "airborne_enemy", "relative_x/60", "relative_y/60",
)

_BASE_NAMES = (
    "round/2000", "cooperation", "width/60", "height/60", "actor_x_fraction", "actor_y_fraction",
    "actor_is_king", "actor_health_fraction", "actor_cheese/100", "actor_action_cooldown/100",
    "actor_movement_cooldown/100", "actor_turning_cooldown/100", "actor_direction_dx", "actor_direction_dy",
    "ally_bank/2500", "enemy_bank/2500", "ally_transferred/5000", "enemy_transferred/5000",
    "ally_cat_damage/10000", "enemy_cat_damage/10000", "ally_kings/5", "enemy_kings/5",
    "ally_babies/100", "enemy_babies/100", "ally_king_hp/3000", "enemy_king_hp/3000",
    "ally_baby_hp/10000", "enemy_baby_hp/10000", "cat_count/20", "cat_hp/80000",
    "ground_cheese_total/10000", "legal_actions/40",
)
GOALS = ("mine", "ground_cheese", "ally_king", "enemy_king", "cat", "enemy_baby")
SCALAR_NAMES = _BASE_NAMES + tuple(f"nearest_{goal}_{part}" for goal in GOALS
                                  for part in ("present", "dx/60", "dy/60", "distance/85")) + (
    "actor_carried", "actor_airborne", "legal_move_fraction", "legal_build_fraction",
    "remaining_round_fraction", "ally_trap_count/35", "enemy_trap_count/35", "king_hp_advantage/3000",
)


class Observation(NamedTuple):
    local: np.ndarray
    global_board: np.ndarray
    scalars: np.ndarray


def _direction(robot):
    direction = robot.get("direction", 8)
    return DIRECTIONS[DIRECTION_NAMES.index(direction) if isinstance(direction, str) else int(direction)]


def _parts(robot):
    if "parts" in robot:
        return robot["parts"]
    kind, x, y = robot["type"], int(robot["x"]), int(robot["y"])
    offsets = range(-1, 2) if kind == "RAT_KING" else (range(2) if kind == "CAT" else range(1))
    return [(x+dx, y+dy) for dy in offsets for dx in offsets]


def encode(state: EngineState) -> Observation:
    """Encode fixed-size local/global tensors and 64 documented scalars.

Global board is padded at the top/right to60x60; on_map separates padding from
terrain. The local crop is centered on the actor. Both use [C,y,x], NORTH=+y.
Nearest-goal vectors describe Euclidean geometry only, not privileged paths or
opponent strategy. Continuous inputs are normalized then clipped to[-5,5].
    """
    if state.terminal:
        raise ValueError("terminal observations are not evaluated by the policy")
    raw, player = state.raw, state.player
    if raw.get("action_version", ACTION_VERSION) != ACTION_VERSION:
        raise ValueError("v2 features require macro_v2 engine states")
    width, height = int(raw["width"]), int(raw["height"])
    if not 1 <= width <= GLOBAL_SIZE or not 1 <= height <= GLOBAL_SIZE:
        raise ValueError(f"v2 supports official maps up to60x60, got{width}x{height}")
    robots = raw["robots"]
    actor = next(r for r in robots if r["id"] == raw["current_id"])
    cx, cy = int(actor["x"]), int(actor["y"])
    board = np.zeros((GLOBAL_CHANNELS, GLOBAL_SIZE, GLOBAL_SIZE), dtype=np.float32)
    board[0, :height, :width] = 1

    def field(key):
        values = np.asarray(raw.get(key, [0] * (width*height)), dtype=np.float32)
        return values.reshape(height, width)

    walls, dirt, mines, cheese = (field(name) for name in ("walls", "dirt", "mines", "cheese"))
    traps_own, traps_other = (field("trapsA" if p == 0 else "trapsB") for p in (player, 1-player))
    board[1, :height, :width] = walls
    board[2, :height, :width] = dirt
    board[3, :height, :width] = mines
    board[4, :height, :width] = np.log1p(np.maximum(cheese, 0)) / np.log(1001)
    for channel, values, trap_type in ((5, traps_own, 1), (6, traps_own, 2),
                                       (7, traps_other, 1), (8, traps_other, 2)):
        board[channel, :height, :width] = values == trap_type
    board[31, :height, :width] = (np.arange(width, dtype=np.float32)[None, :] - cx) / GLOBAL_SIZE
    board[32, :height, :width] = (np.arange(height, dtype=np.float32)[:, None] - cy) / GLOBAL_SIZE

    for robot in robots:
        kind, ally = robot["type"], int(robot["team"]) == player
        category = 4 if kind == "CAT" else (2 if kind == "RAT_KING" else 0) + (0 if ally else 1)
        max_hp = 4000 if kind == "CAT" else (600 if kind == "RAT_KING" else 100)
        dx, dy = _direction(robot)
        for x, y in _parts(robot):
            if not (0 <= x < width and 0 <= y < height):
                continue
            if robot["id"] == actor["id"]:
                board[9, y, x] = 1
            if robot.get("carried", False):
                board[27 if ally else 28, y, x] = 1
                continue
            if robot.get("thrown", False):
                board[29 if ally else 30, y, x] = 1
                continue
            board[10+category, y, x] = 1
            board[15+category, y, x] = float(robot["hp"]) / max_hp
            if kind != "CAT":
                board[20 if ally else 21, y, x] = float(robot.get("cheese", 0)) / 100
            board[22, y, x], board[23, y, x] = dx, dy
            for channel, key in ((24, "action_cooldown"), (25, "movement_cooldown"), (26, "turning_cooldown")):
                board[channel, y, x] = float(robot.get(key, 0)) / 100

    # Terrain and robot channels now share a single definition at both scales.
    radius = LOCAL_SIZE // 2
    local = np.zeros((LOCAL_CHANNELS, LOCAL_SIZE, LOCAL_SIZE), dtype=np.float32)
    x0, x1, y0, y1 = max(0, cx-radius), min(width, cx+radius+1), max(0, cy-radius), min(height, cy+radius+1)
    local[:, y0-cy+radius:y1-cy+radius, x0-cx+radius:x1-cx+radius] = board[:, y0:y1, x0:x1]

    def pair(key, denominator):
        values = raw.get(key, [0, 0])
        return [float(values[player]) / denominator, float(values[1-player]) / denominator]

    def group(kind, team):
        return [r for r in robots if r["type"] == kind and int(r["team"]) == team]

    own_babies, other_babies = group("BABY_RAT", player), group("BABY_RAT", 1-player)
    own_kings, other_kings = group("RAT_KING", player), group("RAT_KING", 1-player)
    cats = [r for r in robots if r["type"] == "CAT"]
    is_king = actor["type"] == "RAT_KING"
    dx, dy = _direction(actor)
    king_hp = [sum(r["hp"] for r in group) for group in (own_kings, other_kings)]
    scalars = [float(raw["round"])/2000, float(raw["cooperation"]), width/60, height/60,
               cx/max(1, width-1), cy/max(1, height-1), float(is_king), float(actor["hp"])/(600 if is_king else 100),
               float(actor.get("cheese", 0))/100,
               *(float(actor.get(k, 0))/100 for k in ("action_cooldown", "movement_cooldown", "turning_cooldown")),
               dx, dy, *pair("bank", 2500), *pair("transferred", 5000), *pair("cat_damage", 10000),
               len(own_kings)/5, len(other_kings)/5, len(own_babies)/100, len(other_babies)/100,
               king_hp[0]/3000, king_hp[1]/3000,
               sum(r["hp"] for r in own_babies)/10000, sum(r["hp"] for r in other_babies)/10000,
               len(cats)/20, sum(r["hp"] for r in cats)/80000,
               float(cheese.sum())/10000, len(state.legal_actions)/ACTION_DIM]

    def locations(values):
        ys, xs = np.nonzero(values)
        return np.stack((xs, ys), axis=-1)

    def robot_locations(group):
        return np.asarray([(r["x"], r["y"]) for r in group], dtype=np.float32).reshape(-1, 2)

    for positions in (locations(mines), locations(cheese), robot_locations(own_kings),
                      robot_locations(other_kings), robot_locations(cats), robot_locations(other_babies)):
        if len(positions) == 0:
            scalars.extend((0, 0, 0, 0))
        else:
            delta = positions - np.array([cx, cy])
            distance2 = (delta*delta).sum(-1)
            nearest = int(distance2.argmin())
            scalars.extend((1, delta[nearest, 0]/60, delta[nearest, 1]/60, np.sqrt(distance2[nearest])/85))
    scalars.extend((float(actor.get("carried", False)), float(actor.get("thrown", False)),
                    sum(1 <= a <= 8 for a in state.legal_actions)/8,
                    sum(9 <= a <= 16 for a in state.legal_actions)/8,
                    max(0, 2000-float(raw["round"]))/2000,
                    np.count_nonzero(traps_own)/35, np.count_nonzero(traps_other)/35,
                    (king_hp[0]-king_hp[1])/3000))
    return Observation(np.clip(local, -5, 5), np.clip(board, -5, 5),
                       np.clip(np.asarray(scalars, dtype=np.float32), -5, 5))


def make_evaluator(network: "PolicyValueNetV2"):
    def evaluate(state: EngineState):
        return network.predict(encode(state), state.legal_actions)
    return evaluate
