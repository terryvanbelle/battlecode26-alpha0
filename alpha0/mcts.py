"""PUCT search over the real engine's deterministic, branchable transitions.

Values on every edge are stored from team A's perspective. Selection converts
them to the player at that node, so two consecutive A turns do not accidentally
negate their value. This matters in Battlecode's spawn-order execution model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, Sequence
import math

import numpy as np


class SearchState(Protocol):
    player: int                 # 0 = team A, 1 = team B
    legal_actions: Sequence[int]
    terminal: bool
    terminal_value_a: float      # only used at a genuine terminal state


Evaluator = Callable[[SearchState], tuple[Mapping[int, float] | np.ndarray, float]]
Transition = Callable[[SearchState, int], SearchState]


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 32
    cpuct: float = 1.5
    dirichlet_alpha: float = 0.3
    exploration_fraction: float = 0.25
    max_depth: int = 64


@dataclass
class _Node:
    state: SearchState
    expanded: bool = False
    priors: dict[int, float] = field(default_factory=dict)
    visits: dict[int, int] = field(default_factory=dict)
    total_value_a: dict[int, float] = field(default_factory=dict)
    children: dict[int, "_Node"] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    visits: dict[int, int]
    policy: np.ndarray
    value: float                 # root player's perspective
    simulations: int
    nodes: int
    max_depth: int

    def select_action(self, rng: np.random.Generator, temperature: float = 1.0) -> int:
        """Temperature changes action sampling, never the stored visit target."""
        actions = np.asarray(list(self.visits), dtype=np.int64)
        counts = np.asarray([self.visits[int(a)] for a in actions], dtype=np.float64)
        if temperature < 0 or not math.isfinite(temperature):
            raise ValueError("temperature must be finite and nonnegative")
        if temperature == 0:
            return int(actions[np.argmax(counts)])
        # Log-space avoids overflow for small positive temperatures.
        positive = counts > 0
        log_weights = np.full(counts.shape, -np.inf)
        log_weights[positive] = np.log(counts[positive]) / temperature
        weights = np.exp(log_weights - np.max(log_weights))
        return int(rng.choice(actions, p=weights / weights.sum()))


def _sign(player: int) -> float:
    if player not in (0, 1):
        raise ValueError(f"expected player 0/1, got {player!r}")
    return 1.0 if player == 0 else -1.0


def search(root_state: SearchState, evaluate: Evaluator, transition: Transition,
           action_dim: int, *, config: SearchConfig | None = None,
           rng: np.random.Generator | None = None,
           add_root_noise: bool = False) -> SearchResult:
    """Run PUCT, returning raw normalized visit counts for policy supervision.

``evaluate`` returns a full action probability vector (or action->probability
mapping) and a value in the state's current player's perspective. ``transition``
must not mutate its input state. It may implement branching by restoring an
engine snapshot or replaying an action history; no approximate simulator is used
by this module. Terminal values are absolute team-A outcomes in [-1, 1].
    """
    cfg = config or SearchConfig()
    rng = rng if rng is not None else np.random.default_rng()
    if cfg.simulations < 1 or cfg.max_depth < 1 or cfg.cpuct < 0:
        raise ValueError("simulations/max_depth must be positive and cpuct nonnegative")
    if not 0 <= cfg.exploration_fraction <= 1 or cfg.dirichlet_alpha <= 0:
        raise ValueError("invalid Dirichlet exploration parameters")
    if root_state.terminal:
        raise ValueError("cannot search a terminal state")
    root = _Node(root_state)
    node_count, deepest = 1, 0

    def expand(node: _Node) -> float:
        if node.state.terminal:
            value = float(node.state.terminal_value_a)
            if not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError("terminal outcome must be finite and in [-1, 1]")
            return value
        actions = tuple(int(a) for a in node.state.legal_actions)
        if not actions or len(actions) != len(set(actions)):
            raise ValueError("nonterminal state needs unique legal actions")
        if min(actions) < 0 or max(actions) >= action_dim:
            raise ValueError("legal action outside policy head")
        probabilities, value = evaluate(node.state)
        value = float(value)
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("network value must be finite and in [-1, 1]")
        prior = np.asarray([probabilities[a] for a in actions], dtype=np.float64)
        if not np.isfinite(prior).all() or (prior < 0).any():
            raise ValueError("policy probabilities must be finite and nonnegative")
        prior = prior / prior.sum() if prior.sum() > 0 else np.full(len(actions), 1 / len(actions))
        node.priors = dict(zip(actions, map(float, prior)))
        node.visits = {a: 0 for a in actions}
        node.total_value_a = {a: 0.0 for a in actions}
        node.expanded = True
        return _sign(node.state.player) * value

    expand(root)
    if add_root_noise:
        actions = list(root.priors)
        noise = rng.dirichlet(np.full(len(actions), cfg.dirichlet_alpha))
        epsilon = cfg.exploration_fraction
        root.priors = {a: (1 - epsilon) * root.priors[a] + epsilon * float(n)
                       for a, n in zip(actions, noise)}

    for _ in range(cfg.simulations):
        node = root
        path: list[tuple[_Node, int]] = []
        depth = 0
        while node.expanded and not node.state.terminal and depth < cfg.max_depth:
            total = sum(node.visits.values())
            player_sign = _sign(node.state.player)

            def score(a: int) -> float:
                count = node.visits[a]
                q = node.total_value_a[a] / count if count else 0.0
                u = cfg.cpuct * node.priors[a] * math.sqrt(total + 1) / (1 + count)
                return player_sign * q + u

            action = max(node.priors, key=score)
            path.append((node, action))
            if action not in node.children:
                node.children[action] = _Node(transition(node.state, action))
                node_count += 1
            node = node.children[action]
            depth += 1
        deepest = max(deepest, depth)
        if node.state.terminal:
            value_a = expand(node)
        elif not node.expanded:
            value_a = expand(node)
        else:
            # A search-depth cutoff is bootstrapped, never mislabeled a win/draw.
            _, value = evaluate(node.state)
            if not math.isfinite(float(value)) or not -1 <= float(value) <= 1:
                raise ValueError("network value must be finite and in [-1, 1]")
            value_a = _sign(node.state.player) * float(value)
        for ancestor, action in path:
            ancestor.visits[action] += 1
            ancestor.total_value_a[action] += value_a

    visits = dict(root.visits)
    policy = np.zeros(action_dim, dtype=np.float32)
    for action, count in visits.items():
        policy[action] = count / cfg.simulations
    value_a = sum(root.total_value_a.values()) / cfg.simulations
    return SearchResult(visits, policy, _sign(root_state.player) * value_a,
                        cfg.simulations, node_count, deepest)
