"""
src/mcts.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)

A real, domain-agnostic Monte Carlo Tree Search (UCT) engine.

What it does (and what the old "bounded MCTS" in planner.py did not): it builds a search tree with
per-node visit counts and value sums; selects children by UCT; expands new children under progressive
widening; plays RANDOM (prior-weighted, seeded) rollouts to a terminal state; evaluates the terminal
state with a caller-supplied reward; and back-propagates the reward to every node on the path. The best
state ever evaluated is kept (anytime), and the tree statistics are returned for `--debug`.

The engine knows nothing about cells, trees, libraries or prompts: callers supply four callbacks.

    actions(state)      -> list of hashable actions legal in `state` (empty list == terminal)
    step(state, action) -> next state (states must be immutable / hashable-ish; see `state_key`)
    reward(state)       -> float in [0, 1]; higher is better (evaluated once per distinct state, cached)
    prior(state, a)     -> optional float >= 0: heuristic preference used to order expansion and to
                           weight rollout sampling (uniform if omitted)
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple


@dataclass
class _Node:
    state: Any
    parent: Optional["_Node"] = None
    action: Any = None
    children: Dict[Any, "_Node"] = field(default_factory=dict)
    ordered_actions: Optional[List[Any]] = None
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class SearchResult:
    best_state: Any
    best_reward: float
    iterations: int
    evaluations: int                 # distinct states actually scored (the rest were cache hits)
    cache_hits: int
    seed: int
    elapsed_ms: float
    stopped_by: str                  # "iterations" | "time" | "exhausted"
    root_stats: List[Dict[str, Any]]
    best_path_stats: List[Dict[str, Any]]


class MonteCarloTreeSearch:
    def __init__(
        self,
        *,
        actions: Callable[[Any], List[Any]],
        step: Callable[[Any, Any], Any],
        reward: Callable[[Any], float],
        prior: Optional[Callable[[Any, Any], float]] = None,
        state_key: Callable[[Any], Hashable] = lambda s: s,
        describe_action: Callable[[Any], str] = str,
        iterations: int = 64,
        exploration: float = math.sqrt(2.0),
        widening_alpha: float = 0.5,
        max_rollout_depth: int = 64,
        max_seconds: Optional[float] = None,
        seed: int = 0,
    ):
        self._actions, self._step, self._reward, self._prior = actions, step, reward, prior
        self._key, self._describe = state_key, describe_action
        self.iterations, self.c, self.alpha = int(iterations), float(exploration), float(widening_alpha)
        self.max_depth, self.max_seconds, self.seed = int(max_rollout_depth), max_seconds, int(seed)
        self._cache: Dict[Hashable, float] = {}
        self._hits = 0

    # ------------------------------------------------------------------ helpers
    def _score(self, state: Any) -> float:
        k = self._key(state)
        if k in self._cache:
            self._hits += 1
            return self._cache[k]
        v = float(self._reward(state))
        v = 0.0 if v != v else min(1.0, max(0.0, v))
        self._cache[k] = v
        return v

    def _ordered(self, node: _Node) -> List[Any]:
        if node.ordered_actions is None:
            acts = list(self._actions(node.state))
            if self._prior is not None:
                acts.sort(key=lambda a: self._prior(node.state, a), reverse=True)   # stable: ties keep caller order
            node.ordered_actions = acts
        return node.ordered_actions

    def _allowed_children(self, node: _Node) -> int:
        # progressive widening: ceil(n ** alpha), at least 1
        return max(1, math.ceil(max(node.visits, 1) ** self.alpha))

    # ------------------------------------------------------------------ search
    def search(self, root_state: Any) -> SearchResult:
        rng = random.Random(self.seed)
        t0 = time.perf_counter()
        root = _Node(state=root_state)
        best_state, best_reward = root_state, self._score(root_state)
        stopped_by, done = "iterations", 0

        for done in range(1, self.iterations + 1):
            if self.max_seconds is not None and (time.perf_counter() - t0) > self.max_seconds:
                stopped_by, done = "time", done - 1
                break
            # 1. selection
            node = root
            while True:
                acts = self._ordered(node)
                if not acts:
                    break
                expandable = [a for a in acts[: self._allowed_children(node)] if a not in node.children]
                if expandable:
                    # 2. expansion: highest-prior untried action within the widening window
                    a = expandable[0]
                    child = _Node(state=self._step(node.state, a), parent=node, action=a)
                    node.children[a] = child
                    node = child
                    break
                window = [node.children[a] for a in acts[: self._allowed_children(node)] if a in node.children]
                node = max(window, key=lambda ch: ch.mean + self.c * math.sqrt(math.log(max(node.visits, 1) + 1) / (ch.visits or 1)))
            # 3. rollout: random, prior-weighted, seeded
            state = node.state
            for _ in range(self.max_depth):
                acts = list(self._actions(state))
                if not acts:
                    break
                if self._prior is None:
                    a = rng.choice(acts)
                else:
                    w = [max(self._prior(state, x), 0.0) + 1e-9 for x in acts]
                    a = rng.choices(acts, weights=w, k=1)[0]
                state = self._step(state, a)
            r = self._score(state)
            if r > best_reward:
                best_state, best_reward = state, r
            # 4. backpropagation
            cur: Optional[_Node] = node
            while cur is not None:
                cur.visits += 1
                cur.value_sum += r
                cur = cur.parent
        else:
            done = self.iterations

        # best (most-visited) line from the root, for the debug panel
        line, cur = [], root
        while cur.children:
            cur = max(cur.children.values(), key=lambda ch: (ch.visits, ch.mean))
            line.append({"action": self._describe(cur.action), "visits": cur.visits, "mean_reward": round(cur.mean, 4)})
        return SearchResult(
            best_state=best_state, best_reward=best_reward, iterations=done, evaluations=len(self._cache),
            cache_hits=self._hits, seed=self.seed, elapsed_ms=(time.perf_counter() - t0) * 1000.0, stopped_by=stopped_by,
            root_stats=sorted(
                ({"action": self._describe(a), "visits": ch.visits, "mean_reward": round(ch.mean, 4)} for a, ch in root.children.items()),
                key=lambda d: (-d["visits"], -d["mean_reward"])),
            best_path_stats=line,
        )
