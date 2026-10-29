from __future__ import annotations

import math
import time
from typing import List, Sequence

from .base import IncrementalPlanner
from .graph import Graph
from .memory_tracker import MemoryTracker
from .priority_queue import PriorityQueue
from .types import (
    INF,
    INVALID_NODE,
    EdgeChange,
    Key,
    SearchResult,
    approx_equal,
    safe_add,
)


class LPAStar(IncrementalPlanner):
    """
    Lifelong Planning A* (LPA*)
    Reference: Koenig, S., Likhachev, M., & Furcy, D. (2004).
    "Lifelong Planning A*". Artificial Intelligence, 155(1-2), 93-146.
    """

    def __init__(self, graph: Graph, start: int, goal: int) -> None:
        super().__init__(graph, start, goal)
        self._g: List[float] = []
        self._rhs: List[float] = []
        self._backpointer: List[int] = []
        self._open_list: PriorityQueue = PriorityQueue()
        self._initialized: bool = False

    @property
    def name(self) -> str:
        return "LPA*"

    def initialize(self) -> None:
        n = self._graph.num_vertices()
        self._g = [INF] * n
        self._rhs = [INF] * n
        self._backpointer = [INVALID_NODE] * n
        self._open_list.clear()
        self._nodes_recalculated = 0
        self._peak_structural_bytes = 0

        if 0 <= self._start < n:
            self._rhs[self._start] = 0.0
            self._open_list.insert_or_update(
                self._start, self._calculate_key(self._start)
            )
        self._initialized = True
        self._record_peak_memory()

    def _calculate_key(self, u: int) -> Key:
        m = min(self._g[u], self._rhs[u])
        return Key(safe_add(m, self._graph.heuristic(u, self._goal)), m)

    def _update_vertex(self, u: int) -> None:
        self._nodes_recalculated += 1

        if u != self._start:
            best_rhs = INF
            best_pred = INVALID_NODE
            for pred in self._graph.predecessors(u):
                edge_cost = self._graph.get_edge_cost(pred, u)
                cand = safe_add(self._g[pred], edge_cost)
                if cand < best_rhs:
                    best_rhs = cand
                    best_pred = pred
            self._rhs[u] = best_rhs
            self._backpointer[u] = best_pred

        if self._open_list.contains(u):
            self._open_list.remove(u)

        if not approx_equal(self._g[u], self._rhs[u]):
            self._open_list.insert_or_update(u, self._calculate_key(u))
        self._record_peak_memory()

    def _compute_shortest_path(self) -> None:
        goal = self._goal
        while not self._open_list.empty() and (
            self._open_list.top_key() < self._calculate_key(goal)
            or not approx_equal(self._rhs[goal], self._g[goal])
        ):
            u, _ = self._open_list.pop()
            if u == INVALID_NODE:
                break

            if self._g[u] > self._rhs[u]:
                self._g[u] = self._rhs[u]
                for edge in self._graph.successors(u):
                    self._update_vertex(edge.to)
            else:
                self._g[u] = INF
                self._update_vertex(u)
                for edge in self._graph.successors(u):
                    self._update_vertex(edge.to)
        self._record_peak_memory()

    def compute_initial_path(self) -> SearchResult:
        if not self._initialized:
            self.initialize()
        self._nodes_recalculated = 0
        self._peak_structural_bytes = self.structural_memory_bytes()

        t0 = time.perf_counter()
        self._compute_shortest_path()
        t1 = time.perf_counter()

        cost = self.current_path_cost()
        found = not math.isinf(cost)
        path = self.extract_path() if found else []

        return SearchResult(
            algorithm_name=self.name,
            path_found=found,
            path_cost=cost,
            path=path,
            compute_time_ms=(t1 - t0) * 1000.0,
            nodes_recalculated=self._nodes_recalculated,
            memory_mb=MemoryTracker.bytes_to_mb(self._peak_structural_bytes),
            process_rss_mb=MemoryTracker.current_process_rss_mb(),
        )

    def replan(self, changes: Sequence[EdgeChange]) -> SearchResult:
        if not self._initialized:
            return self.compute_initial_path()
        self._nodes_recalculated = 0
        self._peak_structural_bytes = self.structural_memory_bytes()

        n = self._graph.num_vertices()
        t0 = time.perf_counter()
        for change in changes:
            if 0 <= change.v < n:
                self._update_vertex(change.v)
        self._compute_shortest_path()
        t1 = time.perf_counter()

        cost = self.current_path_cost()
        found = not math.isinf(cost)
        path = self.extract_path() if found else []

        return SearchResult(
            algorithm_name=self.name,
            path_found=found,
            path_cost=cost,
            path=path,
            compute_time_ms=(t1 - t0) * 1000.0,
            nodes_recalculated=self._nodes_recalculated,
            memory_mb=MemoryTracker.bytes_to_mb(self._peak_structural_bytes),
            process_rss_mb=MemoryTracker.current_process_rss_mb(),
        )

    def update_start(self, new_start: int) -> None:
        if new_start == self._start:
            return
        self._start = new_start
        self.initialize()

    def current_path_cost(self) -> float:
        if not self._initialized or not (0 <= self._goal < len(self._g)):
            return INF
        return self._g[self._goal]

    def extract_path(self) -> List[int]:
        if math.isinf(self.current_path_cost()):
            return []
        path: List[int] = []
        curr = self._goal
        max_steps = self._graph.num_vertices() + 1
        while curr != INVALID_NODE and max_steps > 0:
            max_steps -= 1
            path.append(curr)
            if curr == self._start:
                path.reverse()
                return path
            best_pred = self._backpointer[curr]
            if best_pred == INVALID_NODE:
                best_cost = INF
                for pred in self._graph.predecessors(curr):
                    c = safe_add(self._g[pred], self._graph.get_edge_cost(pred, curr))
                    if c < best_cost:
                        best_cost = c
                        best_pred = pred
            curr = best_pred
        return []

    def structural_memory_bytes(self) -> int:
        # 8 bytes per double (g, rhs) + 4 bytes per NodeId (backpointer) + PQ bytes
        return (
            len(self._g) * 8
            + len(self._rhs) * 8
            + len(self._backpointer) * 4
            + self._open_list.memory_bytes()
        )

    def _record_peak_memory(self) -> None:
        curr = self.structural_memory_bytes()
        if curr > self._peak_structural_bytes:
            self._peak_structural_bytes = curr
