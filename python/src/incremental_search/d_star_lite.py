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


class DStarLite(IncrementalPlanner):
    """
    D* Lite (Optimized version)
    Reference: Koenig, S., & Likhachev, M. (2002).
    "D* Lite". AAAI/IAAI, 15, 476-483.
    """

    def __init__(self, graph: Graph, start: int, goal: int) -> None:
        super().__init__(graph, start, goal)
        self._g: List[float] = []
        self._rhs: List[float] = []
        self._open_list: PriorityQueue = PriorityQueue()
        self._last_start: int = start
        self._km: float = 0.0
        self._initialized: bool = False

    @property
    def name(self) -> str:
        return "D* Lite"

    def initialize(self) -> None:
        n = self._graph.num_vertices()
        self._g = [INF] * n
        self._rhs = [INF] * n
        self._open_list.clear()
        self._km = 0.0
        self._last_start = self._start
        self._nodes_recalculated = 0
        self._peak_structural_bytes = 0

        if 0 <= self._goal < n:
            self._rhs[self._goal] = 0.0
            self._open_list.insert_or_update(
                self._goal, self._calculate_key(self._goal)
            )
        self._initialized = True
        self._record_peak_memory()

    def _calculate_key(self, u: int) -> Key:
        m = min(self._g[u], self._rhs[u])
        return Key(
            safe_add(safe_add(m, self._graph.heuristic(self._start, u)), self._km),
            m,
        )

    def _update_vertex(self, u: int) -> None:
        self._nodes_recalculated += 1

        if u != self._goal:
            best_rhs = INF
            for edge in self._graph.successors(u):
                cand = safe_add(edge.cost, self._g[edge.to])
                if cand < best_rhs:
                    best_rhs = cand
            self._rhs[u] = best_rhs

        if self._open_list.contains(u):
            self._open_list.remove(u)

        if not approx_equal(self._g[u], self._rhs[u]):
            self._open_list.insert_or_update(u, self._calculate_key(u))
        self._record_peak_memory()

    def _compute_shortest_path(self) -> None:
        s = self._start
        while not self._open_list.empty() and (
            self._open_list.top_key() < self._calculate_key(s)
            or self._rhs[s] > self._g[s]
            or not approx_equal(self._rhs[s], self._g[s])
        ):
            u, k_old = self._open_list.pop()
            if u == INVALID_NODE:
                break
            k_new = self._calculate_key(u)

            if k_old < k_new:
                self._open_list.insert_or_update(u, k_new)
            elif self._g[u] > self._rhs[u]:
                self._g[u] = self._rhs[u]
                for pred in self._graph.predecessors(u):
                    self._update_vertex(pred)
            else:
                self._g[u] = INF
                self._update_vertex(u)
                for pred in self._graph.predecessors(u):
                    self._update_vertex(pred)
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
        if self._last_start != self._start:
            self._km += self._graph.heuristic(self._last_start, self._start)
            self._last_start = self._start

        for change in changes:
            if 0 <= change.u < n:
                self._update_vertex(change.u)
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
        self._start = new_start

    def current_path_cost(self) -> float:
        if not self._initialized or not (0 <= self._start < len(self._g)):
            return INF
        return self._g[self._start]

    def extract_path(self) -> List[int]:
        if math.isinf(self.current_path_cost()):
            return []
        path: List[int] = []
        curr = self._start
        max_steps = self._graph.num_vertices() + 1
        while curr != INVALID_NODE and max_steps > 0:
            max_steps -= 1
            path.append(curr)
            if curr == self._goal:
                return path
            best_succ = INVALID_NODE
            best_val = INF
            for edge in self._graph.successors(curr):
                cand = safe_add(edge.cost, self._g[edge.to])
                if cand < best_val:
                    best_val = cand
                    best_succ = edge.to
            if best_succ == INVALID_NODE or math.isinf(best_val):
                break
            curr = best_succ
        return []

    def structural_memory_bytes(self) -> int:
        # 8 bytes per double (g, rhs) + PQ bytes
        return (
            len(self._g) * 8
            + len(self._rhs) * 8
            + self._open_list.memory_bytes()
        )

    def _record_peak_memory(self) -> None:
        curr = self.structural_memory_bytes()
        if curr > self._peak_structural_bytes:
            self._peak_structural_bytes = curr
