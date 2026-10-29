from __future__ import annotations

import math
import time
from typing import List, Sequence

from .base import IncrementalPlanner
from .graph import Graph
from .memory_tracker import MemoryTracker
from .priority_queue import PriorityQueue
from .types import (
    EPSILON,
    INF,
    INVALID_NODE,
    EdgeChange,
    Key,
    SearchResult,
    safe_add,
)


class IncrementalAStar(IncrementalPlanner):
    """
    Incremental A* (Dynamic Search-Tree Repair A*)

    Maintains the forward A* search tree (g-values, parent pointers, and
    closed/open status). When stochastic edge failures or cost changes occur:
      - If an edge (u, v) increases in cost (or fails) and parent[v] == u,
        the subtree rooted at v is invalidated and its boundary vertices
        reachable from valid non-invalidated predecessors are re-inserted
        into the priority queue to repair the search tree incrementally.
      - If an edge (u, v) decreases in cost and g[u] + new_cost < g[v],
        v is updated and re-opened in the priority queue.
    """

    def __init__(self, graph: Graph, start: int, goal: int) -> None:
        super().__init__(graph, start, goal)
        self._g: List[float] = []
        self._parent: List[int] = []
        self._closed: List[bool] = []
        self._open_list: PriorityQueue = PriorityQueue()
        self._initialized: bool = False

    @property
    def name(self) -> str:
        return "Incremental A*"

    def initialize(self) -> None:
        n = self._graph.num_vertices()
        self._g = [INF] * n
        self._parent = [INVALID_NODE] * n
        self._closed = [False] * n
        self._open_list.clear()
        self._nodes_recalculated = 0
        self._peak_structural_bytes = 0

        if 0 <= self._start < n:
            self._g[self._start] = 0.0
            self._open_list.insert_or_update(
                self._start, self._calculate_key(self._start)
            )
        self._initialized = True
        self._record_peak_memory()

    def _calculate_key(self, u: int) -> Key:
        gu = self._g[u]
        return Key(safe_add(gu, self._graph.heuristic(u, self._goal)), gu)

    def _invalidate_subtree(self, root: int) -> None:
        n = self._graph.num_vertices()
        if not (0 <= root < n):
            return

        queue: List[int] = [root]
        in_subtree = [False] * n
        in_subtree[root] = True

        added = True
        while added:
            added = False
            for v in range(n):
                p = self._parent[v]
                if not in_subtree[v] and p != INVALID_NODE and in_subtree[p]:
                    in_subtree[v] = True
                    queue.append(v)
                    added = True

        for v in queue:
            self._g[v] = INF
            self._parent[v] = INVALID_NODE
            self._closed[v] = False
            if self._open_list.contains(v):
                self._open_list.remove(v)
            self._nodes_recalculated += 1

        for v in queue:
            best_g = INF
            best_pred = INVALID_NODE
            for pred in self._graph.predecessors(v):
                if (
                    not in_subtree[pred]
                    and self._closed[pred]
                    and not math.isinf(self._g[pred])
                ):
                    cand = safe_add(self._g[pred], self._graph.get_edge_cost(pred, v))
                    if cand < best_g:
                        best_g = cand
                        best_pred = pred
            if best_pred != INVALID_NODE and not math.isinf(best_g):
                self._g[v] = best_g
                self._parent[v] = best_pred
                self._open_list.insert_or_update(v, self._calculate_key(v))
        self._record_peak_memory()

    def _run_a_star_repair(self) -> None:
        goal = self._goal
        while not self._open_list.empty():
            if (
                not math.isinf(self._g[goal])
                and self._closed[goal]
                and not (self._open_list.top_key() < self._calculate_key(goal))
            ):
                break

            u, _ = self._open_list.pop()
            if u == INVALID_NODE:
                break

            self._closed[u] = True
            self._nodes_recalculated += 1

            gu = self._g[u]
            for edge in self._graph.successors(u):
                if math.isinf(edge.cost):
                    continue
                v = edge.to
                tentative_g = safe_add(gu, edge.cost)
                if tentative_g + EPSILON < self._g[v]:
                    self._g[v] = tentative_g
                    self._parent[v] = u
                    self._closed[v] = False
                    self._open_list.insert_or_update(v, self._calculate_key(v))
                    self._nodes_recalculated += 1
        self._record_peak_memory()

    def compute_initial_path(self) -> SearchResult:
        if not self._initialized:
            self.initialize()
        self._nodes_recalculated = 0
        self._peak_structural_bytes = self.structural_memory_bytes()

        t0 = time.perf_counter()
        self._run_a_star_repair()
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
            if not (0 <= change.u < n and 0 <= change.v < n):
                continue
            if change.new_cost > change.old_cost:
                if self._parent[change.v] == change.u:
                    self._invalidate_subtree(change.v)
            elif change.new_cost < change.old_cost:
                cand = safe_add(self._g[change.u], change.new_cost)
                if cand + EPSILON < self._g[change.v]:
                    self._g[change.v] = cand
                    self._parent[change.v] = change.u
                    self._closed[change.v] = False
                    self._open_list.insert_or_update(
                        change.v, self._calculate_key(change.v)
                    )
                    self._nodes_recalculated += 1

        self._run_a_star_repair()
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
            curr = self._parent[curr]
        return []

    def structural_memory_bytes(self) -> int:
        # 8 bytes per double (g) + 4 bytes per NodeId (parent) + 1 byte per bool (closed) + PQ bytes
        return (
            len(self._g) * 8
            + len(self._parent) * 4
            + len(self._closed) * 1
            + self._open_list.memory_bytes()
        )

    def _record_peak_memory(self) -> None:
        curr = self.structural_memory_bytes()
        if curr > self._peak_structural_bytes:
            self._peak_structural_bytes = curr
