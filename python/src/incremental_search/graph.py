from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Dict, List

from .types import EdgeChange, HeuristicType, INF

_EARTH_RADIUS_METERS = 6371000.0
_DEG_TO_RAD = math.pi / 180.0
_SQRT2_MINUS_1 = 0.41421356237309515


@dataclass(slots=True)
class Edge:
    to: int
    cost: float
    original_cost: float
    failed: bool = False


@dataclass(slots=True)
class Coordinates:
    x: float = 0.0
    y: float = 0.0


class Graph:
    """
    Directed weighted graph supporting dynamic edge failures and restorations,
    fast forward (successors) and backward (predecessors) queries, and
    consistent heuristic evaluation. Mirrors C++ incremental_search::Graph.
    """

    def __init__(
        self,
        num_vertices: int = 0,
        heuristic_type: HeuristicType = HeuristicType.EUCLIDEAN,
    ) -> None:
        self._adj: List[List[Edge]] = [[] for _ in range(num_vertices)]
        self._rev_adj: List[List[int]] = [[] for _ in range(num_vertices)]
        self._coords: List[Coordinates] = [Coordinates() for _ in range(num_vertices)]
        self._num_edges: int = 0
        self._heuristic_type: HeuristicType = heuristic_type

    def _ensure_vertex(self, u: int) -> None:
        if u < 0:
            raise ValueError("Vertex ID must be non-negative")
        while u >= len(self._adj):
            self._adj.append([])
            self._rev_adj.append([])
            self._coords.append(Coordinates())

    def add_vertex(self, x: float = 0.0, y: float = 0.0) -> int:
        vid = len(self._adj)
        self._adj.append([])
        self._rev_adj.append([])
        self._coords.append(Coordinates(x, y))
        return vid

    def set_coordinates(self, u: int, x: float, y: float) -> None:
        self._ensure_vertex(u)
        self._coords[u] = Coordinates(x, y)

    def get_coordinates(self, u: int) -> Coordinates:
        if 0 <= u < len(self._coords):
            return self._coords[u]
        return Coordinates(0.0, 0.0)

    def add_edge(
        self, u: int, v: int, cost: float, bidirectional: bool = False
    ) -> None:
        self._ensure_vertex(max(u, v))
        out_edges = self._adj[u]
        for e in out_edges:
            if e.to == v:
                if cost < e.original_cost:
                    e.cost = cost
                    e.original_cost = cost
                    e.failed = math.isinf(cost)
                if bidirectional:
                    self.add_edge(v, u, cost, False)
                return

        out_edges.append(Edge(to=v, cost=cost, original_cost=cost, failed=math.isinf(cost)))
        self._rev_adj[v].append(u)
        self._num_edges += 1

        if bidirectional:
            self.add_edge(v, u, cost, False)

    def has_edge(self, u: int, v: int) -> bool:
        if not (0 <= u < len(self._adj)):
            return False
        return any(e.to == v for e in self._adj[u])

    def get_edge_cost(self, u: int, v: int) -> float:
        if not (0 <= u < len(self._adj)):
            return INF
        for e in self._adj[u]:
            if e.to == v:
                return e.cost
        return INF

    def get_original_edge_cost(self, u: int, v: int) -> float:
        if not (0 <= u < len(self._adj)):
            return INF
        for e in self._adj[u]:
            if e.to == v:
                return e.original_cost
        return INF

    def update_edge_cost(self, u: int, v: int, new_cost: float) -> EdgeChange:
        if 0 <= u < len(self._adj):
            for e in self._adj[u]:
                if e.to == v:
                    old_cost = e.cost
                    e.cost = new_cost
                    e.failed = math.isinf(new_cost)
                    return EdgeChange(u=u, v=v, old_cost=old_cost, new_cost=new_cost)
        return EdgeChange(u=u, v=v, old_cost=INF, new_cost=INF)

    def fail_edge(self, u: int, v: int) -> EdgeChange:
        return self.update_edge_cost(u, v, INF)

    def restore_edge(self, u: int, v: int) -> EdgeChange:
        if 0 <= u < len(self._adj):
            for e in self._adj[u]:
                if e.to == v:
                    old_cost = e.cost
                    e.cost = e.original_cost
                    e.failed = False
                    return EdgeChange(u=u, v=v, old_cost=old_cost, new_cost=e.cost)
        return EdgeChange(u=u, v=v, old_cost=INF, new_cost=INF)

    def restore_all_edges(self) -> None:
        for out_edges in self._adj:
            for e in out_edges:
                e.cost = e.original_cost
                e.failed = False

    def successors(self, u: int) -> List[Edge]:
        if 0 <= u < len(self._adj):
            return self._adj[u]
        return []

    def predecessors(self, u: int) -> List[int]:
        if 0 <= u < len(self._rev_adj):
            return self._rev_adj[u]
        return []

    def heuristic(self, u: int, v: int) -> float:
        if u == v or self._heuristic_type == HeuristicType.ZERO:
            return 0.0
        cu = self.get_coordinates(u)
        cv = self.get_coordinates(v)
        dx = abs(cu.x - cv.x)
        dy = abs(cu.y - cv.y)

        if self._heuristic_type == HeuristicType.MANHATTAN:
            return dx + dy
        if self._heuristic_type == HeuristicType.OCTILE:
            return max(dx, dy) + _SQRT2_MINUS_1 * min(dx, dy)
        if self._heuristic_type == HeuristicType.HAVERSINE:
            lat1 = cu.y * _DEG_TO_RAD
            lat2 = cv.y * _DEG_TO_RAD
            dlat = (cv.y - cu.y) * _DEG_TO_RAD
            dlon = (cv.x - cu.x) * _DEG_TO_RAD
            a = (
                math.sin(dlat * 0.5) ** 2
                + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon * 0.5) ** 2)
            )
            c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
            return _EARTH_RADIUS_METERS * c
        return math.hypot(dx, dy)

    @property
    def heuristic_type(self) -> HeuristicType:
        return self._heuristic_type

    @heuristic_type.setter
    def heuristic_type(self, value: HeuristicType) -> None:
        self._heuristic_type = value

    def num_vertices(self) -> int:
        return len(self._adj)

    def num_edges(self) -> int:
        return self._num_edges

    def clone(self) -> Graph:
        g = Graph(self.num_vertices(), self._heuristic_type)
        for u in range(self.num_vertices()):
            c = self._coords[u]
            g.set_coordinates(u, c.x, c.y)
        for u in range(self.num_vertices()):
            for e in self._adj[u]:
                g.add_edge(u, e.to, e.original_cost, False)
                if e.cost != e.original_cost:
                    g.update_edge_cost(u, e.to, e.cost)
        return g

    def memory_bytes(self) -> int:
        total = sys.getsizeof(self) + sys.getsizeof(self._coords)
        for vec in self._adj:
            total += sys.getsizeof(vec) + len(vec) * 32
        for vec in self._rev_adj:
            total += sys.getsizeof(vec) + len(vec) * 8
        return total
