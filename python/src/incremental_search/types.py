from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering
from typing import List

INVALID_NODE: int = -1
INF: float = float("inf")
EPSILON: float = 1e-9


def approx_equal(a: float, b: float, tol: float = EPSILON) -> bool:
    if math.isinf(a) and math.isinf(b):
        return (a > 0) == (b > 0)
    return abs(a - b) <= tol


def safe_add(a: float, b: float) -> float:
    if math.isinf(a) or math.isinf(b):
        return INF
    return a + b


class HeuristicType(str, Enum):
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    OCTILE = "octile"
    HAVERSINE = "haversine"
    ZERO = "zero"


@total_ordering
@dataclass(frozen=True, slots=True)
class Key:
    """
    Lexicographical two-component priority key [k1, k2] used equivalently
    across D* Lite, LPA*, and Incremental A*.
    """

    k1: float = INF
    k2: float = INF

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Key):
            return NotImplemented
        if not approx_equal(self.k1, other.k1):
            return self.k1 < other.k1
        return self.k2 < other.k2 - EPSILON

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Key):
            return NotImplemented
        return approx_equal(self.k1, other.k1) and approx_equal(self.k2, other.k2)


@dataclass(frozen=True, slots=True)
class EdgeChange:
    """
    Represents a directed edge weight modification or stochastic edge failure.
    """

    u: int
    v: int
    old_cost: float
    new_cost: float


@dataclass(slots=True)
class SearchResult:
    """
    Quantitative metrics recorded per initial search or replanning event:
      1. compute_time_ms: execution time in milliseconds (time.perf_counter)
      2. nodes_recalculated: vertices processed/updated during search/replanning
      3. memory_mb: structural memory footprint in megabytes
      4. process_rss_mb: process resident set size in megabytes (psutil)
    """

    algorithm_name: str
    path_found: bool = False
    path_cost: float = INF
    path: List[int] = field(default_factory=list)
    compute_time_ms: float = 0.0
    nodes_recalculated: int = 0
    memory_mb: float = 0.0
    process_rss_mb: float = 0.0
