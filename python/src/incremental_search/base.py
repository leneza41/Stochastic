from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from .graph import Graph
from .types import EdgeChange, SearchResult


class IncrementalPlanner(ABC):
    """
    Common abstract interface for all incremental search algorithms
    (D* Lite, LPA*, Incremental A*), ensuring identical execution conditions,
    timing measurement (time.perf_counter), node recalculation accounting,
    and memory profiling.
    """

    def __init__(self, graph: Graph, start: int, goal: int) -> None:
        self._graph: Graph = graph
        self._start: int = start
        self._goal: int = goal
        self._nodes_recalculated: int = 0
        self._peak_structural_bytes: int = 0

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def start(self) -> int:
        return self._start

    @property
    def goal(self) -> int:
        return self._goal

    @abstractmethod
    def initialize(self) -> None:
        ...

    @abstractmethod
    def compute_initial_path(self) -> SearchResult:
        ...

    @abstractmethod
    def replan(self, changes: Sequence[EdgeChange]) -> SearchResult:
        ...

    @abstractmethod
    def update_start(self, new_start: int) -> None:
        ...

    @abstractmethod
    def extract_path(self) -> List[int]:
        ...

    @abstractmethod
    def current_path_cost(self) -> float:
        ...

    @abstractmethod
    def structural_memory_bytes(self) -> int:
        ...
