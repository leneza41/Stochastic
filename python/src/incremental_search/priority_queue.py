from __future__ import annotations

import heapq
from typing import Dict, List, Tuple

from .types import INF, INVALID_NODE, Key


class PriorityQueue:
    """
    Equivalent priority queue used across D* Lite, LPA*, and Incremental A*.
    Uses a binary min-heap with lazy deletion and an active key dictionary
    so that insert_or_update, remove, contains, and top_key operate under
    identical conditions across all three algorithms.
    """

    def __init__(self) -> None:
        self._heap: List[Tuple[Key, int]] = []
        self._active_keys: Dict[int, Key] = {}

    def clear(self) -> None:
        self._heap.clear()
        self._active_keys.clear()

    def _clean_top(self) -> None:
        while self._heap:
            key, node = self._heap[0]
            active = self._active_keys.get(node)
            if active is not None and active == key:
                break
            heapq.heappop(self._heap)

    def empty(self) -> bool:
        self._clean_top()
        return len(self._active_keys) == 0

    def size(self) -> int:
        return len(self._active_keys)

    def contains(self, u: int) -> bool:
        return u in self._active_keys

    def insert_or_update(self, u: int, key: Key) -> None:
        self._active_keys[u] = key
        heapq.heappush(self._heap, (key, u))

    def remove(self, u: int) -> None:
        self._active_keys.pop(u, None)

    def top_key(self) -> Key:
        self._clean_top()
        if not self._heap:
            return Key(INF, INF)
        return self._heap[0][0]

    def top_node(self) -> int:
        self._clean_top()
        if not self._heap:
            return INVALID_NODE
        return self._heap[0][1]

    def pop(self) -> Tuple[int, Key]:
        self._clean_top()
        if not self._heap:
            return INVALID_NODE, Key(INF, INF)
        key, node = heapq.heappop(self._heap)
        self._active_keys.pop(node, None)
        return node, key

    def memory_bytes(self) -> int:
        # Standardized structural byte footprint matching C++ PriorityQueue
        # HeapEntry (Key=16B + NodeId=4B -> 24B aligned) + Active Table entry (~32B)
        return len(self._heap) * 24 + len(self._active_keys) * 32
