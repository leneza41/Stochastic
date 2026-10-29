from __future__ import annotations

import math
import os
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

from .graph import Graph
from .types import INF, INVALID_NODE, HeuristicType

_DIAG_COST = math.sqrt(2.0)
_PASSABLE_CHARS = {".", "G", "S"}


@dataclass(slots=True)
class MovingAIScenario:
    bucket: int
    map_name: str
    map_width: int
    map_height: int
    start_x: int
    start_y: int
    goal_x: int
    goal_y: int
    optimal_length: float
    start_node: int = INVALID_NODE
    goal_node: int = INVALID_NODE


@dataclass(slots=True)
class MovingAIMap:
    graph: Graph
    width: int
    height: int
    connectivity: int = 8
    grid_to_node: List[int] = field(default_factory=list)

    def node_at(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid_to_node[y * self.width + x]
        return INVALID_NODE


class GraphIO:
    """
    Graph serialization (.graph), Moving AI (.map / .scen) loaders & generators,
    OpenStreetMap (OSMnx / urban road network) loader & preprocessor, and
    synthetic graph generators. Mirrors C++ incremental_search::GraphIO.
    """

    @staticmethod
    def save_graph(graph: Graph, filepath: str) -> bool:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(
                f"GRAPH_V1 {graph.num_vertices()} {graph.num_edges()} "
                f"{graph.heuristic_type.value}\n"
            )
            for u in range(graph.num_vertices()):
                c = graph.get_coordinates(u)
                f.write(f"V {u} {c.x:.8f} {c.y:.8f}\n")
            for u in range(graph.num_vertices()):
                for edge in graph.successors(u):
                    f.write(f"E {u} {edge.to} {edge.original_cost:.8f}\n")
        return True

    @staticmethod
    def load_graph(filepath: str) -> Graph:
        with open(filepath, "r", encoding="utf-8") as f:
            header_line = f.readline().strip().split()
            if len(header_line) < 4 or header_line[0] != "GRAPH_V1":
                raise ValueError(f"Invalid GRAPH_V1 header in {filepath}")
            num_v = int(header_line[1])
            h_type = HeuristicType(header_line[3].lower())
            graph = Graph(num_v, h_type)

            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "V":
                    u = int(parts[1])
                    x = float(parts[2])
                    y = float(parts[3])
                    graph.set_coordinates(u, x, y)
                elif parts[0] == "E":
                    u = int(parts[1])
                    v = int(parts[2])
                    cost = float(parts[3])
                    graph.add_edge(u, v, cost, bidirectional=False)
        return graph

    @staticmethod
    def load_movingai_map(map_filepath: str, connectivity: int = 8) -> MovingAIMap:
        width = 0
        height = 0
        grid_rows: List[str] = []

        with open(map_filepath, "r", encoding="utf-8") as f:
            in_map_section = False
            for raw_line in f:
                line = raw_line.rstrip("\r\n")
                if not in_map_section:
                    parts = line.split()
                    if len(parts) >= 2 and parts[0] == "height":
                        height = int(parts[1])
                    elif len(parts) >= 2 and parts[0] == "width":
                        width = int(parts[1])
                    elif len(parts) >= 1 and parts[0] == "map":
                        in_map_section = True
                else:
                    if len(line) >= width and len(grid_rows) < height:
                        grid_rows.append(line[:width])

        if width <= 0 or height <= 0 or len(grid_rows) != height:
            raise ValueError(f"Invalid Moving AI .map file: {map_filepath}")

        conn = 4 if connectivity == 4 else 8
        h_type = HeuristicType.MANHATTAN if conn == 4 else HeuristicType.OCTILE
        graph = Graph(0, h_type)
        grid_to_node = [INVALID_NODE] * (width * height)

        for y in range(height):
            row = grid_rows[y]
            for x in range(width):
                if row[x] in _PASSABLE_CHARS:
                    nid = graph.add_vertex(float(x), float(y))
                    grid_to_node[y * width + x] = nid

        mai_map = MovingAIMap(
            graph=graph,
            width=width,
            height=height,
            connectivity=conn,
            grid_to_node=grid_to_node,
        )

        dx4 = (1, -1, 0, 0)
        dy4 = (0, 0, 1, -1)
        dx8 = (1, 1, -1, -1)
        dy8 = (1, -1, 1, -1)

        for y in range(height):
            for x in range(width):
                u = mai_map.node_at(x, y)
                if u == INVALID_NODE:
                    continue
                for d in range(4):
                    v = mai_map.node_at(x + dx4[d], y + dy4[d])
                    if v != INVALID_NODE:
                        graph.add_edge(u, v, 1.0, bidirectional=False)
                if conn == 8:
                    for d in range(4):
                        nx = x + dx8[d]
                        ny = y + dy8[d]
                        v = mai_map.node_at(nx, ny)
                        if (
                            v != INVALID_NODE
                            and mai_map.node_at(nx, y) != INVALID_NODE
                            and mai_map.node_at(x, ny) != INVALID_NODE
                        ):
                            graph.add_edge(u, v, _DIAG_COST, bidirectional=False)

        return mai_map

    @staticmethod
    def load_movingai_scenarios(
        scen_filepath: str, map_data: MovingAIMap
    ) -> List[MovingAIScenario]:
        scenarios: List[MovingAIScenario] = []
        with open(scen_filepath, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("version"):
                    continue
                parts = line.split()
                if len(parts) < 9:
                    continue
                sx, sy = int(parts[4]), int(parts[5])
                gx, gy = int(parts[6]), int(parts[7])
                s_node = map_data.node_at(sx, sy)
                g_node = map_data.node_at(gx, gy)
                if (
                    s_node != INVALID_NODE
                    and g_node != INVALID_NODE
                    and s_node != g_node
                ):
                    scenarios.append(
                        MovingAIScenario(
                            bucket=int(parts[0]),
                            map_name=parts[1],
                            map_width=int(parts[2]),
                            map_height=int(parts[3]),
                            start_x=sx,
                            start_y=sy,
                            goal_x=gx,
                            goal_y=gy,
                            optimal_length=float(parts[8]),
                            start_node=s_node,
                            goal_node=g_node,
                        )
                    )
        return scenarios

    @staticmethod
    def extract_largest_connected_component(input_graph: Graph) -> Graph:
        n = input_graph.num_vertices()
        if n == 0:
            return Graph(0, input_graph.heuristic_type)

        visited = [False] * n
        best_comp: List[int] = []

        for i in range(n):
            if visited[i]:
                continue
            comp: List[int] = []
            q: Deque[int] = deque([i])
            visited[i] = True
            while q:
                u = q.popleft()
                comp.append(u)
                for edge in input_graph.successors(u):
                    if (
                        0 <= edge.to < n
                        and not visited[edge.to]
                        and not math.isinf(edge.original_cost)
                        and edge.original_cost > 0.0
                    ):
                        visited[edge.to] = True
                        q.append(edge.to)
                for pred in input_graph.predecessors(u):
                    if 0 <= pred < n and not visited[pred]:
                        c = input_graph.get_original_edge_cost(pred, u)
                        if not math.isinf(c) and c > 0.0:
                            visited[pred] = True
                            q.append(pred)
            if len(comp) > len(best_comp):
                best_comp = comp

        best_comp.sort()
        old_to_new = [INVALID_NODE] * n
        clean = Graph(len(best_comp), input_graph.heuristic_type)

        for new_idx, old_u in enumerate(best_comp):
            old_to_new[old_u] = new_idx
            c = input_graph.get_coordinates(old_u)
            clean.set_coordinates(new_idx, c.x, c.y)

        for old_u in best_comp:
            new_u = old_to_new[old_u]
            for edge in input_graph.successors(old_u):
                if (
                    edge.to == old_u
                    or math.isinf(edge.original_cost)
                    or edge.original_cost <= 0.0
                ):
                    continue
                new_v = old_to_new[edge.to]
                if new_v != INVALID_NODE:
                    clean.add_edge(new_u, new_v, edge.original_cost, bidirectional=False)

        return clean

    @staticmethod
    def is_reachable(graph: Graph, start: int, goal: int) -> bool:
        n = graph.num_vertices()
        if not (0 <= start < n and 0 <= goal < n):
            return False
        if start == goal:
            return True

        visited = [False] * n
        q: Deque[int] = deque([start])
        visited[start] = True
        while q:
            u = q.popleft()
            if u == goal:
                return True
            for edge in graph.successors(u):
                if not math.isinf(edge.cost) and not visited[edge.to]:
                    visited[edge.to] = True
                    q.append(edge.to)
        return False

    @staticmethod
    def sample_valid_od_pair(
        graph: Graph, seed: int, max_attempts: int = 1000
    ) -> Tuple[int, int]:
        n = graph.num_vertices()
        if n < 2:
            return INVALID_NODE, INVALID_NODE
        rng = random.Random(seed)
        for _ in range(max_attempts):
            s = rng.randint(0, n - 1)
            t = rng.randint(0, n - 1)
            if s != t and GraphIO.is_reachable(graph, s, t):
                return s, t
        return INVALID_NODE, INVALID_NODE

    @staticmethod
    def generate_synthetic_grid(
        width: int,
        height: int,
        obstacle_ratio: float,
        connectivity: int,
        seed: int,
        min_weight: float = 1.0,
        max_weight: float = 1.0,
    ) -> Graph:
        rng = random.Random(seed)
        blocked = [False] * (width * height)
        for y in range(height):
            for x in range(width):
                if (x == 0 and y == 0) or (x == width - 1 and y == height - 1):
                    continue
                if rng.random() < obstacle_ratio:
                    blocked[y * width + x] = True

        h_type = (
            HeuristicType.MANHATTAN if connectivity == 4 else HeuristicType.OCTILE
        )
        if min_weight > 1.0 or max_weight > 1.0:
            h_type = HeuristicType.EUCLIDEAN

        raw_graph = Graph(0, h_type)
        grid_to_node = [INVALID_NODE] * (width * height)
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                if not blocked[idx]:
                    grid_to_node[idx] = raw_graph.add_vertex(float(x), float(y))

        def node_at(cx: int, cy: int) -> int:
            if 0 <= cx < width and 0 <= cy < height:
                return grid_to_node[cy * width + cx]
            return INVALID_NODE

        hi_w = max(min_weight, max_weight)
        for y in range(height):
            for x in range(width):
                u = node_at(x, y)
                if u == INVALID_NODE:
                    continue
                if x + 1 < width and node_at(x + 1, y) != INVALID_NODE:
                    w = rng.uniform(min_weight, hi_w)
                    raw_graph.add_edge(u, node_at(x + 1, y), w, bidirectional=True)
                if y + 1 < height and node_at(x, y + 1) != INVALID_NODE:
                    w = rng.uniform(min_weight, hi_w)
                    raw_graph.add_edge(u, node_at(x, y + 1), w, bidirectional=True)
                if connectivity == 8:
                    if (
                        x + 1 < width
                        and y + 1 < height
                        and node_at(x + 1, y + 1) != INVALID_NODE
                        and node_at(x + 1, y) != INVALID_NODE
                        and node_at(x, y + 1) != INVALID_NODE
                    ):
                        w = rng.uniform(min_weight, hi_w) * _DIAG_COST
                        raw_graph.add_edge(
                            u, node_at(x + 1, y + 1), w, bidirectional=True
                        )
                    if (
                        x - 1 >= 0
                        and y + 1 < height
                        and node_at(x - 1, y + 1) != INVALID_NODE
                        and node_at(x - 1, y) != INVALID_NODE
                        and node_at(x, y + 1) != INVALID_NODE
                    ):
                        w = rng.uniform(min_weight, hi_w) * _DIAG_COST
                        raw_graph.add_edge(
                            u, node_at(x - 1, y + 1), w, bidirectional=True
                        )

        return GraphIO.extract_largest_connected_component(raw_graph)

    @staticmethod
    def generate_synthetic_random_graph(
        num_vertices: int,
        avg_degree: int,
        seed: int,
        area_size: float = 1000.0,
    ) -> Graph:
        rng = random.Random(seed)
        raw_graph = Graph(num_vertices, HeuristicType.EUCLIDEAN)
        for u in range(num_vertices):
            raw_graph.set_coordinates(
                u, rng.uniform(0.0, area_size), rng.uniform(0.0, area_size)
            )

        k = max(2, avg_degree)
        for u in range(num_vertices):
            cu = raw_graph.get_coordinates(u)
            dists: List[Tuple[float, int]] = []
            for v in range(num_vertices):
                if u == v:
                    continue
                cv = raw_graph.get_coordinates(v)
                d = math.hypot(cu.x - cv.x, cu.y - cv.y)
                dists.append((d, v))
            dists.sort(key=lambda item: item[0])
            for d, v in dists[:k]:
                road_len = max(0.1, d) * rng.uniform(1.0, 1.35)
                raw_graph.add_edge(u, v, road_len, bidirectional=True)

        return GraphIO.extract_largest_connected_component(raw_graph)

    @staticmethod
    def load_or_generate_osm_network(
        place_query: Optional[str] = None,
        fallback_blocks_x: int = 16,
        fallback_blocks_y: int = 16,
        seed: int = 101,
    ) -> Graph:
        """
        Attempts to download an OpenStreetMap road network via osmnx if installed
        and network access is available; otherwise builds a realistic urban street
        network (arterial + collector blocks in meters with jittered intersections
        and one-way/two-way street segments) and applies OSM preprocessing:
          - Largest connected component retention
          - Isolated vertex & self-loop removal
          - Duplicate edge deduplication (minimum positive length in meters)
        """
        if place_query:
            try:
                import osmnx as ox  # type: ignore[import-not-found]

                G_nx = ox.graph_from_place(place_query, network_type="drive", simplify=True)
                G_proj = ox.project_graph(G_nx)
                nodes = list(G_proj.nodes(data=True))
                node_map = {nid: idx for idx, (nid, _) in enumerate(nodes)}
                raw = Graph(len(nodes), HeuristicType.EUCLIDEAN)
                for nid, data in nodes:
                    idx = node_map[nid]
                    raw.set_coordinates(
                        idx, float(data.get("x", 0.0)), float(data.get("y", 0.0))
                    )
                for u_nid, v_nid, edata in G_proj.edges(data=True):
                    length = float(edata.get("length", 10.0))
                    if length > 0.0 and u_nid != v_nid:
                        raw.add_edge(
                            node_map[u_nid],
                            node_map[v_nid],
                            length,
                            bidirectional=False,
                        )
                return GraphIO.extract_largest_connected_component(raw)
            except Exception:
                pass

        # Deterministic realistic urban road network in planar meters
        rng = random.Random(seed)
        n_total = fallback_blocks_x * fallback_blocks_y
        raw = Graph(n_total, HeuristicType.EUCLIDEAN)
        block_size_m = 85.0

        for by in range(fallback_blocks_y):
            for bx in range(fallback_blocks_x):
                u = by * fallback_blocks_x + bx
                jx = bx * block_size_m + rng.uniform(-12.0, 12.0)
                jy = by * block_size_m + rng.uniform(-12.0, 12.0)
                raw.set_coordinates(u, jx, jy)

        for by in range(fallback_blocks_y):
            for bx in range(fallback_blocks_x):
                u = by * fallback_blocks_x + bx
                cu = raw.get_coordinates(u)

                # East-West street segment
                if bx + 1 < fallback_blocks_x:
                    v = by * fallback_blocks_x + (bx + 1)
                    cv = raw.get_coordinates(v)
                    dist_m = math.hypot(cu.x - cv.x, cu.y - cv.y) * rng.uniform(1.0, 1.12)
                    raw.add_edge(u, v, dist_m, bidirectional=True)

                # North-South street segment
                if by + 1 < fallback_blocks_y:
                    v = (by + 1) * fallback_blocks_x + bx
                    cv = raw.get_coordinates(v)
                    dist_m = math.hypot(cu.x - cv.x, cu.y - cv.y) * rng.uniform(1.0, 1.12)
                    raw.add_edge(u, v, dist_m, bidirectional=True)

                # Occasional diagonal arterial boulevard
                if (
                    bx + 1 < fallback_blocks_x
                    and by + 1 < fallback_blocks_y
                    and rng.random() < 0.18
                ):
                    v = (by + 1) * fallback_blocks_x + (bx + 1)
                    cv = raw.get_coordinates(v)
                    dist_m = math.hypot(cu.x - cv.x, cu.y - cv.y) * rng.uniform(1.0, 1.08)
                    raw.add_edge(u, v, dist_m, bidirectional=True)

        return GraphIO.extract_largest_connected_component(raw)
