from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import urllib.parse
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from incremental_search import DStarLite, Graph, GraphIO, HeuristicType, MovingAIMap


def download_osm_via_overpass_api(
    bbox: Tuple[float, float, float, float],
    timeout_s: int = 30,
) -> Optional[Graph]:
    """
    Downloads a real drivable road network directly from the OpenStreetMap Overpass API
    using only Python's standard library (urllib.request), projects coordinates to
    planar meters, and extracts the largest connected component.
    bbox: (south_lat, west_lon, north_lat, east_lon)
    """
    south, west, north, east = bbox
    query = (
        f'[out:json][timeout:{timeout_s}];'
        f'(way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified)$"]'
        f'({south},{west},{north},{east}););'
        f'(._;>;);out body;'
    )
    url = "https://overpass-api.de/api/interpreter"
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"User-Agent": "IncrementalSearchResearch/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"  [Overpass API warning] Could not download bbox {bbox}: {exc}")
        return None

    elements = payload.get("elements", [])
    nodes_latlon: Dict[int, Tuple[float, float]] = {}
    ways: List[Tuple[List[int], bool]] = []

    for el in elements:
        if el.get("type") == "node":
            nodes_latlon[int(el["id"])] = (float(el["lat"]), float(el["lon"]))
        elif el.get("type") == "way":
            nd_list = [int(x) for x in el.get("nodes", [])]
            tags = el.get("tags", {})
            oneway = tags.get("oneway", "no") in ("yes", "1", "true")
            if len(nd_list) >= 2:
                ways.append((nd_list, oneway))

    if not nodes_latlon or not ways:
        return None

    ref_lat = 0.5 * (south + north)
    meters_per_deg_lat = 111132.92
    meters_per_deg_lon = 111412.84 * math.cos(math.radians(ref_lat))

    osm_id_to_vid: Dict[int, int] = {}
    raw_graph = Graph(0, HeuristicType.EUCLIDEAN)

    for nd_list, _ in ways:
        for osm_id in nd_list:
            if osm_id not in osm_id_to_vid and osm_id in nodes_latlon:
                lat, lon = nodes_latlon[osm_id]
                x_m = (lon - west) * meters_per_deg_lon
                y_m = (lat - south) * meters_per_deg_lat
                vid = raw_graph.add_vertex(x_m, y_m)
                osm_id_to_vid[osm_id] = vid

    for nd_list, oneway in ways:
        for i in range(len(nd_list) - 1):
            u_osm, v_osm = nd_list[i], nd_list[i + 1]
            if u_osm in osm_id_to_vid and v_osm in osm_id_to_vid:
                u = osm_id_to_vid[u_osm]
                v = osm_id_to_vid[v_osm]
                if u == v:
                    continue
                cu = raw_graph.get_coordinates(u)
                cv = raw_graph.get_coordinates(v)
                dist_m = max(0.5, math.hypot(cu.x - cv.x, cu.y - cv.y))
                raw_graph.add_edge(u, v, dist_m, bidirectional=not oneway)

    lcc = GraphIO.extract_largest_connected_component(raw_graph)
    return lcc if lcc.num_vertices() >= 20 else None


def _write_movingai_map_and_scen(
    map_filepath: str,
    scen_filepath: str,
    grid_rows: List[str],
    num_scenarios: int,
    seed: int,
) -> MovingAIMap:
    os.makedirs(os.path.dirname(os.path.abspath(map_filepath)), exist_ok=True)
    height = len(grid_rows)
    width = len(grid_rows[0])
    map_filename = os.path.basename(map_filepath)

    with open(map_filepath, "w", encoding="utf-8") as f:
        f.write("type octile\n")
        f.write(f"height {height}\n")
        f.write(f"width {width}\n")
        f.write("map\n")
        for row in grid_rows:
            f.write(f"{row}\n")

    mai_map = GraphIO.load_movingai_map(map_filepath, connectivity=8)
    passable_coords: List[Tuple[int, int, int]] = []
    for y in range(height):
        for x in range(width):
            nid = mai_map.node_at(x, y)
            if nid >= 0:
                passable_coords.append((x, y, nid))

    rng = random.Random(seed)
    scenarios_written = 0
    lines: List[str] = ["version 1.0"]

    attempts = 0
    while scenarios_written < num_scenarios and attempts < num_scenarios * 25:
        attempts += 1
        sx, sy, s_node = rng.choice(passable_coords)
        gx, gy, g_node = rng.choice(passable_coords)
        if s_node == g_node:
            continue
        if not GraphIO.is_reachable(mai_map.graph, s_node, g_node):
            continue
        planner = DStarLite(mai_map.graph, s_node, g_node)
        res = planner.compute_initial_path()
        if not res.path_found:
            continue
        bucket = int(res.path_cost // 4.0)
        lines.append(
            f"{bucket}\t{map_filename}\t{width}\t{height}\t"
            f"{sx}\t{sy}\t{gx}\t{gy}\t{res.path_cost:.8f}"
        )
        scenarios_written += 1

    with open(scen_filepath, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")

    return mai_map


def _build_rooms_grid(width: int = 48, height: int = 48) -> List[str]:
    grid = [["." for _ in range(width)] for _ in range(height)]
    room_size = 12
    for y in range(height):
        for x in range(width):
            if (x % room_size == 0 and x > 0) or (y % room_size == 0 and y > 0):
                grid[y][x] = "@"
    for ry in range(0, height, room_size):
        for rx in range(0, width, room_size):
            door_offset = room_size // 2
            if rx > 0:
                for dy in (-1, 0, 1):
                    if 0 <= ry + door_offset + dy < height:
                        grid[ry + door_offset + dy][rx] = "."
            if ry > 0:
                for dx in (-1, 0, 1):
                    if 0 <= rx + door_offset + dx < width:
                        grid[ry][rx + door_offset + dx] = "."
    return ["".join(row) for row in grid]


def _build_game_obstacles_grid(
    width: int = 48, height: int = 48, seed: int = 202
) -> List[str]:
    rng = random.Random(seed)
    grid = [["." for _ in range(width)] for _ in range(height)]
    for _ in range(18):
        cx = rng.randint(4, width - 5)
        cy = rng.randint(4, height - 5)
        rw = rng.randint(2, 4)
        rh = rng.randint(2, 4)
        ch = "@" if rng.random() < 0.5 else "T"
        for y in range(max(1, cy - rh), min(height - 1, cy + rh)):
            for x in range(max(1, cx - rw), min(width - 1, cx + rw)):
                grid[y][x] = ch
    return ["".join(row) for row in grid]


def _build_braid_maze_grid(
    width: int = 48, height: int = 48, seed: int = 303
) -> List[str]:
    rng = random.Random(seed)
    grid = [["." for _ in range(width)] for _ in range(height)]
    for y in range(2, height - 2, 4):
        for x in range(2, width - 2, 4):
            grid[y][x] = "@"
            dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            rng.shuffle(dirs)
            for dx, dy in dirs[:2]:
                if 1 <= x + dx < width - 1 and 1 <= y + dy < height - 1:
                    grid[y + dy][x + dx] = "@"
    return ["".join(row) for row in grid]


def prepare_all_datasets(
    base_data_dir: str = "data", download_online: bool = False
) -> None:
    osm_dir = os.path.join(base_data_dir, "osm")
    mai_dir = os.path.join(base_data_dir, "movingai")
    syn_dir = os.path.join(base_data_dir, "synthetic")

    os.makedirs(osm_dir, exist_ok=True)
    os.makedirs(mai_dir, exist_ok=True)
    os.makedirs(syn_dir, exist_ok=True)

    # 1. OpenStreetMap Multi-Scale Road Networks (Small, Medium, Large)
    osm_scales = [
        ("osm_urban_small.graph", 15, 15, 101, (20.134, -101.182, 20.142, -101.172)),
        ("osm_urban_medium.graph", 28, 28, 102, (20.128, -101.188, 20.146, -101.168)),
        ("osm_urban_large.graph", 42, 42, 103, (20.120, -101.196, 20.152, -101.160)),
    ]
    for fname, bx, by, seed, bbox in osm_scales:
        g: Optional[Graph] = None
        if download_online:
            print(f"[OSM] Attempting live Overpass API download for {fname} {bbox}...")
            g = download_osm_via_overpass_api(bbox)
        if g is None:
            g = GraphIO.load_or_generate_osm_network(
                place_query=None, fallback_blocks_x=bx, fallback_blocks_y=by, seed=seed
            )
        out_path = os.path.join(osm_dir, fname)
        GraphIO.save_graph(g, out_path)
        print(
            f"[OSM]       Saved {fname:<26} | V={g.num_vertices():<5d} E={g.num_edges():<6d}"
        )

    # 2. Moving AI Benchmark Maps (.map) and Scenarios (.scen)
    mai_maps = [
        ("arena_rooms.map", _build_rooms_grid(48, 48), 35, 201),
        ("dungeon_game.map", _build_game_obstacles_grid(48, 48, 202), 35, 202),
        ("labyrinth_maze.map", _build_braid_maze_grid(48, 48, 303), 35, 303),
    ]
    for map_name, rows, n_scen, seed in mai_maps:
        map_path = os.path.join(mai_dir, map_name)
        scen_path = os.path.join(mai_dir, f"{map_name}.scen")
        mdata = _write_movingai_map_and_scen(map_path, scen_path, rows, n_scen, seed)
        print(
            f"[MovingAI]  Saved {map_name:<26} | V={mdata.graph.num_vertices():<5d} "
            f"E={mdata.graph.num_edges():<6d} Scenarios={n_scen}"
        )

    # 3. Synthetic Controlled Graphs
    syn_configs = [
        (
            "syn_grid_low_density.graph",
            GraphIO.generate_synthetic_grid(35, 35, 0.10, 8, 401, 1.0, 2.5),
        ),
        (
            "syn_grid_high_density.graph",
            GraphIO.generate_synthetic_grid(35, 35, 0.25, 8, 402, 1.0, 2.5),
        ),
        (
            "syn_random_geometric.graph",
            GraphIO.generate_synthetic_random_graph(600, 6, 403, 1000.0),
        ),
    ]
    for fname, g in syn_configs:
        out_path = os.path.join(syn_dir, fname)
        GraphIO.save_graph(g, out_path)
        print(
            f"[Synthetic] Saved {fname:<26} | V={g.num_vertices():<5d} E={g.num_edges():<6d}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare or download OpenStreetMap, Moving AI, and Synthetic datasets."
    )
    parser.add_argument(
        "--data-dir", default="data", help="Target directory for datasets (default: data)"
    )
    parser.add_argument(
        "--download-online",
        action="store_true",
        help="Download live OpenStreetMap road networks from the Overpass API",
    )
    args = parser.parse_args()
    prepare_all_datasets(args.data_dir, download_online=args.download_online)
