from __future__ import annotations

import os
import random
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from incremental_search import DStarLite, GraphIO, MovingAIMap


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
    # Carve 3-cell wide doorways in every room wall
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
    # Place rectangular and circular obstacle clusters (trees 'T' and walls '@')
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


def prepare_all_datasets(base_data_dir: str = "data") -> None:
    osm_dir = os.path.join(base_data_dir, "osm")
    mai_dir = os.path.join(base_data_dir, "movingai")
    syn_dir = os.path.join(base_data_dir, "synthetic")

    os.makedirs(osm_dir, exist_ok=True)
    os.makedirs(mai_dir, exist_ok=True)
    os.makedirs(syn_dir, exist_ok=True)

    # 1. OpenStreetMap Multi-Scale Road Networks (Small, Medium, Large)
    osm_scales = [
        ("osm_urban_small.graph", 15, 15, 101),
        ("osm_urban_medium.graph", 28, 28, 102),
        ("osm_urban_large.graph", 42, 42, 103),
    ]
    for fname, bx, by, seed in osm_scales:
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
    prepare_all_datasets()
