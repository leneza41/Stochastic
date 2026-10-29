from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from incremental_search import (
    DStarLite,
    Graph,
    HeuristicType,
    IncrementalAStar,
    LPAStar,
    SearchResult,
    approx_equal,
)


def build_grid_graph(width: int, height: int) -> Graph:
    g = Graph(width * height, HeuristicType.EUCLIDEAN)
    for y in range(height):
        for x in range(width):
            u = y * width + x
            g.set_coordinates(u, float(x), float(y))
            if x + 1 < width:
                v = y * width + (x + 1)
                g.add_edge(u, v, 1.0, bidirectional=True)
            if y + 1 < height:
                v = (y + 1) * width + x
                g.add_edge(u, v, 1.0, bidirectional=True)
    return g


def print_result(phase: str, res: SearchResult) -> None:
    print(
        f"{phase:<14} | {res.algorithm_name:<16} | "
        f"Cost: {res.path_cost:8.2f} | "
        f"Time(ms): {res.compute_time_ms:9.4f} | "
        f"Nodes: {res.nodes_recalculated:7d} | "
        f"StructMem(MB): {res.memory_mb:8.5f}"
    )


def main() -> int:
    width, height = 30, 30
    g_dstar = build_grid_graph(width, height)
    g_lpa = g_dstar.clone()
    g_inc_a = g_dstar.clone()

    start = 0
    goal = width * height - 1

    dstar = DStarLite(g_dstar, start, goal)
    lpa = LPAStar(g_lpa, start, goal)
    inc_a = IncrementalAStar(g_inc_a, start, goal)

    dstar.initialize()
    lpa.initialize()
    inc_a.initialize()

    print("=== Step 1: Python Incremental Search Equivalence Verification ===")
    r_d = dstar.compute_initial_path()
    r_l = lpa.compute_initial_path()
    r_a = inc_a.compute_initial_path()

    print_result("Initial", r_d)
    print_result("Initial", r_l)
    print_result("Initial", r_a)

    if not (
        approx_equal(r_d.path_cost, r_l.path_cost)
        and approx_equal(r_d.path_cost, r_a.path_cost)
    ):
        print("ERROR: Initial path costs do not match!", file=sys.stderr)
        return 1

    rng = random.Random(42)
    for step in range(1, 6):
        current_path = r_d.path
        if len(current_path) < 2:
            break
        idx = rng.randint(0, len(current_path) - 2)
        u = current_path[idx]
        v = current_path[idx + 1]

        c1_d = g_dstar.fail_edge(u, v)
        c2_d = g_dstar.fail_edge(v, u)
        c1_l = g_lpa.fail_edge(u, v)
        c2_l = g_lpa.fail_edge(v, u)
        c1_a = g_inc_a.fail_edge(u, v)
        c2_a = g_inc_a.fail_edge(v, u)

        r_d = dstar.replan([c1_d, c2_d])
        r_l = lpa.replan([c1_l, c2_l])
        r_a = inc_a.replan([c1_a, c2_a])

        label = f"Failure #{step}"
        print_result(label, r_d)
        print_result(label, r_l)
        print_result(label, r_a)

        if not (
            approx_equal(r_d.path_cost, r_l.path_cost)
            and approx_equal(r_d.path_cost, r_a.path_cost)
        ):
            print(
                f"ERROR: Replanned path costs diverged at step {step}!",
                file=sys.stderr,
            )
            return 1

    print(
        "\nAll Python algorithms converged to identical optimal path costs across all failures."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
