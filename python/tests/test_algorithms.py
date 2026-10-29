from __future__ import annotations

import math
import random
from typing import List

from incremental_search import (
    DStarLite,
    Graph,
    HeuristicType,
    IncrementalAStar,
    LPAStar,
    approx_equal,
)


def create_random_geometric_graph(width: int, height: int, seed: int) -> Graph:
    rng = random.Random(seed)
    g = Graph(width * height, HeuristicType.EUCLIDEAN)
    for y in range(height):
        for x in range(width):
            u = y * width + x
            g.set_coordinates(u, float(x), float(y))
            if x + 1 < width:
                v = y * width + (x + 1)
                w = rng.uniform(1.0, 5.0)
                g.add_edge(u, v, w, bidirectional=True)
            if y + 1 < height:
                v = (y + 1) * width + x
                w = rng.uniform(1.0, 5.0)
                g.add_edge(u, v, w, bidirectional=True)
            if x + 1 < width and y + 1 < height:
                v = (y + 1) * width + (x + 1)
                w = rng.uniform(1.0, 5.0) * math.sqrt(2.0)
                g.add_edge(u, v, w, bidirectional=True)
    return g


def verify_path_validity(graph: Graph, path: List[int], expected_cost: float) -> None:
    assert len(path) >= 1
    total = 0.0
    for i in range(len(path) - 1):
        c = graph.get_edge_cost(path[i], path[i + 1])
        assert not math.isinf(c), f"Edge ({path[i]}, {path[i + 1]}) in path is INF"
        total += c
    assert approx_equal(total, expected_cost, tol=1e-6)


def test_equivalence_across_multiple_seeds() -> None:
    for seed in range(1, 11):
        g_d = create_random_geometric_graph(12, 12, seed)
        g_l = g_d.clone()
        g_a = g_d.clone()

        start = 0
        goal = 12 * 12 - 1

        dstar = DStarLite(g_d, start, goal)
        lpa = LPAStar(g_l, start, goal)
        inc_a = IncrementalAStar(g_a, start, goal)

        res_d = dstar.compute_initial_path()
        res_l = lpa.compute_initial_path()
        res_a = inc_a.compute_initial_path()

        assert res_d.path_found and res_l.path_found and res_a.path_found
        assert approx_equal(res_d.path_cost, res_l.path_cost)
        assert approx_equal(res_d.path_cost, res_a.path_cost)

        verify_path_validity(g_d, res_d.path, res_d.path_cost)
        verify_path_validity(g_l, res_l.path, res_l.path_cost)
        verify_path_validity(g_a, res_a.path, res_a.path_cost)

        rng = random.Random(seed + 100)
        for _ in range(8):
            if not res_d.path_found or len(res_d.path) < 2:
                break
            idx = rng.randint(0, len(res_d.path) - 2)
            u = res_d.path[idx]
            v = res_d.path[idx + 1]

            cd1 = g_d.fail_edge(u, v)
            cd2 = g_d.fail_edge(v, u)
            cl1 = g_l.fail_edge(u, v)
            cl2 = g_l.fail_edge(v, u)
            ca1 = g_a.fail_edge(u, v)
            ca2 = g_a.fail_edge(v, u)

            res_d = dstar.replan([cd1, cd2])
            res_l = lpa.replan([cl1, cl2])
            res_a = inc_a.replan([ca1, ca2])

            assert res_d.path_found == res_l.path_found == res_a.path_found
            assert approx_equal(res_d.path_cost, res_l.path_cost)
            assert approx_equal(res_d.path_cost, res_a.path_cost)

            if res_d.path_found:
                verify_path_validity(g_d, res_d.path, res_d.path_cost)
                verify_path_validity(g_l, res_l.path, res_l.path_cost)
                verify_path_validity(g_a, res_a.path, res_a.path_cost)


def test_complete_disconnection() -> None:
    g_d = Graph(4, HeuristicType.EUCLIDEAN)
    for i in range(4):
        g_d.set_coordinates(i, float(i), 0.0)
    g_d.add_edge(0, 1, 1.0, True)
    g_d.add_edge(1, 2, 1.0, True)
    g_d.add_edge(2, 3, 1.0, True)

    g_l = g_d.clone()
    g_a = g_d.clone()

    dstar = DStarLite(g_d, 0, 3)
    lpa = LPAStar(g_l, 0, 3)
    inc_a = IncrementalAStar(g_a, 0, 3)

    assert approx_equal(dstar.compute_initial_path().path_cost, 3.0)
    assert approx_equal(lpa.compute_initial_path().path_cost, 3.0)
    assert approx_equal(inc_a.compute_initial_path().path_cost, 3.0)

    # Sever bridge (1, 2)
    r_d = dstar.replan([g_d.fail_edge(1, 2), g_d.fail_edge(2, 1)])
    r_l = lpa.replan([g_l.fail_edge(1, 2), g_l.fail_edge(2, 1)])
    r_a = inc_a.replan([g_a.fail_edge(1, 2), g_a.fail_edge(2, 1)])

    assert not r_d.path_found and math.isinf(r_d.path_cost)
    assert not r_l.path_found and math.isinf(r_l.path_cost)
    assert not r_a.path_found and math.isinf(r_a.path_cost)


def test_bottleneck_and_restoration_equivalence() -> None:
    for seed in range(20, 45):
        g_d = create_random_geometric_graph(10, 10, seed)
        g_l = g_d.clone()
        g_a = g_d.clone()

        start, goal = 0, 99
        dstar = DStarLite(g_d, start, goal)
        lpa = LPAStar(g_l, start, goal)
        inc_a = IncrementalAStar(g_a, start, goal)

        res_d = dstar.compute_initial_path()
        res_l = lpa.compute_initial_path()
        res_a = inc_a.compute_initial_path()

        rng = random.Random(seed + 500)
        failed_edges = []
        for step in range(15):
            # Target edges on IncrementalAStar's current path specifically
            target_path = res_a.path if (step % 2 == 0 and len(res_a.path) >= 2) else res_d.path
            if len(target_path) >= 2:
                idx = rng.randint(0, len(target_path) - 2)
                u, v = target_path[idx], target_path[idx + 1]
                failed_edges.append((u, v))
                changes_d = [g_d.fail_edge(u, v), g_d.fail_edge(v, u)]
                changes_l = [g_l.fail_edge(u, v), g_l.fail_edge(v, u)]
                changes_a = [g_a.fail_edge(u, v), g_a.fail_edge(v, u)]
            elif failed_edges:
                u, v = failed_edges.pop(0)
                changes_d = [g_d.restore_edge(u, v), g_d.restore_edge(v, u)]
                changes_l = [g_l.restore_edge(u, v), g_l.restore_edge(v, u)]
                changes_a = [g_a.restore_edge(u, v), g_a.restore_edge(v, u)]
            else:
                break

            res_d = dstar.replan(changes_d)
            res_l = lpa.replan(changes_l)
            res_a = inc_a.replan(changes_a)

            assert res_d.path_found == res_l.path_found == res_a.path_found
            assert approx_equal(res_d.path_cost, res_l.path_cost)
            assert approx_equal(res_d.path_cost, res_a.path_cost)

