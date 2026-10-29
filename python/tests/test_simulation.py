from __future__ import annotations

import os
import tempfile

from incremental_search import (
    FailureMode,
    Graph,
    GraphIO,
    HeuristicType,
    SampleSizeCalculator,
    SimulationConfig,
    StochasticSimulator,
)


def test_sample_size_calculator_formulas() -> None:
    # Matches Section 5.3 illustrative calculation in "Muestra de la Investigación.pdf":
    # S = 4 ms, E = 1 ms, Z = 1.96 -> n_inf = 62, n_finite(N=1000) = 58
    assert SampleSizeCalculator.compute_infinite_population(4.0, 1.0) == 62
    assert SampleSizeCalculator.compute_finite_population(1000, 4.0, 1.0) == 58


def test_largest_connected_component_and_graph_io() -> None:
    raw = Graph(6, HeuristicType.EUCLIDEAN)
    for i in range(6):
        raw.set_coordinates(i, float(i), 0.0)
    raw.add_edge(0, 1, 2.5, bidirectional=True)
    raw.add_edge(1, 2, 3.0, bidirectional=True)
    raw.add_edge(2, 3, 1.5, bidirectional=True)
    # Disconnected smaller component {4, 5}
    raw.add_edge(4, 5, 1.0, bidirectional=True)

    lcc = GraphIO.extract_largest_connected_component(raw)
    assert lcc.num_vertices() == 4
    assert GraphIO.is_reachable(lcc, 0, 3)

    with tempfile.TemporaryDirectory() as tmpdir:
        gpath = os.path.join(tmpdir, "test.graph")
        assert GraphIO.save_graph(lcc, gpath)
        loaded = GraphIO.load_graph(gpath)
        assert loaded.num_vertices() == lcc.num_vertices()
        assert loaded.num_edges() == lcc.num_edges()


def test_movingai_map_and_scen_loader() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        map_path = os.path.join(tmpdir, "mini.map")
        scen_path = os.path.join(tmpdir, "mini.map.scen")
        with open(map_path, "w", encoding="utf-8") as f:
            f.write("type octile\nheight 5\nwidth 5\nmap\n")
            f.write(".....\n")
            f.write(".@@@.\n")
            f.write(".....\n")
            f.write(".@@@.\n")
            f.write(".....\n")

        with open(scen_path, "w", encoding="utf-8") as f:
            f.write("version 1.0\n")
            f.write("0\tmini.map\t5\t5\t0\t0\t4\t4\t6.82842712\n")

        mdata = GraphIO.load_movingai_map(map_path, connectivity=8)
        scens = GraphIO.load_movingai_scenarios(scen_path, mdata)
        assert mdata.width == 5 and mdata.height == 5
        assert mdata.graph.num_vertices() == 19
        assert len(scens) == 1

        cfg = SimulationConfig(
            map_name="mini.map",
            graph_type="movingai",
            seed=99,
            num_failures=4,
            failure_mode=FailureMode.ON_CURRENT_PATH,
        )
        out = StochasticSimulator.run_scenario(
            mdata.graph, scens[0].start_node, scens[0].goal_node, cfg
        )
        assert out.validation_passed
        assert len(out.run_summaries) == 3


def test_stochastic_simulator_reproducibility_across_modes() -> None:
    g = GraphIO.load_or_generate_osm_network(
        place_query=None, fallback_blocks_x=10, fallback_blocks_y=10, seed=55
    )
    s, t = GraphIO.sample_valid_od_pair(g, seed=77)

    for mode in (
        FailureMode.ON_CURRENT_PATH,
        FailureMode.UNIFORM_RANDOM,
        FailureMode.MIXED_PROBABILISTIC,
        FailureMode.CONGESTION_PENALTY,
    ):
        cfg = SimulationConfig(
            map_name="osm_test10x10",
            graph_type="osm",
            seed=555,
            num_failures=6,
            failure_mode=mode,
        )
        run1 = StochasticSimulator.run_scenario(g, s, t, cfg)
        run2 = StochasticSimulator.run_scenario(g, s, t, cfg)

        assert run1.validation_passed
        assert run2.validation_passed
        assert len(run1.failure_events) == len(run2.failure_events)
        for e1, e2 in zip(run1.failure_events, run2.failure_events):
            assert e1.u == e2.u and e1.v == e2.v
