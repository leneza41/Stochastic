#include "incremental_search/graph_io.hpp"
#include "incremental_search/simulation.hpp"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>

using namespace incremental_search;

namespace {

void test_sample_size_formulas() {
    // Matches illustrative example in Section 5.3 of "Muestra de la Investigación.pdf":
    // S = 4 ms, E = 1 ms, Z = 1.96 -> Infinite N: n = 61.47 -> 62 runs
    std::size_t n_inf = SampleSizeCalculator::compute_infinite_population(4.0, 1.0);
    assert(n_inf == 62);

    // Finite N = 1000: n = 57.96 -> 58 runs
    std::size_t n_fin = SampleSizeCalculator::compute_finite_population(1000, 4.0, 1.0);
    assert(n_fin == 58);
}

void test_lcc_and_graph_serialization() {
    Graph raw(6, HeuristicType::EUCLIDEAN);
    for (int i = 0; i < 6; ++i) {
        raw.set_coordinates(i, static_cast<double>(i), 0.0);
    }
    // Component 1: {0, 1, 2, 3}
    raw.add_edge(0, 1, 1.0, true);
    raw.add_edge(1, 2, 1.0, true);
    raw.add_edge(2, 3, 1.0, true);
    // Component 2 (smaller): {4, 5}
    raw.add_edge(4, 5, 1.0, true);

    Graph lcc = GraphIO::extract_largest_connected_component(raw);
    assert(lcc.num_vertices() == 4);
    assert(GraphIO::is_reachable(lcc, 0, 3));

    const std::string tmp_path = "test_tmp_graph.graph";
    assert(GraphIO::save_graph(lcc, tmp_path));
    Graph reloaded = GraphIO::load_graph(tmp_path);
    assert(reloaded.num_vertices() == 4);
    assert(reloaded.num_edges() == lcc.num_edges());
    std::remove(tmp_path.c_str());
}

void test_stochastic_simulator_reproducibility_and_validation() {
    Graph g = GraphIO::generate_synthetic_grid(20, 20, 0.15, 8, 777);
    auto [s, t] = GraphIO::sample_valid_od_pair(g, 888);
    assert(s != INVALID_NODE && t != INVALID_NODE);

    SimulationConfig cfg;
    cfg.map_name = "test_grid20";
    cfg.graph_type = "synthetic";
    cfg.seed = 12345;
    cfg.num_failures = 8;
    cfg.failure_mode = FailureMode::ON_CURRENT_PATH;

    auto run1 = StochasticSimulator::run_scenario(g, s, t, cfg);
    auto run2 = StochasticSimulator::run_scenario(g, s, t, cfg);

    assert(run1.validation_passed);
    assert(run2.validation_passed);
    assert(run1.failure_events.size() == run2.failure_events.size());

    for (std::size_t i = 0; i < run1.failure_events.size(); ++i) {
        assert(run1.failure_events[i].u == run2.failure_events[i].u);
        assert(run1.failure_events[i].v == run2.failure_events[i].v);
    }
}

} // namespace

int main() {
    test_sample_size_formulas();
    test_lcc_and_graph_serialization();
    test_stochastic_simulator_reproducibility_and_validation();
    std::cout << "All C++ Step 2 simulation tests passed successfully!\n";
    return 0;
}
