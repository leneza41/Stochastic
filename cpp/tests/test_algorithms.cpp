#include "incremental_search/d_star_lite.hpp"
#include "incremental_search/graph.hpp"
#include "incremental_search/incremental_a_star.hpp"
#include "incremental_search/lpa_star.hpp"

#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace incremental_search;

namespace {

Graph create_random_geometric_graph(int width, int height, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> weight_dist(1.0, 5.0);

    Graph g(static_cast<std::size_t>(width * height), HeuristicType::EUCLIDEAN);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            NodeId u = y * width + x;
            g.set_coordinates(u, static_cast<double>(x), static_cast<double>(y));
            if (x + 1 < width) {
                NodeId v = y * width + (x + 1);
                double w = weight_dist(rng);
                g.add_edge(u, v, w, true);
            }
            if (y + 1 < height) {
                NodeId v = (y + 1) * width + x;
                double w = weight_dist(rng);
                g.add_edge(u, v, w, true);
            }
            if (x + 1 < width && y + 1 < height) {
                NodeId v = (y + 1) * width + (x + 1);
                double w = weight_dist(rng) * 1.41421356;
                g.add_edge(u, v, w, true);
            }
        }
    }
    return g;
}

void test_equivalence_under_stochastic_failures() {
    for (uint32_t seed = 1; seed <= 10; ++seed) {
        Graph g1 = create_random_geometric_graph(15, 15, seed);
        Graph g2 = create_random_geometric_graph(15, 15, seed);
        Graph g3 = create_random_geometric_graph(15, 15, seed);

        const NodeId start = 0;
        const NodeId goal = 15 * 15 - 1;

        DStarLite dstar(g1, start, goal);
        LPAStar lpa(g2, start, goal);
        IncrementalAStar inc_a(g3, start, goal);

        auto res_d = dstar.compute_initial_path();
        auto res_l = lpa.compute_initial_path();
        auto res_a = inc_a.compute_initial_path();

        assert(res_d.path_found && res_l.path_found && res_a.path_found);
        assert(approx_equal(res_d.path_cost, res_l.path_cost));
        assert(approx_equal(res_d.path_cost, res_a.path_cost));

        std::mt19937 rng(seed + 100);
        for (int f = 0; f < 8; ++f) {
            if (!res_d.path_found || res_d.path.size() < 2) {
                break;
            }
            std::uniform_int_distribution<std::size_t> edge_idx_dist(0, res_d.path.size() - 2);
            std::size_t idx = edge_idx_dist(rng);
            NodeId u = res_d.path[idx];
            NodeId v = res_d.path[idx + 1];

            auto cd1 = g1.fail_edge(u, v);
            auto cd2 = g1.fail_edge(v, u);
            auto cl1 = g2.fail_edge(u, v);
            auto cl2 = g2.fail_edge(v, u);
            auto ca1 = g3.fail_edge(u, v);
            auto ca2 = g3.fail_edge(v, u);

            res_d = dstar.replan({cd1, cd2});
            res_l = lpa.replan({cl1, cl2});
            res_a = inc_a.replan({ca1, ca2});

            assert(res_d.path_found == res_l.path_found);
            assert(res_d.path_found == res_a.path_found);
            assert(approx_equal(res_d.path_cost, res_l.path_cost));
            assert(approx_equal(res_d.path_cost, res_a.path_cost));
            assert(res_d.compute_time_ms >= 0.0);
            assert(res_l.compute_time_ms >= 0.0);
            assert(res_a.compute_time_ms >= 0.0);
            assert(res_d.memory_mb > 0.0);
            assert(res_l.memory_mb > 0.0);
            assert(res_a.memory_mb > 0.0);
        }
    }
}

} // namespace

int main() {
    test_equivalence_under_stochastic_failures();
    std::cout << "All C++ unit and equivalence tests passed successfully!\n";
    return 0;
}
