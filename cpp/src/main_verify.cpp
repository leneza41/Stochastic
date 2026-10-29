#include "incremental_search/d_star_lite.hpp"
#include "incremental_search/graph.hpp"
#include "incremental_search/incremental_a_star.hpp"
#include "incremental_search/lpa_star.hpp"

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace incremental_search;

namespace {

Graph build_grid_graph(int width, int height) {
    Graph g(static_cast<std::size_t>(width * height), HeuristicType::EUCLIDEAN);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            NodeId u = y * width + x;
            g.set_coordinates(u, static_cast<double>(x), static_cast<double>(y));
            if (x + 1 < width) {
                NodeId v = y * width + (x + 1);
                g.add_edge(u, v, 1.0, true);
            }
            if (y + 1 < height) {
                NodeId v = (y + 1) * width + x;
                g.add_edge(u, v, 1.0, true);
            }
        }
    }
    return g;
}

void print_result(const std::string& phase, const SearchResult& res) {
    std::cout << std::left << std::setw(14) << phase << " | "
              << std::setw(16) << res.algorithm_name << " | "
              << "Cost: " << std::setw(8) << std::fixed << std::setprecision(2) << res.path_cost << " | "
              << "Time(ms): " << std::setw(9) << std::setprecision(4) << res.compute_time_ms << " | "
              << "Nodes: " << std::setw(7) << res.nodes_recalculated << " | "
              << "StructMem(MB): " << std::setw(8) << std::setprecision(5) << res.memory_mb
              << "\n";
}

} // namespace

int main() {
    constexpr int W = 30;
    constexpr int H = 30;
    Graph g_dstar = build_grid_graph(W, H);
    Graph g_lpa = build_grid_graph(W, H);
    Graph g_inc_a = build_grid_graph(W, H);

    const NodeId start = 0;
    const NodeId goal = W * H - 1;

    DStarLite dstar(g_dstar, start, goal);
    LPAStar lpa(g_lpa, start, goal);
    IncrementalAStar inc_a(g_inc_a, start, goal);

    dstar.initialize();
    lpa.initialize();
    inc_a.initialize();

    std::cout << "=== Step 1: C++ Incremental Search Equivalence Verification ===\n";
    auto r0_d = dstar.compute_initial_path();
    auto r0_l = lpa.compute_initial_path();
    auto r0_a = inc_a.compute_initial_path();

    print_result("Initial", r0_d);
    print_result("Initial", r0_l);
    print_result("Initial", r0_a);

    if (!approx_equal(r0_d.path_cost, r0_l.path_cost) ||
        !approx_equal(r0_d.path_cost, r0_a.path_cost)) {
        std::cerr << "ERROR: Initial path costs do not match!\n";
        return 1;
    }

    std::mt19937 rng(42);
    for (int step = 1; step <= 5; ++step) {
        const auto& current_path = r0_d.path;
        if (current_path.size() < 2) {
            break;
        }
        std::uniform_int_distribution<std::size_t> dist(0, current_path.size() - 2);
        std::size_t idx = dist(rng);
        NodeId u = current_path[idx];
        NodeId v = current_path[idx + 1];

        auto c1_d = g_dstar.fail_edge(u, v);
        auto c2_d = g_dstar.fail_edge(v, u);
        auto c1_l = g_lpa.fail_edge(u, v);
        auto c2_l = g_lpa.fail_edge(v, u);
        auto c1_a = g_inc_a.fail_edge(u, v);
        auto c2_a = g_inc_a.fail_edge(v, u);

        r0_d = dstar.replan({c1_d, c2_d});
        r0_l = lpa.replan({c1_l, c2_l});
        r0_a = inc_a.replan({c1_a, c2_a});

        std::string label = "Failure #" + std::to_string(step);
        print_result(label, r0_d);
        print_result(label, r0_l);
        print_result(label, r0_a);

        if (!approx_equal(r0_d.path_cost, r0_l.path_cost) ||
            !approx_equal(r0_d.path_cost, r0_a.path_cost)) {
            std::cerr << "ERROR: Replanned path costs diverged at step " << step << "!\n";
            return 1;
        }
    }

    std::cout << "\nAll C++ algorithms converged to identical optimal path costs across all failures.\n";
    return 0;
}
