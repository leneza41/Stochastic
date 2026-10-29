#include "incremental_search/evaluation.hpp"
#include "incremental_search/graph_io.hpp"
#include "incremental_search/simulation.hpp"

#include <iostream>
#include <string>
#include <vector>

using namespace incremental_search;

int main(int argc, char* argv[]) {
    std::string dest_dir = "results/cpp_research_output";
    std::string data_dir = "data";
    std::size_t runs_per_map = 5;

    if (argc >= 2) {
        dest_dir = argv[1];
    }
    if (argc >= 3) {
        runs_per_map = static_cast<std::size_t>(std::stoul(argv[2]));
    }

    std::cout << "=== C++17 Research Benchmark & Evaluation Pipeline (Steps 3 & 4) ===\n"
              << "Destination Folder: " << dest_dir << "\n";

    std::vector<StepExecutionRecord> all_steps;
    std::vector<RunSummaryRecord> all_summaries;

    struct MapEntry {
        std::string graph_type;
        std::string filename;
        std::string map_label;
        uint32_t base_seed;
    };

    const std::vector<MapEntry> catalog = {
        {"osm", "osm_urban_small.graph", "osm_urban_small", 10000},
        {"osm", "osm_urban_medium.graph", "osm_urban_medium", 20000},
        {"movingai", "arena_rooms.map", "arena_rooms.map", 40000},
        {"movingai", "dungeon_game.map", "dungeon_game.map", 50000},
        {"synthetic", "syn_grid_low_density.graph", "syn_grid_low_density", 70000},
        {"synthetic", "syn_grid_high_density.graph", "syn_grid_high_density", 80000}
    };

    for (const auto& entry : catalog) {
        Graph g;
        std::vector<MovingAIScenario> scens;
        try {
            if (entry.graph_type == "movingai") {
                auto mdata = GraphIO::load_movingai_map(data_dir + "/movingai/" + entry.filename, 8);
                g = mdata.graph;
                scens = GraphIO::load_movingai_scenarios(
                    data_dir + "/movingai/" + entry.filename + ".scen", mdata);
            } else {
                g = GraphIO::load_graph(data_dir + "/" + entry.graph_type + "/" + entry.filename);
            }
        } catch (...) {
            g = GraphIO::generate_synthetic_grid(30, 30, 0.15, 8, entry.base_seed);
        }

        // Discarded warmup run
        auto [ws, wt] = scens.empty()
                            ? GraphIO::sample_valid_od_pair(g, entry.base_seed)
                            : std::make_pair(scens[0].start_node, scens[0].goal_node);
        if (ws != INVALID_NODE) {
            SimulationConfig wcfg;
            wcfg.map_name = entry.map_label;
            wcfg.graph_type = entry.graph_type;
            wcfg.seed = entry.base_seed;
            wcfg.num_failures = 3;
            (void)StochasticSimulator::run_scenario(g, ws, wt, wcfg);
        }

        for (std::size_t r = 1; r <= runs_per_map; ++r) {
            uint32_t seed = entry.base_seed + static_cast<uint32_t>(r);
            NodeId s = INVALID_NODE;
            NodeId t = INVALID_NODE;
            if (!scens.empty()) {
                const auto& sc = scens[(r - 1) % scens.size()];
                s = sc.start_node;
                t = sc.goal_node;
            } else {
                std::tie(s, t) = GraphIO::sample_valid_od_pair(g, seed);
            }
            if (s == INVALID_NODE || t == INVALID_NODE) {
                continue;
            }

            SimulationConfig cfg;
            cfg.map_name = entry.map_label;
            cfg.graph_type = entry.graph_type;
            cfg.seed = seed;
            cfg.num_failures = 8;
            cfg.failure_mode = FailureMode::ON_CURRENT_PATH;

            auto out = StochasticSimulator::run_scenario(g, s, t, cfg);
            all_steps.insert(all_steps.end(), out.step_records.begin(), out.step_records.end());
            all_summaries.insert(all_summaries.end(), out.run_summaries.begin(), out.run_summaries.end());
        }
    }

    BenchmarkEvaluator::export_research_report(dest_dir, all_steps, all_summaries);
    std::cout << "C++ research results exported to: " << dest_dir << "\n";
    return 0;
}
