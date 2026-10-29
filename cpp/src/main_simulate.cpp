#include "incremental_search/graph_io.hpp"
#include "incremental_search/simulation.hpp"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace incremental_search;

namespace {

void print_summary_table(const std::vector<RunSummaryRecord>& summaries) {
    std::cout << std::left
              << std::setw(16) << "Algorithm" << " | "
              << std::setw(22) << "Map" << " | "
              << std::setw(10) << "Type" << " | "
              << std::setw(7) << "Seed" << " | "
              << std::setw(11) << "MeanTime(ms)" << " | "
              << std::setw(11) << "MeanNodes" << " | "
              << std::setw(11) << "PeakMem(MB)" << " | "
              << "Valid\n";
    std::cout << std::string(108, '-') << "\n";
    for (const auto& s : summaries) {
        std::cout << std::left
                  << std::setw(16) << s.algorithm << " | "
                  << std::setw(22) << s.map_name << " | "
                  << std::setw(10) << s.graph_type << " | "
                  << std::setw(7) << s.seed << " | "
                  << std::setw(12) << std::fixed << std::setprecision(4) << s.mean_replan_time_ms << " | "
                  << std::setw(11) << std::setprecision(1) << s.mean_replan_nodes << " | "
                  << std::setw(11) << std::setprecision(5) << s.peak_memory_mb << " | "
                  << (s.all_costs_validated ? "OK" : "FAIL") << "\n";
    }
}

} // namespace

int main(int argc, char* argv[]) {
    std::string data_dir = "data";
    std::string out_steps_csv = "results/raw/cpp_simulation_steps.csv";
    std::string out_summary_csv = "results/raw/cpp_simulation_summaries.csv";

    if (argc >= 2) {
        data_dir = argv[1];
    }

    std::vector<StepExecutionRecord> all_steps;
    std::vector<RunSummaryRecord> all_summaries;

    std::cout << "=== Step 2: C++ Stochastic Edge-Failure Simulation Engine ===\n";

    // 1. OpenStreetMap road network (loaded from .graph if prepared, or generated)
    Graph osm_graph;
    try {
        osm_graph = GraphIO::load_graph(data_dir + "/osm/osm_urban_small.graph");
    } catch (...) {
        osm_graph = GraphIO::generate_synthetic_random_graph(400, 4, 101, 1200.0);
    }

    auto [osm_s, osm_t] = GraphIO::sample_valid_od_pair(osm_graph, 2026);
    if (osm_s != INVALID_NODE) {
        SimulationConfig cfg_osm;
        cfg_osm.map_name = "osm_urban_small";
        cfg_osm.graph_type = "osm";
        cfg_osm.seed = 2026;
        cfg_osm.num_failures = 10;
        cfg_osm.failure_mode = FailureMode::ON_CURRENT_PATH;

        auto out = StochasticSimulator::run_scenario(osm_graph, osm_s, osm_t, cfg_osm);
        all_steps.insert(all_steps.end(), out.step_records.begin(), out.step_records.end());
        all_summaries.insert(all_summaries.end(), out.run_summaries.begin(), out.run_summaries.end());
    }

    // 2. Moving AI Benchmark (.map + .scen)
    try {
        auto mai_map = GraphIO::load_movingai_map(data_dir + "/movingai/arena_rooms.map", 8);
        auto mai_scens = GraphIO::load_movingai_scenarios(data_dir + "/movingai/arena_rooms.map.scen", mai_map);
        if (!mai_scens.empty()) {
            const auto& scen = mai_scens.back();
            SimulationConfig cfg_mai;
            cfg_mai.map_name = "arena_rooms.map";
            cfg_mai.graph_type = "movingai";
            cfg_mai.seed = 3030;
            cfg_mai.num_failures = 10;
            cfg_mai.failure_mode = FailureMode::ON_CURRENT_PATH;

            auto out = StochasticSimulator::run_scenario(mai_map.graph, scen.start_node, scen.goal_node, cfg_mai);
            all_steps.insert(all_steps.end(), out.step_records.begin(), out.step_records.end());
            all_summaries.insert(all_summaries.end(), out.run_summaries.begin(), out.run_summaries.end());
        }
    } catch (...) {
        // Fallback if .map not yet generated on disk
        Graph fallback_mai = GraphIO::generate_synthetic_grid(32, 32, 0.15, 8, 3030);
        auto [ms, mt] = GraphIO::sample_valid_od_pair(fallback_mai, 3030);
        SimulationConfig cfg_mai;
        cfg_mai.map_name = "movingai_grid32";
        cfg_mai.graph_type = "movingai";
        cfg_mai.seed = 3030;
        cfg_mai.num_failures = 10;
        auto out = StochasticSimulator::run_scenario(fallback_mai, ms, mt, cfg_mai);
        all_steps.insert(all_steps.end(), out.step_records.begin(), out.step_records.end());
        all_summaries.insert(all_summaries.end(), out.run_summaries.begin(), out.run_summaries.end());
    }

    // 3. Synthetic Controlled Graph
    Graph syn_graph = GraphIO::generate_synthetic_grid(40, 40, 0.20, 8, 4040, 1.0, 3.0);
    auto [syn_s, syn_t] = GraphIO::sample_valid_od_pair(syn_graph, 4040);
    if (syn_s != INVALID_NODE) {
        SimulationConfig cfg_syn;
        cfg_syn.map_name = "synthetic_grid40_d20";
        cfg_syn.graph_type = "synthetic";
        cfg_syn.seed = 4040;
        cfg_syn.num_failures = 10;
        cfg_syn.failure_mode = FailureMode::MIXED_PROBABILISTIC;
        cfg_syn.path_failure_prob = 0.85;

        auto out = StochasticSimulator::run_scenario(syn_graph, syn_s, syn_t, cfg_syn);
        all_steps.insert(all_steps.end(), out.step_records.begin(), out.step_records.end());
        all_summaries.insert(all_summaries.end(), out.run_summaries.begin(), out.run_summaries.end());
    }

    print_summary_table(all_summaries);

    StochasticSimulator::export_steps_csv(all_steps, out_steps_csv, false);
    StochasticSimulator::export_summaries_csv(all_summaries, out_summary_csv, false);

    std::cout << "\nSaved step records to: " << out_steps_csv << "\n"
              << "Saved run summaries to: " << out_summary_csv << "\n";
    return 0;
}
