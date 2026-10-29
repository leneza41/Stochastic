#include "incremental_search/simulation.hpp"
#include "incremental_search/d_star_lite.hpp"
#include "incremental_search/incremental_a_star.hpp"
#include "incremental_search/lpa_star.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>

namespace incremental_search {

std::string failure_mode_to_string(FailureMode mode) {
    switch (mode) {
        case FailureMode::UNIFORM_RANDOM:      return "uniform_random";
        case FailureMode::MIXED_PROBABILISTIC: return "mixed_probabilistic";
        case FailureMode::CONGESTION_PENALTY:  return "congestion_penalty";
        case FailureMode::ON_CURRENT_PATH:
        default:                               return "on_current_path";
    }
}

std::size_t SampleSizeCalculator::compute_infinite_population(double std_dev_s,
                                                              double acceptable_error_e,
                                                              double z_score) {
    if (acceptable_error_e <= 0.0 || std_dev_s <= 0.0) {
        return 30;
    }
    const double num = (z_score * z_score) * (std_dev_s * std_dev_s);
    const double den = acceptable_error_e * acceptable_error_e;
    return static_cast<std::size_t>(std::ceil(num / den));
}

std::size_t SampleSizeCalculator::compute_finite_population(std::size_t population_n,
                                                            double std_dev_s,
                                                            double acceptable_error_e,
                                                            double z_score) {
    if (population_n == 0 || acceptable_error_e <= 0.0 || std_dev_s <= 0.0) {
        return 30;
    }
    const double n_val = static_cast<double>(population_n);
    const double z2_s2 = (z_score * z_score) * (std_dev_s * std_dev_s);
    const double e2 = acceptable_error_e * acceptable_error_e;
    const double num = n_val * z2_s2;
    const double den = e2 * (n_val - 1.0) + z2_s2;
    return static_cast<std::size_t>(std::ceil(num / den));
}

namespace {

std::vector<std::pair<NodeId, NodeId>> collect_active_edges(const Graph& graph) {
    std::vector<std::pair<NodeId, NodeId>> active;
    for (std::size_t u = 0; u < graph.num_vertices(); ++u) {
        for (const auto& e : graph.successors(static_cast<NodeId>(u))) {
            if (!e.failed && !std::isinf(e.cost)) {
                active.emplace_back(static_cast<NodeId>(u), e.to);
            }
        }
    }
    return active;
}

std::pair<NodeId, NodeId> select_stochastic_edge_to_fail(
    const Graph& graph,
    const std::vector<NodeId>& current_path,
    const SimulationConfig& config,
    std::mt19937& rng) {
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

    bool pick_from_path = false;
    if (config.failure_mode == FailureMode::ON_CURRENT_PATH ||
        config.failure_mode == FailureMode::CONGESTION_PENALTY) {
        pick_from_path = true;
    } else if (config.failure_mode == FailureMode::MIXED_PROBABILISTIC) {
        pick_from_path = (prob_dist(rng) < config.path_failure_prob);
    }

    if (pick_from_path && current_path.size() >= 2) {
        std::vector<std::pair<NodeId, NodeId>> path_edges;
        path_edges.reserve(current_path.size() - 1);
        for (std::size_t i = 0; i + 1 < current_path.size(); ++i) {
            NodeId u = current_path[i];
            NodeId v = current_path[i + 1];
            if (!std::isinf(graph.get_edge_cost(u, v))) {
                path_edges.emplace_back(u, v);
            }
        }
        if (!path_edges.empty()) {
            std::uniform_int_distribution<std::size_t> idx_dist(0, path_edges.size() - 1);
            return path_edges[idx_dist(rng)];
        }
    }

    auto active = collect_active_edges(graph);
    if (active.empty()) {
        return {INVALID_NODE, INVALID_NODE};
    }
    std::uniform_int_distribution<std::size_t> idx_dist(0, active.size() - 1);
    return active[idx_dist(rng)];
}

StepExecutionRecord make_step_record(const SimulationConfig& cfg,
                                     std::size_t num_v,
                                     std::size_t num_e,
                                     NodeId start,
                                     NodeId goal,
                                     std::size_t failure_id,
                                     const std::string& affected_edge,
                                     const SearchResult& res,
                                     bool cost_valid) {
    StepExecutionRecord rec;
    rec.algorithm = res.algorithm_name;
    rec.map_name = cfg.map_name;
    rec.graph_type = cfg.graph_type;
    rec.num_vertices = num_v;
    rec.num_edges = num_e;
    rec.seed = cfg.seed;
    rec.start_node = start;
    rec.goal_node = goal;
    rec.failure_id = failure_id;
    rec.affected_edge = affected_edge;
    rec.path_found = res.path_found;
    rec.path_cost = res.path_cost;
    rec.compute_time_ms = res.compute_time_ms;
    rec.nodes_recalculated = res.nodes_recalculated;
    rec.memory_mb = res.memory_mb;
    rec.process_rss_mb = res.process_rss_mb;
    rec.cost_validated = cost_valid;
    return rec;
}

} // namespace

SimulationRunOutput StochasticSimulator::run_scenario(const Graph& base_graph,
                                                      NodeId start,
                                                      NodeId goal,
                                                      const SimulationConfig& config) {
    SimulationRunOutput output;
    output.config = config;
    output.start_node = start;
    output.goal_node = goal;

    // Create three identical graph instances for equivalent execution conditions
    Graph g_dstar = base_graph;
    Graph g_lpa = base_graph;
    Graph g_inca = base_graph;

    const std::size_t num_v = base_graph.num_vertices();
    const std::size_t num_e = base_graph.num_edges();

    DStarLite dstar(g_dstar, start, goal);
    LPAStar lpa(g_lpa, start, goal);
    IncrementalAStar inca(g_inca, start, goal);

    dstar.initialize();
    lpa.initialize();
    inca.initialize();

    SearchResult res_d = dstar.compute_initial_path();
    SearchResult res_l = lpa.compute_initial_path();
    SearchResult res_a = inca.compute_initial_path();

    bool init_valid = approx_equal(res_d.path_cost, res_l.path_cost) &&
                      approx_equal(res_d.path_cost, res_a.path_cost);
    if (!init_valid) {
        output.validation_passed = false;
    }

    output.step_records.push_back(
        make_step_record(config, num_v, num_e, start, goal, 0, "INITIAL", res_d, init_valid));
    output.step_records.push_back(
        make_step_record(config, num_v, num_e, start, goal, 0, "INITIAL", res_l, init_valid));
    output.step_records.push_back(
        make_step_record(config, num_v, num_e, start, goal, 0, "INITIAL", res_a, init_valid));

    std::mt19937 rng(config.seed);

    for (std::size_t f_id = 1; f_id <= config.num_failures; ++f_id) {
        if (!res_d.path_found && config.failure_mode == FailureMode::ON_CURRENT_PATH) {
            break;
        }

        auto [u, v] = select_stochastic_edge_to_fail(g_dstar, res_d.path, config, rng);
        if (u == INVALID_NODE || v == INVALID_NODE) {
            break;
        }

        Weight old_cost = g_dstar.get_edge_cost(u, v);
        Weight new_cost = (config.failure_mode == FailureMode::CONGESTION_PENALTY)
                              ? (old_cost * config.congestion_factor)
                              : INF;

        std::vector<EdgeChange> changes_d;
        std::vector<EdgeChange> changes_l;
        std::vector<EdgeChange> changes_a;

        changes_d.push_back(g_dstar.update_edge_cost(u, v, new_cost));
        changes_l.push_back(g_lpa.update_edge_cost(u, v, new_cost));
        changes_a.push_back(g_inca.update_edge_cost(u, v, new_cost));

        bool bidir_applied = false;
        if (config.bidirectional_failure && g_dstar.has_edge(v, u)) {
            Weight rev_old = g_dstar.get_edge_cost(v, u);
            Weight rev_new = (config.failure_mode == FailureMode::CONGESTION_PENALTY)
                                 ? (rev_old * config.congestion_factor)
                                 : INF;
            changes_d.push_back(g_dstar.update_edge_cost(v, u, rev_new));
            changes_l.push_back(g_lpa.update_edge_cost(v, u, rev_new));
            changes_a.push_back(g_inca.update_edge_cost(v, u, rev_new));
            bidir_applied = true;
        }

        output.failure_events.push_back(
            FailureEventRecord{f_id, u, v, bidir_applied, old_cost, new_cost});

        res_d = dstar.replan(changes_d);
        res_l = lpa.replan(changes_l);
        res_a = inca.replan(changes_a);

        bool step_valid = (res_d.path_found == res_l.path_found) &&
                          (res_d.path_found == res_a.path_found) &&
                          approx_equal(res_d.path_cost, res_l.path_cost) &&
                          approx_equal(res_d.path_cost, res_a.path_cost);
        if (!step_valid) {
            output.validation_passed = false;
        }

        std::string edge_label = std::to_string(u) + (bidir_applied ? "<->" : "->") + std::to_string(v);
        output.step_records.push_back(
            make_step_record(config, num_v, num_e, start, goal, f_id, edge_label, res_d, step_valid));
        output.step_records.push_back(
            make_step_record(config, num_v, num_e, start, goal, f_id, edge_label, res_l, step_valid));
        output.step_records.push_back(
            make_step_record(config, num_v, num_e, start, goal, f_id, edge_label, res_a, step_valid));
    }

    // Aggregate per-algorithm run summaries (Unidad de muestreo)
    const std::vector<std::string> alg_names = {"D* Lite", "LPA*", "Incremental A*"};
    for (const auto& alg : alg_names) {
        RunSummaryRecord summary;
        summary.algorithm = alg;
        summary.map_name = config.map_name;
        summary.graph_type = config.graph_type;
        summary.num_vertices = num_v;
        summary.num_edges = num_e;
        summary.seed = config.seed;
        summary.start_node = start;
        summary.goal_node = goal;
        summary.all_costs_validated = true;

        double sum_time = 0.0;
        double sum_nodes = 0.0;
        double sum_mem = 0.0;
        double peak_mem = 0.0;
        std::size_t replan_count = 0;
        std::size_t succ_count = 0;

        for (const auto& rec : output.step_records) {
            if (rec.algorithm != alg) {
                continue;
            }
            if (!rec.cost_validated) {
                summary.all_costs_validated = false;
            }
            peak_mem = std::max(peak_mem, rec.memory_mb);
            if (rec.failure_id == 0) {
                summary.initial_time_ms = rec.compute_time_ms;
                summary.initial_nodes = rec.nodes_recalculated;
            } else {
                ++replan_count;
                if (rec.path_found) {
                    ++succ_count;
                }
                sum_time += rec.compute_time_ms;
                sum_nodes += static_cast<double>(rec.nodes_recalculated);
                sum_mem += rec.memory_mb;
            }
        }

        summary.total_failures = replan_count;
        summary.successful_replans = succ_count;
        if (replan_count > 0) {
            const double denom = static_cast<double>(replan_count);
            summary.mean_replan_time_ms = sum_time / denom;
            summary.mean_replan_nodes = sum_nodes / denom;
            summary.mean_memory_mb = sum_mem / denom;
        } else {
            summary.mean_memory_mb = peak_mem;
        }
        summary.peak_memory_mb = peak_mem;
        output.run_summaries.push_back(summary);
    }

    return output;
}

bool StochasticSimulator::export_steps_csv(const std::vector<StepExecutionRecord>& records,
                                           const std::string& filepath,
                                           bool append) {
    bool write_header = !append;
    if (append) {
        std::ifstream check(filepath);
        write_header = !check.good() || check.peek() == std::ifstream::traits_type::eof();
    }

    std::ofstream out(filepath, append ? (std::ios::out | std::ios::app) : std::ios::out);
    if (!out.is_open()) {
        return false;
    }

    if (write_header) {
        out << "algorithm,map_name,graph_type,num_vertices,num_edges,seed,"
               "start_node,goal_node,failure_id,affected_edge,path_found,"
               "path_cost,compute_time_ms,nodes_recalculated,memory_mb,process_rss_mb,cost_validated\n";
    }

    out << std::fixed << std::setprecision(6);
    for (const auto& r : records) {
        out << r.algorithm << ","
            << r.map_name << ","
            << r.graph_type << ","
            << r.num_vertices << ","
            << r.num_edges << ","
            << r.seed << ","
            << r.start_node << ","
            << r.goal_node << ","
            << r.failure_id << ","
            << r.affected_edge << ","
            << (r.path_found ? 1 : 0) << ",";
        if (std::isinf(r.path_cost)) {
            out << "INF,";
        } else {
            out << r.path_cost << ",";
        }
        out << r.compute_time_ms << ","
            << r.nodes_recalculated << ","
            << r.memory_mb << ","
            << r.process_rss_mb << ","
            << (r.cost_validated ? 1 : 0) << "\n";
    }
    return true;
}

bool StochasticSimulator::export_summaries_csv(const std::vector<RunSummaryRecord>& summaries,
                                               const std::string& filepath,
                                               bool append) {
    bool write_header = !append;
    if (append) {
        std::ifstream check(filepath);
        write_header = !check.good() || check.peek() == std::ifstream::traits_type::eof();
    }

    std::ofstream out(filepath, append ? (std::ios::out | std::ios::app) : std::ios::out);
    if (!out.is_open()) {
        return false;
    }

    if (write_header) {
        out << "algorithm,map_name,graph_type,num_vertices,num_edges,seed,"
               "start_node,goal_node,total_failures,successful_replans,"
               "initial_time_ms,initial_nodes,mean_replan_time_ms,"
               "mean_replan_nodes,mean_memory_mb,peak_memory_mb,all_costs_validated\n";
    }

    out << std::fixed << std::setprecision(6);
    for (const auto& s : summaries) {
        out << s.algorithm << ","
            << s.map_name << ","
            << s.graph_type << ","
            << s.num_vertices << ","
            << s.num_edges << ","
            << s.seed << ","
            << s.start_node << ","
            << s.goal_node << ","
            << s.total_failures << ","
            << s.successful_replans << ","
            << s.initial_time_ms << ","
            << s.initial_nodes << ","
            << s.mean_replan_time_ms << ","
            << s.mean_replan_nodes << ","
            << s.mean_memory_mb << ","
            << s.peak_memory_mb << ","
            << (s.all_costs_validated ? 1 : 0) << "\n";
    }
    return true;
}

} // namespace incremental_search
