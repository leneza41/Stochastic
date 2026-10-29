#pragma once

#include "incremental_search/graph.hpp"
#include "incremental_search/graph_io.hpp"
#include "incremental_search/types.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace incremental_search {

/**
 * Defines how stochastic edge failures are introduced during a simulation run.
 */
enum class FailureMode {
    ON_CURRENT_PATH,      // Fails edges selected stochastically along the active optimal route
    UNIFORM_RANDOM,       // Fails edges selected uniformly at random across the entire graph
    MIXED_PROBABILISTIC,  // With probability path_failure_prob fails a route edge, else random edge
    CONGESTION_PENALTY    // Multiplies edge weight by congestion_factor instead of complete INF blockage
};

std::string failure_mode_to_string(FailureMode mode);

/**
 * Configuration parameters for a controlled stochastic simulation run.
 */
struct SimulationConfig {
    std::string map_name{"unknown_map"};
    std::string graph_type{"synthetic"}; // "osm", "movingai", "synthetic"
    uint32_t seed{42};
    std::size_t num_failures{10};
    FailureMode failure_mode{FailureMode::ON_CURRENT_PATH};
    double path_failure_prob{0.8};       // Used when failure_mode == MIXED_PROBABILISTIC
    double congestion_factor{5.0};       // Used when failure_mode == CONGESTION_PENALTY
    bool bidirectional_failure{true};    // Fail both (u, v) and (v, u) if reverse edge exists
    bool validate_optimality{true};      // Verify D* Lite, LPA*, Incremental A* match costs
};

/**
 * Record of a single stochastic edge failure event in a sequence.
 */
struct FailureEventRecord {
    std::size_t failure_id{0}; // 1..K (0 = initial search before failures)
    NodeId u{INVALID_NODE};
    NodeId v{INVALID_NODE};
    bool bidirectional{false};
    Weight old_cost{INF};
    Weight new_cost{INF};
};

/**
 * Structured record for a single algorithm at a single replanning step.
 * Directly matches the CSV schema required by "Plan de recolección de datos.pdf":
 * algoritmo, mapa, tipo de grafo, tamaño, semilla, identificador del fallo,
 * arista afectada, tiempo (ms), nodos recalculados y memoria (MB).
 */
struct StepExecutionRecord {
    std::string algorithm;
    std::string map_name;
    std::string graph_type;
    std::size_t num_vertices{0};
    std::size_t num_edges{0};
    uint32_t seed{0};
    NodeId start_node{INVALID_NODE};
    NodeId goal_node{INVALID_NODE};
    std::size_t failure_id{0}; // 0 for initial path, 1..K for each stochastic failure
    std::string affected_edge; // e.g., "12->15" or "INITIAL"
    bool path_found{false};
    Weight path_cost{INF};
    double compute_time_ms{0.0};
    std::size_t nodes_recalculated{0};
    double memory_mb{0.0};
    double process_rss_mb{0.0};
    bool cost_validated{true};
};

/**
 * Aggregated experimental unit summary (Unidad de muestreo = Corrida experimental completa).
 * Stores the average across the K replanning events of a single run.
 */
struct RunSummaryRecord {
    std::string algorithm;
    std::string map_name;
    std::string graph_type;
    std::size_t num_vertices{0};
    std::size_t num_edges{0};
    uint32_t seed{0};
    NodeId start_node{INVALID_NODE};
    NodeId goal_node{INVALID_NODE};
    std::size_t total_failures{0};
    std::size_t successful_replans{0};
    double initial_time_ms{0.0};
    std::size_t initial_nodes{0};
    double mean_replan_time_ms{0.0};
    double mean_replan_nodes{0.0};
    double mean_memory_mb{0.0};
    double peak_memory_mb{0.0};
    bool all_costs_validated{true};
};

/**
 * Result of executing one complete simulation scenario across all three algorithms.
 */
struct SimulationRunOutput {
    SimulationConfig config;
    NodeId start_node{INVALID_NODE};
    NodeId goal_node{INVALID_NODE};
    std::vector<FailureEventRecord> failure_events;
    std::vector<StepExecutionRecord> step_records;
    std::vector<RunSummaryRecord> run_summaries;
    bool validation_passed{true};
};

/**
 * Statistical sample size calculator implementing Section 5 of "Muestra de la Investigación.pdf".
 */
class SampleSizeCalculator {
public:
    static constexpr double Z_95 = 1.96;

    /**
     * Infinite population formula (N > 99,999, e.g., OSM and synthetic scenarios):
     *   n = (Z^2 * S^2) / E^2
     */
    static std::size_t compute_infinite_population(double std_dev_s,
                                                   double acceptable_error_e,
                                                   double z_score = Z_95);

    /**
     * Finite population formula (known N scenarios in a closed Moving AI .scen file):
     *   n = (N * Z^2 * S^2) / (E^2 * (N - 1) + Z^2 * S^2)
     */
    static std::size_t compute_finite_population(std::size_t population_n,
                                                 double std_dev_s,
                                                 double acceptable_error_e,
                                                 double z_score = Z_95);
};

/**
 * Controlled Stochastic Edge-Failure Simulator.
 * Executes D* Lite, LPA*, and Incremental A* on identical graph clones under the
 * exact same pseudo-random sequence of edge failures and records all metrics.
 */
class StochasticSimulator {
public:
    /**
     * Runs a complete experimental unit (initial plan + K stochastic edge failures)
     * for D* Lite, LPA*, and Incremental A* on (start, goal).
     */
    static SimulationRunOutput run_scenario(const Graph& base_graph,
                                            NodeId start,
                                            NodeId goal,
                                            const SimulationConfig& config);

    /**
     * Exports step-by-step records to a structured CSV file.
     */
    static bool export_steps_csv(const std::vector<StepExecutionRecord>& records,
                                 const std::string& filepath,
                                 bool append = false);

    /**
     * Exports run-level summaries (unidad muestral) to a structured CSV file.
     */
    static bool export_summaries_csv(const std::vector<RunSummaryRecord>& summaries,
                                     const std::string& filepath,
                                     bool append = false);
};

} // namespace incremental_search
