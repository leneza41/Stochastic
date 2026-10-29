#pragma once

#include "incremental_search/graph.hpp"
#include "incremental_search/types.hpp"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace incremental_search {

/**
 * Represents a single scenario entry from a Moving AI .scen file.
 */
struct MovingAIScenario {
    int bucket{0};
    std::string map_name;
    int map_width{0};
    int map_height{0};
    int start_x{0};
    int start_y{0};
    int goal_x{0};
    int goal_y{0};
    double optimal_length{0.0};
    NodeId start_node{INVALID_NODE};
    NodeId goal_node{INVALID_NODE};
};

/**
 * Loaded Moving AI map alongside grid-to-vertex coordinate mapping.
 */
struct MovingAIMap {
    Graph graph;
    int width{0};
    int height{0};
    int connectivity{8};
    std::vector<NodeId> grid_to_node; // size = width * height, INVALID_NODE if blocked

    NodeId node_at(int x, int y) const {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            return INVALID_NODE;
        }
        return grid_to_node[static_cast<std::size_t>(y * width + x)];
    }
};

/**
 * Graph input/output, preprocessing (largest connected component, deduplication,
 * reachability), Moving AI (.map / .scen) loaders, and synthetic graph generators.
 */
class GraphIO {
public:
    /**
     * Saves/loads a Graph in the unified text format (.graph) shared between C++ and Python.
     * Header:
     *   GRAPH_V1 <num_vertices> <num_edges> <heuristic_type> <bidirectional_default>
     *   V <id> <x> <y>
     *   E <u> <v> <cost>
     */
    static bool save_graph(const Graph& graph, const std::string& filepath);
    static Graph load_graph(const std::string& filepath);

    /**
     * Parses a Moving AI benchmark .map file and converts passable cells ('.', 'G', 'S')
     * into a graph with explicit 4-neighbor or 8-neighbor connectivity.
     */
    static MovingAIMap load_movingai_map(const std::string& map_filepath,
                                         int connectivity = 8);

    /**
     * Parses a Moving AI .scen file and maps (start_x, start_y) -> start_node
     * and (goal_x, goal_y) -> goal_node using the loaded MovingAIMap.
     */
    static std::vector<MovingAIScenario> load_movingai_scenarios(
        const std::string& scen_filepath,
        const MovingAIMap& map_data);

    /**
     * Generates a controlled synthetic grid graph with random obstacles and
     * weighted edges using a deterministic seed.
     */
    static Graph generate_synthetic_grid(int width,
                                         int height,
                                         double obstacle_ratio,
                                         int connectivity,
                                         uint32_t seed,
                                          double min_weight = 1.0,
                                         double max_weight = 1.0);

    /**
     * Generates a controlled synthetic random geometric road-like graph
     * with specified vertex count, average degree, and seed.
     */
    static Graph generate_synthetic_random_graph(std::size_t num_vertices,
                                                 std::size_t avg_degree,
                                                 uint32_t seed,
                                                 double area_size = 1000.0);

    /**
     * Extracts the largest weakly/strongly connected component, removes isolated
     * vertices and self-loops, and ensures positive weights (OSM preprocessing).
     */
    static Graph extract_largest_connected_component(const Graph& input_graph);

    /**
     * Checks whether a valid directed path with finite cost exists from start to goal.
     */
    static bool is_reachable(const Graph& graph, NodeId start, NodeId goal);

    /**
     * Samples a valid (start, goal) pair using Simple Random Sampling (MAS)
     * with a fixed seed, guaranteeing that goal is reachable from start.
     */
    static std::pair<NodeId, NodeId> sample_valid_od_pair(const Graph& graph,
                                                          uint32_t seed,
                                                          std::size_t max_attempts = 1000);
};

} // namespace incremental_search
