#pragma once

#include "incremental_search/types.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace incremental_search {

enum class HeuristicType {
    EUCLIDEAN,
    MANHATTAN,
    OCTILE,
    HAVERSINE,
    ZERO
};

struct Edge {
    NodeId to{INVALID_NODE};
    Weight cost{1.0};
    Weight original_cost{1.0};
    bool failed{false};
};

struct Coordinates {
    double x{0.0};
    double y{0.0};
};

/**
 * Directed weighted graph supporting dynamic edge failures and restorations,
 * fast forward (successors) and backward (predecessors) queries, and
 * consistent heuristic evaluation.
 */
class Graph {
public:
    explicit Graph(std::size_t num_vertices = 0,
                   HeuristicType heuristic_type = HeuristicType::EUCLIDEAN);

    NodeId add_vertex(double x = 0.0, double y = 0.0);
    void set_coordinates(NodeId u, double x, double y);
    Coordinates get_coordinates(NodeId u) const;

    void add_edge(NodeId u, NodeId v, Weight cost, bool bidirectional = false);
    bool has_edge(NodeId u, NodeId v) const;

    Weight get_edge_cost(NodeId u, NodeId v) const;
    Weight get_original_edge_cost(NodeId u, NodeId v) const;

    /**
     * Updates edge (u, v) to new_cost and returns the EdgeChange record.
     * Setting new_cost = INF simulates a complete edge failure/blockage.
     */
    EdgeChange update_edge_cost(NodeId u, NodeId v, Weight new_cost);
    EdgeChange fail_edge(NodeId u, NodeId v);
    EdgeChange restore_edge(NodeId u, NodeId v);
    void restore_all_edges();

    const std::vector<Edge>& successors(NodeId u) const;
    const std::vector<NodeId>& predecessors(NodeId u) const;

    Weight heuristic(NodeId u, NodeId v) const;
    void set_heuristic_type(HeuristicType type) { heuristic_type_ = type; }
    HeuristicType heuristic_type() const { return heuristic_type_; }

    std::size_t num_vertices() const { return adj_.size(); }
    std::size_t num_edges() const { return num_edges_; }
    std::size_t memory_bytes() const;

private:
    void ensure_vertex(NodeId u);

    std::vector<std::vector<Edge>> adj_;
    std::vector<std::vector<NodeId>> rev_adj_;
    std::vector<Coordinates> coords_;
    std::size_t num_edges_{0};
    HeuristicType heuristic_type_{HeuristicType::EUCLIDEAN};
};

} // namespace incremental_search
