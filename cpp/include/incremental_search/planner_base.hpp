#pragma once

#include "incremental_search/graph.hpp"
#include "incremental_search/types.hpp"
#include <string>
#include <vector>

namespace incremental_search {

/**
 * Common interface for all incremental search algorithms (D* Lite, LPA*, Incremental A*)
 * ensuring identical execution conditions, timing measurement, node recalculation
 * accounting, and memory profiling.
 */
class IncrementalPlanner {
public:
    IncrementalPlanner(const Graph& graph, NodeId start, NodeId goal)
        : graph_(graph), start_(start), goal_(goal) {}

    virtual ~IncrementalPlanner() = default;

    virtual std::string name() const = 0;

    /**
     * Resets internal state and initializes data structures for (start, goal).
     */
    virtual void initialize() = 0;

    /**
     * Computes the initial optimal path from start_ to goal_.
     */
    virtual SearchResult compute_initial_path() = 0;

    /**
     * Replans the optimal path after one or more edge cost changes / failures.
     * Measures exclusively the replanning phase with std::chrono::steady_clock.
     */
    virtual SearchResult replan(const std::vector<EdgeChange>& changes) = 0;

    /**
     * Updates the current agent start vertex (when traversing the path dynamically).
     */
    virtual void update_start(NodeId new_start) = 0;

    /**
     * Extracts the current minimum-cost path from start_ to goal_.
     */
    virtual std::vector<NodeId> extract_path() const = 0;

    /**
     * Returns the optimal cost from start_ to goal_ (INF if unreachable).
     */
    virtual Weight current_path_cost() const = 0;

    /**
     * Returns exact memory usage in bytes of the planner's internal structures.
     */
    virtual std::size_t structural_memory_bytes() const = 0;

    NodeId start() const { return start_; }
    NodeId goal() const { return goal_; }

protected:
    const Graph& graph_;
    NodeId start_;
    NodeId goal_;
    std::size_t nodes_recalculated_{0};
    std::size_t peak_structural_bytes_{0};
};

} // namespace incremental_search
