#pragma once

#include "incremental_search/planner_base.hpp"
#include "incremental_search/priority_queue.hpp"
#include <vector>

namespace incremental_search {

/**
 * Incremental A* (Dynamic Search-Tree Repair A*)
 *
 * Maintains the forward A* search tree (g-values, parent pointers, children sets,
 * and closed/open status). When stochastic edge failures or cost changes occur:
 *   - If an edge (u, v) increases in cost (or fails) and parent[v] == u, the
 *     subtree rooted at v is invalidated and its boundary neighbors with valid
 *     g-values are re-inserted into the priority queue to repair the search tree
 *     incrementally without recomputing unaffected regions from scratch.
 *   - If an edge (u, v) decreases in cost and g[u] + new_cost < g[v], v is
 *     updated and placed in the priority queue.
 *
 * Uses the exact same PriorityQueue, Graph, heuristic, and node recalculation
 * accounting as LPA* and D* Lite for fair experimental comparison.
 */
class IncrementalAStar final : public IncrementalPlanner {
public:
    IncrementalAStar(const Graph& graph, NodeId start, NodeId goal);

    std::string name() const override { return "Incremental A*"; }
    void initialize() override;
    SearchResult compute_initial_path() override;
    SearchResult replan(const std::vector<EdgeChange>& changes) override;
    void update_start(NodeId new_start) override;
    std::vector<NodeId> extract_path() const override;
    Weight current_path_cost() const override;
    std::size_t structural_memory_bytes() const override;

private:
    Key calculate_key(NodeId u) const;
    void invalidate_subtree(NodeId root);
    void run_a_star_repair();
    void record_peak_memory();

    std::vector<Weight> g_;
    std::vector<NodeId> parent_;
    std::vector<uint8_t> closed_;
    PriorityQueue open_list_;
    bool initialized_{false};
};

} // namespace incremental_search
