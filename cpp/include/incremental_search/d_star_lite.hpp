#pragma once

#include "incremental_search/planner_base.hpp"
#include "incremental_search/priority_queue.hpp"
#include <vector>

namespace incremental_search {

/**
 * D* Lite (Optimized version)
 * Reference: Koenig, S., & Likhachev, M. (2002).
 * "D* Lite". AAAI/IAAI, 15, 476-483.
 *
 * Searches backward from goal_ to start_, maintaining g(u), rhs(u), and key
 * modifier km_ so that heuristic priorities remain valid when start_ moves
 * or when edge failures occur along the network.
 */
class DStarLite final : public IncrementalPlanner {
public:
    DStarLite(const Graph& graph, NodeId start, NodeId goal);

    std::string name() const override { return "D* Lite"; }
    void initialize() override;
    SearchResult compute_initial_path() override;
    SearchResult replan(const std::vector<EdgeChange>& changes) override;
    void update_start(NodeId new_start) override;
    std::vector<NodeId> extract_path() const override;
    Weight current_path_cost() const override;
    std::size_t structural_memory_bytes() const override;

private:
    Key calculate_key(NodeId u) const;
    void update_vertex(NodeId u);
    void compute_shortest_path();
    void record_peak_memory();

    std::vector<Weight> g_;
    std::vector<Weight> rhs_;
    PriorityQueue open_list_;
    NodeId last_start_{INVALID_NODE};
    Weight km_{0.0};
    bool initialized_{false};
};

} // namespace incremental_search
