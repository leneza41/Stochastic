#pragma once

#include "incremental_search/planner_base.hpp"
#include "incremental_search/priority_queue.hpp"
#include <vector>

namespace incremental_search {

/**
 * Lifelong Planning A* (LPA*)
 * Reference: Koenig, S., Likhachev, M., & Furcy, D. (2004).
 * "Lifelong Planning A*". Artificial Intelligence, 155(1-2), 93-146.
 *
 * Searches forward from start_ to goal_, maintaining g(u) and rhs(u) values
 * and propagating local inconsistency only through affected vertices.
 */
class LPAStar final : public IncrementalPlanner {
public:
    LPAStar(const Graph& graph, NodeId start, NodeId goal);

    std::string name() const override { return "LPA*"; }
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
    std::vector<NodeId> backpointer_;
    PriorityQueue open_list_;
    bool initialized_{false};
};

} // namespace incremental_search
