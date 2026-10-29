#include "incremental_search/lpa_star.hpp"
#include "incremental_search/memory_tracker.hpp"
#include <algorithm>
#include <chrono>
#include <unordered_set>

namespace incremental_search {

LPAStar::LPAStar(const Graph& graph, NodeId start, NodeId goal)
    : IncrementalPlanner(graph, start, goal) {}

void LPAStar::initialize() {
    const std::size_t n = graph_.num_vertices();
    g_.assign(n, INF);
    rhs_.assign(n, INF);
    backpointer_.assign(n, INVALID_NODE);
    open_list_.clear();
    nodes_recalculated_ = 0;
    peak_structural_bytes_ = 0;

    if (start_ >= 0 && static_cast<std::size_t>(start_) < n) {
        rhs_[static_cast<std::size_t>(start_)] = 0.0;
        open_list_.insert_or_update(start_, calculate_key(start_));
    }
    initialized_ = true;
    record_peak_memory();
}

Key LPAStar::calculate_key(NodeId u) const {
    const Weight m = std::min(g_[static_cast<std::size_t>(u)], rhs_[static_cast<std::size_t>(u)]);
    return Key(safe_add(m, graph_.heuristic(u, goal_)), m);
}

void LPAStar::update_vertex(NodeId u) {
    ++nodes_recalculated_;
    const auto idx = static_cast<std::size_t>(u);

    if (u != start_) {
        Weight best_rhs = INF;
        NodeId best_pred = INVALID_NODE;
        for (NodeId pred : graph_.predecessors(u)) {
            const Weight edge_cost = graph_.get_edge_cost(pred, u);
            const Weight cand = safe_add(g_[static_cast<std::size_t>(pred)], edge_cost);
            if (cand < best_rhs) {
                best_rhs = cand;
                best_pred = pred;
            }
        }
        rhs_[idx] = best_rhs;
        backpointer_[idx] = best_pred;
    }

    if (open_list_.contains(u)) {
        open_list_.remove(u);
    }

    if (!approx_equal(g_[idx], rhs_[idx])) {
        open_list_.insert_or_update(u, calculate_key(u));
    }
    record_peak_memory();
}

void LPAStar::compute_shortest_path() {
    const auto goal_idx = static_cast<std::size_t>(goal_);

    while (!open_list_.empty() &&
           (open_list_.top_key() < calculate_key(goal_) ||
            !approx_equal(rhs_[goal_idx], g_[goal_idx]))) {
        auto [u, k_old] = open_list_.pop();
        if (u == INVALID_NODE) {
            break;
        }
        const auto u_idx = static_cast<std::size_t>(u);

        if (g_[u_idx] > rhs_[u_idx]) {
            g_[u_idx] = rhs_[u_idx];
            for (const auto& edge : graph_.successors(u)) {
                update_vertex(edge.to);
            }
        } else {
            g_[u_idx] = INF;
            update_vertex(u);
            for (const auto& edge : graph_.successors(u)) {
                update_vertex(edge.to);
            }
        }
    }
    record_peak_memory();
}

SearchResult LPAStar::compute_initial_path() {
    if (!initialized_) {
        initialize();
    }
    nodes_recalculated_ = 0;
    peak_structural_bytes_ = structural_memory_bytes();

    const auto t0 = std::chrono::steady_clock::now();
    compute_shortest_path();
    const auto t1 = std::chrono::steady_clock::now();

    SearchResult result;
    result.algorithm_name = name();
    result.compute_time_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.nodes_recalculated = nodes_recalculated_;
    result.path_cost = current_path_cost();
    result.path_found = !std::isinf(result.path_cost);
    if (result.path_found) {
        result.path = extract_path();
    }
    result.memory_mb = MemoryTracker::bytes_to_mb(peak_structural_bytes_);
    result.process_rss_mb = MemoryTracker::current_process_rss_mb();
    return result;
}

SearchResult LPAStar::replan(const std::vector<EdgeChange>& changes) {
    if (!initialized_) {
        return compute_initial_path();
    }
    nodes_recalculated_ = 0;
    peak_structural_bytes_ = structural_memory_bytes();

    const auto t0 = std::chrono::steady_clock::now();
    for (const auto& change : changes) {
        if (change.v >= 0 && static_cast<std::size_t>(change.v) < graph_.num_vertices()) {
            update_vertex(change.v);
        }
    }
    compute_shortest_path();
    const auto t1 = std::chrono::steady_clock::now();

    SearchResult result;
    result.algorithm_name = name();
    result.compute_time_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.nodes_recalculated = nodes_recalculated_;
    result.path_cost = current_path_cost();
    result.path_found = !std::isinf(result.path_cost);
    if (result.path_found) {
        result.path = extract_path();
    }
    result.memory_mb = MemoryTracker::bytes_to_mb(peak_structural_bytes_);
    result.process_rss_mb = MemoryTracker::current_process_rss_mb();
    return result;
}

void LPAStar::update_start(NodeId new_start) {
    if (new_start == start_) {
        return;
    }
    start_ = new_start;
    initialize();
}

Weight LPAStar::current_path_cost() const {
    if (!initialized_ || goal_ < 0 || static_cast<std::size_t>(goal_) >= g_.size()) {
        return INF;
    }
    return g_[static_cast<std::size_t>(goal_)];
}

std::vector<NodeId> LPAStar::extract_path() const {
    std::vector<NodeId> path;
    if (std::isinf(current_path_cost())) {
        return path;
    }
    NodeId curr = goal_;
    std::size_t max_steps = graph_.num_vertices() + 1;
    while (curr != INVALID_NODE && max_steps-- > 0) {
        path.push_back(curr);
        if (curr == start_) {
            std::reverse(path.begin(), path.end());
            return path;
        }
        NodeId best_pred = backpointer_[static_cast<std::size_t>(curr)];
        if (best_pred == INVALID_NODE) {
            Weight best_cost = INF;
            for (NodeId pred : graph_.predecessors(curr)) {
                Weight c = safe_add(g_[static_cast<std::size_t>(pred)],
                                    graph_.get_edge_cost(pred, curr));
                if (c < best_cost) {
                    best_cost = c;
                    best_pred = pred;
                }
            }
        }
        curr = best_pred;
    }
    return {};
}

std::size_t LPAStar::structural_memory_bytes() const {
    return sizeof(*this) +
           g_.capacity() * sizeof(Weight) +
           rhs_.capacity() * sizeof(Weight) +
           backpointer_.capacity() * sizeof(NodeId) +
           open_list_.memory_bytes();
}

void LPAStar::record_peak_memory() {
    peak_structural_bytes_ = std::max(peak_structural_bytes_, structural_memory_bytes());
}

} // namespace incremental_search
