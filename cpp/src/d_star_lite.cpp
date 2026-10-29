#include "incremental_search/d_star_lite.hpp"
#include "incremental_search/memory_tracker.hpp"
#include <algorithm>
#include <chrono>

namespace incremental_search {

DStarLite::DStarLite(const Graph& graph, NodeId start, NodeId goal)
    : IncrementalPlanner(graph, start, goal), last_start_(start) {}

void DStarLite::initialize() {
    const std::size_t n = graph_.num_vertices();
    g_.assign(n, INF);
    rhs_.assign(n, INF);
    open_list_.clear();
    km_ = 0.0;
    last_start_ = start_;
    nodes_recalculated_ = 0;
    peak_structural_bytes_ = 0;

    if (goal_ >= 0 && static_cast<std::size_t>(goal_) < n) {
        rhs_[static_cast<std::size_t>(goal_)] = 0.0;
        open_list_.insert_or_update(goal_, calculate_key(goal_));
    }
    initialized_ = true;
    record_peak_memory();
}

Key DStarLite::calculate_key(NodeId u) const {
    const Weight m = std::min(g_[static_cast<std::size_t>(u)], rhs_[static_cast<std::size_t>(u)]);
    return Key(safe_add(safe_add(m, graph_.heuristic(start_, u)), km_), m);
}

void DStarLite::update_vertex(NodeId u) {
    ++nodes_recalculated_;
    const auto idx = static_cast<std::size_t>(u);

    if (u != goal_) {
        Weight best_rhs = INF;
        for (const auto& edge : graph_.successors(u)) {
            const Weight cand = safe_add(edge.cost, g_[static_cast<std::size_t>(edge.to)]);
            if (cand < best_rhs) {
                best_rhs = cand;
            }
        }
        rhs_[idx] = best_rhs;
    }

    if (open_list_.contains(u)) {
        open_list_.remove(u);
    }

    if (!approx_equal(g_[idx], rhs_[idx])) {
        open_list_.insert_or_update(u, calculate_key(u));
    }
    record_peak_memory();
}

void DStarLite::compute_shortest_path() {
    const auto start_idx = static_cast<std::size_t>(start_);

    while (!open_list_.empty() &&
           (open_list_.top_key() < calculate_key(start_) ||
            rhs_[start_idx] > g_[start_idx] ||
            !approx_equal(rhs_[start_idx], g_[start_idx]))) {
        auto [u, k_old] = open_list_.pop();
        if (u == INVALID_NODE) {
            break;
        }
        const Key k_new = calculate_key(u);
        const auto u_idx = static_cast<std::size_t>(u);

        if (k_old < k_new) {
            open_list_.insert_or_update(u, k_new);
        } else if (g_[u_idx] > rhs_[u_idx]) {
            g_[u_idx] = rhs_[u_idx];
            for (NodeId pred : graph_.predecessors(u)) {
                update_vertex(pred);
            }
        } else {
            g_[u_idx] = INF;
            update_vertex(u);
            for (NodeId pred : graph_.predecessors(u)) {
                update_vertex(pred);
            }
        }
    }
    record_peak_memory();
}

SearchResult DStarLite::compute_initial_path() {
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

SearchResult DStarLite::replan(const std::vector<EdgeChange>& changes) {
    if (!initialized_) {
        return compute_initial_path();
    }
    nodes_recalculated_ = 0;
    peak_structural_bytes_ = structural_memory_bytes();

    const auto t0 = std::chrono::steady_clock::now();
    if (last_start_ != start_) {
        km_ += graph_.heuristic(last_start_, start_);
        last_start_ = start_;
    }

    for (const auto& change : changes) {
        if (change.u >= 0 && static_cast<std::size_t>(change.u) < graph_.num_vertices()) {
            update_vertex(change.u);
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

void DStarLite::update_start(NodeId new_start) {
    start_ = new_start;
}

Weight DStarLite::current_path_cost() const {
    if (!initialized_ || start_ < 0 || static_cast<std::size_t>(start_) >= g_.size()) {
        return INF;
    }
    return g_[static_cast<std::size_t>(start_)];
}

std::vector<NodeId> DStarLite::extract_path() const {
    std::vector<NodeId> path;
    if (std::isinf(current_path_cost())) {
        return path;
    }
    NodeId curr = start_;
    std::size_t max_steps = graph_.num_vertices() + 1;
    while (curr != INVALID_NODE && max_steps-- > 0) {
        path.push_back(curr);
        if (curr == goal_) {
            return path;
        }
        NodeId best_succ = INVALID_NODE;
        Weight best_val = INF;
        for (const auto& edge : graph_.successors(curr)) {
            const Weight cand = safe_add(edge.cost, g_[static_cast<std::size_t>(edge.to)]);
            if (cand < best_val) {
                best_val = cand;
                best_succ = edge.to;
            }
        }
        if (best_succ == INVALID_NODE || std::isinf(best_val)) {
            break;
        }
        curr = best_succ;
    }
    return {};
}

std::size_t DStarLite::structural_memory_bytes() const {
    return sizeof(*this) +
           g_.capacity() * sizeof(Weight) +
           rhs_.capacity() * sizeof(Weight) +
           open_list_.memory_bytes();
}

void DStarLite::record_peak_memory() {
    peak_structural_bytes_ = std::max(peak_structural_bytes_, structural_memory_bytes());
}

} // namespace incremental_search
