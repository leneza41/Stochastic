#include "incremental_search/incremental_a_star.hpp"
#include "incremental_search/memory_tracker.hpp"
#include <algorithm>
#include <chrono>
#include <queue>

namespace incremental_search {

IncrementalAStar::IncrementalAStar(const Graph& graph, NodeId start, NodeId goal)
    : IncrementalPlanner(graph, start, goal) {}

void IncrementalAStar::initialize() {
    const std::size_t n = graph_.num_vertices();
    g_.assign(n, INF);
    parent_.assign(n, INVALID_NODE);
    closed_.assign(n, 0);
    open_list_.clear();
    nodes_recalculated_ = 0;
    peak_structural_bytes_ = 0;

    if (start_ >= 0 && static_cast<std::size_t>(start_) < n) {
        g_[static_cast<std::size_t>(start_)] = 0.0;
        open_list_.insert_or_update(start_, calculate_key(start_));
    }
    initialized_ = true;
    record_peak_memory();
}

Key IncrementalAStar::calculate_key(NodeId u) const {
    const Weight gu = g_[static_cast<std::size_t>(u)];
    return Key(safe_add(gu, graph_.heuristic(u, goal_)), gu);
}

void IncrementalAStar::invalidate_subtree(NodeId root) {
    const std::size_t n = graph_.num_vertices();
    if (root < 0 || static_cast<std::size_t>(root) >= n) {
        return;
    }

    // Build children adjacency from current parent_ tree to traverse only affected subtree
    std::vector<NodeId> queue;
    std::vector<uint8_t> in_subtree(n, 0);
    queue.push_back(root);
    in_subtree[static_cast<std::size_t>(root)] = 1;

    // Collect all descendants in the search tree rooted at root
    bool added = true;
    while (added) {
        added = false;
        for (std::size_t v = 0; v < n; ++v) {
            if (!in_subtree[v] && parent_[v] != INVALID_NODE &&
                in_subtree[static_cast<std::size_t>(parent_[v])]) {
                in_subtree[v] = 1;
                queue.push_back(static_cast<NodeId>(v));
                added = true;
            }
        }
    }

    // Reset invalidated vertices
    for (NodeId v : queue) {
        const auto v_idx = static_cast<std::size_t>(v);
        g_[v_idx] = INF;
        parent_[v_idx] = INVALID_NODE;
        closed_[v_idx] = 0;
        if (open_list_.contains(v)) {
            open_list_.remove(v);
        }
        ++nodes_recalculated_;
    }

    // Reconnect invalidated vertices from best valid closed non-invalidated predecessors
    for (NodeId v : queue) {
        const auto v_idx = static_cast<std::size_t>(v);
        Weight best_g = INF;
        NodeId best_pred = INVALID_NODE;
        for (NodeId pred : graph_.predecessors(v)) {
            const auto pred_idx = static_cast<std::size_t>(pred);
            if (!in_subtree[pred_idx] && closed_[pred_idx] && !std::isinf(g_[pred_idx])) {
                const Weight cand = safe_add(g_[pred_idx], graph_.get_edge_cost(pred, v));
                if (cand < best_g) {
                    best_g = cand;
                    best_pred = pred;
                }
            }
        }
        if (best_pred != INVALID_NODE && !std::isinf(best_g)) {
            g_[v_idx] = best_g;
            parent_[v_idx] = best_pred;
            open_list_.insert_or_update(v, calculate_key(v));
        }
    }
    record_peak_memory();
}

void IncrementalAStar::run_a_star_repair() {
    const auto goal_idx = static_cast<std::size_t>(goal_);

    while (!open_list_.empty()) {
        // Terminate once the open list top key is >= optimal goal cost already closed
        if (!std::isinf(g_[goal_idx]) && closed_[goal_idx] &&
            !(open_list_.top_key() < calculate_key(goal_))) {
            break;
        }

        auto [u, k_old] = open_list_.pop();
        if (u == INVALID_NODE) {
            break;
        }
        const auto u_idx = static_cast<std::size_t>(u);
        closed_[u_idx] = 1;
        ++nodes_recalculated_;

        for (const auto& edge : graph_.successors(u)) {
            if (std::isinf(edge.cost)) {
                continue;
            }
            const NodeId v = edge.to;
            const auto v_idx = static_cast<std::size_t>(v);
            const Weight tentative_g = safe_add(g_[u_idx], edge.cost);

            if (tentative_g + EPSILON < g_[v_idx]) {
                g_[v_idx] = tentative_g;
                parent_[v_idx] = u;
                closed_[v_idx] = 0;
                open_list_.insert_or_update(v, calculate_key(v));
                ++nodes_recalculated_;
            }
        }
    }
    record_peak_memory();
}

SearchResult IncrementalAStar::compute_initial_path() {
    if (!initialized_) {
        initialize();
    }
    nodes_recalculated_ = 0;
    peak_structural_bytes_ = structural_memory_bytes();

    const auto t0 = std::chrono::steady_clock::now();
    run_a_star_repair();
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

SearchResult IncrementalAStar::replan(const std::vector<EdgeChange>& changes) {
    if (!initialized_) {
        return compute_initial_path();
    }
    nodes_recalculated_ = 0;
    peak_structural_bytes_ = structural_memory_bytes();

    const auto t0 = std::chrono::steady_clock::now();
    for (const auto& change : changes) {
        if (change.u < 0 || change.v < 0 ||
            static_cast<std::size_t>(change.u) >= graph_.num_vertices() ||
            static_cast<std::size_t>(change.v) >= graph_.num_vertices()) {
            continue;
        }
        const auto u_idx = static_cast<std::size_t>(change.u);
        const auto v_idx = static_cast<std::size_t>(change.v);

        if (change.new_cost > change.old_cost) {
            // Edge cost increased or failed: if tree uses (u -> v), invalidate subtree at v
            if (parent_[v_idx] == change.u) {
                invalidate_subtree(change.v);
            }
        } else if (change.new_cost < change.old_cost) {
            // Edge cost decreased: check if it improves v
            const Weight cand = safe_add(g_[u_idx], change.new_cost);
            if (cand + EPSILON < g_[v_idx]) {
                g_[v_idx] = cand;
                parent_[v_idx] = change.u;
                closed_[v_idx] = 0;
                open_list_.insert_or_update(change.v, calculate_key(change.v));
                ++nodes_recalculated_;
            }
        }
    }

    run_a_star_repair();
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

void IncrementalAStar::update_start(NodeId new_start) {
    if (new_start == start_) {
        return;
    }
    start_ = new_start;
    initialize();
}

Weight IncrementalAStar::current_path_cost() const {
    if (!initialized_ || goal_ < 0 || static_cast<std::size_t>(goal_) >= g_.size()) {
        return INF;
    }
    return g_[static_cast<std::size_t>(goal_)];
}

std::vector<NodeId> IncrementalAStar::extract_path() const {
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
        curr = parent_[static_cast<std::size_t>(curr)];
    }
    return {};
}

std::size_t IncrementalAStar::structural_memory_bytes() const {
    return sizeof(*this) +
           g_.capacity() * sizeof(Weight) +
           parent_.capacity() * sizeof(NodeId) +
           closed_.capacity() * sizeof(uint8_t) +
           open_list_.memory_bytes();
}

void IncrementalAStar::record_peak_memory() {
    peak_structural_bytes_ = std::max(peak_structural_bytes_, structural_memory_bytes());
}

} // namespace incremental_search
