#include "incremental_search/graph.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace incremental_search {

namespace {
constexpr double EARTH_RADIUS_METERS = 6371000.0;
constexpr double DEG_TO_RAD = 3.14159265358979323846 / 180.0;
} // namespace

Graph::Graph(std::size_t num_vertices, HeuristicType heuristic_type)
    : adj_(num_vertices),
      rev_adj_(num_vertices),
      coords_(num_vertices),
      num_edges_(0),
      heuristic_type_(heuristic_type) {}

void Graph::ensure_vertex(NodeId u) {
    if (u < 0) {
        throw std::invalid_argument("Vertex ID must be non-negative");
    }
    auto idx = static_cast<std::size_t>(u);
    if (idx >= adj_.size()) {
        adj_.resize(idx + 1);
        rev_adj_.resize(idx + 1);
        coords_.resize(idx + 1);
    }
}

NodeId Graph::add_vertex(double x, double y) {
    auto id = static_cast<NodeId>(adj_.size());
    adj_.emplace_back();
    rev_adj_.emplace_back();
    coords_.push_back(Coordinates{x, y});
    return id;
}

void Graph::set_coordinates(NodeId u, double x, double y) {
    ensure_vertex(u);
    coords_[static_cast<std::size_t>(u)] = Coordinates{x, y};
}

Coordinates Graph::get_coordinates(NodeId u) const {
    if (u < 0 || static_cast<std::size_t>(u) >= coords_.size()) {
        return Coordinates{0.0, 0.0};
    }
    return coords_[static_cast<std::size_t>(u)];
}

void Graph::add_edge(NodeId u, NodeId v, Weight cost, bool bidirectional) {
    ensure_vertex(std::max(u, v));
    auto& out_edges = adj_[static_cast<std::size_t>(u)];
    for (auto& e : out_edges) {
        if (e.to == v) {
            if (cost < e.original_cost) {
                e.cost = cost;
                e.original_cost = cost;
                e.failed = std::isinf(cost);
            }
            if (bidirectional) {
                add_edge(v, u, cost, false);
            }
            return;
        }
    }
    out_edges.push_back(Edge{v, cost, cost, std::isinf(cost)});
    rev_adj_[static_cast<std::size_t>(v)].push_back(u);
    ++num_edges_;

    if (bidirectional) {
        add_edge(v, u, cost, false);
    }
}

bool Graph::has_edge(NodeId u, NodeId v) const {
    if (u < 0 || static_cast<std::size_t>(u) >= adj_.size()) {
        return false;
    }
    for (const auto& e : adj_[static_cast<std::size_t>(u)]) {
        if (e.to == v) {
            return true;
        }
    }
    return false;
}

Weight Graph::get_edge_cost(NodeId u, NodeId v) const {
    if (u < 0 || static_cast<std::size_t>(u) >= adj_.size()) {
        return INF;
    }
    for (const auto& e : adj_[static_cast<std::size_t>(u)]) {
        if (e.to == v) {
            return e.cost;
        }
    }
    return INF;
}

Weight Graph::get_original_edge_cost(NodeId u, NodeId v) const {
    if (u < 0 || static_cast<std::size_t>(u) >= adj_.size()) {
        return INF;
    }
    for (const auto& e : adj_[static_cast<std::size_t>(u)]) {
        if (e.to == v) {
            return e.original_cost;
        }
    }
    return INF;
}

EdgeChange Graph::update_edge_cost(NodeId u, NodeId v, Weight new_cost) {
    if (u < 0 || static_cast<std::size_t>(u) >= adj_.size()) {
        return EdgeChange{u, v, INF, INF};
    }
    for (auto& e : adj_[static_cast<std::size_t>(u)]) {
        if (e.to == v) {
            Weight old_cost = e.cost;
            e.cost = new_cost;
            e.failed = std::isinf(new_cost);
            return EdgeChange{u, v, old_cost, new_cost};
        }
    }
    return EdgeChange{u, v, INF, INF};
}

EdgeChange Graph::fail_edge(NodeId u, NodeId v) {
    return update_edge_cost(u, v, INF);
}

EdgeChange Graph::restore_edge(NodeId u, NodeId v) {
    if (u < 0 || static_cast<std::size_t>(u) >= adj_.size()) {
        return EdgeChange{u, v, INF, INF};
    }
    for (auto& e : adj_[static_cast<std::size_t>(u)]) {
        if (e.to == v) {
            Weight old_cost = e.cost;
            e.cost = e.original_cost;
            e.failed = false;
            return EdgeChange{u, v, old_cost, e.cost};
        }
    }
    return EdgeChange{u, v, INF, INF};
}

void Graph::restore_all_edges() {
    for (auto& out_edges : adj_) {
        for (auto& e : out_edges) {
            e.cost = e.original_cost;
            e.failed = false;
        }
    }
}

const std::vector<Edge>& Graph::successors(NodeId u) const {
    static const std::vector<Edge> empty_edges;
    if (u < 0 || static_cast<std::size_t>(u) >= adj_.size()) {
        return empty_edges;
    }
    return adj_[static_cast<std::size_t>(u)];
}

const std::vector<NodeId>& Graph::predecessors(NodeId u) const {
    static const std::vector<NodeId> empty_nodes;
    if (u < 0 || static_cast<std::size_t>(u) >= rev_adj_.size()) {
        return empty_nodes;
    }
    return rev_adj_[static_cast<std::size_t>(u)];
}

Weight Graph::heuristic(NodeId u, NodeId v) const {
    if (u == v || heuristic_type_ == HeuristicType::ZERO) {
        return 0.0;
    }
    const auto cu = get_coordinates(u);
    const auto cv = get_coordinates(v);
    const double dx = std::fabs(cu.x - cv.x);
    const double dy = std::fabs(cu.y - cv.y);

    switch (heuristic_type_) {
        case HeuristicType::MANHATTAN:
            return dx + dy;
        case HeuristicType::OCTILE: {
            constexpr double SQRT2_MINUS_1 = 0.41421356237309515;
            return std::max(dx, dy) + SQRT2_MINUS_1 * std::min(dx, dy);
        }
        case HeuristicType::HAVERSINE: {
            // Coordinates stored as x = lon, y = lat in degrees
            const double lat1 = cu.y * DEG_TO_RAD;
            const double lat2 = cv.y * DEG_TO_RAD;
            const double dlat = (cv.y - cu.y) * DEG_TO_RAD;
            const double dlon = (cv.x - cu.x) * DEG_TO_RAD;
            const double a = std::sin(dlat * 0.5) * std::sin(dlat * 0.5) +
                             std::cos(lat1) * std::cos(lat2) *
                                 std::sin(dlon * 0.5) * std::sin(dlon * 0.5);
            const double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(std::max(0.0, 1.0 - a)));
            return EARTH_RADIUS_METERS * c;
        }
        case HeuristicType::EUCLIDEAN:
        default:
            return std::hypot(dx, dy);
    }
}

std::size_t Graph::memory_bytes() const {
    std::size_t total = sizeof(*this);
    total += coords_.capacity() * sizeof(Coordinates);
    for (const auto& vec : adj_) {
        total += sizeof(vec) + vec.capacity() * sizeof(Edge);
    }
    for (const auto& vec : rev_adj_) {
        total += sizeof(vec) + vec.capacity() * sizeof(NodeId);
    }
    return total;
}

} // namespace incremental_search
