#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace incremental_search {

using NodeId = int32_t;
using Weight = double;

constexpr NodeId INVALID_NODE = -1;
constexpr Weight INF = std::numeric_limits<Weight>::infinity();
constexpr Weight EPSILON = 1e-9;

inline bool approx_equal(Weight a, Weight b, Weight tol = EPSILON) {
    if (std::isinf(a) && std::isinf(b)) {
        return (a > 0) == (b > 0);
    }
    return std::fabs(a - b) <= tol;
}

inline Weight safe_add(Weight a, Weight b) {
    if (std::isinf(a) || std::isinf(b)) {
        return INF;
    }
    return a + b;
}

/**
 * Lexicographical two-component priority key [k1, k2] used equivalently
 * across D* Lite, LPA*, and Incremental A*.
 */
struct Key {
    Weight k1{INF};
    Weight k2{INF};

    constexpr Key() = default;
    constexpr Key(Weight first, Weight second) : k1(first), k2(second) {}

    bool operator<(const Key& other) const {
        if (!approx_equal(k1, other.k1)) {
            return k1 < other.k1;
        }
        return k2 < other.k2 - EPSILON;
    }

    bool operator>(const Key& other) const {
        return other < *this;
    }

    bool operator<=(const Key& other) const {
        return !(*this > other);
    }

    bool operator>=(const Key& other) const {
        return !(*this < other);
    }

    bool operator==(const Key& other) const {
        return approx_equal(k1, other.k1) && approx_equal(k2, other.k2);
    }

    bool operator!=(const Key& other) const {
        return !(*this == other);
    }
};

/**
 * Represents a directed edge weight modification or stochastic edge failure.
 */
struct EdgeChange {
    NodeId u{INVALID_NODE};
    NodeId v{INVALID_NODE};
    Weight old_cost{INF};
    Weight new_cost{INF};
};

/**
 * Quantitative metrics recorded per initial search or replanning event.
 * Matches the research protocol variables:
 *   1. compute_time_ms: execution time in milliseconds (std::chrono::steady_clock)
 *   2. nodes_recalculated: vertices processed/updated during search/replanning
 *   3. memory_mb: resident/structural memory footprint in megabytes
 */
struct SearchResult {
    std::string algorithm_name;
    bool path_found{false};
    Weight path_cost{INF};
    std::vector<NodeId> path;
    double compute_time_ms{0.0};
    std::size_t nodes_recalculated{0};
    double memory_mb{0.0};
    double process_rss_mb{0.0};
};

} // namespace incremental_search
