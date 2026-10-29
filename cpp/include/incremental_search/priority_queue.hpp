#pragma once

#include "incremental_search/types.hpp"
#include <queue>
#include <unordered_map>
#include <vector>

namespace incremental_search {

/**
 * Equivalent priority queue used across D* Lite, LPA*, and Incremental A*.
 * Uses a binary min-heap with lazy deletion and an active key table so that
 * insert, update, remove, contains, and top_key operate under identical
 * asymptotic and constant-factor conditions.
 */
class PriorityQueue {
public:
    PriorityQueue() = default;

    void clear() {
        while (!heap_.empty()) {
            heap_.pop();
        }
        active_keys_.clear();
    }

    bool empty() const {
        clean_top();
        return active_keys_.empty();
    }

    std::size_t size() const {
        return active_keys_.size();
    }

    bool contains(NodeId u) const {
        return active_keys_.find(u) != active_keys_.end();
    }

    void insert_or_update(NodeId u, const Key& key) {
        active_keys_[u] = key;
        heap_.push(HeapEntry{key, u});
    }

    void remove(NodeId u) {
        active_keys_.erase(u);
    }

    Key top_key() const {
        clean_top();
        if (heap_.empty()) {
            return Key(INF, INF);
        }
        return heap_.top().key;
    }

    NodeId top_node() const {
        clean_top();
        if (heap_.empty()) {
            return INVALID_NODE;
        }
        return heap_.top().node;
    }

    std::pair<NodeId, Key> pop() {
        clean_top();
        if (heap_.empty()) {
            return {INVALID_NODE, Key(INF, INF)};
        }
        HeapEntry top = heap_.top();
        heap_.pop();
        active_keys_.erase(top.node);
        return {top.node, top.key};
    }

    std::size_t memory_bytes() const {
        return heap_.size() * sizeof(HeapEntry) +
               active_keys_.size() * (sizeof(NodeId) + sizeof(Key) + sizeof(void*));
    }

private:
    struct HeapEntry {
        Key key;
        NodeId node;

        bool operator>(const HeapEntry& other) const {
            if (key != other.key) {
                return key > other.key;
            }
            return node > other.node;
        }
    };

    void clean_top() const {
        while (!heap_.empty()) {
            const HeapEntry& entry = heap_.top();
            auto it = active_keys_.find(entry.node);
            if (it != active_keys_.end() && it->second == entry.key) {
                break;
            }
            heap_.pop();
        }
    }

    mutable std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap_;
    std::unordered_map<NodeId, Key> active_keys_;
};

} // namespace incremental_search
