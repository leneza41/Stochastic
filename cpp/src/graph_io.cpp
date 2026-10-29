#include "incremental_search/graph_io.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace incremental_search {

namespace {

std::string heuristic_to_token(HeuristicType h) {
    switch (h) {
        case HeuristicType::MANHATTAN: return "manhattan";
        case HeuristicType::OCTILE:    return "octile";
        case HeuristicType::HAVERSINE: return "haversine";
        case HeuristicType::ZERO:      return "zero";
        case HeuristicType::EUCLIDEAN:
        default:                       return "euclidean";
    }
}

HeuristicType token_to_heuristic(const std::string& token) {
    if (token == "manhattan") return HeuristicType::MANHATTAN;
    if (token == "octile")    return HeuristicType::OCTILE;
    if (token == "haversine") return HeuristicType::HAVERSINE;
    if (token == "zero")      return HeuristicType::ZERO;
    return HeuristicType::EUCLIDEAN;
}

bool is_passable_movingai_char(char c) {
    return c == '.' || c == 'G' || c == 'S';
}

} // namespace

bool GraphIO::save_graph(const Graph& graph, const std::string& filepath) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        return false;
    }
    out << std::fixed << std::setprecision(8);
    out << "GRAPH_V1 " << graph.num_vertices() << " " << graph.num_edges() << " "
        << heuristic_to_token(graph.heuristic_type()) << "\n";

    for (std::size_t u = 0; u < graph.num_vertices(); ++u) {
        const auto c = graph.get_coordinates(static_cast<NodeId>(u));
        out << "V " << u << " " << c.x << " " << c.y << "\n";
    }

    for (std::size_t u = 0; u < graph.num_vertices(); ++u) {
        for (const auto& edge : graph.successors(static_cast<NodeId>(u))) {
            out << "E " << u << " " << edge.to << " " << edge.original_cost << "\n";
        }
    }
    return true;
}

Graph GraphIO::load_graph(const std::string& filepath) {
    std::ifstream in(filepath);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open graph file: " + filepath);
    }

    std::string header;
    std::size_t num_v = 0;
    std::size_t num_e = 0;
    std::string h_token;
    if (!(in >> header >> num_v >> num_e >> h_token) || header != "GRAPH_V1") {
        throw std::runtime_error("Invalid GRAPH_V1 header in: " + filepath);
    }

    Graph graph(num_v, token_to_heuristic(h_token));
    std::string line_type;
    while (in >> line_type) {
        if (line_type == "V") {
            NodeId u = 0;
            double x = 0.0;
            double y = 0.0;
            in >> u >> x >> y;
            graph.set_coordinates(u, x, y);
        } else if (line_type == "E") {
            NodeId u = 0;
            NodeId v = 0;
            Weight cost = 0.0;
            in >> u >> v >> cost;
            graph.add_edge(u, v, cost, false);
        } else {
            std::string dummy;
            std::getline(in, dummy);
        }
    }
    return graph;
}

MovingAIMap GraphIO::load_movingai_map(const std::string& map_filepath, int connectivity) {
    std::ifstream in(map_filepath);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open Moving AI .map file: " + map_filepath);
    }

    int height = 0;
    int width = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        std::istringstream iss(line);
        std::string key;
        iss >> key;
        if (key == "height") {
            iss >> height;
        } else if (key == "width") {
            iss >> width;
        } else if (key == "map") {
            break;
        }
    }

    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid dimensions in Moving AI .map file: " + map_filepath);
    }

    std::vector<std::string> grid_rows;
    grid_rows.reserve(static_cast<std::size_t>(height));
    while (static_cast<int>(grid_rows.size()) < height && std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (static_cast<int>(line.size()) >= width) {
            grid_rows.push_back(line.substr(0, static_cast<std::size_t>(width)));
        }
    }

    if (static_cast<int>(grid_rows.size()) != height) {
        throw std::runtime_error("Incomplete grid rows in Moving AI .map file: " + map_filepath);
    }

    MovingAIMap result;
    result.width = width;
    result.height = height;
    result.connectivity = (connectivity == 4) ? 4 : 8;
    result.grid_to_node.assign(static_cast<std::size_t>(width * height), INVALID_NODE);

    HeuristicType h_type = (result.connectivity == 4)
                               ? HeuristicType::MANHATTAN
                               : HeuristicType::OCTILE;
    result.graph = Graph(0, h_type);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (is_passable_movingai_char(grid_rows[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)])) {
                NodeId id = result.graph.add_vertex(static_cast<double>(x), static_cast<double>(y));
                result.grid_to_node[static_cast<std::size_t>(y * width + x)] = id;
            }
        }
    }

    const int dx4[4] = {1, -1, 0, 0};
    const int dy4[4] = {0, 0, 1, -1};
    const int dx8[4] = {1, 1, -1, -1};
    const int dy8[4] = {1, -1, 1, -1};
    constexpr Weight DIAG_COST = 1.4142135623730950488;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            NodeId u = result.node_at(x, y);
            if (u == INVALID_NODE) {
                continue;
            }
            for (int d = 0; d < 4; ++d) {
                NodeId v = result.node_at(x + dx4[d], y + dy4[d]);
                if (v != INVALID_NODE) {
                    result.graph.add_edge(u, v, 1.0, false);
                }
            }
            if (result.connectivity == 8) {
                for (int d = 0; d < 4; ++d) {
                    int nx = x + dx8[d];
                    int ny = y + dy8[d];
                    NodeId v = result.node_at(nx, ny);
                    // Prevent corner cutting: both cardinal neighbors must be passable
                    if (v != INVALID_NODE &&
                        result.node_at(nx, y) != INVALID_NODE &&
                        result.node_at(x, ny) != INVALID_NODE) {
                        result.graph.add_edge(u, v, DIAG_COST, false);
                    }
                }
            }
        }
    }

    return result;
}

std::vector<MovingAIScenario> GraphIO::load_movingai_scenarios(
    const std::string& scen_filepath,
    const MovingAIMap& map_data) {
    std::ifstream in(scen_filepath);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open Moving AI .scen file: " + scen_filepath);
    }

    std::vector<MovingAIScenario> scenarios;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line.rfind("version", 0) == 0) {
            continue;
        }
        std::istringstream iss(line);
        MovingAIScenario s;
        if (iss >> s.bucket >> s.map_name >> s.map_width >> s.map_height
                >> s.start_x >> s.start_y >> s.goal_x >> s.goal_y >> s.optimal_length) {
            s.start_node = map_data.node_at(s.start_x, s.start_y);
            s.goal_node = map_data.node_at(s.goal_x, s.goal_y);
            if (s.start_node != INVALID_NODE && s.goal_node != INVALID_NODE &&
                s.start_node != s.goal_node) {
                scenarios.push_back(s);
            }
        }
    }
    return scenarios;
}

Graph GraphIO::generate_synthetic_grid(int width,
                                       int height,
                                       double obstacle_ratio,
                                       int connectivity,
                                       uint32_t seed,
                                       double min_weight,
                                       double max_weight) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_real_distribution<double> weight_dist(min_weight, std::max(min_weight, max_weight));

    std::vector<uint8_t> blocked(static_cast<std::size_t>(width * height), 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if ((x == 0 && y == 0) || (x == width - 1 && y == height - 1)) {
                continue;
            }
            if (prob_dist(rng) < obstacle_ratio) {
                blocked[static_cast<std::size_t>(y * width + x)] = 1;
            }
        }
    }

    HeuristicType h_type = (connectivity == 4) ? HeuristicType::MANHATTAN : HeuristicType::OCTILE;
    if (min_weight > 1.0 || max_weight > 1.0) {
        h_type = HeuristicType::EUCLIDEAN;
    }

    Graph raw_graph(0, h_type);
    std::vector<NodeId> grid_to_node(static_cast<std::size_t>(width * height), INVALID_NODE);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const auto idx = static_cast<std::size_t>(y * width + x);
            if (!blocked[idx]) {
                grid_to_node[idx] = raw_graph.add_vertex(static_cast<double>(x), static_cast<double>(y));
            }
        }
    }

    auto node_at = [&](int x, int y) -> NodeId {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            return INVALID_NODE;
        }
        return grid_to_node[static_cast<std::size_t>(y * width + x)];
    };

    constexpr Weight DIAG_SCALE = 1.4142135623730950488;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            NodeId u = node_at(x, y);
            if (u == INVALID_NODE) {
                continue;
            }
            if (x + 1 < width && node_at(x + 1, y) != INVALID_NODE) {
                Weight w = weight_dist(rng);
                raw_graph.add_edge(u, node_at(x + 1, y), w, true);
            }
            if (y + 1 < height && node_at(x, y + 1) != INVALID_NODE) {
                Weight w = weight_dist(rng);
                raw_graph.add_edge(u, node_at(x, y + 1), w, true);
            }
            if (connectivity == 8) {
                if (x + 1 < width && y + 1 < height &&
                    node_at(x + 1, y + 1) != INVALID_NODE &&
                    node_at(x + 1, y) != INVALID_NODE &&
                    node_at(x, y + 1) != INVALID_NODE) {
                    Weight w = weight_dist(rng) * DIAG_SCALE;
                    raw_graph.add_edge(u, node_at(x + 1, y + 1), w, true);
                }
                if (x - 1 >= 0 && y + 1 < height &&
                    node_at(x - 1, y + 1) != INVALID_NODE &&
                    node_at(x - 1, y) != INVALID_NODE &&
                    node_at(x, y + 1) != INVALID_NODE) {
                    Weight w = weight_dist(rng) * DIAG_SCALE;
                    raw_graph.add_edge(u, node_at(x - 1, y + 1), w, true);
                }
            }
        }
    }

    return extract_largest_connected_component(raw_graph);
}

Graph GraphIO::generate_synthetic_random_graph(std::size_t num_vertices,
                                               std::size_t avg_degree,
                                               uint32_t seed,
                                               double area_size) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> coord_dist(0.0, area_size);
    std::uniform_real_distribution<double> roughness_dist(1.0, 1.35);

    Graph raw_graph(num_vertices, HeuristicType::EUCLIDEAN);
    for (std::size_t u = 0; u < num_vertices; ++u) {
        raw_graph.set_coordinates(static_cast<NodeId>(u), coord_dist(rng), coord_dist(rng));
    }

    const std::size_t k = std::max<std::size_t>(2, avg_degree);
    for (std::size_t u = 0; u < num_vertices; ++u) {
        std::vector<std::pair<double, NodeId>> dists;
        dists.reserve(num_vertices - 1);
        const auto cu = raw_graph.get_coordinates(static_cast<NodeId>(u));
        for (std::size_t v = 0; v < num_vertices; ++v) {
            if (u == v) {
                continue;
            }
            const auto cv = raw_graph.get_coordinates(static_cast<NodeId>(v));
            double d = std::hypot(cu.x - cv.x, cu.y - cv.y);
            dists.emplace_back(d, static_cast<NodeId>(v));
        }
        std::size_t limit = std::min(k, dists.size());
        std::partial_sort(dists.begin(),
                          dists.begin() + static_cast<std::ptrdiff_t>(limit),
                          dists.end());
        for (std::size_t i = 0; i < limit; ++i) {
            double euclidean = std::max(0.1, dists[i].first);
            Weight road_len = euclidean * roughness_dist(rng);
            raw_graph.add_edge(static_cast<NodeId>(u), dists[i].second, road_len, true);
        }
    }

    return extract_largest_connected_component(raw_graph);
}

Graph GraphIO::extract_largest_connected_component(const Graph& input_graph) {
    const std::size_t n = input_graph.num_vertices();
    if (n == 0) {
        return Graph(0, input_graph.heuristic_type());
    }

    std::vector<uint8_t> visited(n, 0);
    std::vector<NodeId> best_component;

    for (std::size_t i = 0; i < n; ++i) {
        if (visited[i]) {
            continue;
        }
        std::vector<NodeId> comp;
        std::queue<NodeId> q;
        q.push(static_cast<NodeId>(i));
        visited[i] = 1;

        while (!q.empty()) {
            NodeId u = q.front();
            q.pop();
            comp.push_back(u);

            for (const auto& edge : input_graph.successors(u)) {
                if (edge.to >= 0 && !visited[static_cast<std::size_t>(edge.to)] &&
                    !std::isinf(edge.original_cost) && edge.original_cost > 0.0) {
                    visited[static_cast<std::size_t>(edge.to)] = 1;
                    q.push(edge.to);
                }
            }
            for (NodeId pred : input_graph.predecessors(u)) {
                if (pred >= 0 && !visited[static_cast<std::size_t>(pred)]) {
                    Weight c = input_graph.get_original_edge_cost(pred, u);
                    if (!std::isinf(c) && c > 0.0) {
                        visited[static_cast<std::size_t>(pred)] = 1;
                        q.push(pred);
                    }
                }
            }
        }

        if (comp.size() > best_component.size()) {
            best_component = std::move(comp);
        }
    }

    std::sort(best_component.begin(), best_component.end());
    std::vector<NodeId> old_to_new(n, INVALID_NODE);
    Graph clean_graph(best_component.size(), input_graph.heuristic_type());

    for (std::size_t new_idx = 0; new_idx < best_component.size(); ++new_idx) {
        NodeId old_u = best_component[new_idx];
        old_to_new[static_cast<std::size_t>(old_u)] = static_cast<NodeId>(new_idx);
        const auto coords = input_graph.get_coordinates(old_u);
        clean_graph.set_coordinates(static_cast<NodeId>(new_idx), coords.x, coords.y);
    }

    for (NodeId old_u : best_component) {
        NodeId new_u = old_to_new[static_cast<std::size_t>(old_u)];
        for (const auto& edge : input_graph.successors(old_u)) {
            if (edge.to == old_u || std::isinf(edge.original_cost) || edge.original_cost <= 0.0) {
                continue; // Skip self-loops and non-positive edges
            }
            NodeId new_v = old_to_new[static_cast<std::size_t>(edge.to)];
            if (new_v != INVALID_NODE) {
                clean_graph.add_edge(new_u, new_v, edge.original_cost, false);
            }
        }
    }

    return clean_graph;
}

bool GraphIO::is_reachable(const Graph& graph, NodeId start, NodeId goal) {
    const std::size_t n = graph.num_vertices();
    if (start < 0 || goal < 0 ||
        static_cast<std::size_t>(start) >= n ||
        static_cast<std::size_t>(goal) >= n) {
        return false;
    }
    if (start == goal) {
        return true;
    }

    std::vector<uint8_t> visited(n, 0);
    std::queue<NodeId> q;
    q.push(start);
    visited[static_cast<std::size_t>(start)] = 1;

    while (!q.empty()) {
        NodeId u = q.front();
        q.pop();
        if (u == goal) {
            return true;
        }
        for (const auto& edge : graph.successors(u)) {
            if (!std::isinf(edge.cost) && !visited[static_cast<std::size_t>(edge.to)]) {
                visited[static_cast<std::size_t>(edge.to)] = 1;
                q.push(edge.to);
            }
        }
    }
    return false;
}

std::pair<NodeId, NodeId> GraphIO::sample_valid_od_pair(const Graph& graph,
                                                        uint32_t seed,
                                                        std::size_t max_attempts) {
    const std::size_t n = graph.num_vertices();
    if (n < 2) {
        return {INVALID_NODE, INVALID_NODE};
    }

    std::mt19937 rng(seed);
    std::uniform_int_distribution<NodeId> node_dist(0, static_cast<NodeId>(n - 1));

    for (std::size_t attempt = 0; attempt < max_attempts; ++attempt) {
        NodeId s = node_dist(rng);
        NodeId t = node_dist(rng);
        if (s != t && is_reachable(graph, s, t)) {
            return {s, t};
        }
    }
    return {INVALID_NODE, INVALID_NODE};
}

} // namespace incremental_search
