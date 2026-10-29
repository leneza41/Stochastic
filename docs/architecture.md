# Architecture & Equivalent Execution Conditions (Step 1)

## 1. Objective Alignment
This codebase implements **Specific Objective 1** of the research project:
> *Implementar los algoritmos de búsqueda incremental seleccionados D\* Lite, LPA\*, Incremental A\* en C++/Python, bajo condiciones equivalentes de ejecución.*

## 2. Equivalent Execution Conditions
To guarantee that empirical differences in **compute time (ms)**, **recalculated nodes**, and **memory usage (MB)** reflect solely algorithmic properties rather than implementation biases, all three algorithms share:

1. **Identical Graph Topology & Heuristic Engine (`Graph`)**:
   - Directed weighted adjacency lists (`successors(u)`) and reverse adjacency lists (`predecessors(u)`).
   - Dynamic edge cost modification (`fail_edge`, `restore_edge`, `update_edge_cost`) returning standardized `EdgeChange` records.
   - Unified admissible and consistent heuristics (`EUCLIDEAN`, `MANHATTAN`, `OCTILE`, `HAVERSINE`, `ZERO`).
2. **Identical Priority Queue (`PriorityQueue`)**:
   - Lexicographical two-component keys `Key(k1, k2)` with floating-point tolerance (`EPSILON = 1e-9`).
   - Binary min-heap with lazy deletion and active key lookup (`insert_or_update`, `remove`, `contains`, `top_key`, `pop`).
3. **Identical Instrumentation & Metrics (`SearchResult`, `MemoryTracker`)**:
   - **Compute Time (ms)**: High-resolution monotonic clock (`std::chrono::steady_clock` in C++, `time.perf_counter` in Python) applied strictly to the replanning phase.
   - **Nodes Recalculated**: Internal counter incremented whenever a vertex state is updated or expanded during replanning.
   - **Memory Usage (MB)**: Peak structural byte accounting (`g`, `rhs`/`parent`, and `PriorityQueue` buffers) alongside OS Resident Set Size (`GetProcessMemoryInfo` / `getrusage` / `psutil`).

## 3. Implemented Algorithms

| Algorithm | Search Direction | State Vectors | Replanning Mechanism |
| :--- | :--- | :--- | :--- |
| **D\* Lite** (Koenig & Likhachev, 2002) | Backward (`goal` $\to$ `start`) | `g[u]`, `rhs[u]`, `km` | Updates `u` for changed edge `(u, v)` and propagates local inconsistency backward to `start`. |
| **LPA\*** (Koenig, Likhachev & Furcy, 2004) | Forward (`start` $\to$ `goal`) | `g[u]`, `rhs[u]`, `backpointer[u]` | Updates `v` for changed edge `(u, v)` and propagates local inconsistency forward to `goal`. |
| **Incremental A\*** (Dynamic Tree-Repair A\*) | Forward (`start` $\to$ `goal`) | `g[u]`, `parent[u]`, `closed[u]` | Invalidates only the affected subtree rooted at `v` when tree edge `(u, v)` fails, re-opening frontier boundary nodes. |
