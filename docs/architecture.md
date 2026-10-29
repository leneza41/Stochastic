# Architecture, Equivalent Execution Conditions & Stochastic Simulation (Steps 1 & 2)

## 1. Objective Alignment
This codebase implements **Specific Objectives 1 and 2** of the research project:
1. *Implementar los algoritmos de búsqueda incremental seleccionados D\* Lite, LPA\*, Incremental A\* en C++/Python, bajo condiciones equivalentes de ejecución.*
2. *Diseñar un módulo de simulación que introduzca fallos estocásticos de aristas de manera controlada sobre grafos obtenidos de OpenStreetMap y MovingAI.*

---

## 2. Equivalent Execution Conditions (Step 1)
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

---

## 3. Controlled Stochastic Simulation Module (Step 2)

### 3.1. Sampling Frame & Graph Preparation (`GraphIO`)
Following *Muestra de la Investigación* and *Plan de recolección de datos*:
- **OpenStreetMap (`data/osm/`)**:
  - Supports `osmnx` road network acquisition and multi-scale urban street network generation (Small, Medium, Large).
  - Automatically extracts the **largest connected component** (`extract_largest_connected_component`), removes isolated nodes and self-loops, deduplicates parallel edges keeping the minimum positive length in meters, and serializes to the unified `.graph` format read identically by C++ and Python.
- **Moving AI Benchmarks (`data/movingai/`)**:
  - Standard `.map` parser (`load_movingai_map`) with explicit 4-neighbor (`MANHATTAN`) or 8-neighbor (`OCTILE`) connectivity and corner-cutting prevention.
  - Standard `.scen` parser (`load_movingai_scenarios`) indexing valid `(start, goal)` pairs across distance buckets.
- **Synthetic Controlled Graphs (`data/synthetic/`)**:
  - Parameterized generators for obstacle grids (`generate_synthetic_grid`) and random geometric road-like graphs (`generate_synthetic_random_graph`) with controlled size, density, and PRNG seed.

### 3.2. Stochastic Edge-Failure Modes (`FailureMode` & `StochasticSimulator`)
Each simulation run (`run_scenario`) represents one complete **experimental unit** (*unidad de muestreo*):
1. Clones the base graph into three identical instances for **D\* Lite**, **LPA\***, and **Incremental A\***.
2. Computes and validates the initial optimal path ($k = 0$).
3. Uses a seeded PRNG (`std::mt19937` in C++, `random.Random` in Python) to inject a sequence of $K$ repeated stochastic edge failures under one of four controlled regimes:
   - `ON_CURRENT_PATH`: Fails edges selected uniformly at random along the active optimal path.
   - `UNIFORM_RANDOM`: Fails active edges uniformly across the entire topology.
   - `MIXED_PROBABILISTIC`: Fails an active path edge with probability $p_{\text{path}}$ and a random network edge with probability $1 - p_{\text{path}}$.
   - `CONGESTION_PENALTY`: Multiplies the edge weight by a severe congestion factor $\alpha > 1$ instead of $\infty$.
4. Validates at every step that all three algorithms converge to the exact same optimal path cost (`cost_validated`) and exports structured CSV logs (`export_steps_csv` and `export_summaries_csv`).

### 3.3. Statistical Sample Size Determination (`SampleSizeCalculator`)
Implements the sampling formulas for $95\%$ confidence ($Z = 1.96$):
- **Infinite Population ($N > 99,999$, OSM & Synthetic)**:
  $$n = \left\lceil \frac{Z^2 \cdot S^2}{E^2} \right\rceil$$
- **Finite Population (Known $N$ in Moving AI `.scen` files)**:
  $$n = \left\lceil \frac{N \cdot Z^2 \cdot S^2}{E^2 (N - 1) + Z^2 \cdot S^2} \right\rceil$$
