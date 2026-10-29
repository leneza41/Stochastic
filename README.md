# Incremental Pathfinding under Stochastic Edge Failures

**Authors:**
- Leonardo Núñez Zavala
- Alan Yael Fonseca Ruiz
- Roberto Gordillo Herrera

## Overview
Empirical evaluation of incremental graph search algorithms (**D\* Lite**, **LPA\***, and **Incremental A\***) under real-time stochastic edge failures on networks derived from **OpenStreetMap**, **Moving AI**, and **synthetic graphs**.

See the complete **[Windows 11 Execution Manual](docs/WINDOWS11_EXECUTION_MANUAL.md)** for prerequisites and step-by-step instructions.

---

## 1. How to Populate the Input Datasets (`data/`)

The repository uses `data/osm/`, `data/movingai/`, and `data/synthetic/` for input graphs. Because large dataset files are ignored by Git, populate them locally on Windows 11 before or during execution using one of the following methods:

### Method A — Automatic Local Population (Recommended, Zero External Dependencies)
Run `prepare_datasets.py` from PowerShell at the repository root:
```powershell
python python\prepare_datasets.py
```
This automatically populates:
- **`data/osm/`**: `osm_urban_small.graph`, `osm_urban_medium.graph`, `osm_urban_large.graph` (preprocessed with largest connected component retention, isolated vertex removal, and positive edge lengths in meters).
- **`data/movingai/`**: `arena_rooms.map`, `dungeon_game.map`, `labyrinth_maze.map` and their corresponding `.map.scen` scenario files.
- **`data/synthetic/`**: `syn_grid_low_density.graph`, `syn_grid_high_density.graph`, `syn_random_geometric.graph`.

### Method B — Live Download from OpenStreetMap Overpass API
To download real road network extracts directly from the public **Overpass API** (`https://overpass-api.de/api/interpreter`) using Python's standard library:
```powershell
python python\prepare_datasets.py --download-online
```
*(Alternatively, if `osmnx` is installed via `pip install -r requirements.txt`, pass any city query to `GraphIO.load_or_generate_osm_network("Morelia, Michoacán, Mexico")`).*

### Method C — Adding Custom Moving AI Benchmark Files
1. Download any `.map` and `.map.scen` benchmark pair from [https://movingai.com/benchmarks/grids.html](https://movingai.com/benchmarks/grids.html).
2. Place both files inside `data/movingai/`.
3. Load them via `GraphIO.load_movingai_map()` and `GraphIO.load_movingai_scenarios()` in C++ or Python.

---

## 2. How to Generate the Research Results (Steps 3 & 4 Pipeline)

Run the complete benchmark, statistical evaluation, and figure/report generator specifying the **destination folder path** as an argument. *(If `data/` has not been populated yet, the pipeline automatically populates it first).*

### Option A — Python CLI (Zero External Dependencies Required on Windows 11)
```powershell
python python\run_research_pipeline.py "C:\Path\To\Output_Results"
```
To run with a full **30-run pilot sample per map** and **10 sequential stochastic failures per run**:
```powershell
python python\run_research_pipeline.py "C:\Path\To\Output_Results" --runs-per-map 30 --failures-per-run 10
```

### Option B — Windows 11 PowerShell Wrapper (`run_experiments.ps1`)
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_experiments.ps1 "C:\Path\To\Output_Results"
```

### Opening the Generated Research Dashboard
```powershell
Start-Process "C:\Path\To\Output_Results\04_research_report\RESEARCH_DASHBOARD.html"
```

### Generated Output Folder Structure
```text
<Output_Results>/
├── 01_raw_data/
│   ├── replanning_steps_raw.csv                       # Step-by-step measurements after every failure
│   ├── experimental_runs_raw.csv                      # Run-level averages (unidades de muestreo)
│   └── system_environment.json                        # CPU, RAM, and Windows 11 build metadata
├── 02_statistical_tables/
│   ├── table1_descriptive_by_algorithm_and_source.csv # Mean, SD, Median, IQR, P95, IQR Outliers
│   ├── table2_performance_by_failure_mode.csv         # Metrics stratified by stochastic failure mode
│   ├── table3_scale_sensitivity_analysis.csv          # Small vs. Medium vs. Large network scale analysis
│   ├── table4_sample_size_estimation.csv              # Z=1.96 infinite & finite sample size calculations
│   ├── table5_friedman_wilcoxon_tests.csv             # Friedman test, Kendall's W & Wilcoxon-Holm post-hoc
│   └── table6_efficiency_resource_tradeoff.csv        # Composite efficiency-resource trade-off index
├── 03_figures/
│   ├── fig1_replan_time_comparison.svg                # Replanning compute time (ms) by graph source
│   ├── fig2_nodes_recalculated_comparison.svg         # Recalculated nodes by graph source
│   ├── fig3_memory_consumption_comparison.svg         # Structural memory footprint (KB) by algorithm
│   ├── fig4_convergence_over_failure_sequence.svg     # Convergence trajectory over failures k = 1..K
│   ├── fig5_scale_effect_small_to_large.svg           # Scale sensitivity (Small, Medium, Large)
│   └── fig6_efficiency_memory_tradeoff_scatter.svg    # Trade-off plane: Time vs. Memory & Nodes
└── 04_research_report/
    ├── RESEARCH_REPORT.md                             # Complete scientific report in Markdown
    └── RESEARCH_DASHBOARD.html                        # Self-contained interactive HTML dashboard
```

---

## 3. Repository Structure

```text
Stochastic/
├── CMakeLists.txt                     # C++17 build configuration (core, verify, simulate, benchmark, tests)
├── run_experiments.ps1                # Windows 11 PowerShell launcher accepting destination path
├── pyproject.toml                     # Python package & pytest configuration
├── requirements.txt                   # Optional Python dependencies
├── docs/
│   ├── WINDOWS11_EXECUTION_MANUAL.md  # Complete manual & prerequisites for Windows 11
│   └── architecture.md                # Equivalent execution conditions & simulation design
├── data/                              # Input graph datasets (Step 2)
│   ├── osm/                           # OpenStreetMap multi-scale road networks (.graph)
│   ├── movingai/                      # Moving AI benchmark maps (.map) & scenarios (.scen)
│   └── synthetic/                     # Controlled synthetic grids & geometric graphs (.graph)
├── results/                           # Default experimental outputs directory
├── cpp/                               # C++17 Implementation
│   ├── include/incremental_search/    # Headers (Graph, PriorityQueue, Planners, GraphIO, Simulation, Evaluation)
│   ├── src/                           # Core sources, main_verify.cpp, main_simulate.cpp, main_benchmark_results.cpp
│   └── tests/                         # C++ unit, equivalence & simulation tests
└── python/                            # Python Implementation
    ├── src/incremental_search/        # Python package (Planners, GraphIO, Simulation, Evaluation, ReportBuilder)
    ├── prepare_datasets.py            # Generates/downloads OSM, Moving AI, and Synthetic datasets
    ├── verify_equivalence.py          # Step 1 equivalence verification runner
    ├── run_simulation.py              # Step 2 stochastic edge-failure simulation runner
    ├── run_research_pipeline.py       # Steps 3 & 4 full research benchmark, stats, SVG figures & reports
    └── tests/                         # Test suites (test_algorithms.py, test_simulation.py, test_evaluation.py)
```
