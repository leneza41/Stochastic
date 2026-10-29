# Incremental Pathfinding under Stochastic Edge Failures

**Authors:**
- Leonardo Núñez Zavala
- Alan Yael Fonseca Ruiz
- Roberto Gordillo Herrera

## Overview
Empirical evaluation of incremental graph search algorithms (**D\* Lite**, **LPA\***, and **Incremental A\***) under real-time stochastic edge failures on networks derived from **OpenStreetMap**, **Moving AI**, and **synthetic graphs**.

See the complete **[Windows 11 Execution Manual](docs/WINDOWS11_EXECUTION_MANUAL.md)** for prerequisites and step-by-step instructions.

---

## Quick Start on Windows 11 (Steps 3 & 4 Research Pipeline)

Run the complete benchmark, statistical evaluation, and figure/report generator specifying the **destination folder path** as an argument:

### Option A — Python CLI (Zero External Dependencies Required on Windows 11)
```powershell
python python\run_research_pipeline.py "C:\Path\To\Output_Results"
```
To run with a full 30-run pilot sample per map and 10 sequential stochastic failures per run:
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

---

## Repository Structure

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
    ├── prepare_datasets.py            # Generates/prepares OSM, Moving AI, and Synthetic datasets
    ├── verify_equivalence.py          # Step 1 equivalence verification runner
    ├── run_simulation.py              # Step 2 stochastic edge-failure simulation runner
    ├── run_research_pipeline.py       # Steps 3 & 4 full research benchmark, stats, SVG figures & reports
    └── tests/                         # Test suites (test_algorithms.py, test_simulation.py, test_evaluation.py)
```
