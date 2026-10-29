# Incremental Pathfinding under Stochastic Edge Failures

**Authors:**
- Leonardo Núñez Zavala
- Alan Yael Fonseca Ruiz
- Roberto Gordillo Herrera

## Overview
Empirical evaluation of incremental graph search algorithms (**D\* Lite**, **LPA\***, and **Incremental A\***) under real-time stochastic edge failures on networks derived from **OpenStreetMap**, **Moving AI**, and **synthetic graphs**.

## Repository Structure

```text
Stochastic/
├── CMakeLists.txt                     # C++17 build configuration (core, verify, simulate, tests)
├── pyproject.toml                     # Python package & pytest configuration
├── requirements.txt                   # Python dependencies
├── docs/
│   └── architecture.md                # Equivalent execution conditions & simulation design
├── data/                              # Input graph datasets (Step 2)
│   ├── osm/                           # OpenStreetMap multi-scale road networks (.graph)
│   ├── movingai/                      # Moving AI benchmark maps (.map) & scenarios (.scen)
│   └── synthetic/                     # Controlled synthetic grids & geometric graphs (.graph)
├── results/                           # Experimental outputs (Steps 2, 3 & 4)
│   ├── raw/                           # Raw CSV step-by-step and run-summary logs
│   └── processed/                     # Aggregated tables, statistics & figures
├── cpp/                               # C++17 Implementation
│   ├── include/incremental_search/    # Headers (Graph, PriorityQueue, Planners, GraphIO, Simulation)
│   ├── src/                           # Core sources, main_verify.cpp, main_simulate.cpp
│   └── tests/                         # C++ unit, equivalence & simulation tests
└── python/                            # Python Implementation
    ├── src/incremental_search/        # Python package mirroring C++ API
    ├── prepare_datasets.py            # Generates/prepares OSM, Moving AI, and Synthetic datasets
    ├── verify_equivalence.py          # Step 1 equivalence verification runner
    ├── run_simulation.py              # Step 2 stochastic edge-failure simulation runner
    └── tests/                         # Pytest test suites (test_algorithms.py, test_simulation.py)
```

## Running the Project

### Python (Steps 1 & 2)
```powershell
# 1. Prepare OSM, Moving AI (.map/.scen), and Synthetic datasets
python python/prepare_datasets.py

# 2. Run Step 1 equivalence verification
python python/verify_equivalence.py

# 3. Run Step 2 stochastic edge-failure simulations & export CSVs
python python/run_simulation.py

# 4. Run full test suite
python -m pytest
```

### C++17 (CMake / MSVC / GCC / Clang)
```powershell
mkdir build
cd build
cmake ..
cmake --build . --config Release
ctest --output-on-failure -C Release
./Release/run_simulation ..
```
