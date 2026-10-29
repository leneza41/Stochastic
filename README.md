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
├── CMakeLists.txt                     # C++17 build configuration
├── pyproject.toml                     # Python package & pytest configuration
├── requirements.txt                   # Python dependencies
├── docs/
│   └── architecture.md                # Equivalent execution conditions & design
├── data/                              # Input graph datasets (Step 2)
│   ├── osm/                           # OpenStreetMap road networks
│   ├── movingai/                      # Moving AI .map and .scen benchmarks
│   └── synthetic/                     # Controlled random & grid graphs
├── results/                           # Experimental outputs (Steps 3 & 4)
│   ├── raw/                           # Raw CSV benchmark logs
│   └── processed/                     # Aggregated tables, statistics & figures
├── cpp/                               # C++17 Implementation
│   ├── include/incremental_search/    # Headers (Graph, PriorityQueue, Planners)
│   ├── src/                           # Core sources & verification CLI
│   └── tests/                         # C++ unit & equivalence tests
└── python/                            # Python Implementation
    ├── src/incremental_search/        # Python package mirroring C++ API
    ├── verify_equivalence.py          # Equivalence verification runner
    └── tests/                         # Pytest test suite
```

## Step 1: Running Equivalence Verification & Tests

### Python
```powershell
python python/verify_equivalence.py
python -m pytest
```

### C++ (CMake / MSVC / GCC / Clang)
```powershell
mkdir build
cd build
cmake ..
cmake --build . --config Release
ctest --output-on-failure -C Release
```
