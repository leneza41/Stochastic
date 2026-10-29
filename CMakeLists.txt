cmake_minimum_required(VERSION 3.16)
project(IncrementalSearchStochastic VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /O2 /EHsc)
else()
    add_compile_options(-Wall -Wextra -O3)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)

set(CORE_SOURCES
    cpp/src/graph.cpp
    cpp/src/lpa_star.cpp
    cpp/src/d_star_lite.cpp
    cpp/src/incremental_a_star.cpp
    cpp/src/graph_io.cpp
    cpp/src/simulation.cpp
)

add_library(incremental_search_core STATIC ${CORE_SOURCES})

add_executable(verify_equivalence cpp/src/main_verify.cpp)
target_link_libraries(verify_equivalence PRIVATE incremental_search_core)

add_executable(run_simulation cpp/src/main_simulate.cpp)
target_link_libraries(run_simulation PRIVATE incremental_search_core)

add_executable(test_algorithms cpp/tests/test_algorithms.cpp)
target_link_libraries(test_algorithms PRIVATE incremental_search_core)

add_executable(test_simulation cpp/tests/test_simulation.cpp)
target_link_libraries(test_simulation PRIVATE incremental_search_core)

enable_testing()
add_test(NAME AlgorithmEquivalenceTests COMMAND test_algorithms)
add_test(NAME StochasticSimulationTests COMMAND test_simulation)
