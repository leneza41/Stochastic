from __future__ import annotations

import os
import statistics
import sys
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from incremental_search import (
    FailureMode,
    GraphIO,
    RunSummaryRecord,
    SampleSizeCalculator,
    SimulationConfig,
    StepExecutionRecord,
    StochasticSimulator,
)
from prepare_datasets import prepare_all_datasets


def print_summary_table(summaries: List[RunSummaryRecord]) -> None:
    print(
        f"{'Algorithm':<16} | {'Map':<26} | {'Type':<10} | {'Seed':<6} | "
        f"{'MeanTime(ms)':<12} | {'MeanNodes':<10} | {'PeakMem(MB)':<11} | Valid"
    )
    print("-" * 112)
    for s in summaries:
        status = "OK" if s.all_costs_validated else "FAIL"
        print(
            f"{s.algorithm:<16} | {s.map_name:<26} | {s.graph_type:<10} | {s.seed:<6d} | "
            f"{s.mean_replan_time_ms:12.4f} | {s.mean_replan_nodes:10.1f} | "
            f"{s.peak_memory_mb:11.5f} | {status}"
        )


def main() -> int:
    data_dir = "data"
    if not os.path.exists(os.path.join(data_dir, "osm", "osm_urban_small.graph")):
        print("Preparing datasets in data/{osm,movingai,synthetic}...")
        prepare_all_datasets(data_dir)

    all_steps: List[StepExecutionRecord] = []
    all_summaries: List[RunSummaryRecord] = []

    print("\n=== Step 2: Python Stochastic Edge-Failure Simulation Engine ===")

    # 1. OpenStreetMap simulations
    for osm_file, seed in [
        ("osm_urban_small.graph", 1001),
        ("osm_urban_medium.graph", 1002),
    ]:
        g_osm = GraphIO.load_graph(os.path.join(data_dir, "osm", osm_file))
        s, t = GraphIO.sample_valid_od_pair(g_osm, seed)
        cfg = SimulationConfig(
            map_name=osm_file.replace(".graph", ""),
            graph_type="osm",
            seed=seed,
            num_failures=10,
            failure_mode=FailureMode.ON_CURRENT_PATH,
        )
        out = StochasticSimulator.run_scenario(g_osm, s, t, cfg)
        if not out.validation_passed:
            print(f"Validation failed on {osm_file}!", file=sys.stderr)
            return 1
        all_steps.extend(out.step_records)
        all_summaries.extend(out.run_summaries)

    # 2. Moving AI simulations (.map + .scen)
    for map_file, seed in [
        ("arena_rooms.map", 2001),
        ("dungeon_game.map", 2002),
        ("labyrinth_maze.map", 2003),
    ]:
        mdata = GraphIO.load_movingai_map(os.path.join(data_dir, "movingai", map_file), 8)
        scens = GraphIO.load_movingai_scenarios(
            os.path.join(data_dir, "movingai", f"{map_file}.scen"), mdata
        )
        scen = scens[-1]  # Choose longest distance scenario bucket
        cfg = SimulationConfig(
            map_name=map_file,
            graph_type="movingai",
            seed=seed,
            num_failures=10,
            failure_mode=FailureMode.ON_CURRENT_PATH,
        )
        out = StochasticSimulator.run_scenario(
            mdata.graph, scen.start_node, scen.goal_node, cfg
        )
        if not out.validation_passed:
            print(f"Validation failed on {map_file}!", file=sys.stderr)
            return 1
        all_steps.extend(out.step_records)
        all_summaries.extend(out.run_summaries)

    # 3. Synthetic graph simulations
    for syn_file, mode, seed in [
        ("syn_grid_low_density.graph", FailureMode.ON_CURRENT_PATH, 3001),
        ("syn_grid_high_density.graph", FailureMode.MIXED_PROBABILISTIC, 3002),
        ("syn_random_geometric.graph", FailureMode.CONGESTION_PENALTY, 3003),
    ]:
        g_syn = GraphIO.load_graph(os.path.join(data_dir, "synthetic", syn_file))
        s, t = GraphIO.sample_valid_od_pair(g_syn, seed)
        cfg = SimulationConfig(
            map_name=syn_file.replace(".graph", ""),
            graph_type="synthetic",
            seed=seed,
            num_failures=10,
            failure_mode=mode,
        )
        out = StochasticSimulator.run_scenario(g_syn, s, t, cfg)
        if not out.validation_passed:
            print(f"Validation failed on {syn_file}!", file=sys.stderr)
            return 1
        all_steps.extend(out.step_records)
        all_summaries.extend(out.run_summaries)

    print_summary_table(all_summaries)

    steps_csv = "results/raw/python_simulation_steps.csv"
    summaries_csv = "results/raw/python_simulation_summaries.csv"
    StochasticSimulator.export_steps_csv(all_steps, steps_csv, append=False)
    StochasticSimulator.export_summaries_csv(all_summaries, summaries_csv, append=False)

    # Demonstrate sample size calculation from pilot metrics
    dstar_times = [
        s.mean_replan_time_ms for s in all_summaries if s.algorithm == "D* Lite"
    ]
    if len(dstar_times) >= 2:
        mean_t = statistics.mean(dstar_times)
        std_t = statistics.stdev(dstar_times)
        err_e = max(0.05 * mean_t, 1e-4)
        n_inf = SampleSizeCalculator.compute_infinite_population(std_t, err_e)
        n_fin = SampleSizeCalculator.compute_finite_population(1000, std_t, err_e)
        print(
            f"\n[Pilot Sample Size Estimation (Z=1.96)] "
            f"S={std_t:.4f} ms, E={err_e:.4f} ms -> "
            f"n_infinite={n_inf}, n_finite(N=1000)={n_fin}"
        )

    print(f"\nSaved step records to:  {steps_csv}")
    print(f"Saved run summaries to: {summaries_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
