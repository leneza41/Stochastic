from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from incremental_search import (
    FailureMode,
    GraphIO,
    ResearchReportBuilder,
    RunSummaryRecord,
    SimulationConfig,
    StepExecutionRecord,
    StochasticSimulator,
    SystemEnvironmentInfo,
)
from prepare_datasets import prepare_all_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Steps 3 & 4 Research Pipeline: Quantifies, evaluates, and compares "
            "D* Lite, LPA*, and Incremental A* under stochastic edge failures and "
            "exports all research tables, figures, and reports to the destination folder."
        )
    )
    parser.add_argument(
        "destination_path",
        nargs="?",
        default=None,
        help="Destination folder path where all research results, tables, figures, and reports will be generated.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        default="results/research_output",
        help="Alternative flag to specify the destination folder path (default: results/research_output).",
    )
    parser.add_argument(
        "--runs-per-map",
        "-n",
        type=int,
        default=8,
        help="Number of independent experimental runs (unidades muestrales) per map (use 30 for full pilot sample).",
    )
    parser.add_argument(
        "--failures-per-run",
        "-k",
        type=int,
        default=8,
        help="Number of sequential stochastic edge failures injected per run (default: 8).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing input datasets (default: data).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dest_dir = args.destination_path if args.destination_path else args.output_dir
    abs_dest = os.path.abspath(dest_dir)
    data_dir = os.path.abspath(args.data_dir)

    print("=" * 96)
    print("  INCREMENTAL SEARCH UNDER STOCHASTIC EDGE FAILURES - RESEARCH PIPELINE (STEPS 3 & 4)")
    print("=" * 96)
    env_info = SystemEnvironmentInfo.capture()
    print(
        f"  OS & Hardware : {env_info.os_name} {env_info.os_release} (Build {env_info.os_version}) | "
        f"CPU: {env_info.logical_cores} logical cores | RAM: {env_info.total_ram_gb:.2f} GB"
    )
    print(f"  Python Runtime: {env_info.python_version}")
    print(f"  Destination   : {abs_dest}")
    print(
        f"  Configuration : {args.runs_per_map} runs/map (+1 discarded warmup/map) x "
        f"{args.failures_per_run} stochastic failures/run"
    )
    print("-" * 96)

    if not os.path.exists(os.path.join(data_dir, "osm", "osm_urban_small.graph")):
        print("[1/4] Preparing canonical datasets (OpenStreetMap, Moving AI, Synthetic)...")
        prepare_all_datasets(data_dir)
    else:
        print("[1/4] Verified canonical datasets in:", data_dir)

    # Map catalog across the 3 sampling frame sources and 3 scale tiers
    map_catalog = [
        ("osm", "osm_urban_small.graph", "osm_urban_small", "Pequeña", 10000),
        ("osm", "osm_urban_medium.graph", "osm_urban_medium", "Mediana", 20000),
        ("osm", "osm_urban_large.graph", "osm_urban_large", "Grande", 30000),
        ("movingai", "arena_rooms.map", "arena_rooms.map", "Grande", 40000),
        ("movingai", "dungeon_game.map", "dungeon_game.map", "Grande", 50000),
        ("movingai", "labyrinth_maze.map", "labyrinth_maze.map", "Grande", 60000),
        ("synthetic", "syn_grid_low_density.graph", "syn_grid_low_density", "Mediana", 70000),
        ("synthetic", "syn_grid_high_density.graph", "syn_grid_high_density", "Mediana", 80000),
        ("synthetic", "syn_random_geometric.graph", "syn_random_geometric", "Mediana", 90000),
    ]

    failure_modes_cycle = [
        FailureMode.ON_CURRENT_PATH,
        FailureMode.ON_CURRENT_PATH,
        FailureMode.MIXED_PROBABILISTIC,
        FailureMode.CONGESTION_PENALTY,
        FailureMode.UNIFORM_RANDOM,
    ]

    all_steps: List[StepExecutionRecord] = []
    all_summaries: List[RunSummaryRecord] = []
    failure_mode_by_run_key: Dict[Tuple[str, int], str] = {}
    scale_by_map: Dict[str, str] = {}

    print("[2/4] Executing automated benchmark harness with warmup runs...")
    t_pipeline_start = time.perf_counter()

    for graph_type, filename, map_label, scale_label, base_seed in map_catalog:
        scale_by_map[map_label] = scale_label

        if graph_type == "movingai":
            map_path = os.path.join(data_dir, "movingai", filename)
            scen_path = os.path.join(data_dir, "movingai", f"{filename}.scen")
            mdata = GraphIO.load_movingai_map(map_path, connectivity=8)
            base_graph = mdata.graph
            scenarios = GraphIO.load_movingai_scenarios(scen_path, mdata)
        else:
            g_path = os.path.join(data_dir, graph_type, filename)
            base_graph = GraphIO.load_graph(g_path)
            scenarios = []

        # Discarded Warmup Run (as required by "Plan de recolección de datos.pdf")
        if scenarios:
            w_s, w_t = scenarios[0].start_node, scenarios[0].goal_node
        else:
            w_s, w_t = GraphIO.sample_valid_od_pair(base_graph, base_seed)

        warmup_cfg = SimulationConfig(
            map_name=map_label,
            graph_type=graph_type,
            seed=base_seed,
            num_failures=3,
            failure_mode=FailureMode.ON_CURRENT_PATH,
        )
        _ = StochasticSimulator.run_scenario(base_graph, w_s, w_t, warmup_cfg)

        # Measured Experimental Runs
        for run_idx in range(1, args.runs_per_map + 1):
            seed = base_seed + run_idx
            mode = failure_modes_cycle[(run_idx - 1) % len(failure_modes_cycle)]

            if scenarios:
                scen = scenarios[(run_idx - 1) % len(scenarios)]
                s_node, t_node = scen.start_node, scen.goal_node
            else:
                s_node, t_node = GraphIO.sample_valid_od_pair(base_graph, seed)

            cfg = SimulationConfig(
                map_name=map_label,
                graph_type=graph_type,
                seed=seed,
                num_failures=args.failures_per_run,
                failure_mode=mode,
                path_failure_prob=0.80,
                congestion_factor=5.0,
            )

            out = StochasticSimulator.run_scenario(base_graph, s_node, t_node, cfg)
            if not out.validation_passed:
                print(
                    f"ERROR: Optimality validation failed on {map_label} (seed={seed})!",
                    file=sys.stderr,
                )
                return 1

            failure_mode_by_run_key[(map_label, seed)] = mode.value
            all_steps.extend(out.step_records)
            all_summaries.extend(out.run_summaries)

        print(
            f"      Completed {map_label:<24} ({graph_type:<9} | Escala {scale_label:<7}) "
            f"-> {args.runs_per_map} runs validated [OK]"
        )

    elapsed_s = time.perf_counter() - t_pipeline_start
    print(f"[3/4] Benchmark execution finished in {elapsed_s:.2f}s. Computing statistical analyses & figures...")

    artifacts = ResearchReportBuilder.build_complete_output_folder(
        destination_dir=abs_dest,
        step_records=all_steps,
        run_summaries=all_summaries,
        env_info=env_info,
        failure_mode_by_run_key=failure_mode_by_run_key,
        scale_by_map=scale_by_map,
    )

    print("[4/4] Research output folder generated successfully!\n")
    print("=" * 96)
    print(f"  GENERATED RESEARCH ARTIFACTS IN: {abs_dest}")
    print("=" * 96)
    print(f"  - Raw Step Measurements CSV : {artifacts['steps_csv']}")
    print(f"  - Raw Run Summaries CSV     : {artifacts['runs_csv']}")
    print(f"  - System Environment JSON   : {artifacts['env_json']}")
    print(f"  - Table 1 (Descriptive)     : {artifacts['table1_csv']}")
    print(f"  - Table 2 (Failure Modes)   : {artifacts['table2_csv']}")
    print(f"  - Table 3 (Scale Analysis)  : {artifacts['table3_csv']}")
    print(f"  - Table 4 (Sample Size)     : {artifacts['table4_csv']}")
    print(f"  - Table 5 (Friedman/Wilcox) : {artifacts['table5_csv']}")
    print(f"  - Table 6 (Trade-off Index) : {artifacts['table6_csv']}")
    print(f"  - Research Report (Markdown): {artifacts['markdown_report']}")
    print(f"  - Research Dashboard (HTML) : {artifacts['html_dashboard']}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
