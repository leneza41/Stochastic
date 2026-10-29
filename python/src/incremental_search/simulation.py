from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

from .d_star_lite import DStarLite
from .graph import Graph
from .incremental_a_star import IncrementalAStar
from .lpa_star import LPAStar
from .types import INF, INVALID_NODE, EdgeChange, SearchResult, approx_equal


class FailureMode(str, Enum):
    ON_CURRENT_PATH = "on_current_path"
    UNIFORM_RANDOM = "uniform_random"
    MIXED_PROBABILISTIC = "mixed_probabilistic"
    CONGESTION_PENALTY = "congestion_penalty"


@dataclass(slots=True)
class SimulationConfig:
    map_name: str = "unknown_map"
    graph_type: str = "synthetic"  # "osm", "movingai", "synthetic"
    seed: int = 42
    num_failures: int = 10
    failure_mode: FailureMode = FailureMode.ON_CURRENT_PATH
    path_failure_prob: float = 0.8
    congestion_factor: float = 5.0
    bidirectional_failure: bool = True
    validate_optimality: bool = True


@dataclass(slots=True)
class FailureEventRecord:
    failure_id: int
    u: int
    v: int
    bidirectional: bool
    old_cost: float
    new_cost: float


@dataclass(slots=True)
class StepExecutionRecord:
    algorithm: str
    map_name: str
    graph_type: str
    num_vertices: int
    num_edges: int
    seed: int
    start_node: int
    goal_node: int
    failure_id: int
    affected_edge: str
    path_found: bool
    path_cost: float
    compute_time_ms: float
    nodes_recalculated: int
    memory_mb: float
    process_rss_mb: float
    cost_validated: bool


@dataclass(slots=True)
class RunSummaryRecord:
    algorithm: str
    map_name: str
    graph_type: str
    num_vertices: int
    num_edges: int
    seed: int
    start_node: int
    goal_node: int
    total_failures: int
    successful_replans: int
    initial_time_ms: float
    initial_nodes: int
    mean_replan_time_ms: float
    mean_replan_nodes: float
    mean_memory_mb: float
    peak_memory_mb: float
    all_costs_validated: bool


@dataclass(slots=True)
class SimulationRunOutput:
    config: SimulationConfig
    start_node: int
    goal_node: int
    failure_events: List[FailureEventRecord] = field(default_factory=list)
    step_records: List[StepExecutionRecord] = field(default_factory=list)
    run_summaries: List[RunSummaryRecord] = field(default_factory=list)
    validation_passed: bool = True


class SampleSizeCalculator:
    """
    Implements Section 5 of "Muestra de la Investigación.pdf" (Z = 1.96 for 95% confidence).
    """

    Z_95: float = 1.96

    @staticmethod
    def compute_infinite_population(
        std_dev_s: float, acceptable_error_e: float, z_score: float = Z_95
    ) -> int:
        if acceptable_error_e <= 0.0 or std_dev_s <= 0.0:
            return 30
        num = (z_score**2) * (std_dev_s**2)
        den = acceptable_error_e**2
        return int(math.ceil(num / den))

    @staticmethod
    def compute_finite_population(
        population_n: int,
        std_dev_s: float,
        acceptable_error_e: float,
        z_score: float = Z_95,
    ) -> int:
        if population_n <= 0 or acceptable_error_e <= 0.0 or std_dev_s <= 0.0:
            return 30
        z2_s2 = (z_score**2) * (std_dev_s**2)
        e2 = acceptable_error_e**2
        num = float(population_n) * z2_s2
        den = e2 * float(population_n - 1) + z2_s2
        return int(math.ceil(num / den))


class StochasticSimulator:
    """
    Controlled Stochastic Edge-Failure Simulator.
    Executes D* Lite, LPA*, and Incremental A* on identical graph clones under the
    exact same pseudo-random sequence of edge failures and records all metrics.
    """

    @staticmethod
    def _collect_active_edges(graph: Graph) -> List[Tuple[int, int]]:
        active: List[Tuple[int, int]] = []
        for u in range(graph.num_vertices()):
            for e in graph.successors(u):
                if not e.failed and not math.isinf(e.cost):
                    active.append((u, e.to))
        return active

    @staticmethod
    def _select_stochastic_edge_to_fail(
        graph: Graph,
        current_path: List[int],
        config: SimulationConfig,
        rng: random.Random,
    ) -> Tuple[int, int]:
        pick_from_path = False
        if config.failure_mode in (
            FailureMode.ON_CURRENT_PATH,
            FailureMode.CONGESTION_PENALTY,
        ):
            pick_from_path = True
        elif config.failure_mode == FailureMode.MIXED_PROBABILISTIC:
            pick_from_path = rng.random() < config.path_failure_prob

        if pick_from_path and len(current_path) >= 2:
            path_edges: List[Tuple[int, int]] = []
            for i in range(len(current_path) - 1):
                u, v = current_path[i], current_path[i + 1]
                if not math.isinf(graph.get_edge_cost(u, v)):
                    path_edges.append((u, v))
            if path_edges:
                idx = rng.randint(0, len(path_edges) - 1)
                return path_edges[idx]

        active = StochasticSimulator._collect_active_edges(graph)
        if not active:
            return INVALID_NODE, INVALID_NODE
        idx = rng.randint(0, len(active) - 1)
        return active[idx]

    @staticmethod
    def _make_step_record(
        cfg: SimulationConfig,
        num_v: int,
        num_e: int,
        start: int,
        goal: int,
        failure_id: int,
        affected_edge: str,
        res: SearchResult,
        cost_valid: bool,
    ) -> StepExecutionRecord:
        return StepExecutionRecord(
            algorithm=res.algorithm_name,
            map_name=cfg.map_name,
            graph_type=cfg.graph_type,
            num_vertices=num_v,
            num_edges=num_e,
            seed=cfg.seed,
            start_node=start,
            goal_node=goal,
            failure_id=failure_id,
            affected_edge=affected_edge,
            path_found=res.path_found,
            path_cost=res.path_cost,
            compute_time_ms=res.compute_time_ms,
            nodes_recalculated=res.nodes_recalculated,
            memory_mb=res.memory_mb,
            process_rss_mb=res.process_rss_mb,
            cost_validated=cost_valid,
        )

    @staticmethod
    def run_scenario(
        base_graph: Graph,
        start: int,
        goal: int,
        config: SimulationConfig,
    ) -> SimulationRunOutput:
        output = SimulationRunOutput(config=config, start_node=start, goal_node=goal)

        g_dstar = base_graph.clone()
        g_lpa = base_graph.clone()
        g_inca = base_graph.clone()

        num_v = base_graph.num_vertices()
        num_e = base_graph.num_edges()

        dstar = DStarLite(g_dstar, start, goal)
        lpa = LPAStar(g_lpa, start, goal)
        inca = IncrementalAStar(g_inca, start, goal)

        dstar.initialize()
        lpa.initialize()
        inca.initialize()

        res_d = dstar.compute_initial_path()
        res_l = lpa.compute_initial_path()
        res_a = inca.compute_initial_path()

        init_valid = approx_equal(res_d.path_cost, res_l.path_cost) and approx_equal(
            res_d.path_cost, res_a.path_cost
        )
        if not init_valid:
            output.validation_passed = False

        for res in (res_d, res_l, res_a):
            output.step_records.append(
                StochasticSimulator._make_step_record(
                    config, num_v, num_e, start, goal, 0, "INITIAL", res, init_valid
                )
            )

        rng = random.Random(config.seed)

        for f_id in range(1, config.num_failures + 1):
            if (
                not res_d.path_found
                and config.failure_mode == FailureMode.ON_CURRENT_PATH
            ):
                break

            u, v = StochasticSimulator._select_stochastic_edge_to_fail(
                g_dstar, res_d.path, config, rng
            )
            if u == INVALID_NODE or v == INVALID_NODE:
                break

            old_cost = g_dstar.get_edge_cost(u, v)
            new_cost = (
                old_cost * config.congestion_factor
                if config.failure_mode == FailureMode.CONGESTION_PENALTY
                else INF
            )

            changes_d: List[EdgeChange] = [g_dstar.update_edge_cost(u, v, new_cost)]
            changes_l: List[EdgeChange] = [g_lpa.update_edge_cost(u, v, new_cost)]
            changes_a: List[EdgeChange] = [g_inca.update_edge_cost(u, v, new_cost)]

            bidir_applied = False
            if config.bidirectional_failure and g_dstar.has_edge(v, u):
                rev_old = g_dstar.get_edge_cost(v, u)
                rev_new = (
                    rev_old * config.congestion_factor
                    if config.failure_mode == FailureMode.CONGESTION_PENALTY
                    else INF
                )
                changes_d.append(g_dstar.update_edge_cost(v, u, rev_new))
                changes_l.append(g_lpa.update_edge_cost(v, u, rev_new))
                changes_a.append(g_inca.update_edge_cost(v, u, rev_new))
                bidir_applied = True

            output.failure_events.append(
                FailureEventRecord(
                    failure_id=f_id,
                    u=u,
                    v=v,
                    bidirectional=bidir_applied,
                    old_cost=old_cost,
                    new_cost=new_cost,
                )
            )

            res_d = dstar.replan(changes_d)
            res_l = lpa.replan(changes_l)
            res_a = inca.replan(changes_a)

            step_valid = (
                (res_d.path_found == res_l.path_found == res_a.path_found)
                and approx_equal(res_d.path_cost, res_l.path_cost)
                and approx_equal(res_d.path_cost, res_a.path_cost)
            )
            if not step_valid:
                output.validation_passed = False

            edge_label = f"{u}{'<->' if bidir_applied else '->'}{v}"
            for res in (res_d, res_l, res_a):
                output.step_records.append(
                    StochasticSimulator._make_step_record(
                        config,
                        num_v,
                        num_e,
                        start,
                        goal,
                        f_id,
                        edge_label,
                        res,
                        step_valid,
                    )
                )

        for alg in ("D* Lite", "LPA*", "Incremental A*"):
            alg_steps = [r for r in output.step_records if r.algorithm == alg]
            init_step = next((r for r in alg_steps if r.failure_id == 0), None)
            replan_steps = [r for r in alg_steps if r.failure_id > 0]

            replan_count = len(replan_steps)
            succ_count = sum(1 for r in replan_steps if r.path_found)
            peak_mem = max((r.memory_mb for r in alg_steps), default=0.0)
            all_valid = all(r.cost_validated for r in alg_steps)

            if replan_count > 0:
                mean_time = sum(r.compute_time_ms for r in replan_steps) / replan_count
                mean_nodes = (
                    sum(r.nodes_recalculated for r in replan_steps) / replan_count
                )
                mean_mem = sum(r.memory_mb for r in replan_steps) / replan_count
            else:
                mean_time = 0.0
                mean_nodes = 0.0
                mean_mem = peak_mem

            output.run_summaries.append(
                RunSummaryRecord(
                    algorithm=alg,
                    map_name=config.map_name,
                    graph_type=config.graph_type,
                    num_vertices=num_v,
                    num_edges=num_e,
                    seed=config.seed,
                    start_node=start,
                    goal_node=goal,
                    total_failures=replan_count,
                    successful_replans=succ_count,
                    initial_time_ms=init_step.compute_time_ms if init_step else 0.0,
                    initial_nodes=init_step.nodes_recalculated if init_step else 0,
                    mean_replan_time_ms=mean_time,
                    mean_replan_nodes=mean_nodes,
                    mean_memory_mb=mean_mem,
                    peak_memory_mb=peak_mem,
                    all_costs_validated=all_valid,
                )
            )

        return output

    @staticmethod
    def export_steps_csv(
        records: List[StepExecutionRecord], filepath: str, append: bool = False
    ) -> bool:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        write_header = not append or not os.path.exists(filepath) or os.path.getsize(filepath) == 0
        mode = "a" if append else "w"
        with open(filepath, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "algorithm",
                        "map_name",
                        "graph_type",
                        "num_vertices",
                        "num_edges",
                        "seed",
                        "start_node",
                        "goal_node",
                        "failure_id",
                        "affected_edge",
                        "path_found",
                        "path_cost",
                        "compute_time_ms",
                        "nodes_recalculated",
                        "memory_mb",
                        "process_rss_mb",
                        "cost_validated",
                    ]
                )
            for r in records:
                cost_str = "INF" if math.isinf(r.path_cost) else f"{r.path_cost:.6f}"
                writer.writerow(
                    [
                        r.algorithm,
                        r.map_name,
                        r.graph_type,
                        r.num_vertices,
                        r.num_edges,
                        r.seed,
                        r.start_node,
                        r.goal_node,
                        r.failure_id,
                        r.affected_edge,
                        1 if r.path_found else 0,
                        cost_str,
                        f"{r.compute_time_ms:.6f}",
                        r.nodes_recalculated,
                        f"{r.memory_mb:.6f}",
                        f"{r.process_rss_mb:.6f}",
                        1 if r.cost_validated else 0,
                    ]
                )
        return True

    @staticmethod
    def export_summaries_csv(
        summaries: List[RunSummaryRecord], filepath: str, append: bool = False
    ) -> bool:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        write_header = not append or not os.path.exists(filepath) or os.path.getsize(filepath) == 0
        mode = "a" if append else "w"
        with open(filepath, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "algorithm",
                        "map_name",
                        "graph_type",
                        "num_vertices",
                        "num_edges",
                        "seed",
                        "start_node",
                        "goal_node",
                        "total_failures",
                        "successful_replans",
                        "initial_time_ms",
                        "initial_nodes",
                        "mean_replan_time_ms",
                        "mean_replan_nodes",
                        "mean_memory_mb",
                        "peak_memory_mb",
                        "all_costs_validated",
                    ]
                )
            for s in summaries:
                writer.writerow(
                    [
                        s.algorithm,
                        s.map_name,
                        s.graph_type,
                        s.num_vertices,
                        s.num_edges,
                        s.seed,
                        s.start_node,
                        s.goal_node,
                        s.total_failures,
                        s.successful_replans,
                        f"{s.initial_time_ms:.6f}",
                        s.initial_nodes,
                        f"{s.mean_replan_time_ms:.6f}",
                        f"{s.mean_replan_nodes:.6f}",
                        f"{s.mean_memory_mb:.6f}",
                        f"{s.peak_memory_mb:.6f}",
                        1 if s.all_costs_validated else 0,
                    ]
                )
        return True
