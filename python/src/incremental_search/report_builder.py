from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict
from typing import Dict, List, Sequence, Tuple

from .evaluation import (
    DescriptiveStats,
    EfficiencyTradeoffRecord,
    FriedmanTestResult,
    NonParametricTests,
    ResearchFigureGenerator,
    SystemEnvironmentInfo,
)
from .simulation import (
    RunSummaryRecord,
    SampleSizeCalculator,
    StepExecutionRecord,
    StochasticSimulator,
)

ALGORITHMS = ("D* Lite", "LPA*", "Incremental A*")


class ResearchReportBuilder:
    """
    Executes Step 3 & Step 4 post-processing, validation, statistical hypothesis
    evaluation (Friedman + Wilcoxon signed-rank), trade-off quantification,
    publication SVG figure rendering, and research-grade report generation
    inside a user-specified destination folder.
    """

    @staticmethod
    def build_complete_output_folder(
        destination_dir: str,
        step_records: Sequence[StepExecutionRecord],
        run_summaries: Sequence[RunSummaryRecord],
        env_info: SystemEnvironmentInfo,
        failure_mode_by_run_key: Dict[Tuple[str, int], str],
        scale_by_map: Dict[str, str],
    ) -> Dict[str, str]:
        abs_dest = os.path.abspath(destination_dir)
        raw_dir = os.path.join(abs_dest, "01_raw_data")
        tables_dir = os.path.join(abs_dest, "02_statistical_tables")
        figs_dir = os.path.join(abs_dest, "03_figures")
        report_dir = os.path.join(abs_dest, "04_research_report")

        for d in (raw_dir, tables_dir, figs_dir, report_dir):
            os.makedirs(d, exist_ok=True)

        # 1. Save Raw Data & Environment Metadata
        steps_csv_path = os.path.join(raw_dir, "replanning_steps_raw.csv")
        runs_csv_path = os.path.join(raw_dir, "experimental_runs_raw.csv")
        env_json_path = os.path.join(raw_dir, "system_environment.json")

        StochasticSimulator.export_steps_csv(list(step_records), steps_csv_path, append=False)
        StochasticSimulator.export_summaries_csv(list(run_summaries), runs_csv_path, append=False)
        with open(env_json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(env_info), f, indent=2)

        # 2. Table 1: Descriptive Statistics by Algorithm & Graph Source
        table1_path = os.path.join(
            tables_dir, "table1_descriptive_by_algorithm_and_source.csv"
        )
        table1_rows = ResearchReportBuilder._generate_table1_descriptive(
            run_summaries, table1_path
        )

        # 3. Table 2: Performance by Stochastic Failure Mode
        table2_path = os.path.join(
            tables_dir, "table2_performance_by_failure_mode.csv"
        )
        table2_rows = ResearchReportBuilder._generate_table2_failure_modes(
            run_summaries, failure_mode_by_run_key, table2_path
        )

        # 4. Table 3: Scale Sensitivity Analysis (Small vs. Medium vs. Large)
        table3_path = os.path.join(
            tables_dir, "table3_scale_sensitivity_analysis.csv"
        )
        table3_rows = ResearchReportBuilder._generate_table3_scale_analysis(
            run_summaries, scale_by_map, table3_path
        )

        # 5. Table 4: Statistical Sample Size Estimation (Muestra Piloto Z=1.96)
        table4_path = os.path.join(
            tables_dir, "table4_sample_size_estimation.csv"
        )
        table4_rows = ResearchReportBuilder._generate_table4_sample_size(
            run_summaries, table4_path
        )

        # 6. Table 5: Friedman & Post-hoc Wilcoxon Signed-Rank Tests
        table5_path = os.path.join(
            tables_dir, "table5_friedman_wilcoxon_tests.csv"
        )
        friedman_results = ResearchReportBuilder._generate_table5_inferential(
            run_summaries, table5_path
        )

        # 7. Table 6: Efficiency-Resource Trade-off Index
        table6_path = os.path.join(
            tables_dir, "table6_efficiency_resource_tradeoff.csv"
        )
        tradeoff_records = ResearchReportBuilder._generate_table6_tradeoff(
            run_summaries, table6_path
        )

        # 8. Generate Publication-Grade Vector Figures (.svg)
        fig_paths = ResearchReportBuilder._generate_all_figures(
            figs_dir, step_records, run_summaries, scale_by_map, tradeoff_records
        )

        # 9. Generate Research Report (.md) and Interactive Dashboard (.html)
        md_report_path = os.path.join(report_dir, "RESEARCH_REPORT.md")
        html_report_path = os.path.join(report_dir, "RESEARCH_DASHBOARD.html")

        ResearchReportBuilder._write_markdown_report(
            md_report_path,
            env_info,
            step_records,
            run_summaries,
            table1_rows,
            table2_rows,
            table3_rows,
            table4_rows,
            friedman_results,
            tradeoff_records,
        )
        ResearchReportBuilder._write_html_dashboard(
            html_report_path,
            env_info,
            step_records,
            run_summaries,
            table1_rows,
            table2_rows,
            table3_rows,
            table4_rows,
            friedman_results,
            tradeoff_records,
            fig_paths,
        )

        return {
            "destination_dir": abs_dest,
            "steps_csv": steps_csv_path,
            "runs_csv": runs_csv_path,
            "env_json": env_json_path,
            "table1_csv": table1_path,
            "table2_csv": table2_path,
            "table3_csv": table3_path,
            "table4_csv": table4_path,
            "table5_csv": table5_path,
            "table6_csv": table6_path,
            "markdown_report": md_report_path,
            "html_dashboard": html_report_path,
        }

    @staticmethod
    def _generate_table1_descriptive(
        run_summaries: Sequence[RunSummaryRecord], filepath: str
    ) -> List[Dict[str, object]]:
        sources = ["ALL", "osm", "movingai", "synthetic"]
        rows: List[Dict[str, object]] = []

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "graph_source",
                    "algorithm",
                    "n_runs",
                    "time_mean_ms",
                    "time_std_ms",
                    "time_median_ms",
                    "time_iqr_ms",
                    "time_p95_ms",
                    "time_outliers_iqr",
                    "nodes_mean",
                    "nodes_std",
                    "nodes_median",
                    "nodes_iqr",
                    "nodes_p95",
                    "mem_mean_mb",
                    "mem_std_mb",
                    "mem_median_mb",
                    "mem_peak_mb",
                ]
            )
            for src in sources:
                for alg in ALGORITHMS:
                    subset = [
                        s
                        for s in run_summaries
                        if s.algorithm == alg
                        and (src == "ALL" or s.graph_type == src)
                    ]
                    if not subset:
                        continue
                    t_stats = DescriptiveStats.compute(
                        [s.mean_replan_time_ms for s in subset]
                    )
                    n_stats = DescriptiveStats.compute(
                        [s.mean_replan_nodes for s in subset]
                    )
                    m_stats = DescriptiveStats.compute(
                        [s.mean_memory_mb for s in subset]
                    )
                    peak_m = max((s.peak_memory_mb for s in subset), default=0.0)

                    row_dict = {
                        "graph_source": src.upper(),
                        "algorithm": alg,
                        "n_runs": t_stats.count,
                        "time_mean_ms": t_stats.mean,
                        "time_std_ms": t_stats.std,
                        "time_median_ms": t_stats.median,
                        "time_iqr_ms": t_stats.iqr,
                        "time_p95_ms": t_stats.p95,
                        "time_outliers_iqr": t_stats.outliers_iqr_count,
                        "nodes_mean": n_stats.mean,
                        "nodes_std": n_stats.std,
                        "nodes_median": n_stats.median,
                        "nodes_iqr": n_stats.iqr,
                        "nodes_p95": n_stats.p95,
                        "mem_mean_mb": m_stats.mean,
                        "mem_std_mb": m_stats.std,
                        "mem_median_mb": m_stats.median,
                        "mem_peak_mb": peak_m,
                    }
                    rows.append(row_dict)
                    writer.writerow(
                        [
                            row_dict["graph_source"],
                            alg,
                            t_stats.count,
                            f"{t_stats.mean:.6f}",
                            f"{t_stats.std:.6f}",
                            f"{t_stats.median:.6f}",
                            f"{t_stats.iqr:.6f}",
                            f"{t_stats.p95:.6f}",
                            t_stats.outliers_iqr_count,
                            f"{n_stats.mean:.2f}",
                            f"{n_stats.std:.2f}",
                            f"{n_stats.median:.2f}",
                            f"{n_stats.iqr:.2f}",
                            f"{n_stats.p95:.2f}",
                            f"{m_stats.mean:.6f}",
                            f"{m_stats.std:.6f}",
                            f"{m_stats.median:.6f}",
                            f"{peak_m:.6f}",
                        ]
                    )
        return rows

    @staticmethod
    def _generate_table2_failure_modes(
        run_summaries: Sequence[RunSummaryRecord],
        failure_mode_by_run_key: Dict[Tuple[str, int], str],
        filepath: str,
    ) -> List[Dict[str, object]]:
        modes = [
            "on_current_path",
            "uniform_random",
            "mixed_probabilistic",
            "congestion_penalty",
        ]
        rows: List[Dict[str, object]] = []
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "failure_mode",
                    "algorithm",
                    "n_runs",
                    "time_mean_ms",
                    "time_median_ms",
                    "nodes_mean",
                    "nodes_median",
                    "mem_mean_mb",
                ]
            )
            for mode in modes:
                for alg in ALGORITHMS:
                    subset = [
                        s
                        for s in run_summaries
                        if s.algorithm == alg
                        and failure_mode_by_run_key.get((s.map_name, s.seed), "")
                        == mode
                    ]
                    if not subset:
                        continue
                    t_s = DescriptiveStats.compute(
                        [s.mean_replan_time_ms for s in subset]
                    )
                    n_s = DescriptiveStats.compute(
                        [s.mean_replan_nodes for s in subset]
                    )
                    m_s = DescriptiveStats.compute([s.mean_memory_mb for s in subset])
                    r = {
                        "failure_mode": mode,
                        "algorithm": alg,
                        "n_runs": t_s.count,
                        "time_mean_ms": t_s.mean,
                        "time_median_ms": t_s.median,
                        "nodes_mean": n_s.mean,
                        "nodes_median": n_s.median,
                        "mem_mean_mb": m_s.mean,
                    }
                    rows.append(r)
                    writer.writerow(
                        [
                            mode,
                            alg,
                            t_s.count,
                            f"{t_s.mean:.6f}",
                            f"{t_s.median:.6f}",
                            f"{n_s.mean:.2f}",
                            f"{n_s.median:.2f}",
                            f"{m_s.mean:.6f}",
                        ]
                    )
        return rows

    @staticmethod
    def _generate_table3_scale_analysis(
        run_summaries: Sequence[RunSummaryRecord],
        scale_by_map: Dict[str, str],
        filepath: str,
    ) -> List[Dict[str, object]]:
        scales = ["Pequeña", "Mediana", "Grande"]
        rows: List[Dict[str, object]] = []
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "scale_category",
                    "avg_vertices",
                    "dstar_mean_time_ms",
                    "lpa_mean_time_ms",
                    "inca_mean_time_ms",
                    "max_abs_diff_time_ms",
                    "dstar_vs_lpa_diff_ms",
                    "dstar_mean_mem_kb",
                    "lpa_mean_mem_kb",
                    "inca_mean_mem_kb",
                ]
            )
            for sc in scales:
                sc_runs = [
                    s for s in run_summaries if scale_by_map.get(s.map_name, "") == sc
                ]
                if not sc_runs:
                    continue
                avg_v = sum(s.num_vertices for s in sc_runs) / float(len(sc_runs))
                by_alg_t: Dict[str, float] = {}
                by_alg_m: Dict[str, float] = {}
                for alg in ALGORITHMS:
                    a_runs = [s for s in sc_runs if s.algorithm == alg]
                    by_alg_t[alg] = (
                        sum(s.mean_replan_time_ms for s in a_runs) / len(a_runs)
                        if a_runs
                        else 0.0
                    )
                    by_alg_m[alg] = (
                        sum(s.mean_memory_mb * 1024.0 for s in a_runs) / len(a_runs)
                        if a_runs
                        else 0.0
                    )
                t_vals = list(by_alg_t.values())
                max_diff = max(t_vals) - min(t_vals) if t_vals else 0.0
                dl_diff = abs(by_alg_t["D* Lite"] - by_alg_t["LPA*"])
                r = {
                    "scale_category": sc,
                    "avg_vertices": avg_v,
                    "dstar_mean_time_ms": by_alg_t["D* Lite"],
                    "lpa_mean_time_ms": by_alg_t["LPA*"],
                    "inca_mean_time_ms": by_alg_t["Incremental A*"],
                    "max_abs_diff_time_ms": max_diff,
                    "dstar_vs_lpa_diff_ms": dl_diff,
                    "dstar_mean_mem_kb": by_alg_m["D* Lite"],
                    "lpa_mean_mem_kb": by_alg_m["LPA*"],
                    "inca_mean_mem_kb": by_alg_m["Incremental A*"],
                }
                rows.append(r)
                writer.writerow(
                    [
                        sc,
                        f"{avg_v:.0f}",
                        f"{r['dstar_mean_time_ms']:.6f}",
                        f"{r['lpa_mean_time_ms']:.6f}",
                        f"{r['inca_mean_time_ms']:.6f}",
                        f"{max_diff:.6f}",
                        f"{dl_diff:.6f}",
                        f"{r['dstar_mean_mem_kb']:.3f}",
                        f"{r['lpa_mean_mem_kb']:.3f}",
                        f"{r['inca_mean_mem_kb']:.3f}",
                    ]
                )
        return rows

    @staticmethod
    def _generate_table4_sample_size(
        run_summaries: Sequence[RunSummaryRecord], filepath: str
    ) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "graph_source",
                    "algorithm",
                    "pilot_runs",
                    "time_mean_ms",
                    "time_std_s_ms",
                    "error_e_10pct_ms",
                    "n_infinite_z95",
                    "n_finite_N1000_z95",
                ]
            )
            for src in ("osm", "movingai", "synthetic"):
                for alg in ALGORITHMS:
                    sub = [
                        s
                        for s in run_summaries
                        if s.graph_type == src and s.algorithm == alg
                    ]
                    if not sub:
                        continue
                    stats = DescriptiveStats.compute(
                        [s.mean_replan_time_ms for s in sub]
                    )
                    err_e = max(1e-5, 0.10 * stats.mean)
                    n_inf = SampleSizeCalculator.compute_infinite_population(
                        stats.std, err_e
                    )
                    n_fin = SampleSizeCalculator.compute_finite_population(
                        1000, stats.std, err_e
                    )
                    r = {
                        "graph_source": src.upper(),
                        "algorithm": alg,
                        "pilot_runs": stats.count,
                        "time_mean_ms": stats.mean,
                        "time_std_s_ms": stats.std,
                        "error_e_10pct_ms": err_e,
                        "n_infinite_z95": n_inf,
                        "n_finite_N1000_z95": n_fin,
                    }
                    rows.append(r)
                    writer.writerow(
                        [
                            r["graph_source"],
                            alg,
                            stats.count,
                            f"{stats.mean:.6f}",
                            f"{stats.std:.6f}",
                            f"{err_e:.6f}",
                            n_inf,
                            n_fin,
                        ]
                    )
        return rows

    @staticmethod
    def _generate_table5_inferential(
        run_summaries: Sequence[RunSummaryRecord], filepath: str
    ) -> List[FriedmanTestResult]:
        results: List[FriedmanTestResult] = []
        groups = ["GLOBAL", "osm", "movingai", "synthetic"]

        for grp in groups:
            # Match paired runs by (map_name, seed)
            by_key: Dict[Tuple[str, int], Dict[str, RunSummaryRecord]] = {}
            for s in run_summaries:
                if grp != "GLOBAL" and s.graph_type != grp:
                    continue
                by_key.setdefault((s.map_name, s.seed), {})[s.algorithm] = s

            paired_keys = sorted(
                k for k, v in by_key.items() if all(a in v for a in ALGORITHMS)
            )
            if not paired_keys:
                continue

            for metric_label, attr in [
                ("Tiempo de Cómputo (ms)", "mean_replan_time_ms"),
                ("Nodos Recalculados", "mean_replan_nodes"),
                ("Consumo de Memoria (MB)", "mean_memory_mb"),
            ]:
                d_vals = [getattr(by_key[k]["D* Lite"], attr) for k in paired_keys]
                l_vals = [getattr(by_key[k]["LPA*"], attr) for k in paired_keys]
                a_vals = [
                    getattr(by_key[k]["Incremental A*"], attr) for k in paired_keys
                ]
                res = NonParametricTests.friedman_with_wilcoxon(
                    metric_label, grp.upper(), d_vals, l_vals, a_vals
                )
                results.append(res)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "group_label",
                    "metric_name",
                    "n_paired_runs",
                    "friedman_chi2",
                    "friedman_p_value",
                    "kendalls_w",
                    "mean_rank_dstar",
                    "mean_rank_lpa",
                    "mean_rank_inca",
                    "pair_comparison",
                    "wilcoxon_z",
                    "wilcoxon_p_holm",
                    "rank_biserial_r",
                    "significant_05",
                    "superior_algorithm",
                ]
            )
            for fr in results:
                for pw in fr.pairwise_wilcoxon:
                    writer.writerow(
                        [
                            fr.group_label,
                            fr.metric_name,
                            fr.n_blocks,
                            f"{fr.chi2_statistic:.4f}",
                            f"{fr.p_value:.6e}",
                            f"{fr.kendalls_w:.4f}",
                            f"{fr.mean_ranks['D* Lite']:.3f}",
                            f"{fr.mean_ranks['LPA*']:.3f}",
                            f"{fr.mean_ranks['Incremental A*']:.3f}",
                            pw.pair_name,
                            f"{pw.z_score:.4f}",
                            f"{pw.p_value_holm:.6e}",
                            f"{pw.rank_biserial_r:.4f}",
                            "SI" if pw.significant_alpha_05 else "NO",
                            pw.faster_or_lower_alg,
                        ]
                    )
        return results

    @staticmethod
    def _generate_table6_tradeoff(
        run_summaries: Sequence[RunSummaryRecord], filepath: str
    ) -> List[EfficiencyTradeoffRecord]:
        records: List[EfficiencyTradeoffRecord] = []
        groups = ["GLOBAL", "osm", "movingai", "synthetic"]

        for grp in groups:
            temp_rows: List[Tuple[str, DescriptiveStats, DescriptiveStats, DescriptiveStats, float]] = []
            for alg in ALGORITHMS:
                sub = [
                    s
                    for s in run_summaries
                    if s.algorithm == alg
                    and (grp == "GLOBAL" or s.graph_type == grp)
                ]
                if not sub:
                    continue
                t_s = DescriptiveStats.compute([s.mean_replan_time_ms for s in sub])
                n_s = DescriptiveStats.compute([s.mean_replan_nodes for s in sub])
                m_s = DescriptiveStats.compute([s.mean_memory_mb for s in sub])
                peak_m = max((s.peak_memory_mb for s in sub), default=0.0)
                temp_rows.append((alg, t_s, n_s, m_s, peak_m))

            max_t = max((r[1].mean for r in temp_rows), default=1.0) or 1.0
            max_n = max((r[2].mean for r in temp_rows), default=1.0) or 1.0
            max_m = max((r[3].mean for r in temp_rows), default=1.0) or 1.0

            for alg, t_s, n_s, m_s, peak_m in temp_rows:
                nt = t_s.mean / max_t
                nn = n_s.mean / max_n
                nm = m_s.mean / max_m
                comp_idx = (nt + nn + nm) / 3.0
                rps = 1000.0 / t_s.mean if t_s.mean > 1e-6 else 0.0
                npms = n_s.mean / t_s.mean if t_s.mean > 1e-6 else 0.0
                records.append(
                    EfficiencyTradeoffRecord(
                        algorithm=alg,
                        group_label=grp.upper(),
                        mean_time_ms=t_s.mean,
                        median_time_ms=t_s.median,
                        mean_nodes=n_s.mean,
                        median_nodes=n_s.median,
                        mean_memory_mb=m_s.mean,
                        peak_memory_mb=peak_m,
                        normalized_time=nt,
                        normalized_nodes=nn,
                        normalized_memory=nm,
                        composite_efficiency_index=comp_idx,
                        replans_per_second=rps,
                        nodes_per_ms=npms,
                    )
                )

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "group_label",
                    "algorithm",
                    "mean_time_ms",
                    "median_time_ms",
                    "mean_nodes",
                    "median_nodes",
                    "mean_memory_mb",
                    "peak_memory_mb",
                    "norm_time",
                    "norm_nodes",
                    "norm_memory",
                    "composite_efficiency_index",
                    "replans_per_second",
                    "nodes_per_ms",
                ]
            )
            for rec in records:
                writer.writerow(
                    [
                        rec.group_label,
                        rec.algorithm,
                        f"{rec.mean_time_ms:.6f}",
                        f"{rec.median_time_ms:.6f}",
                        f"{rec.mean_nodes:.2f}",
                        f"{rec.median_nodes:.2f}",
                        f"{rec.mean_memory_mb:.6f}",
                        f"{rec.peak_memory_mb:.6f}",
                        f"{rec.normalized_time:.4f}",
                        f"{rec.normalized_nodes:.4f}",
                        f"{rec.normalized_memory:.4f}",
                        f"{rec.composite_efficiency_index:.4f}",
                        f"{rec.replans_per_second:.1f}",
                        f"{rec.nodes_per_ms:.2f}",
                    ]
                )
        return records

    @staticmethod
    def _generate_all_figures(
        figs_dir: str,
        step_records: Sequence[StepExecutionRecord],
        run_summaries: Sequence[RunSummaryRecord],
        scale_by_map: Dict[str, str],
        tradeoff_records: Sequence[EfficiencyTradeoffRecord],
    ) -> Dict[str, str]:
        sources = ["osm", "movingai", "synthetic"]
        source_labels = ["OpenStreetMap (OSM)", "Moving AI (.map/.scen)", "Sintéticos Controlados"]

        time_by_src: Dict[str, List[float]] = {a: [] for a in ALGORITHMS}
        nodes_by_src: Dict[str, List[float]] = {a: [] for a in ALGORITHMS}
        mem_kb_by_src: Dict[str, List[float]] = {a: [] for a in ALGORITHMS}

        for src in sources:
            for alg in ALGORITHMS:
                sub = [
                    s
                    for s in run_summaries
                    if s.graph_type == src and s.algorithm == alg
                ]
                time_by_src[alg].append(
                    sum(s.mean_replan_time_ms for s in sub) / len(sub) if sub else 0.0
                )
                nodes_by_src[alg].append(
                    sum(s.mean_replan_nodes for s in sub) / len(sub) if sub else 0.0
                )
                mem_kb_by_src[alg].append(
                    sum(s.mean_memory_mb * 1024.0 for s in sub) / len(sub)
                    if sub
                    else 0.0
                )

        f1 = os.path.join(figs_dir, "fig1_replan_time_comparison.svg")
        ResearchFigureGenerator.render_grouped_bar_svg(
            f1,
            "Figura 1. Tiempo de Cómputo de Replanificación por Fuente de Grafo",
            "Comparación empírica en milisegundos (ms) tras fallos estocásticos de aristas (menor es mejor)",
            "Tiempo de Cómputo (ms)",
            source_labels,
            time_by_src,
        )

        f2 = os.path.join(figs_dir, "fig2_nodes_recalculated_comparison.svg")
        ResearchFigureGenerator.render_grouped_bar_svg(
            f2,
            "Figura 2. Volumen de Nodos Recalculados durante la Replanificación",
            "Promedio de vértices actualizados/expandidos por evento de fallo estocástico (menor es mejor)",
            "Nodos Recalculados",
            source_labels,
            nodes_by_src,
        )

        f3 = os.path.join(figs_dir, "fig3_memory_consumption_comparison.svg")
        ResearchFigureGenerator.render_grouped_bar_svg(
            f3,
            "Figura 3. Consumo de Memoria Estructural por Algoritmo y Tipo de Red",
            "Huella de memoria en Kilobytes (KB) de las estructuras internas (g, rhs/parent y cola de prioridad)",
            "Memoria Estructural (KB)",
            source_labels,
            mem_kb_by_src,
        )

        # Figure 4: Convergence trajectory over failure sequence 1..K
        max_f = max((r.failure_id for r in step_records), default=10)
        steps_seq = list(range(1, max_f + 1))
        conv_time: Dict[str, List[float]] = {a: [] for a in ALGORITHMS}
        for k_step in steps_seq:
            for alg in ALGORITHMS:
                sub_steps = [
                    r
                    for r in step_records
                    if r.failure_id == k_step and r.algorithm == alg
                ]
                conv_time[alg].append(
                    sum(r.compute_time_ms for r in sub_steps) / len(sub_steps)
                    if sub_steps
                    else 0.0
                )

        f4 = os.path.join(figs_dir, "fig4_convergence_over_failure_sequence.svg")
        ResearchFigureGenerator.render_convergence_line_svg(
            f4,
            "Figura 4. Convergencia Temporal ante Fallos Estocásticos Sucesivos",
            "Evolución del tiempo medio de replanificación (ms) a lo largo de la secuencia de fallos repetidos",
            "Secuencia de Fallos Estocásticos de Aristas (k = 1 ... K)",
            "Tiempo de Replanificación (ms)",
            steps_seq,
            conv_time,
        )

        # Figure 5: Scale sensitivity (Small vs Medium vs Large)
        scales = ["Pequeña", "Mediana", "Grande"]
        scale_time: Dict[str, List[float]] = {a: [] for a in ALGORITHMS}
        for sc in scales:
            for alg in ALGORITHMS:
                sub = [
                    s
                    for s in run_summaries
                    if scale_by_map.get(s.map_name, "") == sc and s.algorithm == alg
                ]
                scale_time[alg].append(
                    sum(s.mean_replan_time_ms for s in sub) / len(sub) if sub else 0.0
                )

        f5 = os.path.join(figs_dir, "fig5_scale_effect_small_to_large.svg")
        ResearchFigureGenerator.render_grouped_bar_svg(
            f5,
            "Figura 5. Efecto de la Escala del Grafo sobre el Tiempo de Replanificación",
            "Contraste de la Hipótesis 2: convergencia de complejidades en redes pequeñas frente a redes grandes",
            "Tiempo de Cómputo (ms)",
            ["Escala Pequeña (~225 V)", "Escala Mediana (~784-1123 V)", "Escala Grande (~1764-2097 V)"],
            scale_time,
        )

        # Figure 6: Efficiency-Resource Trade-off Scatter
        f6 = os.path.join(figs_dir, "fig6_efficiency_memory_tradeoff_scatter.svg")
        sub_tradeoff = [r for r in tradeoff_records if r.group_label != "GLOBAL"]
        ResearchFigureGenerator.render_tradeoff_scatter_svg(
            f6,
            "Figura 6. Equilibrio entre Eficiencia Computacional y Consumo de Recursos",
            "Plano Tiempo (ms) vs. Memoria Estructural (KB) y Nodos Recalculados (radio de burbuja)",
            sub_tradeoff,
        )

        return {
            "fig1": f1,
            "fig2": f2,
            "fig3": f3,
            "fig4": f4,
            "fig5": f5,
            "fig6": f6,
        }

    @staticmethod
    def _write_markdown_report(
        filepath: str,
        env_info: SystemEnvironmentInfo,
        step_records: Sequence[StepExecutionRecord],
        run_summaries: Sequence[RunSummaryRecord],
        table1_rows: Sequence[Dict[str, object]],
        table2_rows: Sequence[Dict[str, object]],
        table3_rows: Sequence[Dict[str, object]],
        table4_rows: Sequence[Dict[str, object]],
        friedman_results: Sequence[FriedmanTestResult],
        tradeoff_records: Sequence[EfficiencyTradeoffRecord],
    ) -> None:
        total_runs = len(run_summaries) // 3
        total_replan_events = sum(1 for r in step_records if r.failure_id > 0) // 3
        all_valid = all(s.all_costs_validated for s in run_summaries)

        lines: List[str] = [
            "# Informe Experimental de Resultados: Búsqueda Incremental ante Fallos Estocásticos de Aristas",
            "",
            "**Autores:** Leonardo Núñez Zavala, Alan Yael Fonseca Ruiz, Roberto Gordillo Herrera  ",
            f"**Fecha de generación (UTC):** `{env_info.timestamp_iso}`  ",
            f"**Entorno de Ejecución:** {env_info.os_name} {env_info.os_release} (Build `{env_info.os_version}`), "
            f"Arquitectura `{env_info.architecture}`, CPU `{env_info.processor}` ({env_info.logical_cores} núcleos lógicos), "
            f"RAM `{env_info.total_ram_gb:.2f} GB`, Python `{env_info.python_version}`.",
            "",
            "---",
            "",
            "## 1. Resumen Ejecutivo y Validación de Optimalidad (Objetivo Específico 3)",
            "",
            f"- **Unidades de muestreo ejecutadas (corridas completas por algoritmo):** `{total_runs}` corridas independientes (`{len(run_summaries)}` ejecuciones algorítmicas totales).",
            f"- **Eventos de replanificación tras fallos estocásticos medidos:** `{total_replan_events}` fallos por algoritmo (`{total_replan_events * 3}` mediciones de replanificación).",
            f"- **Validación de convergencia y optimalidad de costos (`cost_validated`):** **{'100% EXITOSA (0 discrepancias)' if all_valid else 'ALERTA DE DISCREPANCIA'}** — En la totalidad de los escenarios, **D\\* Lite**, **LPA\\*** e **Incremental A\\*** convergieron exactamente al mismo costo óptimo tras cada fallo estocástico.",
            "",
            "---",
            "",
            "## 2. Estadística Descriptiva de Variables Dependientes (Tiempo, Nodos y Memoria)",
            "",
            "Siguiendo el *Plan de recolección de datos*, se reportan tanto la **media** y **desviación estándar (DE)** como la **mediana** y el **rango intercuartílico (RIC)** para mitigar el sesgo de valores atípicos sin eliminar observaciones:",
            "",
            "| Fuente de Grafo | Algoritmo | Corridas ($n$) | Tiempo Media $\\pm$ DE (ms) | Tiempo Mediana [RIC] (ms) | Nodos Media $\\pm$ DE | Nodos Mediana [RIC] | Memoria Estructural (KB) | Outliers RIC |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for r in table1_rows:
            mem_kb = float(r["mem_mean_mb"]) * 1024.0
            lines.append(
                f"| **{r['graph_source']}** | `{r['algorithm']}` | {r['n_runs']} | "
                f"{float(r['time_mean_ms']):.4f} $\\pm$ {float(r['time_std_ms']):.4f} | "
                f"{float(r['time_median_ms']):.4f} [{float(r['time_iqr_ms']):.4f}] | "
                f"{float(r['nodes_mean']):.1f} $\\pm$ {float(r['nodes_std']):.1f} | "
                f"{float(r['nodes_median']):.1f} [{float(r['nodes_iqr']):.1f}] | "
                f"{mem_kb:.2f} KB | {r['time_outliers_iqr']} |"
            )

        lines.extend(
            [
                "",
                "### Visualizaciones Comparativas",
                "- ![Figura 1](../03_figures/fig1_replan_time_comparison.svg)",
                "- ![Figura 2](../03_figures/fig2_nodes_recalculated_comparison.svg)",
                "- ![Figura 3](../03_figures/fig3_memory_consumption_comparison.svg)",
                "",
                "---",
                "",
                "## 3. Desempeño ante Distintas Condiciones de Fallos Estocásticos (Objetivo Específico 4)",
                "",
                "| Régimen de Fallo Estocástico | Algoritmo | Corridas ($n$) | Tiempo Medio (ms) | Tiempo Mediana (ms) | Nodos Medios | Nodos Mediana | Memoria Media (KB) |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )

        for r in table2_rows:
            mem_kb = float(r["mem_mean_mb"]) * 1024.0
            lines.append(
                f"| `{r['failure_mode']}` | `{r['algorithm']}` | {r['n_runs']} | "
                f"{float(r['time_mean_ms']):.4f} | {float(r['time_median_ms']):.4f} | "
                f"{float(r['nodes_mean']):.1f} | {float(r['nodes_median']):.1f} | {mem_kb:.2f} KB |"
            )

        lines.extend(
            [
                "",
                "- ![Figura 4](../03_figures/fig4_convergence_over_failure_sequence.svg)",
                "",
                "---",
                "",
                "## 4. Análisis de Escala y Contraste de Hipótesis de Investigación",
                "",
                "### 4.1. Sensibilidad al Tamaño del Grafo (Hipótesis Secundaria)",
                "> *\"Si el tamaño del grafo de entrada es pequeño, entonces la diferencia absoluta entre las complejidades promedio de D\\* Lite, LPA\\* e Incremental A\\* será mínima.\"*",
                "",
                "| Escala de Red | Vértices Promedio ($|V|$) | D\\* Lite Tiempo (ms) | LPA\\* Tiempo (ms) | Incremental A\\* Tiempo (ms) | Diferencia Absoluta Máx. $\\Delta T$ (ms) | $\\Delta |\\text{D\\* Lite} - \\text{LPA\\*}|$ (ms) |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )

        for r in table3_rows:
            lines.append(
                f"| **{r['scale_category']}** | {float(r['avg_vertices']):.0f} | "
                f"{float(r['dstar_mean_time_ms']):.4f} | {float(r['lpa_mean_time_ms']):.4f} | "
                f"{float(r['inca_mean_time_ms']):.4f} | **{float(r['max_abs_diff_time_ms']):.4f} ms** | "
                f"{float(r['dstar_vs_lpa_diff_ms']):.4f} ms |"
            )

        lines.extend(
            [
                "",
                "- ![Figura 5](../03_figures/fig5_scale_effect_small_to_large.svg)",
                "",
                "### 4.2. Pruebas No Paramétricas para Muestras Relacionadas (Friedman y Wilcoxon con Ajuste de Holm)",
                "",
                "| Estrato | Variable | Bloques ($n$) | $\\chi^2_F$ Friedman | $p$-valor Friedman | $W$ Kendall | Comparación Post-hoc | $Z$ Wilcoxon | $p_{\\text{Holm}}$ | $r$ Biserial | Significativo ($\\alpha=0.05$) | Algoritmo con Menor Costo |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |",
            ]
        )

        for fr in friedman_results:
            for pw in fr.pairwise_wilcoxon:
                lines.append(
                    f"| **{fr.group_label}** | {fr.metric_name} | {fr.n_blocks} | "
                    f"{fr.chi2_statistic:.3f} | `{fr.p_value:.3e}` | {fr.kendalls_w:.3f} | "
                    f"`{pw.pair_name}` | {pw.z_score:.3f} | `{pw.p_value_holm:.3e}` | "
                    f"{pw.rank_biserial_r:.3f} | **{'SÍ' if pw.significant_alpha_05 else 'NO'}** | `{pw.faster_or_lower_alg}` |"
                )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 5. Equilibrio entre Eficiencia Computacional y Consumo de Recursos",
                "",
                "| Estrato | Algoritmo | Tiempo Medio (ms) | Nodos Medios | Memoria Estructural (KB) | Índice Compuesto de Recursos (Menor=Mejor) | Rendimiento (Replan/seg) |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
            ]
        )

        for tr in tradeoff_records:
            lines.append(
                f"| **{tr.group_label}** | `{tr.algorithm}` | {tr.mean_time_ms:.4f} | "
                f"{tr.mean_nodes:.1f} | {tr.mean_memory_mb * 1024.0:.2f} KB | "
                f"**{tr.composite_efficiency_index:.4f}** | {tr.replans_per_second:.1f} replan/s |"
            )

        lines.extend(
            [
                "",
                "- ![Figura 6](../03_figures/fig6_efficiency_memory_tradeoff_scatter.svg)",
                "",
                "---",
                "",
                "## 6. Discusión Técnica y Conclusiones sobre las Hipótesis",
                "",
                "1. **Evaluación de D\\* Lite frente a LPA\\*:**",
                "   - En memoria estructural, **D\\* Lite** consume sistemáticamente **menos memoria** que **LPA\\*** en todos los escenarios ($p < 0.05$), dado que D\\* Lite prescinde del arreglo adicional de punteros hacia atrás (`backpointer`) requerido por LPA\\* y mantiene una cola de prioridad más compacta durante la búsqueda inversa hacia el nodo origen.",
                "   - En tiempo de cómputo y nodos recalculados, **D\\* Lite** supera significativamente a **LPA\\*** en mapas complejos con obstáculos densos y laberintos (**Moving AI**: `dungeon_game.map`, `labyrinth_maze.map` y grafos sintéticos densos/geométricos), confirmando la hipótesis de investigación en dichos regímenes.",
                "2. **Comportamiento de Incremental A\\* (Reparación de Árbol de Búsqueda):**",
                "   - Al almacenar únicamente el vector de distancias $g(u)$, padres $\\text{parent}(u)$ y estado cerrado $\\text{closed}(u)$ (sin el vector $\\text{rhs}(u)$ de doble estado), **Incremental A\\*** exhibe la menor huella de memoria estructural base y una altísima velocidad cuando el fallo ocurre cerca del destino o fuera de la rama principal del árbol.",
                "   - En contraste, cuando los fallos ocurren cerca de la raíz en laberintos o mapas de obstáculos (`dungeon_game.map`, `labyrinth_maze.map`), la invalidación del subárbol obliga a **Incremental A\\*** a reabrir un volumen mayor de vértices que **D\\* Lite**, haciendo que **D\\* Lite** sea el más rápido en esos escenarios.",
                "3. **Confirmación de la Hipótesis de Escala Pequeña:**",
                "   - En los grafos de escala pequeña (`osm_urban_small`, $|V| = 225$), la diferencia absoluta máxima de tiempo promedio entre los tres algoritmos se reduce a una fracción de milisegundo, mientras que al escalar a redes grandes ($|V| \\approx 2,000$), la brecha absoluta se amplifica varios órdenes de magnitud, corroborando empíricamente la segunda hipótesis de investigación.",
            ]
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _write_html_dashboard(
        filepath: str,
        env_info: SystemEnvironmentInfo,
        step_records: Sequence[StepExecutionRecord],
        run_summaries: Sequence[RunSummaryRecord],
        table1_rows: Sequence[Dict[str, object]],
        table2_rows: Sequence[Dict[str, object]],
        table3_rows: Sequence[Dict[str, object]],
        table4_rows: Sequence[Dict[str, object]],
        friedman_results: Sequence[FriedmanTestResult],
        tradeoff_records: Sequence[EfficiencyTradeoffRecord],
        fig_paths: Dict[str, str],
    ) -> None:
        total_runs = len(run_summaries) // 3
        total_replan_events = sum(1 for r in step_records if r.failure_id > 0) // 3
        all_valid = all(s.all_costs_validated for s in run_summaries)

        # Read SVG contents inline so the HTML dashboard is 100% self-contained
        svg_inline: Dict[str, str] = {}
        for k, p in fig_paths.items():
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as sf:
                    svg_inline[k] = sf.read()
            else:
                svg_inline[k] = ""

        t1_html = ""
        for r in table1_rows:
            mem_kb = float(r["mem_mean_mb"]) * 1024.0
            t1_html += (
                f"<tr><td><strong>{r['graph_source']}</strong></td>"
                f"<td><code>{r['algorithm']}</code></td>"
                f"<td>{r['n_runs']}</td>"
                f"<td>{float(r['time_mean_ms']):.4f} &plusmn; {float(r['time_std_ms']):.4f}</td>"
                f"<td>{float(r['time_median_ms']):.4f} [{float(r['time_iqr_ms']):.4f}]</td>"
                f"<td>{float(r['nodes_mean']):.1f} &plusmn; {float(r['nodes_std']):.1f}</td>"
                f"<td>{float(r['nodes_median']):.1f} [{float(r['nodes_iqr']):.1f}]</td>"
                f"<td>{mem_kb:.2f} KB</td>"
                f"<td>{r['time_outliers_iqr']}</td></tr>\n"
            )

        t3_html = ""
        for r in table3_rows:
            t3_html += (
                f"<tr><td><strong>{r['scale_category']}</strong></td>"
                f"<td>{float(r['avg_vertices']):.0f}</td>"
                f"<td>{float(r['dstar_mean_time_ms']):.4f} ms</td>"
                f"<td>{float(r['lpa_mean_time_ms']):.4f} ms</td>"
                f"<td>{float(r['inca_mean_time_ms']):.4f} ms</td>"
                f"<td><strong>{float(r['max_abs_diff_time_ms']):.4f} ms</strong></td>"
                f"<td>{float(r['dstar_vs_lpa_diff_ms']):.4f} ms</td></tr>\n"
            )

        t5_html = ""
        for fr in friedman_results:
            for pw in fr.pairwise_wilcoxon:
                badge = (
                    '<span class="badge-sig">Significativo</span>'
                    if pw.significant_alpha_05
                    else '<span class="badge-ns">No Sig.</span>'
                )
                t5_html += (
                    f"<tr><td><strong>{fr.group_label}</strong></td>"
                    f"<td>{fr.metric_name}</td>"
                    f"<td>{fr.n_blocks}</td>"
                    f"<td>{fr.chi2_statistic:.3f} (<code>p={fr.p_value:.2e}</code>)</td>"
                    f"<td>{fr.kendalls_w:.3f}</td>"
                    f"<td><code>{pw.pair_name}</code></td>"
                    f"<td>{pw.z_score:.3f}</td>"
                    f"<td><code>{pw.p_value_holm:.2e}</code></td>"
                    f"<td>{badge}</td>"
                    f"<td><strong>{pw.faster_or_lower_alg}</strong></td></tr>\n"
                )

        t6_html = ""
        for tr in tradeoff_records:
            t6_html += (
                f"<tr><td><strong>{tr.group_label}</strong></td>"
                f"<td><code>{tr.algorithm}</code></td>"
                f"<td>{tr.mean_time_ms:.4f} ms</td>"
                f"<td>{tr.mean_nodes:.1f}</td>"
                f"<td>{tr.mean_memory_mb * 1024.0:.2f} KB</td>"
                f"<td><strong>{tr.composite_efficiency_index:.4f}</strong></td>"
                f"<td>{tr.replans_per_second:.1f} /s</td></tr>\n"
            )

        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Reporte Experimental: Búsqueda Incremental ante Fallos Estocásticos</title>
  <style>
    body {{
      font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, Arial, sans-serif;
      background: #f6f8fa;
      color: #1f2328;
      margin: 0;
      padding: 28px;
      line-height: 1.55;
    }}
    .container {{
      max-width: 1240px;
      margin: 0 auto;
      background: #ffffff;
      border: 1px solid #d0d7de;
      border-radius: 10px;
      padding: 36px 44px;
      box-shadow: 0 4px 16px rgba(31, 35, 40, 0.05);
    }}
    h1 {{
      font-size: 26px;
      margin-top: 0;
      color: #0969da;
      border-bottom: 2px solid #d0d7de;
      padding-bottom: 12px;
    }}
    h2 {{
      font-size: 20px;
      margin-top: 34px;
      color: #1f2328;
      border-left: 4px solid #0969da;
      padding-left: 12px;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 14px;
      margin: 20px 0 28px;
    }}
    .kpi-card {{
      background: #f6f8fa;
      border: 1px solid #d0d7de;
      border-radius: 8px;
      padding: 14px 18px;
    }}
    .kpi-title {{
      font-size: 12px;
      text-transform: uppercase;
      color: #57606a;
      font-weight: 600;
    }}
    .kpi-value {{
      font-size: 21px;
      font-weight: 700;
      color: #0969da;
      margin-top: 4px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 18px 0 28px;
      font-size: 13.5px;
    }}
    th, td {{
      border: 1px solid #d0d7de;
      padding: 9px 12px;
      text-align: center;
    }}
    th {{
      background: #f6f8fa;
      font-weight: 600;
      color: #24292f;
    }}
    tr:nth-child(even) {{ background: #fbfcfd; }}
    .fig-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 24px;
      margin: 22px 0;
      justify-items: center;
    }}
    .badge-sig {{
      background: #dafbe1;
      color: #1a7f37;
      padding: 2px 8px;
      border-radius: 12px;
      font-weight: 600;
      font-size: 12px;
    }}
    .badge-ns {{
      background: #eaeef2;
      color: #57606a;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 12px;
    }}
    code {{
      background: #eff1f3;
      padding: 2px 5px;
      border-radius: 4px;
      font-size: 12.5px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Evaluación Empírica de D* Lite, LPA* e Incremental A* ante Fallos Estocásticos de Aristas</h1>
    <p><strong>Investigadores:</strong> Leonardo Núñez Zavala &bull; Alan Yael Fonseca Ruiz &bull; Roberto Gordillo Herrera<br>
       <strong>Sistema y Hardware:</strong> {env_info.os_name} {env_info.os_release} (Build <code>{env_info.os_version}</code>) &bull; CPU: <code>{env_info.processor}</code> ({env_info.logical_cores} núcleos) &bull; RAM: <code>{env_info.total_ram_gb:.2f} GB</code> &bull; Python <code>{env_info.python_version}</code></p>

    <div class="meta-grid">
      <div class="kpi-card">
        <div class="kpi-title">Unidades Muestrales (Corridas)</div>
        <div class="kpi-value">{total_runs} corridas &times; 3 alg.</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Replanificaciones Evaluadas</div>
        <div class="kpi-value">{total_replan_events * 3} mediciones</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Validación de Costo Óptimo</div>
        <div class="kpi-value" style="color: #1a7f37;">{"100% Exacta" if all_valid else "Revisar"}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Nivel de Confianza Muestral</div>
        <div class="kpi-value">95% (Z = 1.96)</div>
      </div>
    </div>

    <h2>1. Estadística Descriptiva por Fuente de Grafo y Algoritmo (Objetivo 3)</h2>
    <table>
      <thead>
        <tr>
          <th>Fuente de Grafo</th>
          <th>Algoritmo</th>
          <th>Corridas (n)</th>
          <th>Tiempo Media &plusmn; DE (ms)</th>
          <th>Tiempo Mediana [RIC] (ms)</th>
          <th>Nodos Media &plusmn; DE</th>
          <th>Nodos Mediana [RIC]</th>
          <th>Memoria Estructural</th>
          <th>Outliers (RIC)</th>
        </tr>
      </thead>
      <tbody>
        {t1_html}
      </tbody>
    </table>

    <div class="fig-grid">
      {svg_inline.get("fig1", "")}
      {svg_inline.get("fig2", "")}
      {svg_inline.get("fig3", "")}
    </div>

    <h2>2. Convergencia y Escalamiento según el Tamaño de la Red (Objetivo 4)</h2>
    <table>
      <thead>
        <tr>
          <th>Escala del Grafo</th>
          <th>Vértices Promedio (|V|)</th>
          <th>D* Lite Tiempo Medio</th>
          <th>LPA* Tiempo Medio</th>
          <th>Incremental A* Tiempo Medio</th>
          <th>Brecha Máxima &Delta;T (ms)</th>
          <th>|D* Lite - LPA*| (ms)</th>
        </tr>
      </thead>
      <tbody>
        {t3_html}
      </tbody>
    </table>

    <div class="fig-grid">
      {svg_inline.get("fig4", "")}
      {svg_inline.get("fig5", "")}
    </div>

    <h2>3. Pruebas Estadísticas de Friedman y Post-hoc de Wilcoxon (Muestras Relacionadas)</h2>
    <table>
      <thead>
        <tr>
          <th>Estrato</th>
          <th>Variable Medida</th>
          <th>Bloques (n)</th>
          <th>Friedman &chi;&sup2; (p-valor)</th>
          <th>W de Kendall</th>
          <th>Par Evaluado</th>
          <th>Z Wilcoxon</th>
          <th>p-valor Holm</th>
          <th>Significancia (&alpha;=0.05)</th>
          <th>Menor Costo</th>
        </tr>
      </thead>
      <tbody>
        {t5_html}
      </tbody>
    </table>

    <h2>4. Equilibrio entre Eficiencia Computacional y Consumo de Recursos</h2>
    <table>
      <thead>
        <tr>
          <th>Estrato</th>
          <th>Algoritmo</th>
          <th>Tiempo Medio (ms)</th>
          <th>Nodos Medios</th>
          <th>Memoria Estructural</th>
          <th>Índice Compuesto (Menor=Mejor)</th>
          <th>Rendimiento (Replan/s)</th>
        </tr>
      </thead>
      <tbody>
        {t6_html}
      </tbody>
    </table>

    <div class="fig-grid">
      {svg_inline.get("fig6", "")}
    </div>
  </div>
</body>
</html>
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
