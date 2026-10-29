from __future__ import annotations

import os
import tempfile

from incremental_search import (
    DescriptiveStats,
    FailureMode,
    GraphIO,
    NonParametricTests,
    ResearchReportBuilder,
    SimulationConfig,
    StochasticSimulator,
    SystemEnvironmentInfo,
)


def test_descriptive_stats_and_iqr_outliers() -> None:
    vals = [1.0, 1.1, 1.2, 1.15, 1.05, 1.18, 1.12, 15.0]
    s = DescriptiveStats.compute(vals)
    assert s.count == 8
    assert s.median > 1.0 and s.median < 1.3
    assert s.outliers_iqr_count >= 1
    assert s.p95 >= 1.18


def test_friedman_and_wilcoxon_holm_tests() -> None:
    dstar = [1.1, 1.2, 1.0, 1.3, 1.15, 1.05, 1.22, 1.18, 1.09, 1.11]
    lpa = [2.4, 2.5, 2.3, 2.7, 2.45, 2.35, 2.60, 2.55, 2.40, 2.50]
    inca = [1.8, 1.9, 1.7, 2.0, 1.85, 1.75, 1.95, 1.88, 1.79, 1.82]

    res = NonParametricTests.friedman_with_wilcoxon(
        "Tiempo de Cómputo (ms)", "TEST_STRATUM", dstar, lpa, inca
    )
    assert res.n_blocks == 10
    assert res.chi2_statistic > 10.0
    assert res.p_value < 0.01
    assert res.significant_alpha_05
    assert res.mean_ranks["D* Lite"] < res.mean_ranks["Incremental A*"] < res.mean_ranks["LPA*"]
    assert len(res.pairwise_wilcoxon) == 3
    for pw in res.pairwise_wilcoxon:
        assert pw.p_value_holm < 0.05
        assert pw.significant_alpha_05


def test_research_report_builder_creates_full_folder() -> None:
    g = GraphIO.generate_synthetic_grid(18, 18, 0.15, 8, 901)
    s, t = GraphIO.sample_valid_od_pair(g, 901)

    steps = []
    summaries = []
    modes_map = {}
    scale_map = {"mini_syn": "Pequeña"}

    for run_i in range(1, 5):
        seed = 900 + run_i
        cfg = SimulationConfig(
            map_name="mini_syn",
            graph_type="synthetic",
            seed=seed,
            num_failures=4,
            failure_mode=FailureMode.ON_CURRENT_PATH,
        )
        out = StochasticSimulator.run_scenario(g, s, t, cfg)
        steps.extend(out.step_records)
        summaries.extend(out.run_summaries)
        modes_map[("mini_syn", seed)] = FailureMode.ON_CURRENT_PATH.value

    env = SystemEnvironmentInfo.capture()
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts = ResearchReportBuilder.build_complete_output_folder(
            destination_dir=tmpdir,
            step_records=steps,
            run_summaries=summaries,
            env_info=env,
            failure_mode_by_run_key=modes_map,
            scale_by_map=scale_map,
        )
        for key, path in artifacts.items():
            assert os.path.exists(path), f"Missing artifact {key}: {path}"
        assert os.path.exists(
            os.path.join(tmpdir, "03_figures", "fig1_replan_time_comparison.svg")
        )
        assert os.path.exists(
            os.path.join(tmpdir, "03_figures", "fig6_efficiency_memory_tradeoff_scatter.svg")
        )
