#include "incremental_search/evaluation.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <tuple>

namespace incremental_search {

namespace {

double percentile_sorted(const std::vector<double>& s, double q) {
    if (s.empty()) {
        return 0.0;
    }
    if (s.size() == 1) {
        return s.front();
    }
    const double pos = static_cast<double>(s.size() - 1) * q;
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const auto hi = static_cast<std::size_t>(std::ceil(pos));
    if (lo == hi) {
        return s[lo];
    }
    const double frac = pos - static_cast<double>(lo);
    return s[lo] * (1.0 - frac) + s[hi] * frac;
}

} // namespace

DescriptiveStats DescriptiveStats::compute(const std::vector<double>& values) {
    std::vector<double> s;
    s.reserve(values.size());
    for (double v : values) {
        if (!std::isnan(v) && !std::isinf(v)) {
            s.push_back(v);
        }
    }
    DescriptiveStats out;
    out.count = s.size();
    if (s.empty()) {
        return out;
    }
    std::sort(s.begin(), s.end());

    double sum = 0.0;
    for (double v : s) {
        sum += v;
    }
    out.mean = sum / static_cast<double>(s.size());

    if (s.size() > 1) {
        double sq = 0.0;
        for (double v : s) {
            const double d = v - out.mean;
            sq += d * d;
        }
        out.std_dev = std::sqrt(sq / static_cast<double>(s.size() - 1));
    }

    out.p25 = percentile_sorted(s, 0.25);
    out.median = percentile_sorted(s, 0.50);
    out.p75 = percentile_sorted(s, 0.75);
    out.p95 = percentile_sorted(s, 0.95);
    out.iqr = out.p75 - out.p25;
    out.min_val = s.front();
    out.max_val = s.back();

    const double low_fence = out.p25 - 1.5 * out.iqr;
    const double high_fence = out.p75 + 1.5 * out.iqr;
    for (double v : s) {
        if (v < low_fence || v > high_fence) {
            ++out.outliers_iqr_count;
        }
    }
    return out;
}

FriedmanResult BenchmarkEvaluator::compute_friedman_test(
    const std::string& metric_name,
    const std::string& group_label,
    const std::vector<double>& dstar_vals,
    const std::vector<double>& lpa_vals,
    const std::vector<double>& inca_vals) {
    FriedmanResult res;
    res.metric_name = metric_name;
    res.group_label = group_label;
    const std::size_t n = std::min({dstar_vals.size(), lpa_vals.size(), inca_vals.size()});
    res.n_blocks = n;
    if (n == 0) {
        return res;
    }

    double rank_sums[3] = {0.0, 0.0, 0.0};
    double tie_sum = 0.0;

    for (std::size_t idx = 0; idx < n; ++idx) {
        std::pair<double, int> row[3] = {
            {dstar_vals[idx], 0},
            {lpa_vals[idx], 1},
            {inca_vals[idx], 2}
        };
        std::sort(std::begin(row), std::end(row));
        int i = 0;
        while (i < 3) {
            int j = i;
            while (j + 1 < 3 && std::fabs(row[j + 1].first - row[i].first) <= 1e-12) {
                ++j;
            }
            const double avg_r = 0.5 * static_cast<double>((i + 1) + (j + 1));
            const int t_len = j - i + 1;
            if (t_len > 1) {
                tie_sum += static_cast<double>(t_len * t_len * t_len - t_len);
            }
            for (int m = i; m <= j; ++m) {
                rank_sums[row[m].second] += avg_r;
            }
            i = j + 1;
        }
    }

    const double nd = static_cast<double>(n);
    const double raw_q =
        (12.0 / (nd * 3.0 * 4.0)) *
            (rank_sums[0] * rank_sums[0] + rank_sums[1] * rank_sums[1] + rank_sums[2] * rank_sums[2]) -
        3.0 * nd * 4.0;
    const double denom_tie = 1.0 - tie_sum / (nd * 24.0);
    res.chi2_statistic = (denom_tie > 1e-9) ? std::max(0.0, raw_q / denom_tie) : 0.0;
    res.p_value = std::exp(-0.5 * res.chi2_statistic);
    res.kendalls_w = std::min(1.0, res.chi2_statistic / (nd * 2.0));
    res.mean_ranks["D* Lite"] = rank_sums[0] / nd;
    res.mean_ranks["LPA*"] = rank_sums[1] / nd;
    res.mean_ranks["Incremental A*"] = rank_sums[2] / nd;
    res.significant_alpha_05 = (res.p_value < 0.05);
    return res;
}

bool BenchmarkEvaluator::export_research_report(
    const std::string& destination_dir,
    const std::vector<StepExecutionRecord>& step_records,
    const std::vector<RunSummaryRecord>& run_summaries) {
    namespace fs = std::filesystem;
    fs::path dest(destination_dir);
    fs::path raw_dir = dest / "01_raw_data";
    fs::path tables_dir = dest / "02_statistical_tables";
    fs::path report_dir = dest / "04_research_report";

    fs::create_directories(raw_dir);
    fs::create_directories(tables_dir);
    fs::create_directories(report_dir);

    StochasticSimulator::export_steps_csv(
        step_records, (raw_dir / "cpp_replanning_steps_raw.csv").string(), false);
    StochasticSimulator::export_summaries_csv(
        run_summaries, (raw_dir / "cpp_experimental_runs_raw.csv").string(), false);

    std::ofstream desc_csv((tables_dir / "cpp_table1_descriptive.csv").string());
    desc_csv << "graph_source,algorithm,n_runs,time_mean_ms,time_std_ms,time_median_ms,"
                "time_iqr_ms,nodes_mean,nodes_median,mem_mean_mb,mem_peak_mb\n";
    desc_csv << std::fixed << std::setprecision(6);

    const std::vector<std::string> sources = {"ALL", "osm", "movingai", "synthetic"};
    const std::vector<std::string> algs = {"D* Lite", "LPA*", "Incremental A*"};

    for (const auto& src : sources) {
        for (const auto& alg : algs) {
            std::vector<double> t_vals, n_vals, m_vals;
            double peak_m = 0.0;
            for (const auto& s : run_summaries) {
                if (s.algorithm == alg && (src == "ALL" || s.graph_type == src)) {
                    t_vals.push_back(s.mean_replan_time_ms);
                    n_vals.push_back(s.mean_replan_nodes);
                    m_vals.push_back(s.mean_memory_mb);
                    peak_m = std::max(peak_m, s.peak_memory_mb);
                }
            }
            if (t_vals.empty()) {
                continue;
            }
            auto ts = DescriptiveStats::compute(t_vals);
            auto ns = DescriptiveStats::compute(n_vals);
            auto ms = DescriptiveStats::compute(m_vals);
            desc_csv << src << "," << alg << "," << ts.count << ","
                     << ts.mean << "," << ts.std_dev << "," << ts.median << "," << ts.iqr << ","
                     << ns.mean << "," << ns.median << "," << ms.mean << "," << peak_m << "\n";
        }
    }

    return true;
}

} // namespace incremental_search
