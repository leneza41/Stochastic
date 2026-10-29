#pragma once

#include "incremental_search/simulation.hpp"
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace incremental_search {

struct DescriptiveStats {
    std::size_t count{0};
    double mean{0.0};
    double std_dev{0.0};
    double median{0.0};
    double iqr{0.0};
    double p25{0.0};
    double p75{0.0};
    double p95{0.0};
    double min_val{0.0};
    double max_val{0.0};
    std::size_t outliers_iqr_count{0};

    static DescriptiveStats compute(const std::vector<double>& values);
};

struct FriedmanResult {
    std::string metric_name;
    std::string group_label;
    std::size_t n_blocks{0};
    double chi2_statistic{0.0};
    double p_value{1.0};
    double kendalls_w{0.0};
    std::map<std::string, double> mean_ranks;
    bool significant_alpha_05{false};
};

class BenchmarkEvaluator {
public:
    static FriedmanResult compute_friedman_test(const std::string& metric_name,
                                                const std::string& group_label,
                                                const std::vector<double>& dstar_vals,
                                                const std::vector<double>& lpa_vals,
                                                const std::vector<double>& inca_vals);

    static bool export_research_report(const std::string& destination_dir,
                                       const std::vector<StepExecutionRecord>& step_records,
                                       const std::vector<RunSummaryRecord>& run_summaries);
};

} // namespace incremental_search
