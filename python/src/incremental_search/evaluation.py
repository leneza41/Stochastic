from __future__ import annotations

import csv
import ctypes
import datetime
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .simulation import (
    FailureMode,
    RunSummaryRecord,
    SampleSizeCalculator,
    StepExecutionRecord,
)


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


@dataclass(slots=True)
class SystemEnvironmentInfo:
    os_name: str
    os_release: str
    os_version: str
    architecture: str
    processor: str
    logical_cores: int
    total_ram_gb: float
    python_version: str
    timestamp_iso: str

    @staticmethod
    def capture() -> SystemEnvironmentInfo:
        ram_gb = 0.0
        if sys.platform == "win32":
            try:
                stat = _MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    ram_gb = float(stat.ullTotalPhys) / (1024.0**3)
            except Exception:
                pass

        return SystemEnvironmentInfo(
            os_name=platform.system(),
            os_release=platform.release(),
            os_version=platform.version(),
            architecture=platform.machine(),
            processor=platform.processor() or "x86_64 / AMD64 Family",
            logical_cores=os.cpu_count() or 1,
            total_ram_gb=round(ram_gb, 2),
            python_version=platform.python_version(),
            timestamp_iso=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )


@dataclass(slots=True)
class DescriptiveStats:
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    iqr: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    outliers_iqr_count: int = 0

    @staticmethod
    def _percentile(sorted_vals: Sequence[float], q: float) -> float:
        if not sorted_vals:
            return 0.0
        if len(sorted_vals) == 1:
            return float(sorted_vals[0])
        pos = (len(sorted_vals) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(sorted_vals[lo])
        frac = pos - lo
        return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac

    @staticmethod
    def compute(values: Sequence[float]) -> DescriptiveStats:
        clean = [float(x) for x in values if not math.isnan(x) and not math.isinf(x)]
        n = len(clean)
        if n == 0:
            return DescriptiveStats()
        s = sorted(clean)
        mean_v = sum(s) / n
        if n > 1:
            var_v = sum((x - mean_v) ** 2 for x in s) / (n - 1)
            std_v = math.sqrt(max(0.0, var_v))
        else:
            std_v = 0.0

        p25 = DescriptiveStats._percentile(s, 0.25)
        med = DescriptiveStats._percentile(s, 0.50)
        p75 = DescriptiveStats._percentile(s, 0.75)
        p95 = DescriptiveStats._percentile(s, 0.95)
        iqr_v = p75 - p25
        low_fence = p25 - 1.5 * iqr_v
        high_fence = p75 + 1.5 * iqr_v
        outliers = sum(1 for x in s if x < low_fence or x > high_fence)

        return DescriptiveStats(
            count=n,
            mean=mean_v,
            std=std_v,
            median=med,
            iqr=iqr_v,
            p25=p25,
            p75=p75,
            p95=p95,
            min_val=s[0],
            max_val=s[-1],
            outliers_iqr_count=outliers,
        )


@dataclass(slots=True)
class WilcoxonPairwiseResult:
    pair_name: str
    alg_a: str
    alg_b: str
    n_pairs: int
    w_plus: float
    w_minus: float
    w_stat: float
    z_score: float
    p_value_two_sided: float
    p_value_holm: float
    rank_biserial_r: float
    significant_alpha_05: bool
    faster_or_lower_alg: str


@dataclass(slots=True)
class FriedmanTestResult:
    metric_name: str
    group_label: str
    n_blocks: int
    chi2_statistic: float
    df: int
    p_value: float
    kendalls_w: float
    mean_ranks: Dict[str, float]
    significant_alpha_05: bool
    pairwise_wilcoxon: List[WilcoxonPairwiseResult] = field(default_factory=list)


@dataclass(slots=True)
class EfficiencyTradeoffRecord:
    algorithm: str
    group_label: str
    mean_time_ms: float
    median_time_ms: float
    mean_nodes: float
    median_nodes: float
    mean_memory_mb: float
    peak_memory_mb: float
    normalized_time: float
    normalized_nodes: float
    normalized_memory: float
    composite_efficiency_index: float  # Lower is better (0.0 = ideal, 1.0 = worst)
    replans_per_second: float
    nodes_per_ms: float


class NonParametricTests:
    """
    Implements Friedman test for k=3 related samples and post-hoc Wilcoxon
    signed-rank tests with Holm-Bonferroni adjustment (as specified in
    "Plan de recolección de datos.pdf").
    """

    @staticmethod
    def _normal_two_sided_p(z: float) -> float:
        return math.erfc(abs(z) / math.sqrt(2.0))

    @staticmethod
    def _chi2_df2_p_value(chi2_val: float) -> float:
        # For df = 2, the chi-square survival function has closed form: exp(-x / 2)
        if chi2_val <= 0.0:
            return 1.0
        return math.exp(-0.5 * chi2_val)

    @staticmethod
    def wilcoxon_signed_rank(
        x: Sequence[float],
        y: Sequence[float],
        alg_a: str,
        alg_b: str,
    ) -> WilcoxonPairwiseResult:
        diffs: List[Tuple[float, int]] = []
        for a, b in zip(x, y):
            d = float(a) - float(b)
            if abs(d) > 1e-12:
                diffs.append((abs(d), 1 if d > 0 else -1))

        n = len(diffs)
        if n == 0:
            return WilcoxonPairwiseResult(
                pair_name=f"{alg_a} vs {alg_b}",
                alg_a=alg_a,
                alg_b=alg_b,
                n_pairs=0,
                w_plus=0.0,
                w_minus=0.0,
                w_stat=0.0,
                z_score=0.0,
                p_value_two_sided=1.0,
                p_value_holm=1.0,
                rank_biserial_r=0.0,
                significant_alpha_05=False,
                faster_or_lower_alg="Tie",
            )

        diffs.sort(key=lambda item: item[0])
        w_plus = 0.0
        w_minus = 0.0
        tie_correction_sum = 0.0

        i = 0
        while i < n:
            j = i
            while j + 1 < n and abs(diffs[j + 1][0] - diffs[i][0]) <= 1e-12:
                j += 1
            avg_rank = 0.5 * ((i + 1) + (j + 1))
            t_count = j - i + 1
            if t_count > 1:
                tie_correction_sum += float(t_count**3 - t_count)
            for k in range(i, j + 1):
                if diffs[k][1] > 0:
                    w_plus += avg_rank
                else:
                    w_minus += avg_rank
            i = j + 1

        w_stat = min(w_plus, w_minus)
        mean_w = n * (n + 1) / 4.0
        var_w = (n * (n + 1) * (2 * n + 1) - 0.5 * tie_correction_sum) / 24.0
        if var_w > 0.0:
            continuity = 0.5 if abs(w_stat - mean_w) >= 0.5 else 0.0
            z = (abs(w_stat - mean_w) - continuity) / math.sqrt(var_w)
        else:
            z = 0.0

        p_two = NonParametricTests._normal_two_sided_p(z)
        total_rank_sum = n * (n + 1) / 2.0
        r_biserial = (
            (w_plus - w_minus) / total_rank_sum if total_rank_sum > 0 else 0.0
        )
        better = alg_a if w_plus < w_minus else (alg_b if w_minus < w_plus else "Tie")

        return WilcoxonPairwiseResult(
            pair_name=f"{alg_a} vs {alg_b}",
            alg_a=alg_a,
            alg_b=alg_b,
            n_pairs=n,
            w_plus=w_plus,
            w_minus=w_minus,
            w_stat=w_stat,
            z_score=z,
            p_value_two_sided=p_two,
            p_value_holm=p_two,
            rank_biserial_r=r_biserial,
            significant_alpha_05=(p_two < 0.05),
            faster_or_lower_alg=better,
        )

    @staticmethod
    def friedman_with_wilcoxon(
        metric_name: str,
        group_label: str,
        dstar_vals: Sequence[float],
        lpa_vals: Sequence[float],
        inca_vals: Sequence[float],
    ) -> FriedmanTestResult:
        n = min(len(dstar_vals), len(lpa_vals), len(inca_vals))
        alg_names = ["D* Lite", "LPA*", "Incremental A*"]
        if n == 0:
            return FriedmanTestResult(
                metric_name=metric_name,
                group_label=group_label,
                n_blocks=0,
                chi2_statistic=0.0,
                df=2,
                p_value=1.0,
                kendalls_w=0.0,
                mean_ranks={a: 2.0 for a in alg_names},
                significant_alpha_05=False,
            )

        rank_sums = [0.0, 0.0, 0.0]
        tie_sum = 0.0

        for idx in range(n):
            row = [
                (float(dstar_vals[idx]), 0),
                (float(lpa_vals[idx]), 1),
                (float(inca_vals[idx]), 2),
            ]
            row.sort(key=lambda item: item[0])
            i = 0
            while i < 3:
                j = i
                while j + 1 < 3 and abs(row[j + 1][0] - row[i][0]) <= 1e-12:
                    j += 1
                avg_r = 0.5 * ((i + 1) + (j + 1))
                t_len = j - i + 1
                if t_len > 1:
                    tie_sum += float(t_len**3 - t_len)
                for m in range(i, j + 1):
                    rank_sums[row[m][1]] += avg_r
                i = j + 1

        k = 3
        raw_q = (12.0 / (n * k * (k + 1))) * sum(r**2 for r in rank_sums) - 3.0 * n * (
            k + 1
        )
        denom_tie = 1.0 - tie_sum / float(n * k * (k**2 - 1))
        chi2_f = raw_q / denom_tie if denom_tie > 1e-9 else 0.0
        chi2_f = max(0.0, chi2_f)
        p_val = NonParametricTests._chi2_df2_p_value(chi2_f)
        kendalls_w = min(1.0, chi2_f / float(n * (k - 1)))

        mean_ranks = {alg_names[i]: rank_sums[i] / float(n) for i in range(3)}

        pairs = [
            NonParametricTests.wilcoxon_signed_rank(
                dstar_vals[:n], lpa_vals[:n], "D* Lite", "LPA*"
            ),
            NonParametricTests.wilcoxon_signed_rank(
                dstar_vals[:n], inca_vals[:n], "D* Lite", "Incremental A*"
            ),
            NonParametricTests.wilcoxon_signed_rank(
                lpa_vals[:n], inca_vals[:n], "LPA*", "Incremental A*"
            ),
        ]

        # Apply Holm-Bonferroni step-down adjustment across the 3 pairwise comparisons
        order = sorted(range(len(pairs)), key=lambda idx: pairs[idx].p_value_two_sided)
        running_max = 0.0
        for rank_idx, orig_idx in enumerate(order):
            multiplier = len(pairs) - rank_idx
            adj_p = min(1.0, pairs[orig_idx].p_value_two_sided * multiplier)
            running_max = max(running_max, adj_p)
            pairs[orig_idx].p_value_holm = running_max
            pairs[orig_idx].significant_alpha_05 = running_max < 0.05

        return FriedmanTestResult(
            metric_name=metric_name,
            group_label=group_label,
            n_blocks=n,
            chi2_statistic=chi2_f,
            df=2,
            p_value=p_val,
            kendalls_w=kendalls_w,
            mean_ranks=mean_ranks,
            significant_alpha_05=(p_val < 0.05),
            pairwise_wilcoxon=pairs,
        )


class ResearchFigureGenerator:
    """
    Generates publication-quality vector graphics (.svg) for all comparative
    metrics, convergence profiles, scale sensitivity, and resource trade-offs.
    """

    _ALG_COLORS = {
        "D* Lite": "#1f77b4",       # Deep Academic Blue
        "LPA*": "#ff7f0e",          # Warm Amber/Orange
        "Incremental A*": "#2ca02c", # Emerald Green
    }

    @staticmethod
    def render_grouped_bar_svg(
        filepath: str,
        title: str,
        subtitle: str,
        y_label: str,
        categories: Sequence[str],
        series_data: Dict[str, Sequence[float]],
        series_errors: Optional[Dict[str, Sequence[float]]] = None,
    ) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        width, height = 880, 480
        pad_left, pad_right, pad_top, pad_bottom = 90, 40, 80, 85
        plot_w = width - pad_left - pad_right
        plot_h = height - pad_top - pad_bottom

        alg_names = ["D* Lite", "LPA*", "Incremental A*"]
        max_val = 0.0
        for alg in alg_names:
            vals = series_data.get(alg, [])
            errs = series_errors.get(alg, [0.0] * len(vals)) if series_errors else [0.0] * len(vals)
            for v, e in zip(vals, errs):
                max_val = max(max_val, v + e)
        if max_val <= 0.0:
            max_val = 1.0
        y_max = max_val * 1.18

        lines: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            '<style>',
            '  .bg { fill: #ffffff; stroke: #d0d7de; stroke-width: 1; }',
            '  .title { font-family: "Segoe UI", Arial, sans-serif; font-size: 17px; font-weight: 700; fill: #1f2328; }',
            '  .subtitle { font-family: "Segoe UI", Arial, sans-serif; font-size: 12.5px; fill: #57606a; }',
            '  .axis-label { font-family: "Segoe UI", Arial, sans-serif; font-size: 12.5px; font-weight: 600; fill: #24292f; }',
            '  .tick { font-family: "Segoe UI", Arial, sans-serif; font-size: 11px; fill: #424a53; }',
            '  .val-label { font-family: "Segoe UI", Arial, sans-serif; font-size: 10px; font-weight: 600; fill: #1f2328; }',
            '  .grid { stroke: #eaeef2; stroke-width: 1; stroke-dasharray: 4,3; }',
            '  .axis { stroke: #424a53; stroke-width: 1.5; }',
            '</style>',
            f'<rect class="bg" x="0" y="0" width="{width}" height="{height}" rx="6"/>',
            f'<text class="title" x="{pad_left}" y="32">{title}</text>',
            f'<text class="subtitle" x="{pad_left}" y="52">{subtitle}</text>',
        ]

        # Legend
        leg_x = width - pad_right - 330
        for idx, alg in enumerate(alg_names):
            lx = leg_x + idx * 112
            color = ResearchFigureGenerator._ALG_COLORS[alg]
            lines.append(f'<rect x="{lx}" y="22" width="13" height="13" rx="2" fill="{color}"/>')
            lines.append(f'<text class="tick" x="{lx + 18}" y="33">{alg}</text>')

        # Y grid & ticks
        num_ticks = 5
        for t in range(num_ticks + 1):
            frac = t / float(num_ticks)
            val = y_max * frac
            py = pad_top + plot_h - frac * plot_h
            lines.append(
                f'<line class="grid" x1="{pad_left}" y1="{py:.1f}" x2="{width - pad_right}" y2="{py:.1f}"/>'
            )
            lbl = f"{val:.3f}" if y_max < 1.0 else (f"{val:.2f}" if y_max < 25.0 else f"{val:.0f}")
            lines.append(
                f'<text class="tick" x="{pad_left - 10}" y="{py + 4:.1f}" text-anchor="end">{lbl}</text>'
            )

        # Y-axis title
        cy = pad_top + plot_h * 0.5
        lines.append(
            f'<text class="axis-label" transform="translate(24,{cy:.1f}) rotate(-90)" text-anchor="middle">{y_label}</text>'
        )

        # Grouped Bars
        n_cats = max(1, len(categories))
        group_w = plot_w / float(n_cats)
        bar_w = min(38.0, (group_w * 0.72) / 3.0)

        for c_idx, cat in enumerate(categories):
            group_cx = pad_left + (c_idx + 0.5) * group_w
            start_x = group_cx - 1.5 * bar_w - 2.0
            for a_idx, alg in enumerate(alg_names):
                vals = series_data.get(alg, [])
                val = vals[c_idx] if c_idx < len(vals) else 0.0
                bh = max(1.0, (val / y_max) * plot_h)
                bx = start_x + a_idx * (bar_w + 2.0)
                by = pad_top + plot_h - bh
                color = ResearchFigureGenerator._ALG_COLORS[alg]
                lines.append(
                    f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{color}" rx="2" opacity="0.92"/>'
                )
                val_txt = f"{val:.4f}" if y_max < 0.2 else (f"{val:.2f}" if y_max < 50.0 else f"{val:.0f}")
                lines.append(
                    f'<text class="val-label" x="{bx + bar_w * 0.5:.1f}" y="{by - 6:.1f}" text-anchor="middle">{val_txt}</text>'
                )

            lines.append(
                f'<text class="axis-label" x="{group_cx:.1f}" y="{pad_top + plot_h + 28}" text-anchor="middle">{cat}</text>'
            )

        lines.append(
            f'<line class="axis" x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + plot_h}"/>'
        )
        lines.append(
            f'<line class="axis" x1="{pad_left}" y1="{pad_top + plot_h}" x2="{width - pad_right}" y2="{pad_top + plot_h}"/>'
        )
        lines.append("</svg>")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def render_convergence_line_svg(
        filepath: str,
        title: str,
        subtitle: str,
        x_label: str,
        y_label: str,
        steps: Sequence[int],
        series_data: Dict[str, Sequence[float]],
    ) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        width, height = 880, 460
        pad_left, pad_right, pad_top, pad_bottom = 90, 40, 80, 75
        plot_w = width - pad_left - pad_right
        plot_h = height - pad_top - pad_bottom

        alg_names = ["D* Lite", "LPA*", "Incremental A*"]
        max_val = 0.0
        for alg in alg_names:
            for v in series_data.get(alg, []):
                max_val = max(max_val, v)
        y_max = max(1e-3, max_val * 1.18)

        lines: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            '<style>',
            '  .bg { fill: #ffffff; stroke: #d0d7de; stroke-width: 1; }',
            '  .title { font-family: "Segoe UI", Arial, sans-serif; font-size: 17px; font-weight: 700; fill: #1f2328; }',
            '  .subtitle { font-family: "Segoe UI", Arial, sans-serif; font-size: 12.5px; fill: #57606a; }',
            '  .axis-label { font-family: "Segoe UI", Arial, sans-serif; font-size: 12.5px; font-weight: 600; fill: #24292f; }',
            '  .tick { font-family: "Segoe UI", Arial, sans-serif; font-size: 11px; fill: #424a53; }',
            '  .grid { stroke: #eaeef2; stroke-width: 1; stroke-dasharray: 4,3; }',
            '  .axis { stroke: #424a53; stroke-width: 1.5; }',
            '</style>',
            f'<rect class="bg" x="0" y="0" width="{width}" height="{height}" rx="6"/>',
            f'<text class="title" x="{pad_left}" y="32">{title}</text>',
            f'<text class="subtitle" x="{pad_left}" y="52">{subtitle}</text>',
        ]

        leg_x = width - pad_right - 330
        for idx, alg in enumerate(alg_names):
            lx = leg_x + idx * 112
            color = ResearchFigureGenerator._ALG_COLORS[alg]
            lines.append(f'<rect x="{lx}" y="22" width="13" height="13" rx="2" fill="{color}"/>')
            lines.append(f'<text class="tick" x="{lx + 18}" y="33">{alg}</text>')

        num_ticks = 5
        for t in range(num_ticks + 1):
            frac = t / float(num_ticks)
            val = y_max * frac
            py = pad_top + plot_h - frac * plot_h
            lines.append(
                f'<line class="grid" x1="{pad_left}" y1="{py:.1f}" x2="{width - pad_right}" y2="{py:.1f}"/>'
            )
            lbl = f"{val:.3f}" if y_max < 2.0 else f"{val:.2f}"
            lines.append(
                f'<text class="tick" x="{pad_left - 10}" y="{py + 4:.1f}" text-anchor="end">{lbl}</text>'
            )

        n_pts = max(1, len(steps))
        for s_idx, step_num in enumerate(steps):
            px = (
                pad_left + (s_idx / float(max(1, n_pts - 1))) * plot_w
                if n_pts > 1
                else pad_left + plot_w * 0.5
            )
            lines.append(
                f'<text class="tick" x="{px:.1f}" y="{pad_top + plot_h + 22}" text-anchor="middle">Fallo #{step_num}</text>'
            )

        for alg in alg_names:
            vals = series_data.get(alg, [])
            if not vals:
                continue
            color = ResearchFigureGenerator._ALG_COLORS[alg]
            pts: List[str] = []
            for s_idx, val in enumerate(vals):
                px = (
                    pad_left + (s_idx / float(max(1, n_pts - 1))) * plot_w
                    if n_pts > 1
                    else pad_left + plot_w * 0.5
                )
                py = pad_top + plot_h - (val / y_max) * plot_h
                pts.append(f"{px:.1f},{py:.1f}")
            poly_pts = " ".join(pts)
            lines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.6" points="{poly_pts}"/>'
            )
            for pt in pts:
                cx_s, cy_s = pt.split(",")
                lines.append(
                    f'<circle cx="{cx_s}" cy="{cy_s}" r="4.2" fill="{color}" stroke="#ffffff" stroke-width="1.2"/>'
                )

        cy = pad_top + plot_h * 0.5
        lines.append(
            f'<text class="axis-label" transform="translate(24,{cy:.1f}) rotate(-90)" text-anchor="middle">{y_label}</text>'
        )
        lines.append(
            f'<text class="axis-label" x="{pad_left + plot_w * 0.5:.1f}" y="{height - 18}" text-anchor="middle">{x_label}</text>'
        )
        lines.append(
            f'<line class="axis" x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + plot_h}"/>'
        )
        lines.append(
            f'<line class="axis" x1="{pad_left}" y1="{pad_top + plot_h}" x2="{width - pad_right}" y2="{pad_top + plot_h}"/>'
        )
        lines.append("</svg>")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def render_tradeoff_scatter_svg(
        filepath: str,
        title: str,
        subtitle: str,
        records: Sequence[EfficiencyTradeoffRecord],
    ) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        width, height = 880, 480
        pad_left, pad_right, pad_top, pad_bottom = 95, 45, 80, 75
        plot_w = width - pad_left - pad_right
        plot_h = height - pad_top - pad_bottom

        max_mem_kb = max((r.mean_memory_mb * 1024.0 for r in records), default=10.0) * 1.18
        max_time = max((r.mean_time_ms for r in records), default=1.0) * 1.18
        max_nodes = max((r.mean_nodes for r in records), default=100.0)

        lines: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            '<style>',
            '  .bg { fill: #ffffff; stroke: #d0d7de; stroke-width: 1; }',
            '  .title { font-family: "Segoe UI", Arial, sans-serif; font-size: 17px; font-weight: 700; fill: #1f2328; }',
            '  .subtitle { font-family: "Segoe UI", Arial, sans-serif; font-size: 12.5px; fill: #57606a; }',
            '  .axis-label { font-family: "Segoe UI", Arial, sans-serif; font-size: 12.5px; font-weight: 600; fill: #24292f; }',
            '  .tick { font-family: "Segoe UI", Arial, sans-serif; font-size: 11px; fill: #424a53; }',
            '  .pt-label { font-family: "Segoe UI", Arial, sans-serif; font-size: 10px; font-weight: 600; fill: #24292f; }',
            '  .grid { stroke: #eaeef2; stroke-width: 1; stroke-dasharray: 4,3; }',
            '  .axis { stroke: #424a53; stroke-width: 1.5; }',
            '</style>',
            f'<rect class="bg" x="0" y="0" width="{width}" height="{height}" rx="6"/>',
            f'<text class="title" x="{pad_left}" y="32">{title}</text>',
            f'<text class="subtitle" x="{pad_left}" y="52">{subtitle}</text>',
        ]

        alg_names = ["D* Lite", "LPA*", "Incremental A*"]
        leg_x = width - pad_right - 330
        for idx, alg in enumerate(alg_names):
            lx = leg_x + idx * 112
            color = ResearchFigureGenerator._ALG_COLORS[alg]
            lines.append(f'<circle cx="{lx + 6}" cy="28" r="6" fill="{color}"/>')
            lines.append(f'<text class="tick" x="{lx + 16}" y="32">{alg}</text>')

        for t in range(6):
            frac = t / 5.0
            py = pad_top + plot_h - frac * plot_h
            val_y = max_time * frac
            lines.append(
                f'<line class="grid" x1="{pad_left}" y1="{py:.1f}" x2="{width - pad_right}" y2="{py:.1f}"/>'
            )
            lines.append(
                f'<text class="tick" x="{pad_left - 10}" y="{py + 4:.1f}" text-anchor="end">{val_y:.2f}</text>'
            )
            px = pad_left + frac * plot_w
            val_x = max_mem_kb * frac
            lines.append(
                f'<text class="tick" x="{px:.1f}" y="{pad_top + plot_h + 22}" text-anchor="middle">{val_x:.1f} KB</text>'
            )

        for r in records:
            mem_kb = r.mean_memory_mb * 1024.0
            px = pad_left + (mem_kb / max_mem_kb) * plot_w
            py = pad_top + plot_h - (r.mean_time_ms / max_time) * plot_h
            radius = 7.0 + 14.0 * math.sqrt(max(0.0, r.mean_nodes) / max(1.0, max_nodes))
            color = ResearchFigureGenerator._ALG_COLORS.get(r.algorithm, "#333333")
            lines.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" fill="{color}" opacity="0.72" stroke="{color}" stroke-width="1.5"/>'
            )
            lines.append(
                f'<text class="pt-label" x="{px:.1f}" y="{py - radius - 4:.1f}" text-anchor="middle">{r.group_label}</text>'
            )

        cy = pad_top + plot_h * 0.5
        lines.append(
            f'<text class="axis-label" transform="translate(24,{cy:.1f}) rotate(-90)" text-anchor="middle">Tiempo Promedio de Replanificación (ms)</text>'
        )
        lines.append(
            f'<text class="axis-label" x="{pad_left + plot_w * 0.5:.1f}" y="{height - 18}" text-anchor="middle">Memoria Estructural Promedio (KB) [Tamaño de burbuja = Nodos Recalculados]</text>'
        )
        lines.append(
            f'<line class="axis" x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + plot_h}"/>'
        )
        lines.append(
            f'<line class="axis" x1="{pad_left}" y1="{pad_top + plot_h}" x2="{width - pad_right}" y2="{pad_top + plot_h}"/>'
        )
        lines.append("</svg>")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
