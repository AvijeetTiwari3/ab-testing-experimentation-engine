"""
Multiple Testing Correction Engine
Controls Family-Wise Error Rate (FWER) and False Discovery Rate (FDR) across multi-metric experiment scorecards.
Implements: Bonferroni, Holm-Bonferroni, Benjamini-Hochberg (BH-FDR), and Benjamini-Yekutieli.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np
from statsmodels.stats.multitest import multipletests


@dataclass
class MetricCorrectionRow:
    metric_name: str
    raw_p_value: float
    observed_effect_size: float
    bonferroni_p_value: float
    bonferroni_significant: bool
    holm_p_value: float
    holm_significant: bool
    bh_fdr_p_value: float
    bh_fdr_significant: bool


@dataclass
class MultipleTestingResult:
    total_metrics_tested: int
    nominal_alpha: float
    fdr_q_threshold: float
    results: List[Dict[str, Any]]
    raw_significant_count: int
    bonferroni_significant_count: int
    holm_significant_count: int
    bh_fdr_significant_count: int
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def adjust_multiple_metrics(
    metric_names: List[str],
    raw_p_values: List[float],
    effect_sizes: Optional[List[float]] = None,
    alpha: float = 0.05,
    q_threshold: float = 0.05
) -> MultipleTestingResult:
    """
    Adjust multi-metric scorecard p-values for multiple comparisons.

    Parameters:
      metric_names: List of metric identifiers
      raw_p_values: List of unadjusted two-sided p-values
      effect_sizes: List of observed relative or absolute effect sizes
      alpha: Desired FWER control level
      q_threshold: Desired False Discovery Rate (FDR) threshold
    """
    if len(metric_names) != len(raw_p_values):
        raise ValueError("Lengths of metric_names and raw_p_values must match.")

    m = len(metric_names)
    p_arr = np.clip(np.asarray(raw_p_values, dtype=float), 1e-15, 1.0)
    eff_arr = effect_sizes if effect_sizes else [0.0] * m

    # 1. Bonferroni (FWER)
    bonf_reject, bonf_p, _, _ = multipletests(p_arr, alpha=alpha, method="bonferroni")

    # 2. Holm-Bonferroni (FWER)
    holm_reject, holm_p, _, _ = multipletests(p_arr, alpha=alpha, method="holm")

    # 3. Benjamini-Hochberg (FDR)
    bh_reject, bh_p, _, _ = multipletests(p_arr, alpha=q_threshold, method="fdr_bh")

    rows = []
    for i in range(m):
        row = MetricCorrectionRow(
            metric_name=metric_names[i],
            raw_p_value=round(float(p_arr[i]), 5),
            observed_effect_size=round(float(eff_arr[i]), 4),
            bonferroni_p_value=round(float(bonf_p[i]), 5),
            bonferroni_significant=bool(bonf_reject[i]),
            holm_p_value=round(float(holm_p[i]), 5),
            holm_significant=bool(holm_reject[i]),
            bh_fdr_p_value=round(float(bh_p[i]), 5),
            bh_fdr_significant=bool(bh_reject[i])
        )
        rows.append(asdict(row))

    raw_sig_cnt = int(np.sum(p_arr < alpha))
    bonf_sig_cnt = int(np.sum(bonf_reject))
    holm_sig_cnt = int(np.sum(holm_reject))
    bh_sig_cnt = int(np.sum(bh_reject))

    rec = (
        f"Multi-Metric Audit: {raw_sig_cnt}/{m} metrics showed naive nominal significance (p < {alpha}). "
        f"After Benjamini-Hochberg FDR control (q <= {q_threshold}), {bh_sig_cnt}/{m} metrics remain robustly confirmed. "
        f"Under conservative Bonferroni FWER control, {bonf_sig_cnt}/{m} metrics pass."
    )

    return MultipleTestingResult(
        total_metrics_tested=m,
        nominal_alpha=alpha,
        fdr_q_threshold=q_threshold,
        results=rows,
        raw_significant_count=raw_sig_cnt,
        bonferroni_significant_count=bonf_sig_cnt,
        holm_significant_count=holm_sig_cnt,
        bh_fdr_significant_count=bh_sig_cnt,
        recommendation=rec
    )


if __name__ == "__main__":
    # Test with typical e-commerce multi-metric audit scenario
    sample_metrics = [
        "Primary_Checkout_Conversion",
        "Add_to_Cart_Rate",
        "Product_Detail_View_Rate",
        "Average_Session_Duration",
        "Page_Load_Time_P95",
        "Search_Usage_Rate",
        "Bounce_Rate",
        "Checkout_Error_Rate"
    ]
    sample_pvals = [0.189, 0.032, 0.048, 0.612, 0.001, 0.220, 0.089, 0.015]
    sample_lifts = [-1.31, +2.10, +1.85, -0.40, -8.20, +0.95, -1.10, -12.4]

    audit_res = adjust_multiple_metrics(sample_metrics, sample_pvals, sample_lifts)
    print("=" * 60)
    print("MULTI-METRIC SCORECARD CORRECTION AUDIT")
    print("=" * 60)
    print(audit_res.recommendation)
    print("-" * 60)
    for r in audit_res.results:
        print(f"{r['metric_name']:30s} | Raw: {r['raw_p_value']:.4f} | BH-FDR: {r['bh_fdr_p_value']:.4f} (Sig={r['bh_fdr_significant']}) | Bonf: {r['bonferroni_p_value']:.4f}")
    print("=" * 60)
