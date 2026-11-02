"""
Segment Analysis & Heterogeneous Treatment Effect (HTE) Engine
Performs subgroup decomposition, interaction testing, and Simpson's Paradox detection.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class SegmentMetricRow:
    segment_name: str
    control_count: int
    control_conversions: int
    control_rate: float
    treatment_count: int
    treatment_conversions: int
    treatment_rate: float
    absolute_lift: float
    relative_lift_pct: float
    z_statistic: float
    p_value: float
    is_significant: bool
    traffic_share_pct: float


@dataclass
class SegmentAnalysisResult:
    segmentation_variable: str
    segments: List[Dict[str, Any]]
    aggregate_control_rate: float
    aggregate_treatment_rate: float
    aggregate_relative_lift_pct: float
    simpsons_paradox_detected: bool
    heterogeneity_detected: bool
    summary_insight: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze_cohort_segments(
    df: pd.DataFrame,
    segment_column: str = "country",
    group_column: str = "group",
    outcome_column: str = "converted",
    alpha: float = 0.05
) -> SegmentAnalysisResult:
    """
    Deconstruct experiment effects across distinct user segments.
    Detects Heterogeneous Treatment Effects (HTE) and Simpson's Paradox.
    """
    if segment_column not in df.columns:
        raise ValueError(f"Segment column '{segment_column}' not found in dataframe.")

    total_n = len(df)
    unique_segments = sorted(list(df[segment_column].dropna().unique()))

    # Global aggregate metrics
    agg_ctrl_df = df[df[group_column] == "control"]
    agg_trt_df = df[df[group_column] == "treatment"]

    agg_ctrl_rate = float(agg_ctrl_df[outcome_column].mean())
    agg_trt_rate = float(agg_trt_df[outcome_column].mean())
    agg_rel_lift = (
        ((agg_trt_rate - agg_ctrl_rate) / agg_ctrl_rate * 100.0)
        if agg_ctrl_rate > 0 else 0.0
    )

    segment_rows = []
    lifts = []

    for seg in unique_segments:
        sub_df = df[df[segment_column] == seg]
        n_seg = len(sub_df)
        if n_seg < 30:
            continue

        ctrl_sub = sub_df[sub_df[group_column] == "control"][outcome_column]
        trt_sub = sub_df[sub_df[group_column] == "treatment"][outcome_column]

        n_c = len(ctrl_sub)
        n_t = len(trt_sub)
        conv_c = int(ctrl_sub.sum())
        conv_t = int(trt_sub.sum())

        if n_c == 0 or n_t == 0:
            continue

        r_c = conv_c / n_c
        r_t = conv_t / n_t
        diff = r_t - r_c
        rel = (diff / r_c * 100.0) if r_c > 0 else 0.0
        lifts.append(rel)

        # Standard two-proportion z-test per segment
        p_pool = (conv_c + conv_t) / (n_c + n_t)
        se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_c + 1.0 / n_t))
        if se > 0:
            z = diff / se
            p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
        else:
            z = 0.0
            p_val = 1.0

        row = SegmentMetricRow(
            segment_name=str(seg),
            control_count=n_c,
            control_conversions=conv_c,
            control_rate=round(r_c, 5),
            treatment_count=n_t,
            treatment_conversions=conv_t,
            treatment_rate=round(r_t, 5),
            absolute_lift=round(diff, 5),
            relative_lift_pct=round(rel, 2),
            z_statistic=round(float(z), 4),
            p_value=round(float(p_val), 5),
            is_significant=bool(p_val < alpha),
            traffic_share_pct=round((n_seg / total_n) * 100.0, 2)
        )
        segment_rows.append(asdict(row))

    # Simpson's Paradox Check:
    # Occurs if aggregate lift sign contradicts ALL segment lift signs
    pos_lifts = [l for l in lifts if l > 0]
    neg_lifts = [l for l in lifts if l < 0]

    simpsons_paradox = False
    if agg_rel_lift < 0 and len(pos_lifts) == len(lifts) and len(lifts) > 0:
        simpsons_paradox = True
    elif agg_rel_lift > 0 and len(neg_lifts) == len(lifts) and len(lifts) > 0:
        simpsons_paradox = True

    # Heterogeneity: do lifts vary wildly between segments?
    std_lifts = float(np.std(lifts)) if len(lifts) > 1 else 0.0
    heterogeneity = bool(std_lifts > 3.0)  # >3% variation across segments

    if simpsons_paradox:
        insight = "CRITICAL: Simpson's Paradox detected! Segment mix shift is distorting aggregate numbers."
    elif heterogeneity:
        insight = "HETEROGENEOUS EFFECTS: Treatment impact varies significantly across sub-cohorts. Consider personalized roll-out."
    else:
        insight = "HOMOGENEOUS PERFORMANCE: Subgroups perform uniformly with the aggregate null/consistent effect."

    return SegmentAnalysisResult(
        segmentation_variable=segment_column,
        segments=segment_rows,
        aggregate_control_rate=round(agg_ctrl_rate, 5),
        aggregate_treatment_rate=round(agg_trt_rate, 5),
        aggregate_relative_lift_pct=round(agg_rel_lift, 2),
        simpsons_paradox_detected=simpsons_paradox,
        heterogeneity_detected=heterogeneity,
        summary_insight=insight
    )


if __name__ == "__main__":
    import os
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    clean_csv = os.path.join(base_dir, "data", "processed", "clean_experiment_data.csv")
    df = pd.read_csv(clean_csv)

    seg_res = analyze_cohort_segments(df, segment_column="country")
    print("=" * 60)
    print("SUBGROUP / HETEROGENEOUS TREATMENT EFFECT AUDIT")
    print("=" * 60)
    print(f"Segment Variable: {seg_res.segmentation_variable}")
    print(f"Aggregate Lift: {seg_res.aggregate_relative_lift_pct}%")
    print(f"Simpson's Paradox Detected: {seg_res.simpsons_paradox_detected}")
    print(f"Insight: {seg_res.summary_insight}")
    print("-" * 60)
    for s in seg_res.segments:
        print(f"Country: {s['segment_name']:4s} | Share: {s['traffic_share_pct']:5.1f}% | Ctrl: {s['control_rate']:.4f} | Trt: {s['treatment_rate']:.4f} | Lift: {s['relative_lift_pct']:+5.2f}% | p-val: {s['p_value']:.4f}")
    print("=" * 60)
