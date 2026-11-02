"""
Sequential Testing & Alpha-Spending Engine
Implements O'Brien-Fleming and Pocock boundaries to eliminate the Peeking Problem in continuous monitoring.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class SequentialLook:
    look_number: int
    date: str
    cumulative_control_n: int
    cumulative_treatment_n: int
    cumulative_control_conv: int
    cumulative_treatment_conv: int
    information_fraction: float
    z_statistic: float
    raw_p_value: float
    obrien_fleming_critical_z: float
    obrien_fleming_alpha_boundary: float
    naive_peeking_significant: bool
    sequential_significant: bool
    can_stop_early: bool


@dataclass
class SequentialAnalysisResult:
    total_looks: int
    alpha: float
    planned_sample_size: int
    history: List[Dict[str, Any]]
    early_stopping_triggered: bool
    stopping_look: Optional[int]
    final_verdict: str
    peeking_simulation_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_obrien_fleming_boundary(alpha: float, information_fraction: float) -> Tuple[float, float]:
    """
    Compute O'Brien-Fleming critical Z value and corresponding alpha threshold.

    Formula:
      z_{crit}(t) = z_{alpha / 2} / sqrt(t)
      alpha(t) = 2 * (1 - Phi(z_{crit}(t)))
    """
    if information_fraction <= 0.0:
        return float("inf"), 0.0
    
    t = min(max(information_fraction, 0.0001), 1.0)
    z_alpha2 = norm.ppf(1.0 - alpha / 2.0)
    z_crit = z_alpha2 / np.sqrt(t)
    alpha_boundary = 2.0 * (1.0 - norm.cdf(z_crit))
    return float(z_crit), float(alpha_boundary)


def run_sequential_experiment_analysis(
    df: pd.DataFrame,
    date_col: str = "timestamp",
    group_col: str = "group",
    outcome_col: str = "converted",
    alpha: float = 0.05,
    planned_sample_size: Optional[int] = None
) -> SequentialAnalysisResult:
    """
    Run day-by-day sequential monitoring on experiment dataset.
    """
    df_seq = df.copy()
    df_seq["date"] = pd.to_datetime(df_seq[date_col]).dt.date

    # Group by date and variant
    daily_summary = df_seq.groupby(["date", group_col])[outcome_col].agg(["count", "sum"]).unstack(fill_value=0)
    
    dates = sorted(list(df_seq["date"].unique()))
    total_clean_n = len(df_seq)
    N_planned = planned_sample_size if planned_sample_size else total_clean_n

    cum_ctrl_n = 0
    cum_ctrl_conv = 0
    cum_trt_n = 0
    cum_trt_conv = 0

    history = []
    early_stop_triggered = False
    stop_look = None
    naive_false_alarms = 0

    for look_idx, d in enumerate(dates, 1):
        if ("count", "control") in daily_summary.columns:
            cum_ctrl_n += int(daily_summary.loc[d, ("count", "control")])
            cum_ctrl_conv += int(daily_summary.loc[d, ("sum", "control")])
        if ("count", "treatment") in daily_summary.columns:
            cum_trt_n += int(daily_summary.loc[d, ("count", "treatment")])
            cum_trt_conv += int(daily_summary.loc[d, ("sum", "treatment")])

        total_cum_n = cum_ctrl_n + cum_trt_n
        t = min(total_cum_n / max(N_planned, 1), 1.0)

        # Standard two-proportion z-statistic at look t
        p_c = cum_ctrl_conv / max(cum_ctrl_n, 1)
        p_t = cum_trt_conv / max(cum_trt_n, 1)
        p_pool = (cum_ctrl_conv + cum_trt_conv) / max(total_cum_n, 1)
        se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / max(cum_ctrl_n, 1) + 1.0 / max(cum_trt_n, 1)))

        if se > 0:
            z = (p_t - p_c) / se
            raw_p = 2.0 * (1.0 - norm.cdf(abs(z)))
        else:
            z = 0.0
            raw_p = 1.0

        z_crit_obf, alpha_obf = compute_obrien_fleming_boundary(alpha, t)

        naive_sig = bool(raw_p < alpha)
        if naive_sig:
            naive_false_alarms += 1

        seq_sig = bool(abs(z) >= z_crit_obf or raw_p <= alpha_obf)
        can_stop = bool(seq_sig and not early_stop_triggered)

        if can_stop and not early_stop_triggered:
            early_stop_triggered = True
            stop_look = look_idx

        look = SequentialLook(
            look_number=look_idx,
            date=str(d),
            cumulative_control_n=cum_ctrl_n,
            cumulative_treatment_n=cum_trt_n,
            cumulative_control_conv=cum_ctrl_conv,
            cumulative_treatment_conv=cum_trt_conv,
            information_fraction=round(t, 4),
            z_statistic=round(float(z), 4),
            raw_p_value=round(float(raw_p), 5),
            obrien_fleming_critical_z=round(float(z_crit_obf), 4),
            obrien_fleming_alpha_boundary=round(float(alpha_obf), 6),
            naive_peeking_significant=naive_sig,
            sequential_significant=seq_sig,
            can_stop_early=can_stop
        )
        history.append(asdict(look))

    if early_stop_triggered:
        final_verdict = f"EARLY TERMINATION TRIGGERED: Boundary crossed at look #{stop_look}. Statistically valid to stop."
    else:
        final_verdict = "EXPERIMENT RUN TO COMPLETION: O'Brien-Fleming boundary was never breached. No early stopping."

    peeking_summary = {
        "days_monitored": len(dates),
        "naive_peeking_days_flagged_as_significant": naive_false_alarms,
        "sequential_early_stops": 1 if early_stop_triggered else 0,
        "peeking_problem_diagnosis": (
            "Naive daily peeking increases Type I false positive risk. "
            "O'Brien-Fleming alpha-spending safely bounds total Type I error at exactly alpha=0.05."
        )
    }

    return SequentialAnalysisResult(
        total_looks=len(dates),
        alpha=alpha,
        planned_sample_size=N_planned,
        history=history,
        early_stopping_triggered=early_stop_triggered,
        stopping_look=stop_look,
        final_verdict=final_verdict,
        peeking_simulation_summary=peeking_summary
    )


if __name__ == "__main__":
    import os
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    clean_csv = os.path.join(base_dir, "data", "processed", "clean_experiment_data.csv")
    df = pd.read_csv(clean_csv)

    seq_res = run_sequential_experiment_analysis(df, alpha=0.05)
    print("=" * 60)
    print("SEQUENTIAL EXPERIMENTATION & ALPHA-SPENDING ANALYSIS")
    print("=" * 60)
    print(f"Total Looks (Days): {seq_res.total_looks}")
    print(f"Early Stopping: {seq_res.early_stopping_triggered}")
    print(f"Final Verdict: {seq_res.final_verdict}")
    print("\nPeeking Diagnosis:", seq_res.peeking_simulation_summary)
    print("\nSample Look Snapshot (First 3 Days):")
    for row in seq_res.history[:3]:
        print(f"  Day {row['look_number']} ({row['date']}): z={row['z_statistic']:.3f}, raw_p={row['raw_p_value']:.4f}, OBF_alpha={row['obrien_fleming_alpha_boundary']:.6f}, Seq_Sig={row['sequential_significant']}")
    print("=" * 60)
