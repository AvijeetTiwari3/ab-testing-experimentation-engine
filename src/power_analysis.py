"""
Pre-Experiment Power Analysis & Sample Size Computation Engine
Implements statistical sizing for proportion metrics and continuous metrics.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import numpy as np
from scipy.stats import norm


@dataclass
class PowerAnalysisResult:
    baseline_conversion_rate: float
    mde_absolute: float
    mde_relative: float
    alpha: float
    power: float
    required_sample_per_variant: int
    total_required_sample: int
    estimated_duration_days: float
    daily_traffic: int
    power_curve: Optional[Dict[str, List[float]]] = None

    def to_dict(self):
        return asdict(self)


def calculate_sample_size_proportions(
    baseline_rate: float,
    mde_absolute: float,
    alpha: float = 0.05,
    power: float = 0.80,
    daily_traffic: int = 10000,
    test_type: str = "two_sided"
) -> PowerAnalysisResult:
    """
    Compute required sample size per variant for a two-sample proportion test.

    Formula:
      n = [ (z_{1 - alpha/2} * sqrt(2 * p_bar * (1 - p_bar)) + z_{1 - beta} * sqrt(p1(1-p1) + p2(1-p2))) / (p2 - p1) ]^2

    Parameters:
      baseline_rate: Baseline conversion rate (e.g. 0.12 for 12%)
      mde_absolute: Minimum detectable effect as absolute difference (e.g. 0.01 for 1% lift)
      alpha: Type I error rate (default 0.05)
      power: Statistical power 1 - beta (default 0.80)
      daily_traffic: Estimated total daily traffic across all variants
      test_type: 'two_sided' or 'one_sided'
    """
    if not (0 < baseline_rate < 1):
        raise ValueError("Baseline rate must be between 0 and 1.")
    if mde_absolute == 0:
        raise ValueError("MDE cannot be zero.")

    p1 = baseline_rate
    p2 = baseline_rate + mde_absolute
    if not (0 < p2 < 1):
        raise ValueError(f"Target rate p2={p2} out of bounds (0, 1).")

    z_alpha = norm.ppf(1.0 - alpha / (2.0 if test_type == "two_sided" else 1.0))
    z_beta = norm.ppf(power)

    p_bar = (p1 + p2) / 2.0
    term1 = z_alpha * np.sqrt(2.0 * p_bar * (1.0 - p_bar))
    term2 = z_beta * np.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    
    n_per_variant = int(np.ceil(((term1 + term2) / abs(p2 - p1)) ** 2))
    total_n = n_per_variant * 2
    duration_days = round(total_n / max(daily_traffic, 1), 1)
    mde_rel = round((mde_absolute / baseline_rate) * 100.0, 2)

    # Generate power curve across various sample sizes
    sample_multipliers = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    sample_sizes = [int(n_per_variant * m) for m in sample_multipliers]
    computed_powers = []
    for n_s in sample_sizes:
        if n_s <= 0:
            computed_powers.append(0.0)
            continue
        se = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n_s)
        z_stat = abs(p2 - p1) / se
        # Power = Phi(z_stat - z_alpha)
        pw = float(norm.cdf(z_stat - z_alpha))
        computed_powers.append(round(pw, 4))

    power_curve = {
        "sample_size_per_variant": sample_sizes,
        "statistical_power": computed_powers
    }

    return PowerAnalysisResult(
        baseline_conversion_rate=baseline_rate,
        mde_absolute=mde_absolute,
        mde_relative=mde_rel,
        alpha=alpha,
        power=power,
        required_sample_per_variant=n_per_variant,
        total_required_sample=total_n,
        estimated_duration_days=duration_days,
        daily_traffic=daily_traffic,
        power_curve=power_curve
    )


def calculate_mde(
    baseline_rate: float,
    sample_size_per_variant: int,
    alpha: float = 0.05,
    power: float = 0.80,
    test_type: str = "two_sided"
) -> float:
    """
    Calculate the Minimum Detectable Effect (MDE) achievable given fixed sample size per variant.
    """
    z_alpha = norm.ppf(1.0 - alpha / (2.0 if test_type == "two_sided" else 1.0))
    z_beta = norm.ppf(power)
    
    p = baseline_rate
    mde = (z_alpha + z_beta) * np.sqrt(2.0 * p * (1.0 - p) / sample_size_per_variant)
    return float(mde)


if __name__ == "__main__":
    res = calculate_sample_size_proportions(
        baseline_rate=0.12,
        mde_absolute=0.005,  # 0.5% absolute lift (e.g. from 12% to 12.5%)
        alpha=0.05,
        power=0.80,
        daily_traffic=15000
    )
    print("=" * 60)
    print("A/B TEST POWER ANALYSIS & SAMPLE SIZING")
    print("=" * 60)
    for k, v in res.to_dict().items():
        if k != "power_curve":
            print(f"{k:32s}: {v}")
    print("=" * 60)
    print("Power Curve Sample:", res.power_curve)
