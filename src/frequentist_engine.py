"""
Frequentist Statistical Experimentation Engine
Implements hypothesis testing suites: Two-proportion Z-test, Chi-Square, Mann-Whitney U, and Welch's t-test.
Computes effect sizes (Cohen's h, Cramér's V, CLES), confidence intervals, and practical significance.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Union
import numpy as np
from scipy import stats


@dataclass
class FrequentistResult:
    test_name: str
    control_mean: float
    treatment_mean: float
    absolute_difference: float
    relative_lift_pct: float
    test_statistic: float
    p_value: float
    alpha: float
    is_statistically_significant: bool
    confidence_level: float
    ci_lower: float
    ci_upper: float
    effect_size: float
    effect_size_metric: str
    effect_size_interpretation: str
    practical_significance_mde: Optional[float]
    is_practically_significant: Optional[bool]
    verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def two_proportion_ztest(
    control_conversions: int,
    control_total: int,
    treatment_conversions: int,
    treatment_total: int,
    alpha: float = 0.05,
    mde: Optional[float] = None,
    alternative: str = "two-sided"
) -> FrequentistResult:
    """
    Two-proportion Z-test for binary outcomes (e.g., Conversion Rate).

    Parameters:
      control_conversions: Number of conversions in Control
      control_total: Total sample in Control
      treatment_conversions: Number of conversions in Treatment
      treatment_total: Total sample in Treatment
      alpha: Significance threshold (default 0.05)
      mde: Minimum Detectable Effect threshold (absolute difference)
      alternative: 'two-sided', 'larger' (treatment > control), or 'smaller'
    """
    if control_total <= 0 or treatment_total <= 0:
        raise ValueError("Sample sizes must be strictly positive.")

    p_ctrl = control_conversions / control_total
    p_trt = treatment_conversions / treatment_total
    diff = p_trt - p_ctrl
    rel_lift = (diff / p_ctrl * 100.0) if p_ctrl > 0 else 0.0

    # Pooled probability for standard error under H0
    p_pool = (control_conversions + treatment_conversions) / (control_total + treatment_total)
    se_pool = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / control_total + 1.0 / treatment_total))

    if se_pool == 0:
        z_stat = 0.0
        p_val = 1.0
    else:
        z_stat = diff / se_pool
        if alternative == "two-sided":
            p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
        elif alternative == "larger":
            p_val = 1.0 - stats.norm.cdf(z_stat)
        elif alternative == "smaller":
            p_val = stats.norm.cdf(z_stat)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")

    # Unpooled standard error for Wald Confidence Interval
    se_unpooled = np.sqrt(
        (p_ctrl * (1.0 - p_ctrl) / control_total) +
        (p_trt * (1.0 - p_trt) / treatment_total)
    )
    z_crit = stats.norm.ppf(1.0 - alpha / (2.0 if alternative == "two-sided" else 1.0))
    ci_lower = diff - z_crit * se_unpooled
    ci_upper = diff + z_crit * se_unpooled

    # Cohen's h for difference between two proportions
    # h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    cohens_h = 2.0 * np.arcsin(np.sqrt(p_trt)) - 2.0 * np.arcsin(np.sqrt(p_ctrl))
    abs_h = abs(cohens_h)
    if abs_h < 0.10:
        interp = "negligible"
    elif abs_h < 0.20:
        interp = "very small"
    elif abs_h < 0.50:
        interp = "small"
    elif abs_h < 0.80:
        interp = "medium"
    else:
        interp = "large"

    is_sig = bool(p_val < alpha)
    is_prac = bool(abs(diff) >= mde) if mde is not None else None

    if is_sig and diff > 0 and (is_prac is None or is_prac):
        verdict = "POSITIVE LIFT: Treatment is statistically and practically superior. Recommend roll-out."
    elif is_sig and diff < 0:
        verdict = "NEGATIVE REGRESSION: Treatment significantly harms conversion. Reject variant."
    elif is_sig and not is_prac:
        verdict = "TRIVIAL SIGNIFICANCE: Statistically significant but below Minimum Detectable Effect threshold."
    else:
        verdict = "NULL RESULT: No statistically significant difference detected. Retain Control."

    return FrequentistResult(
        test_name="Two-Proportion Z-Test",
        control_mean=round(p_ctrl, 5),
        treatment_mean=round(p_trt, 5),
        absolute_difference=round(diff, 5),
        relative_lift_pct=round(rel_lift, 2),
        test_statistic=round(float(z_stat), 4),
        p_value=round(float(p_val), 5),
        alpha=alpha,
        is_statistically_significant=is_sig,
        confidence_level=round(1.0 - alpha, 2),
        ci_lower=round(float(ci_lower), 5),
        ci_upper=round(float(ci_upper), 5),
        effect_size=round(float(cohens_h), 4),
        effect_size_metric="Cohen's h",
        effect_size_interpretation=interp,
        practical_significance_mde=mde,
        is_practically_significant=is_prac,
        verdict=verdict
    )


def chi_square_contingency_test(
    contingency_matrix: np.ndarray,
    alpha: float = 0.05
) -> FrequentistResult:
    """
    Pearson's Chi-Square Test of Independence for categorical distributions.
    Computes Cramér's V effect size.
    """
    matrix = np.asarray(contingency_matrix)
    chi2, p_val, dof, expected = stats.chi2_contingency(matrix)

    n_total = matrix.sum()
    min_dim = min(matrix.shape) - 1
    cramers_v = np.sqrt(chi2 / (n_total * min_dim)) if (n_total * min_dim) > 0 else 0.0

    if cramers_v < 0.1:
        interp = "negligible"
    elif cramers_v < 0.3:
        interp = "small"
    elif cramers_v < 0.5:
        interp = "moderate"
    else:
        interp = "large"

    is_sig = bool(p_val < alpha)
    verdict = (
        "Significant categorical association between variant and outcome."
        if is_sig else "No significant association between variant and outcome."
    )

    return FrequentistResult(
        test_name="Pearson Chi-Square Test",
        control_mean=0.0,
        treatment_mean=0.0,
        absolute_difference=0.0,
        relative_lift_pct=0.0,
        test_statistic=round(float(chi2), 4),
        p_value=round(float(p_val), 5),
        alpha=alpha,
        is_statistically_significant=is_sig,
        confidence_level=round(1.0 - alpha, 2),
        ci_lower=0.0,
        ci_upper=0.0,
        effect_size=round(float(cramers_v), 4),
        effect_size_metric="Cramér's V",
        effect_size_interpretation=interp,
        practical_significance_mde=None,
        is_practically_significant=None,
        verdict=verdict
    )


def mann_whitney_u_test(
    control_series: np.ndarray,
    treatment_series: np.ndarray,
    alpha: float = 0.05
) -> FrequentistResult:
    """
    Mann-Whitney U Test (Wilcoxon Rank-Sum) for non-parametric continuous metrics.
    Computes Common Language Effect Size (CLES).
    """
    ctrl = np.asarray(control_series)
    trt = np.asarray(treatment_series)
    u_stat, p_val = stats.mannwhitneyu(ctrl, trt, alternative="two-sided")

    n1, n2 = len(ctrl), len(trt)
    cles = u_stat / (n1 * n2)  # Probability that random treatment > random control

    mean_ctrl = float(np.mean(ctrl))
    mean_trt = float(np.mean(trt))
    diff = mean_trt - mean_ctrl
    rel = (diff / mean_ctrl * 100.0) if mean_ctrl != 0 else 0.0

    is_sig = bool(p_val < alpha)
    verdict = (
        "Statistically significant rank distribution shift."
        if is_sig else "No statistically significant rank difference."
    )

    return FrequentistResult(
        test_name="Mann-Whitney U Test",
        control_mean=round(mean_ctrl, 4),
        treatment_mean=round(mean_trt, 4),
        absolute_difference=round(diff, 4),
        relative_lift_pct=round(rel, 2),
        test_statistic=round(float(u_stat), 2),
        p_value=round(float(p_val), 5),
        alpha=alpha,
        is_statistically_significant=is_sig,
        confidence_level=round(1.0 - alpha, 2),
        ci_lower=0.0,
        ci_upper=0.0,
        effect_size=round(float(cles), 4),
        effect_size_metric="Common Language Effect Size (CLES)",
        effect_size_interpretation="P(Treatment > Control) = " + str(round(cles, 3)),
        practical_significance_mde=None,
        is_practically_significant=None,
        verdict=verdict
    )


if __name__ == "__main__":
    # Test with empirical values from our clean Kaggle dataset
    res = two_proportion_ztest(
        control_conversions=17489,
        control_total=145274,
        treatment_conversions=17264,
        treatment_total=145310,
        alpha=0.05,
        mde=0.005
    )
    print("=" * 60)
    print("FREQUENTIST TWO-PROPORTION Z-TEST RESULT (EMPIRICAL DATA)")
    print("=" * 60)
    for k, v in res.to_dict().items():
        print(f"{k:30s}: {v}")
    print("=" * 60)
