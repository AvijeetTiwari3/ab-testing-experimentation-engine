"""
Bayesian A/B Testing & Decision Engine
Implements Beta-Binomial Conjugate model, Monte Carlo posterior sampling,
Highest Density Interval (HDI), Expected Loss (Risk), and ROPE analysis.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
import numpy as np
from scipy.stats import beta


@dataclass
class BayesianResult:
    prior_alpha: float
    prior_beta: float
    posterior_control_alpha: float
    posterior_control_beta: float
    posterior_treatment_alpha: float
    posterior_treatment_beta: float
    control_posterior_mean: float
    treatment_posterior_mean: float
    prob_treatment_superior: float
    expected_absolute_lift: float
    expected_relative_lift_pct: float
    credible_interval_lower: float
    credible_interval_upper: float
    risk_choosing_treatment: float
    risk_choosing_control: float
    rope_probability: float
    rope_width: float
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def bayesian_ab_test(
    control_conversions: int,
    control_total: int,
    treatment_conversions: int,
    treatment_total: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    rope_width: float = 0.001,
    n_simulations: int = 100_000,
    random_seed: int = 42
) -> BayesianResult:
    """
    Execute Bayesian Beta-Binomial experimentation analysis.

    Parameters:
      control_conversions: Successes in control
      control_total: Trials in control
      treatment_conversions: Successes in treatment
      treatment_total: Trials in treatment
      prior_alpha: Prior alpha parameter (1.0 = Uniform uninformative prior)
      prior_beta: Prior beta parameter (1.0 = Uniform uninformative prior)
      rope_width: Margin for Region of Practical Equivalence (e.g. 0.001 = ±0.1% lift)
      n_simulations: Number of Monte Carlo samples drawn from posteriors
      random_seed: Seed for reproducible sampling
    """
    # 1. Closed-form Beta posterior parameter updates
    post_ctrl_a = prior_alpha + control_conversions
    post_ctrl_b = prior_beta + (control_total - control_conversions)

    post_trt_a = prior_alpha + treatment_conversions
    post_trt_b = prior_beta + (treatment_total - treatment_conversions)

    mean_ctrl = post_ctrl_a / (post_ctrl_a + post_ctrl_b)
    mean_trt = post_trt_a / (post_trt_a + post_trt_b)

    # 2. Monte Carlo sampling from analytical posteriors
    rng = np.random.default_rng(random_seed)
    samples_ctrl = rng.beta(post_ctrl_a, post_ctrl_b, size=n_simulations)
    samples_trt = rng.beta(post_trt_a, post_trt_b, size=n_simulations)

    # 3. Posterior Probability of Superiority
    prob_trt_superior = float(np.mean(samples_trt > samples_ctrl))

    # 4. Lift distribution and 95% Credible Interval (HDI approximation)
    lift_distribution = samples_trt - samples_ctrl
    exp_abs_lift = float(np.mean(lift_distribution))
    exp_rel_lift = float(exp_abs_lift / mean_ctrl * 100.0)

    ci_lower = float(np.percentile(lift_distribution, 2.5))
    ci_upper = float(np.percentile(lift_distribution, 97.5))

    # 5. Expected Loss (Bayesian Risk Analysis)
    # If we choose Treatment, loss is max(Control - Treatment, 0)
    risk_trt = float(np.mean(np.maximum(samples_ctrl - samples_trt, 0.0)))
    # If we choose Control, loss is max(Treatment - Control, 0)
    risk_ctrl = float(np.mean(np.maximum(samples_trt - samples_ctrl, 0.0)))

    # 6. Region of Practical Equivalence (ROPE)
    rope_prob = float(np.mean(np.abs(lift_distribution) < rope_width))

    # 7. Decision Policy Matrix
    if prob_trt_superior >= 0.95 and risk_trt < (rope_width / 2.0):
        recommendation = "DECISION: SHIP TREATMENT [P(Treatment > Control) >= 95% with negligible risk]"
    elif prob_trt_superior <= 0.05 or (mean_trt < mean_ctrl and prob_trt_superior < 0.20):
        recommendation = "DECISION: RETAIN CONTROL [Treatment has low probability of superiority and negative expected lift]"
    elif rope_prob >= 0.75:
        recommendation = "DECISION: PRACTICALLY EQUIVALENT [No meaningful business difference within ROPE window]"
    else:
        recommendation = "DECISION: INCONCLUSIVE [Collect more samples or evaluate downstream business metrics]"

    return BayesianResult(
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        posterior_control_alpha=float(post_ctrl_a),
        posterior_control_beta=float(post_ctrl_b),
        posterior_treatment_alpha=float(post_trt_a),
        posterior_treatment_beta=float(post_trt_b),
        control_posterior_mean=round(float(mean_ctrl), 5),
        treatment_posterior_mean=round(float(mean_trt), 5),
        prob_treatment_superior=round(prob_trt_superior, 4),
        expected_absolute_lift=round(exp_abs_lift, 5),
        expected_relative_lift_pct=round(exp_rel_lift, 2),
        credible_interval_lower=round(ci_lower, 5),
        credible_interval_upper=round(ci_upper, 5),
        risk_choosing_treatment=round(risk_trt, 6),
        risk_choosing_control=round(risk_ctrl, 6),
        rope_probability=round(rope_prob, 4),
        rope_width=rope_width,
        recommendation=recommendation
    )


if __name__ == "__main__":
    res = bayesian_ab_test(
        control_conversions=17489,
        control_total=145274,
        treatment_conversions=17264,
        treatment_total=145310,
        prior_alpha=1.0,
        prior_beta=1.0,
        rope_width=0.001
    )
    print("=" * 60)
    print("BAYESIAN A/B TEST EXPERIMENTATION RESULT")
    print("=" * 60)
    for k, v in res.to_dict().items():
        print(f"{k:32s}: {v}")
    print("=" * 60)
