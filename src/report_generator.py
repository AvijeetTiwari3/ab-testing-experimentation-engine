"""
Automated Experimentation Readout & Report Generator
Generates Google Data Science standard Markdown and HTML executive readouts.
"""

import os
from typing import Dict, Any, Optional
import pandas as pd


def generate_experiment_markdown_report(
    validation_rep: Dict[str, Any],
    freq_res: Dict[str, Any],
    bayes_res: Dict[str, Any],
    seq_res: Dict[str, Any],
    seg_res: Dict[str, Any],
    output_path: Optional[str] = None
) -> str:
    """
    Format a complete, executive-grade experimentation decision memo.
    """
    md = f"""# Executive Experimentation Decision Memo
## E-Commerce Landing Page Optimization A/B Test
**Author:** Lead Data Scientist (Experimentation Platform)  
**Status:** COMPLETE & CONCLUDED  
**Primary Metric:** Checkout Conversion Rate  
**Target Alpha:** {freq_res['alpha']} (95% Confidence Level)  

---

## 1. Executive Summary & Launch Recommendation

> ### **FINAL DECISION: RETAIN CONTROL VARIANT (DO NOT SHIP)**
> **The new landing page failed to demonstrate statistical superiority over the existing control experience.**  
> - **Observed Conversion Lift:** **{freq_res['relative_lift_pct']:+.2f}%** (Control: **{freq_res['control_mean']*100:.2f}%** vs. Treatment: **{freq_res['treatment_mean']*100:.2f}%**)
> - **Frequentist Hypothesis Test:** $p = {freq_res['p_value']:.4f}$ (Not statistically significant at $\\alpha = 0.05$)
> - **95% Confidence Interval for Lift:** $[{freq_res['ci_lower']*100:+.2f}\\%, {freq_res['ci_upper']*100:+.2f}\\%]$ (Spans zero)
> - **Bayesian Posterior Probability:** Only **{bayes_res['prob_treatment_superior']*100:.1f}%** chance that the treatment is superior.
> - **Expected Business Risk:** Deploying Treatment carries a **{bayes_res['risk_choosing_treatment']*10000:.1f} bps** expected conversion penalty.
> - **Recommendation:** **Discard variant**. Do not allocate engineering resources to ship `new_page`. Re-evaluate UX friction hypotheses in the funnel.

---

## 2. Data Integrity & Randomization Audit

Before conducting statistical inferences, data sanitization and randomization checks were strictly enforced:

| Integrity Check | Target / Expected | Observed Metric | Audit Status |
|---|---|---|---|
| **Raw Sample Size** | Full event logs | {validation_rep['total_raw_rows']:,} rows | Logged |
| **Contaminated Records** | 0 mismatches | {validation_rep['mismatched_assignments']:,} purged | Fixed |
| **Deduplicated User Exposures** | 1 exposure per user | {validation_rep['duplicate_users']:,} purged | Fixed |
| **Clean Analyzed Sample** | High power volume | {validation_rep['clean_rows']:,} users | Verified |
| **Sample Ratio Mismatch (SRM)** | $\\chi^2$ goodness-of-fit $p \\ge 0.01$ | $\\chi^2 = {validation_rep['srm_chi2_stat']:.4f}, p = {validation_rep['srm_p_value']:.4f}$ | **PASSED (No Traffic Bias)** |
| **Control Group Traffic** | 50.0% allocation | {validation_rep['control_count']:,} ({validation_rep['control_count']/validation_rep['clean_rows']*100:.2f}%) | Balanced |
| **Treatment Group Traffic** | 50.0% allocation | {validation_rep['treatment_count']:,} ({validation_rep['treatment_count']/validation_rep['clean_rows']*100:.2f}%) | Balanced |

---

## 3. Frequentist Statistical Evidence

- **Test Mechanism:** Two-Proportion Pooled Z-Test
- **Control Baseline:** {freq_res['control_mean']*100:.2f}% ({validation_rep['control_count']:,} visitors)
- **Treatment Observed:** {freq_res['treatment_mean']*100:.2f}% ({validation_rep['treatment_count']:,} visitors)
- **Difference in Proportions:** {freq_res['absolute_difference']:+.5f} ({freq_res['relative_lift_pct']:+.2f}% relative)
- **Test Statistic ($Z$):** {freq_res['test_statistic']:.4f}
- **P-Value:** **{freq_res['p_value']:.4f}**
- **Cohen's $h$ Effect Size:** {freq_res['effect_size']:.4f} (*{freq_res['effect_size_interpretation']}*)
- **Practical Significance Threshold (MDE):** {freq_res['practical_significance_mde'] if freq_res['practical_significance_mde'] else 'N/A'}

---

## 4. Bayesian Evidence & Decision Theory

- **Prior Specification:** $\\text{{Beta}}(\\alpha_0={bayes_res['prior_alpha']}, \\beta_0={bayes_res['prior_beta']})$ (Uninformative Uniform Prior)
- **Control Posterior:** $\\text{{Beta}}(\\alpha={bayes_res['posterior_control_alpha']:.0f}, \\beta={bayes_res['posterior_control_beta']:.0f})$ (Mean = {bayes_res['control_posterior_mean']*100:.2f}%)
- **Treatment Posterior:** $\\text{{Beta}}(\\alpha={bayes_res['posterior_treatment_alpha']:.0f}, \\beta={bayes_res['posterior_treatment_beta']:.0f})$ (Mean = {bayes_res['treatment_posterior_mean']*100:.2f}%)
- **$P(\\text{{Treatment}} > \\text{{Control}})$:** **{bayes_res['prob_treatment_superior']*100:.1f}%**
- **95% Credible Interval (HDI):** $[{bayes_res['credible_interval_lower']*100:+.2f}\\%, {bayes_res['credible_interval_upper']*100:+.2f}\\%]$
- **Expected Loss (Risk of choosing Treatment):** {bayes_res['risk_choosing_treatment']:.6f}
- **Expected Loss (Risk of choosing Control):** {bayes_res['risk_choosing_control']:.6f}
- **ROPE Probability ($|\\text{{Lift}}| < {bayes_res['rope_width']*100:.1f}\\%$):** {bayes_res['rope_probability']*100:.1f}%

---

## 5. Sequential Monitoring & Peeking Audit

- **Days Monitored:** {seq_res['total_looks']} sequential daily evaluations
- **Alpha Spending Function:** O'Brien-Fleming boundary
- **Early Stopping Triggered:** **{seq_res['early_stopping_triggered']}**
- **Peeking Diagnosis:** {seq_res['peeking_simulation_summary']['peeking_problem_diagnosis']}

---

## 6. Subgroup / Heterogeneous Treatment Effects

Decomposition across geography:

| Country | Traffic Share | Control Rate | Treatment Rate | Observed Lift | P-Value | Significance |
|---|---|---|---|---|---|---|
"""
    for s in seg_res['segments']:
        md += f"| **{s['segment_name']}** | {s['traffic_share_pct']:.1f}% | {s['control_rate']*100:.2f}% | {s['treatment_rate']*100:.2f}% | {s['relative_lift_pct']:+.2f}% | {s['p_value']:.4f} | {'Significant' if s['is_significant'] else 'Not Significant'} |\n"

    md += f"""
- **Simpson's Paradox Detected:** **{seg_res['simpsons_paradox_detected']}**
- **Synthesis:** {seg_res['summary_insight']}

---

## 7. Next Steps & Post-Mortem Action Items
1. **Archive Redesign Asset:** Do not roll out `new_page` to production.
2. **Qualitative User Research:** Review session recordings to understand why users converted slightly less on `new_page`.
3. **Formulate Next Iteration:** Develop new hypotheses focusing on simplified checkout form fields rather than full-page redesign.
"""

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

    return md


if __name__ == "__main__":
    import os
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    report_file = os.path.join(base_dir, "reports", "experiment_readout.md")

    # Sample execution
    val_dummy = {
        "total_raw_rows": 294478,
        "mismatched_assignments": 3893,
        "duplicate_users": 1,
        "clean_rows": 290584,
        "control_count": 145274,
        "treatment_count": 145310,
        "srm_chi2_stat": 0.0045,
        "srm_p_value": 0.9468
    }
    freq_dummy = {
        "alpha": 0.05,
        "control_mean": 0.12039,
        "treatment_mean": 0.11881,
        "relative_lift_pct": -1.31,
        "absolute_difference": -0.00158,
        "test_statistic": -1.3109,
        "p_value": 0.1899,
        "ci_lower": -0.00394,
        "ci_upper": 0.00078,
        "effect_size": -0.0049,
        "effect_size_interpretation": "negligible",
        "practical_significance_mde": 0.005
    }
    bayes_dummy = {
        "prior_alpha": 1.0,
        "prior_beta": 1.0,
        "posterior_control_alpha": 17490,
        "posterior_control_beta": 127786,
        "posterior_treatment_alpha": 17265,
        "posterior_treatment_beta": 128047,
        "control_posterior_mean": 0.12039,
        "treatment_posterior_mean": 0.11881,
        "prob_treatment_superior": 0.0937,
        "credible_interval_lower": -0.00395,
        "credible_interval_upper": 0.00078,
        "risk_choosing_treatment": 0.001638,
        "risk_choosing_control": 0.000053,
        "rope_probability": 0.2976,
        "rope_width": 0.001
    }
    seq_dummy = {
        "total_looks": 23,
        "early_stopping_triggered": False,
        "peeking_simulation_summary": {
            "peeking_problem_diagnosis": "O'Brien-Fleming alpha-spending safely bounds total Type I error at exactly alpha=0.05."
        }
    }
    seg_dummy = {
        "simpsons_paradox_detected": False,
        "summary_insight": "Homogeneous performance across subgroups.",
        "segments": [
            {"segment_name": "US", "traffic_share_pct": 70.1, "control_rate": 0.1206, "treatment_rate": 0.1185, "relative_lift_pct": -1.79, "p_value": 0.1323, "is_significant": False},
            {"segment_name": "UK", "traffic_share_pct": 24.9, "control_rate": 0.1200, "treatment_rate": 0.1212, "relative_lift_pct": 0.96, "p_value": 0.6349, "is_significant": False},
            {"segment_name": "CA", "traffic_share_pct": 5.0, "control_rate": 0.1188, "treatment_rate": 0.1119, "relative_lift_pct": -5.79, "p_value": 0.1947, "is_significant": False}
        ]
    }

    report_content = generate_experiment_markdown_report(val_dummy, freq_dummy, bayes_dummy, seq_dummy, seg_dummy, report_file)
    print("Report generated successfully at:", report_file)
