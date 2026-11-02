"""
Production Streamlit Analytics Dashboard
A/B Testing & Statistical Experimentation Platform (Google Data Scientist Standard)
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import beta, norm

# Ensure project modules are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.power_analysis import calculate_sample_size_proportions
from src.frequentist_engine import two_proportion_ztest
from src.bayesian_engine import bayesian_ab_test
from src.multiple_testing import adjust_multiple_metrics
from src.segment_analyzer import analyze_cohort_segments
from src.sequential_engine import run_sequential_experiment_analysis

st.set_page_config(
    page_title="A/B Testing Analytics Engine | Google DS Standard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #1a73e8;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_processed_experiment_data():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(base_dir, "data", "processed", "clean_experiment_data.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


df_clean = load_processed_experiment_data()

# Sidebar Navigation
st.sidebar.title("🔬 Experimentation Engine")
st.sidebar.caption("Google Data Scientist Portfolio standard")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation Modules",
    [
        "1. Executive Decision Memo",
        "2. Data Integrity & SRM Audit",
        "3. Pre-Experiment Power Sizing",
        "4. Frequentist Hypothesis Testing",
        "5. Bayesian Posterior Engine",
        "6. Sequential Testing & Peeking",
        "7. Multi-Metric Scorecard (FDR)",
        "8. Subgroup & Simpson's Paradox"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Dataset:** Real Kaggle E-Commerce A/B Test (294,478 raw rows, 290,584 verified users).\n"
    "**Stack:** Python, SciPy, FastAPI, Streamlit, Plotly, Render."
)


# Module 1: Executive Decision Memo
if menu == "1. Executive Decision Memo":
    st.title("📋 Executive Experimentation Decision Memo")
    st.subheader("Landing Page Checkout Redesign (290,584 Verified Users)")

    st.error("### 🛑 FINAL RECOMMENDATION: RETAIN CONTROL VARIANT (DO NOT SHIP)")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Control Conversion Rate", value="12.04%", delta=None)
    with col2:
        st.metric(label="Treatment Conversion Rate", value="11.88%", delta="-0.16% (Relative: -1.31%)", delta_color="inverse")
    with col3:
        st.metric(label="Two-Tailed P-Value", value="0.1899", delta="Not Significant (>0.05)", delta_color="off")
    with col4:
        st.metric(label="P(Treatment > Control)", value="9.37%", delta="Severe Downside Risk", delta_color="inverse")

    st.markdown("""
    #### 🔍 Key Takeaways for Leadership:
    1. **Zero Evidence of Lift:** The redesigned landing page did not improve checkout conversions. In fact, point estimates indicate a minor conversion degradation (**-1.31% relative lift**).
    2. **Hypothesis Rejected:** With $p = 0.1899$, the observed deviation is fully consistent with random noise under the Null Hypothesis.
    3. **Bayesian Risk Assessment:** Choosing the treatment variant carries a **31x higher risk of business loss** compared to staying with the proven control page.
    4. **Actionable Next Steps:** Deprecate `new_page` code branch; shift experimentation roadmap to micro-optimizations (frictionless payment buttons) rather than cosmetic overhauls.
    """)

    if df_clean is not None:
        st.markdown("---")
        st.markdown("#### Empirical Conversion Comparison")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Control (old_page)",
            x=["Baseline"],
            y=[0.12039],
            error_y=dict(type='data', array=[0.00168]),
            marker_color="#1a73e8",
            text=["12.04%"],
            textposition="auto"
        ))
        fig.add_trace(go.Bar(
            name="Treatment (new_page)",
            x=["Baseline"],
            y=[0.11881],
            error_y=dict(type='data', array=[0.00167]),
            marker_color="#ea4335",
            text=["11.88%"],
            textposition="auto"
        ))
        fig.update_layout(
            barmode='group',
            yaxis_title="Conversion Rate",
            yaxis=dict(range=[0.10, 0.13], tickformat=".1%"),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


# Module 2: Data Integrity & SRM Audit
elif menu == "2. Data Integrity & SRM Audit":
    st.title("🛡️ Data Integrity & Sample Ratio Mismatch (SRM) Audit")
    st.markdown("Industrial experiments require strict data hygiene before calculating inferences.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Raw Event Rows", "294,478")
    with col2:
        st.metric("Contaminated Mismatches", "3,893", delta="Purged (Group!=Page)", delta_color="inverse")
    with col3:
        st.metric("Unique Verified Users", "290,584")
    with col4:
        st.metric("SRM Chi-Square P-Value", "0.9468", delta="PASSED (p >= 0.01)", delta_color="normal")

    st.success("✅ **Sample Ratio Mismatch Test Passed!** Pearson $\\chi^2 = 0.0045$, $p = 0.9468$. Traffic was allocated in a mathematically pure 50.006% to 49.994% ratio.")

    if df_clean is not None:
        group_counts = df_clean["group"].value_counts().reset_index()
        group_counts.columns = ["Variant", "User Count"]
        fig = px.pie(group_counts, names="Variant", values="User Count", title="Traffic Allocation Distribution (50/50 Target)", color="Variant", color_discrete_map={"control": "#1a73e8", "treatment": "#34a853"})
        st.plotly_chart(fig, use_container_width=True)


# Module 3: Pre-Experiment Power Sizing
elif menu == "3. Pre-Experiment Power Sizing":
    st.title("📐 Pre-Experiment Statistical Power & Sample Sizing")
    st.markdown("Size experiments properly to prevent underpowered tests and avoid wasteful traffic allocation.")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        baseline_p = st.slider("Baseline Conversion Rate (%)", 1.0, 30.0, 12.0, 0.5) / 100.0
    with col_s2:
        mde_pct = st.slider("Minimum Detectable Effect (Absolute %)", 0.1, 3.0, 0.5, 0.1) / 100.0
    with col_s3:
        alpha_val = st.selectbox("Significance Level (Alpha)", [0.01, 0.05, 0.10], index=1)
    with col_s4:
        power_val = st.selectbox("Statistical Power (1 - Beta)", [0.80, 0.85, 0.90, 0.95], index=0)

    daily_traffic = st.number_input("Expected Total Daily Traffic across Variants", value=15000, step=1000)

    res = calculate_sample_size_proportions(
        baseline_rate=baseline_p,
        mde_absolute=mde_pct,
        alpha=alpha_val,
        power=power_val,
        daily_traffic=daily_traffic
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Required Sample / Variant", f"{res.required_sample_per_variant:,} users")
    with c2:
        st.metric("Total Required Sample", f"{res.total_required_sample:,} users")
    with c3:
        st.metric("Estimated Run Duration", f"{res.estimated_duration_days} days")

    if res.power_curve:
        st.markdown("#### Statistical Power Curve")
        curve_df = pd.DataFrame(res.power_curve)
        fig = px.line(
            curve_df,
            x="sample_size_per_variant",
            y="statistical_power",
            markers=True,
            title="Statistical Power vs. Sample Size per Variant",
            labels={"sample_size_per_variant": "Sample Size per Variant", "statistical_power": "Statistical Power (1 - Beta)"}
        )
        fig.add_hline(y=power_val, line_dash="dash", line_color="green", annotation_text=f"Target Power ({power_val})")
        st.plotly_chart(fig, use_container_width=True)


# Module 4: Frequentist Hypothesis Testing
elif menu == "4. Frequentist Hypothesis Testing":
    st.title("📊 Frequentist Hypothesis Testing Engine")
    st.markdown("Two-Proportion Z-Test with Wald Confidence Intervals and Cohen's effect size.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Control Configuration")
        c_n = st.number_input("Control Sample Size", value=145274, step=1000)
        c_conv = st.number_input("Control Conversions", value=17489, step=100)
    with col2:
        st.markdown("##### Treatment Configuration")
        t_n = st.number_input("Treatment Sample Size", value=145310, step=1000)
        t_conv = st.number_input("Treatment Conversions", value=17264, step=100)

    alpha_freq = st.selectbox("Significance Threshold (Alpha)", [0.01, 0.05, 0.10], index=1, key="freq_alpha")
    mde_freq = st.number_input("Practical Significance MDE (Absolute)", value=0.005, step=0.001, format="%.4f")

    if st.button("Run Frequentist Hypothesis Test", type="primary"):
        res = two_proportion_ztest(c_conv, c_n, t_conv, t_n, alpha=alpha_freq, mde=mde_freq)
        st.markdown("---")
        if res.is_statistically_significant:
            st.success(f"**Verdict:** {res.verdict}")
        else:
            st.warning(f"**Verdict:** {res.verdict}")

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Z-Statistic", res.test_statistic)
        with mc2:
            st.metric("P-Value", res.p_value)
        with mc3:
            st.metric("Observed Relative Lift", f"{res.relative_lift_pct:+0.2f}%")
        with mc4:
            st.metric("Cohen's h", f"{res.effect_size} ({res.effect_size_interpretation})")

        st.info(f"**95% Confidence Interval for Absolute Lift:** [{res.ci_lower:+.5f}, {res.ci_upper:+.5f}]")


# Module 5: Bayesian Posterior Engine
elif menu == "5. Bayesian Posterior Engine":
    st.title("🎲 Bayesian Posterior Inference & Decision Theory")
    st.markdown("Beta-Binomial conjugate updating, posterior Monte Carlo sampling, and Expected Loss (Risk).")

    col1, col2 = st.columns(2)
    with col1:
        p_alpha = st.number_input("Beta Prior Alpha", value=1.0, step=0.5)
        p_beta = st.number_input("Beta Prior Beta", value=1.0, step=0.5)
    with col2:
        rope = st.number_input("ROPE Margin (Region of Practical Equivalence)", value=0.001, format="%.4f")
        n_sims = st.selectbox("Monte Carlo Posterior Draws", [10000, 50000, 100000], index=1)

    if st.button("Compute Bayesian Posteriors", type="primary"):
        with st.spinner("Generating posterior distributions..."):
            res = bayesian_ab_test(
                control_conversions=17489,
                control_total=145274,
                treatment_conversions=17264,
                treatment_total=145310,
                prior_alpha=p_alpha,
                prior_beta=p_beta,
                rope_width=rope,
                n_simulations=n_sims
            )

        st.markdown("---")
        st.info(f"**{res.recommendation}**")

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.metric("P(Treatment > Control)", f"{res.prob_treatment_superior*100:.1f}%")
        with b2:
            st.metric("Expected Lift", f"{res.expected_relative_lift_pct:+0.2f}%")
        with b3:
            st.metric("Expected Loss (Trt Risk)", f"{res.risk_choosing_treatment:.6f}")
        with b4:
            st.metric("ROPE Probability", f"{res.rope_probability*100:.1f}%")

        # Plot Posteriors
        x_vals = np.linspace(0.115, 0.125, 500)
        pdf_ctrl = beta.pdf(x_vals, res.posterior_control_alpha, res.posterior_control_beta)
        pdf_trt = beta.pdf(x_vals, res.posterior_treatment_alpha, res.posterior_treatment_beta)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=pdf_ctrl, mode='lines', name='Control Posterior Beta', line=dict(color='#1a73e8', width=3)))
        fig.add_trace(go.Scatter(x=x_vals, y=pdf_trt, mode='lines', name='Treatment Posterior Beta', line=dict(color='#ea4335', width=3)))
        fig.update_layout(title="Posterior Probability Densities for Conversion Rates", xaxis_title="Conversion Rate", yaxis_title="Probability Density", height=450)
        st.plotly_chart(fig, use_container_width=True)


# Module 6: Sequential Testing & Peeking
elif menu == "6. Sequential Testing & Peeking":
    st.title("⏱️ Sequential Testing & Peeking Simulator")
    st.markdown("O'Brien-Fleming alpha-spending bounds total Type I error across continuous daily inspections.")

    if df_clean is not None:
        seq_res = run_sequential_experiment_analysis(df_clean, alpha=0.05)
        st.success(f"**Audit Status:** {seq_res.final_verdict}")

        hist_df = pd.DataFrame(seq_res.history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df["date"], y=hist_df["raw_p_value"], mode="lines+markers", name="Raw Daily P-Value", line=dict(color="#fbbc04", width=2)))
        fig.add_trace(go.Scatter(x=hist_df["date"], y=hist_df["obrien_fleming_alpha_boundary"], mode="lines", name="O'Brien-Fleming Alpha Boundary", line=dict(color="#ea4335", dash="dash", width=3)))
        fig.add_hline(y=0.05, line_dash="dot", line_color="gray", annotation_text="Naive Alpha (0.05)")

        fig.update_layout(title="Sequential P-Value Trajectory vs. O'Brien-Fleming Alpha Boundary", yaxis_title="Significance Threshold", xaxis_title="Experiment Date", height=450)
        st.plotly_chart(fig, use_container_width=True)


# Module 7: Multi-Metric Scorecard (FDR)
elif menu == "7. Multi-Metric Scorecard (FDR)":
    st.title("🎯 Multi-Metric Scorecard & FDR Control")
    st.markdown("Avoid false discoveries when evaluating secondary, guardrail, and business metrics.")

    sample_metrics = ["Primary_Conversion", "Add_to_Cart_Rate", "Session_Duration", "Page_Load_Time", "Checkout_Errors", "Bounce_Rate"]
    sample_pvals = [0.189, 0.032, 0.612, 0.001, 0.015, 0.089]
    sample_lifts = [-1.31, +2.10, -0.40, -8.20, -12.4, -1.10]

    mult_res = adjust_multiple_metrics(sample_metrics, sample_pvals, sample_lifts)
    st.info(mult_res.recommendation)
    st.dataframe(pd.DataFrame(mult_res.results), use_container_width=True)


# Module 8: Subgroup & Simpson's Paradox
elif menu == "8. Subgroup & Simpson's Paradox":
    st.title("🌍 Subgroup Analysis & Simpson's Paradox Detection")
    st.markdown("Heterogeneous Treatment Effect (HTE) evaluation across user geography.")

    if df_clean is not None:
        seg_res = analyze_cohort_segments(df_clean, segment_column="country")
        st.info(f"**Diagnosis:** {seg_res.summary_insight} (Simpson's Paradox Detected: {seg_res.simpsons_paradox_detected})")

        seg_df = pd.DataFrame(seg_res.segments)
        st.dataframe(seg_df, use_container_width=True)

        fig = px.bar(
            seg_df,
            x="segment_name",
            y="relative_lift_pct",
            color="relative_lift_pct",
            color_continuous_scale="RdBu",
            title="Observed Relative Lift by Geographic Cohort (%)",
            labels={"segment_name": "Country", "relative_lift_pct": "Relative Lift (%)"}
        )
        fig.add_hline(y=0, line_dash="solid", line_color="black")
        st.plotly_chart(fig, use_container_width=True)
