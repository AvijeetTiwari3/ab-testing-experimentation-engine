"""Statistical Analysis Router"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi import APIRouter, HTTPException
import pandas as pd
from api.schemas import FrequentistTestRequest, BayesianTestRequest, MultipleTestingRequest
from src.frequentist_engine import two_proportion_ztest
from src.bayesian_engine import bayesian_ab_test
from src.multiple_testing import adjust_multiple_metrics
from src.segment_analyzer import analyze_cohort_segments
from src.sequential_engine import run_sequential_experiment_analysis

router = APIRouter()

# Cached dataset path
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "clean_experiment_data.csv"))


@router.post("/frequentist", summary="Run Frequentist Two-Proportion Z-Test")
def frequentist_endpoint(req: FrequentistTestRequest):
    try:
        res = two_proportion_ztest(
            control_conversions=req.control_conversions,
            control_total=req.control_total,
            treatment_conversions=req.treatment_conversions,
            treatment_total=req.treatment_total,
            alpha=req.alpha,
            mde=req.mde,
            alternative=req.alternative
        )
        return res.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/bayesian", summary="Run Bayesian Beta-Binomial Test")
def bayesian_endpoint(req: BayesianTestRequest):
    try:
        res = bayesian_ab_test(
            control_conversions=req.control_conversions,
            control_total=req.control_total,
            treatment_conversions=req.treatment_conversions,
            treatment_total=req.treatment_total,
            prior_alpha=req.prior_alpha,
            prior_beta=req.prior_beta,
            rope_width=req.rope_width,
            n_simulations=req.n_simulations
        )
        return res.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/multiple-testing", summary="Apply Multiple Testing Corrections (Bonferroni, Holm, BH-FDR)")
def multiple_testing_endpoint(req: MultipleTestingRequest):
    try:
        res = adjust_multiple_metrics(
            metric_names=req.metric_names,
            raw_p_values=req.raw_p_values,
            effect_sizes=req.effect_sizes,
            alpha=req.alpha,
            q_threshold=req.q_threshold
        )
        return res.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/empirical-readout", summary="Full Executive Readout on Real Kaggle Dataset (290,584 users)")
def empirical_readout_endpoint():
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="Processed dataset not found.")

    df = pd.read_csv(DATA_PATH)

    ctrl_sub = df[df["group"] == "control"]
    trt_sub = df[df["group"] == "treatment"]

    n_ctrl = len(ctrl_sub)
    n_trt = len(trt_sub)
    conv_ctrl = int(ctrl_sub["converted"].sum())
    conv_trt = int(trt_sub["converted"].sum())

    freq_res = two_proportion_ztest(conv_ctrl, n_ctrl, conv_trt, n_trt, alpha=0.05, mde=0.005)
    bayes_res = bayesian_ab_test(conv_ctrl, n_ctrl, conv_trt, n_trt, rope_width=0.001, n_simulations=50000)
    seg_res = analyze_cohort_segments(df, segment_column="country")

    return {
        "experiment_name": "E-Commerce Landing Page A/B Experiment",
        "sample_size": {
            "total_users": len(df),
            "control_users": n_ctrl,
            "treatment_users": n_trt
        },
        "conversion_rates": {
            "control": round(conv_ctrl / n_ctrl, 5),
            "treatment": round(conv_trt / n_trt, 5),
            "relative_lift_pct": freq_res.relative_lift_pct
        },
        "frequentist_evidence": freq_res.to_dict(),
        "bayesian_evidence": bayes_res.to_dict(),
        "geographic_segments": seg_res.to_dict(),
        "final_recommendation": "RETAIN CONTROL (Do not deploy treatment variant to production)"
    }
