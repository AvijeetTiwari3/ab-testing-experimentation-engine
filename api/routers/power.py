"""Power Analysis and Sample Size Calculator router"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi import APIRouter, HTTPException
from api.schemas import PowerAnalysisRequest, MDERequest
from src.power_analysis import calculate_sample_size_proportions, calculate_mde

router = APIRouter()


@router.post("/sample-size", summary="Compute Sample Size & Power Curve")
def compute_sample_size_endpoint(req: PowerAnalysisRequest):
    try:
        res = calculate_sample_size_proportions(
            baseline_rate=req.baseline_conversion_rate,
            mde_absolute=req.mde_absolute,
            alpha=req.alpha,
            power=req.power,
            daily_traffic=req.daily_traffic,
            test_type=req.test_type
        )
        return res.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/mde", summary="Compute Minimum Detectable Effect given Sample Size")
def compute_mde_endpoint(req: MDERequest):
    try:
        mde = calculate_mde(
            baseline_rate=req.baseline_conversion_rate,
            sample_size_per_variant=req.sample_size_per_variant,
            alpha=req.alpha,
            power=req.power
        )
        return {
            "baseline_conversion_rate": req.baseline_conversion_rate,
            "sample_size_per_variant": req.sample_size_per_variant,
            "mde_absolute": round(mde, 5),
            "mde_relative_pct": round((mde / req.baseline_conversion_rate) * 100.0, 2),
            "alpha": req.alpha,
            "power": req.power
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
