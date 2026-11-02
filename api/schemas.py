"""
Pydantic V2 Schemas for A/B Testing Engine REST API
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PowerAnalysisRequest(BaseModel):
    baseline_conversion_rate: float = Field(..., ge=0.0001, le=0.9999, description="Baseline conversion rate (e.g. 0.12)", json_schema_extra={"example": 0.12})
    mde_absolute: float = Field(..., ge=0.0001, le=0.50, description="Minimum Detectable Effect absolute difference (e.g. 0.005)", json_schema_extra={"example": 0.005})
    alpha: float = Field(default=0.05, ge=0.001, le=0.20, description="Significance level alpha", json_schema_extra={"example": 0.05})
    power: float = Field(default=0.80, ge=0.50, le=0.999, description="Statistical power 1 - beta", json_schema_extra={"example": 0.80})
    daily_traffic: int = Field(default=15000, ge=1, description="Estimated total daily active users across variants", json_schema_extra={"example": 15000})
    test_type: str = Field(default="two_sided", description="two_sided or one_sided", json_schema_extra={"example": "two_sided"})


class MDERequest(BaseModel):
    baseline_conversion_rate: float = Field(..., ge=0.0001, le=0.9999, json_schema_extra={"example": 0.12})
    sample_size_per_variant: int = Field(..., ge=10, json_schema_extra={"example": 50000})
    alpha: float = Field(default=0.05, json_schema_extra={"example": 0.05})
    power: float = Field(default=0.80, json_schema_extra={"example": 0.80})


class FrequentistTestRequest(BaseModel):
    control_conversions: int = Field(..., ge=0, json_schema_extra={"example": 17489})
    control_total: int = Field(..., ge=1, json_schema_extra={"example": 145274})
    treatment_conversions: int = Field(..., ge=0, json_schema_extra={"example": 17264})
    treatment_total: int = Field(..., ge=1, json_schema_extra={"example": 145310})
    alpha: float = Field(default=0.05, ge=0.001, le=0.20, json_schema_extra={"example": 0.05})
    mde: Optional[float] = Field(default=0.005, description="Practical significance MDE", json_schema_extra={"example": 0.005})
    alternative: str = Field(default="two-sided", json_schema_extra={"example": "two-sided"})


class BayesianTestRequest(BaseModel):
    control_conversions: int = Field(..., ge=0, json_schema_extra={"example": 17489})
    control_total: int = Field(..., ge=1, json_schema_extra={"example": 145274})
    treatment_conversions: int = Field(..., ge=0, json_schema_extra={"example": 17264})
    treatment_total: int = Field(..., ge=1, json_schema_extra={"example": 145310})
    prior_alpha: float = Field(default=1.0, ge=0.01, json_schema_extra={"example": 1.0})
    prior_beta: float = Field(default=1.0, ge=0.01, json_schema_extra={"example": 1.0})
    rope_width: float = Field(default=0.001, ge=0.00001, json_schema_extra={"example": 0.001})
    n_simulations: int = Field(default=50000, ge=1000, le=500000, json_schema_extra={"example": 50000})


class MultipleTestingRequest(BaseModel):
    metric_names: List[str] = Field(..., min_length=2, json_schema_extra={"example": ["Conversion_Rate", "Add_To_Cart", "Bounce_Rate"]})
    raw_p_values: List[float] = Field(..., min_length=2, json_schema_extra={"example": [0.189, 0.032, 0.048]})
    effect_sizes: Optional[List[float]] = Field(default=None, json_schema_extra={"example": [-1.31, 2.10, -0.40]})
    alpha: float = Field(default=0.05, json_schema_extra={"example": 0.05})
    q_threshold: float = Field(default=0.05, json_schema_extra={"example": 0.05})
