"""
Data Validation & Integrity Engine
Validates experiment data against industrial experimentation standards (SRM, assignment consistency, deduplication).
"""

import os
from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from scipy.stats import chisquare


@dataclass
class ValidationReport:
    total_raw_rows: int
    mismatched_assignments: int
    duplicate_users: int
    clean_rows: int
    control_count: int
    treatment_count: int
    control_conversion_rate: float
    treatment_conversion_rate: float
    srm_chi2_stat: float
    srm_p_value: float
    srm_passed: bool
    countries_matched: int
    warnings: List[str]

    def to_dict(self):
        return asdict(self)


def validate_and_clean_dataset(
    ab_data_path: str,
    countries_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> Tuple[pd.DataFrame, ValidationReport]:
    """
    Validate, sanitize, and enrich experiment dataset.

    Steps performed:
    1. Verify columns and non-empty schema.
    2. Identify and purge contaminated records where group != landing_page mapping.
    3. Remove duplicate user exposures (retaining first interaction to avoid novelty bias).
    4. Perform Sample Ratio Mismatch (SRM) chi-square goodness-of-fit test.
    5. Optionally join demographic/country segmentation attributes.
    6. Persist clean dataset if output_path is provided.
    """
    warnings = []

    if not os.path.exists(ab_data_path):
        raise FileNotFoundError(f"Experiment dataset not found at: {ab_data_path}")

    df = pd.read_csv(ab_data_path)
    total_raw_rows = len(df)

    required_cols = {"user_id", "timestamp", "group", "landing_page", "converted"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing mandatory experiment columns: {missing}")

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Step 1: Detect and remove mismatched assignments
    # Control should see old_page, Treatment should see new_page
    valid_mask = (
        ((df["group"] == "control") & (df["landing_page"] == "old_page")) |
        ((df["group"] == "treatment") & (df["landing_page"] == "new_page"))
    )
    mismatched_count = int((~valid_mask).sum())
    if mismatched_count > 0:
        warnings.append(
            f"Purged {mismatched_count} contaminated records where group assignment contradicted landing page variant."
        )
    df_clean = df[valid_mask].copy()

    # Step 2: Sort chronologically and deduplicate users
    df_clean = df_clean.sort_values("timestamp")
    duplicate_users = int(df_clean.duplicated(subset=["user_id"]).sum())
    if duplicate_users > 0:
        warnings.append(
            f"Detected and dropped {duplicate_users} duplicate user exposures, keeping first interaction timestamp."
        )
    df_clean = df_clean.drop_duplicates(subset=["user_id"], keep="first").reset_index(drop=True)

    # Step 3: Sample Ratio Mismatch (SRM) Test
    # Tests whether traffic assignment deviates significantly from expected 50:50 allocation
    observed_counts = df_clean["group"].value_counts()
    n_ctrl = int(observed_counts.get("control", 0))
    n_trt = int(observed_counts.get("treatment", 0))
    total_clean = len(df_clean)

    expected = [total_clean / 2.0, total_clean / 2.0]
    observed = [n_ctrl, n_trt]
    chi2_stat, srm_p_value = chisquare(observed, expected)
    srm_passed = bool(srm_p_value >= 0.01)

    if not srm_passed:
        warnings.append(
            f"CRITICAL: Sample Ratio Mismatch detected (p={srm_p_value:.5f} < 0.01). "
            f"Traffic allocation is biased. Results may have systematic selection bias."
        )

    # Step 4: Merge country segmentation if provided
    countries_matched = 0
    if countries_path and os.path.exists(countries_path):
        df_countries = pd.read_csv(countries_path)
        if "user_id" in df_countries.columns and "country" in df_countries.columns:
            df_countries = df_countries.drop_duplicates(subset=["user_id"])
            df_clean = df_clean.merge(df_countries, on="user_id", how="left")
            countries_matched = int(df_clean["country"].notna().sum())
            df_clean["country"] = df_clean["country"].fillna("UNKNOWN")

    ctrl_rate = float(df_clean[df_clean["group"] == "control"]["converted"].mean())
    trt_rate = float(df_clean[df_clean["group"] == "treatment"]["converted"].mean())

    report = ValidationReport(
        total_raw_rows=total_raw_rows,
        mismatched_assignments=mismatched_count,
        duplicate_users=duplicate_users,
        clean_rows=total_clean,
        control_count=n_ctrl,
        treatment_count=n_trt,
        control_conversion_rate=ctrl_rate,
        treatment_conversion_rate=trt_rate,
        srm_chi2_stat=float(chi2_stat),
        srm_p_value=float(srm_p_value),
        srm_passed=srm_passed,
        countries_matched=countries_matched,
        warnings=warnings
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_clean.to_csv(output_path, index=False)

    return df_clean, report


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_ab = os.path.join(base_dir, "data", "raw", "ab_data.csv")
    raw_countries = os.path.join(base_dir, "data", "raw", "countries.csv")
    out_clean = os.path.join(base_dir, "data", "processed", "clean_experiment_data.csv")

    clean_df, rep = validate_and_clean_dataset(raw_ab, raw_countries, out_clean)
    print("=" * 60)
    print("A/B EXPERIMENT DATA VALIDATION & INTEGRITY REPORT")
    print("=" * 60)
    for k, v in rep.to_dict().items():
        print(f"{k:30s}: {v}")
    print("=" * 60)
    print(f"Clean dataset saved to: {out_clean} ({clean_df.shape[0]} rows)")
