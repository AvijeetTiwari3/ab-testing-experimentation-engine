"""Unit tests for Data Validator & SRM Engine"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if 'src' in sys.modules:
    del sys.modules['src']

from src.data_validator import validate_and_clean_dataset


def test_validator_with_synthetic_sample():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create small test dataset
        csv_path = os.path.join(tmpdir, "test_ab.csv")
        data = {
            "user_id": [1, 2, 3, 4, 1], # 1 duplicate
            "timestamp": ["2026-01-01 10:00:00", "2026-01-01 11:00:00", "2026-01-01 12:00:00", "2026-01-01 13:00:00", "2026-01-01 14:00:00"],
            "group": ["control", "control", "treatment", "treatment", "control"],
            "landing_page": ["old_page", "new_page", "new_page", "old_page", "old_page"], # rows 2 & 4 mismatched
            "converted": [0, 1, 1, 0, 0]
        }
        pd.DataFrame(data).to_csv(csv_path, index=False)

        clean_df, report = validate_and_clean_dataset(csv_path)

        assert report.total_raw_rows == 5
        assert report.mismatched_assignments == 2
        assert len(clean_df) == 2  # remaining valid unique rows
        assert report.clean_rows == 2
