"""Unit tests for Power Analysis & Sizing Engine"""

import pytest
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if 'src' in sys.modules:
    del sys.modules['src']

from src.power_analysis import calculate_sample_size_proportions, calculate_mde


def test_power_analysis_symmetry_and_direction():
    res = calculate_sample_size_proportions(
        baseline_rate=0.10,
        mde_absolute=0.01,
        alpha=0.05,
        power=0.80,
        daily_traffic=5000
    )
    assert res.required_sample_per_variant > 1000
    assert res.total_required_sample == res.required_sample_per_variant * 2
    assert res.estimated_duration_days > 0


def test_mde_calculation():
    mde = calculate_mde(baseline_rate=0.12, sample_size_per_variant=50000, alpha=0.05, power=0.80)
    assert 0.001 < mde < 0.01
