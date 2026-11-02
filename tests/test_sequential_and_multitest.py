"""Unit tests for Sequential and Multiple Testing Engines"""

import pytest
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if 'src' in sys.modules:
    del sys.modules['src']

from src.sequential_engine import compute_obrien_fleming_boundary
from src.multiple_testing import adjust_multiple_metrics


def test_obrien_fleming_boundary_decay():
    # Early look (t=0.1) should have very high critical z
    z_early, alpha_early = compute_obrien_fleming_boundary(alpha=0.05, information_fraction=0.1)
    # Late look (t=1.0) should approach standard z_alpha (1.96)
    z_late, alpha_late = compute_obrien_fleming_boundary(alpha=0.05, information_fraction=1.0)

    assert z_early > z_late
    assert alpha_early < alpha_late
    assert round(z_late, 2) == 1.96


def test_multiple_testing_fdr_vs_bonferroni():
    metric_names = ["m1", "m2", "m3", "m4", "m5"]
    p_vals = [0.001, 0.015, 0.035, 0.045, 0.200]
    res = adjust_multiple_metrics(metric_names, p_vals, alpha=0.05, q_threshold=0.05)

    assert res.total_metrics_tested == 5
    assert res.raw_significant_count == 4
    # FDR should be less or equal conservative than Bonferroni
    assert res.bh_fdr_significant_count >= res.bonferroni_significant_count
