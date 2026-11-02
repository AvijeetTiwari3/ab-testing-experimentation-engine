"""Unit tests for Bayesian Engine"""

import pytest
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if 'src' in sys.modules:
    del sys.modules['src']

from src.bayesian_engine import bayesian_ab_test


def test_bayesian_equal_groups():
    res = bayesian_ab_test(500, 5000, 500, 5000, n_simulations=10000)
    assert 0.45 <= res.prob_treatment_superior <= 0.55


def test_bayesian_strong_treatment():
    res = bayesian_ab_test(100, 1000, 250, 1000, n_simulations=10000)
    assert res.prob_treatment_superior > 0.99
    assert res.expected_relative_lift_pct > 100
    assert "SHIP TREATMENT" in res.recommendation
