"""Unit tests for Frequentist Engine"""

import pytest
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if 'src' in sys.modules:
    del sys.modules['src']

from src.frequentist_engine import two_proportion_ztest


def test_two_proportion_identical_groups():
    res = two_proportion_ztest(500, 5000, 500, 5000, alpha=0.05)
    assert not res.is_statistically_significant
    assert res.p_value > 0.90
    assert abs(res.relative_lift_pct) < 0.001


def test_two_proportion_clear_winner():
    res = two_proportion_ztest(100, 1000, 250, 1000, alpha=0.05, mde=0.02)
    assert res.is_statistically_significant
    assert res.p_value < 0.0001
    assert res.relative_lift_pct > 50
    assert res.is_practically_significant
