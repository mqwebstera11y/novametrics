"""
tests/test_LTV.py — Unit tests for LTV.py

Run with: pytest tests/test_LTV.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from LTV import compute_LTV, build_survival_curve


# ── build_survival_curve ──────────────────────────────────────────────────────

def test_survival_curve_length():
    curve = build_survival_curve(0.32, 0.05, months=12)
    assert len(curve) == 13  # months+1

def test_survival_curve_starts_at_one():
    curve = build_survival_curve(0.32, 0.05)
    assert curve[0] == 1.0

def test_survival_curve_month1_anchor():
    curve = build_survival_curve(0.32, 0.05)
    assert curve[1] == pytest.approx(0.32)

def test_survival_curve_decreasing():
    curve = build_survival_curve(0.40, 0.05)
    for i in range(len(curve) - 1):
        assert curve[i] >= curve[i + 1]

def test_survival_curve_churn_applied():
    curve = build_survival_curve(0.40, 0.10, months=3)
    assert curve[2] == pytest.approx(0.40 * 0.90)
    assert curve[3] == pytest.approx(0.40 * 0.90 * 0.90)


# ── compute_LTV ───────────────────────────────────────────────────────────────

def test_ltv_known_simple():
    """With 100% survival and 1 month, LTV = revenue - 0 loss."""
    curve = [1.0, 1.0]  # one month, nobody churns
    result = compute_LTV(curve, monthly_net_revenue=10.0, cac=5.0, months=1)
    assert result["gross_revenue"] == pytest.approx(10.0)
    assert result["LTV"] == pytest.approx(10.0)
    assert result["payback_month"] == 1

def test_ltv_cac_not_recovered():
    """High CAC, low survival — payback should be None."""
    curve = build_survival_curve(0.10, 0.50, months=12)
    result = compute_LTV(curve, monthly_net_revenue=5.0, cac=100.0, months=12)
    assert result["payback_month"] is None
    assert result["LTV"] < 0

def test_ltv_treatment_greater_than_control():
    """Treatment curve (higher retention) must produce higher LTV."""
    ctrl  = build_survival_curve(0.321, 0.05, months=12)
    treat = build_survival_curve(0.357, 0.05, months=12)
    ltv_ctrl  = compute_LTV(ctrl,  11.49, 40.0)
    ltv_treat = compute_LTV(treat, 11.49, 40.0)
    assert ltv_treat["LTV"] > ltv_ctrl["LTV"]

def test_ltv_incremental_matches_headline():
    """Verify the headline numbers from Job 4."""
    ctrl  = build_survival_curve(0.321, 0.05, months=12)
    treat = build_survival_curve(0.357, 0.05, months=12)
    ltv_ctrl  = compute_LTV(ctrl,  11.49, 40.0)
    ltv_treat = compute_LTV(treat, 11.49, 40.0)
    incremental = round(ltv_treat["LTV"] - ltv_ctrl["LTV"], 2)
    # Should be approximately $5.93 (within $1 tolerance for rounding)
    assert 4.0 <= incremental <= 8.0

def test_ltv_wrong_survival_length():
    with pytest.raises(ValueError, match="length"):
        compute_LTV([1.0, 0.5], 10.0, 40.0, months=12)

def test_ltv_survival_must_start_at_one():
    with pytest.raises(ValueError, match="1.0"):
        bad_curve = [0.9] + [0.8] * 12
        compute_LTV(bad_curve, 10.0, 40.0, months=12)

def test_ltv_cumrev_length():
    curve = build_survival_curve(0.35, 0.05, months=12)
    result = compute_LTV(curve, 11.49, 40.0)
    assert len(result["cumrev"]) == 13  # months+1

def test_ltv_cumrev_monotone():
    curve = build_survival_curve(0.35, 0.05, months=12)
    result = compute_LTV(curve, 11.49, 40.0)
    for i in range(len(result["cumrev"]) - 1):
        assert result["cumrev"][i] <= result["cumrev"][i + 1]
