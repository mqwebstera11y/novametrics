"""
tests/test_experiment.py — Unit tests for experiment.py

Run with: pytest tests/test_experiment.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from experiment import assign_ab_group, power_calc, compute_verdict


# ── assign_ab_group ───────────────────────────────────────────────────────────

def test_assignment_deterministic():
    """Same user_id must always get the same group."""
    uid = "AEKSXGXY2HYFHZ5NO6LR"
    assert assign_ab_group(uid) == assign_ab_group(uid)
    assert assign_ab_group(uid) == assign_ab_group(uid)

def test_assignment_returns_valid_group():
    for uid in ["user1", "user2", "user3", "abc", "xyz123"]:
        assert assign_ab_group(uid) in ("control", "treatment")

def test_assignment_roughly_balanced():
    """Over 10,000 users, split should be close to 50/50."""
    groups = [assign_ab_group(f"user_{i}") for i in range(10_000)]
    n_treat = sum(1 for g in groups if g == "treatment")
    ratio = n_treat / 10_000
    assert 0.45 <= ratio <= 0.55

def test_assignment_different_users_can_differ():
    """Different users should not all get the same group."""
    groups = {assign_ab_group(f"u{i}") for i in range(100)}
    assert len(groups) == 2  # both groups must appear


# ── power_calc ────────────────────────────────────────────────────────────────

def test_power_calc_returns_int():
    n = power_calc(0.32, mde=0.02)
    assert isinstance(n, int)
    assert n > 0

def test_power_calc_larger_mde_needs_fewer():
    n_small_mde = power_calc(0.32, mde=0.01)
    n_large_mde = power_calc(0.32, mde=0.05)
    assert n_small_mde > n_large_mde

def test_power_calc_known_value():
    """At p=0.32, MDE=2pp, alpha=0.05, power=0.80 → roughly 7,000–9,000 per arm."""
    n = power_calc(0.32, mde=0.02, alpha=0.05, power=0.80)
    assert 5_000 <= n <= 12_000


# ── compute_verdict ───────────────────────────────────────────────────────────

def test_verdict_significant():
    v = compute_verdict(0.035, 0.321, 0.356, 3.72, 0.0001, True)
    assert "statistically significant" in v
    assert "NOT" not in v
    assert "+3.5pp" in v

def test_verdict_not_significant():
    v = compute_verdict(0.010, 0.321, 0.331, 0.95, 0.17, False)
    assert "NOT statistically significant" in v

def test_verdict_contains_rates():
    v = compute_verdict(0.035, 0.321, 0.356, 3.72, 0.0001, True)
    assert "32.1%" in v
    assert "35.6%" in v

def test_verdict_is_string():
    v = compute_verdict(0.035, 0.321, 0.356, 3.72, 0.0001, True)
    assert isinstance(v, str)
    assert len(v) > 30
