"""
experiment.py — A/B framework: power calculation, assignment, z-test, verdict.

All functions are stateless and fully unit-testable.
Extracted from Job 4 notebook — do not rewrite the core logic.
"""
from __future__ import annotations

import hashlib
import math


def power_calc(
    p_base: float,
    mde: float = 0.02,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Compute required sample size per arm for a two-proportion z-test.

    Parameters
    ----------
    p_base:
        Baseline retention rate (proportion).
    mde:
        Minimum detectable effect in percentage points (default 0.02 = 2pp).
    alpha:
        Significance level (default 0.05).
    power:
        Desired statistical power (default 0.80).

    Returns
    -------
    int
        Required number of users per arm.
    """
    from scipy import stats
    z_alpha2 = stats.norm.ppf(1 - alpha / 2)
    z_beta   = stats.norm.ppf(power)
    n = 2 * ((z_alpha2 + z_beta) ** 2 * p_base * (1 - p_base)) / (mde ** 2)
    return math.ceil(n)


def assign_ab_group(user_id: str) -> str:
    """
    Deterministically assign a user to control or treatment via MD5 hash.

    Same user_id always returns the same group — guaranteed.
    Split is approximately 50/50 across large populations.

    Parameters
    ----------
    user_id:
        Any string user identifier.

    Returns
    -------
    str
        "treatment" or "control".
    """
    digest = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return "treatment" if digest % 2 == 0 else "control"


def run_ztest(
    n_treatment: int,
    n_control: int,
    retained_treatment: int,
    retained_control: int,
    alpha: float = 0.05,
) -> dict:
    """
    Run a one-sided two-proportion z-test (treatment > control).

    Parameters
    ----------
    n_treatment, n_control:
        Total users in each arm.
    retained_treatment, retained_control:
        Number of retained users in each arm.
    alpha:
        Significance level.

    Returns
    -------
    dict with keys: z_stat, p_value, significant, r_treatment, r_control, lift_pp
    """
    from statsmodels.stats.proportion import proportions_ztest
    import numpy as np

    count = np.array([retained_treatment, retained_control])
    nobs  = np.array([n_treatment, n_control])
    z_stat, p_value = proportions_ztest(count, nobs, alternative="larger")

    r_ctrl  = retained_control  / n_control
    r_treat = retained_treatment / n_treatment

    return {
        "z_stat":      round(float(z_stat), 4),
        "p_value":     round(float(p_value), 4),
        "significant": bool(p_value < alpha),
        "r_control":   round(r_ctrl, 4),
        "r_treatment": round(r_treat, 4),
        "lift_pp":     round(r_treat - r_ctrl, 4),
    }


def compute_verdict(
    lift_pp: float,
    r_control: float,
    r_treatment: float,
    z_stat: float,
    p_value: float,
    significant: bool,
    alpha: float = 0.05,
) -> str:
    """
    Generate a plain-English verdict string for the Vercel frontend.

    Parameters
    ----------
    lift_pp:
        Absolute retention lift (treatment minus control).
    r_control, r_treatment:
        Day-30 retention rates as proportions.
    z_stat:
        z-statistic from the two-proportion z-test.
    p_value:
        p-value from the test.
    significant:
        Whether the result is statistically significant.
    alpha:
        Significance level used.

    Returns
    -------
    str
        One-sentence plain-English verdict.
    """
    sig_str = "statistically significant" if significant else "NOT statistically significant"
    return (
        f"Treatment improves 30-day retention by {lift_pp*100:+.1f}pp "
        f"({r_control:.1%} \u2192 {r_treatment:.1%}). "
        f"Result is {sig_str} "
        f"(z={z_stat:.2f}, p={p_value:.4f}, \u03b1={alpha})."
    )
