"""
LTV.py — 12-month LTV calculator and payback period.

Extracted verbatim from Job 4 notebook. Do not rewrite.

Formula is structurally identical to PD x LGD in credit risk:
    PD  (probability of default)  = P(churn at month m)
    LGD (loss given default)      = unrecovered_CAC(m)
    Expected loss                 = sum of PD x LGD across all months

The only terminology change: "loan" -> "subscription user".
"""
from __future__ import annotations


def compute_LTV(
    survival_probs: list[float],
    monthly_net_revenue: float,
    cac: float,
    months: int = 12,
) -> dict:
    """
    Compute 12-month cohort LTV using the blueprint PD x LGD formula.

    Parameters
    ----------
    survival_probs:
        List of length months+1. Index 0 = 1.0 (start of month 1).
        Each element is the probability a user is still active at that month.
    monthly_net_revenue:
        Net revenue per active user per month (subscription fee minus
        ongoing infrastructure cost). Single float, same for both groups.
    cac:
        One-time customer acquisition cost. CONFIG PARAM.
    months:
        Number of months to project. Default 12.

    Returns
    -------
    dict with keys:
        gross_revenue  : float — total expected revenue over `months`.
        expected_loss  : float — expected unrecovered CAC weighted by churn.
        LTV            : float — net LTV = gross_revenue - expected_loss.
        payback_month  : int or None — first month where cumrev >= CAC.
        cumrev         : list[float] — cumulative revenue by month (length months+1).
    """
    if len(survival_probs) != months + 1:
        raise ValueError(
            f"survival_probs must have length months+1={months+1}, "
            f"got {len(survival_probs)}."
        )
    if survival_probs[0] != 1.0:
        raise ValueError(
            f"survival_probs[0] must be 1.0 (100% at start), got {survival_probs[0]}."
        )

    # Step 1 — cumulative revenue at each month
    cumrev = [0.0]
    for m in range(1, months + 1):
        cumrev.append(cumrev[-1] + monthly_net_revenue * survival_probs[m])

    # Step 2 — unrecovered CAC at each month
    unrecovered = [max(0.0, cac - cumrev[m]) for m in range(months + 1)]

    # Step 3 — probability of churning at exactly month m
    p_churn = [0.0]
    for m in range(1, months + 1):
        p_churn.append(survival_probs[m - 1] - survival_probs[m])

    # Step 4 — expected unrecovered CAC (weighted by churn probability)
    e_loss = sum(p_churn[m] * unrecovered[m] for m in range(1, months + 1))
    # Users who survive all months but still haven't recovered CAC
    e_loss += survival_probs[months] * max(0.0, cac - cumrev[months])

    # Step 5 — gross revenue
    gross_revenue = cumrev[months]

    # Step 6 — LTV
    ltv = gross_revenue - e_loss

    # Payback month
    payback_month = next(
        (m for m in range(1, months + 1) if cumrev[m] >= cac), None
    )

    return {
        "gross_revenue": round(gross_revenue, 2),
        "expected_loss": round(e_loss, 2),
        "LTV":           round(ltv, 2),
        "payback_month": payback_month,
        "cumrev":        [round(v, 2) for v in cumrev],
    }


def build_survival_curve(
    p_month1: float,
    monthly_churn: float,
    months: int = 12,
) -> list[float]:
    """
    Build a survival curve from a month-1 anchor and constant monthly churn.

    Parameters
    ----------
    p_month1:
        Probability of surviving to month 1 (day-30 retention rate).
    monthly_churn:
        Constant monthly churn rate applied from month 2 onward.
    months:
        Number of months to project. Default 12.

    Returns
    -------
    list of float, length months+1.
        Index 0 = 1.0, index 1 = p_month1, subsequent months decay
        by (1 - monthly_churn).
    """
    curve = [1.0, p_month1]
    for _ in range(2, months + 1):
        curve.append(curve[-1] * (1 - monthly_churn))
    return curve
