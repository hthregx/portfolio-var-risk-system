"""Independent VaR backtesting utilities for Phase 2.

Inputs use decimal returns, not percentages.
Kupiec p-values use the asymptotic chi-square(1) distribution.
No training, file reading or file writing occurs in this module.
"""

from math import erfc, log, log1p, sqrt
from numbers import Integral, Real

import numpy as np


def _probability(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
    ):
        raise ValueError(f"{name} must be a real number in (0, 1).")

    value = float(value)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1).")
    return value


def _count(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
    ):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def _paired_returns(actual, quantile):
    actual = np.asarray(actual, dtype=float)
    quantile = np.asarray(quantile, dtype=float)

    if actual.ndim != 1 or quantile.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")
    if actual.size == 0 or actual.shape != quantile.shape:
        raise ValueError("Inputs must be nonempty and have equal length.")
    if not np.isfinite(actual).all():
        raise ValueError("Actual returns contain NaN or infinity.")
    if not np.isfinite(quantile).all():
        raise ValueError("Quantile forecasts contain NaN or infinity.")

    return actual, quantile


def violation_flags(actual, quantile):
    """Strict comparison: equality is not a violation."""
    actual, quantile = _paired_returns(actual, quantile)
    return actual < quantile


def kupiec_uc(
    n_targets,
    n_violations,
    alpha=0.05,
    significance=0.05,
):
    """Kupiec unconditional coverage likelihood-ratio test.

    H0: violation probability equals alpha.

    Boundary cases x=0 and x=n use the convention 0*log(0)=0.
    The returned p-value is asymptotic, not an exact binomial p-value.
    """
    n = _count(n_targets, "n_targets")
    x = _count(n_violations, "n_violations")
    alpha = _probability(alpha, "alpha")
    significance = _probability(significance, "significance")

    if n <= 0:
        raise ValueError("n_targets must be positive.")
    if not 0 <= x <= n:
        raise ValueError("n_violations must be between 0 and n_targets.")

    observed = x / n

    # Log likelihood under H0.
    log_likelihood_null = (
        x * log(alpha) + (n - x) * log1p(-alpha)
    )

    # Log likelihood at the unrestricted maximum.
    log_likelihood_fitted = 0.0
    if x > 0:
        log_likelihood_fitted += x * log(observed)
    if x < n:
        log_likelihood_fitted += (n - x) * log1p(-observed)

    lr = max(
        0.0,
        2.0 * (log_likelihood_fitted - log_likelihood_null),
    )

    # Survival function of chi-square with one degree of freedom.
    p_value = erfc(sqrt(lr / 2.0))
    reject = p_value < significance

    return {
        "kupiec_lr": float(lr),
        "kupiec_p_value": float(p_value),
        "kupiec_significance": significance,
        "kupiec_reject": bool(reject),
        "kupiec_decision": (
            "reject_H0" if reject else "do_not_reject_H0"
        ),
        "kupiec_reference_distribution": "chi_square_df1_asymptotic",
        "expected_violations": n * alpha,
        "expected_nonviolations": n * (1.0 - alpha),
        "boundary_observation": bool(x == 0 or x == n),
    }


def backtest_summary(
    actual,
    quantile,
    alpha=0.05,
    significance=0.05,
):
    """Compute metrics directly from actual returns and forecasts."""
    alpha = _probability(alpha, "alpha")
    actual, quantile = _paired_returns(actual, quantile)

    flags = actual < quantile
    n = actual.size
    x = int(flags.sum())

    error = actual - quantile
    losses = np.where(
        error >= 0.0,
        alpha * error,
        (alpha - 1.0) * error,
    )

    result = {
        "n_targets": int(n),
        "n_violations": x,
        "quantile_alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "violation_rate": x / n,
        "coverage": 1.0 - x / n,
        "mean_pinball_loss": float(losses.mean()),
        "average_var": float(np.maximum(0.0, -quantile).mean()),
    }

    result.update(
        kupiec_uc(
            n_targets=int(n),
            n_violations=x,
            alpha=alpha,
            significance=significance,
        )
    )
    return result