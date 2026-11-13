import math

import numpy as np
import pytest

from src.phase2.backtesting_b import (
    backtest_summary,
    kupiec_uc,
    violation_flags,
)


def test_strict_violation_rule():
    actual = [-0.03, -0.02, 0.01]
    quantile = [-0.02, -0.02, -0.02]

    np.testing.assert_array_equal(
        violation_flags(actual, quantile),
        [True, False, False],
    )


def test_hand_calculated_metrics():
    result = backtest_summary(
        actual=[-0.04, -0.02, 0.01],
        quantile=[-0.02, -0.02, -0.02],
    )

    # Pinball losses: 0.019, 0, 0.0015.
    assert result["n_targets"] == 3
    assert result["n_violations"] == 1
    assert result["violation_rate"] == pytest.approx(1 / 3)
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["mean_pinball_loss"] == pytest.approx(0.0205 / 3)
    assert result["average_var"] == pytest.approx(0.02)


def test_var_magnitude_is_clipped_at_zero():
    result = backtest_summary(
        actual=[0.0, 0.02],
        quantile=[0.01, -0.03],
    )
    assert result["average_var"] == pytest.approx(0.015)


def test_nominal_exception_rate():
    result = kupiec_uc(100, 5)

    assert result["kupiec_lr"] == pytest.approx(0.0, abs=1e-12)
    assert result["kupiec_p_value"] == pytest.approx(1.0)
    assert result["kupiec_reject"] is False


def test_known_likelihood_ratio():
    # n=10, x=5, alpha=0.1:
    # LR = 10 * log(25/9) = 10.216512475319814.
    result = kupiec_uc(10, 5, alpha=0.1)

    assert result["kupiec_lr"] == pytest.approx(
        10.216512475319814, abs=1e-12
    )
    assert result["kupiec_reject"] is True


def test_known_chi_square_tail_probability():
    # With n=1, x=0 and alpha=1-exp(-0.5), LR=1.
    result = kupiec_uc(
        1, 0, alpha=1.0 - math.exp(-0.5)
    )

    assert result["kupiec_lr"] == pytest.approx(1.0)
    assert result["kupiec_p_value"] == pytest.approx(
        0.31731050786291415, abs=1e-12
    )


@pytest.mark.parametrize("x", [0, 100])
def test_boundary_counts_are_finite(x):
    result = kupiec_uc(100, x)

    expected = (
        -200 * math.log(0.95)
        if x == 0
        else -200 * math.log(0.05)
    )

    assert result["kupiec_lr"] == pytest.approx(expected)
    assert math.isfinite(result["kupiec_p_value"])
    assert 0.0 <= result["kupiec_p_value"] <= 1.0
    assert result["boundary_observation"] is True


def test_significance_does_not_change_statistic():
    first = kupiec_uc(1, 0, alpha=1 - math.exp(-0.5), significance=0.05)
    second = kupiec_uc(1, 0, alpha=1 - math.exp(-0.5), significance=0.5)

    assert first["kupiec_lr"] == second["kupiec_lr"]
    assert first["kupiec_p_value"] == second["kupiec_p_value"]
    assert first["kupiec_reject"] is False
    assert second["kupiec_reject"] is True


@pytest.mark.parametrize(
    "n,x",
    [(0, 0), (-1, 0), (10, -1), (10, 11), (10.5, 1), (True, 0)],
)
def test_invalid_counts(n, x):
    with pytest.raises(ValueError):
        kupiec_uc(n, x)


@pytest.mark.parametrize("value", [0, 1, -0.1, np.nan, np.inf, True])
def test_invalid_alpha(value):
    with pytest.raises(ValueError):
        kupiec_uc(100, 5, alpha=value)


@pytest.mark.parametrize("value", [0, 1, np.nan, True])
def test_invalid_significance(value):
    with pytest.raises(ValueError):
        kupiec_uc(100, 5, significance=value)


@pytest.mark.parametrize(
    "actual,quantile",
    [
        ([], []),
        ([0.0], [0.0, 0.1]),
        ([np.nan], [0.0]),
        ([0.0], [np.inf]),
        ([[0.0]], [[0.0]]),
    ],
)
def test_invalid_vectors(actual, quantile):
    with pytest.raises(ValueError):
        backtest_summary(actual, quantile)