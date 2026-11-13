import numpy as np
import pytest

from src.phase2.metrics_b import (
    average_var,
    pinball_loss,
    summarize_metrics,
    violation_rate,
)


def test_hand_calculated_metrics():
    actual = [-0.04, -0.02, 0.01]
    quantile = [-0.02, -0.02, -0.02]

    np.testing.assert_allclose(
        pinball_loss(actual, quantile),
        [0.019, 0.0, 0.0015],
        rtol=0, atol=1e-12,
    )

    result = summarize_metrics(actual, quantile)
    assert result["n_targets"] == 3
    assert result["n_violations"] == 1
    assert result["mean_pinball_loss"] == pytest.approx(
        0.006833333333333334
    )
    assert result["violation_rate"] == pytest.approx(1 / 3)
    assert result["average_var"] == pytest.approx(0.02)


def test_equality_is_not_violation():
    actual = [-0.02, 0.0, 0.01]
    assert violation_rate(actual, actual) == 0
    np.testing.assert_allclose(pinball_loss(actual, actual), 0)


def test_positive_quantile_keeps_original_violation_rule():
    # Actual dương vẫn có thể thấp hơn quantile dương.
    result = summarize_metrics([0.005], [0.01])
    assert result["n_violations"] == 1
    assert result["average_var"] == 0
    assert result["mean_pinball_loss"] == pytest.approx(0.00475)


def test_var_clips_each_observation_before_averaging():
    # VaR từng dòng là [0.02, 0, 0].
    assert average_var([-0.02, 0.01, 0.0]) == pytest.approx(0.02 / 3)


def test_median_pinball_is_half_absolute_error():
    np.testing.assert_allclose(
        pinball_loss([-0.03, 0.01], [-0.01, 0.0], alpha=0.5),
        [0.01, 0.005],
    )


@pytest.mark.parametrize(
    "actual, quantile",
    [
        ([], []),
        ([0.0, 0.01], [0.0]),
        ([[0.0]], [[0.0]]),
        ([np.nan], [0.0]),
        ([0.0], [np.inf]),
    ],
)
def test_invalid_pairs_are_rejected(actual, quantile):
    with pytest.raises(ValueError):
        summarize_metrics(actual, quantile)


@pytest.mark.parametrize("alpha", [0, 1, -0.05, np.nan, True])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError):
        pinball_loss([0.0], [-0.01], alpha=alpha)


@pytest.mark.parametrize("quantile", [[], [np.nan], [np.inf]])
def test_average_var_rejects_invalid_input(quantile):
    with pytest.raises(ValueError):
        average_var(quantile)