from statistics import NormalDist

import numpy as np
import pytest

from src.models.ewma_var import ewma_var_forecast


def test_ewma_var_forecast_returns_expected_values():
    returns = [-0.02, 0.01, -0.03, 0.015]

    forecast = ewma_var_forecast(returns, alpha=0.05, decay=0.94)

    assert set(forecast) == {"quantile_return", "var"}
    assert forecast["quantile_return"] == pytest.approx(-0.032970810928, abs=1e-12)
    assert forecast["var"] == pytest.approx(0.032970810928, abs=1e-12)


def test_ewma_var_forecast_uses_default_decay_094():
    returns = [-0.02, 0.01, -0.03, 0.015]

    default_forecast = ewma_var_forecast(returns)
    explicit_forecast = ewma_var_forecast(returns, decay=0.94)

    assert default_forecast == explicit_forecast


def test_ewma_var_forecast_single_observation_uses_squared_return_initialization():
    forecast = ewma_var_forecast([-0.02], alpha=0.05)
    expected_quantile = NormalDist().inv_cdf(0.05) * 0.02

    assert forecast["quantile_return"] == pytest.approx(expected_quantile, abs=1e-12)
    assert forecast["var"] == pytest.approx(-expected_quantile, abs=1e-12)


def test_ewma_var_forecast_uses_zero_floor_for_positive_quantile():
    forecast = ewma_var_forecast([-0.02, 0.01, -0.03], alpha=0.95)

    assert forecast["quantile_return"] > 0.0
    assert forecast["var"] == 0.0


def test_ewma_var_forecast_does_not_mutate_input():
    returns = np.array([-0.02, 0.01, -0.03, 0.015], dtype="float64")
    original = returns.copy()

    ewma_var_forecast(returns)

    np.testing.assert_array_equal(returns, original)


@pytest.mark.parametrize("returns", [[], [0.01, np.nan], [0.01, np.inf], [0.01, -np.inf]])
def test_ewma_var_forecast_rejects_empty_or_nonfinite_returns(returns):
    with pytest.raises(ValueError):
        ewma_var_forecast(returns)


def test_ewma_var_forecast_rejects_multidimensional_returns():
    with pytest.raises(ValueError, match="one-dimensional"):
        ewma_var_forecast([[0.01, 0.02], [0.03, 0.04]])


def test_ewma_var_forecast_rejects_nonnumeric_returns():
    with pytest.raises(ValueError, match="numeric"):
        ewma_var_forecast([0.01, "invalid", 0.02])


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.01, 1.01])
def test_ewma_var_forecast_rejects_alpha_outside_open_unit_interval(alpha):
    with pytest.raises(ValueError, match="between 0 and 1"):
        ewma_var_forecast([-0.02, 0.01, 0.03], alpha=alpha)


def test_ewma_var_forecast_rejects_nonnumeric_alpha():
    with pytest.raises(ValueError, match="numeric"):
        ewma_var_forecast([-0.02, 0.01, 0.03], alpha="invalid")


@pytest.mark.parametrize("decay", [0.0, 1.0, -0.01, 1.01])
def test_ewma_var_forecast_rejects_decay_outside_open_unit_interval(decay):
    with pytest.raises(ValueError, match="between 0 and 1"):
        ewma_var_forecast([-0.02, 0.01, 0.03], decay=decay)


def test_ewma_var_forecast_rejects_nonnumeric_decay():
    with pytest.raises(ValueError, match="numeric"):
        ewma_var_forecast([-0.02, 0.01, 0.03], decay="invalid")
