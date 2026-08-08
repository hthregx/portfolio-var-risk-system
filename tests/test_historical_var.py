import numpy as np
import pytest

from src.models.historical_var import historical_var_forecast


def test_historical_var_forecast_returns_expected_values():
    returns = [-0.05, -0.03, 0.00, 0.01, 0.02]

    forecast = historical_var_forecast(returns, alpha=0.05)

    assert set(forecast) == {"quantile_return", "var"}
    assert forecast["quantile_return"] == pytest.approx(-0.046, abs=1e-12)
    assert forecast["var"] == pytest.approx(0.046, abs=1e-12)


def test_historical_var_forecast_uses_zero_floor_for_positive_quantile():
    forecast = historical_var_forecast([0.01, 0.02, 0.03], alpha=0.05)

    assert forecast["quantile_return"] > 0.0
    assert forecast["var"] == 0.0


def test_historical_var_forecast_does_not_mutate_input():
    returns = np.array([-0.05, -0.03, 0.00, 0.01, 0.02], dtype="float64")
    original = returns.copy()

    historical_var_forecast(returns)

    np.testing.assert_array_equal(returns, original)


@pytest.mark.parametrize("returns", [[], [0.01, np.nan], [0.01, np.inf], [0.01, -np.inf]])
def test_historical_var_forecast_rejects_empty_or_nonfinite_returns(returns):
    with pytest.raises(ValueError):
        historical_var_forecast(returns)


def test_historical_var_forecast_rejects_multidimensional_returns():
    with pytest.raises(ValueError, match="one-dimensional"):
        historical_var_forecast([[0.01, 0.02], [0.03, 0.04]])


def test_historical_var_forecast_rejects_nonnumeric_returns():
    with pytest.raises(ValueError, match="numeric"):
        historical_var_forecast([0.01, "invalid", 0.02])


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.01, 1.01])
def test_historical_var_forecast_rejects_alpha_outside_open_unit_interval(alpha):
    with pytest.raises(ValueError, match="between 0 and 1"):
        historical_var_forecast([-0.02, 0.01, 0.03], alpha=alpha)


def test_historical_var_forecast_rejects_nonnumeric_alpha():
    with pytest.raises(ValueError, match="numeric"):
        historical_var_forecast([-0.02, 0.01, 0.03], alpha="invalid")
