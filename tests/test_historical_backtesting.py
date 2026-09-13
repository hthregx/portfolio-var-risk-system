import numpy as np
import pandas as pd
import pytest

import src.backtesting.historical as historical_backtesting
from src.backtesting.historical import calculate_rolling_historical_var


EXPECTED_COLUMNS = [
    "window_start_date",
    "window_end_date",
    "forecast_date",
    "target_date",
    "observations",
    "quantile_return",
    "historical_var",
    "target_return",
]


def _sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-08",
                ]
            ),
            "portfolio_simple_return": [
                0.01,
                -0.02,
                0.03,
                -0.01,
                0.02,
            ],
        }
    )


def test_calculate_rolling_historical_var_small_sample() -> None:
    result = calculate_rolling_historical_var(
        _sample_data(),
        window_size=3,
        confidence_level=0.95,
    )

    assert result.columns.tolist() == EXPECTED_COLUMNS
    assert len(result) == 2

    expected_dates = [
        {
            "window_start_date": pd.Timestamp("2024-01-02"),
            "window_end_date": pd.Timestamp("2024-01-04"),
            "forecast_date": pd.Timestamp("2024-01-04"),
            "target_date": pd.Timestamp("2024-01-05"),
        },
        {
            "window_start_date": pd.Timestamp("2024-01-03"),
            "window_end_date": pd.Timestamp("2024-01-05"),
            "forecast_date": pd.Timestamp("2024-01-05"),
            "target_date": pd.Timestamp("2024-01-08"),
        },
    ]

    expected_numeric = [
        {
            "quantile_return": -0.017,
            "historical_var": 0.017,
            "target_return": -0.01,
        },
        {
            "quantile_return": -0.019,
            "historical_var": 0.019,
            "target_return": 0.02,
        },
    ]

    for position in range(2):
        row = result.iloc[position]

        for column, expected in expected_dates[position].items():
            assert row[column] == expected

        assert row["observations"] == 3

        for column, expected in expected_numeric[position].items():
            assert np.isclose(
                float(row[column]),
                expected,
                atol=1e-12,
                rtol=0.0,
            )


def test_calculate_rolling_historical_var_sorts_dates_without_lookahead() -> None:
    data = _sample_data().iloc[[3, 0, 4, 1, 2]].reset_index(drop=True)

    result = calculate_rolling_historical_var(
        data,
        window_size=3,
    )

    assert result["target_date"].is_monotonic_increasing
    assert result["forecast_date"].equals(
        result["window_end_date"]
    )
    assert (
        result["forecast_date"]
        < result["target_date"]
    ).all()


def test_calculate_rolling_historical_var_maps_confidence_to_alpha(
    monkeypatch,
) -> None:
    calls = []

    def fake_forecast(returns, alpha=0.05):
        calls.append(
            {
                "returns": np.asarray(
                    returns,
                    dtype="float64",
                ).copy(),
                "alpha": alpha,
            }
        )
        return {
            "quantile_return": -0.01,
            "var": 0.01,
        }

    monkeypatch.setattr(
        historical_backtesting,
        "historical_var_forecast",
        fake_forecast,
    )

    result = calculate_rolling_historical_var(
        _sample_data().iloc[:4].copy(),
        window_size=2,
        confidence_level=0.90,
    )

    assert len(result) == 2
    assert len(calls) == 2

    assert np.isclose(
        calls[0]["alpha"],
        0.10,
        atol=1e-12,
        rtol=0.0,
    )
    assert np.isclose(
        calls[1]["alpha"],
        0.10,
        atol=1e-12,
        rtol=0.0,
    )

    np.testing.assert_allclose(
        calls[0]["returns"],
        np.array([0.01, -0.02]),
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        calls[1]["returns"],
        np.array([-0.02, 0.03]),
        atol=1e-12,
        rtol=0.0,
    )


def test_calculate_rolling_historical_var_rejects_non_dataframe() -> None:
    with pytest.raises(
        ValueError,
        match="Data must be provided as a pandas DataFrame.",
    ):
        calculate_rolling_historical_var([0.01, -0.02, 0.03])


def test_calculate_rolling_historical_var_rejects_empty_dataframe() -> None:
    with pytest.raises(
        ValueError,
        match="Data cannot be empty.",
    ):
        calculate_rolling_historical_var(pd.DataFrame())


def test_calculate_rolling_historical_var_rejects_missing_columns() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04"]
            )
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Missing required columns: \['portfolio_simple_return'\]",
    ):
        calculate_rolling_historical_var(data, window_size=2)


@pytest.mark.parametrize(
    "window_size",
    [True, 1, 0, -1, 2.5, "3"],
)
def test_calculate_rolling_historical_var_rejects_invalid_window(
    window_size,
) -> None:
    with pytest.raises(
        ValueError,
        match="Window size must be an integer greater than or equal to 2.",
    ):
        calculate_rolling_historical_var(
            _sample_data(),
            window_size=window_size,
        )


def test_calculate_rolling_historical_var_rejects_duplicate_dates() -> None:
    data = _sample_data()
    data.loc[1, "date"] = data.loc[0, "date"]

    with pytest.raises(
        ValueError,
        match="Duplicate dates are not allowed.",
    ):
        calculate_rolling_historical_var(data, window_size=3)


def test_calculate_rolling_historical_var_rejects_non_numeric_returns() -> None:
    data = _sample_data()
    data["portfolio_simple_return"] = data["portfolio_simple_return"].astype(object)
    data.loc[1, "portfolio_simple_return"] = "invalid"

    with pytest.raises(
        ValueError,
        match="Returns must contain only numeric values.",
    ):
        calculate_rolling_historical_var(data, window_size=3)


def test_calculate_rolling_historical_var_rejects_missing_returns() -> None:
    data = _sample_data()
    data.loc[1, "portfolio_simple_return"] = np.nan

    with pytest.raises(
        ValueError,
        match="Returns cannot contain missing values.",
    ):
        calculate_rolling_historical_var(data, window_size=3)


def test_calculate_rolling_historical_var_rejects_nonfinite_returns() -> None:
    data = _sample_data()
    data.loc[1, "portfolio_simple_return"] = np.inf

    with pytest.raises(
        ValueError,
        match="Returns must contain only finite values.",
    ):
        calculate_rolling_historical_var(data, window_size=3)


def test_calculate_rolling_historical_var_rejects_insufficient_observations() -> None:
    data = _sample_data().iloc[:3].copy()

    with pytest.raises(
        ValueError,
        match="Data must contain more observations than the rolling window.",
    ):
        calculate_rolling_historical_var(
            data,
            window_size=3,
        )


def test_calculate_rolling_historical_var_rejects_nonnumeric_confidence() -> None:
    with pytest.raises(
        ValueError,
        match="Confidence level must be numeric.",
    ):
        calculate_rolling_historical_var(
            _sample_data(),
            window_size=3,
            confidence_level="invalid",
        )


@pytest.mark.parametrize(
    "confidence_level",
    [0.0, 1.0, -0.1, 1.1, np.nan, np.inf, -np.inf],
)
def test_calculate_rolling_historical_var_rejects_invalid_confidence(
    confidence_level,
) -> None:
    with pytest.raises(
        ValueError,
        match="Confidence level must be finite and strictly between 0 and 1.",
    ):
        calculate_rolling_historical_var(
            _sample_data(),
            window_size=3,
            confidence_level=confidence_level,
        )
