from functools import partial

import numpy as np
import pandas as pd
import pytest

from src.backtesting.walk_forward import walk_forward
from src.models.ewma_var import ewma_var_forecast
from src.models.historical_var import historical_var_forecast


def leakage_data():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-08",
                    "2024-01-09",
                    "2024-01-10",
                    "2024-01-11",
                ]
            ),
            "portfolio_simple_return": [
                0.010,
                -0.020,
                0.030,
                -0.010,
                0.015,
                -0.025,
                0.005,
                -0.012,
            ],
        }
    )


@pytest.mark.parametrize(
    ("model", "mode", "method"),
    [
        (
            partial(
                historical_var_forecast,
                alpha=0.05,
            ),
            "rolling",
            "historical",
        ),
        (
            partial(
                ewma_var_forecast,
                alpha=0.05,
                decay=0.94,
            ),
            "expanding",
            "ewma",
        ),
    ],
)
def test_forecast_origin_is_strictly_before_target(
    model,
    mode,
    method,
):
    result = walk_forward(
        leakage_data(),
        model,
        window_size=3,
        mode=mode,
        method=method,
    )

    assert (
        result["forecast_date"]
        < result["target_date"]
    ).all()

    assert (
        result["window_end_date"]
        == result["forecast_date"]
    ).all()
@pytest.mark.parametrize(
    "mode",
    ["rolling", "expanding"],
)
def test_training_dates_never_reach_target(mode):
    data = leakage_data().sort_values(
        "date"
    ).reset_index(drop=True)

    window_size = 3
    captured = []

    def recording_model(train_returns):
        captured.append(
            np.array(
                train_returns,
                copy=True,
            )
        )

        return {
            "quantile_return": -0.02,
            "var": 0.02,
        }

    result = walk_forward(
        data,
        recording_model,
        window_size=window_size,
        mode=mode,
        method="recording",
    )

    for output_index, target_index in enumerate(
        range(window_size, len(data))
    ):
        if mode == "rolling":
            start_index = target_index - window_size
        else:
            start_index = 0

        training_dates = data.loc[
            start_index:target_index - 1,
            "date",
        ]

        forecast_date = result.iloc[
            output_index
        ]["forecast_date"]

        target_date = result.iloc[
            output_index
        ]["target_date"]

        assert training_dates.max() <= forecast_date
        assert training_dates.max() < target_date

        expected_returns = data.loc[
            start_index:target_index - 1,
            "portfolio_simple_return",
        ].to_numpy()

        np.testing.assert_allclose(
            captured[output_index],
            expected_returns,
            atol=1e-12,
            rtol=0.0,
        )
@pytest.mark.parametrize(
    ("model", "mode", "method"),
    [
        (
            partial(
                historical_var_forecast,
                alpha=0.05,
            ),
            "rolling",
            "historical",
        ),
        (
            partial(
                ewma_var_forecast,
                alpha=0.05,
                decay=0.94,
            ),
            "expanding",
            "ewma",
        ),
    ],
)
def test_future_returns_do_not_change_past_forecast(
    model,
    mode,
    method,
):
    original = leakage_data()

    original_result = walk_forward(
        original,
        model,
        window_size=3,
        mode=mode,
        method=method,
    )

    row_index = 1

    target_date = original_result.iloc[
        row_index
    ]["target_date"]

    perturbed = original.copy()

    future_mask = (
        perturbed["date"] > target_date
    )

    perturbed.loc[
        future_mask,
        "portfolio_simple_return",
    ] = 9.0

    perturbed_result = walk_forward(
        perturbed,
        model,
        window_size=3,
        mode=mode,
        method=method,
    )

    original_row = original_result.loc[
        original_result["target_date"]
        == target_date
    ].iloc[0]

    perturbed_row = perturbed_result.loc[
        perturbed_result["target_date"]
        == target_date
    ].iloc[0]

    assert (
        original_row["forecast_date"]
        == perturbed_row["forecast_date"]
    )

    assert (
        original_row["target_date"]
        == perturbed_row["target_date"]
    )

    assert original_row[
        "quantile_return"
    ] == pytest.approx(
        perturbed_row["quantile_return"],
        abs=1e-12,
    )

    assert original_row[
        "var"
    ] == pytest.approx(
        perturbed_row["var"],
        abs=1e-12,
    )
@pytest.mark.parametrize(
    ("model", "mode", "method"),
    [
        (
            partial(
                historical_var_forecast,
                alpha=0.05,
            ),
            "rolling",
            "historical",
        ),
        (
            partial(
                ewma_var_forecast,
                alpha=0.05,
                decay=0.94,
            ),
            "expanding",
            "ewma",
        ),
    ],
)
def test_target_dates_and_actual_returns_align(
    model,
    mode,
    method,
):
    data = leakage_data().sort_values(
        "date"
    ).reset_index(drop=True)

    result = walk_forward(
        data,
        model,
        window_size=3,
        mode=mode,
        method=method,
    )

    expected_targets = (
        data.iloc[3:]["date"]
        .reset_index(drop=True)
    )

    expected_actual = (
        data.iloc[3:][
            "portfolio_simple_return"
        ]
        .reset_index(drop=True)
    )

    pd.testing.assert_series_equal(
        result["target_date"].reset_index(
            drop=True
        ),
        expected_targets,
        check_names=False,
    )

    np.testing.assert_allclose(
        result["actual_return"],
        expected_actual,
        atol=1e-12,
        rtol=0.0,
    )

    assert not result[
        "target_date"
    ].duplicated().any()


def test_historical_and_ewma_use_same_targets():
    data = leakage_data()

    historical = walk_forward(
        data,
        partial(
            historical_var_forecast,
            alpha=0.05,
        ),
        window_size=3,
        mode="rolling",
        method="historical",
    )

    ewma = walk_forward(
        data,
        partial(
            ewma_var_forecast,
            alpha=0.05,
            decay=0.94,
        ),
        window_size=3,
        mode="expanding",
        method="ewma",
    )

    pd.testing.assert_series_equal(
        historical["target_date"],
        ewma["target_date"],
        check_names=False,
    )

    np.testing.assert_allclose(
        historical["actual_return"],
        ewma["actual_return"],
        atol=1e-12,
        rtol=0.0,
    )


def test_equal_actual_and_quantile_is_not_violation():
    data = leakage_data().iloc[:4].copy()

    target_value = -0.02

    data.loc[
        data.index[-1],
        "portfolio_simple_return",
    ] = target_value

    def equality_model(train_returns):
        return {
            "quantile_return": target_value,
            "var": abs(target_value),
        }

    result = walk_forward(
        data,
        equality_model,
        window_size=3,
        mode="rolling",
        method="equality",
    )

    assert (
        result.iloc[0]["actual_return"]
        == result.iloc[0]["quantile_return"]
    )

    assert bool(
        result.iloc[0]["violation"]
    ) is False