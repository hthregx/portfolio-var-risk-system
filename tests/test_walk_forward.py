from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtesting.walk_forward import (
    PREDICTION_COLUMNS,
    save_predictions,
    walk_forward,
)
from src.models.historical_var import historical_var_forecast
from src.models.ewma_var import ewma_var_forecast


def sample_data():
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
                ]
            ),
            "portfolio_simple_return": [
                0.01,
                -0.02,
                0.03,
                -0.01,
                0.02,
                -0.03,
            ],
        }
    )


def dummy_model(train_returns):
    return {
        "quantile_return": -0.02,
        "var": 0.02,
    }


# B-08.2 / B-08.4 — Schema and rolling evaluation
def test_rolling_count_schema_targets_and_windows():
    calls = []

    def recording_model(train_returns):
        calls.append(np.array(train_returns, copy=True))
        return dummy_model(train_returns)

    result = walk_forward(
        sample_data(),
        recording_model,
        window_size=3,
        mode="rolling",
        method="dummy",
    )

    assert len(result) == 3
    assert result.columns.tolist() == PREDICTION_COLUMNS
    assert result["observations"].tolist() == [3, 3, 3]
    assert result.iloc[0]["target_date"] == pd.Timestamp("2024-01-05")
    assert result.iloc[-1]["target_date"] == pd.Timestamp("2024-01-09")

    expected = [
        [0.01, -0.02, 0.03],
        [-0.02, 0.03, -0.01],
        [0.03, -0.01, 0.02],
    ]

    for actual, wanted in zip(calls, expected, strict=True):
        np.testing.assert_allclose(
            actual,
            wanted,
            atol=1e-12,
            rtol=0.0,
        )


# B-08.5 — Expanding evaluation
def test_expanding_windows():
    result = walk_forward(
        sample_data(),
        dummy_model,
        window_size=3,
        mode="expanding",
    )

    assert result["observations"].tolist() == [3, 4, 5]


# B-08.6 — Independent no-lookahead tests
def test_no_lookahead():
    result = walk_forward(
        sample_data(),
        dummy_model,
        window_size=3,
    )

    assert (
        result["forecast_date"]
        < result["target_date"]
    ).all()


# B-08.7 — Date alignment
def test_date_alignment():
    result = walk_forward(
        sample_data(),
        dummy_model,
        window_size=3,
    )

    assert (
        result["forecast_date"]
        == result["window_end_date"]
    ).all()

    assert (
        result["forecast_date"]
        < result["target_date"]
    ).all()


# B-08.8 — Violation generation
@pytest.mark.parametrize(
    ("target_return", "expected"),
    [
        (-0.03, True),
        (-0.02, False),
        (-0.01, False),
    ],
)
def test_violation_rule(target_return, expected):
    data = sample_data().iloc[:4].copy()
    data.loc[3, "portfolio_simple_return"] = target_return

    result = walk_forward(
        data,
        dummy_model,
        window_size=3,
    )

    assert bool(result.iloc[0]["violation"]) is expected


# B-08.9 — Runtime logging
def test_runtime_logging():
    result_1 = walk_forward(
        sample_data(),
        dummy_model,
        window_size=3,
        method="dummy",
    )

    result_2 = walk_forward(
        sample_data(),
        dummy_model,
        window_size=3,
        method="dummy",
    )

    runtime = result_1.attrs["runtime_seconds"]

    assert runtime >= 0
    assert np.isfinite(runtime)

    pd.testing.assert_frame_equal(
        result_1,
        result_2,
    )


# B-08.10 — Save predictions
def test_save_predictions(tmp_path):
    predictions = walk_forward(
        sample_data(),
        dummy_model,
        window_size=3,
        method="dummy",
    )

    output_path = tmp_path / "predictions.csv"

    saved_path = save_predictions(
        predictions,
        output_path,
    )

    assert saved_path.exists()

    saved = pd.read_csv(saved_path)

    assert len(saved) == len(predictions)
    assert saved.columns.tolist() == PREDICTION_COLUMNS


# B-08.11 — Prediction round-trip validation
def test_prediction_round_trip(tmp_path):
    predictions = walk_forward(
        sample_data(),
        dummy_model,
        window_size=3,
        method="dummy",
    )

    output_path = tmp_path / "predictions.csv"

    save_predictions(
        predictions,
        output_path,
    )

    reloaded = pd.read_csv(
        output_path,
        parse_dates=[
            "window_end_date",
            "forecast_date",
            "target_date",
        ],
    )

    assert len(reloaded) == len(predictions)
    assert reloaded.columns.tolist() == predictions.columns.tolist()

    for column in [
        "window_end_date",
        "forecast_date",
        "target_date",
    ]:
        pd.testing.assert_series_equal(
            reloaded[column],
            predictions[column],
            check_names=False,
        )

    for column in [
        "actual_return",
        "quantile_return",
        "var",
    ]:
        np.testing.assert_allclose(
            reloaded[column],
            predictions[column],
            atol=1e-12,
            rtol=0.0,
        )

    assert (
        reloaded["violation"].astype(bool).to_numpy()
        == predictions["violation"].astype(bool).to_numpy()
    ).all()

    assert (
        reloaded["method"].tolist()
        == predictions["method"].tolist()
    )


# B-08.12 — Historical Simulation integration
def test_runner_works_with_historical_model():
    model = partial(
        historical_var_forecast,
        alpha=0.05,
    )

    result = walk_forward(
        sample_data(),
        model,
        window_size=3,
        mode="rolling",
        method="historical",
    )

    assert len(result) == 3
    assert (result["method"] == "historical").all()
    assert (result["var"] >= 0).all()


# B-08.13 — Historical baseline regression
def test_historical_regression_against_baseline():
    portfolio_path = Path(
        "data/processed/portfolio_returns.csv"
    )
    baseline_path = Path(
        "data/processed/historical_var_backtest.csv"
    )

    if not portfolio_path.exists() or not baseline_path.exists():
        pytest.skip(
            "Historical baseline artifacts unavailable."
        )

    portfolio = pd.read_csv(portfolio_path)

    baseline = pd.read_csv(
        baseline_path,
        parse_dates=[
            "forecast_date",
            "target_date",
        ],
    )

    model = partial(
        historical_var_forecast,
        alpha=0.05,
    )

    result = walk_forward(
        portfolio,
        model,
        window_size=250,
        mode="rolling",
        method="historical",
    )

    assert len(result) == 1387
    assert len(baseline) == 1387

    assert result.iloc[0]["target_date"] == pd.Timestamp("2020-12-31")
    assert result.iloc[-1]["target_date"] == pd.Timestamp("2026-07-28")

    pd.testing.assert_series_equal(
        result["target_date"].reset_index(drop=True),
        baseline["target_date"].reset_index(drop=True),
        check_names=False,
    )

    np.testing.assert_allclose(
        result["actual_return"],
        baseline["target_return"],
        atol=1e-12,
        rtol=0.0,
    )

    np.testing.assert_allclose(
        result["quantile_return"],
        baseline["quantile_return"],
        atol=1e-12,
        rtol=0.0,
    )

    np.testing.assert_allclose(
        result["var"],
        baseline["historical_var"],
        atol=1e-12,
        rtol=0.0,
    )

    violations = (
        result["actual_return"]
        < result["quantile_return"]
    )

    assert int(violations.sum()) == 75
    assert float(violations.mean()) == pytest.approx(
        75 / 1387,
        abs=1e-12,
    )

    if "violation" in baseline.columns:
        assert (
            violations.to_numpy()
            == baseline["violation"].astype(bool).to_numpy()
        ).all()


# B-08.14 — Synthetic runner tests
@pytest.mark.parametrize(
    "window_size",
    [0, -1, 6],
)
def test_invalid_window(window_size):
    with pytest.raises(ValueError):
        walk_forward(
            sample_data(),
            dummy_model,
            window_size=window_size,
        )


# B-08.15 — Model-independent runner test
def test_model_independent_runner():
    result = walk_forward(
        sample_data(),
        dummy_model,
        window_size=3,
        method="dummy",
    )

    assert len(result) == 3
    assert (result["method"] == "dummy").all()


# B-08.16 — Failure-path validation
@pytest.mark.parametrize(
    "case",
    [
        "invalid_mode",
        "empty_data",
        "missing_return",
        "non_numeric",
        "duplicate_date",
        "model_none",
        "model_missing_var",
        "model_nan",
        "model_inf",
    ],
)
def test_failure_paths(case):
    data = sample_data()
    model = dummy_model
    kwargs = {"window_size": 3}

    if case == "invalid_mode":
        kwargs["mode"] = "invalid"

    elif case == "empty_data":
        data = pd.DataFrame()

    elif case == "missing_return":
        data.loc[1, "portfolio_simple_return"] = np.nan

    elif case == "non_numeric":
        data["portfolio_simple_return"] = (
            data["portfolio_simple_return"].astype(object)
        )
        data.loc[0, "portfolio_simple_return"] = "bad"

    elif case == "duplicate_date":
        data.loc[1, "date"] = data.loc[0, "date"]

    elif case == "model_none":
        model = lambda x: None

    elif case == "model_missing_var":
        model = lambda x: {
            "quantile_return": -0.02
        }

    elif case == "model_nan":
        model = lambda x: {
            "quantile_return": np.nan,
            "var": 0.02,
        }

    elif case == "model_inf":
        model = lambda x: {
            "quantile_return": -0.02,
            "var": np.inf,
        }

    with pytest.raises(ValueError):
        walk_forward(
            data,
            model,
            **kwargs,
        )


# B-08.17 — End-to-end smoke test
def test_historical_smoke_end_to_end(tmp_path):
    portfolio_path = Path(
        "data/processed/portfolio_returns.csv"
    )

    if not portfolio_path.exists():
        pytest.skip(
            "Processed portfolio data unavailable."
        )

    data = pd.read_csv(
        portfolio_path
    )

    model = partial(
        historical_var_forecast,
        alpha=0.05,
    )

    predictions = walk_forward(
        data,
        model,
        window_size=250,
        mode="rolling",
        method="historical",
    )

    output_path = (
        tmp_path
        / "historical_predictions.csv"
    )

    save_predictions(
        predictions,
        output_path,
    )

    reloaded = pd.read_csv(
        output_path
    )

    assert len(reloaded) == 1387
    assert output_path.exists()

    assert (
        predictions["forecast_date"]
        < predictions["target_date"]
    ).all()

    assert (
        predictions["quantile_return"]
        .notna()
        .all()
    )


# B-08.18 — Pinball Loss compatibility
def test_pinball_loss_fields_available():
    result = walk_forward(
        sample_data(),
        dummy_model,
        window_size=3,
        method="dummy",
    )

    assert "actual_return" in result.columns
    assert "quantile_return" in result.columns

def test_runner_works_with_ewma_model():
    result = walk_forward(
        sample_data(),
        partial(
            ewma_var_forecast,
            alpha=0.05,
            decay=0.94,
        ),
        window_size=3,
        mode="expanding",
        method="ewma",
    )

    assert len(result) == 3
    assert (result["method"] == "ewma").all()
    assert result["quantile_return"].notna().all()
    assert result["var"].notna().all()
    assert (result["var"] >= 0.0).all()
    assert (
        result["forecast_date"]
        < result["target_date"]
    ).all()


def test_historical_and_ewma_share_schema():
    historical = walk_forward(
        sample_data(),
        partial(
            historical_var_forecast,
            alpha=0.05,
        ),
        window_size=3,
        mode="rolling",
        method="historical",
    )

    ewma = walk_forward(
        sample_data(),
        partial(
            ewma_var_forecast,
            alpha=0.05,
            decay=0.94,
        ),
        window_size=3,
        mode="expanding",
        method="ewma",
    )

    assert historical.columns.tolist() == ewma.columns.tolist()
    assert historical.columns.tolist() == PREDICTION_COLUMNS


def test_ewma_runner_is_deterministic():
    model = partial(
        ewma_var_forecast,
        alpha=0.05,
        decay=0.94,
    )

    first = walk_forward(
        sample_data(),
        model,
        window_size=3,
        mode="expanding",
        method="ewma",
    )

    second = walk_forward(
        sample_data(),
        model,
        window_size=3,
        mode="expanding",
        method="ewma",
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )