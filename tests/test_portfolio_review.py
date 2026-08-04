from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.portfolio import (
    calculate_portfolio_return,
    create_equal_weights,
    validate_weights,
)


def test_create_equal_weights_rejects_empty_asset_list() -> None:
    """An equal-weight portfolio requires at least one ticker."""
    with pytest.raises(
        ValueError,
        match="At least one ticker is required",
    ):
        create_equal_weights([])


def test_validate_weights_rejects_total_below_one() -> None:
    """Weights below 100% must be rejected."""
    weights = pd.Series(
        [0.30, 0.30, 0.30],
        index=["HPG", "FPT", "MWG"],
        dtype="float64",
    )

    with pytest.raises(
        ValueError,
        match="Portfolio weights must sum to 1.0",
    ):
        validate_weights(weights)


def test_validate_weights_rejects_nan_weight() -> None:
    """Missing weights must not pass validation."""
    weights = pd.Series(
        [0.50, np.nan, 0.50],
        index=["HPG", "FPT", "MWG"],
        dtype="float64",
    )

    with pytest.raises(
        ValueError,
        match="Portfolio weights cannot contain missing values",
    ):
        validate_weights(weights)


def test_validate_weights_rejects_infinite_weight() -> None:
    """Infinite weights must not pass validation."""
    weights = pd.Series(
        [0.50, np.inf, 0.50],
        index=["HPG", "FPT", "MWG"],
        dtype="float64",
    )

    with pytest.raises(
        ValueError,
        match="Portfolio weights must contain only finite values",
    ):
        validate_weights(weights)


def test_validate_weights_rejects_empty_series() -> None:
    """An empty weight vector is invalid."""
    weights = pd.Series(dtype="float64")

    with pytest.raises(
        ValueError,
        match="Weights cannot be empty",
    ):
        validate_weights(weights)


def test_portfolio_return_preserves_exact_zero() -> None:
    """Opposing equal-weight returns must aggregate to exactly zero."""
    dates = pd.to_datetime(["2026-08-03"])

    returns = pd.DataFrame(
        {
            "HPG": [0.03],
            "FPT": [0.00],
            "MWG": [-0.03],
        },
        index=dates,
        dtype="float64",
    )

    weights = create_equal_weights(
        ["HPG", "FPT", "MWG"]
    )

    result = calculate_portfolio_return(
        returns,
        weights,
    )

    assert result.iloc[0] == pytest.approx(0.0)
    pd.testing.assert_index_equal(
        result.index,
        returns.index,
    )


def test_nan_asset_return_is_not_treated_as_zero() -> None:
    """A missing asset return must raise instead of being silently zero-filled."""
    returns = pd.DataFrame(
        {
            "HPG": [np.nan],
            "FPT": [0.02],
            "MWG": [-0.02],
        },
        index=pd.to_datetime(["2026-08-03"]),
        dtype="float64",
    )

    weights = create_equal_weights(
        ["HPG", "FPT", "MWG"]
    )

    with pytest.raises(
        ValueError,
        match="Asset returns cannot contain missing values",
    ):
        calculate_portfolio_return(
            returns,
            weights,
        )


def test_equal_weight_portfolio_real_data_regression() -> None:
    """Regression test using canonical processed data."""

    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"

    return_series: dict[str, pd.Series] = {}

    for ticker in ["HPG", "FPT", "MWG"]:
        file_path = processed_dir / f"{ticker}_clean.csv"

        assert file_path.exists(), (
            f"Missing processed data file: {file_path}"
        )

        data = pd.read_csv(file_path)

        if "date" in data.columns:
            date_column = "date"
        elif "time" in data.columns:
            date_column = "time"
        else:
            raise AssertionError(
                f"{ticker}_clean.csv must contain 'date' or 'time' column."
            )

        if "close" not in data.columns:
            raise AssertionError(
                f"{ticker}_clean.csv must contain 'close' column."
            )

        data[date_column] = pd.to_datetime(
            data[date_column],
            errors="raise",
        )

        return_series[ticker] = (
            data
            .sort_values(date_column)
            .set_index(date_column)["close"]
            .pct_change(fill_method=None)
            .rename(ticker)
        )

    aligned_returns = (
        pd.concat(return_series, axis=1)
        .dropna()
        .sort_index()
    )

    weights = create_equal_weights(
        ["HPG", "FPT", "MWG"]
    )

    result = calculate_portfolio_return(
        aligned_returns,
        weights,
    )

    # Regression invariants
    assert len(result) == 1637

    assert result.min() == pytest.approx(
        -0.06983986,
        abs=1e-6,
    )

    assert result.max() == pytest.approx(
        0.06887777,
        abs=1e-6,
    )