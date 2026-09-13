from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.gb_market_features import build_market_features
from src.gb_return_features import build_return_features
from src.validation.final_results_contract import (
    validate_final_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data/sample/final_predictions_contract_sample.csv"


def load_predictions():
    return pd.read_csv(PRED)


def test_valid_temporal_fixture_passes():
    validate_final_predictions(load_predictions())


def test_future_forecast_date_rejected():
    df = load_predictions()
    df.loc[0, "forecast_date"] = df.loc[0, "target_date"]

    with pytest.raises(ValueError, match="forecast_date"):
        validate_final_predictions(df)


def test_actual_return_mismatch_rejected():
    df = load_predictions()

    mask = (
        (df["method"] == "ewma")
        & (df["target_date"] == "2024-01-03")
    )

    df.loc[mask, "actual_return"] = 0.99

    with pytest.raises(ValueError, match="actual_return"):
        validate_final_predictions(df)


def test_missing_target_for_one_method_rejected():
    df = load_predictions()

    mask = (
        (df["method"] == "ewma")
        & (df["target_date"] == "2024-01-05")
    )

    df = df.loc[~mask].copy()

    with pytest.raises(ValueError, match="target-date universe"):
        validate_final_predictions(df)


def test_return_feature_has_no_target_future_leakage():
    dates = pd.bdate_range("2023-01-02", periods=100)

    returns = pd.Series(
        np.linspace(-0.02, 0.02, len(dates)),
        index=dates,
        name="portfolio_simple_return",
    )

    target = dates[70]

    baseline = build_return_features(returns)

    changed = returns.copy()
    changed.loc[target:] += 0.5

    audited = build_return_features(changed)

    pd.testing.assert_series_equal(
        baseline.loc[target],
        audited.loc[target],
    )


def test_market_feature_has_no_target_future_leakage():
    dates = pd.bdate_range("2023-01-02", periods=100)

    stocks = {}

    for i, ticker in enumerate(("HPG", "FPT", "MWG")):
        close = 50 + i * 10 + np.arange(100) * 0.1

        stocks[ticker] = pd.DataFrame({
            "date": dates,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": 1000 + i * 100 + np.arange(100) * 10,
        })

    market = pd.DataFrame({
        "date": dates,
        "close": 1000 + np.arange(100),
    })

    target = dates[70]

    baseline = build_market_features(stocks, market)

    changed_stocks = {
        ticker: frame.copy()
        for ticker, frame in stocks.items()
    }

    for frame in changed_stocks.values():
        mask = frame["date"] >= target
        frame.loc[mask, ["high", "low", "close"]] *= 2
        frame.loc[mask, "volume"] *= 10

    changed_market = market.copy()
    changed_market.loc[
        changed_market["date"] >= target,
        "close",
    ] *= 2

    audited = build_market_features(
        changed_stocks,
        changed_market,
    )

    pd.testing.assert_series_equal(
        baseline.loc[target],
        audited.loc[target],
    )