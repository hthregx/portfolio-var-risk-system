from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.gb_market_features import FEATURE_COLUMNS, build_market_features


def make_stock(periods=40, volume_base=100.0):
    dates = pd.bdate_range("2024-01-01", periods=periods)
    close = np.arange(periods, dtype=float) + 100.0

    return pd.DataFrame({
        "date": dates,
        "high": close + 2.0,
        "low": close - 2.0,
        "close": close,
        "volume": np.arange(periods, dtype=float) + volume_base,
    })


def make_market(periods=40):
    dates = pd.bdate_range("2024-01-01", periods=periods)

    pattern = np.array([
        0.010,
        -0.020,
        0.035,
        0.005,
        -0.015,
        0.025,
        -0.010,
        0.018,
    ])

    returns = np.resize(pattern, periods - 1)

    close = np.empty(periods, dtype=float)
    close[0] = 1000.0
    close[1:] = close[0] * np.cumprod(1.0 + returns)

    return pd.DataFrame({
        "date": dates,
        "close": close,
    })


def make_inputs(periods=40):
    stocks = {
        "HPG": make_stock(periods, 100.0),
        "FPT": make_stock(periods, 200.0),
        "MWG": make_stock(periods, 300.0),
    }

    return stocks, make_market(periods)


def test_required_input_schema():
    stocks, market = make_inputs()
    stocks["HPG"] = stocks["HPG"].drop(columns="volume")

    with pytest.raises(ValueError, match="missing required columns"):
        build_market_features(stocks, market)


def test_required_market_schema():
    stocks, market = make_inputs()
    market = market.drop(columns="close")

    with pytest.raises(ValueError, match="missing required columns"):
        build_market_features(stocks, market)


def test_missing_stock_rejected():
    stocks, market = make_inputs()
    stocks.pop("MWG")

    with pytest.raises(ValueError, match="Missing required stock inputs"):
        build_market_features(stocks, market)


def test_range_arithmetic():
    stocks, market = make_inputs()
    result = build_market_features(stocks, market)

    forecast_pos = 0

    expected = np.mean([
        (df.loc[forecast_pos, "high"] - df.loc[forecast_pos, "low"])
        / df.loc[forecast_pos, "close"]
        for df in stocks.values()
    ])

    assert result.iloc[1]["portfolio_range"] == pytest.approx(expected)


def test_volume_change_arithmetic():
    stocks, market = make_inputs()
    result = build_market_features(stocks, market)

    expected = np.mean([
        df.loc[1, "volume"] / df.loc[0, "volume"] - 1.0
        for df in stocks.values()
    ])

    assert result.iloc[2]["volume_change"] == pytest.approx(expected)


def test_relative_volume_20_arithmetic():
    stocks, market = make_inputs()
    result = build_market_features(stocks, market)

    expected = np.mean([
        df["volume"].iloc[19] / df["volume"].iloc[:20].mean() - 1.0
        for df in stocks.values()
    ])

    assert result.iloc[20]["relative_volume_20"] == pytest.approx(expected)


def test_market_return_lag_1_arithmetic():
    stocks, market = make_inputs()
    result = build_market_features(stocks, market)

    expected = (
        market.loc[1, "close"]
        / market.loc[0, "close"]
        - 1.0
    )

    assert result.iloc[2]["market_return_lag_1"] == pytest.approx(expected)


def test_market_return_lag_5_arithmetic():
    stocks, market = make_inputs()
    result = build_market_features(stocks, market)

    target_position = 6
    forecast_position = target_position - 1
    source_return_position = forecast_position - 4

    expected = (
        market.loc[source_return_position, "close"]
        / market.loc[source_return_position - 1, "close"]
        - 1.0
    )

    assert result.iloc[target_position][
        "market_return_lag_5"
    ] == pytest.approx(expected)

    # Guard against accidentally using the latest market return.
    latest_available_return = (
        market.loc[forecast_position, "close"]
        / market.loc[forecast_position - 1, "close"]
        - 1.0
    )

    assert not np.isclose(expected, latest_available_return)


def test_market_vol_20_arithmetic():
    stocks, market = make_inputs(50)
    result = build_market_features(stocks, market)

    returns = market["close"].pct_change(fill_method=None)

    forecast_position = 20

    expected = returns.iloc[
        forecast_position - 19:
        forecast_position + 1
    ].std(ddof=1)

    assert result.iloc[21]["market_vol_20"] == pytest.approx(expected)


def test_exact_date_alignment_and_shape():
    stocks, market = make_inputs()
    result = build_market_features(stocks, market)

    expected_index = pd.DatetimeIndex(
        stocks["HPG"]["date"],
        name="date",
    )

    pd.testing.assert_index_equal(result.index, expected_index)

    assert result.shape == (40, len(FEATURE_COLUMNS))
    assert result.columns.tolist() == FEATURE_COLUMNS


def test_missing_market_date_no_future_backfill():
    stocks, market = make_inputs()

    missing_date = market.loc[10, "date"]

    market_missing = market.loc[
        ~market["date"].eq(missing_date)
    ].reset_index(drop=True)

    result = build_market_features(stocks, market_missing)

    # Target row 11 uses forecast/source date at position 10.
    # That market date is absent, so a future row must not be substituted.
    assert pd.isna(
        result.iloc[11]["market_return_lag_1"]
    )


def test_finite_output_after_warmup():
    stocks, market = make_inputs(50)
    result = build_market_features(stocks, market)

    assert np.isfinite(
        result.iloc[21:].to_numpy()
    ).all()


def test_missing_volume_rejected():
    stocks, market = make_inputs()

    stocks["HPG"].loc[10, "volume"] = np.nan

    with pytest.raises(ValueError, match="non-finite OHLCV values"):
        build_market_features(stocks, market)


def test_zero_previous_volume_produces_nan():
    stocks, market = make_inputs()

    for frame in stocks.values():
        frame.loc[0, "volume"] = 0.0

    result = build_market_features(stocks, market)

    assert pd.isna(
        result.iloc[2]["volume_change"]
    )


def test_negative_volume_rejected():
    stocks, market = make_inputs()

    stocks["FPT"].loc[5, "volume"] = -1.0

    with pytest.raises(ValueError, match="negative volume"):
        build_market_features(stocks, market)


def test_extreme_volume_preserved_not_clipped():
    stocks, market = make_inputs()

    stocks["HPG"].loc[10, "volume"] = 1_000_000_000.0

    result = build_market_features(stocks, market)

    expected = np.mean([
        df.loc[10, "volume"] / df.loc[9, "volume"] - 1.0
        for df in stocks.values()
    ])

    assert result.iloc[11]["volume_change"] == pytest.approx(expected)


def test_target_date_no_lookahead():
    stocks, market = make_inputs()
    original = build_market_features(stocks, market)

    target_date = original.index[25]

    changed_stocks = {
        symbol: frame.copy()
        for symbol, frame in stocks.items()
    }

    for frame in changed_stocks.values():
        mask = frame["date"].eq(target_date)

        frame.loc[mask, "high"] *= 10.0
        frame.loc[mask, "low"] *= 0.5
        frame.loc[mask, "close"] *= 5.0
        frame.loc[mask, "volume"] *= 100.0

    changed_market = market.copy()
    changed_market.loc[
        changed_market["date"].eq(target_date),
        "close",
    ] *= 10.0

    changed = build_market_features(
        changed_stocks,
        changed_market,
    )

    pd.testing.assert_series_equal(
        original.loc[target_date],
        changed.loc[target_date],
    )


def test_future_perturbation():
    stocks, market = make_inputs(50)
    original = build_market_features(stocks, market)

    boundary = original.index[30]

    changed_stocks = {
        symbol: frame.copy()
        for symbol, frame in stocks.items()
    }

    for frame in changed_stocks.values():
        mask = frame["date"] > boundary

        frame.loc[mask, ["high", "low", "close"]] *= 10.0
        frame.loc[mask, "volume"] *= 100.0

    changed_market = market.copy()
    changed_market.loc[
        changed_market["date"] > boundary,
        "close",
    ] *= 20.0

    changed = build_market_features(
        changed_stocks,
        changed_market,
    )

    pd.testing.assert_frame_equal(
        original.loc[:boundary],
        changed.loc[:boundary],
    )


def test_determinism():
    stocks, market = make_inputs()

    first = build_market_features(stocks, market)
    second = build_market_features(stocks, market)

    pd.testing.assert_frame_equal(first, second)