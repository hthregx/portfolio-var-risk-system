from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


STOCKS = ("HPG", "FPT", "MWG")

STOCK_REQUIRED_COLUMNS = {
    "date",
    "high",
    "low",
    "close",
    "volume",
}

MARKET_REQUIRED_COLUMNS = {
    "date",
    "close",
}

FEATURE_COLUMNS = [
    "portfolio_range",
    "volume_change",
    "relative_volume_20",
    "market_return_lag_1",
    "market_return_lag_5",
    "market_vol_20",
]


def _prepare_stock_frame(
    frame: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{symbol} input must be a pandas DataFrame.")

    if frame.empty:
        raise ValueError(f"{symbol} input cannot be empty.")

    data = frame.copy()

    missing = sorted(STOCK_REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(
            f"{symbol} is missing required columns: {missing}."
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    ).dt.normalize()

    if data["date"].duplicated().any():
        raise ValueError(f"{symbol} contains duplicate dates.")

    if not data["date"].is_monotonic_increasing:
        raise ValueError(f"{symbol} dates must be sorted.")

    numeric_columns = ["high", "low", "close", "volume"]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    values = data[numeric_columns].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{symbol} contains non-finite OHLCV values."
        )

    if (data["close"] <= 0).any():
        raise ValueError(
            f"{symbol} close values must be positive."
        )

    if (data["high"] < data["low"]).any():
        raise ValueError(
            f"{symbol} contains high < low."
        )

    if (data["volume"] < 0).any():
        raise ValueError(
            f"{symbol} contains negative volume."
        )

    return data.set_index("date")


def _prepare_market_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Market input must be a pandas DataFrame.")

    if frame.empty:
        raise ValueError("Market input cannot be empty.")

    data = frame.copy()

    if "date" not in data.columns and "time" in data.columns:
        data = data.rename(columns={"time": "date"})

    missing = sorted(MARKET_REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(
            f"Market input is missing required columns: {missing}."
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    ).dt.normalize()

    if data["date"].duplicated().any():
        raise ValueError("Market input contains duplicate dates.")

    if not data["date"].is_monotonic_increasing:
        raise ValueError("Market dates must be sorted.")

    data["close"] = pd.to_numeric(
        data["close"],
        errors="raise",
    )

    close_values = data["close"].to_numpy(dtype=float)

    if not np.isfinite(close_values).all():
        raise ValueError(
            "Market input contains non-finite close values."
        )

    if (data["close"] <= 0).any():
        raise ValueError(
            "Market close values must be positive."
        )

    return data.set_index("date")


def _safe_volume_change(
    volume: pd.Series,
) -> pd.Series:
    previous = volume.shift(1)

    result = volume.div(previous).sub(1.0)

    result = result.mask(previous.eq(0))

    return result


def _relative_volume(
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    rolling_mean = volume.rolling(
        window=window,
        min_periods=window,
    ).mean()

    result = volume.div(rolling_mean).sub(1.0)

    result = result.mask(rolling_mean.eq(0))

    return result


def build_market_features(
    stock_frames: Mapping[str, pd.DataFrame],
    market_frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the canonical market/liquidity feature family.

    Output rows are indexed by target date T. Features for T use
    information available only through the immediately preceding
    common trading observation, which acts as forecast date F.

    The canonical calendars use exact-date alignment. Missing market
    observations remain missing; future observations are never used
    as backfill.
    """
    if not isinstance(stock_frames, Mapping):
        raise TypeError(
            "stock_frames must be a mapping of ticker to DataFrame."
        )

    missing_symbols = [
        symbol
        for symbol in STOCKS
        if symbol not in stock_frames
    ]

    if missing_symbols:
        raise ValueError(
            f"Missing required stock inputs: {missing_symbols}."
        )

    stocks = {
        symbol: _prepare_stock_frame(
            stock_frames[symbol],
            symbol,
        )
        for symbol in STOCKS
    }

    market = _prepare_market_frame(market_frame)

    common_index = stocks["HPG"].index

    for symbol in STOCKS[1:]:
        common_index = common_index.intersection(
            stocks[symbol].index
        )

    common_index = common_index.sort_values()

    if common_index.empty:
        raise ValueError(
            "No common stock trading dates are available."
        )

    per_stock_range = []
    per_stock_volume_change = []
    per_stock_relative_volume = []

    for symbol in STOCKS:
        frame = stocks[symbol].reindex(common_index)

        range_feature = (
            frame["high"].sub(frame["low"])
            .div(frame["close"])
        )

        volume_change = _safe_volume_change(
            frame["volume"]
        )

        relative_volume = _relative_volume(
            frame["volume"],
            window=20,
        )

        per_stock_range.append(range_feature)
        per_stock_volume_change.append(volume_change)
        per_stock_relative_volume.append(relative_volume)

    portfolio_range = pd.concat(
        per_stock_range,
        axis=1,
    ).mean(axis=1)

    portfolio_volume_change = pd.concat(
        per_stock_volume_change,
        axis=1,
    ).mean(axis=1)

    portfolio_relative_volume = pd.concat(
        per_stock_relative_volume,
        axis=1,
    ).mean(axis=1)

    market_aligned = market.reindex(common_index)

    market_return = market_aligned["close"].pct_change(
        fill_method=None,
    )

    market_return_lag_1 = market_return
    market_return_lag_5 = market_return.shift(4)

    market_vol_20 = market_return.rolling(
        window=20,
        min_periods=20,
    ).std(ddof=1)

    source_features = pd.DataFrame(
        {
            "portfolio_range": portfolio_range,
            "volume_change": portfolio_volume_change,
            "relative_volume_20": portfolio_relative_volume,
            "market_return_lag_1": market_return_lag_1,
            "market_return_lag_5": market_return_lag_5,
            "market_vol_20": market_vol_20,
        },
        index=common_index,
    )

    # Source row at forecast date F becomes the feature row for
    # the next target date T. This shift prevents target-date
    # OHLCV or VN-Index information from entering its own feature row.
    features = source_features.shift(1)

    features.index.name = "date"

    return features[FEATURE_COLUMNS].astype(float)