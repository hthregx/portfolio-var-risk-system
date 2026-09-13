from __future__ import annotations

import numpy as np
import pandas as pd


PORTFOLIO_RETURN_NAME = "portfolio_simple_return"
RETURN_LAGS = (1, 2, 5)
VOLATILITY_WINDOWS = (5, 20, 60)


def _validate_portfolio_return(
    portfolio_return: pd.Series,
) -> pd.Series:
    """Validate the canonical portfolio simple-return series."""
    if not isinstance(portfolio_return, pd.Series):
        raise ValueError(
            "Portfolio returns must be provided as a pandas Series."
        )

    if portfolio_return.empty:
        raise ValueError(
            "Portfolio returns cannot be empty."
        )

    if portfolio_return.name != PORTFOLIO_RETURN_NAME:
        raise ValueError(
            "Portfolio return series must be named "
            f"'{PORTFOLIO_RETURN_NAME}'."
        )

    if not isinstance(
        portfolio_return.index,
        pd.DatetimeIndex,
    ):
        raise ValueError(
            "Portfolio returns must use a DatetimeIndex."
        )

    if portfolio_return.index.hasnans:
        raise ValueError(
            "Portfolio return index cannot contain missing dates."
        )

    if portfolio_return.index.has_duplicates:
        raise ValueError(
            "Portfolio return dates must be unique."
        )

    if not portfolio_return.index.is_monotonic_increasing:
        raise ValueError(
            "Portfolio return dates must be sorted "
            "in increasing order."
        )

    try:
        numeric_returns = pd.to_numeric(
            portfolio_return,
            errors="raise",
        ).astype("float64")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Portfolio returns must be numeric."
        ) from error

    if numeric_returns.isna().any():
        raise ValueError(
            "Portfolio returns cannot contain missing values."
        )

    if not np.isfinite(
        numeric_returns.to_numpy()
    ).all():
        raise ValueError(
            "Portfolio returns must contain only finite values."
        )

    if (numeric_returns <= -1.0).any():
        raise ValueError(
            "Portfolio simple returns must be greater than -1.0 "
            "to construct a valid wealth index."
        )

    return pd.Series(
        numeric_returns.to_numpy(copy=True),
        index=portfolio_return.index.copy(),
        name=PORTFOLIO_RETURN_NAME,
        dtype="float64",
    )


def build_return_features(
    portfolio_return: pd.Series,
) -> pd.DataFrame:
    """
    Build target-date-aligned return-history features.

    Each output row is indexed by a target date. Features for that
    target use only portfolio returns observed strictly before the
    target date. Warm-up observations remain missing by design.
    """
    validated_returns = _validate_portfolio_return(
        portfolio_return
    )

    features = pd.DataFrame(
        index=validated_returns.index.copy()
    )

    for lag in RETURN_LAGS:
        features[f"return_lag_{lag}"] = (
            validated_returns.shift(lag)
        )

    available_history = validated_returns.shift(1)

    for window in VOLATILITY_WINDOWS:
        features[f"rolling_vol_{window}"] = (
            available_history
            .rolling(
                window=window,
                min_periods=window,
            )
            .std(ddof=1)
        )

    wealth_index = (
        1.0 + validated_returns
    ).cumprod()

    running_peak = wealth_index.cummax()

    contemporaneous_drawdown = (
        wealth_index
        .div(running_peak)
        .sub(1.0)
    )

    features["drawdown"] = (
        contemporaneous_drawdown.shift(1)
    )

    return features.astype("float64")
