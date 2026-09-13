from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def create_equal_weights(tickers: Sequence[str]) -> pd.Series:
    """
    Create equal portfolio weights for a sequence of ticker symbols.
    """
    if isinstance(tickers, (str, bytes)):
        raise ValueError(
            "Tickers must be provided as a sequence of strings."
        )

    if len(tickers) == 0:
        raise ValueError(
            "At least one ticker is required."
        )

    normalized_tickers = []

    for ticker in tickers:
        if not isinstance(ticker, str):
            raise ValueError(
                "Each ticker must be a string."
            )

        normalized_ticker = ticker.strip().upper()

        if not normalized_ticker:
            raise ValueError(
                "Ticker symbols cannot be empty."
            )

        normalized_tickers.append(
            normalized_ticker
        )

    if len(set(normalized_tickers)) != len(normalized_tickers):
        raise ValueError(
            "Ticker symbols must be unique."
        )

    equal_weight = 1.0 / len(
        normalized_tickers
    )

    weights = pd.Series(
        equal_weight,
        index=normalized_tickers,
        name="weight",
        dtype="float64",
    )

    return weights


def validate_weights(
    weights: pd.Series,
    tolerance: float = 1e-12,
) -> pd.Series:
    """
    Validate and normalize a long-only portfolio weight vector.
    """
    if not isinstance(weights, pd.Series):
        raise ValueError(
            "Weights must be provided as a pandas Series."
        )

    if weights.empty:
        raise ValueError(
            "Weights cannot be empty."
        )

    if not isinstance(tolerance, (int, float)):
        raise ValueError(
            "Tolerance must be numeric."
        )

    tolerance = float(tolerance)

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(
            "Tolerance must be finite and greater than zero."
        )

    normalized_tickers = []

    for ticker in weights.index:
        if not isinstance(ticker, str):
            raise ValueError(
                "Each weight index must be a ticker string."
            )

        normalized_ticker = ticker.strip().upper()

        if not normalized_ticker:
            raise ValueError(
                "Ticker symbols cannot be empty."
            )

        normalized_tickers.append(
            normalized_ticker
        )

    if len(set(normalized_tickers)) != len(normalized_tickers):
        raise ValueError(
            "Ticker symbols must be unique."
        )

    try:
        numeric_weights = pd.to_numeric(
            weights,
            errors="raise",
        ).astype("float64")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "All portfolio weights must be numeric."
        ) from error

    if numeric_weights.isna().any():
        raise ValueError(
            "Portfolio weights cannot contain missing values."
        )

    if not np.isfinite(
        numeric_weights.to_numpy()
    ).all():
        raise ValueError(
            "Portfolio weights must contain only finite values."
        )

    if (numeric_weights < 0.0).any():
        raise ValueError(
            "Portfolio weights cannot be negative."
        )

    total_weight = float(
        numeric_weights.sum()
    )

    if not np.isclose(
        total_weight,
        1.0,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(
            "Portfolio weights must sum to 1.0 "
            f"within tolerance {tolerance}. "
            f"Received total: {total_weight}."
        )

    validated_weights = pd.Series(
        numeric_weights.to_numpy(),
        index=normalized_tickers,
        name="weight",
        dtype="float64",
    )

    return validated_weights


def calculate_portfolio_return(
    returns: pd.DataFrame,
    weights: pd.Series,
    tolerance: float = 1e-12,
) -> pd.Series:
    """
    Calculate portfolio simple returns from aligned asset returns and weights.
    """
    if not isinstance(returns, pd.DataFrame):
        raise ValueError(
            "Returns must be provided as a pandas DataFrame."
        )

    if returns.empty:
        raise ValueError(
            "Returns cannot be empty."
        )

    normalized_columns = []

    for ticker in returns.columns:
        if not isinstance(ticker, str):
            raise ValueError(
                "Each return column must be a ticker string."
            )

        normalized_ticker = ticker.strip().upper()

        if not normalized_ticker:
            raise ValueError(
                "Ticker symbols cannot be empty."
            )

        normalized_columns.append(
            normalized_ticker
        )

    if len(set(normalized_columns)) != len(normalized_columns):
        raise ValueError(
            "Return ticker columns must be unique."
        )

    validated_weights = validate_weights(
        weights,
        tolerance=tolerance,
    )

    return_tickers = set(
        normalized_columns
    )

    weight_tickers = set(
        validated_weights.index
    )

    if return_tickers != weight_tickers:
        missing_returns = sorted(
            weight_tickers - return_tickers
        )

        missing_weights = sorted(
            return_tickers - weight_tickers
        )

        raise ValueError(
            "Return tickers and weight tickers must match exactly. "
            f"Missing return columns: {missing_returns}. "
            f"Missing weights: {missing_weights}."
        )

    normalized_returns = returns.copy()

    normalized_returns.columns = (
        normalized_columns
    )

    try:
        numeric_returns = (
            normalized_returns
            .apply(
                pd.to_numeric,
                errors="raise",
            )
            .astype("float64")
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "All asset returns must be numeric."
        ) from error

    if numeric_returns.isna().any().any():
        raise ValueError(
            "Asset returns cannot contain missing values."
        )

    if not np.isfinite(
        numeric_returns.to_numpy()
    ).all():
        raise ValueError(
            "Asset returns must contain only finite values."
        )

    aligned_returns = numeric_returns.loc[
        :,
        validated_weights.index,
    ]

    portfolio_return = (
        aligned_returns
        .mul(
            validated_weights,
            axis="columns",
        )
        .sum(axis="columns")
        .astype("float64")
    )

    portfolio_return.name = (
        "portfolio_simple_return"
    )

    return portfolio_return


def calculate_portfolio_log_return(
    portfolio_return: pd.Series,
) -> pd.Series:
    """
    Convert portfolio simple returns to exact portfolio log returns.
    """
    if not isinstance(portfolio_return, pd.Series):
        raise ValueError(
            "Portfolio returns must be provided as a pandas Series."
        )

    if portfolio_return.empty:
        raise ValueError(
            "Portfolio returns cannot be empty."
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
            "to compute log returns."
        )

    portfolio_log_return = pd.Series(
        np.log1p(
            numeric_returns.to_numpy()
        ),
        index=numeric_returns.index,
        name="portfolio_log_return",
        dtype="float64",
    )

    return portfolio_log_return