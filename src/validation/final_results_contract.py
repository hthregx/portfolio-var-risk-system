from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)


METHODS = {
    "historical_simulation",
    "ewma",
    "gradient_boosting",
}

PREDICTION_COLUMNS = [
    "forecast_date",
    "target_date",
    "method",
    "actual_return",
    "quantile_return",
    "var",
    "violation",
    "runtime_seconds",
    "config_id",
]

METRIC_COLUMNS = [
    "method",
    "forecast_count",
    "violation_count",
    "violation_rate",
    "pinball_loss",
    "average_var",
    "minimum_var",
    "maximum_var",
    "total_runtime_seconds",
    "test_start",
    "test_end",
    "config_id",
]


def _base_check(df, columns, name):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"{name} must be a non-empty DataFrame.")

    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} missing required columns: {sorted(missing)}"
        )

    if df[columns].isna().any().any():
        raise ValueError("Missing values are not allowed.")


def _numeric(df, columns):
    for column in columns:
        if not is_numeric_dtype(df[column]):
            raise ValueError(
                f"{column} must have numeric dtype."
            )

        if not np.isfinite(
            df[column].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{column} must contain only finite values."
            )


def _methods(df):
    if set(df["method"]) != METHODS:
        raise ValueError(
            f"Method values must be exactly {sorted(METHODS)}."
        )


def validate_final_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    _base_check(
        predictions,
        PREDICTION_COLUMNS,
        "Predictions",
    )

    df = predictions.copy()

    _numeric(
        df,
        [
            "actual_return",
            "quantile_return",
            "var",
            "runtime_seconds",
        ],
    )

    if not is_bool_dtype(df["violation"]):
        raise ValueError(
            "violation must have boolean dtype."
        )

    _methods(df)

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"],
        errors="raise",
    )
    df["target_date"] = pd.to_datetime(
        df["target_date"],
        errors="raise",
    )

    if not (
        df["forecast_date"] < df["target_date"]
    ).all():
        raise ValueError(
            "forecast_date must be before target_date."
        )

    if df[["method", "target_date"]].duplicated().any():
        raise ValueError(
            "Duplicate method/target_date key."
        )

    if (df["runtime_seconds"] < 0).any():
        raise ValueError(
            "runtime_seconds cannot be negative."
        )

    expected_var = np.maximum(
        0.0,
        -df["quantile_return"].to_numpy(dtype=float),
    )

    if not np.allclose(
        df["var"],
        expected_var,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError(
            "VaR sign convention violated."
        )

    expected_violation = (
        df["actual_return"].to_numpy(dtype=float)
        < df["quantile_return"].to_numpy(dtype=float)
    )

    if not np.array_equal(
        df["violation"].to_numpy(),
        expected_violation,
    ):
        raise ValueError(
            "Strict violation rule violated."
        )

    universes = [
        set(group["target_date"])
        for _, group in df.groupby("method")
    ]

    if any(
        universe != universes[0]
        for universe in universes[1:]
    ):
        raise ValueError(
            "Methods must share the same target-date universe."
        )

    if (
        df.groupby("target_date")["actual_return"]
        .nunique()
        .ne(1)
        .any()
    ):
        raise ValueError(
            "actual_return must match across methods."
        )

    return df[PREDICTION_COLUMNS].copy()


def validate_final_metrics(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    _base_check(
        metrics,
        METRIC_COLUMNS,
        "Metrics",
    )

    df = metrics.copy()

    for column in (
        "forecast_count",
        "violation_count",
    ):
        if not is_integer_dtype(df[column]):
            raise ValueError(
                f"{column} must have integer dtype."
            )

    _numeric(
        df,
        [
            "forecast_count",
            "violation_count",
            "violation_rate",
            "pinball_loss",
            "average_var",
            "minimum_var",
            "maximum_var",
            "total_runtime_seconds",
        ],
    )

    _methods(df)

    if df["method"].duplicated().any():
        raise ValueError(
            "Metrics contain duplicate methods."
        )

    df["test_start"] = pd.to_datetime(
        df["test_start"],
        errors="raise",
    )
    df["test_end"] = pd.to_datetime(
        df["test_end"],
        errors="raise",
    )

    if (df["test_start"] > df["test_end"]).any():
        raise ValueError(
            "test_start cannot be after test_end."
        )

    if (
        (df["forecast_count"] <= 0)
        | (df["violation_count"] < 0)
        | (
            df["violation_count"]
            > df["forecast_count"]
        )
    ).any():
        raise ValueError(
            "Invalid forecast/violation counts."
        )

    expected_rate = (
        df["violation_count"]
        / df["forecast_count"]
    )

    if not np.allclose(
        df["violation_rate"],
        expected_rate,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError(
            "violation_rate does not match counts."
        )

    if (
        df[
            [
                "average_var",
                "minimum_var",
                "maximum_var",
                "total_runtime_seconds",
            ]
        ]
        < 0
    ).any().any():
        raise ValueError(
            "VaR/runtime metrics cannot be negative."
        )

    if (
        (df["minimum_var"] > df["maximum_var"])
        | (df["average_var"] < df["minimum_var"])
        | (df["average_var"] > df["maximum_var"])
    ).any():
        raise ValueError(
            "Invalid VaR summary range."
        )

    return df[METRIC_COLUMNS].copy()