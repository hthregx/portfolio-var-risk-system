from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


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


def _validate_alpha(alpha: float) -> float:
    """Validate and normalize the quantile level."""
    if isinstance(alpha, bool) or not isinstance(
        alpha,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError("Alpha must be numeric.")

    value = float(alpha)

    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(
            "Alpha must be finite and strictly between 0 and 1."
        )

    return value


def pinball_loss(
    actual_return,
    quantile_return,
    alpha: float = 0.05,
) -> float:
    """Return mean quantile Pinball Loss."""
    alpha_value = _validate_alpha(alpha)

    try:
        actual = np.asarray(
            actual_return,
            dtype="float64",
        )
        predicted = np.asarray(
            quantile_return,
            dtype="float64",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Actual and predicted returns must be numeric."
        ) from error

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError(
            "Actual and predicted returns must be one-dimensional."
        )

    if actual.size == 0:
        raise ValueError(
            "Pinball Loss requires at least one observation."
        )

    if actual.shape != predicted.shape:
        raise ValueError(
            "Actual and predicted return shapes must match."
        )

    if not np.isfinite(actual).all():
        raise ValueError(
            "Actual returns must contain only finite values."
        )

    if not np.isfinite(predicted).all():
        raise ValueError(
            "Predicted quantiles must contain only finite values."
        )

    error = actual - predicted

    losses = np.maximum(
        alpha_value * error,
        (alpha_value - 1.0) * error,
    )

    return float(losses.mean())


def _validate_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the prediction table used for final metrics."""
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError(
            "Predictions must be provided as a pandas DataFrame."
        )

    if predictions.empty:
        raise ValueError(
            "Predictions cannot be empty."
        )

    required_columns = {
        "target_date",
        "actual_return",
        "quantile_return",
        "var",
        "violation",
        "method",
    }

    missing_columns = required_columns - set(predictions.columns)

    if missing_columns:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing_columns)}"
        )

    frame = predictions.copy()

    frame["target_date"] = pd.to_datetime(
        frame["target_date"],
        errors="raise",
    )

    if frame[["method", "target_date"]].duplicated().any():
        raise ValueError(
            "Predictions contain duplicate method/target-date pairs."
        )

    for column in (
        "actual_return",
        "quantile_return",
        "var",
    ):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        ).astype("float64")

        if not np.isfinite(
            frame[column].to_numpy()
        ).all():
            raise ValueError(
                f"{column} must contain only finite values."
            )

    if (frame["var"] < 0.0).any():
        raise ValueError(
            "VaR values cannot be negative."
        )

    expected_var = np.maximum(
        0.0,
        -frame["quantile_return"].to_numpy(dtype="float64"),
    )

    if not np.allclose(
        frame["var"].to_numpy(dtype="float64"),
        expected_var,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError(
            "Prediction table violates the VaR sign convention."
        )

    expected_violation = (
        frame["actual_return"].to_numpy(dtype="float64")
        <
        frame["quantile_return"].to_numpy(dtype="float64")
    )

    actual_violation = frame["violation"].to_numpy(dtype=bool)

    if not np.array_equal(
        actual_violation,
        expected_violation,
    ):
        raise ValueError(
            "Prediction table violates the strict violation rule."
        )

    if frame["method"].isna().any():
        raise ValueError(
            "Method labels cannot be missing."
        )

    if (
        frame["method"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError(
            "Method labels cannot be empty."
        )

    return frame


def compute_final_metrics(
    predictions: pd.DataFrame,
    *,
    runtimes: Mapping[str, float],
    config_ids: Mapping[str, str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute one final metric row per forecasting method."""
    alpha_value = _validate_alpha(alpha)
    frame = _validate_predictions(predictions)

    methods = list(
        dict.fromkeys(
            frame["method"].astype(str)
        )
    )

    if set(runtimes) != set(methods):
        raise ValueError(
            "Runtime keys must exactly match prediction methods."
        )

    if set(config_ids) != set(methods):
        raise ValueError(
            "Config ID keys must exactly match prediction methods."
        )

    records = []

    for method in methods:
        method_frame = (
            frame.loc[
                frame["method"].astype(str) == method
            ]
            .sort_values("target_date")
            .reset_index(drop=True)
        )

        runtime = float(runtimes[method])

        if not np.isfinite(runtime) or runtime <= 0.0:
            raise ValueError(
                f"Runtime for {method} must be positive and finite."
            )

        config_id = str(config_ids[method]).strip()

        if not config_id:
            raise ValueError(
                f"Config ID for {method} cannot be empty."
            )

        actual = method_frame[
            "actual_return"
        ].to_numpy(dtype="float64")

        predicted = method_frame[
            "quantile_return"
        ].to_numpy(dtype="float64")

        var_values = method_frame[
            "var"
        ].to_numpy(dtype="float64")

        violation_values = method_frame[
            "violation"
        ].to_numpy(dtype=bool)

        records.append(
            {
                "method": method,
                "forecast_count": int(len(method_frame)),
                "violation_count": int(
                    violation_values.sum()
                ),
                "violation_rate": float(
                    violation_values.mean()
                ),
                "pinball_loss": pinball_loss(
                    actual,
                    predicted,
                    alpha=alpha_value,
                ),
                "average_var": float(
                    var_values.mean()
                ),
                "minimum_var": float(
                    var_values.min()
                ),
                "maximum_var": float(
                    var_values.max()
                ),
                "total_runtime_seconds": runtime,
                "test_start": method_frame[
                    "target_date"
                ].iloc[0],
                "test_end": method_frame[
                    "target_date"
                ].iloc[-1],
                "config_id": config_id,
            }
        )

    result = pd.DataFrame(
        records,
        columns=METRIC_COLUMNS,
    )

    return result


__all__ = [
    "METRIC_COLUMNS",
    "compute_final_metrics",
    "pinball_loss",
]
