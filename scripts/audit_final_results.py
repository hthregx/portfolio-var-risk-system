from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.validation.final_results_contract import (
    METRIC_COLUMNS,
    validate_final_metrics,
    validate_final_predictions,
)


DEFAULT_ALPHA = 0.05


def pinball_loss(actual, quantile, alpha=DEFAULT_ALPHA):
    alpha = float(alpha)

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    actual = np.asarray(actual, dtype=float)
    quantile = np.asarray(quantile, dtype=float)

    if actual.ndim != 1 or actual.shape != quantile.shape:
        raise ValueError(
            "actual and quantile must be matching 1D arrays."
        )

    if (
        not np.isfinite(actual).all()
        or not np.isfinite(quantile).all()
    ):
        raise ValueError("Pinball inputs must be finite.")

    error = actual - quantile

    return float(
        np.maximum(
            alpha * error,
            (alpha - 1.0) * error,
        ).mean()
    )


def recompute_metrics(
    predictions,
    alpha=DEFAULT_ALPHA,
):
    frame = validate_final_predictions(predictions)
    records = []

    for method, part in frame.groupby(
        "method",
        sort=False,
    ):
        part = part.sort_values("target_date")

        actual = part[
            "actual_return"
        ].to_numpy(dtype=float)

        quantile = part[
            "quantile_return"
        ].to_numpy(dtype=float)

        var = part[
            "var"
        ].to_numpy(dtype=float)

        violation = actual < quantile

        config_ids = (
            part["config_id"]
            .astype(str)
            .unique()
        )

        if len(config_ids) != 1:
            raise ValueError(
                f"{method} must have exactly one config_id."
            )

        runtime_values = part[
            "runtime_seconds"
        ].to_numpy(dtype=float)

        if not np.allclose(
            runtime_values,
            runtime_values[0],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{method} must have one consistent "
                "runtime_seconds value."
            )

        records.append({
            "method": method,
            "forecast_count": int(len(part)),
            "violation_count": int(
                violation.sum()
            ),
            "violation_rate": float(
                violation.mean()
            ),
            "pinball_loss": pinball_loss(
                actual,
                quantile,
                alpha=alpha,
            ),
            "average_var": float(
                var.mean()
            ),
            "minimum_var": float(
                var.min()
            ),
            "maximum_var": float(
                var.max()
            ),
            "total_runtime_seconds": float(
                runtime_values[0]
            ),
            "test_start": part[
                "target_date"
            ].iloc[0],
            "test_end": part[
                "target_date"
            ].iloc[-1],
            "config_id": config_ids[0],
        })

    return pd.DataFrame(
        records,
        columns=METRIC_COLUMNS,
    )


def compare_metrics(
    recomputed,
    reported,
    *,
    atol=1e-12,
):
    expected = (
        validate_final_metrics(recomputed)
        .sort_values("method")
        .reset_index(drop=True)
    )

    actual = (
        validate_final_metrics(reported)
        .sort_values("method")
        .reset_index(drop=True)
    )

    if expected["method"].tolist() != actual[
        "method"
    ].tolist():
        raise ValueError(
            "Metric methods do not match."
        )

    for column in (
        "forecast_count",
        "violation_count",
        "config_id",
    ):
        if not expected[column].equals(
            actual[column]
        ):
            raise ValueError(
                f"Metric mismatch in {column}."
            )

    for column in (
        "test_start",
        "test_end",
    ):
        if not pd.to_datetime(
            expected[column]
        ).equals(
            pd.to_datetime(actual[column])
        ):
            raise ValueError(
                f"Metric mismatch in {column}."
            )

    for column in (
        "violation_rate",
        "pinball_loss",
        "average_var",
        "minimum_var",
        "maximum_var",
        "total_runtime_seconds",
    ):
        if not np.allclose(
            expected[column].to_numpy(
                dtype=float
            ),
            actual[column].to_numpy(
                dtype=float
            ),
            rtol=0.0,
            atol=atol,
        ):
            raise ValueError(
                f"Metric mismatch in {column}."
            )

    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--metrics",
        type=Path,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
    )

    args = parser.parse_args()

    predictions = pd.read_csv(
        args.predictions
    )

    recomputed = recompute_metrics(
        predictions,
        alpha=args.alpha,
    )

    print(
        recomputed.to_string(
            index=False
        )
    )

    if args.metrics:
        reported = pd.read_csv(
            args.metrics
        )

        compare_metrics(
            recomputed,
            reported,
        )

        print(
            "reported metrics comparison: PASS"
        )

    print(
        "independent final-results audit: PASS"
    )


if __name__ == "__main__":
    main()