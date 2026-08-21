from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = REPO_ROOT / "results" / "final_predictions.csv"
METRICS_PATH = REPO_ROOT / "results" / "final_metrics.csv"
OUTPUT_PATH = REPO_ROOT / "results" / "final_metric_comparison.csv"

ALPHA = 0.05

METHOD_ORDER = [
    "historical_simulation",
    "ewma",
    "gradient_boosting",
]

OUTPUT_COLUMNS = [
    "method",
    "forecast_count",
    "violation_count",
    "violation_rate",
    "nominal_violation_rate",
    "calibration_distance",
    "pinball_loss",
    "average_var",
    "minimum_var",
    "maximum_var",
    "total_runtime_seconds",
    "test_start",
    "test_end",
    "config_id",
    "calibration_leader",
    "pinball_leader",
    "lowest_average_var",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def pinball_loss(
    actual: np.ndarray,
    predicted_quantile: np.ndarray,
    alpha: float,
) -> float:
    """Return mean quantile / Pinball Loss."""

    actual_values = np.asarray(actual, dtype="float64")
    predicted_values = np.asarray(
        predicted_quantile,
        dtype="float64",
    )

    require(
        actual_values.shape == predicted_values.shape,
        "Actual and predicted arrays must share the same shape.",
    )

    require(
        np.isfinite(actual_values).all()
        and np.isfinite(predicted_values).all(),
        "Pinball inputs must be finite.",
    )

    error = actual_values - predicted_values

    loss = np.maximum(
        alpha * error,
        (alpha - 1.0) * error,
    )

    return float(loss.mean())


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and minimally validate canonical Day-21 artifacts."""

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["forecast_date", "target_date"],
    )

    metrics = pd.read_csv(
        METRICS_PATH,
        parse_dates=["test_start", "test_end"],
    )

    require(
        len(predictions) == 1194,
        f"Expected 1194 predictions, found {len(predictions)}.",
    )

    require(
        len(metrics) == 3,
        f"Expected 3 metric rows, found {len(metrics)}.",
    )

    require(
        set(predictions["method"]) == set(METHOD_ORDER),
        "Prediction method set is incorrect.",
    )

    require(
        set(metrics["method"]) == set(METHOD_ORDER),
        "Metric method set is incorrect.",
    )

    require(
        not predictions[
            ["method", "target_date"]
        ].duplicated().any(),
        "Duplicate method/target-date pairs found.",
    )

    return predictions, metrics


def recompute_method_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute evaluation metrics directly from prediction rows."""

    records: list[dict] = []

    for method in METHOD_ORDER:
        frame = (
            predictions.loc[
                predictions["method"] == method
            ]
            .sort_values("target_date")
            .reset_index(drop=True)
        )

        require(
            len(frame) == 398,
            f"{method}: expected 398 forecasts.",
        )

        actual = frame["actual_return"].to_numpy(
            dtype="float64"
        )

        quantile = frame["quantile_return"].to_numpy(
            dtype="float64"
        )

        var = frame["var"].to_numpy(
            dtype="float64"
        )

        violation = frame["violation"].to_numpy(
            dtype=bool
        )

        expected_violation = actual < quantile

        require(
            np.array_equal(
                violation,
                expected_violation,
            ),
            f"{method}: violation semantics changed.",
        )

        runtime_values = frame[
            "runtime_seconds"
        ].to_numpy(dtype="float64")

        require(
            np.isfinite(runtime_values).all(),
            f"{method}: runtime is non-finite.",
        )

        require(
            np.allclose(
                runtime_values,
                runtime_values[0],
                rtol=0.0,
                atol=1e-15,
            ),
            f"{method}: runtime is not method-level constant.",
        )

        config_ids = frame[
            "config_id"
        ].astype(str).unique()

        require(
            len(config_ids) == 1,
            f"{method}: expected one config_id.",
        )

        violation_count = int(
            violation.sum()
        )

        violation_rate = float(
            violation.mean()
        )

        records.append(
            {
                "method": method,
                "forecast_count": int(len(frame)),
                "violation_count": violation_count,
                "violation_rate": violation_rate,
                "nominal_violation_rate": ALPHA,
                "calibration_distance": abs(
                    violation_rate - ALPHA
                ),
                "pinball_loss": pinball_loss(
                    actual,
                    quantile,
                    ALPHA,
                ),
                "average_var": float(var.mean()),
                "minimum_var": float(var.min()),
                "maximum_var": float(var.max()),
                "total_runtime_seconds": float(
                    runtime_values[0]
                ),
                "test_start": frame[
                    "target_date"
                ].iloc[0],
                "test_end": frame[
                    "target_date"
                ].iloc[-1],
                "config_id": config_ids[0],
            }
        )

    return pd.DataFrame(records)


def validate_against_canonical_metrics(
    recomputed: pd.DataFrame,
    canonical: pd.DataFrame,
) -> None:
    """Require independent recomputation to match final_metrics.csv."""

    left = recomputed.set_index("method")
    right = canonical.set_index("method")

    integer_columns = [
        "forecast_count",
        "violation_count",
    ]

    float_columns = [
        "violation_rate",
        "pinball_loss",
        "average_var",
        "minimum_var",
        "maximum_var",
        "total_runtime_seconds",
    ]

    for method in METHOD_ORDER:
        for column in integer_columns:
            require(
                int(left.loc[method, column])
                == int(right.loc[method, column]),
                f"{method}: {column} mismatch.",
            )

        for column in float_columns:
            require(
                np.isclose(
                    float(left.loc[method, column]),
                    float(right.loc[method, column]),
                    rtol=0.0,
                    atol=1e-15,
                ),
                f"{method}: {column} mismatch.",
            )

        require(
            pd.Timestamp(
                left.loc[method, "test_start"]
            )
            == pd.Timestamp(
                right.loc[method, "test_start"]
            ),
            f"{method}: test_start mismatch.",
        )

        require(
            pd.Timestamp(
                left.loc[method, "test_end"]
            )
            == pd.Timestamp(
                right.loc[method, "test_end"]
            ),
            f"{method}: test_end mismatch.",
        )

        require(
            str(left.loc[method, "config_id"])
            == str(right.loc[method, "config_id"]),
            f"{method}: config_id mismatch.",
        )


def add_criterion_flags(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Mark criterion leaders without creating an overall model ranking."""

    frame = comparison.copy()

    calibration_best = float(
        frame["calibration_distance"].min()
    )

    pinball_best = float(
        frame["pinball_loss"].min()
    )

    average_var_lowest = float(
        frame["average_var"].min()
    )

    frame["calibration_leader"] = np.isclose(
        frame["calibration_distance"],
        calibration_best,
        rtol=0.0,
        atol=1e-15,
    )

    frame["pinball_leader"] = np.isclose(
        frame["pinball_loss"],
        pinball_best,
        rtol=0.0,
        atol=1e-15,
    )

    frame["lowest_average_var"] = np.isclose(
        frame["average_var"],
        average_var_lowest,
        rtol=0.0,
        atol=1e-15,
    )

    return frame


def build_comparison() -> pd.DataFrame:
    """Build the deterministic Day-22 core metric comparison."""

    predictions, canonical_metrics = load_inputs()

    comparison = recompute_method_metrics(
        predictions
    )

    validate_against_canonical_metrics(
        comparison,
        canonical_metrics,
    )

    comparison = add_criterion_flags(
        comparison
    )

    comparison = comparison[
        OUTPUT_COLUMNS
    ]

    return comparison


def write_comparison(
    comparison: pd.DataFrame,
) -> None:
    """Write the analysis result after all validation passes."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )


def print_summary(
    comparison: pd.DataFrame,
) -> None:
    """Print criterion-specific evidence without declaring a winner."""

    print("")
    print("=== DAY 22 A CORE METRIC COMPARISON ===")
    print("")

    display_columns = [
        "method",
        "violation_count",
        "violation_rate",
        "calibration_distance",
        "pinball_loss",
        "average_var",
    ]

    print(
        comparison[
            display_columns
        ].to_string(index=False)
    )

    calibration_method = comparison.loc[
        comparison["calibration_leader"],
        "method",
    ].iloc[0]

    pinball_method = comparison.loc[
        comparison["pinball_leader"],
        "method",
    ].iloc[0]

    lowest_var_method = comparison.loc[
        comparison["lowest_average_var"],
        "method",
    ].iloc[0]

    print("")
    print("[CRITERION LEADERS]")
    print(
        "Closest to 5% violation rate :",
        calibration_method,
    )
    print(
        "Lowest Pinball Loss          :",
        pinball_method,
    )
    print(
        "Lowest Average VaR           :",
        lowest_var_method,
    )
    print("")
    print(
        "Overall model winner declared: NO"
    )


def main() -> None:
    comparison = build_comparison()

    write_comparison(comparison)

    print_summary(comparison)

    print("")
    print(
        f"Wrote: {OUTPUT_PATH.relative_to(REPO_ROOT)}"
    )
    print(
        "PASS - DAY 22 A2 CORE METRIC COMPARISON COMPLETE"
    )


if __name__ == "__main__":
    main()
