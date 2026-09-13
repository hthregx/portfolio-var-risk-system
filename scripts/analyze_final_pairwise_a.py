from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    REPO_ROOT
    / "results"
    / "final_predictions.csv"
)

METRIC_COMPARISON_PATH = (
    REPO_ROOT
    / "results"
    / "final_metric_comparison.csv"
)

OUTPUT_PATH = (
    REPO_ROOT
    / "results"
    / "final_pairwise_diagnostics.csv"
)

ALPHA = 0.05

METHODS = [
    "historical_simulation",
    "ewma",
    "gradient_boosting",
]

METHOD_PREFIX = {
    "historical_simulation": "historical",
    "ewma": "ewma",
    "gradient_boosting": "gb",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def observation_pinball_loss(
    actual: np.ndarray,
    quantile: np.ndarray,
    alpha: float = ALPHA,
) -> np.ndarray:
    """Return observation-level quantile loss."""

    actual_values = np.asarray(
        actual,
        dtype="float64",
    )

    quantile_values = np.asarray(
        quantile,
        dtype="float64",
    )

    require(
        actual_values.shape == quantile_values.shape,
        "Actual and quantile arrays must share shape.",
    )

    require(
        np.isfinite(actual_values).all()
        and np.isfinite(quantile_values).all(),
        "Pinball inputs must be finite.",
    )

    error = actual_values - quantile_values

    return np.maximum(
        alpha * error,
        (alpha - 1.0) * error,
    )


def load_predictions() -> pd.DataFrame:
    """Load and validate canonical Day-21 predictions."""

    frame = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=[
            "forecast_date",
            "target_date",
        ],
    )

    require(
        len(frame) == 1194,
        f"Expected 1194 rows, found {len(frame)}.",
    )

    require(
        set(frame["method"]) == set(METHODS),
        "Unexpected prediction method set.",
    )

    require(
        not frame[
            ["method", "target_date"]
        ].duplicated().any(),
        "Duplicate method/target-date pairs found.",
    )

    counts = (
        frame.groupby("method")
        .size()
        .to_dict()
    )

    expected_counts = {
        "historical_simulation": 398,
        "ewma": 398,
        "gradient_boosting": 398,
    }

    require(
        counts == expected_counts,
        f"Unexpected method counts: {counts}",
    )

    return frame


def build_pairwise_diagnostics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Create one aligned diagnostics row per target date."""

    method_frames: dict[str, pd.DataFrame] = {}

    for method in METHODS:
        frame = (
            predictions.loc[
                predictions["method"] == method
            ]
            .sort_values("target_date")
            .reset_index(drop=True)
            .copy()
        )

        frame[
            "observation_pinball_loss"
        ] = observation_pinball_loss(
            frame[
                "actual_return"
            ].to_numpy(dtype="float64"),
            frame[
                "quantile_return"
            ].to_numpy(dtype="float64"),
        )

        method_frames[method] = frame

    reference = method_frames[
        "historical_simulation"
    ]

    target_dates = reference[
        "target_date"
    ].to_numpy()

    actual_return = reference[
        "actual_return"
    ].to_numpy(dtype="float64")

    for method in (
        "ewma",
        "gradient_boosting",
    ):
        frame = method_frames[method]

        require(
            np.array_equal(
                frame[
                    "target_date"
                ].to_numpy(),
                target_dates,
            ),
            f"{method}: target-date sequence differs.",
        )

        require(
            np.allclose(
                frame[
                    "actual_return"
                ].to_numpy(dtype="float64"),
                actual_return,
                rtol=0.0,
                atol=1e-15,
            ),
            f"{method}: actual returns differ.",
        )

    output = pd.DataFrame(
        {
            "target_date": reference["target_date"],
            "actual_return": actual_return,
        }
    )

    for method in METHODS:
        prefix = METHOD_PREFIX[method]
        frame = method_frames[method]

        output[
            f"{prefix}_quantile_return"
        ] = frame[
            "quantile_return"
        ].to_numpy(dtype="float64")

        output[
            f"{prefix}_var"
        ] = frame[
            "var"
        ].to_numpy(dtype="float64")

        output[
            f"{prefix}_violation"
        ] = frame[
            "violation"
        ].to_numpy(dtype=bool)

        output[
            f"{prefix}_pinball_loss"
        ] = frame[
            "observation_pinball_loss"
        ].to_numpy(dtype="float64")

    output[
        "gb_minus_historical_pinball"
    ] = (
        output["gb_pinball_loss"]
        - output["historical_pinball_loss"]
    )

    output[
        "gb_minus_ewma_pinball"
    ] = (
        output["gb_pinball_loss"]
        - output["ewma_pinball_loss"]
    )

    output[
        "ewma_minus_historical_pinball"
    ] = (
        output["ewma_pinball_loss"]
        - output["historical_pinball_loss"]
    )

    output[
        "gb_minus_historical_var"
    ] = (
        output["gb_var"]
        - output["historical_var"]
    )

    output[
        "gb_minus_ewma_var"
    ] = (
        output["gb_var"]
        - output["ewma_var"]
    )

    output[
        "ewma_minus_historical_var"
    ] = (
        output["ewma_var"]
        - output["historical_var"]
    )

    output[
        "abs_gb_historical_quantile_diff"
    ] = np.abs(
        output["gb_quantile_return"]
        - output["historical_quantile_return"]
    )

    output[
        "abs_gb_ewma_quantile_diff"
    ] = np.abs(
        output["gb_quantile_return"]
        - output["ewma_quantile_return"]
    )

    output[
        "abs_ewma_historical_quantile_diff"
    ] = np.abs(
        output["ewma_quantile_return"]
        - output["historical_quantile_return"]
    )

    loss_columns = [
        "historical_pinball_loss",
        "ewma_pinball_loss",
        "gb_pinball_loss",
    ]

    loss_matrix = output[
        loss_columns
    ].to_numpy(dtype="float64")

    minimum_loss = loss_matrix.min(axis=1)

    output["lowest_pinball_loss"] = minimum_loss

    method_labels = [
        "historical_simulation",
        "ewma",
        "gradient_boosting",
    ]

    winner_labels: list[str] = []

    for row_index in range(len(output)):
        matches = np.isclose(
            loss_matrix[row_index],
            minimum_loss[row_index],
            rtol=0.0,
            atol=1e-15,
        )

        winners = [
            method_labels[index]
            for index, matched in enumerate(matches)
            if matched
        ]

        winner_labels.append(
            winners[0]
            if len(winners) == 1
            else "tie"
        )

    output[
        "lowest_pinball_method"
    ] = winner_labels

    require(
        len(output) == 398,
        "Expected 398 paired target rows.",
    )

    require(
        output["target_date"].is_unique,
        "Target dates must be unique.",
    )

    numeric = output.select_dtypes(
        include=[np.number]
    )

    require(
        np.isfinite(
            numeric.to_numpy(dtype="float64")
        ).all(),
        "Non-finite paired diagnostics found.",
    )

    return output


def validate_aggregate_reconciliation(
    paired: pd.DataFrame,
) -> None:
    """Reconcile paired losses to A2 aggregate metrics."""

    comparison = (
        pd.read_csv(
            METRIC_COMPARISON_PATH
        )
        .set_index("method")
    )

    observed = {
        "historical_simulation": float(
            paired[
                "historical_pinball_loss"
            ].mean()
        ),
        "ewma": float(
            paired[
                "ewma_pinball_loss"
            ].mean()
        ),
        "gradient_boosting": float(
            paired[
                "gb_pinball_loss"
            ].mean()
        ),
    }

    for method, mean_loss in observed.items():
        expected = float(
            comparison.loc[
                method,
                "pinball_loss",
            ]
        )

        require(
            np.isclose(
                mean_loss,
                expected,
                rtol=0.0,
                atol=1e-15,
            ),
            (
                f"{method}: paired Pinball mean "
                "does not reconcile to A2."
            ),
        )

    pair_checks = [
        (
            "gb_minus_historical_pinball",
            observed["gradient_boosting"]
            - observed["historical_simulation"],
        ),
        (
            "gb_minus_ewma_pinball",
            observed["gradient_boosting"]
            - observed["ewma"],
        ),
        (
            "ewma_minus_historical_pinball",
            observed["ewma"]
            - observed["historical_simulation"],
        ),
    ]

    for column, expected in pair_checks:
        observed_delta = float(
            paired[column].mean()
        )

        require(
            np.isclose(
                observed_delta,
                expected,
                rtol=0.0,
                atol=1e-15,
            ),
            f"{column}: paired mean mismatch.",
        )


def print_summary(
    paired: pd.DataFrame,
) -> None:
    """Print descriptive paired evidence."""

    print("")
    print(
        "=== DAY 22 A3 PAIRED FORECAST DIAGNOSTICS ==="
    )
    print("")

    print(
        "Paired target dates :",
        len(paired),
    )

    print(
        "Start date          :",
        paired["target_date"].min().date(),
    )

    print(
        "End date            :",
        paired["target_date"].max().date(),
    )

    print("")
    print(
        "[MEAN OBSERVATION-LEVEL PINBALL LOSS]"
    )

    print(
        "Historical :",
        f"{paired['historical_pinball_loss'].mean():.15f}",
    )

    print(
        "EWMA       :",
        f"{paired['ewma_pinball_loss'].mean():.15f}",
    )

    print(
        "GB         :",
        f"{paired['gb_pinball_loss'].mean():.15f}",
    )

    print("")
    print("[MEAN PAIRED LOSS DIFFERENCE]")

    print(
        "GB - Historical  :",
        f"{paired['gb_minus_historical_pinball'].mean():.15f}",
    )

    print(
        "GB - EWMA        :",
        f"{paired['gb_minus_ewma_pinball'].mean():.15f}",
    )

    print(
        "EWMA - Historical:",
        f"{paired['ewma_minus_historical_pinball'].mean():.15f}",
    )

    print("")
    print("[LOWEST LOSS BY TARGET DATE]")

    counts = (
        paired["lowest_pinball_method"]
        .value_counts()
        .sort_index()
    )

    for label, count in counts.items():
        print(
            f"{label:24s}: {int(count)}"
        )

    print("")
    print(
        "Interpretation deferred to A3.2: "
        "distribution/concentration analysis."
    )


def main() -> None:
    predictions = load_predictions()

    paired = build_pairwise_diagnostics(
        predictions
    )

    validate_aggregate_reconciliation(
        paired
    )

    paired.to_csv(
        OUTPUT_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    print_summary(paired)

    print("")
    print(
        "Wrote:",
        OUTPUT_PATH.relative_to(REPO_ROOT),
    )

    print(
        "PASS - DAY 22 A3.1 PAIRED DIAGNOSTICS COMPLETE"
    )


if __name__ == "__main__":
    main()
