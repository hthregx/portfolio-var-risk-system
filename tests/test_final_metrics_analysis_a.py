from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_final_metrics_a import (
    ALPHA,
    METHOD_ORDER,
    OUTPUT_COLUMNS,
    add_criterion_flags,
    build_comparison,
    pinball_loss,
)


def test_pinball_loss_matches_manual_quantile_formula() -> None:
    actual = np.array(
        [-0.04, -0.01, 0.02],
        dtype="float64",
    )

    predicted = np.array(
        [-0.03, -0.02, -0.01],
        dtype="float64",
    )

    error = actual - predicted

    expected = np.maximum(
        ALPHA * error,
        (ALPHA - 1.0) * error,
    ).mean()

    observed = pinball_loss(
        actual,
        predicted,
        ALPHA,
    )

    assert np.isclose(
        observed,
        expected,
        rtol=0.0,
        atol=1e-15,
    )


def test_build_comparison_matches_day22_canonical_results() -> None:
    frame = build_comparison()

    assert frame.columns.tolist() == OUTPUT_COLUMNS
    assert frame["method"].tolist() == METHOD_ORDER
    assert len(frame) == 3

    lookup = frame.set_index("method")

    expected = {
        "historical_simulation": {
            "violations": 27,
            "rate": 0.0678391959798995,
            "pinball": 0.0018964642326659775,
            "average_var": 0.021330189279201585,
        },
        "ewma": {
            "violations": 22,
            "rate": 0.055276381909547742,
            "pinball": 0.0019683212526850121,
            "average_var": 0.025084042489536107,
        },
        "gradient_boosting": {
            "violations": 24,
            "rate": 0.06030150753768844,
            "pinball": 0.0017581309509574873,
            "average_var": 0.023349057036283791,
        },
    }

    for method, values in expected.items():
        row = lookup.loc[method]

        assert int(row["forecast_count"]) == 398
        assert int(row["violation_count"]) == values["violations"]

        assert np.isclose(
            float(row["violation_rate"]),
            values["rate"],
            rtol=0.0,
            atol=1e-15,
        )

        assert np.isclose(
            float(row["pinball_loss"]),
            values["pinball"],
            rtol=0.0,
            atol=1e-15,
        )

        assert np.isclose(
            float(row["average_var"]),
            values["average_var"],
            rtol=0.0,
            atol=1e-15,
        )


def test_criterion_flags_are_specific_not_overall_ranking() -> None:
    frame = build_comparison().set_index("method")

    assert bool(
        frame.loc[
            "ewma",
            "calibration_leader",
        ]
    )

    assert bool(
        frame.loc[
            "gradient_boosting",
            "pinball_leader",
        ]
    )

    assert bool(
        frame.loc[
            "historical_simulation",
            "lowest_average_var",
        ]
    )

    forbidden_columns = {
        "winner",
        "overall_winner",
        "rank",
        "overall_rank",
        "best_model",
    }

    assert forbidden_columns.isdisjoint(
        set(frame.columns)
    )


def test_calibration_distance_is_absolute_distance_from_alpha() -> None:
    frame = build_comparison()

    expected = np.abs(
        frame["violation_rate"].to_numpy(dtype="float64")
        - ALPHA
    )

    assert np.allclose(
        frame["calibration_distance"].to_numpy(dtype="float64"),
        expected,
        rtol=0.0,
        atol=1e-15,
    )


def test_exported_comparison_matches_recomputed_table() -> None:
    exported = pd.read_csv(
        "results/final_metric_comparison.csv"
    )

    recomputed = build_comparison().copy()

    for column in ("test_start", "test_end"):
        recomputed[column] = (
            pd.to_datetime(recomputed[column])
            .dt.strftime("%Y-%m-%d")
        )

    assert exported.columns.tolist() == recomputed.columns.tolist()
    assert exported["method"].tolist() == recomputed["method"].tolist()

    numeric_columns = [
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
    ]

    assert np.allclose(
        exported[numeric_columns].to_numpy(dtype="float64"),
        recomputed[numeric_columns].to_numpy(dtype="float64"),
        rtol=0.0,
        atol=1e-15,
    )

    for column in (
        "config_id",
        "test_start",
        "test_end",
        "calibration_leader",
        "pinball_leader",
        "lowest_average_var",
    ):
        assert exported[column].astype(str).tolist() == (
            recomputed[column].astype(str).tolist()
        )
