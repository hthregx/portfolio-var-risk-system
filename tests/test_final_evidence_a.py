from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = ROOT / "results" / "final_predictions.csv"
METRICS_PATH = ROOT / "results" / "final_metrics.csv"
COMPARISON_PATH = ROOT / "results" / "final_metric_comparison.csv"
VALIDATION_PATH = ROOT / "results" / "final_evidence_validation_a.csv"

FIGURE_PATHS = [
    ROOT / "results" / "figures" / "final_violation_rate_a.png",
    ROOT / "results" / "figures" / "final_pinball_loss_a.png",
    ROOT / "results" / "figures" / "final_average_var_a.png",
]

EXPECTED_METHODS = {
    "historical_simulation",
    "ewma",
    "gradient_boosting",
}

EXPECTED_METRICS = {
    "historical_simulation": {
        "forecast_count": 398,
        "violation_count": 27,
        "violation_rate": 0.0678391959798995,
        "pinball_loss": 0.0018964642326659775,
        "average_var": 0.021330189279201585,
        "minimum_var": 0.017088385227830519,
        "maximum_var": 0.027020197746921468,
        "config_id": "historical_w250",
    },
    "ewma": {
        "forecast_count": 398,
        "violation_count": 22,
        "violation_rate": 0.055276381909547742,
        "pinball_loss": 0.0019683212526850121,
        "average_var": 0.025084042489536107,
        "minimum_var": 0.014005656462523988,
        "maximum_var": 0.056446630577662971,
        "config_id": "ewma_d094",
    },
    "gradient_boosting": {
        "forecast_count": 398,
        "violation_count": 24,
        "violation_rate": 0.06030150753768844,
        "pinball_loss": 0.0017581309509574873,
        "average_var": 0.023349057036283791,
        "minimum_var": 0.01374247307033119,
        "maximum_var": 0.048139015621043925,
        "config_id": "gb_G04",
    },
}

TOLERANCE = 1e-12
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_PATH)


@pytest.fixture(scope="module")
def metrics() -> pd.DataFrame:
    return pd.read_csv(METRICS_PATH)


@pytest.fixture(scope="module")
def comparison() -> pd.DataFrame:
    return pd.read_csv(COMPARISON_PATH)


def test_prediction_shape_and_method_contract(predictions: pd.DataFrame) -> None:
    assert len(predictions) == 1194
    assert set(predictions["method"]) == EXPECTED_METHODS

    counts = predictions.groupby("method").size().to_dict()

    assert counts == {
        "historical_simulation": 398,
        "ewma": 398,
        "gradient_boosting": 398,
    }

    duplicate_count = predictions.duplicated(
        subset=["method", "target_date"]
    ).sum()

    assert duplicate_count == 0


def test_prediction_dates_are_aligned(predictions: pd.DataFrame) -> None:
    target_dates_by_method = {
        method: tuple(
            sorted(
                predictions.loc[
                    predictions["method"].eq(method),
                    "target_date",
                ]
            )
        )
        for method in EXPECTED_METHODS
    }

    baseline = target_dates_by_method["historical_simulation"]

    assert len(baseline) == 398
    assert baseline[0] == "2024-12-18"
    assert baseline[-1] == "2026-07-28"

    for dates in target_dates_by_method.values():
        assert dates == baseline


def test_realized_returns_are_shared_across_methods(
    predictions: pd.DataFrame,
) -> None:
    actual_counts = (
        predictions.groupby("target_date")["actual_return"]
        .nunique(dropna=False)
    )

    assert len(actual_counts) == 398
    assert actual_counts.eq(1).all()


def test_strict_violation_and_var_identity(
    predictions: pd.DataFrame,
) -> None:
    actual = pd.to_numeric(predictions["actual_return"], errors="raise")
    quantile = pd.to_numeric(predictions["quantile_return"], errors="raise")
    var = pd.to_numeric(predictions["var"], errors="raise")

    stored_violation = (
        predictions["violation"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

    assert stored_violation.notna().all()

    expected_violation = actual.lt(quantile)
    expected_var = (-quantile).clip(lower=0.0)

    assert stored_violation.equals(expected_violation)
    assert var.ge(0.0).all()
    assert (var - expected_var).abs().le(TOLERANCE).all()


@pytest.mark.parametrize(
    "method",
    [
        "historical_simulation",
        "ewma",
        "gradient_boosting",
    ],
)
def test_frozen_metrics(
    metrics: pd.DataFrame,
    method: str,
) -> None:
    row = metrics.loc[metrics["method"].eq(method)]

    assert len(row) == 1

    row = row.iloc[0]
    expected = EXPECTED_METRICS[method]

    assert int(row["forecast_count"]) == expected["forecast_count"]
    assert int(row["violation_count"]) == expected["violation_count"]
    assert row["config_id"] == expected["config_id"]
    assert row["test_start"] == "2024-12-18"
    assert row["test_end"] == "2026-07-28"

    for field in (
        "violation_rate",
        "pinball_loss",
        "average_var",
        "minimum_var",
        "maximum_var",
    ):
        assert float(row[field]) == pytest.approx(
            expected[field],
            abs=TOLERANCE,
        )


def test_criterion_specific_leaders(
    comparison: pd.DataFrame,
) -> None:
    assert len(comparison) == 3
    assert set(comparison["method"]) == EXPECTED_METHODS

    calibration_leader = comparison.loc[
        comparison["calibration_distance"].idxmin(),
        "method",
    ]

    pinball_leader = comparison.loc[
        comparison["pinball_loss"].idxmin(),
        "method",
    ]

    lowest_average_var = comparison.loc[
        comparison["average_var"].idxmin(),
        "method",
    ]

    assert calibration_leader == "ewma"
    assert pinball_leader == "gradient_boosting"
    assert lowest_average_var == "historical_simulation"


def test_a_owned_figure_outputs_are_valid_pngs() -> None:
    assert len(FIGURE_PATHS) == 3

    for path in FIGURE_PATHS:
        assert path.is_file()
        assert path.stat().st_size >= 5000

        with path.open("rb") as handle:
            assert handle.read(8) == PNG_SIGNATURE


def test_final_evidence_validation_artifact() -> None:
    validation = pd.read_csv(VALIDATION_PATH)

    assert list(validation.columns) == [
        "check",
        "expected",
        "actual",
        "tolerance",
        "status",
        "source",
    ]

    assert len(validation) == 26
    assert validation["check"].is_unique
    assert validation["status"].eq("PASS").all()

    tolerances = pd.to_numeric(
        validation["tolerance"],
        errors="raise",
    )

    assert tolerances.eq(TOLERANCE).all()