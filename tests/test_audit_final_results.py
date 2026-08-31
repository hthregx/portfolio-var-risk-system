from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_final_results import (
    compare_metrics,
    recompute_metrics,
)


ROOT = Path(__file__).resolve().parents[1]

PRED = (
    ROOT
    / "data/sample/final_predictions_contract_sample.csv"
)

METRICS = (
    ROOT
    / "data/sample/final_metrics_contract_sample.csv"
)


def load_data():
    return (
        pd.read_csv(PRED),
        pd.read_csv(METRICS),
    )


def test_fixture_metrics_match():
    pred, reported = load_data()

    recomputed = recompute_metrics(
        pred,
        alpha=0.05,
    )

    assert compare_metrics(
        recomputed,
        reported,
    )


def test_manual_metric_values():
    pred, _ = load_data()

    result = recompute_metrics(
        pred,
        alpha=0.05,
    ).set_index("method")

    assert result.loc[
        "historical_simulation",
        "forecast_count",
    ] == 3

    assert result.loc[
        "historical_simulation",
        "violation_count",
    ] == 1

    assert result.loc[
        "ewma",
        "violation_count",
    ] == 1

    assert result.loc[
        "gradient_boosting",
        "violation_count",
    ] == 0

    assert result.loc[
        "historical_simulation",
        "violation_rate",
    ] == pytest.approx(
        1 / 3
    )

    assert result.loc[
        "historical_simulation",
        "pinball_loss",
    ] == pytest.approx(
        0.003416666666666667
    )

    assert result.loc[
        "ewma",
        "pinball_loss",
    ] == pytest.approx(
        0.002
    )

    assert result.loc[
        "gradient_boosting",
        "pinball_loss",
    ] == pytest.approx(
        0.0003333333333333333
    )


def test_var_summary_values():
    pred, _ = load_data()

    result = recompute_metrics(
        pred,
        alpha=0.05,
    ).set_index("method")

    assert result.loc[
        "historical_simulation",
        "average_var",
    ] == pytest.approx(
        0.013333333333333334
    )

    assert result.loc[
        "historical_simulation",
        "minimum_var",
    ] == pytest.approx(
        0.0
    )

    assert result.loc[
        "historical_simulation",
        "maximum_var",
    ] == pytest.approx(
        0.020
    )

    assert result.loc[
        "ewma",
        "minimum_var",
    ] == pytest.approx(
        0.010
    )

    assert result.loc[
        "gradient_boosting",
        "maximum_var",
    ] == pytest.approx(
        0.030
    )


def test_wrong_alpha_is_rejected():
    pred, reported = load_data()

    recomputed = recompute_metrics(
        pred,
        alpha=0.10,
    )

    with pytest.raises(
        ValueError,
        match="pinball_loss",
    ):
        compare_metrics(
            recomputed,
            reported,
        )


def test_wrong_violation_is_rejected():
    pred, _ = load_data()

    pred.loc[0, "violation"] = True

    with pytest.raises(
        ValueError,
        match="violation",
    ):
        recompute_metrics(
            pred,
            alpha=0.05,
        )