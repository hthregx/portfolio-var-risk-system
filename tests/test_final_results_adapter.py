from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.reporting.final_results_adapter import (
    build_dashboard_data,
    build_exception_table,
    build_latest_var_summary,
    build_model_comparison,
    build_report_summary,
)


ROOT = Path(__file__).resolve().parents[1]

PRED = ROOT / "data/sample/final_predictions_contract_sample.csv"
METRICS = ROOT / "data/sample/final_metrics_contract_sample.csv"


def load_data():
    return pd.read_csv(PRED), pd.read_csv(METRICS)


def test_model_comparison():
    _, metrics = load_data()

    result = build_model_comparison(metrics)

    assert len(result) == 3
    assert result["method"].is_unique


def test_exception_table():
    pred, _ = load_data()

    result = build_exception_table(pred)

    assert len(result) == 2
    assert set(result["method"]) == {
        "historical_simulation",
        "ewma",
    }


def test_latest_var_summary():
    pred, _ = load_data()

    result = build_latest_var_summary(pred)

    assert len(result) == 3
    assert result["method"].is_unique
    assert (
        pd.to_datetime(result["target_date"])
        == pd.Timestamp("2024-01-05")
    ).all()


def test_dashboard_deterministic():
    pred, metrics = load_data()

    first = build_dashboard_data(pred, metrics)
    second = build_dashboard_data(pred, metrics)

    for key in first:
        pd.testing.assert_frame_equal(
            first[key],
            second[key],
        )


def test_report_summary_matches_fixture():
    pred, metrics = load_data()

    result = build_report_summary(pred, metrics)

    row = result.set_index("method").loc[
        "historical_simulation"
    ]

    assert row["forecast_count"] == 3
    assert row["average_var"] == pytest.approx(
    0.013333333333333334
)
    assert row["violation_rate_display"] == "33.33%"


def test_adapter_does_not_mutate_inputs():
    pred, metrics = load_data()

    pred_before = pred.copy(deep=True)
    metrics_before = metrics.copy(deep=True)

    build_dashboard_data(pred, metrics)
    build_report_summary(pred, metrics)

    pd.testing.assert_frame_equal(
        pred,
        pred_before,
    )

    pd.testing.assert_frame_equal(
        metrics,
        metrics_before,
    )