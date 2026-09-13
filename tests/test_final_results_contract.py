from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.validation.final_results_contract import (
    validate_final_metrics,
    validate_final_predictions,
)


ROOT = Path(__file__).resolve().parents[1]

PRED_PATH = ROOT / "data/sample/final_predictions_contract_sample.csv"
METRIC_PATH = ROOT / "data/sample/final_metrics_contract_sample.csv"


def load_predictions():
    return pd.read_csv(PRED_PATH)


def load_metrics():
    return pd.read_csv(METRIC_PATH)


def test_valid_fixtures_pass():
    assert not validate_final_predictions(
        load_predictions()
    ).empty

    assert not validate_final_metrics(
        load_metrics()
    ).empty


@pytest.mark.parametrize(
    "column",
    [
        "forecast_date",
        "target_date",
        "method",
        "actual_return",
        "quantile_return",
        "var",
        "violation",
        "runtime_seconds",
        "config_id",
    ],
)
def test_missing_prediction_column_rejected(column):
    df = load_predictions().drop(columns=[column])

    with pytest.raises(ValueError, match="missing required"):
        validate_final_predictions(df)


def test_duplicate_key_rejected():
    df = load_predictions()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        validate_final_predictions(df)


def test_invalid_method_rejected():
    df = load_predictions()
    df.loc[0, "method"] = "invalid"

    with pytest.raises(ValueError, match="Method"):
        validate_final_predictions(df)


def test_nonfinite_rejected():
    df = load_predictions()
    df.loc[0, "actual_return"] = np.inf

    with pytest.raises(ValueError, match="finite"):
        validate_final_predictions(df)


def test_forecast_not_before_target_rejected():
    df = load_predictions()
    df.loc[0, "forecast_date"] = df.loc[0, "target_date"]

    with pytest.raises(ValueError, match="forecast_date"):
        validate_final_predictions(df)


def test_wrong_var_sign_rejected():
    df = load_predictions()
    df.loc[0, "var"] = 999.0

    with pytest.raises(ValueError, match="VaR"):
        validate_final_predictions(df)


def test_wrong_violation_rejected():
    df = load_predictions()
    df.loc[0, "violation"] = True

    with pytest.raises(ValueError, match="violation"):
        validate_final_predictions(df)


def test_target_universe_mismatch_rejected():
    df = load_predictions()

    mask = (
        (df["method"] == "ewma")
        & (df["target_date"] == "2024-01-05")
    )

    df = df.loc[~mask].copy()

    with pytest.raises(ValueError, match="target-date universe"):
        validate_final_predictions(df)


def test_actual_return_mismatch_rejected():
    df = load_predictions()

    mask = (
        (df["method"] == "ewma")
        & (df["target_date"] == "2024-01-03")
    )

    df.loc[mask, "actual_return"] = 0.99

    with pytest.raises(ValueError, match="actual_return"):
        validate_final_predictions(df)


def test_duplicate_metric_method_rejected():
    df = load_metrics()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        validate_final_metrics(df)


def test_invalid_metric_range_rejected():
    df = load_metrics()
    df.loc[0, "violation_rate"] = 1.5

    with pytest.raises(ValueError, match="violation_rate"):
        validate_final_metrics(df)
def test_prediction_numeric_dtype_rejected():
    df = load_predictions()
    df["actual_return"] = (
        df["actual_return"].astype(str)
    )

    with pytest.raises(
        ValueError,
        match="numeric dtype",
    ):
        validate_final_predictions(df)


def test_violation_non_boolean_rejected():
    df = load_predictions()
    df["violation"] = (
        df["violation"]
        .map({True: "yes", False: "no"})
    )

    with pytest.raises(
        ValueError,
        match="boolean dtype",
    ):
        validate_final_predictions(df)


def test_metric_count_non_integer_rejected():
    df = load_metrics()
    df["forecast_count"] = (
        df["forecast_count"]
        .astype(float)
    )

    with pytest.raises(
        ValueError,
        match="integer dtype",
    ):
        validate_final_metrics(df)


def test_prediction_required_null_rejected():
    df = load_predictions()
    df.loc[0, "config_id"] = None

    with pytest.raises(
        ValueError,
        match="Missing values",
    ):
        validate_final_predictions(df)


def test_metric_required_null_rejected():
    df = load_metrics()
    df.loc[0, "config_id"] = None

    with pytest.raises(
        ValueError,
        match="Missing values",
    ):
        validate_final_metrics(df)


def test_fractional_metric_count_rejected():
    df = load_metrics()
    df["forecast_count"] = (
        df["forecast_count"]
        .astype(float)
    )
    df.loc[0, "forecast_count"] = 3.5

    with pytest.raises(
        ValueError,
        match="integer dtype",
    ):
        validate_final_metrics(df)