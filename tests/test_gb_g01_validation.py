from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_gb_g01_validation import (
    ALPHA,
    EXPERIMENT_ID,
    PREDICTION_COLUMNS,
    compute_metrics,
    run_g01,
)
from src.gb_market_features import FEATURE_COLUMNS


@pytest.fixture
def synthetic_dataset():
    dates = pd.bdate_range(
        "2021-01-01",
        "2023-12-29",
    )

    rng = np.random.default_rng(42)

    data = pd.DataFrame(
        rng.normal(
            size=(len(dates), len(FEATURE_COLUMNS))
        ),
        index=dates,
        columns=FEATURE_COLUMNS,
    )

    data["portfolio_simple_return"] = rng.normal(
        loc=0.0,
        scale=0.02,
        size=len(dates),
    )

    return data


@pytest.fixture
def g01_result(monkeypatch, synthetic_dataset):
    monkeypatch.setattr(
        "scripts.run_gb_g01_validation.load_dataset",
        lambda: synthetic_dataset,
    )

    return run_g01(write_outputs=False)


def test_runner_executes(g01_result):
    pred, exp, meta = g01_result

    assert len(pred) > 0
    assert exp["experiment_id"] == EXPERIMENT_ID
    assert meta["experiment_id"] == EXPERIMENT_ID


def test_prediction_schema_exact(g01_result):
    pred, _, _ = g01_result

    assert pred.columns.tolist() == PREDICTION_COLUMNS


def test_finite_quantiles_and_var(g01_result):
    pred, _, _ = g01_result

    assert np.isfinite(pred["quantile_return"]).all()
    assert np.isfinite(pred["var"]).all()
    assert (pred["var"] >= 0.0).all()


def test_strict_violation_rule():
    actual = pd.Series([
        -0.03,
        -0.02,
        -0.01,
    ])

    quantile = pd.Series([
        -0.02,
        -0.02,
        -0.02,
    ])

    result = actual < quantile

    assert result.tolist() == [
        True,
        False,
        False,
    ]


def test_date_ordering(g01_result):
    pred, _, _ = g01_result

    forecast = pd.to_datetime(
        pred["forecast_date"]
    )

    target = pd.to_datetime(
        pred["target_date"]
    )

    assert (forecast < target).all()


def test_no_duplicate_target_dates(g01_result):
    pred, _, _ = g01_result

    assert pred["target_date"].is_unique


def test_prediction_count(g01_result):
    pred, exp, meta = g01_result

    assert len(pred) > 0
    assert len(pred) == exp["n_validation"]
    assert len(pred) == meta["prediction_count"]


def test_violation_count_consistent(g01_result):
    pred, exp, meta = g01_result

    count = int(
        pred["violation"].sum()
    )

    assert count == exp["violations"]
    assert count == meta["violation_count"]


def test_metrics_recompute_from_predictions(g01_result):
    pred, exp, _ = g01_result

    result = compute_metrics(pred)

    assert result["violations"] == exp["violations"]

    assert result["violation_rate"] == pytest.approx(
        exp["violation_rate"]
    )

    assert result["pinball_loss"] == pytest.approx(
        exp["pinball_loss"]
    )

    assert result["average_var"] == pytest.approx(
        exp["average_var"]
    )


def test_alpha_and_var_formula(g01_result):
    pred, exp, meta = g01_result

    assert ALPHA == 0.05
    assert exp["alpha"] == 0.05
    assert meta["alpha"] == 0.05

    expected = np.maximum(
        0.0,
        -pred["quantile_return"].to_numpy(),
    )

    np.testing.assert_allclose(
        pred["var"].to_numpy(),
        expected,
    )


def test_metadata_fields_present(g01_result):
    _, _, meta = g01_result

    required = {
        "experiment_id",
        "target",
        "alpha",
        "horizon",
        "feature_names",
        "model_family",
        "package_version",
        "random_seed",
        "validation_boundary",
        "source_commit",
        "runtime_seconds",
    }

    assert required.issubset(meta)
    assert meta["experiment_id"] == "G01"
    assert meta["runtime_seconds"] >= 0.0
    assert meta["reserved_later_used"] is False


def test_quantile_distribution_diagnostics(g01_result):
    _, _, meta = g01_result

    required = {
        "quantile_min",
        "quantile_max",
        "quantile_mean",
        "quantile_median",
        "quantile_p05",
        "quantile_p95",
        "var_min",
        "var_max",
        "obvious_outlier_count",
    }

    assert required.issubset(meta)
    assert meta["quantile_min"] <= meta["quantile_max"]
    assert meta["var_min"] >= 0.0
    assert meta["var_min"] <= meta["var_max"]
    assert meta["obvious_outlier_count"] >= 0


def test_average_var_policy(g01_result):
    _, _, meta = g01_result

    assert (
        meta["average_var_selection_policy"]
        == "descriptive_only_not_standalone_selection"
    )


def test_fixed_seed_is_deterministic(
    monkeypatch,
    synthetic_dataset,
):
    monkeypatch.setattr(
        "scripts.run_gb_g01_validation.load_dataset",
        lambda: synthetic_dataset,
    )

    p1, e1, _ = run_g01(write_outputs=False)
    p2, e2, _ = run_g01(write_outputs=False)

    np.testing.assert_allclose(
        p1["quantile_return"],
        p2["quantile_return"],
    )

    np.testing.assert_allclose(
        p1["var"],
        p2["var"],
    )

    assert p1["violation"].equals(
        p2["violation"]
    )

    assert e1["violations"] == e2["violations"]

    assert e1["violation_rate"] == pytest.approx(
        e2["violation_rate"]
    )

    assert e1["pinball_loss"] == pytest.approx(
        e2["pinball_loss"]
    )

    assert e1["average_var"] == pytest.approx(
        e2["average_var"]
    )