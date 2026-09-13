from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_gb_tuning_b import (
    ALPHA,
    EXPERIMENTS,
    FEATURES,
    PREDICTION_COLUMNS,
    TARGET,
    compute_metrics,
    run_all,
)


@pytest.fixture
def synthetic_dataset():
    dates = pd.bdate_range(
        "2021-01-01",
        "2023-12-29",
    )

    rng = np.random.default_rng(42)

    data = pd.DataFrame(
        rng.normal(
            size=(len(dates), len(FEATURES))
        ),
        index=dates,
        columns=FEATURES,
    )

    data[TARGET] = rng.normal(
        0.0,
        0.02,
        len(dates),
    )

    return data


@pytest.fixture
def tuning_result(
    monkeypatch,
    synthetic_dataset,
):
    monkeypatch.setattr(
        "scripts.run_gb_tuning_b.load_dataset",
        lambda: synthetic_dataset,
    )

    return run_all(write_outputs=False)


def test_runner_executes(tuning_result):
    pred, summary = tuning_result

    assert not pred.empty
    assert not summary.empty


def test_experiment_ids(tuning_result):
    pred, summary = tuning_result

    expected = {"G01", "G05", "G06", "G07"}

    assert set(pred["experiment_id"]) == expected
    assert set(summary["experiment_id"]) == expected


def test_predeclared_configs():
    assert EXPERIMENTS == {
        "G01": (2, 0.05, 100),
        "G05": (2, 0.05, 50),
        "G06": (2, 0.05, 200),
        "G07": (2, 0.10, 100),
    }

    assert ALPHA == 0.05


def test_prediction_schema_exact(tuning_result):
    pred, _ = tuning_result

    assert pred.columns.tolist() == PREDICTION_COLUMNS


def test_finite_quantile_and_var(tuning_result):
    pred, _ = tuning_result

    assert np.isfinite(
        pred["quantile_return"]
    ).all()

    assert np.isfinite(
        pred["var"]
    ).all()

    assert (
        pred["var"] >= 0.0
    ).all()


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

    assert (actual < quantile).tolist() == [
        True,
        False,
        False,
    ]


def test_date_ordering(tuning_result):
    pred, _ = tuning_result

    forecast = pd.to_datetime(
        pred["forecast_date"]
    )

    target = pd.to_datetime(
        pred["target_date"]
    )

    assert (forecast < target).all()


def test_unique_targets_per_experiment(
    tuning_result,
):
    pred, _ = tuning_result

    for _, part in pred.groupby(
        "experiment_id"
    ):
        assert part["target_date"].is_unique


def test_same_validation_dates(tuning_result):
    pred, _ = tuning_result

    groups = [
        tuple(part["target_date"])
        for _, part in pred.groupby(
            "experiment_id"
        )
    ]

    assert all(
        dates == groups[0]
        for dates in groups[1:]
    )


def test_metrics_recompute(tuning_result):
    pred, summary = tuning_result

    summary = summary.set_index(
        "experiment_id"
    )

    for exp, part in pred.groupby(
        "experiment_id"
    ):
        result = compute_metrics(part)
        row = summary.loc[exp]

        assert (
            result["violations"]
            == row["violations"]
        )

        assert result[
            "violation_rate"
        ] == pytest.approx(
            row["violation_rate"]
        )

        assert result[
            "pinball_loss"
        ] == pytest.approx(
            row["pinball_loss"]
        )

        assert result[
            "average_var"
        ] == pytest.approx(
            row["average_var"]
        )


def test_runtime_non_negative(tuning_result):
    _, summary = tuning_result

    assert (
        summary["runtime_seconds"] >= 0.0
    ).all()


def test_fixed_seed_deterministic(
    monkeypatch,
    synthetic_dataset,
):
    monkeypatch.setattr(
        "scripts.run_gb_tuning_b.load_dataset",
        lambda: synthetic_dataset,
    )

    p1, _ = run_all(write_outputs=False)
    p2, _ = run_all(write_outputs=False)

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