"""
Tests for the A-19 Gradient Boosting validation-tuning runner.

PROMPT GỐC ĐỂ LƯU:
Verify the locked G01-G04 experiment grid, canonical feature/target
frame, fixed 739-row validation split, expanding no-look-ahead
training semantics, strict violation rule, Pinball Loss alpha=.05,
reserved-later exclusion, input immutability, and orchestration
without performing expensive real-model tuning inside unit tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import run_gb_tuning_a as tuning


EXPECTED_FEATURE_COLUMNS = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_5",
    "rolling_vol_5",
    "rolling_vol_20",
    "rolling_vol_60",
    "drawdown",
]


class FakeGradientBoostingVaR:
    """Lightweight estimator double for runner-contract tests."""

    init_calls: list[dict] = []
    fit_calls: list[dict] = []
    forecast_calls: list[pd.Timestamp] = []

    def __init__(self, **kwargs):
        self.__class__.init_calls.append(
            dict(kwargs)
        )

    @classmethod
    def reset(cls):
        cls.init_calls = []
        cls.fit_calls = []
        cls.forecast_calls = []

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ):
        if not features.index.equals(
            target.index
        ):
            raise AssertionError(
                "Fake model received misaligned X/y."
            )

        self.__class__.fit_calls.append(
            {
                "n_train": len(features),
                "train_start": features.index.min(),
                "train_end": features.index.max(),
                "feature_columns": tuple(
                    features.columns
                ),
            }
        )

        return self

    def forecast(
        self,
        features: pd.DataFrame,
    ):
        if len(features) != 1:
            raise AssertionError(
                "Expected exactly one target row."
            )

        self.__class__.forecast_calls.append(
            features.index[0]
        )

        quantile_return = -0.01

        return {
            "quantile_return": quantile_return,
            "var": max(
                0.0,
                -quantile_return,
            ),
        }


@pytest.fixture
def canonical_frame():
    return tuning.load_feature_target_frame()


@pytest.fixture
def fake_model(monkeypatch):
    FakeGradientBoostingVaR.reset()

    monkeypatch.setattr(
        tuning,
        "GradientBoostingVaR",
        FakeGradientBoostingVaR,
    )

    return FakeGradientBoostingVaR


def test_locked_experiment_grid():
    observed = [
        (
            config.experiment_id,
            config.max_depth,
            config.learning_rate,
            config.n_estimators,
        )
        for config in tuning.EXPERIMENTS
    ]

    assert observed == [
        ("G01", 2, 0.05, 100),
        ("G02", 1, 0.05, 100),
        ("G03", 3, 0.05, 100),
        ("G04", 2, 0.03, 100),
    ]


def test_locked_global_contract():
    assert tuning.ALPHA == pytest.approx(
        0.05
    )

    assert tuning.VALIDATION_START == (
        pd.Timestamp("2021-12-31")
    )

    assert tuning.VALIDATION_END == (
        pd.Timestamp("2024-12-17")
    )


def test_feature_target_frame_contract(
    canonical_frame,
):
    frame = canonical_frame

    assert len(frame) == 1577

    assert frame.index.is_unique
    assert frame.index.is_monotonic_increasing

    assert frame.index.min() == pd.Timestamp(
        "2020-04-06"
    )

    assert frame.index.max() == pd.Timestamp(
        "2026-07-28"
    )

    assert list(frame.columns) == (
        EXPECTED_FEATURE_COLUMNS
        + ["target_return"]
    )

    assert not frame.isna().any().any()

    assert np.isfinite(
        frame.to_numpy()
    ).all()


def test_locked_split_counts(
    canonical_frame,
):
    index = canonical_frame.index

    pre_validation = index < (
        tuning.VALIDATION_START
    )

    validation = (
        (index >= tuning.VALIDATION_START)
        & (index <= tuning.VALIDATION_END)
    )

    reserved_later = (
        index > tuning.VALIDATION_END
    )

    assert int(
        pre_validation.sum()
    ) == 440

    assert int(
        validation.sum()
    ) == 739

    assert int(
        reserved_later.sum()
    ) == 398


def test_pinball_loss_known_value():
    actual = np.array(
        [-0.03, 0.01],
        dtype=float,
    )

    predicted = np.array(
        [-0.02, 0.00],
        dtype=float,
    )

    observed = tuning.pinball_loss(
        actual,
        predicted,
    )

    assert observed == pytest.approx(
        0.005
    )


def test_pinball_loss_exact_predictions_zero():
    actual = np.array(
        [-0.04, 0.00, 0.02],
        dtype=float,
    )

    assert tuning.pinball_loss(
        actual,
        actual.copy(),
    ) == pytest.approx(0.0)


def test_run_experiment_expanding_contract(
    canonical_frame,
    fake_model,
):
    config = tuning.EXPERIMENTS[1]

    predictions, metrics = (
        tuning.run_experiment(
            canonical_frame,
            config,
        )
    )

    assert len(predictions) == 739

    assert predictions[
        "target_date"
    ].min() == tuning.VALIDATION_START

    assert predictions[
        "target_date"
    ].max() == tuning.VALIDATION_END

    assert (
        predictions["forecast_date"]
        < predictions["target_date"]
    ).all()

    assert predictions[
        "n_train"
    ].iloc[0] == 440

    assert predictions[
        "n_train"
    ].iloc[-1] == 1178

    assert len(
        fake_model.fit_calls
    ) == 739

    assert len(
        fake_model.forecast_calls
    ) == 739

    for fit_call, target_date in zip(
        fake_model.fit_calls,
        fake_model.forecast_calls,
        strict=True,
    ):
        assert (
            fit_call["train_end"]
            < target_date
        )

        assert (
            fit_call["feature_columns"]
            == tuple(
                EXPECTED_FEATURE_COLUMNS
            )
        )

    expected_violation = (
        predictions["actual_return"]
        < predictions["quantile_return"]
    )

    assert predictions[
        "violation"
    ].equals(
        expected_violation
    )

    assert metrics[
        "violation_count"
    ] == int(
        expected_violation.sum()
    )

    assert metrics[
        "violation_rate"
    ] == pytest.approx(
        expected_violation.mean()
    )

    assert metrics[
        "average_var"
    ] == pytest.approx(
        0.01
    )

    assert metrics[
        "validation_rows"
    ] == 739

    assert metrics[
        "validation_start"
    ] == "2021-12-31"

    assert metrics[
        "validation_end"
    ] == "2024-12-17"

    expected_pinball = (
        tuning.pinball_loss(
            predictions[
                "actual_return"
            ].to_numpy(),
            predictions[
                "quantile_return"
            ].to_numpy(),
        )
    )

    assert metrics[
        "pinball_loss"
    ] == pytest.approx(
        expected_pinball
    )

    first_init = (
        fake_model.init_calls[0]
    )

    assert first_init == {
        "alpha": 0.05,
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 1,
        "min_samples_leaf": 5,
        "subsample": 1.0,
        "random_state": 42,
    }


def test_run_experiment_does_not_mutate_frame(
    canonical_frame,
    fake_model,
):
    before = canonical_frame.copy(
        deep=True
    )

    tuning.run_experiment(
        canonical_frame,
        tuning.EXPERIMENTS[0],
    )

    pd.testing.assert_frame_equal(
        canonical_frame,
        before,
    )


def test_run_all_uses_locked_grid(
    monkeypatch,
):
    dummy_frame = pd.DataFrame(
        {
            "target_return": [0.0],
        },
        index=[
            pd.Timestamp("2020-01-01")
        ],
    )

    observed_ids = []

    def fake_load():
        return dummy_frame.copy()

    def fake_run(
        frame,
        config,
    ):
        observed_ids.append(
            config.experiment_id
        )

        predictions = pd.DataFrame(
            {
                "experiment_id": [
                    config.experiment_id
                ]
            }
        )

        metrics = {
            "experiment_id":
                config.experiment_id
        }

        return predictions, metrics

    monkeypatch.setattr(
        tuning,
        "load_feature_target_frame",
        fake_load,
    )

    monkeypatch.setattr(
        tuning,
        "run_experiment",
        fake_run,
    )

    predictions, summary = (
        tuning.run_all_experiments()
    )

    assert observed_ids == [
        "G01",
        "G02",
        "G03",
        "G04",
    ]

    assert predictions[
        "experiment_id"
    ].tolist() == observed_ids

    assert summary[
        "experiment_id"
    ].tolist() == observed_ids
