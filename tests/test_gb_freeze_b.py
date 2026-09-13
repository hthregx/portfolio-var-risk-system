from __future__ import annotations

import json

import pandas as pd

from scripts.audit_gb_freeze_b import (
    EXPECTED_CONFIG,
    EXPECTED_FEATURES,
    METADATA_PATH,
    audit_evaluation_protocol,
    audit_feature_semantics,
    audit_future_perturbation,
)
from src.gb_return_features import build_return_features


def test_frozen_g04_config():
    metadata = json.loads(
        METADATA_PATH.read_text()
    )
    gb = metadata["methods"]["gradient_boosting"]

    for key, expected in EXPECTED_CONFIG.items():
        assert gb[key] == expected

    assert gb["features"] == EXPECTED_FEATURES


def test_feature_semantics():
    returns, features = audit_feature_semantics()
    t = returns.index[70]

    assert list(features.columns) == EXPECTED_FEATURES

    assert (
        features.loc[t, "return_lag_1"]
        == returns.shift(1).loc[t]
    )
    assert (
        features.loc[t, "return_lag_2"]
        == returns.shift(2).loc[t]
    )
    assert (
        features.loc[t, "return_lag_5"]
        == returns.shift(5).loc[t]
    )


def test_rolling_vol_is_backward_looking():
    returns, features = audit_feature_semantics()
    t = returns.index[70]
    history = returns.shift(1)

    for window in (5, 20, 60):
        expected = (
            history
            .rolling(
                window,
                min_periods=window,
            )
            .std(ddof=1)
            .loc[t]
        )

        assert (
            features.loc[
                t,
                f"rolling_vol_{window}",
            ]
            == expected
        )


def test_drawdown_uses_prior_history():
    dates = pd.date_range(
        "2024-01-01",
        periods=100,
        freq="D",
    )

    returns = pd.Series(
        [0.01] * 40
        + [-0.20]
        + [0.01] * 59,
        index=dates,
        name="portfolio_simple_return",
        dtype=float,
    )

    original = build_return_features(
        returns
    )

    changed = returns.copy()
    changed.iloc[70:] += 0.5

    modified = build_return_features(
        changed
    )

    pd.testing.assert_series_equal(
        original.loc[
            :dates[70],
            "drawdown",
        ],
        modified.loc[
            :dates[70],
            "drawdown",
        ],
    )


def test_future_perturbation():
    returns, features = (
        audit_feature_semantics()
    )

    audit_future_perturbation(
        returns,
        features,
    )


def test_target_return_not_a_feature():
    _, features = audit_feature_semantics()

    forbidden = {
        "portfolio_simple_return",
        "actual_return",
        "target_return",
    }

    assert forbidden.isdisjoint(
        features.columns
    )

def test_evaluation_protocol_is_chronological():
    audit_evaluation_protocol()