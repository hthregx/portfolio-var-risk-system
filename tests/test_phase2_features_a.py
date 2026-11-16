import numpy as np
import pandas as pd
import pytest

from src.gb_return_features import build_return_features
from src.phase2.features_a import (
    ALL_FEATURES,
    BASELINE,
    build_features_a,
)


def sample():
    dates = pd.bdate_range("2025-01-01", periods=150)
    x = np.arange(150, dtype=float)

    values = (
        0.013 * np.sin(x * 0.37)
        + 0.006 * np.cos(x * 0.11)
        - 0.001
    )

    return pd.Series(
        values,
        index=dates,
        name="portfolio_simple_return",
    )


def test_feature_count_and_order():
    result = build_features_a(sample())

    assert result.shape == (150, 12)
    assert tuple(result.columns) == ALL_FEATURES


def test_baseline_is_identical_to_frozen_builder():
    r = sample()

    expected = build_return_features(r)
    actual = build_features_a(r)

    pd.testing.assert_frame_equal(
        actual[list(BASELINE)],
        expected,
    )

    pd.testing.assert_frame_equal(
        build_features_a(r, extended=False),
        expected,
    )


def test_60_day_warmup_is_preserved():
    result = build_features_a(sample())

    assert result.iloc[:60].isna().any(axis=1).all()
    assert result.iloc[60:].notna().all().all()

    assert len(result.dropna()) == 90


def test_manual_values_use_prior_observations():
    r = sample()
    feature = build_features_a(r).iloc[80]

    prior = r.iloc[:80].to_numpy()

    momentum5 = np.prod(1 + prior[-5:]) - 1
    momentum20 = np.prod(1 + prior[-20:]) - 1

    downside = np.sqrt(
        np.mean(np.minimum(prior[-20:], 0.0) ** 2)
    )

    q05 = np.quantile(
        prior[-60:],
        0.05,
        method="linear",
    )

    value_range = (
        np.max(prior[-20:]) - np.min(prior[-20:])
    )

    assert feature["momentum_5"] == pytest.approx(momentum5)
    assert feature["momentum_20"] == pytest.approx(momentum20)
    assert feature["downside_vol_20"] == pytest.approx(downside)
    assert feature["rolling_q05_60"] == pytest.approx(q05)
    assert feature["rolling_range_20"] == pytest.approx(value_range)


def test_target_return_cannot_affect_its_features():
    original = sample()
    target = 80

    before = build_features_a(original)

    modified = original.copy()
    modified.iloc[target] = 0.80

    after = build_features_a(modified)

    pd.testing.assert_series_equal(
        before.iloc[target],
        after.iloc[target],
    )

    assert (
        before.iloc[target + 1]["return_lag_1"]
        != after.iloc[target + 1]["return_lag_1"]
    )


def test_future_returns_cannot_change_earlier_features():
    original = sample()
    before = build_features_a(original)

    changed = original.copy()
    changed.iloc[90:] = 0.60

    after = build_features_a(changed)

    pd.testing.assert_frame_equal(
        before.iloc[:91],
        after.iloc[:91],
    )


def test_deterministic_and_no_input_mutation():
    original = sample()
    old = original.copy(deep=True)

    a = build_features_a(original)
    b = build_features_a(original)

    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_series_equal(original, old)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_invalid_returns_rejected(bad):
    r = sample()
    r.iloc[10] = bad

    with pytest.raises(ValueError):
        build_features_a(r)
