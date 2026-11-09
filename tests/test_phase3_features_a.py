"""Phase3 causal feature tests: no future data, baseline drift or silent NaNs."""
import numpy as np
import pandas as pd
import pytest

from src.phase2.features_a import ALL_FEATURES, build_features_a
from src.phase3.features_a import (
    EXTRA_TO_18, EXTRA_TO_25, FEATURES_18, FEATURES_25,
    build_phase3_features, select_feature_set,
)


def synthetic(n=240):
    x = np.arange(n, dtype=float)
    values = .012 * np.sin(x / 4.2) + .009 * np.cos(x / 11.0) + .000023 * (x % 17)
    return pd.Series(values, index=pd.bdate_range("2023-01-02", periods=n), name="portfolio_simple_return")


def test_feature_contract_unique_counts():
    assert len(ALL_FEATURES) == 12
    assert len(FEATURES_18) == 18
    assert len(FEATURES_25) == 25
    assert len(set(FEATURES_25)) == 25
    assert FEATURES_25[:12] == tuple(ALL_FEATURES)
    assert FEATURES_18[12:] == EXTRA_TO_18
    assert FEATURES_25[18:] == EXTRA_TO_25


def test_first_12_values_unchanged():
    r = synthetic()
    p3 = build_phase3_features(r)
    p2 = build_features_a(r)
    pd.testing.assert_frame_equal(p3[list(ALL_FEATURES)], p2, check_exact=True)


def test_exact_lags_and_rolling_formulas():
    r = synthetic()
    f = build_phase3_features(r)
    ix = 150
    hist = r.iloc[:ix]
    assert f.iloc[ix].return_lag_3 == pytest.approx(r.iloc[ix-3])
    assert f.iloc[ix].return_lag_10 == pytest.approx(r.iloc[ix-10])
    assert f.iloc[ix].rolling_mean_10 == pytest.approx(hist.iloc[-10:].mean())
    assert f.iloc[ix].rolling_q95_60 == pytest.approx(hist.iloc[-60:].quantile(.95))
    assert f.iloc[ix].rolling_q01_60 == pytest.approx(hist.iloc[-60:].quantile(.01))
    assert f.iloc[ix].mean_abs_return_20 == pytest.approx(hist.iloc[-20:].abs().mean())
    assert f.iloc[ix].rolling_skew_20 == pytest.approx(hist.iloc[-20:].skew())
    assert f.iloc[ix].rolling_kurt_60 == pytest.approx(hist.iloc[-60:].kurt())


def test_ewma_uses_only_prior_squared_returns():
    r = synthetic()
    f = build_phase3_features(r)
    expected = np.sqrt(r.shift(1).pow(2).ewm(alpha=.06, adjust=False).mean())
    np.testing.assert_allclose(f.ewma_vol_094.iloc[65:], expected.iloc[65:], rtol=1e-12)


def test_no_leakage_if_target_return_changes():
    a = synthetic()
    before = build_phase3_features(a)
    b = a.copy()
    b.iloc[141] += .25
    after = build_phase3_features(b)
    pd.testing.assert_frame_equal(before.iloc[:142], after.iloc[:142], check_exact=False, rtol=1e-12, atol=1e-12)
    assert not np.allclose(before.iloc[143:].to_numpy(), after.iloc[143:].to_numpy(), equal_nan=True)


def test_no_leakage_if_future_returns_change():
    a = synthetic()
    before = build_phase3_features(a)
    b = a.copy()
    b.iloc[180:] = .04
    after = build_phase3_features(b)
    pd.testing.assert_frame_equal(before.iloc[:181], after.iloc[:181], check_exact=False, rtol=1e-12, atol=1e-12)


def test_60_warmup_then_finite_for_regular_series():
    f = build_phase3_features(synthetic())
    assert len(f) == 240
    assert f.iloc[:60].isna().any(axis=1).all()
    assert f.iloc[60:].notna().all().all()
    assert np.isfinite(f.iloc[60:].to_numpy()).all()
    assert f.index.equals(synthetic().index)


def test_select_only_fixed_18_and_25():
    full = build_phase3_features(synthetic())
    assert select_feature_set(full, 18).columns.tolist() == list(FEATURES_18)
    assert select_feature_set(full, 25).columns.tolist() == list(FEATURES_25)
    with pytest.raises(ValueError):
        select_feature_set(full, 19)


def test_invalid_source_fails():
    with pytest.raises(ValueError):
        build_phase3_features(synthetic().iloc[::-1])
    bad = synthetic()
    bad.iloc[20] = np.nan
    with pytest.raises(ValueError):
        build_phase3_features(bad)


def test_target_and_future_perturbation_not_influence_prior_features():
    r = synthetic()
    base = build_phase3_features(r)
    changed = r.copy()
    changed.iloc[120:] = changed.iloc[120:] * (-1)
    other = build_phase3_features(changed)
    pd.testing.assert_frame_equal(base.iloc[:121], other.iloc[:121], check_exact=False, rtol=1e-12, atol=1e-12)
