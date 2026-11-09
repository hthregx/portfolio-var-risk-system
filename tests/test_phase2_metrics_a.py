import math

import numpy as np
import pandas as pd
import pytest

from src.phase2.metrics_a import (
    backtest_one, christoffersen_ind, kupiec_uc,
    pinball, validate_paired, validate_predictions,
)


def fixture(n=35, model="a"):
    dates = pd.bdate_range("2025-01-01", periods=n)
    q = -np.full(n, 0.02)
    actual = np.linspace(-0.04, 0.02, n)
    return pd.DataFrame({"model": model, "fold": "holdout_fixed_fit",
                         "as_of": dates - pd.Timedelta(days=1),
                         "target_date": dates,
                         "quantile_return": q, "var_return": .02,
                         "actual_return": actual,
                         "violation": actual < q})


def test_pinball_hand_example():
    np.testing.assert_allclose(pinball([-0.02, 0.02], [-.01, -.01]), [.0095, .0015])


def test_kupiec_rate_and_pvalue():
    v = [True] + [False] * 19
    out = kupiec_uc(v)
    assert out["n"] == 20 and out["violations"] == 1
    assert out["lr_uc"] == pytest.approx(0)
    assert out["p_uc"] == pytest.approx(1)


def test_kupiec_zero_violations_finite():
    out = kupiec_uc([False] * 100)
    assert math.isfinite(out["lr_uc"])
    assert 0 <= out["p_uc"] <= 1


def test_christoffersen_transitions():
    # 0->0:1, 0->1:2, 1->0:2, 1->1:1
    out = christoffersen_ind([0, 0, 1, 1, 0, 1, 0])
    assert [out[k] for k in ("n00", "n01", "n10", "n11")] == [1, 2, 2, 1]
    assert 0 < out["p_ind"] <= 1


def test_christoffersen_independence_nonnegative():
    out = christoffersen_ind([False] * 40)
    assert out["lr_ind"] >= 0
    assert math.isfinite(out["p_ind"])


def test_backtest_quantile_rule():
    f = fixture()
    out = backtest_one(f)
    assert out["violations"] == int(f.violation.sum())
    assert out["n"] == len(f)
    assert 0 <= out["p_cc"] <= 1


def test_reject_current_day_origin():
    f = fixture()
    f.loc[0, "as_of"] = f.loc[0, "target_date"]
    with pytest.raises(ValueError, match="Forecast date"):
        validate_predictions(f)


def test_reject_wrong_violation_equality():
    f = fixture()
    f.loc[0, "actual_return"] = f.loc[0, "quantile_return"]
    f.loc[0, "violation"] = True
    with pytest.raises(ValueError, match="Violate rule"):
        validate_predictions(f)


def test_reject_wrong_sign_var():
    f = fixture()
    f.loc[0, "var_return"] = -.02
    with pytest.raises(ValueError, match="VaR sign"):
        validate_predictions(f)


def test_reject_nonidentical_holdout_dates():
    a, b = fixture(), fixture(model="b")
    b.loc[len(b) - 1, "target_date"] += pd.Timedelta(days=7)
    with pytest.raises(ValueError, match="Different target dates"):
        validate_paired(a, b)


def test_reject_nonidentical_realized_returns():
    a, b = fixture(), fixture(model="b")
    b.loc[0, "actual_return"] += 0.003
    b.loc[0, "violation"] = bool(b.loc[0, "actual_return"] < b.loc[0, "quantile_return"])
    with pytest.raises(ValueError, match="Different realized"):
        validate_paired(a, b)


def test_reject_duplicate_targets():
    f = fixture()
    f.loc[1, "target_date"] = f.loc[0, "target_date"]
    f.loc[1, "as_of"] = f.loc[0, "as_of"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_predictions(f)
