import numpy as np
import pandas as pd
import pytest

from src.phase2.aligned_runner_b import (
    BASELINE, FEATURES, baseline_quantiles, build_features_b, compare_reference,
    fit_predict, load_snapshot, locked_split, prediction_frame, run, summarize,
    verify_reference_hash,
)


def sample_returns(n=360):
    values = np.random.default_rng(42).normal(0, .015, n)
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=n),
                     name="portfolio_simple_return")


def test_features_have_exact_schema_and_causal_extra_arithmetic():
    r = sample_returns()
    f = build_features_b(r)
    assert list(f.columns) == FEATURES
    assert f.notna().all(axis=1).sum() == len(r) - 60
    t = 100
    assert f.momentum_5.iloc[t] == pytest.approx(np.prod(1 + r.iloc[t-5:t]) - 1)
    assert f.downside_vol_20.iloc[t] == pytest.approx(np.sqrt(np.mean(np.minimum(r.iloc[t-20:t], 0) ** 2)))
    assert f.rolling_q05_60.iloc[t] == pytest.approx(np.quantile(r.iloc[t-60:t], .05))
    assert f.rolling_range_20.iloc[t] == pytest.approx(np.ptp(r.iloc[t-20:t]))
    changed = r.copy()
    changed.iloc[t:] = .5
    pd.testing.assert_frame_equal(f.iloc[:t+1], build_features_b(changed).iloc[:t+1])


def test_baselines_use_raw_history_and_exclude_target_return():
    r = sample_returns()
    before = baseline_quantiles(r)
    changed = r.copy()
    changed.iloc[260:] = .5
    after = baseline_quantiles(changed)
    for name in before:
        pd.testing.assert_series_equal(before[name].iloc[:11], after[name].iloc[:11])
    assert before["historical_w250"].iloc[0] == pytest.approx(np.quantile(r.iloc[:250], .05))


def test_gb_ignores_validation_labels_and_enforces_temporal_split():
    r = sample_returns()
    f = build_features_b(r).iloc[60:].copy()
    f["target_date"] = f.index
    f["as_of"] = pd.Series(r.index, index=r.index).shift(1).loc[f.index]
    f["actual_return"] = r.loc[f.index]
    train, valid = f.iloc[:200], f.iloc[200:210]
    before = fit_predict("gb_G04_7f", train, valid, "dev_1")
    changed = valid.copy()
    changed["actual_return"] = .5
    after = fit_predict("gb_G04_7f", train, changed, "dev_1")
    np.testing.assert_array_equal(before.quantile_return, after.quantile_return)
    with pytest.raises(ValueError, match="Training"):
        fit_predict("gb_G04_7f", valid, train, "dev_1")


def test_metrics_match_hand_calculation_including_equality():
    valid = pd.DataFrame(dict(as_of=pd.to_datetime(["2020-01-01"]*3),
                              target_date=pd.date_range("2020-01-02", periods=3),
                              actual_return=[-.03, -.02, .01]))
    pred = prediction_frame("toy", "holdout", valid, [-.02]*3)
    result = summarize(pred).iloc[0]
    assert result.violations == 1  # equality is not an exception
    assert result.pinball_loss == pytest.approx((.0095 + 0 + .0015) / 3)
    assert result.average_var == pytest.approx(.02)


def test_reference_requires_equal_dates_origins_folds_and_actuals():
    valid = pd.DataFrame(dict(as_of=pd.to_datetime(["2020-01-01"]),
                              target_date=pd.to_datetime(["2020-01-02"]), actual_return=[-.03]))
    obs = prediction_frame("gb_G04_7f", "dev_1", valid, [-.02])
    ref = obs.copy()
    ref["model"] += "_a"
    assert compare_reference(obs, ref)[0]["status"] == "PASS"
    for column, value in [("actual_return", .1), ("as_of", "2019-12-31"),
                          ("target_date", "2020-01-03"), ("fold", "dev_2")]:
        bad = ref.copy()
        bad[column] = value
        with pytest.raises(ValueError):
            compare_reference(obs, bad)
    bad = ref.copy()
    bad["quantile_return"] += .001
    assert compare_reference(obs, bad)[0]["status"] == "MISMATCH"


def test_snapshot_hash_and_existing_output_are_rejected(tmp_path):
    p = tmp_path / "returns.csv"
    p.write_text("changed data")
    m = tmp_path / "manifest.json"
    m.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        load_snapshot(p, m)
    with pytest.raises(FileExistsError):
        run(p, m, tmp_path)


def test_wrong_development_size_is_rejected():
    frame = pd.DataFrame({"target_date": pd.bdate_range("2020-01-01", periods=100)})
    with pytest.raises(ValueError, match="sizes"):
        locked_split(frame)


def test_git_line_endings_allowed_but_content_changes_rejected(tmp_path):
    import hashlib
    p = tmp_path / "reference.csv"
    p.write_bytes(b"a,b\n1,2\n")
    expected = hashlib.sha256(b"a,b\r\n1,2\r\n").hexdigest()
    assert verify_reference_hash(p, expected) == "git_line_endings_restored_CRLF"
    p.write_bytes(b"a,b\n1,3\n")
    with pytest.raises(ValueError, match="hash"):
        verify_reference_hash(p, expected)