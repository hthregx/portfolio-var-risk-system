"""A-only Phase3 diagnostics tests: no dependence on external KBS network."""
import numpy as np
import pandas as pd
import pytest

from src.phase3.diagnostics_a import (
    pinball, diagnostics, expanding_fold_boundaries,
    assert_feature_frames, high_correlations, make_ranking,
)


def mock_frames():
    before = pd.bdate_range(end="2024-12-17", periods=1179)
    after = pd.bdate_range(start="2024-12-18", periods=468)
    dates = before.append(after)
    asof = dates - pd.offsets.BDay(1)
    frames = {}
    names = {}
    for count in (12, 18, 25):
        cols = [f"f{x}" for x in range(count)]
        names[count] = cols
        frame = pd.DataFrame({"target_date":dates, "as_of":asof})
        for j, name in enumerate(cols):
            frame[name] = np.sin((np.arange(len(dates)) + j) / (7 + j))
        frame["actual_return"] = .01 * np.cos(np.arange(len(dates))/11)
        frames[count] = frame
    return frames, names


def test_pinball_by_hand_at_five_percent():
    result = pinball([-0.04, .02], [-.02, -.02])
    np.testing.assert_allclose(result, [.019, .002], atol=1e-12)


def test_pinball_strict_validation():
    with pytest.raises(ValueError):
        pinball([1, 2], [3])
    with pytest.raises(ValueError):
        pinball([float("nan")], [0])
    with pytest.raises(ValueError):
        pinball([0], [0], 1.0)


def test_diagnostics_violations_strict_less_than():
    r = diagnostics([-.05, -.02], [-.02, -.02], [-.05, -.03, 0], [-.05, -.02, -.02])
    assert r["violations"] == 1  # equality is NOT a violation
    assert r["violation_rate"] == pytest.approx(1 / 3)
    assert r["val_pinball"] > 0
    assert r["train_pinball"] >= 0
    assert r["avg_var"] == pytest.approx(.03)


def test_five_folds_equal_to_frozen_phase2():
    splits = expanding_fold_boundaries(1179)
    assert [len(tr) for tr, _ in splits] == [199, 395, 591, 787, 983]
    assert all(len(va) == 196 for _, va in splits)
    assert splits[-1][1][-1] == 1178
    for tr, va in splits:
        assert tr[-1] < va[0]


def test_invalid_number_of_folds_or_small_dev():
    with pytest.raises(ValueError):
        expanding_fold_boundaries(1179, n_splits=4)
    with pytest.raises(ValueError):
        expanding_fold_boundaries(220)


def test_12_18_25_feature_alignment():
    f, n = mock_frames()
    dev, length = assert_feature_frames(f, n)
    assert length == 1179
    assert all(x.shape[0] == length for x in dev.values())
    assert all((x.target_date <= pd.Timestamp("2024-12-17")).all() for x in dev.values())


def test_alignment_rejects_label_change():
    f, n = mock_frames()
    f[25].loc[50, "actual_return"] += .003
    with pytest.raises((ValueError, AssertionError)):
        assert_feature_frames(f, n)


def test_alignment_rejects_date_shift():
    f, n = mock_frames()
    f[18].loc[50, "as_of"] = f[18].loc[50, "target_date"]
    with pytest.raises(ValueError):
        assert_feature_frames(f, n)


def test_alignment_rejects_target_nan():
    f, n = mock_frames()
    f[25].loc[100, "f17"] = np.nan
    with pytest.raises(ValueError):
        assert_feature_frames(f, n)


def test_high_correlation_screen_positive_and_empty():
    q = pd.DataFrame({"a":np.arange(20), "b":2*np.arange(20), "c":np.sin(np.arange(20))})
    r = high_correlations(q, ["a","b","c"])
    assert (r.feature_a == "a").any() and (r.feature_b == "b").any()
    assert r.iloc[0].abs_pearson == pytest.approx(1.0)
    e = high_correlations(q, ["c"])
    assert len(e) == 0 and e.columns.tolist() == ["feature_a","feature_b","abs_pearson"]


def test_high_correlation_invalid_threshold():
    with pytest.raises(ValueError):
        high_correlations(pd.DataFrame({"a":[1,2]}), ["a"], 1.0)


def test_make_ranking_stable_fold_comparison():
    rows = []
    for fold in range(1, 6):
        for size in (12,18,25):
            loss = {12:.020,18:.019,25:.021}[size] + fold*.0001
            rows.append({"fold":fold,"feature_count":size,"train_pinball":.01,
                         "val_pinball":loss,"generalization_gap":loss-.01,
                         "violation_rate":.05})
    result = make_ranking(pd.DataFrame(rows))
    assert result.iloc[0].feature_count == 18
    assert result.iloc[0].fold_wins_vs_12 == 5
    assert result.iloc[0].improvement_vs_12_pct > 0
    assert result[result.feature_count==25].iloc[0].improvement_vs_12_pct < 0


def test_make_ranking_rejects_missing_folds():
    frame = pd.DataFrame({"feature_count":[12,18,25], "fold":[1,1,1],
                          "val_pinball":[.02,.01,.03],"train_pinball":[.01,.01,.01],
                          "generalization_gap":[.01,0,.02],"violation_rate":[.05,.05,.05]})
    with pytest.raises(ValueError):
        make_ranking(frame)
