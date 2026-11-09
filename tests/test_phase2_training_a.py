import numpy as np
import pandas as pd
import pytest

from scripts.run_phase2_training_a import (
    ALPHA, DEV_END, CANDIDATES, pinball, evaluation,
    five_expanding_folds, split_dev_holdout, predict,
)


def synthetic_table():
    dates = pd.bdate_range("2020-01-01", periods=1500)
    n = len(dates)
    x = np.arange(n)
    frame = pd.DataFrame({
        "target_date": dates,
        "as_of": dates - pd.Timedelta(days=1),
        "actual_return": np.sin(x / 12) * .015,
    })
    for i, name in enumerate(CANDIDATES["gbv2_12f_c1_a"][0]):
        frame[name] = np.cos((x - i) / 15) * .013
    return frame


def test_pinball_hand_calculation():
    assert ALPHA == .05
    result = pinball([-0.02, .02], [-.01, -.01])
    np.testing.assert_allclose(result, [.0095, .0015])


def test_strict_violation_definition():
    frame = pd.DataFrame({"actual_return": [-.02, -.01, .01],
                          "quantile_return": [-.01, -.01, -.01]})
    m = evaluation(frame)
    assert m["violations"] == 1
    assert m["violation_rate"] == pytest.approx(1 / 3)


def test_five_fold_expanding_order():
    folds = five_expanding_folds(1154)
    assert len(folds) == 5
    assert all(tr[-1] < va[0] for tr, va in folds)
    assert all(np.intersect1d(tr, va).size == 0 for tr, va in folds)
    assert all(np.array_equal(folds[i+1][0], np.arange(folds[i][1][-1]+1))
               for i in range(4))


def test_holdout_is_excluded_from_training():
    dev, holdout = split_dev_holdout(synthetic_table())
    assert dev.target_date.max() <= pd.Timestamp(DEV_END)
    assert holdout.target_date.min() > pd.Timestamp(DEV_END)


def test_reject_overlapping_train_validation():
    data = synthetic_table()
    with pytest.raises(ValueError, match="leakage"):
        predict("gb_G04_7f_a", data.iloc[:400], data.iloc[399:405], "bad")


def test_deterministic_training():
    pytest.importorskip("sklearn")
    data = synthetic_table()
    a = predict("gb_G04_7f_a", data.iloc[:400], data.iloc[400:409], "trial")
    b = predict("gb_G04_7f_a", data.iloc[:400], data.iloc[400:409], "trial")
    pd.testing.assert_frame_equal(a, b)
    assert (a.as_of < a.target_date).all()
