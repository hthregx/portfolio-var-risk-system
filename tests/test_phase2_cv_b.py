import numpy as np
import pandas as pd
import pytest

from src.phase2.cv_b import expanding_splits


def test_five_expanding_folds():
    dates = pd.bdate_range("2020-01-01", periods=1645)
    splits = expanding_splits(dates, n_splits=5)

    assert len(splits) == 5
    expected_train_sizes = [275, 549, 823, 1097, 1371]

    previous_train = None
    previous_val = None

    for (train, val), expected_size in zip(
        splits, expected_train_sizes
    ):
        assert len(train) == expected_size
        assert len(val) == 274

        # Train liên tục từ đầu, validation ngay sau train.
        np.testing.assert_array_equal(
            train, np.arange(expected_size)
        )
        np.testing.assert_array_equal(
            val, np.arange(expected_size, expected_size + 274)
        )

        # Không chồng lấn và mọi ngày train đều trước validation.
        assert np.intersect1d(train, val).size == 0
        assert dates[train].max() < dates[val].min()

        if previous_train is not None:
            # Train mới chứa toàn bộ train và validation trước.
            np.testing.assert_array_equal(
                train,
                np.concatenate([previous_train, previous_val]),
            )

        previous_train = train
        previous_val = val

    # Validation không trùng, không bỏ sót mẫu sau train đầu.
    all_val = np.concatenate([val for _, val in splits])
    assert len(np.unique(all_val)) == len(all_val)
    np.testing.assert_array_equal(all_val, np.arange(275, 1645))


def test_small_hand_calculated_example():
    dates = pd.bdate_range("2026-01-01", periods=11)
    splits = expanding_splits(dates, n_splits=2)

    # 11 // 3 = 3 mẫu validation; train đầu = 11 - 2*3 = 5.
    expected = [
        ([0, 1, 2, 3, 4], [5, 6, 7]),
        ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10]),
    ]

    for (train, val), (want_train, want_val) in zip(
        splits, expected
    ):
        np.testing.assert_array_equal(train, want_train)
        np.testing.assert_array_equal(val, want_val)


@pytest.mark.parametrize(
    "dates",
    [
        ["2026-01-02", "2026-01-02", "2026-01-05"],
        ["2026-01-05", "2026-01-02", "2026-01-06"],
        ["2026-01-02", None, "2026-01-06"],
    ],
    ids=["duplicate_dates", "unsorted_dates", "missing_date"],
)
def test_reject_invalid_dates(dates):
    with pytest.raises(ValueError):
        expanding_splits(pd.DatetimeIndex(dates), n_splits=2)


@pytest.mark.parametrize("n_splits", [0, 1, -1, 2.5, True])
def test_reject_invalid_split_count(n_splits):
    dates = pd.bdate_range("2026-01-01", periods=20)

    with pytest.raises(ValueError):
        expanding_splits(dates, n_splits=n_splits)


@pytest.mark.parametrize("size", [0, 5])
def test_reject_insufficient_samples(size):
    dates = pd.bdate_range("2026-01-01", periods=size)

    with pytest.raises(ValueError):
        expanding_splits(dates, n_splits=5)