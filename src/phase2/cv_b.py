
import numpy as np
import pandas as pd


def expanding_splits(dates, n_splits=5):
    """Trả về các cặp chỉ số train/validation theo thứ tự thời gian."""
    dates = pd.DatetimeIndex(dates)

    if dates.hasnans or dates.has_duplicates:
        raise ValueError("Ngày không được thiếu hoặc trùng.")
    if not dates.is_monotonic_increasing:
        raise ValueError("Ngày phải tăng dần; không tự sắp xếp.")
    if (
        isinstance(n_splits, bool)
        or not isinstance(n_splits, int)
        or n_splits < 2
    ):
        raise ValueError("n_splits phải là số nguyên >= 2.")

    n = len(dates)
    val_size = n // (n_splits + 1)
    if val_size == 0:
        raise ValueError("Không đủ mẫu để chia CV.")

    # Phần dư được giữ trong tập train đầu tiên.
    first_val = n - n_splits * val_size
    splits = []

    for start in range(first_val, n, val_size):
        train = np.arange(start)
        val = np.arange(start, start + val_size)

        if dates[train[-1]] >= dates[val[0]]:
            raise ValueError("Train phải kết thúc trước validation.")

        splits.append((train, val))

    return splits