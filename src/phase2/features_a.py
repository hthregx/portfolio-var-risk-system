"""A-owned Phase 2 features; all information strictly precedes target."""

import numpy as np
import pandas as pd

from src.gb_return_features import build_return_features


BASELINE = (
    "return_lag_1",
    "return_lag_2",
    "return_lag_5",
    "rolling_vol_5",
    "rolling_vol_20",
    "rolling_vol_60",
    "drawdown",
)

EXTENDED = (
    "momentum_5",
    "momentum_20",
    "downside_vol_20",
    "rolling_q05_60",
    "rolling_range_20",
)

ALL_FEATURES = BASELINE + EXTENDED


def build_features_a(
    returns: pd.Series,
    *,
    extended: bool = True,
) -> pd.DataFrame:
    """Align lagged features to target trading dates.

    For target t, feature values use returns at dates < t.
    No imputation, no global scaling and no target leakage.
    """
    baseline = build_return_features(returns)

    if tuple(baseline.columns) != BASELINE:
        raise RuntimeError("Frozen baseline feature schema changed")

    if not extended:
        return baseline.copy()

    history = returns.astype("float64").shift(1)

    additional = pd.DataFrame(index=baseline.index)

    for window in (5, 20):
        additional[f"momentum_{window}"] = (
            np.expm1(
                np.log1p(history)
                .rolling(window, min_periods=window)
                .sum()
            )
        )

    negative_squared = history.clip(upper=0.0).pow(2)

    additional["downside_vol_20"] = np.sqrt(
        negative_squared.rolling(
            20, min_periods=20
        ).mean()
    )

    additional["rolling_q05_60"] = history.rolling(
        60, min_periods=60
    ).quantile(0.05)

    rolling_20 = history.rolling(20, min_periods=20)

    additional["rolling_range_20"] = (
        rolling_20.max() - rolling_20.min()
    )

    result = pd.concat(
        [baseline, additional],
        axis=1,
    )

    if tuple(result.columns) != ALL_FEATURES:
        raise RuntimeError("Unexpected Phase 2 feature schema")

    return result.astype("float64")
