"""Additional causal, target-date aligned portfolio-return features for Phase 3.

No target return is used as a feature. All engineered windows end at t-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.phase2.features_a import ALL_FEATURES, build_features_a

EXTRA_TO_18 = (
    "return_lag_3",
    "return_lag_10",
    "rolling_mean_10",
    "rolling_vol_10",
    "upside_vol_20",
    "rolling_q95_60",
)
EXTRA_TO_25 = (
    "rolling_skew_20",
    "rolling_kurt_60",
    "rolling_q01_60",
    "vol_ratio_5_20",
    "vol_ratio_20_60",
    "mean_abs_return_20",
    "ewma_vol_094",
)
FEATURES_18 = tuple(ALL_FEATURES) + EXTRA_TO_18
FEATURES_25 = FEATURES_18 + EXTRA_TO_25


def build_phase3_features(returns: pd.Series) -> pd.DataFrame:
    """Return 25 features; the first 12 reproduce the Phase 2 implementation.

    For forecast target date t, every feature is computed only from r at
    prior dates. The 60-observation warmup is intentionally left as NaN.
    No imputation or global normalization occurs.
    """
    baseline = build_features_a(returns)
    if tuple(baseline.columns) != tuple(ALL_FEATURES):
        raise RuntimeError("Phase 2 12-feature contract changed")
    history = returns.astype("float64").shift(1)
    r = pd.DataFrame(index=baseline.index)
    r["return_lag_3"] = returns.shift(3)
    r["return_lag_10"] = returns.shift(10)
    r["rolling_mean_10"] = history.rolling(10, min_periods=10).mean()
    r["rolling_vol_10"] = history.rolling(10, min_periods=10).std(ddof=1)
    r["upside_vol_20"] = np.sqrt(
        history.clip(lower=0).pow(2).rolling(20, min_periods=20).mean()
    )
    r["rolling_q95_60"] = history.rolling(60, min_periods=60).quantile(.95)
    r["rolling_skew_20"] = history.rolling(20, min_periods=20).skew()
    r["rolling_kurt_60"] = history.rolling(60, min_periods=60).kurt()
    r["rolling_q01_60"] = history.rolling(60, min_periods=60).quantile(.01)
    # Stabilize low-volatility denominators without filling missing values.
    r["vol_ratio_5_20"] = baseline["rolling_vol_5"] / (
        baseline["rolling_vol_20"] + 1e-12
    )
    r["vol_ratio_20_60"] = baseline["rolling_vol_20"] / (
        baseline["rolling_vol_60"] + 1e-12
    )
    r["mean_abs_return_20"] = history.abs().rolling(20, min_periods=20).mean()
    # A fixed-decay trailing variance proxy; note initialization convention.
    r["ewma_vol_094"] = np.sqrt(
        history.pow(2).ewm(alpha=.06, adjust=False).mean()
    )
    out = pd.concat([baseline, r], axis=1).astype("float64")
    if tuple(out.columns) != FEATURES_25 or len(set(out.columns)) != 25:
        raise RuntimeError("Unexpected Phase 3 feature order")
    return out


def select_feature_set(full: pd.DataFrame, count: int) -> pd.DataFrame:
    if count == 18:
        return full.loc[:, list(FEATURES_18)].copy()
    if count == 25:
        return full.loc[:, list(FEATURES_25)].copy()
    raise ValueError("Only Phase 3 feature counts 18 and 25 are supported")
