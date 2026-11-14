"""Independent Phase 2 VaR backtesting diagnostics owned by A.

All coverage tests are descriptive retrospective diagnostics, not a
claim that a model is statistically proven to be well calibrated.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

ALPHA = 0.05
EXPECTED_COLUMNS = {
    "model", "fold", "as_of", "target_date", "quantile_return",
    "var_return", "actual_return", "violation",
}


def _xlogp(count: int, prob: float) -> float:
    """Log likelihood contribution; define 0*log(0) = 0."""
    if count == 0:
        return 0.0
    if not 0 < prob <= 1:
        return -math.inf
    return count * math.log(prob)


def _binomial_log_likelihood(failures: int, total: int, probability: float) -> float:
    return _xlogp(failures, probability) + _xlogp(total - failures, 1 - probability)


def kupiec_uc(violations, alpha: float = ALPHA) -> dict:
    """Kupiec (1995) unconditional coverage LR; chi-square(1) asymptotic p-value."""
    events = np.asarray(violations)
    if events.ndim != 1 or len(events) < 2:
        raise ValueError("At least two violation observations required")
    if not np.isin(events, [0, 1, False, True]).all():
        raise ValueError("Violation values must be boolean")
    if not 0 < alpha < 1:
        raise ValueError("Invalid alpha")
    n = int(len(events))
    n1 = int(events.astype(int).sum())
    mle = n1 / n
    lr = max(0.0, 2 * (_binomial_log_likelihood(n1, n, mle)
                        - _binomial_log_likelihood(n1, n, alpha)))
    p = math.erfc(math.sqrt(lr / 2))  # chi-square survival, df=1
    return {"n": n, "violations": n1, "violation_rate": mle,
            "expected_violations": n * alpha, "lr_uc": lr, "p_uc": p}


def christoffersen_ind(violations) -> dict:
    """Christoffersen (1998) first-order independence test, chi-square(1)."""
    v = np.asarray(violations)
    if v.ndim != 1 or len(v) < 3 or not np.isin(v, [0, 1, True, False]).all():
        raise ValueError("Need at least 3 boolean violations")
    v = v.astype(bool)
    a, b = v[:-1], v[1:]
    n00 = int(((~a) & (~b)).sum())
    n01 = int(((~a) & b).sum())
    n10 = int((a & (~b)).sum())
    n11 = int((a & b).sum())
    n0 = n00 + n01
    n1 = n10 + n11
    pi = (n01 + n11) / (n0 + n1)
    p01 = n01 / n0 if n0 else 0.0
    p11 = n11 / n1 if n1 else 0.0
    ll0 = (_xlogp(n01 + n11, pi) + _xlogp(n00 + n10, 1 - pi))
    ll1 = (_xlogp(n01, p01) + _xlogp(n00, 1 - p01)
           + _xlogp(n11, p11) + _xlogp(n10, 1 - p11))
    lr = max(0.0, 2 * (ll1 - ll0))
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "lr_ind": lr, "p_ind": math.erfc(math.sqrt(lr / 2))}


def pinball(actual, quantile, alpha: float = ALPHA) -> np.ndarray:
    y, q = np.asarray(actual, float), np.asarray(quantile, float)
    if y.ndim != 1 or y.shape != q.shape or y.size == 0:
        raise ValueError("Invalid actual/quantile shape")
    if not (np.isfinite(y).all() and np.isfinite(q).all()):
        raise ValueError("Actual and quantile must be finite")
    return np.where(y < q, (1 - alpha) * (q - y), alpha * (y - q))


def validate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Strict schema, target dating, sign and violation rule validation."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Empty predictions")
    if not EXPECTED_COLUMNS.issubset(df.columns):
        raise ValueError(f"Missing prediction columns: {EXPECTED_COLUMNS - set(df.columns)}")
    frame = df.copy()
    for column in ("target_date", "as_of"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
        if frame[column].isna().any():
            raise ValueError("Missing prediction dates")
    if not (frame["as_of"] < frame["target_date"]).all():
        raise ValueError("Forecast date not strictly before target")
    if frame[["model", "target_date"]].duplicated().any():
        raise ValueError("Duplicate model/target pairs")
    for column in ("quantile_return", "var_return", "actual_return"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(float)).all():
            raise ValueError("Nonfinite prediction values")
    if (frame["actual_return"] <= -1).any():
        raise ValueError("Invalid simple returns")
    if not np.allclose(frame["var_return"], np.maximum(0, -frame["quantile_return"]), atol=1e-12, rtol=0):
        raise ValueError("VaR sign/quantile mismatch")
    if not np.isin(frame["violation"], [0, 1, False, True]).all():
        raise ValueError("Nonboolean violation column")
    violations = frame["actual_return"].to_numpy() < frame["quantile_return"].to_numpy()
    if not np.array_equal(frame["violation"].to_numpy().astype(bool), violations):
        raise ValueError("Violate rule actual < quantile does not match")
    return frame


def backtest_one(df: pd.DataFrame, alpha: float = ALPHA) -> dict:
    frame = validate_predictions(df).sort_values("target_date")
    if frame["model"].nunique() != 1 or frame["target_date"].duplicated().any():
        raise ValueError("Run backtest on exactly one model")
    v = (frame["actual_return"] < frame["quantile_return"]).to_numpy()
    uc = kupiec_uc(v, alpha=alpha)
    ind = christoffersen_ind(v)
    lrcc = uc["lr_uc"] + ind["lr_ind"]
    # Chi-square(2) survival, asymptotic.
    return {"model": str(frame["model"].iloc[0]), **uc, **ind,
            "lr_cc": lrcc, "p_cc": math.exp(-lrcc / 2),
            "mean_pinball_loss": float(pinball(frame["actual_return"], frame["quantile_return"], alpha).mean()),
            "mean_var": float(frame["var_return"].mean()),
            "first_target": frame["target_date"].min().date().isoformat(),
            "last_target": frame["target_date"].max().date().isoformat()}


def validate_paired(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Comparable if every target date AND every realized return is identical."""
    a = validate_predictions(a).sort_values("target_date").reset_index(drop=True)
    b = validate_predictions(b).sort_values("target_date").reset_index(drop=True)
    if len(a) != len(b) or not a["target_date"].equals(b["target_date"]):
        raise ValueError("Different target dates; models not directly comparable")
    if not a["as_of"].equals(b["as_of"]):
        raise ValueError("Different forecast origins; models not directly comparable")
    if not np.allclose(a["actual_return"], b["actual_return"], atol=1e-13, rtol=0):
        raise ValueError("Different realized returns; models not directly comparable")
    return a, b
