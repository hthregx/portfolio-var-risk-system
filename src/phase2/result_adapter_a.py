"""A-owned adapters for completed Phase-2 retrospective VaR experiments.

Never relabel B's development predictions as A's independent holdout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRED_COLUMNS = (
    "model", "fold", "as_of", "target_date",
    "quantile_return", "var_return", "actual_return",
)
ALPHA = 0.05


def normalize_predictions(raw: pd.DataFrame, *, source: str, protocol: str) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("Prediction data must be a nonempty DataFrame")
    if source not in ("A", "B"):
        raise ValueError("Unknown prediction owner")
    if not isinstance(protocol, str) or not protocol.strip():
        raise ValueError("Forecast protocol must be declared")
    if not set(PRED_COLUMNS).issubset(raw.columns):
        raise ValueError("Missing core prediction columns")
    frame = raw[list(PRED_COLUMNS)].copy()
    for label in ("model", "fold"):
        if frame[label].isna().any() or not frame[label].map(
            lambda x: isinstance(x, str) and bool(x.strip())
        ).all():
            raise ValueError("Empty model or fold labels")
    for label in ("as_of", "target_date"):
        frame[label] = pd.to_datetime(frame[label], errors="raise")
        if frame[label].isna().any():
            raise ValueError("Missing forecast dates")
        if (frame[label].dt.time != pd.Timestamp("00:00").time()).any():
            raise ValueError("Dates must be calendar dates without timestamps")
    if not (frame.as_of < frame.target_date).all():
        raise ValueError("Forecast must predate target")
    if frame[["model", "target_date"]].duplicated().any():
        raise ValueError("Duplicate model/target dates")
    for label in ("quantile_return", "var_return", "actual_return"):
        frame[label] = pd.to_numeric(frame[label], errors="raise").astype(float)
        if not np.isfinite(frame[label].to_numpy()).all():
            raise ValueError("Nonfinite numerical output")
    if (frame.actual_return <= -1).any() or (frame.quantile_return < -1).any():
        raise ValueError("Invalid simple return")
    if not np.allclose(
        frame.var_return, np.maximum(0.0, -frame.quantile_return),
        atol=1e-12, rtol=0,
    ):
        raise ValueError("VaR magnitude/sign mismatch")
    if "violation" in raw.columns:
        provided = raw["violation"]
        if not provided.isin([0, 1, True, False]).all():
            raise ValueError("Invalid violation flags")
        expected = frame.actual_return.to_numpy() < frame.quantile_return.to_numpy()
        if not np.array_equal(provided.to_numpy().astype(bool), expected):
            raise ValueError("Incorrect reported exceptions")
    frame.insert(0, "source", source)
    frame.insert(1, "protocol", protocol)
    frame["violation"] = frame.actual_return < frame.quantile_return
    return frame.sort_values(["model", "target_date"]).reset_index(drop=True)


def matched_holdout(frame: pd.DataFrame, baseline: str, candidate: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reject comparisons on unequal dates or unequal realized observations."""
    if set(frame["source"].unique()) != {"A"}:
        raise ValueError("Only A's own paired holdout may be directly compared")
    if set(frame["protocol"].unique()) != {"fixed_fit_retrospective_holdout"}:
        raise ValueError("Mismatched forecast protocol")
    if set(frame["model"].unique()) != {baseline, candidate}:
        raise ValueError("Expected exactly two A holdout models")
    a = frame.loc[frame.model == baseline].sort_values("target_date").reset_index(drop=True)
    b = frame.loc[frame.model == candidate].sort_values("target_date").reset_index(drop=True)
    if not a.target_date.equals(b.target_date) or not a.as_of.equals(b.as_of):
        raise ValueError("Unequal forecast target/as-of dates")
    if not np.allclose(a.actual_return, b.actual_return, rtol=0, atol=1e-13):
        raise ValueError("Unequal observed returns")
    return a, b


def mean_pinball(y, q, alpha: float = ALPHA) -> float:
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    if y.ndim != 1 or q.shape != y.shape or y.size == 0:
        raise ValueError("Invalid paired pinball arrays")
    if not np.isfinite(y).all() or not np.isfinite(q).all():
        raise ValueError("Nonfinite paired pinball arrays")
    return float(np.where(y < q, (1 - alpha) * (q - y), alpha * (y - q)).mean())
