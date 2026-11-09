"""Phase 3 A-only quantile-CV diagnostics. No Phase 2 state changes."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinball(y, quantile, alpha: float = ALPHA) -> np.ndarray:
    """Per-observation pinball loss for the lower alpha quantile."""
    y = np.asarray(y, dtype=np.float64)
    quantile = np.asarray(quantile, dtype=np.float64)
    if not (0 < alpha < 1) or y.ndim != 1 or quantile.shape != y.shape or y.size == 0:
        raise ValueError("Invalid pinball inputs")
    if not (np.isfinite(y).all() and np.isfinite(quantile).all()):
        raise ValueError("Nonfinite pinball inputs")
    return np.maximum(alpha * (y - quantile), (alpha - 1) * (y - quantile))


def diagnostics(y_train, q_train, y_val, q_val) -> dict:
    """Compute train and validation metrics without mixing their observations."""
    train_loss = float(pinball(y_train, q_train).mean())
    val_loss = float(pinball(y_val, q_val).mean())
    y_val = np.asarray(y_val, float)
    q_val = np.asarray(q_val, float)
    count = int(np.sum(y_val < q_val))
    return {
        "train_pinball": train_loss,
        "val_pinball": val_loss,
        "generalization_gap": val_loss - train_loss,
        "val_over_train_ratio": val_loss / train_loss if train_loss > 1e-12 else np.nan,
        "violations": count,
        "violation_rate": count / len(y_val),
        "avg_var": float(np.maximum(0, -q_val).mean()),
    }


def expanding_fold_boundaries(n: int, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fixed A2/A3 expanding folds, contiguous non-shuffled validation blocks."""
    if n_splits != 5 or n < 360:
        raise ValueError("Expected at least 360 development observations and five folds")
    width = n // 6
    first = n - 5 * width
    result = []
    for i in range(5):
        tr = np.arange(first + i * width)
        va = np.arange(first + i * width, first + (i + 1) * width)
        if tr.size == 0 or va.size == 0 or tr[-1] >= va[0]:
            raise ValueError("Invalid expanding fold")
        result.append((tr, va))
    return result


def assert_feature_frames(frames: dict, names_by_size: dict, cutoff: str = "2024-12-17") -> tuple[dict, int]:
    """Require identical T+1 targets, y, as_of and original 12 feature values."""
    if set(frames) != {12, 18, 25} or set(names_by_size) != {12, 18, 25}:
        raise ValueError("Expected three feature sets: 12, 18, 25")
    baseline = frames[12]
    expected_dates = None
    for k in (12, 18, 25):
        df = frames[k]
        cols = ["target_date", "as_of", *names_by_size[k], "actual_return"]
        if df.columns.tolist() != cols or len(df) != 1647:
            raise ValueError(f"Feature {k}: column contract or row count mismatch")
        if (df.target_date.isna().any() or df.as_of.isna().any()
                or df.target_date.duplicated().any() or not df.target_date.is_monotonic_increasing
                or not (df.as_of < df.target_date).all()):
            raise ValueError(f"Feature {k}: invalid dates/lookahead")
        if not np.isfinite(df[names_by_size[k] + ["actual_return"]].to_numpy(float)).all():
            raise ValueError(f"Feature {k}: NaN/Inf after warmup")
        if expected_dates is None:
            expected_dates = df.target_date
        else:
            if not (df.target_date.equals(expected_dates) and df.as_of.equals(baseline.as_of)):
                raise ValueError(f"Feature {k}: different target/as_of dates")
            np.testing.assert_allclose(df.actual_return, baseline.actual_return, rtol=0, atol=1e-13)
            np.testing.assert_allclose(df[names_by_size[12]], baseline[names_by_size[12]], rtol=1e-10, atol=1e-12)
    dev_mask = baseline.target_date <= pd.Timestamp(cutoff)
    dev_n = int(dev_mask.sum())
    if dev_n != 1179 or len(baseline) - dev_n != 468:
        raise ValueError("Development/legacy holdout split drift")
    return {k: v.loc[dev_mask].copy().reset_index(drop=True) for k, v in frames.items()}, dev_n


def high_correlations(frame: pd.DataFrame, columns: list[str], threshold: float = 0.95) -> pd.DataFrame:
    """Flag collinearity for human review, never delete features automatically."""
    if not 0 < threshold < 1:
        raise ValueError("Invalid threshold")
    corr = frame[columns].corr(method="pearson").abs()
    flagged = []
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            coef = float(corr.loc[left, right])
            if np.isfinite(coef) and coef >= threshold:
                flagged.append({"feature_a": left, "feature_b": right, "abs_pearson": coef})
    return pd.DataFrame(flagged, columns=["feature_a", "feature_b", "abs_pearson"]).sort_values(
        "abs_pearson", ascending=False, kind="stable"
    ).reset_index(drop=True)


def make_ranking(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = fold_metrics.groupby("feature_count", sort=True).agg(
        cv_pinball=("val_pinball", "mean"),
        mean_train_pinball=("train_pinball", "mean"),
        avg_gap=("generalization_gap", "mean"),
        avg_violation_rate=("violation_rate", "mean"),
        worst_fold_pinball=("val_pinball", "max"),
    ).reset_index()
    baseline = metrics.loc[metrics.feature_count == 12, "cv_pinball"]
    if len(baseline) != 1:
        raise ValueError("12-feature baseline absent")
    base = float(baseline.iloc[0])
    metrics["improvement_vs_12_pct"] = 100 * (base - metrics.cv_pinball) / base
    pivot = fold_metrics.pivot(index="fold", columns="feature_count", values="val_pinball")
    if list(pivot.columns) != [12, 18, 25] or len(pivot) != 5:
        raise ValueError("Expected 5 common folds of 12/18/25")
    metrics["fold_wins_vs_12"] = metrics.feature_count.map(
        {int(k): int((pivot[k] < pivot[12] - 1e-12).sum()) for k in pivot.columns}
    )
    metrics["folds_best"] = metrics.feature_count.map(
        pivot.idxmin(axis=1).value_counts().to_dict()
    ).fillna(0).astype(int)
    return metrics.sort_values(["cv_pinball", "feature_count"], kind="stable").reset_index(drop=True)
