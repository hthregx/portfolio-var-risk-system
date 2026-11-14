"""Independent retrospective audit of B's published Phase 2 baseline forecasts.

This is an audit utility, not an approval to tune on Phase 1 final dates.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

COLUMNS = (
    'model', 'fold', 'as_of', 'target_date', 'quantile_return',
    'var_return', 'actual_return',
)


def audited_frame(frame: pd.DataFrame, expected_name: str) -> pd.DataFrame:
    """Reject missing, duplicate, stale, ill-signed or shifted forecast rows."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError('Missing or empty predictions')
    if frame.columns.tolist() != list(COLUMNS):
        raise ValueError('Unexpected prediction schema')
    df = frame.copy()
    if df['model'].nunique() != 1 or df['model'].iat[0] != expected_name:
        raise ValueError('Model identifier does not match')
    for name in ('as_of', 'target_date'):
        df[name] = pd.to_datetime(df[name], format='%Y-%m-%d', errors='raise')
        if df[name].isna().any():
            raise ValueError('Missing forecast date')
    if df['target_date'].duplicated().any() or not df['target_date'].is_monotonic_increasing:
        raise ValueError('Target dates are duplicated or not increasing')
    if not (df['as_of'] < df['target_date']).all():
        raise ValueError('as_of is not before target')
    for name in ('quantile_return', 'var_return', 'actual_return'):
        df[name] = pd.to_numeric(df[name], errors='raise')
        if not np.isfinite(df[name].to_numpy(float)).all():
            raise ValueError(f'Nonfinite {name}')
    if (df['actual_return'] <= -1).any():
        raise ValueError('Invalid simple return')
    q = df['quantile_return'].to_numpy(float)
    v = df['var_return'].to_numpy(float)
    if not np.allclose(v, np.maximum(0.0, -q), rtol=0, atol=1e-12):
        raise ValueError('VaR sign does not match quantile')
    return df


def verify_target_alignment(frame: pd.DataFrame, reference: pd.Series) -> None:
    """Compare all B target/as-of/actual rows against A's immutable snapshot."""
    if not isinstance(reference.index, pd.DatetimeIndex) or reference.index.has_duplicates:
        raise ValueError('Invalid independent reference')
    df = frame.copy()
    locations = reference.index.get_indexer(df['target_date'])
    if (locations <= 0).any():
        raise ValueError('B target is missing or without history in A data')
    prev_dates = reference.index[locations - 1]
    if not np.array_equal(df['as_of'].to_numpy(), prev_dates.to_numpy()):
        raise ValueError('B as_of is not the previous trading session')
    expected_actual = reference.iloc[locations].to_numpy(float)
    observed_actual = df['actual_return'].to_numpy(float)
    if not np.allclose(expected_actual, observed_actual, atol=2e-12, rtol=1e-10):
        worst = float(np.max(np.abs(expected_actual - observed_actual)))
        raise ValueError(f'B realized returns differ from independent A data: {worst}')


def historical_quantiles(returns: np.ndarray, warmup: int = 250, alpha: float = .05) -> np.ndarray:
    r = np.asarray(returns, float)
    if r.ndim != 1 or len(r) <= warmup or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError('Invalid returns')
    return np.asarray([
        np.quantile(r[i - warmup:i], alpha, method='linear')
        for i in range(warmup, len(r))
    ])


def ewma_quantiles_independent(returns: np.ndarray, warmup: int = 250,
                               decay: float = .94, alpha: float = .05) -> np.ndarray:
    r = np.asarray(returns, float)
    if r.ndim != 1 or len(r) <= warmup or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError('Invalid returns')
    if not (0 < decay < 1 and 0 < alpha < 1):
        raise ValueError('Invalid alpha/decay')
    sigma2 = float(r[0] ** 2)
    z = NormalDist().inv_cdf(alpha)
    out = []
    for i in range(1, len(r)):
        if i >= warmup:
            out.append(z * np.sqrt(sigma2))
        sigma2 = decay * sigma2 + (1 - decay) * float(r[i] ** 2)
    return np.asarray(out, float)


def verify_quantiles(observed: pd.DataFrame, expected: np.ndarray, tol: float = 1e-9) -> dict:
    actual = observed['quantile_return'].to_numpy(float)
    expected = np.asarray(expected, float)
    if len(actual) != len(expected) or not np.isfinite(expected).all():
        raise ValueError('Independent forecast count/values differ')
    differences = np.abs(actual - expected)
    worst = float(differences.max())
    if worst > tol:
        i = int(np.argmax(differences))
        raise ValueError(f'Independent predictions mismatch; index={i}, max_abs={worst:.12g}')
    return {'n': len(actual), 'max_abs_difference': worst, 'mean_abs_difference': float(differences.mean())}


def b_five_expanding_indices(n: int):
    """Independent calculation of the fixed 5 expanding folds used by B."""
    if n < 6:
        raise ValueError('Insufficient folds')
    size = n // 6
    first = n - 5 * size
    return [(np.arange(first + i * size),
             np.arange(first + i * size, first + (i + 1) * size))
            for i in range(5)]
