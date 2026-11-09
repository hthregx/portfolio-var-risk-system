"""Independent Phase 3 A-only nested temporal CV helpers. No holdout reads."""
from __future__ import annotations

import numpy as np
import pandas as pd

ALPHA = 0.05
OUTER_TRAIN0 = 579
OUTER_VALID = 200
OUTER_FOLDS = 3
# Fixed a priori, to avoid huge data-driven hyperparameter sweeps.
SELECTION_MIN_REL_GAIN = 0.01
SELECTION_MIN_FOLD_WINS = 2


def pinball(y, q, alpha: float = ALPHA) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if y.ndim != 1 or q.shape != y.shape or y.size == 0 or not (0 < alpha < 1):
        raise ValueError('Invalid quantile loss arguments')
    if not np.isfinite(y).all() or not np.isfinite(q).all():
        raise ValueError('Nonfinite quantile loss arguments')
    return np.maximum(alpha * (y-q), (alpha-1) * (y-q))


def outer_splits(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Three non-overlapping validation blocks inside the 1179-day DEVELOPMENT ONLY."""
    if n != 1179:
        raise ValueError('Unexpected dev observations; refusing to change temporal protocol')
    out = []
    for i in range(OUTER_FOLDS):
        stop = OUTER_TRAIN0 + i * OUTER_VALID
        tr = np.arange(stop)
        va = np.arange(stop, stop + OUTER_VALID)
        if tr[-1] >= va[0] or va[-1] >= n:
            raise ValueError('Outer fold leakage or overflow')
        out.append((tr, va))
    if not np.array_equal(np.concatenate([v for _,v in out]), np.arange(OUTER_TRAIN0,n)):
        raise ValueError('Missing or duplicated outer validation dates')
    return out


def inner_splits(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Chronological three-fold tuning inside *each* outer training period."""
    if n < 500 or n > 1179:
        raise ValueError('Outer training size invalid')
    width = min(120, n//6)
    start = n-3*width
    if start < 200:
        raise ValueError('Inner train history too short')
    out=[]
    for i in range(3):
        stop=start+i*width
        tr=np.arange(stop)
        va=np.arange(stop,stop+width)
        if tr[-1]>=va[0] or va[-1]>=n:
            raise ValueError('Inner fold leakage')
        out.append((tr,va))
    return out


def candidate_parameters() -> dict[str,dict]:
    base = dict(loss='quantile', alpha=.05, n_estimators=150,
                learning_rate=.03, max_depth=2, min_samples_leaf=10,
                subsample=1.0, random_state=42)
    # Bounded four-way challenge to the Phase 2 C1 contract.
    return {
        'C1_frozen': base,
        'shallow_leaf15': dict(base,n_estimators=120,max_depth=1,min_samples_leaf=15),
        'leaf20': dict(base,min_samples_leaf=20),
        'subsample080': dict(base,subsample=.8),
        'slow_regularized': dict(base,n_estimators=200,learning_rate=.02,
                                 min_samples_leaf=15,subsample=.9),
    }


def choose_from_inner(scores: pd.DataFrame) -> tuple[str, dict]:
    """Pre-registered rule: >=1% mean gain and >=2/3 wins vs C1 or retain C1.

    Do not peek at outer-fold predictions to select hyperparameters.
    """
    required={'candidate','inner_fold','pinball_loss'}
    if not required.issubset(scores.columns):
        raise ValueError('Inner score columns missing')
    if len(scores)!=15 or scores.duplicated(['candidate','inner_fold']).any():
        raise ValueError('Unexpected inner scores')
    pool=candidate_parameters()
    if set(scores.candidate)!=set(pool) or set(scores.inner_fold)!={1,2,3}:
        raise ValueError('Inner candidate/fold protocol changed')
    pivot=scores.pivot(index='inner_fold', columns='candidate',values='pinball_loss')
    if not np.isfinite(pivot.to_numpy(float)).all() or (pivot.to_numpy(float)<0).any():
        raise ValueError('Invalid inner loss')
    original=float(pivot['C1_frozen'].mean())
    qualified=[]
    details={}
    for k in pool:
        v=float(pivot[k].mean())
        wins=int((pivot[k] < pivot['C1_frozen'] - 1e-12).sum())
        gain=(original-v)/original if original>0 else 0.0
        details[k]=dict(mean_loss=v, gain_vs_c1_pct=100*gain,fold_wins_vs_c1=wins,
                        qualifies=(k!='C1_frozen' and gain>=SELECTION_MIN_REL_GAIN and wins>=SELECTION_MIN_FOLD_WINS))
        if details[k]['qualifies']:
            qualified.append(k)
    selected=min(qualified, key=lambda x:(details[x]['mean_loss'],x)) if qualified else 'C1_frozen'
    return selected,details


def validate_dev_frame(frame: pd.DataFrame, expected_features: tuple[str,...],
                       dev_end: str='2024-12-17') -> pd.DataFrame:
    columns=['target_date','as_of',*expected_features,'actual_return']
    if frame.columns.tolist()!=columns or len(frame)!=1647:
        raise ValueError('Unrecognized Phase 2 feature matrix')
    if frame.target_date.isna().any() or frame.as_of.isna().any():
        raise ValueError('Missing dates')
    if not frame.target_date.is_monotonic_increasing or frame.target_date.duplicated().any():
        raise ValueError('Target dates invalid')
    if not (frame.as_of < frame.target_date).all():
        raise ValueError('Lookahead in as_of')
    if not np.isfinite(frame[[*expected_features,'actual_return']].to_numpy(float)).all():
        raise ValueError('Nonfinite feature/target')
    dev=frame[frame.target_date <= pd.Timestamp(dev_end)].copy().reset_index(drop=True)
    if len(dev)!=1179 or len(frame)-len(dev)!=468:
        raise ValueError('Development split drift')
    if dev.target_date.max()!=pd.Timestamp(dev_end):
        raise ValueError('Last development trading date changed')
    return dev


def violation_summary(y, q) -> dict:
    y,q=np.asarray(y,float),np.asarray(q,float)
    pin=pinball(y,q)
    k=int((y<q).sum())
    return {'n':int(len(y)), 'pinball_loss':float(pin.mean()),
            'violations':k,'violation_rate':k/len(y),
            'average_var':float(np.maximum(0,-q).mean())}
