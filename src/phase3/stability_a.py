"""Phase 3 A-only walk-forward refit schedules and out-of-fold stability metrics.

Forecasts target 5% portfolio-return quantiles (NOT direction or return level).
All labels and feature construction are inherited from the locked Phase 2 data.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from src.phase3.tuning_a import pinball

ALPHA = 0.05
FIRST_VALID = 579
VALID_N = 600
OUTER_LENGTH = 200
REFIT_EVERY = 20
ROLLING_TRAIN = 504
POLICIES = ('C1_frozen', 'nested_selected', 'expanding_refit20', 'rolling504_refit20')


def walkforward_schedule(n: int = 1179, *, interval: int = REFIT_EVERY,
                         first: int = FIRST_VALID, rolling: int | None = None):
    """For each target block, fit only on rows strictly before its first target."""
    if not isinstance(n, int) or not isinstance(interval, int) or interval < 1:
        raise ValueError('Invalid schedule dimensions')
    if not isinstance(first, int) or first < 200 or n <= first:
        raise ValueError('Insufficient causal training history')
    if rolling is not None and (not isinstance(rolling, int) or rolling < 200 or rolling > first):
        raise ValueError('Invalid rolling training window')
    blocks = []
    for lo in range(first, n, interval):
        hi = min(n, lo + interval)
        train_lo = 0 if rolling is None else max(0, lo-rolling)
        tr = np.arange(train_lo, lo)
        va = np.arange(lo,hi)
        if tr.size == 0 or tr[-1] >= va[0]:
            raise RuntimeError('Non-causal walkforward schedule')
        blocks.append((tr,va))
    if not np.array_equal(np.concatenate([v for _,v in blocks]),np.arange(first,n)):
        raise RuntimeError('Schedule missing targets')
    return blocks


def coerce_bool(flags):
    values = list(flags)
    out=[]
    for x in values:
        if isinstance(x,(bool,np.bool_)):
            out.append(bool(x))
        elif isinstance(x,str) and x.lower() in ('true','false'):
            out.append(x.lower()=='true')
        else:
            raise ValueError('Invalid violation flag')
    return np.asarray(out, dtype=bool)


def verify_predictions(df: pd.DataFrame, dates, asof, actual,
                       policies=POLICIES) -> pd.DataFrame:
    required={'policy','outer_fold','as_of','target_date','quantile_return',
              'var_return','actual_return','violation'}
    if not required.issubset(df.columns):
        raise ValueError('Missing prediction columns')
    dates=np.asarray(dates,str);asof=np.asarray(asof,str)
    actual=np.asarray(actual,float)
    if not (len(dates)==len(asof)==len(actual)>0 and np.isfinite(actual).all()):
        raise ValueError('Bad reference timeline')
    if len(set(dates))!=len(dates) or not np.all(asof<dates) or not np.all(dates[:-1]<dates[1:]):
        raise ValueError('Reference dates invalid')
    if len(df)!=len(dates)*len(policies) or set(df.policy)!=set(policies):
        raise ValueError('Prediction policy count mismatch')
    out=[]
    for label in policies:
        part=df.loc[df.policy==label].sort_values('target_date').reset_index(drop=True)
        if part.duplicated('target_date').any() or len(part)!=len(dates):
            raise ValueError('Duplicate or missing target')
        if not (np.array_equal(part.target_date.astype(str).to_numpy(),dates)
                and np.array_equal(part.as_of.astype(str).to_numpy(),asof)):
            raise ValueError('Target/as_of misalignment')
        y=part.actual_return.to_numpy(float)
        q=part.quantile_return.to_numpy(float)
        v=part.var_return.to_numpy(float)
        if not np.isfinite(np.column_stack([y,q,v])).all():
            raise ValueError('Nonfinite forecast')
        if not np.allclose(y,actual,rtol=0,atol=1e-12):
            raise ValueError('Actual return mismatch')
        if not np.allclose(v,np.maximum(0,-q),rtol=0,atol=1e-12):
            raise ValueError('VaR sign convention mismatch')
        if not np.array_equal(coerce_bool(part.violation),y<q):
            raise ValueError('Violation flag inconsistency')
        if len(dates)==VALID_N:
            expected_fold=1+np.arange(VALID_N)//OUTER_LENGTH
            if not np.array_equal(part.outer_fold.to_numpy(int), expected_fold):
                raise ValueError('Wrong outer-fold assignment')
        out.append(part)
    return pd.concat(out,ignore_index=True)


def _xlogy(p: float, x: int) -> float:
    return 0.0 if x == 0 else x*math.log(p) if p>0 else float('-inf')


def coverage_tests(flags, alpha:float=ALPHA) -> dict:
    """Kupiec UC / Christoffersen independence and conditional coverage.

    Chi-square(1) p-value = erfc(sqrt(LR/2)); chi-square(2) = exp(-LR/2).
    Boundary likelihoods use the 0*log(0) convention.
    """
    f=coerce_bool(flags)
    if not (0<alpha<1) or len(f)<3:
        raise ValueError('Invalid exceptions for coverage')
    n=len(f); k=int(f.sum());p=k/n
    ll_null=_xlogy(alpha,k)+_xlogy(1-alpha,n-k)
    ll_alt=_xlogy(p,k)+_xlogy(1-p,n-k)
    lr_uc=max(0.,2*(ll_alt-ll_null))
    a,b=f[:-1],f[1:]
    n00=int(((~a)&(~b)).sum());n01=int(((~a)&b).sum())
    n10=int((a&(~b)).sum());n11=int((a&b).sum())
    nn=n00+n01+n10+n11
    pi=(n01+n11)/nn
    pi0=n01/(n00+n01) if n00+n01 else 0.0
    pi1=n11/(n10+n11) if n10+n11 else 0.0
    lli0=_xlogy(pi,n01+n11)+_xlogy(1-pi,n00+n10)
    lli1=(_xlogy(pi0,n01)+_xlogy(1-pi0,n00)
          +_xlogy(pi1,n11)+_xlogy(1-pi1,n10))
    lr_ind=max(0.,2*(lli1-lli0))
    lr_cc=lr_uc+lr_ind
    return {'n':n,'violations':k,'violation_rate':p,
            'kupiec_p_uc':math.erfc(math.sqrt(lr_uc/2)),
            'christoffersen_p_ind':math.erfc(math.sqrt(lr_ind/2)),
            'conditional_p_cc':math.exp(-lr_cc/2),
            'transitions_00':n00,'transitions_01':n01,
            'transitions_10':n10,'transitions_11':n11}


def summarize(df:pd.DataFrame, policies=POLICIES):
    records=[]
    for label in policies:
        p=df.loc[df.policy==label].sort_values('target_date')
        y=p.actual_return.to_numpy(float);q=p.quantile_return.to_numpy(float)
        v=p.var_return.to_numpy(float)
        c=coverage_tests(y<q)
        records.append({'policy':label, 'mean_pinball':float(pinball(y,q).mean()),
                        'average_var':float(v.mean()),**c})
    res=pd.DataFrame(records)
    base=float(res.loc[res.policy=='C1_frozen','mean_pinball'].iloc[0])
    res['relative_loss_improvement_vs_c1_pct']=(base-res.mean_pinball)/base*100
    return res


def rolling_quality(df:pd.DataFrame, *, window:int=60):
    if window < 20:
        raise ValueError('Rolling window too small')
    rows=[]
    for label,part in df.groupby('policy',sort=False):
        p=part.sort_values('target_date').reset_index(drop=True)
        y=p.actual_return.to_numpy(float);q=p.quantile_return.to_numpy(float)
        loss=pinball(y,q)
        exceptions=(y<q).astype(float)
        rolling_pin=pd.Series(loss).rolling(window,min_periods=window).mean()
        rolling_vi=pd.Series(exceptions).rolling(window,min_periods=window).mean()
        for j in range(window-1,len(p)):
            rows.append({'policy':label,'target_date':p.target_date.iloc[j],
                         'window':window,'rolling_pinball':float(rolling_pin.iloc[j]),
                         'rolling_violation_rate':float(rolling_vi.iloc[j])})
    return pd.DataFrame(rows)


def period_metrics(df:pd.DataFrame):
    rows=[]
    for (policy,outer),p in df.groupby(['policy','outer_fold'],sort=True):
        p=p.sort_values('target_date')
        y=p.actual_return.to_numpy(float);q=p.quantile_return.to_numpy(float)
        rows.append({'policy':policy,'outer_fold':int(outer),'n':len(p),
                     'first_target':str(p.target_date.iloc[0]),
                     'last_target':str(p.target_date.iloc[-1]),
                     'mean_pinball':float(pinball(y,q).mean()),
                     'violation_rate':float((y<q).mean()),
                     'violations':int((y<q).sum())})
    return pd.DataFrame(rows)


def block_bootstrap_paired(base_loss, alternative_loss, *, blocks=(200,200,200),
                           block_length:int=20, repetitions:int=1500,seed:int=20261109):
    """Descriptive circular moving-block CI on paired observed loss differences.

    Preserves within-fold temporal dependence; NOT a confirmatory p-value.
    Positive difference = lower loss for alternative.
    """
    b=np.asarray(base_loss,float);a=np.asarray(alternative_loss,float)
    if (b.ndim!=1 or a.shape!=b.shape or len(b)!=sum(blocks) or
        not np.isfinite(b).all() or not np.isfinite(a).all() or
        min(blocks)<block_length or block_length<2 or repetitions<100):
        raise ValueError('Invalid bootstrap inputs')
    diffs=b-a
    rng=np.random.default_rng(seed)
    estimates=np.empty(repetitions)
    cuts=np.cumsum((0,*blocks))
    for t in range(repetitions):
        parts=[]
        for f,n in enumerate(blocks):
            one=diffs[cuts[f]:cuts[f+1]]
            starts=rng.integers(0,n,size=math.ceil(n/block_length))
            idx=(starts[:,None]+np.arange(block_length)[None,:])%n
            parts.append(one[idx.ravel()[:n]])
        estimates[t]=np.concatenate(parts).mean()
    lo,hi=np.quantile(estimates,[.025,.975])
    return {'paired_mean_loss_advantage':float(diffs.mean()),
            'block_ci_95_low':float(lo),'block_ci_95_high':float(hi),
            'ci_includes_zero':bool(lo<=0<=hi),
            'block_length':block_length,'bootstrap_repetitions':repetitions,
            'bootstrap_seed':seed,
            'scope':'EXPLORATORY (reused development data, not untouched future test)'}
