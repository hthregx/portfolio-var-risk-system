"""P3-05 A: causal refit cadence, expanding-vs-rolling and stability on development.

Read-only on locked Phase2, P3-01, P3-03, P3-04 artifacts. Does not read the
2024-12-18 onward historical holdout for training, validation, or selection.
This is EXPLORATORY retrospective evidence, not a production selection decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import GradientBoostingRegressor

from src.phase2.features_a import ALL_FEATURES
from src.phase3.tuning_a import (
    candidate_parameters, choose_from_inner, pinball, validate_dev_frame
)
from src.phase3.stability_a import (
    ALPHA, FIRST_VALID, VALID_N, OUTER_LENGTH, REFIT_EVERY,
    ROLLING_TRAIN, POLICIES, walkforward_schedule, verify_predictions,
    summarize, period_metrics, rolling_quality, block_bootstrap_paired
)

ROOT=Path(__file__).resolve().parents[1]
CUTOFF='2026-11-06'
SOURCE_HASH='6389586ae8f1edbabb31760e0b555619186114b3f77d09212362fc2e1c40aadf'
FEATURES12_HASH='14ac5dd4ce7dd6090420b9cee06c8ac40935c54691b4daf95d063b71a075eccc'
DEV_END='2024-12-17'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authenticated_inputs(repo:Path, tuning_dir:Path):
    source=repo/f'results/phase2/a/features_{CUTOFF}.csv'
    lock_file=repo/f'results/phase3/a/baseline_{CUTOFF}/baseline_lock.json'
    diag_file=repo/f'results/phase3/a/diagnostics_{CUTOFF}/manifest.json'
    meta_file=tuning_dir/'manifest.json'
    for path in (source,lock_file,diag_file,meta_file):
        if not path.is_file():
            raise FileNotFoundError(f'Missing required verified input: {path}')
    lock=json.loads(lock_file.read_text(encoding='utf-8-sig'))
    meta=json.loads(meta_file.read_text(encoding='utf-8-sig'))
    checks={
        'original_12f_sha':sha256(source)==FEATURES12_HASH,
        'source_kbs_sha':meta.get('source_sha256')==SOURCE_HASH==lock.get('source_sha256'),
        'original_features_p304':meta.get('features_12_sha256')==FEATURES12_HASH==lock.get('features_sha256'),
        'p301_lock_sha':meta.get('baseline_lock_sha256')==sha256(lock_file),
        'p303_diagnostics_sha':meta.get('p3_03_manifest_sha256')==sha256(diag_file),
        'cutoff':meta.get('snapshot_cutoff')==CUTOFF and meta.get('training_cutoff')==DEV_END,
        'nested_policy':meta.get('outer_model_fit_count')==6 and meta.get('inner_fit_count')==45,
        'holdout_guard':'NEVER READ' in meta.get('holdout_policy','')
    }
    if not all(checks.values()):
        raise ValueError('Frozen provenance mismatch: '+repr([k for k,v in checks.items() if not v]))
    for name,record in meta.get('artifacts',{}).items():
        path=tuning_dir/name
        if not path.is_file() or sha256(path)!=record['sha256']:
            raise ValueError('P3-04 artifact missing or SHA256 mismatch: '+name)
    frame=pd.read_csv(source,parse_dates=['target_date','as_of'],float_precision='round_trip')
    dev=validate_dev_frame(frame,tuple(ALL_FEATURES),DEV_END)
    if len(dev)!=1179 or dev.target_date.max().strftime('%Y-%m-%d')!=DEV_END:
        raise ValueError('Development protocol unexpectedly changed')
    raw=pd.read_csv(tuning_dir/'outer_cv_predictions.csv',float_precision='round_trip')
    raw=raw.copy()
    expected_dates=dev.target_date.iloc[FIRST_VALID:].dt.strftime('%Y-%m-%d').to_numpy()
    expected_asof=dev.as_of.iloc[FIRST_VALID:].dt.strftime('%Y-%m-%d').to_numpy()
    actual=dev.actual_return.iloc[FIRST_VALID:].to_numpy(float)
    old=verify_predictions(raw,expected_dates,expected_asof,actual,
                           policies=('C1_frozen','nested_selected'))
    if not len(old)==1200:
        raise ValueError('Nested outer predictions unexpected size')
    metrics=pd.read_csv(tuning_dir/'outer_cv_metrics.csv')
    if not len(metrics)==6 or metrics.duplicated(['outer_fold','policy']).any():
        raise ValueError('Unexpected nested outer summary')
    for (fold,policy),part in old.groupby(['outer_fold','policy']):
        row=metrics[(metrics.outer_fold==fold)&(metrics.policy==policy)]
        if len(part)!=200 or len(row)!=1:
            raise ValueError('Broken outer-fold pairing')
        loss=pinball(part.actual_return.to_numpy(float),part.quantile_return.to_numpy(float)).mean()
        if not np.isclose(loss,float(row.valid_pinball.iloc[0]),rtol=0,atol=1e-12):
            raise ValueError('P3-04 outer loss inconsistent with predictions')
    inner=pd.read_csv(tuning_dir/'inner_cv_scores.csv')
    chosen=pd.read_csv(tuning_dir/'inner_selected_configs.csv')
    if len(inner)!=45 or len(chosen)!=3:
        raise ValueError('Unexpected nested selection records')
    for outer in (1,2,3):
        selected,_=choose_from_inner(inner[inner.outer_fold==outer][['candidate','inner_fold','pinball_loss']])
        got=chosen.loc[chosen.outer_fold==outer,'selected_from_inner'].tolist()
        if got!=[selected]:
            raise ValueError('Nested hyperparameters not selected by inner-only rule')
        prediction_models=old[(old.outer_fold==outer)&(old.policy=='nested_selected')].model.unique().tolist()
        if prediction_models!=[selected]:
            raise ValueError('Nested selected model changed')
    return dev,old,meta,sha256(meta_file)


def walk_forward_predict(dev, interval:int=REFIT_EVERY, window:int|None=None):
    X=dev[list(ALL_FEATURES)].to_numpy(float)
    y=dev.actual_return.to_numpy(float)
    target=dev.target_date.dt.strftime('%Y-%m-%d').to_numpy()
    asof=dev.as_of.dt.strftime('%Y-%m-%d').to_numpy()
    label='expanding_refit20' if window is None else 'rolling504_refit20'
    schedules=walkforward_schedule(len(dev),interval=interval,first=FIRST_VALID,rolling=window)
    result=[];fitlog=[]
    for block,(train,valid) in enumerate(schedules,1):
        if not (max(train)<min(valid) and target[train[-1]]<target[valid[0]]
                and all(asof[valid]<target[valid])):
            raise ValueError('Training/validation or feature look-ahead')
        model=GradientBoostingRegressor(**candidate_parameters()['C1_frozen'])
        model.fit(X[train],y[train])
        predicted=model.predict(X[valid])
        if not np.isfinite(predicted).all():
            raise ValueError('Nonfinite quantile forecast')
        fold=1+(valid-FIRST_VALID)//OUTER_LENGTH
        for k,q in zip(range(len(valid)),predicted):
            i=valid[k]
            result.append({'policy':label,'model':'C1_frozen',
                           'outer_fold':int(fold[k]),'as_of':str(asof[i]),
                           'target_date':str(target[i]),'quantile_return':float(q),
                           'var_return':float(max(0.,-q)),
                           'actual_return':float(y[i]),'violation':bool(y[i]<q)})
        fitlog.append({'policy':label,'refit_block':block,'train_n':len(train),
                       'train_start':str(target[train[0]]),'train_end':str(target[train[-1]]),
                       'valid_start':str(target[valid[0]]),'valid_end':str(target[valid[-1]]),
                       'valid_n':len(valid),'fit_random_state':42})
        if block in (1,10,20,len(schedules)):
            print(f'{label.upper()} BLOCK {block:02d}/{len(schedules)}: train={len(train)} valid={len(valid)} PASS',flush=True)
    return pd.DataFrame(result),pd.DataFrame(fitlog)


def graphics(df, rolling, periods, figdir:Path):
    figdir.mkdir(parents=True,exist_ok=False)
    all_fig=[]
    fig,ax=plt.subplots(figsize=(11,4.5))
    for policy,segment in rolling.groupby('policy'):
        ax.plot(pd.to_datetime(segment.target_date),segment.rolling_pinball,label=policy,lw=1.25)
    ax.set(title='Rolling 60-session pinball loss (exploratory nested development)',
           xlabel='Target date',ylabel='Rolling mean pinball loss (alpha=0.05)')
    ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2);fig.autofmt_xdate();fig.tight_layout()
    file=figdir/'rolling_60d_pinball.png';fig.savefig(file,dpi=150);plt.close(fig);all_fig.append(file)
    fig,ax=plt.subplots(figsize=(11,4.5))
    for policy,segment in rolling.groupby('policy'):
        ax.plot(pd.to_datetime(segment.target_date),100*segment.rolling_violation_rate,label=policy,lw=1.25)
    ax.axhline(5,color='k',linestyle='--',lw=1,label='nominal 5%')
    ax.set(title='Rolling 60-session VaR exception rate (exploratory)',
           xlabel='Target date',ylabel='Exceptions (%)')
    ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2);fig.autofmt_xdate();fig.tight_layout()
    file=figdir/'rolling_60d_coverage.png';fig.savefig(file,dpi=150);plt.close(fig);all_fig.append(file)
    fig,ax=plt.subplots(figsize=(10,4.5))
    tick=np.arange(1,4)
    for k,policy in enumerate(POLICIES):
        x=periods.loc[periods.policy==policy].sort_values('outer_fold')
        ax.bar(tick+(k-1.5)*.19,x.mean_pinball,width=.18,label=policy)
    ax.set(xticks=tick,xlabel='Outer validation period',ylabel='Mean pinball loss',
           title='Stability of quantile forecasts by chronological regime')
    ax.legend(fontsize=8,ncol=2);ax.grid(axis='y',alpha=.2);fig.tight_layout()
    file=figdir/'three_regime_pinball.png';fig.savefig(file,dpi=150);plt.close(fig);all_fig.append(file)
    fig,ax=plt.subplots(figsize=(11,4.2))
    for j,policy in enumerate(POLICIES):
        violations=df[(df.policy==policy)&(df.violation.astype(bool))]
        dates=pd.to_datetime(violations.target_date)
        ax.scatter(dates,np.full(len(dates),j),s=18,alpha=.7,label=policy)
    ax.set(yticks=range(len(POLICIES)),yticklabels=POLICIES,
           title='Actual VaR violation events (development only)',xlabel='Target date',ylim=(-.6,len(POLICIES)-.4))
    ax.grid(axis='x',alpha=.2);fig.autofmt_xdate();fig.tight_layout()
    file=figdir/'exception_timeline.png';fig.savefig(file,dpi=150);plt.close(fig);all_fig.append(file)
    return all_fig


def run(repo:Path,tuning_dir:Path,out:Path):
    repo=repo.resolve();tuning_dir=tuning_dir.resolve();out=out.resolve()
    out.relative_to(repo)
    if out.exists():
        raise FileExistsError('P3-05 directory already exists; refusing to overwrite')
    dev,old,p304,p304_hash=authenticated_inputs(repo,tuning_dir)
    print('P3-04 MANIFEST + 7 ARTIFACT HASHES: PASS',flush=True)
    print('SOURCE CUTOFF:',CUTOFF,'DEV ONLY:',len(dev),'P3-04 OOF:',len(old),flush=True)
    expanding,log_exp=walk_forward_predict(dev,window=None)
    rolling,log_roll=walk_forward_predict(dev,window=ROLLING_TRAIN)
    combined=pd.concat([old,expanding,rolling],ignore_index=True)
    dates=dev.target_date.iloc[FIRST_VALID:].dt.strftime('%Y-%m-%d').to_numpy()
    asof=dev.as_of.iloc[FIRST_VALID:].dt.strftime('%Y-%m-%d').to_numpy()
    actual=dev.actual_return.iloc[FIRST_VALID:].to_numpy(float)
    combined=verify_predictions(combined,dates,asof,actual)
    metrics=summarize(combined)
    by_regime=period_metrics(combined)
    by_rolling=rolling_quality(combined,window=60)
    trainlog=pd.concat([log_exp,log_roll],ignore_index=True)
    if not (len(combined)==2400 and len(trainlog)==60 and
            len(by_regime)==12 and len(by_rolling)==2164):
        raise RuntimeError('Incomplete P3-05 output')
    events=combined[combined.violation.astype(bool)].copy().reset_index(drop=True)
    reference=combined[combined.policy=='C1_frozen'].sort_values('target_date')
    ref_loss=pinball(reference.actual_return,reference.quantile_return)
    paired=[]
    for policy in POLICIES[1:]:
        other=combined[combined.policy==policy].sort_values('target_date')
        measure=block_bootstrap_paired(ref_loss,pinball(other.actual_return,other.quantile_return))
        paired.append({'baseline':'C1_frozen','alternative':policy,**measure})
    paired_df=pd.DataFrame(paired)
    if metrics.policy.tolist()!=list(POLICIES):
        raise RuntimeError('Unexpected model ordering')
    out.mkdir(parents=True,exist_ok=False)
    tables={'walkforward_predictions':combined,'refit_schedule':trainlog,
            'policy_metrics':metrics,'regime_metrics':by_regime,
            'rolling_60d_metrics':by_rolling,'paired_block_bootstrap':paired_df,
            'exception_events':events}
    artifacts={}
    for name,df in tables.items():
        p=out/f'{name}.csv';df.to_csv(p,index=False,float_format='%.17g')
        artifacts[p.name]={'sha256':sha256(p),'rows':len(df)}
    for p in graphics(combined,by_rolling,by_regime,out/'figures'):
        artifacts[p.relative_to(out).as_posix()]={'sha256':sha256(p)}
    metadata={'owner':'A','stage':'EXPLORATORY development_only_not_live_forecast',
              'experiment':'Phase3 P3-05 expanding versus rolling refit cadence',
              'snapshot_cutoff':CUTOFF,'development_last':DEV_END,
              'source_sha256':SOURCE_HASH,'features12_sha256':FEATURES12_HASH,
              'p304_manifest_sha256':p304_hash,
              'protocol':'Only first 1179 development targets; last 600 validated; C1 refit once per 20 targets with prior observed features; either expanding or last 504 rows; compare to P3-04 200-day frozen fits.',
              'refit_every_n_sessions':REFIT_EVERY,'rolling_window':ROLLING_TRAIN,
              'frozen_reference':'GBv2 C1 12 features, no hyperparameter retuning',
              'frozen_baseline':'P3-04 C1_frozen (3x200); not a live forecast',
              'original_holdout_used':False,'selection_from_p305':False,
              'cautions':['This retrospective development history was examined in Phase 2/3 and is not a clean future performance estimate.',
                          'C1, expanding and rolling models differ in refit schedules, so this is a refit-policy comparison, not purely feature selection.',
                          'Moving block bootstrap intervals are descriptive, not confirmatory p-values; 60-day coverage windows are correlated and noisy.',
                          'Kupiec and Christoffersen non-rejection is not proof of perfect coverage.',
                          'No production model is chosen by the already-inspected holdout.'],
              'artifacts':artifacts,
              'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
              'sklearn':sklearn_version,'script_sha256':sha256(__file__),
              'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()}
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nSTABILITY METRICS (EXPLORATORY; LOWER PINBALL IS BETTER):')
    print(metrics[['policy','n','violations','violation_rate','mean_pinball',
                   'kupiec_p_uc','christoffersen_p_ind','relative_loss_improvement_vs_c1_pct']].to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('\nPAIRED BLOCK BOOTSTRAP (DESCRIPTIVE, NOT DEFINITIVE):')
    print(paired_df[['alternative','paired_mean_loss_advantage','block_ci_95_low',
                    'block_ci_95_high','ci_includes_zero']].to_string(index=False))
    print('WALKFORWARD REFITS:',len(trainlog),'PREVIOUS OUTER FOLD OBSERVATIONS:',len(old))
    print('PREDICTIONS:',len(combined),'ROLLING WINDOWS:',len(by_rolling),
          'REGIME SUMMARIES:',len(by_regime),'FIGURES:',4)
    print('OLD HOLDOUT USED: NO; PHASE2 MODIFIED: NO; WINNER SELECTED: NO')
    print('OUTPUT:',out)
    print('STEP P3-05 WALKFORWARD STABILITY PASS')
    return metadata


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tuning-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    run(ROOT,a.tuning_dir,a.out)


if __name__=='__main__':
    main()
