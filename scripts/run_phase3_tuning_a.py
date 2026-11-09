"""P3-04 A: bounded hyperparameter search using nested expanding CV, development ONLY.

12F was pre-screened in P3-03; use this as exploratory evidence on reused
Phase 2 development history, NOT as an unbiased estimate on unseen future dates.
No Phase 2 holdout data are read for fitting or choosing hyperparameters.
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
from src.phase3.tuning_a import (outer_splits,inner_splits,candidate_parameters,
    choose_from_inner,pinball,violation_summary,validate_dev_frame,
    SELECTION_MIN_REL_GAIN,SELECTION_MIN_FOLD_WINS)

ROOT=Path(__file__).resolve().parents[1]
CUTOFF='2026-11-06'
DEV_END='2024-12-17'
SOURCE_HASH='6389586ae8f1edbabb31760e0b555619186114b3f77d09212362fc2e1c40aadf'
FEAT12_HASH='14ac5dd4ce7dd6090420b9cee06c8ac40935c54691b4daf95d063b71a075eccc'
FEAT18_HASH='609926f3af7b2cea81fdd63d5a9e47050bee2128d81b11e83c03d7eb5b818288'
FEAT25_HASH='b45094fea0f8814d19b0505b26f8d6556c184190fa73dc6b953b14c1245874fa'


def sha(path:Path)->str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_authenticated_development(repo:Path,diagnostics_dir:Path,lock_file:Path):
    baseline=repo/f'results/phase2/a/features_{CUTOFF}.csv'
    old_meta=baseline.with_name(f'features_{CUTOFF}_metadata.json')
    diag_meta=diagnostics_dir/'manifest.json'
    ranking=diagnostics_dir/'cv_ranking.csv'
    lock=json.loads(lock_file.read_text(encoding='utf-8-sig'))
    d=json.loads(diag_meta.read_text(encoding='utf-8-sig'))
    m=json.loads(old_meta.read_text(encoding='utf-8-sig'))
    if (sha(baseline)!=FEAT12_HASH or m['output_sha256']!=FEAT12_HASH
            or m['input_sha256']!=SOURCE_HASH
            or sha(baseline)!=lock['features_sha256']
            or lock['source_sha256']!=SOURCE_HASH
            or lock['training_cutoff']!=DEV_END
            or lock['selected_on_development_cv']!='gbv2_12f_c1_a'):
        raise ValueError('Frozen Phase 2 baseline provenance mismatch')
    if (d['source_sha256']!=SOURCE_HASH or d['baseline_lock_sha256']!=sha(lock_file)
            or d['old_phase2_holdout_untouched'] is not True
            or d['feature_csv_hashes']!={'12':FEAT12_HASH,'18':FEAT18_HASH,'25':FEAT25_HASH}
            or d['best_feature_count_exploratory_cv_only']!=12
            or d['phase2_12f_reproduction_max_abs_diff']>1e-9):
        raise ValueError('P3-03 diagnostic provenance or feature selection mismatch')
    for filename, entry in d['artifacts'].items():
        relative = filename if filename.startswith('figures/') else filename + '.csv'
        target=diagnostics_dir/relative
        if not target.is_file() or sha(target)!=entry['sha256']:
            raise ValueError('P3-03 diagnostic output changed: '+filename)
    if sha(ranking)!=d['artifacts']['cv_ranking']['sha256']:
        raise ValueError('Exploratory feature ranking changed')
    ranked=pd.read_csv(ranking)
    if (int(ranked.iloc[0].feature_count)!=12 or len(ranked)!=3 or
            not np.isfinite(float(ranked.iloc[0].cv_pinball)) or float(ranked.iloc[0].cv_pinball)<=0):
        raise ValueError('Unexpected prior feature screening result')
    frame=pd.read_csv(baseline,parse_dates=['target_date','as_of'],float_precision='round_trip')
    dev=validate_dev_frame(frame, tuple(ALL_FEATURES),DEV_END)
    return dev,d,lock,sha(diag_meta),sha(lock_file)


def run(repo:Path,diag_dir:Path,lock_file:Path,out:Path):
    repo,diag_dir,lock_file,out=repo.resolve(),diag_dir.resolve(),lock_file.resolve(),out.resolve()
    out.relative_to(repo)
    if out.exists():
        raise FileExistsError('P3-04 results already exist; no overwrite allowed')
    dev,d,lock,diag_sha,lock_sha=load_authenticated_development(repo,diag_dir,lock_file)
    X=dev[list(ALL_FEATURES)].to_numpy(np.float64)
    y=dev.actual_return.to_numpy(np.float64)
    dates=dev.target_date.dt.strftime('%Y-%m-%d').to_numpy()
    asof=dev.as_of.dt.strftime('%Y-%m-%d').to_numpy()
    configs=candidate_parameters()
    inner_rows=[]
    outer_rows=[]
    predictions=[]
    choices=[]
    for outer_id,(outer_train,outer_val) in enumerate(outer_splits(len(dev)),1):
        one=[]
        for inner_id,(itr,iva) in enumerate(inner_splits(len(outer_train)),1):
            if not (itr[-1] < iva[0] and outer_train[iva[-1]] < outer_val[0]):
                raise ValueError('Inner/outer leakage')
            for candidate,params in configs.items():
                fitted=GradientBoostingRegressor(**params)
                fitted.fit(X[itr],y[itr])
                q=fitted.predict(X[iva])
                loss=float(pinball(y[iva],q).mean())
                row={'outer_fold':outer_id,'inner_fold':inner_id,'candidate':candidate,
                     'train_n':len(itr),'valid_n':len(iva),
                     'train_end':dates[itr[-1]],'valid_start':dates[iva[0]],
                     'valid_end':dates[iva[-1]],'pinball_loss':loss}
                one.append(row)
                inner_rows.append(row)
        selected, rationale=choose_from_inner(pd.DataFrame(one))
        choices.append({'outer_fold':outer_id,'selected_from_inner':selected,
                        'inner_c1_pinball':rationale['C1_frozen']['mean_loss'],
                        'inner_selected_pinball':rationale[selected]['mean_loss'],
                        'inner_gain_vs_c1_pct':rationale[selected]['gain_vs_c1_pct'],
                        'inner_fold_wins_vs_c1':rationale[selected]['fold_wins_vs_c1'],
                        'outer_train_n':len(outer_train),'outer_valid_n':len(outer_val),
                        'outer_train_end':dates[outer_train[-1]],
                        'outer_valid_start':dates[outer_val[0]],
                        'outer_valid_end':dates[outer_val[-1]]})
        for label, candidate in [('C1_frozen', 'C1_frozen'),('nested_selected',selected)]:
            fitted=GradientBoostingRegressor(**configs[candidate])
            fitted.fit(X[outer_train],y[outer_train])
            qv=fitted.predict(X[outer_val])
            qt=fitted.predict(X[outer_train])
            val=violation_summary(y[outer_val],qv)
            train_loss=float(pinball(y[outer_train],qt).mean())
            outer_rows.append({'outer_fold':outer_id,'policy':label,'model':candidate,
                               'train_n':len(outer_train),'valid_n':len(outer_val),
                               'train_pinball':train_loss,
                               'valid_pinball':val['pinball_loss'],
                               'train_validation_gap':val['pinball_loss']-train_loss,
                               **val})
            predictions.extend({'outer_fold':outer_id,'policy':label,'model':candidate,
                                'as_of':a,'target_date':day,'quantile_return':float(q),
                                'var_return':float(max(0,-q)),
                                'actual_return':float(actual),
                                'violation':bool(actual<q)}
                               for a,day,q,actual in zip(asof[outer_val],dates[outer_val],qv,y[outer_val]))
        print(f'OUTER {outer_id}: TRAIN {len(outer_train)} / VALID {len(outer_val)} / INNER 3 FOLDS / SELECTED {selected} PASS',flush=True)
    inner_df=pd.DataFrame(inner_rows)
    outer_df=pd.DataFrame(outer_rows)
    preds_df=pd.DataFrame(predictions)
    selection_df=pd.DataFrame(choices)
    if not (len(inner_df)==45 and len(outer_df)==6 and len(preds_df)==1200
            and len(selection_df)==3 and
            (preds_df.as_of < preds_df.target_date).all() and
            not preds_df.duplicated(['policy','target_date']).any() and
            preds_df.groupby('policy').size().to_dict()=={'C1_frozen':600,'nested_selected':600}):
        raise RuntimeError('Unexpected nested temporal experiment size/alignment')
    # Exact paired date and actuals match.
    c1=preds_df[preds_df.policy=='C1_frozen'].reset_index(drop=True)
    sel=preds_df[preds_df.policy=='nested_selected'].reset_index(drop=True)
    if not (c1.target_date.equals(sel.target_date) and
            c1.as_of.equals(sel.as_of) and
            np.array_equal(c1.actual_return,sel.actual_return)):
        raise ValueError('Paired outer-fold predictions not aligned')
    overview=outer_df.groupby('policy').agg(outer_mean_pinball=('valid_pinball','mean'),
        train_mean_pinball=('train_pinball','mean'),
        mean_gap=('train_validation_gap','mean'),
        mean_violation_rate=('violation_rate','mean'),
        worst_outer_pinball=('valid_pinball','max')).reset_index()
    val0=float(overview.loc[overview.policy=='C1_frozen','outer_mean_pinball'].iloc[0]);
    val1=float(overview.loc[overview.policy=='nested_selected','outer_mean_pinball'].iloc[0]);
    gain=100*(val0-val1)/val0
    # Persist only once validations and fitting have finished.
    out.mkdir(parents=True,exist_ok=False)
    csvs={'inner_cv_scores':inner_df,'outer_cv_metrics':outer_df,
          'outer_cv_predictions':preds_df,'inner_selected_configs':selection_df,
          'nested_comparison':overview}
    artifacts={}
    for name,table in csvs.items():
        p=out/f'{name}.csv'
        table.to_csv(p,index=False,float_format='%.17g')
        artifacts[p.name]={'sha256':sha(p),'rows':len(table)}
    figdir=out/'figures';figdir.mkdir()
    fig,ax=plt.subplots(figsize=(9,4.5))
    for label in ['C1_frozen','nested_selected']:
        f=outer_df[outer_df.policy==label]
        ax.plot(f.outer_fold,f.valid_pinball,marker='o',label=label)
    ax.set(xlabel='Chronological outer fold',ylabel='Pinball loss (alpha=0.05)',
           title='Nested temporal CV - C1 versus inner-selected policy')
    ax.set_xticks([1,2,3]);ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(figdir/'nested_outer_pinball.png',dpi=155);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    ix=np.arange(3)
    for offset,label in [(-.18,'C1_frozen'),(.18,'nested_selected')]:
        data=outer_df[outer_df.policy==label].sort_values('outer_fold')
        ax.bar(ix+offset,data.train_validation_gap,.36,label=label)
    ax.set(xticks=ix,xticklabels=['1','2','3'],xlabel='Outer fold',
           ylabel='Validation minus train pinball',
           title='Generalization gaps (diagnostic only)')
    ax.grid(axis='y',alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(figdir/'nested_generalization_gap.png',dpi=155);plt.close(fig)
    for p in sorted(figdir.glob('*.png')):
        artifacts[p.relative_to(out).as_posix()]={'sha256':sha(p)}
    report={'owner':'A','experiment':'phase3_nested_temporal_tuning_12f',
            'stage':'EXPLORATORY development_only_not_live_forecast',
            'data_source':'KBS','snapshot_cutoff':CUTOFF,'training_cutoff':DEV_END,
            'source_sha256':SOURCE_HASH,'features_12_sha256':FEAT12_HASH,
            'baseline_lock_sha256':lock_sha,'p3_03_manifest_sha256':diag_sha,
            'feature_count':12,'candidate_configs':configs,
            'candidate_selection_rule':f'inner CV mean pinball >= {SELECTION_MIN_REL_GAIN:.1%} relative improvement AND >= {SELECTION_MIN_FOLD_WINS}/3 inner wins versus C1, else C1',
            'nested_protocol':'3 expanding outer folds with disjoint 200-session validation; each outer training period uses 3 inner expanding folds for config choice',
            'outer_train_sizes':[579,779,979],
            'outer_validation_sessions_per_fold':200,
            'inner_fit_count':45,'outer_model_fit_count':6,
            'selection_by_outer_fold':selection_df.to_dict(orient='records'),
            'outer_oof_mean_pinball_c1':val0,
            'outer_oof_mean_pinball_inner_selected':val1,
            'outer_oof_relative_gain_pct_exploratory':gain,
            'holdout_policy':'NEVER READ historical Phase 1/Phase 2 holdout; do not select final production model on outer performance',
            'limitations':["Prior development history was already inspected during Phase 2/3; outer nested folds reduce within-run tuning bias but are not pristine future evidence.",
                          'No claims of live performance; use new future timestamps for independent confirmation.',
                          'This step tests tuning with preselected 12 features only; feature-count selection was exploratory on reused development data.'],
            'artifacts':artifacts,'versions':{'python':platform.python_version(),
                       'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn_version},
            'code_sha256':sha(__file__),
            'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()}
    (out/'manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nNESTED OUTER METRICS (EXPLORATORY):')
    print(outer_df[['outer_fold','policy','model','valid_pinball','violation_rate','train_validation_gap']].to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('\nCOMPARISON:')
    print(overview.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('MEAN OUTER PINBALL CHANGE (%):',round(gain,4))
    print('SELECTIONS:',selection_df.selected_from_inner.tolist())
    print('INNER FITS:',len(inner_df),'OUTER PREDICTIONS:',len(preds_df),'FIGURES: 2')
    print('PREVIOUS HOLDOUT USED: NO')
    print('OUTPUT:',out)
    print('STEP P3-04 NESTED TUNING PASS')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--diagnostics-dir',type=Path,required=True)
    p.add_argument('--baseline-lock',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    run(ROOT,a.diagnostics_dir,a.baseline_lock,a.out)


if __name__=='__main__':
    main()
