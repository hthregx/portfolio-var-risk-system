#!/usr/bin/env python3
"""Freeze and exactly reproduce original C1 (12F) VaR, Phase2 KBS snapshot 2026-11-06.

No hyperparameter tuning, no 7/30/seasonal experiments, no use of holdout for selection.
Run: python scripts/final_c1/run_final_c1.py --root .
Audit: python scripts/final_c1/run_final_c1.py --root . --verify-only
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.ensemble import GradientBoostingRegressor
import sklearn
import joblib

OUT_REL = Path('results/final_C1_frozen_2026-11-19')
CONFIG_REL = Path('configs/c1_final_frozen.json')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



def sha_matches_eol(path: Path, expected: str) -> bool:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() == expected:
        return True
    if path.suffix.lower() not in {'.json', '.csv', '.md', '.yml', '.yaml'}:
        return False
    lf = raw.replace(b'\r\n', b'\n')
    if hashlib.sha256(lf).hexdigest() == expected:
        return True
    crlf = lf.replace(b'\n', b'\r\n')
    return hashlib.sha256(crlf).hexdigest() == expected


def assert_close(actual, expected, *, label, atol=3e-10):
    if not np.isclose(float(actual), float(expected), atol=atol, rtol=0):
        raise AssertionError(f'{label} DOES NOT MATCH reference: actual={actual}, expected={expected}')


def quantile_loss_each(actual, q, a=0.05):
    e = np.asarray(actual, float) - np.asarray(q, float)
    return np.maximum(a * e, (a - 1.0) * e)


def binomial_log_lik(k, n, p):
    if n == 0: return 0.0
    if p <= 0.0: return 0.0 if k == 0 else float('-inf')
    if p >= 1.0: return 0.0 if k == n else float('-inf')
    return float(k * np.log(p) + (n-k) * np.log1p(-p))


def evaluate_backtest(y, q, alpha):
    y, q = np.asarray(y, float), np.asarray(q, float)
    if len(y) == 0 or y.shape != q.shape: raise ValueError('Invalid backtesting vectors')
    violation = y < q
    n = int(len(y)); k = int(violation.sum()); rate = k / n
    l0 = binomial_log_lik(k, n, alpha)
    l1 = binomial_log_lik(k, n, rate)
    lr_uc = max(0.0, 2.0 * (l1 - l0))
    v = violation.astype(int)
    n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
    n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
    n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
    n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))
    pi0 = n01/(n00+n01) if n00+n01 else 0.0
    pi1 = n11/(n10+n11) if n10+n11 else 0.0
    pi = (n01+n11)/(n00+n01+n10+n11) if n > 1 else 0.0
    null = binomial_log_lik(n01+n11, n-1, pi) if n > 1 else 0.0
    alt = binomial_log_lik(n01, n00+n01, pi0) + binomial_log_lik(n11, n10+n11, pi1)
    lr_ind = max(0.0, 2.0*(alt-null))
    return {
        'n': n, 'breaches': k, 'breach_rate': rate, 'coverage_rate': 1.0-rate,
        'expected_breach_rate': alpha,
        'pinball_loss': float(quantile_loss_each(y, q, alpha).mean()),
        'mean_var': float(np.maximum(0.0, -q).mean()),
        'kupiec_lr_uc': lr_uc, 'kupiec_p': float(chi2.sf(lr_uc,1)),
        'christoffersen_lr_ind': lr_ind, 'christoffersen_independence_p': float(chi2.sf(lr_ind,1)),
        'conditional_coverage_lr': lr_uc+lr_ind,
        'conditional_coverage_p': float(chi2.sf(lr_uc+lr_ind,2)),
        'transition_counts': {'n00':n00,'n01':n01,'n10':n10,'n11':n11},
    }


def load_inputs(root: Path, cfg: dict):
    path = root / cfg['input_features_csv']
    if not path.is_file():
        raise FileNotFoundError(f'Missing REAL features CSV: {path}. Use Phase2 snapshot; do not use synthetic test data.')
    digest = sha(path)
    if digest != cfg['input_features_sha256']:
        if not sha_matches_eol(path, cfg['input_features_sha256']):
            raise ValueError(f'INPUT SHA256 MISMATCH\nexpected={cfg["input_features_sha256"]}\nactual={digest}\nNO results are published.')
        print('SOURCE CHECK: LF/CRLF-only checkout difference; canonical hash verified.', flush=True)
        # The canonical frozen SHA represents the same numerical CSV content.
        # Preserve it in result metadata and verify-only comparisons.
        digest = cfg['input_features_sha256']
    df = pd.read_csv(path, parse_dates=['target_date','as_of'], float_precision='round_trip')
    expected_columns = ['target_date','as_of',*cfg['feature_names'],'actual_return']
    if df.columns.tolist() != expected_columns: raise ValueError('Feature schema mismatch')
    if len(df) != cfg['data_rows']: raise ValueError('Data length mismatch')
    if df.target_date.isna().any() or df.as_of.isna().any(): raise ValueError('Missing date')
    if df.target_date.duplicated().any(): raise ValueError('Duplicate target date')
    if not df.target_date.is_monotonic_increasing: raise ValueError('Out-of-order dates')
    if not (df.as_of < df.target_date).all(): raise ValueError('Lookahead leakage in feature timing')
    if df.target_date.iloc[-1].date().isoformat() != cfg['snapshot_cutoff']:
        raise ValueError('Snapshot date mismatch')
    cols = [*cfg['feature_names'],'actual_return']
    if not np.isfinite(df[cols].to_numpy(float)).all(): raise ValueError('Nonfinite feature or target values')
    development = df.loc[df.target_date <= cfg['development_end_inclusive']].copy().reset_index(drop=True)
    holdout = df.loc[df.target_date >= cfg['holdout_start_inclusive']].copy().reset_index(drop=True)
    if len(development) != cfg['development_rows'] or len(holdout) != cfg['holdout_rows']:
        raise ValueError(f'Dev/Holdout mismatch: {len(development)} / {len(holdout)}')
    if development.target_date.max() >= holdout.target_date.min(): raise ValueError('Train/Test date overlap')
    return df, development, holdout, digest


def time_splits(n, cfg):
    n_folds = 5
    width = cfg['cv_validation_rows_per_fold']
    first = cfg['cv_first_train_rows']
    if first + n_folds * width != n: raise ValueError('CV count mismatches frozen configuration')
    for j in range(n_folds):
        split = first+j*width
        yield j+1, np.arange(split), np.arange(split, split+width)


def fit_predict(train, valid, cfg):
    x = cfg['feature_names']
    params = {k: v for k, v in cfg['model'].items() if k != 'class'}
    mdl = GradientBoostingRegressor(**params)
    mdl.fit(train[x],train['actual_return'].to_numpy(float))
    q = mdl.predict(valid[x])
    return mdl, np.asarray(q, float)


def reproduce_cv(dev, cfg):
    rows=[]; preds=[]
    for fold, tr_idx, val_idx in time_splits(len(dev),cfg):
        tr=dev.iloc[tr_idx]; va=dev.iloc[val_idx]
        if tr.target_date.max() >= va.target_date.min(): raise ValueError('Time leakage')
        _m,q = fit_predict(tr,va,cfg)
        y=va.actual_return.to_numpy(float)
        breaches=y<q
        rows.append({
            'fold':fold,'train_n':len(tr),'validation_n':len(va),
            'train_start':tr.target_date.iloc[0].date().isoformat(),
            'train_end':tr.target_date.iloc[-1].date().isoformat(),
            'validation_start':va.target_date.iloc[0].date().isoformat(),
            'validation_end':va.target_date.iloc[-1].date().isoformat(),
            'pinball_loss':float(quantile_loss_each(y,q,cfg['quantile_alpha']).mean()),
            'breaches':int(breaches.sum()),'breach_rate':float(breaches.mean())
        })
        p=va[['as_of','target_date','actual_return']].copy()
        p['fold']=fold;p['q05_return']=q;p['VaR95']=np.maximum(0.,-q);p['breach']=breaches
        preds.append(p)
        print(f'[CV {fold}/5] train={len(tr):4d} valid={len(va):3d} loss={rows[-1]["pinball_loss"]:.9f}',flush=True)
    met=pd.DataFrame(rows)
    pred=pd.concat(preds,ignore_index=True)
    assert_close(met.pinball_loss.mean(),cfg['canonical_reference_cv_pinball'],label='C1 frozen CV mean Pinball')
    if len(pred)!=cfg['canonical_reference_cv_n']: raise AssertionError('Frozen CV n mismatch')
    return met,pred


def compare_references(root,cfg,holdout_result):
    p=root/cfg['canonical_reference_predictions']
    if not p.exists(): raise FileNotFoundError(f'Missing canonical GitHub prediction reference: {p}')
    ref=pd.read_csv(p,parse_dates=['target_date'],float_precision='round_trip')
    ref=ref[ref['model']==cfg['canonical_reference_model']].copy().reset_index(drop=True)
    if len(ref)!=len(holdout_result): raise AssertionError('Reference C1 row count mismatch')
    if not np.array_equal(ref.target_date.to_numpy(),holdout_result.target_date.to_numpy()):
        raise AssertionError('Reference dates not aligned')
    differences=np.abs(ref.quantile_return.to_numpy(float)-holdout_result.q05_return.to_numpy(float))
    max_diff=float(differences.max())
    if max_diff>1e-8: raise AssertionError(f'REPRODUCTION FAILURE: max_abs_quantile_difference={max_diff:.12g}; check scikit-learn version and input.')
    if not np.array_equal(ref.actual_return.to_numpy(float),holdout_result.actual_return.to_numpy(float)):
        assert_close(np.max(np.abs(ref.actual_return.to_numpy(float)-holdout_result.actual_return.to_numpy(float))),0.0,label='reference actual return',atol=1e-13)
    return {'reference_csv':str(cfg['canonical_reference_predictions']), 'max_abs_quantile_difference':max_diff,'reproduced_reference_predictions':True}


def plot_var_and_breaches(hold,metrics,output):
    x=hold.target_date
    y=hold.actual_return.to_numpy(float)*100
    q=hold.q05_return.to_numpy(float)*100
    breach=hold.breach.to_numpy(bool)
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(14,8.6),gridspec_kw={'height_ratios':[3.7,1.2]},sharex=True)
    fig.patch.set_facecolor('#ffffff')
    ax.plot(x,y,lw=0.9,color='#7290a7',alpha=.8,label='Actual portfolio return (%)')
    ax.plot(x,q,lw=1.65,color='#ef8f42',label='Forecast 5th-percentile threshold (negative VaR)')
    ax.scatter(x[breach],y[breach],s=37,color='#c62d38',marker='x',lw=1.7,zorder=6,label=f'VaR exceedances: {int(breach.sum())}')
    ax.axhline(0,lw=.8,color='#9ba6b0',alpha=.65)
    ax.set_ylabel('Daily return (%)',fontsize=11)
    ax.set_title('FROZEN C1 (12 features)  |  1-day VaR 95% backtest',fontsize=17,weight='bold',pad=13)
    ax.grid(axis='y',color='#c9d0d9',alpha=.4)
    ax.legend(loc='upper right',fontsize=9,frameon=False,ncol=1)
    ax.text(.01,.02,f'Holdout: {metrics["n"]} sessions | Exceedances: {metrics["breaches"]} ({metrics["breach_rate"]:.2%}) | Expected: 5%',transform=ax.transAxes,fontsize=10,color='#263849',bbox={'boxstyle':'round,pad=.4','facecolor':'white','edgecolor':'#dfe4e9','alpha':.92})
    rolling=pd.Series(breach.astype(float),index=x).rolling(60,min_periods=60).mean()*100
    ax2.plot(rolling.index,rolling.values,color='#254e7b',lw=1.65,label='Rolling exceedance rate (60 trading days)')
    ax2.axhline(5,color='#cf4b4b',ls='--',lw=1.3,label='Nominal 5% exceedance rate')
    ax2.set_ylabel('Breach rate (%)',fontsize=11)
    ax2.set_xlabel('Validation target date',fontsize=11)
    ax2.grid(axis='y',color='#c9d0d9',alpha=.4)
    ax2.legend(loc='upper right',fontsize=9,frameon=False)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout(pad=2)
    fig.savefig(output,dpi=220,bbox_inches='tight')
    plt.close(fig)


def plot_coverage(hold,metrics,output):
    k=metrics['breaches']; n=metrics['n']
    fig,axs=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={'width_ratios':[1.25,1]})
    fig.patch.set_facecolor('white')
    axs[0].hist(hold.q05_return.to_numpy(float)*(-100),bins=35,color='#254e7b',alpha=.85)
    axs[0].set_xlabel('Forecasted 5% tail loss threshold (% of portfolio)')
    axs[0].set_ylabel('Count of sessions')
    axs[0].set_title('C1 VaR forecast distribution',weight='bold')
    axs[0].grid(axis='y',alpha=.25)
    vals=[5.0,100*k/n]
    bars=axs[1].bar(['Nominal','Observed'],vals,color=['#8ea0b1','#c62d38'],width=.48)
    axs[1].set_ylim(0,max(vals)*1.45)
    axs[1].set_ylabel('Exceedance rate (%)')
    axs[1].set_title('Coverage quality',weight='bold')
    for bar, v in zip(bars,vals): axs[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+.12,f'{v:.2f}%',ha='center',fontsize=11)
    axs[1].grid(axis='y',alpha=.25)
    fig.suptitle(f'Kupiec p={metrics["kupiec_p"]:.3f}  |  Independence p={metrics["christoffersen_independence_p"]:.3f}',fontsize=12)
    fig.tight_layout()
    fig.savefig(output,dpi=220,bbox_inches='tight');plt.close(fig)


def plot_training_validation(dev,cfg,output,csv):
    folds=list(time_splits(len(dev),cfg))
    _,train_idx,val_idx=folds[-1]
    tr=dev.iloc[train_idx];va=dev.iloc[val_idx]
    mdl,_=fit_predict(tr,va,cfg)
    a=cfg['quantile_alpha']
    trloss=[float(quantile_loss_each(tr.actual_return.to_numpy(float),p,a).mean()) for p in mdl.staged_predict(tr[cfg['feature_names']])]
    vloss=[float(quantile_loss_each(va.actual_return.to_numpy(float),p,a).mean()) for p in mdl.staged_predict(va[cfg['feature_names']])]
    p=pd.DataFrame({'boosting_iteration':np.arange(1,len(trloss)+1),'training_pinball':trloss,'validation_pinball':vloss})
    p.to_csv(csv,index=False,float_format='%.17g')
    fig,ax=plt.subplots(figsize=(11,5.5))
    ax.plot(p.boosting_iteration,p.training_pinball*1000,lw=2,color='#254e7b',label='Train (fold 5)')
    ax.plot(p.boosting_iteration,p.validation_pinball*1000,lw=2,color='#e18d45',label='Validation (fold 5)')
    ax.set_xlabel('Number of trees / boosting iterations')
    ax.set_ylabel('Pinball Loss x1000')
    ax.set_title('Frozen C1 | Training vs. Validation Loss — Fold 5',weight='bold',fontsize=15)
    ax.grid(alpha=.3);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(output,dpi=220,bbox_inches='tight');plt.close(fig)


def read_metrics(path):
    if not path.exists(): raise FileNotFoundError(f'Missing frozen result for --verify-only: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--verify-only',action='store_true',help='Recompute and compare frozen metrics without changing files')
    args=p.parse_args()
    root=args.root.resolve(); cfg_path=root/CONFIG_REL
    if not cfg_path.is_file(): raise FileNotFoundError(f'Missing frozen config: {cfg_path}')
    cfg=json.loads(cfg_path.read_text(encoding='utf-8'))
    if cfg['id'] != 'C1_12F_FROZEN_PHASE2_2026_11_06': raise ValueError('Wrong frozen model ID')
    print('PROJECT:',root,flush=True)
    print('DATA CONFIG SHA256:',sha(cfg_path),flush=True)
    _full,dev,hold,digest=load_inputs(root,cfg)
    print(f'PINNED REAL DATA PASS | sha256={digest} | dev={len(dev)} | holdout={len(hold)}',flush=True)
    fold_metrics,cv_pred=reproduce_cv(dev,cfg)
    mdl,q=fit_predict(dev,hold,cfg)
    y=hold.actual_return.to_numpy(float)
    breach=(y<q)
    out_hold=hold[['as_of','target_date','actual_return']].copy()
    out_hold['q05_return']=q
    out_hold['VaR95']=np.maximum(0.,-q)
    out_hold['breach']=breach
    out_hold['pinball_loss_per_day']=quantile_loss_each(y,q,cfg['quantile_alpha'])
    metrics=evaluate_backtest(y,q,cfg['quantile_alpha'])
    assert_close(metrics['pinball_loss'],cfg['canonical_reference_holdout_pinball'],label='C1 holdout pinball')
    if metrics['breaches']!=cfg['canonical_reference_holdout_breaches']: raise AssertionError('C1 holdout breach count mismatch')
    refcheck=compare_references(root,cfg,out_hold)
    outdir=root/OUT_REL
    report={
        'model_id':cfg['id'],'config_sha256':sha(cfg_path),'input_features_sha256':digest,
        'original_reference_branch':'phase2/develop',
        'evaluation':'retrospective_holdout_not_pristine',
        'model_params':cfg['model'],'feature_names':cfg['feature_names'],
        'cv_mean_pinball':float(fold_metrics.pinball_loss.mean()),
        'cv_breach_rate':float(fold_metrics.breaches.sum()/fold_metrics.validation_n.sum()),
        'cv_n':int(fold_metrics.validation_n.sum()),
        'holdout':metrics,'reference_reproduction':refcheck,
        'environment':{'python':platform.python_version(),'scikit_learn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__},
        'reproducible_metrics':True,
        'notes':['This is original frozen C1, not the later experimental C1 Optimized.','Not a live forecast; snapshot cutoff 2026-11-06.',cfg['limitation']],
    }
    print(f'CV PINBALL: {report["cv_mean_pinball"]:.12f} | breaches {report["cv_breach_rate"]:.2%}',flush=True)
    print(f'HOLDOUT PINBALL: {metrics["pinball_loss"]:.12f} | breaches {metrics["breaches"]}/{metrics["n"]} ({metrics["breach_rate"]:.2%})',flush=True)
    print(f'BACKTEST Kupiec={metrics["kupiec_p"]:.6f} | Christoffersen independence={metrics["christoffersen_independence_p"]:.6f}',flush=True)
    print(f'REFERENCE REPRODUCTION: PASS | max absolute q05 error = {refcheck["max_abs_quantile_difference"]:.12g}',flush=True)

    if args.verify_only:
        old=read_metrics(outdir/'FINAL_C1_BACKTEST.json')
        if old['input_features_sha256']!=digest or not sha_matches_eol(cfg_path, old['config_sha256']): raise AssertionError('Saved input/config hash differs')
        for key in ['cv_mean_pinball','cv_breach_rate']:
            assert_close(report[key],old[key],label=f'Saved {key}')
        for key in ['pinball_loss','breach_rate','kupiec_p','christoffersen_independence_p']:
            assert_close(report['holdout'][key],old['holdout'][key],label=f'Saved {key}')
        csv=outdir/'FINAL_C1_HOLDOUT.csv'
        if not csv.exists(): raise FileNotFoundError(csv)
        saved=pd.read_csv(csv,parse_dates=['target_date'],float_precision='round_trip')
        if len(saved)!=len(out_hold) or not np.array_equal(saved.target_date.to_numpy(),out_hold.target_date.to_numpy()): raise AssertionError('Saved holdout dates differ')
        if np.max(np.abs(saved.q05_return.to_numpy(float)-q))>1e-10: raise AssertionError('Saved holdout forecast differs')
        manifest=read_metrics(outdir/'ARTIFACT_SHA256.json')
        for filename, expected_sha in manifest.items():
            fp=outdir/filename
            if not fp.is_file() or not sha_matches_eol(fp, expected_sha):
                raise AssertionError(f'Frozen submission file is missing or modified: {filename}')
        archived=joblib.load(outdir/'FINAL_C1_MODEL.joblib')
        archived_q=np.asarray(archived.predict(hold[cfg['feature_names']]),float)
        if np.max(np.abs(archived_q-q))>1e-10:
            raise AssertionError('Saved C1 model predictions do not reproduce')
        print('REPRODUCIBILITY VERIFIED: PASS | frozen input + original reference + all saved files + serialized model + metrics',flush=True)
        return

    outdir.mkdir(parents=True,exist_ok=True)
    fold_metrics.to_csv(outdir/'FINAL_C1_CV_METRICS.csv',index=False,float_format='%.17g')
    cv_pred.to_csv(outdir/'FINAL_C1_CV_PREDICTIONS.csv',index=False,float_format='%.17g')
    out_hold.to_csv(outdir/'FINAL_C1_HOLDOUT.csv',index=False,float_format='%.17g')
    joblib.dump(mdl, outdir/'FINAL_C1_MODEL.joblib', compress=3)
    (outdir/'FINAL_C1_BACKTEST.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    plot_var_and_breaches(out_hold,metrics,outdir/'01_C1_VaR_with_Breaches.png')
    plot_coverage(out_hold,metrics,outdir/'02_C1_Coverage.png')
    plot_training_validation(dev,cfg,outdir/'03_C1_Training_Validation_Loss.png',outdir/'FINAL_C1_TRAIN_VALID_LOSS_FOLD5.csv')
    md=f'''# KẾT QUẢ CUỐI CÙNG — C1 gốc đã khóa\n\n**Một mô hình duy nhất:** Gradient Boosting quantile regression C1 (12 đặc trưng; 150 cây; depth 2; learning rate 0.03; min_samples_leaf 10). Dự báo **phân vị 5% lợi suất danh mục một phiên tiếp theo**, tương ứng VaR 95%.\n\n- Nguồn dữ liệu: KBS; HPG/FPT/MWG tỷ trọng 1/3, snapshot đến ngày **06/11/2026**.\n- SHA256 bộ features thực: `{digest}`.\n- 1.179 quan sát development, 468 phiên holdout hồi cứu; CV 5 folds không xáo trộn thời gian.\n- CV mean Pinball Loss: **{report['cv_mean_pinball']:.9f}**.\n- Holdout Pinball Loss: **{metrics['pinball_loss']:.9f}**.\n- Vi phạm VaR: **{metrics['breaches']}/{metrics['n']} = {metrics['breach_rate']:.2%}** (kỳ vọng 5%).\n- Kupiec p-value: **{metrics['kupiec_p']:.5f}**. Christoffersen independence p-value: **{metrics['christoffersen_independence_p']:.5f}**.\n- Sai khác lớn nhất so với file dự báo C1 gốc đã công bố trong repository: **{refcheck['max_abs_quantile_difference']:.3g}**.\n\n## Giải thích biểu đồ\n\n`01_C1_VaR_with_Breaches.png`: đường xanh xám là lợi suất thực tế; đường cam là **ngưỡng phân vị 5% được dự báo trước phiên** (giá trị âm, tương ứng với ngưỡng VaR). Dấu X đỏ đánh dấu ngày có lợi suất thực tế thấp hơn ngưỡng. Biểu đồ dưới cho thấy tỷ lệ vi phạm trượt 60 phiên, đối chiếu mức danh nghĩa 5%. **Không đánh giá chất lượng VaR bằng việc đường cam bám sát lợi suất thực tế**.\n\n`02_C1_Coverage.png`: phân phối VaR và mức vi phạm thực tế so với mức 5%.\n\n`03_C1_Training_Validation_Loss.png`: đường loss theo 150 cây ở Fold 5 để quan sát quá trình học; không thay thế backtesting.\n\n## Giới hạn\n\n- Đây là một phép **tái tạo thực nghiệm cũ**, không phải dự báo thị trường trực tiếp ngày 19/11/2026.\n- Holdout hồi cứu giao thoa giai đoạn Phase 1 đã phân tích, **không phải holdout hoàn toàn chưa từng được xem xét**.\n- Tỷ lệ vi phạm gần 5% và p-value lớn **không chứng minh mô hình hoàn hảo**; chỉ cho biết kiểm định chưa bác bỏ giả thuyết ở mức ý nghĩa tương ứng.\n- Không thực hiện tối ưu tham số hoặc chọn lại mô hình dựa vào holdout.\n'''
    (outdir/'REPORT_FINAL_C1_VI.md').write_text(md,encoding='utf-8')
    (outdir/'ARTIFACT_SHA256.json').write_text(json.dumps({p.name:sha(p) for p in outdir.iterdir() if p.is_file() and p.name!='ARTIFACT_SHA256.json'},indent=2),encoding='utf-8')
    print('OUTPUT:',outdir,flush=True)
    print('FROZEN C1 REAL-DATA REPRODUCTION: PASS | CV + holdout + reference forecast + plots',flush=True)

if __name__=='__main__':
    main()
