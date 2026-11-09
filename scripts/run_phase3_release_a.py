"""P3-06: Phase3 A-only technical evidence and offline release, without training.

Read only frozen Phase2 evidence, P3-01..05 outputs and Phase3 source code.
Never reads the original Phase2 holdout predictions, metrics or return dates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from src.phase3.release_a import (
    CUTOFF, SOURCE_SHA, FEATURE12_SHA, TRAINING_END, POLICIES,
    sha256, audit_manifest_artifacts, verify_feature_ranking,
    verify_policy_predictions, decision_gate, standalone_html
)

ROOT=Path(__file__).resolve().parents[1]
STAGES=('diagnostics','tuning','stability')


def run(root:Path, out:Path):
    root=root.resolve();out=out.resolve()
    out.relative_to(root)
    if out.exists():
        raise FileExistsError('P3-06 immutable release directory already exists')
    feature12=root/f'results/phase2/a/features_{CUTOFF}.csv'
    lock_file=root/f'results/phase3/a/baseline_{CUTOFF}/baseline_lock.json'
    stage_dirs={x:root/f'results/phase3/a/{x}_{CUTOFF}' for x in STAGES}
    for path in [feature12,lock_file,*[p/'manifest.json' for p in stage_dirs.values()]]:
        if not path.is_file():
            raise FileNotFoundError('Missing release evidence: '+str(path))
    if sha256(feature12)!=FEATURE12_SHA:
        raise ValueError('Frozen Phase2 features changed')
    lock=json.loads(lock_file.read_text(encoding='utf-8-sig'))
    if (lock.get('source_sha256')!=SOURCE_SHA or
            lock.get('features_sha256')!=FEATURE12_SHA or
            lock.get('training_cutoff')!=TRAINING_END or
            lock.get('selected_on_development_cv')!='gbv2_12f_c1_a'):
        raise ValueError('Frozen P3-01 lock invalid')
    manifests={x:json.loads((stage_dirs[x]/'manifest.json').read_text(encoding='utf-8-sig'))
               for x in STAGES}
    diag,tuning,stability=(manifests[x] for x in STAGES)
    if (diag.get('source_sha256')!=SOURCE_SHA or
            diag.get('baseline_lock_sha256')!=sha256(lock_file) or
            diag.get('old_phase2_holdout_untouched') is not True):
        raise ValueError('P3-03 evidence inconsistent')
    if (tuning.get('source_sha256')!=SOURCE_SHA or
            tuning.get('baseline_lock_sha256')!=sha256(lock_file) or
            tuning.get('p3_03_manifest_sha256')!=sha256(stage_dirs['diagnostics']/'manifest.json') or
            tuning.get('snapshot_cutoff')!=CUTOFF or
            tuning.get('training_cutoff')!=TRAINING_END or
            tuning.get('inner_fit_count')!=45 or
            'NEVER READ' not in tuning.get('holdout_policy','')):
        raise ValueError('P3-04 evidence inconsistent')
    if (stability.get('source_sha256')!=SOURCE_SHA or
            stability.get('features12_sha256')!=FEATURE12_SHA or
            stability.get('p304_manifest_sha256')!=sha256(stage_dirs['tuning']/'manifest.json') or
            stability.get('snapshot_cutoff')!=CUTOFF or
            stability.get('development_last')!=TRAINING_END or
            stability.get('original_holdout_used') is not False or
            stability.get('selection_from_p305') is not False):
        raise ValueError('P3-05 evidence inconsistent')

    audited={}
    for name in STAGES:
        audited[name]=audit_manifest_artifacts(stage_dirs[name],manifests[name],
                                               bare_csv_keys=name=='diagnostics')
    # Feature-count outputs contain independent hashes for 18 and 25 features.
    feature_dir=root/f'results/phase3/a/features_{CUTOFF}'
    feature_hashes={'12':sha256(feature12)}
    base_frame=pd.read_csv(feature12,float_precision='round_trip')
    base_columns=base_frame.columns.tolist()
    for n in (18,25):
        f=feature_dir/f'features_{n}_{CUTOFF}.csv'
        meta_path=feature_dir/f'features_{n}_{CUTOFF}_metadata.json'
        meta=json.loads(meta_path.read_text(encoding='utf-8-sig'))
        if (sha256(f)!=meta.get('sha256') or
                meta.get('source_sha256')!=SOURCE_SHA or
                meta.get('feature_count')!=n):
            raise ValueError(f'Phase3 {n} feature hash/schema mismatch')
        df=pd.read_csv(f,float_precision='round_trip')
        if (len(df)!=1647 or df.columns[:14].tolist()!=base_columns[:14] or
                df.columns[-1]!='actual_return' or len(df.columns)!=n+3):
            raise ValueError(f'Phase3 {n} feature shape or column order changed')
        if not np.allclose(df[base_columns[2:-1]].to_numpy(float),
                           base_frame[base_columns[2:-1]].to_numpy(float),atol=1e-12,rtol=1e-10):
            raise ValueError('Frozen 12 features differ from Phase3 extra features')
        if not (df.target_date.equals(base_frame.target_date) and
                df.as_of.equals(base_frame.as_of) and
                np.allclose(df.actual_return.to_numpy(float),base_frame.actual_return.to_numpy(float),atol=1e-12,rtol=0)):
            raise ValueError('Feature target or label mismatch')
        feature_hashes[str(n)]=sha256(f)

    rankings=pd.read_csv(stage_dirs['diagnostics']/'cv_ranking.csv')
    folds=pd.read_csv(stage_dirs['diagnostics']/'cv_fold_metrics.csv')
    ranked=verify_feature_ranking(rankings,folds)
    stats=pd.read_csv(stage_dirs['stability']/'policy_metrics.csv')
    preds=pd.read_csv(stage_dirs['stability']/'walkforward_predictions.csv')
    summary=verify_policy_predictions(preds,stats,expected_n=600)
    paired=pd.read_csv(stage_dirs['stability']/'paired_block_bootstrap.csv')
    gate=decision_gate(paired)
    figures=sorted((stage_dirs['stability']/'figures').glob('*.png'))
    if len(figures)!=4:
        raise ValueError('Expected four P3-05 plots')
    for figure in figures:
        if figure.relative_to(stage_dirs['stability']).as_posix() not in audited['stability']:
            raise ValueError('Unauthenticated figure')
    oof=pd.read_csv(stage_dirs['tuning']/'outer_cv_predictions.csv')
    if len(oof)!=1200:
        raise ValueError('Nested outer output count changed')
    schedule=pd.read_csv(stage_dirs['stability']/'refit_schedule.csv')
    if len(schedule)!=60:
        raise ValueError('Refit schedule count changed')

    out.mkdir(parents=True,exist_ok=False)
    title='Portfolio VaR â€” Phase 3 A Exploratory Research Review'
    view=standalone_html(title,ranked,stats,gate,figures)
    html_path=out/'phase3_review_offline.html'
    html_path.write_text(view,encoding='utf-8')
    summary_path=out/'experiment_summary.md'
    def md_table(frame, columns):
        header='| '+' | '.join(columns)+' |'
        divider='| '+' | '.join(['---']*len(columns))+' |'
        data=['| '+' | '.join(str(row[col]) for col in columns)+' |' for _,row in frame[columns].iterrows()]
        return '\n'.join([header,divider,*data])

    lines=[
        '# Phase 3 A â€” Exploratory VaR optimization summary',
        '',
        '## Scope',
        '- Objective: conditional 5th-percentile portfolio simple-return forecast (VaR 95% for T+1), not trend classification.',
        f'- KBS snapshot cutoff: {CUTOFF}; development through {TRAINING_END}.',
        '- Phase 2 baseline: GBv2 C1 12 features, frozen and SHA256-verified.',
        '- Original historical Phase 2 holdout NOT used for selection or Phase 3 evaluation.',
        '',
        '## Feature-count screen (exploratory development CV)',
        md_table(ranked,['feature_count','cv_pinball']),
        '',
        '## Policies on 600 paired development targets',
        md_table(stats,['policy','n','violations','violation_rate','mean_pinball']),
        '',
        '## Result / promotion gate',
        f'- {gate["status"]}: {gate["reason"]}',
        '- No claim that rolling refitting is significantly or prospectively better.',
        '- Retain GBv2 C1 as the research reference; do not promote any experiment to the live forecast registry.',
        '',
        '## Limitations and next evaluation',
        '- Historical development data were examined during earlier experiments: all comparisons remain exploratory.',
        '- A clean future, pre-registered timestamped evaluation period is needed for confirmation.',
        '- Backtesting non-rejection is not evidence of perfect model calibration.',
        '- Check original B snapshot hash, common A/B protocol, and unresolved VNINDEX flags separately.',
        '- Retrospective predictions cannot be registered as genuine historical real-time forecasts.',
        '',
        '## Reproducibility',
        '- All P3-03..P3-05 artifacts and Figure SHA256 hashes audited against manifests.',
        '- A 600-session paired prediction and 60-refit schedule audit was performed without any model fit here.',
        '- See `release_manifest.json` for full evidence SHA256 and input provenance.',
        ''
    ]
    summary_path.write_text('\n'.join(lines),encoding='utf-8')
    report={
        'phase':'Phase 3 A','status':'AUDIT_PASS_EXPLORATORY_WITH_LIMITATIONS',
        'snapshot_cutoff':CUTOFF,'development_last':TRAINING_END,
        'source_sha256':SOURCE_SHA,'features12_sha256':FEATURE12_SHA,
        'feature_hashes':feature_hashes,
        'baseline_lock_sha256':sha256(lock_file),
        'phase3_manifests_sha256':{name:sha256(stage_dirs[name]/'manifest.json') for name in STAGES},
        'audited_prior_artifacts_sha256':audited,
        'feature_count_cv_ranking':ranked[['feature_count','cv_pinball']].to_dict(orient='records'),
        'walkforward_policies':summary,
        'paired_bootstrap_ci_includes_zero':{str(r.alternative):bool(r.ci_includes_zero) for r in paired.itertuples()},
        'promotion_gate':gate,
        'reference_model':'gbv2_12f_c1_a',
        'new_production_model_selected':False,'original_holdout_used':False,
        'phase2_modified':False,
        'output_artifacts':{p.name:sha256(p) for p in (html_path,summary_path)},
        'python':platform.python_version(),
        'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'source_script_sha256':sha256(Path(__file__)),
    }
    (out/'release_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('PHASE 3 MANIFESTS AND ARTIFACT SHA256: PASS',flush=True)
    print('FEATURE SETS: 12/18/25; 12F PRIMARY BASELINE IMMUTABLE',flush=True)
    print('FEATURE CV RANKING (EXPLORATORY):\n',ranked[['feature_count','cv_pinball']].to_string(index=False),flush=True)
    print('WALKFORWARD SUMMARY (EXPLORATORY):\n',stats[['policy','n','violations','violation_rate','mean_pinball']].to_string(index=False),flush=True)
    print('PROMOTION GATE:',gate['status'],flush=True)
    print('PHASE3 FIGURES:',len(figures),'PREDICTION ROWS:',len(preds),'REFITS:',len(schedule),flush=True)
    print('OLD HOLDOUT USED: NO; PHASE2 MODIFIED: NO; PRODUCTION MODEL PROMOTED: NO')
    print('OFFLINE HTML:',html_path,flush=True)
    print('RELEASE MANIFEST:',out/'release_manifest.json',flush=True)
    print('STEP P3-06 RELEASE AUDIT PASS WITH LIMITATIONS',flush=True)
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    run(ROOT,args.out)


if __name__=='__main__':
    main()
