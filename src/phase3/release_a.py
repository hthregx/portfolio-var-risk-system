"""Phase 3 A: conservative, read-only evidence audit and standalone HTML release.

This module performs NO model fitting, NO model promotion and NO holdout inspection.
P3-06 outputs are an exploratory development-evidence package only.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

CUTOFF = '2026-11-06'
TRAINING_END = '2024-12-17'
SOURCE_SHA = '6389586ae8f1edbabb31760e0b555619186114b3f77d09212362fc2e1c40aadf'
FEATURE12_SHA = '14ac5dd4ce7dd6090420b9cee06c8ac40935c54691b4daf95d063b71a075eccc'
POLICIES = ('C1_frozen', 'nested_selected', 'expanding_refit20', 'rolling504_refit20')


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_artifact_path(root: Path, key: str, *, bare_csv_keys: bool = False) -> Path:
    """Resolve manifest paths without traversal, absolute paths or OS ambiguities."""
    if not isinstance(key, str) or not key or '\0' in key:
        raise ValueError('Invalid artifact key')
    k = key.replace('\\', '/')
    candidate = PurePosixPath(k)
    if (candidate.is_absolute() or ':' in candidate.parts[0] or
            '..' in candidate.parts or k.endswith('/') or k.startswith('//')):
        raise ValueError('Unsafe artifact path: '+key)
    if bare_csv_keys and candidate.suffix == '':
        k += '.csv'
    base = Path(root).resolve()
    path = (base / k).resolve()
    if path == base or base not in path.parents:
        raise ValueError('Artifact path escapes stage directory')
    return path


def audit_manifest_artifacts(stage: Path, manifest: dict, *, bare_csv_keys=False) -> dict:
    records = manifest.get('artifacts')
    if not isinstance(records, dict) or not records:
        raise ValueError('Missing artifact registry')
    audited = {}
    for key, record in records.items():
        path = safe_artifact_path(stage, key, bare_csv_keys=bare_csv_keys)
        if not path.is_file():
            raise FileNotFoundError('Missing artifact: '+str(path))
        digest = sha256(path)
        if digest != record.get('sha256'):
            raise ValueError('SHA256 mismatch: '+key)
        if 'rows' in record and path.suffix.lower() == '.csv':
            # pandas avoids miscounting a trailing newline as an extra record.
            count = len(pd.read_csv(path))
            if count != int(record['rows']):
                raise ValueError('Row count mismatch: '+key)
        audited[path.relative_to(stage).as_posix()] = digest
    return audited


def verify_feature_ranking(ranking: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if set(ranking.feature_count.astype(int)) != {12,18,25} or len(ranking) != 3:
        raise ValueError('Expected exactly 12/18/25 feature ranking')
    if not {'feature_count','cv_pinball'}.issubset(ranking.columns):
        raise ValueError('Missing ranking fields')
    if not {'feature_count','fold','val_pinball'}.issubset(fold_metrics.columns):
        raise ValueError('Missing CV fold fields')
    for n in (12,18,25):
        block = fold_metrics.loc[fold_metrics.feature_count == n]
        if (len(block) != 5 or block.fold.nunique() != 5 or
                not np.isfinite(block.val_pinball.to_numpy(float)).all()):
            raise ValueError('Invalid 5-fold records')
        mean = float(block.val_pinball.mean())
        reported = float(ranking.loc[ranking.feature_count == n,'cv_pinball'].iloc[0])
        if not np.isclose(mean, reported, rtol=0, atol=1e-12):
            raise ValueError('CV ranking does not match fold scores')
    ranked = ranking.sort_values('cv_pinball',kind='stable').reset_index(drop=True)
    if int(ranked.feature_count.iloc[0]) != 12:
        raise ValueError('Frozen experimental interpretation changed; inspect manually')
    return ranked


def _bool_flags(values) -> np.ndarray:
    out=[]
    for x in values:
        if isinstance(x,(bool,np.bool_)):
            out.append(bool(x))
        elif isinstance(x,str) and x.lower() in ('true','false'):
            out.append(x.lower()=='true')
        else:
            raise ValueError('Invalid violation value')
    return np.asarray(out,dtype=bool)


def pinball(y, q, alpha=0.05):
    y=np.asarray(y,float);q=np.asarray(q,float)
    if y.ndim!=1 or y.shape!=q.shape or not np.isfinite(y).all() or not np.isfinite(q).all():
        raise ValueError('Invalid prediction arrays')
    return np.maximum(alpha*(y-q),(alpha-1)*(y-q))


def verify_policy_predictions(df: pd.DataFrame, metrics:pd.DataFrame, expected_n:int=600) -> dict:
    required={'policy','target_date','as_of','actual_return','quantile_return','var_return','violation','outer_fold'}
    if not required.issubset(df.columns):
        raise ValueError('Missing prediction columns')
    if set(df.policy)!=set(POLICIES) or len(df)!=expected_n*4:
        raise ValueError('Incorrect policy count')
    if set(metrics.policy)!=set(POLICIES) or len(metrics)!=4:
        raise ValueError('Incorrect metric rows')
    reference=None
    summary={}
    for policy in POLICIES:
        part=df[df.policy==policy].sort_values('target_date').reset_index(drop=True)
        if len(part)!=expected_n or part.target_date.duplicated().any():
            raise ValueError('Missing or duplicate target')
        dates=part.target_date.astype(str).to_numpy()
        asof=part.as_of.astype(str).to_numpy()
        actual=part.actual_return.to_numpy(float)
        quantile=part.quantile_return.to_numpy(float)
        var=part.var_return.to_numpy(float)
        if not (np.isfinite(np.column_stack([actual,quantile,var])).all() and
                np.all(asof < dates) and np.all(dates[:-1] < dates[1:])):
            raise ValueError('Invalid dates or nonfinite prediction')
        if not np.allclose(var,np.maximum(0,-quantile),atol=1e-12,rtol=0):
            raise ValueError('VaR sign inconsistency')
        violations=actual<quantile
        if not np.array_equal(_bool_flags(part.violation),violations):
            raise ValueError('Exception flags wrong')
        if expected_n == 600 and not np.array_equal(part.outer_fold.to_numpy(int),1+np.arange(600)//200):
            raise ValueError('Wrong chronological outer folds')
        if reference is None:
            reference=(dates,asof,actual)
        else:
            if not (np.array_equal(dates,reference[0]) and np.array_equal(asof,reference[1]) and
                    np.allclose(actual,reference[2],atol=1e-12,rtol=0)):
                raise ValueError('Policies evaluated on unequal target dates or returns')
        mean=float(pinball(actual,quantile).mean())
        row=metrics.loc[metrics.policy==policy].iloc[0]
        for field,computed in (('n',expected_n),('violations',int(violations.sum())),
                                ('violation_rate',float(violations.mean())),('mean_pinball',mean)):
            if not np.isclose(float(row[field]),float(computed),atol=1e-12,rtol=0):
                raise ValueError(f'Published {policy} {field} inconsistent')
        summary[policy]={'n':len(part),'mean_pinball':mean,
                         'violations':int(violations.sum()),'violation_rate':float(violations.mean())}
    return summary


def decision_gate(paired:pd.DataFrame) -> dict:
    required={'alternative','block_ci_95_low','block_ci_95_high','ci_includes_zero'}
    if not required.issubset(paired.columns) or len(paired)!=3:
        raise ValueError('Incomplete paired bootstrap rows')
    if set(paired.alternative)!=set(POLICIES[1:]):
        raise ValueError('Unexpected policies')
    for row in paired.itertuples():
        lo=float(row.block_ci_95_low);hi=float(row.block_ci_95_high)
        if not (np.isfinite([lo,hi]).all() and lo <= hi and
                bool(row.ci_includes_zero)==(lo<=0<=hi)):
            raise ValueError('Inconsistent CI')
    if not all(paired.ci_includes_zero.astype(bool)):
        return {'status':'REVIEW_REQUIRED','candidate_promoted':False,
                'reason':'Some descriptive intervals exclude zero; manual review and fresh evaluation required'}
    return {'status':'NO_CONFIRMED_IMPROVEMENT','candidate_promoted':False,
            'reference_model':'gbv2_12f_c1_a',
            'reason':'All descriptive paired block-bootstrap intervals include zero; original inspected history is not a pristine test'}


def standalone_html(title:str, ranking:pd.DataFrame, metrics:pd.DataFrame, gate:dict,
                    images:list[Path]) -> str:
    def table(df):
        return df.to_html(index=False,border=0,classes='data-table',float_format=lambda x:f'{x:.6f}',escape=True)
    items=[]
    for p in images:
        if not p.is_file() or p.suffix.lower()!='.png':
            raise ValueError('Missing release figure')
        encoded=base64.b64encode(p.read_bytes()).decode('ascii')
        items.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(p.stem)}"/><figcaption>{html.escape(p.stem)}</figcaption></figure>')
    return ('<!doctype html><html lang="vi"><head><meta charset="utf-8"/>'
            '<meta name="viewport" content="width=device-width,initial-scale=1"/>'
            f'<title>{html.escape(title)}</title><style>'
            'body{font-family:system-ui,Arial,sans-serif;margin:2rem auto;max-width:1080px;padding:0 1rem;color:#243146;background:#f8fafc}'
            'h1,h2{color:#172a44}section{background:white;border:1px solid #e2e8f0;border-radius:12px;margin:1rem 0;padding:1rem 1.5rem}'
            '.data-table{width:100%;border-collapse:collapse;font-size:.88rem}td,th{text-align:left;padding:.5rem;border-bottom:1px solid #e5e7eb}'
            'figure{margin:1.5rem 0}img{max-width:100%;height:auto}figcaption{color:#64748b;font-size:.8rem}'
            '.note{color:#9a3412}p{line-height:1.6}</style></head><body>'
            f'<h1>{html.escape(title)}</h1><p class="note"><b>EXPLORATORY DEVELOPMENT ONLY.</b> '
            'No new untouched holdout was evaluated, and no production model is promoted.</p>'
            '<section><h2>Feature-count CV</h2>'+table(ranking)+'</section>'
            '<section><h2>Paired walk-forward policies (600 development targets each)</h2>'+table(metrics)+'</section>'
            f'<section><h2>Model decision</h2><p>{html.escape(gate["status"])} â€” {html.escape(gate["reason"])}</p></section>'
            '<section><h2>Stability plots</h2>'+''.join(items)+'</section>'
            '<section><h2>Research limitations</h2><p>Historical development folds were previously inspected. '
            'Results must not be presented as independent forward validation. Coverage tests not rejecting a hypothesis do not prove calibration. '
            'The original Phase 2 468-session holdout is intentionally not used for Phase 3 tuning.</p></section>'
            '</body></html>')
