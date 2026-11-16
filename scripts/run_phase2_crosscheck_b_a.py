"""Reconstruct B's frozen 04-Nov baseline forecasts from A's KBS snapshot.

No B or A prediction file is modified. The A input must include the B cutoff.
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

from src.gb_return_features import build_return_features
from src.phase2.crosscheck_b_a import (
    audited_frame, verify_target_alignment, historical_quantiles,
    ewma_quantiles_independent, verify_quantiles, b_five_expanding_indices,
)

ROOT = Path(__file__).resolve().parents[1]
CUTOFF_B = '2026-11-04'
EXPECTED_B_HASH = 'f85bc33ce54d449e393cd81878e9de8627de88d0c309e7f821568b7177692f53'
ALPHA = .05


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_b(path, name, ref):
    raw = pd.read_csv(path, dtype={'as_of': str, 'target_date': str}, float_precision='round_trip')
    frame = audited_frame(raw, name)
    if frame['target_date'].max().date().isoformat() != CUTOFF_B:
        raise ValueError('B last prediction does not match frozen cutoff')
    verify_target_alignment(frame, ref)
    return frame


def compare_gb(b, ref, metadata):
    import sklearn
    from sklearn.ensemble import GradientBoostingRegressor

    features = build_return_features(ref).dropna()
    if features.columns.tolist() != metadata['features']:
        raise ValueError('B seven-feature schema does not match frozen features')
    if len(features) != int(metadata['valid_feature_rows']):
        raise ValueError('B feature count does not match A reconstruction')
    dates = features.index
    params = metadata['parameters']
    if params.get('alpha') != ALPHA or params.get('loss') != 'quantile':
        raise ValueError('Unexpected B GB objective')
    predictions = []
    for i, (train, valid) in enumerate(b_five_expanding_indices(len(features)), 1):
        fold = f'development_fold_{i}'
        got = b.loc[b['fold'] == fold]
        val_dates = dates[valid]
        if len(got) != len(valid) or not np.array_equal(
            got['target_date'].to_numpy(), val_dates.to_numpy()
        ):
            raise ValueError(f'B fold {i}: target dates do not match independent split')
        expected_record = metadata['folds'][i - 1]
        if (int(expected_record['n_train']) != len(train)
                or int(expected_record['n_val']) != len(valid)
                or expected_record['train_end'] != dates[train[-1]].date().isoformat()
                or expected_record['val_start'] != dates[valid[0]].date().isoformat()):
            raise ValueError(f'B fold {i}: published metadata changed')
        model = GradientBoostingRegressor(**params)
        model.fit(features.iloc[train], ref.loc[dates[train]].to_numpy(float))
        q = model.predict(features.iloc[valid])
        stats = verify_quantiles(got, q, tol=2e-8)
        predictions.append({'fold': fold, **stats})
        print(f'GB FOLD {i}: {len(valid)} TARGETS, MAX ABS DIFF={stats["max_abs_difference"]:.10g} PASS', flush=True)
    return {
        'n': int(len(b)),
        'max_abs_difference': max(v['max_abs_difference'] for v in predictions),
        'mean_abs_difference': sum(v['mean_abs_difference'] * v['n'] for v in predictions) / len(b),
        'per_fold': predictions,
        'reproducibility_environment': f'scikit-learn {sklearn.__version__}; Python {platform.python_version()}',
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    out = args.out_dir.resolve()
    out.relative_to(ROOT)
    if out.exists():
        raise FileExistsError('A5 output exists; never overwrite')

    a_base = args.data_root / 'data/phase2'
    a_csv = a_base / 'portfolio_returns.csv'
    a_manifest = json.loads((a_base / 'latest_manifest.json').read_text(encoding='utf-8'))
    a_hash = sha(a_csv)
    if a_manifest['source'] != 'KBS' or a_hash != a_manifest['portfolio_sha256']:
        raise ValueError('Independent A KBS manifest/hash mismatch')
    if a_manifest['cutoff'] < CUTOFF_B:
        raise ValueError('A snapshot cannot cover B published cutoff')

    a_full = pd.read_csv(a_csv, dtype={'date': str}, float_precision='round_trip')
    if a_full['date'].duplicated().any() or not a_full['date'].is_monotonic_increasing:
        raise ValueError('A snapshot duplicate/unsorted dates')
    b_period = a_full.loc[a_full['date'].le(CUTOFF_B)].copy()
    if len(b_period) != 1705 or b_period['date'].iat[-1] != CUTOFF_B:
        raise ValueError('A data cannot replicate B date span/row count')
    ref = pd.Series(b_period['portfolio_simple_return'].to_numpy(float),
                    index=pd.to_datetime(b_period['date']),
                    name='portfolio_simple_return')

    b_dir = ROOT / 'results/phase2/b'
    required = {
        'hs': ('historical_w250', 'hs.csv', 'hs_metadata.json'),
        'ewma': ('ewma_d094', 'ewma.csv', 'ewma_metadata.json'),
        'gb': ('gb_G04', 'gb_baseline.csv', 'gb_baseline_metadata.json'),
    }
    data = {}
    metas = {}
    for key, (model, filename, metafile) in required.items():
        metadata = json.loads((b_dir / metafile).read_text(encoding='utf-8'))
        if metadata['cutoff'] != CUTOFF_B or metadata['input_sha256'] != EXPECTED_B_HASH:
            raise ValueError(f'{key}: B input fingerprint inconsistent')
        frame = load_b(b_dir / filename, model, ref)
        if len(frame) != metadata['prediction_rows']:
            raise ValueError(f'{key}: B output row count differs from metadata')
        data[key] = frame
        metas[key] = metadata
    print('B CSV SCHEMA/AS_OF/ACTUAL: PASS', flush=True)

    base_dates = ref.index[250:]
    for key in ('hs', 'ewma'):
        if not np.array_equal(data[key]['target_date'].to_numpy(), base_dates.to_numpy()):
            raise ValueError(f'{key}: warmup target dates changed')
    r = ref.to_numpy(float)
    hs_stats = verify_quantiles(data['hs'], historical_quantiles(r), tol=2e-12)
    ewma_stats = verify_quantiles(data['ewma'], ewma_quantiles_independent(r), tol=2e-12)
    print('HISTORICAL 250/ALPHA 0.05 INDEPENDENT REPRODUCE: PASS', flush=True)
    print('EWMA 0.94/ALPHA 0.05 INDEPENDENT REPRODUCE: PASS', flush=True)
    gb_stats = compare_gb(data['gb'], ref, metas['gb'])
    print('GB G04 5-FOLD INDEPENDENT REPRODUCE: PASS', flush=True)

    frame_names = {'hs': 'Historical Simulation w250',
                   'ewma': 'EWMA(0.94)', 'gb': 'GB G04 7 features'}
    report = []
    for k, stats in [('hs', hs_stats), ('ewma', ewma_stats), ('gb', gb_stats)]:
        frame = data[k]
        q = frame['quantile_return'].to_numpy(float)
        y = frame['actual_return'].to_numpy(float)
        pinball = np.where(y < q, .95 * (q-y), .05 * (y-q)).mean()
        report.append({
            'model': frame_names[k], 'prediction_rows': len(frame),
            'first_target': frame['target_date'].iat[0].date().isoformat(),
            'last_target': frame['target_date'].iat[-1].date().isoformat(),
            'exception_count': int((y < q).sum()),
            'violation_rate': float((y < q).mean()),
            'mean_pinball_loss': float(pinball),
            'max_abs_reproduction_difference': stats['max_abs_difference'],
            'status': 'TECHNICAL_RECONSTRUCTION_PASS',
        })
    tbl = pd.DataFrame(report)

    # A has chosen on pre-Phase1-evaluation CV; B used folds extending into that period.
    b_used_phase1_eval_in_cv = bool(
        (data['gb']['target_date'] >= pd.Timestamp('2024-12-18')).any()
    )
    if not b_used_phase1_eval_in_cv:
        raise ValueError('Unexpected B CV horizon; protocol has changed')

    report_meta = {
        'step': 'A5_independent_B_baseline_reconstruction',
        'data_ownership': 'A KBS 2026-11-06 snapshot is read-only; B committed outputs are read-only',
        'a_source_sha256': a_hash,
        'b_published_source_sha256': EXPECTED_B_HASH,
        'b_original_snapshot_bytes_available': False,
        'b_snapshot_hash_verified_against_original_bytes': False,
        'b_actual_returns_match_a_on_all_forecast_dates': True,
        'b_baselines_reproduced': ['hs', 'ewma', 'gb_G04'],
        'gb_detailed_results': gb_stats['per_fold'],
        'protocol_flags': {
            'b_gb_cv_extends_into_historically_published_phase1_final_evaluation': b_used_phase1_eval_in_cv,
            'a_and_b_gb_forecasts_directly_comparable': False,
            'reason': 'A and B use different cutoffs and folds; B GB CV includes historically published Phase 1 evaluation dates. Do not claim common holdout.',
            'b_gbv2_retrain_available': False,
            'b_gb_cv_development_only_label_requires_qualification': True,
        },
        'research_conclusion': 'TECHNICAL BASELINE AUDIT ONLY; not a joint independent evaluation or B reproduction of A GBv2',
        'git_commit_before_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    }
    out.mkdir(parents=True, exist_ok=False)
    csv_out = out / 'b_baseline_audit.csv'
    tbl.to_csv(csv_out, index=False, float_format='%.17g')
    report_meta['audit_csv_sha256'] = sha(csv_out)
    (out / 'audit_metadata.json').write_text(json.dumps(report_meta, indent=2), encoding='utf-8')
    print('\nINDEPENDENT B BASELINE AUDIT:')
    print(tbl.to_string(index=False))
    print('\nB CV EXTENDS INTO PHASE 1 HISTORICAL EVALUATION:', b_used_phase1_eval_in_cv)
    print('A AND B GB HOLDOUTS DIRECTLY COMPARABLE: NO')
    print('B ORIGINAL 2026-11-04 SOURCE BYTES HASH VERIFIED: NO (not provided)')
    print('OUTPUT:', out)
    print('STEP A5 TECHNICAL BASELINES PASS')

if __name__ == '__main__':
    main()
