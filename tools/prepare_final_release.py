#!/usr/bin/env python3
"""Validate the frozen C1 research package and create final submission docs.

Not a model-fitting or optimization script. Idempotently annotates README and
creates docs/submission/FINAL_SUBMISSION_2026_11_19.md + machine-readable audit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

SOURCE_BRANCH = 'submission/c1-frozen-20261119'
C1_COMMIT = 'a21a8556413d0c0c40fe03b3b6597389a5abf20d'
FROZEN_SOURCE_SHA = '14ac5dd4ce7dd6090420b9cee06c8ac40935c54691b4daf95d063b71a075eccc'
REFERENCE = 0.0016369612941317788
C1_PATH = 'results/final_C1_frozen_2026-11-19'
REQUIRED = [
    'configs/c1_final_frozen.json',
    'scripts/final_c1/run_final_c1.py',
    'tests/test_final_c1_frozen.py',
    '.github/workflows/c1_frozen_reproduction.yml',
    'results/phase2/a/features_2026-11-06.csv',
    'results/phase2/a/training_2026-11-06/holdout_predictions.csv',
    f'{C1_PATH}/01_C1_VaR_with_Breaches.png',
    f'{C1_PATH}/02_C1_Coverage.png',
    f'{C1_PATH}/03_C1_Training_Validation_Loss.png',
    f'{C1_PATH}/FINAL_C1_BACKTEST.json',
    f'{C1_PATH}/FINAL_C1_MODEL.joblib',
    f'{C1_PATH}/FINAL_C1_HOLDOUT.csv',
    f'{C1_PATH}/FINAL_C1_CV_METRICS.csv',
    f'{C1_PATH}/ARTIFACT_SHA256.json',
    'app/app.py',
    'requirements.txt',
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eol_equivalent(path: Path, expected: str) -> bool:
    """Only accept exact byte identity or an LF/CRLF-only checkout conversion."""
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()==expected:
        return True
    if path.suffix.lower() not in {'.json','.csv','.md','.yml','.yaml','.py','.txt'}:
        return False
    normalized=raw.replace(b'\r\n',b'\n')
    return (hashlib.sha256(normalized).hexdigest()==expected or
            hashlib.sha256(normalized.replace(b'\n',b'\r\n')).hexdigest()==expected)


def portable_sha(path: Path) -> str:
    """Portable manifest digest: LF-only for text, raw bytes for binary."""
    raw=path.read_bytes()
    if path.suffix.lower() in {'.json','.csv','.md','.yml','.yaml','.py','.txt'}:
        raw=raw.replace(b'\r\n',b'\n')
    return hashlib.sha256(raw).hexdigest()


def check(root: Path):
    missing = [p for p in REQUIRED if not (root / p).is_file()]
    if missing:
        raise RuntimeError(f'Missing release files: {missing}')
    cfg = json.loads((root / 'configs/c1_final_frozen.json').read_text(encoding='utf-8'))
    result = json.loads((root / C1_PATH / 'FINAL_C1_BACKTEST.json').read_text(encoding='utf-8'))
    if cfg['id'] != 'C1_12F_FROZEN_PHASE2_2026_11_06':
        raise RuntimeError('Wrong final model')
    if cfg['snapshot_cutoff'] != '2026-11-06':
        raise RuntimeError('Wrong snapshot cutoff')
    input_path = root / cfg['input_features_csv']
    actual_sha = sha(input_path)
    if not eol_equivalent(input_path, FROZEN_SOURCE_SHA):
        raise RuntimeError(
            'Real source data differs beyond line endings; STOP. '
            f'expected_sha256={FROZEN_SOURCE_SHA}, actual_sha256={actual_sha}'
        )
    if actual_sha != FROZEN_SOURCE_SHA:
        print('INPUT CHECK: LF/CRLF-ONLY DIFFERENCE VERIFIED; canonical frozen SHA256 retained.')
    else:
        print('INPUT CHECK: EXACT RAW SHA256 PASS')
    if cfg['input_features_sha256'] != FROZEN_SOURCE_SHA or result['input_features_sha256'] != FROZEN_SOURCE_SHA:
        raise RuntimeError('Data SHA mismatch between config, output and source')
    if cfg['feature_names'] != result['feature_names'] or len(cfg['feature_names']) != 12:
        raise RuntimeError('Final features mismatch')
    if cfg['canonical_reference_holdout_n'] != 468 or cfg['canonical_reference_holdout_breaches'] != 25:
        raise RuntimeError('Unexpected canonical reference')
    if result['holdout']['n'] != 468 or result['holdout']['breaches'] != 25:
        raise RuntimeError('Unexpected holdout length/violations')
    if abs(result['holdout']['pinball_loss'] - REFERENCE) > 1e-12:
        raise RuntimeError('Final C1 holdout Pinball Loss drift')
    if abs(result['cv_mean_pinball'] - cfg['canonical_reference_cv_pinball']) > 1e-12:
        raise RuntimeError('Final CV metric drift')
    if not result['reference_reproduction'].get('reproduced_reference_predictions'):
        raise RuntimeError('Canonical predictions were not reproduced')
    return cfg, result


def make_docs(root: Path, cfg: dict, result: dict, commit: str, main_merge_sha: str):
    folder=root/'docs'/'submission'
    folder.mkdir(parents=True,exist_ok=True)
    rate=result['holdout']['breach_rate']*100
    data_digest = FROZEN_SOURCE_SHA
    doc=f'''# Final Code Submission — Portfolio VaR Risk System

**Submission snapshot:** 2026-11-19  
**Release source branch:** `{SOURCE_BRANCH}`  
**Frozen C1 reference commit:** `{C1_COMMIT}`  
**Release base / main integration commit:** `{main_merge_sha}`  
**Prepared from repository commit:** `{commit}`

## 1. Scope of the submission

One-day 95% Value at Risk (VaR) for an equal-weight portfolio of HPG, FPT and MWG. The production/research system includes data validation, VaR estimation, backtesting, dashboard presentation and frozen research artifacts.

**Final highlighted risk model: C1 Gradient Boosting Quantile Regression (12 features).** Historical Simulation, EWMA, and earlier G04 remain as **historical comparators**. The prior Phase 1 G04 research contract is preserved as historical evidence; it is not replaced or silently rewritten.

The later experiments involving 7/30-session features, calendar seasonality, volatility normalization or recency weighting are **research explorations, not the selected final model**. Those experimental branches must not be presented as a confirmed improvement.

## 2. Frozen C1 specification

| Field | Value |
|---|---|
| Horizon | 1 trading session |
| Confidence | 95% (5% lower-tail quantile) |
| Source data cutoff | 2026-11-06 |
| Portfolio | HPG/FPT/MWG, equal weight |
| Estimator | GradientBoostingRegressor(loss=quantile) |
| Features | 12, using historical returns/risk statistics |
| Estimators | 150 |
| Learning rate | 0.03 |
| Max depth | 2 |
| Min leaf | 10 |
| Development observations | 1,179 |
| Retrospective holdout observations | 468 |
| Input SHA-256 | `{data_digest}` |

## 3. Reproduced evidence

| Metric | Result |
|---|---:|
| CV mean Pinball Loss | {result['cv_mean_pinball']:.12f} |
| Holdout Pinball Loss | {result['holdout']['pinball_loss']:.12f} |
| VaR breaches | {result['holdout']['breaches']} / {result['holdout']['n']} |
| Breach rate | {rate:.4f}% |
| Kupiec p-value | {result['holdout']['kupiec_p']:.6f} |
| Christoffersen independence p-value | {result['holdout']['christoffersen_independence_p']:.6f} |

The canonical C1 predictions were reproduced byte-for-byte at the numeric-value level, with zero maximum absolute quantile difference in the recorded validation. **A VaR quantile is not a point prediction of next-day actual returns.** A breach occurs when the realized portfolio return falls below the forecasted 5% quantile.

## 4. Important files

- `app/app.py`: Streamlit dashboard.
- `configs/c1_final_frozen.json`: immutable C1 model/data contract.
- `scripts/final_c1/run_final_c1.py`: reproducibility runner.
- `results/final_C1_frozen_2026-11-19/01_C1_VaR_with_Breaches.png`: VaR chart with 25 breach markers.
- `results/final_C1_frozen_2026-11-19/02_C1_Coverage.png`: coverage diagnostics.
- `results/final_C1_frozen_2026-11-19/FINAL_C1_BACKTEST.json`: risk statistics.
- `results/final_C1_frozen_2026-11-19/FINAL_C1_HOLDOUT.csv`: forecast and observed returns.
- `tests/test_final_c1_frozen.py`: frozen-model verification tests.

## 5. Reproduce and review

Run from the repository root after installing `requirements.txt`:

```bash
python -m pytest -q
python scripts/final_c1/run_final_c1.py --root . --verify-only
python -m streamlit run app/app.py
```

The final submission process must also verify the branch from a **fresh GitHub clone**, not only the local working directory.

## 6. Scientific boundaries

- The source snapshot is from **November 6, 2026**, not current live market data.
- The C1 holdout is historical and overlaps data inspected during Phase 1; it must **not** be called pristine or never-seen.
- Kupiec and independence p-values above 0.05 indicate no rejection of the tested hypotheses; they are **not proof** of perfect VaR quality.
- Older Phase 1 documentation and frozen research contracts remain available and are historical context rather than the single C1 result being highlighted in the submission.
- Do not change any locked C1 metric, input hash or reference prediction merely to improve a graph.

## 7. Closing conditions

Only describe the **entire repository** as finalized after full test-suite PASS, C1 `--verify-only` PASS, Streamlit smoke PASS, a clean fresh GitHub clone replay and final GitHub branch/PR integration. Keep the verified release commit SHA and submit that exact ref or tag.
'''
    (folder/'FINAL_SUBMISSION_2026_11_19.md').write_text(doc,encoding='utf-8',newline='\n')
    README = root/'README.md'
    before=README.read_text(encoding='utf-8')
    lead='''<!-- FINAL_SUBMISSION_2026_11_19 -->
> **Final submission (2026-11-19):** The selected final risk model is **C1 Gradient Boosting Quantile Regression, 12 features**, using the verified KBS data snapshot through **2026-11-06**. Its 468-session retrospective holdout has **25 VaR breaches (5.34%)** and **Pinball Loss 0.001636961294**. Reproduction code, data fingerprint, backtesting and figures: [`docs/submission/FINAL_SUBMISSION_2026_11_19.md`](docs/submission/FINAL_SUBMISSION_2026_11_19.md).
>
> The earlier G04/HS/EWMA frozen evaluation documented below remains the **historical Phase 1 contract**. It is not the C1 final result. Data are **not live**, and the retrospective holdout is **not pristine**.

'''
    if '<!-- FINAL_SUBMISSION_2026_11_19 -->' not in before:
        README.write_text(lead+before,encoding='utf-8',newline='\n')
    manifest={
       'schema':'portfolio-var-final-submission-v1',
       'date':'2026-11-19',
       'selected_model':cfg['id'],
       'source_branch':SOURCE_BRANCH,
       'c1_frozen_commit':C1_COMMIT,
       'main_merge_commit':main_merge_sha,
       'input_sha256':data_digest,
       'input_checkout_raw_sha256':sha(root/cfg['input_features_csv']),
       'input_line_endings_equivalent':eol_equivalent(root/cfg['input_features_csv'],data_digest),
       'artifact_hash_policy':'SHA256 of LF-normalized text, raw bytes for binary',
       'holdout_pinball':result['holdout']['pinball_loss'],
       'holdout_breaches':result['holdout']['breaches'],
       'holdout_n':result['holdout']['n'],
       'artifact_sha256':{p:portable_sha(root/p) for p in REQUIRED},
    }
    (folder/'FINAL_SUBMISSION_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    return folder


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path.cwd())
    ap.add_argument('--commit',default='unknown')
    ap.add_argument('--main-merge-sha',default='none')
    args=ap.parse_args()
    root=args.root.resolve()
    cfg, report=check(root)
    path=make_docs(root,cfg,report,args.commit,args.main_merge_sha)
    print('FINAL C1 ARTIFACT AUDIT: PASS')
    print('RELEASE DOCUMENTS READY:',path)
    print('DATA SHA256:',FROZEN_SOURCE_SHA)
    print('HOLDOUT:',report['holdout']['breaches'],'/',report['holdout']['n'])

if __name__=='__main__':main()
