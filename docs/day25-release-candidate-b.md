# Day 25 Release Candidate — Member B

## Objective

Package and verify the frozen release candidate without changing models, predictions, metrics, evaluation semantics, or historical provenance.

## Release Scope

Base:

```text
feature/eda-analysis
de7843f3c68cb0371cd1ab49d7a53e26fd7809d3
```

The release manifest contains only explicitly selected tracked release-critical artifacts.

## Frozen Contracts

```text
Portfolio: equal-weight HPG/FPT/MWG
Horizon: 1 day
Confidence: 0.95
Alpha: 0.05
Historical window: 250
EWMA decay: 0.94
GB config: gb_G04
Prediction rows: 1194
Methods: 3
Forecasts/method: 398
Evaluation: 2024-12-18 → 2026-07-28
Violation: actual_return < quantile_return
VaR: max(0, -quantile_return)
```

Exact G04 features:

```text
return_lag_1
return_lag_2
return_lag_5
rolling_vol_5
rolling_vol_20
rolling_vol_60
drawdown
```

## Artifact Manifest

Contract:

```text
configs/release_candidate_b.yaml
```

Build:

```bash
python scripts/build_release_candidate_b.py
```

Outputs:

```text
results/release_artifact_ledger_b.csv
results/release_candidate_manifest_b.json
```

Ledger schema:

```text
path,role,git_blob_oid,sha256_git_blob,blob_bytes,status
```

Integrity uses canonical Git blob bytes, not working-tree bytes.

## Integrity Verification

Run the builder twice and compare:

```bash
shasum -a 256 results/release_artifact_ledger_b.csv results/release_candidate_manifest_b.json
```

Observed deterministic hashes:

```text
ledger:
88611bee6a928c772ebd14de803a792c7cc1b696b9ced993f9e6420e51e28484

manifest:
17d1d7b24b3e547e82db197ad0ad18031f8b2196c908daf1339bba9a6a9d5df5
```

Both hashes were identical across consecutive builds.

## Clean Environment Procedure

```bash
git clone <repository-url>
cd portfolio-var-risk-system
git checkout <release-commit>
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Private canonical data is not required for the portable release QA gate.

## Evidence Rebuild Procedure

```bash
python scripts/build_final_evidence_a.py
python scripts/build_release_candidate_b.py
python scripts/validate_release_candidate_b.py
```

This rebuilds release evidence only. A full final model rerun is not a Day 25 acceptance requirement.

## Test Procedure

Task-specific QA:

```bash
python -m pytest tests/test_release_candidate_b.py -q
```

Observed:

```text
10 passed
```

Full repository portable QA:

```bash
MODEL_FREEZE_USE_SURROGATE_DATA=1 python -m pytest -q
```

Observed:

```text
394 passed, 1 skipped, 0 failed
```

The surrogate mode is repository-supported portable QA and does not replace a separate canonical-data reproduction workflow.

## Expected Outputs

```text
configs/release_candidate_b.yaml
scripts/build_release_candidate_b.py
results/release_artifact_ledger_b.csv
results/release_candidate_manifest_b.json
scripts/validate_release_candidate_b.py
results/release_smoke_validation_b.csv
docs/day25-release-candidate-b.md
tests/test_release_candidate_b.py
```

Smoke validation observed:

```text
27 PASS
0 FAIL
B3_RELEASE_SMOKE_VALIDATION_PASS
```

## Data Boundary

Do not package:

```text
data/**
.venv/**
.env
.env.*
**/__pycache__/**
.pytest_cache/**
**/.ipynb_checkpoints/**
*.log
logs/**
credentials/**
private/**
```

`data/processed/portfolio_returns.csv` is local/private and is not part of the release manifest.

## Known Limitations

The evaluation is not a pristine, never-inspected test set.

Results are specific to the frozen HPG/FPT/MWG portfolio, evaluation period, and model configurations. Runtime values are not portable benchmarks.

A full final model rerun is not a
Day 25 release acceptance gate.

## Provenance Boundary

Locked integration base:

```text
de7843f3c68cb0371cd1ab49d7a53e26fd7809d3
```

Artifact SHA256 values are computed from canonical Git blob bytes.

The Day 25 builder commit SHA is not embedded in its own manifest to avoid self-referential provenance.

## Failure Recovery

If validation fails:

1. Inspect `results/release_smoke_validation_b.csv`.
2. Check tracked paths, Git blobs, hashes, exclusions, and frozen contracts.
3. Rebuild the manifest and rerun validation.
4. Do not modify frozen outputs or historical evidence merely to force PASS.

## Release Acceptance Criteria

PASS when:

- all manifest artifacts are tracked, unique, resolvable, and outside excluded paths;
- Git-blob SHA256 verification passes;
- manifest generation is deterministic;
- Historical=250, EWMA=.94, GB=`gb_G04` with exact 7 features;
- predictions=1194, 3 methods, 398 forecasts/method;
- Day 24 validation and figures are present;
- smoke validation has 0 FAIL;
- task QA has 0 failures;
- full-repository portable QA has 0 failures;
- non-pristine evaluation limitation is documented;
- private canonical data is excluded;
- full final model rerun is not required for Day 25 acceptance.

Observed Day 25: **10 task tests passed; 394 repository tests passed, 1 skipped, 0 failed; smoke validation 27/27 PASS.**
