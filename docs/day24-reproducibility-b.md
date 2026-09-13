# Day 24 — Release Reproducibility Runbook

## Purpose

This runbook verifies that the frozen VaR release can be audited and reproduced without changing the frozen model contract or canonical results.

It separates FAST RELEASE VALIDATION from EXPENSIVE FULL MODEL REPRODUCTION.

## Repository Branch Contract

Day 24 Member B branch:

`release/day24-reproducibility-b`

Validation must not modify the frozen model specification, historical metadata, or canonical evaluation artifacts.

## Environment Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Required Data

Canonical processed data is required at:

`data/processed/portfolio_returns.csv`

This dataset is intentionally not version-controlled. A fresh clone must obtain the approved canonical data through the project data handoff before full reproduction.

Do not commit or stage the canonical dataset.

## Validation Commands

### FAST RELEASE VALIDATION

Run:

```bash
python scripts/audit_release_reproducibility_b.py
```

Expected result:

```text
B1: 22/22 PASS
B2: 61/61 PASS
B1_FROZEN_MODEL_SPECIFICATION_PASS
B2_CROSS_LAYER_CONTRACT_PASS
B3_ARTIFACT_PROVENANCE_PASS
```

EWMA decay `0.94` is retained for continuity with the frozen contract. It must not be described as the validation winner; `0.90` had stronger sensitivity evidence.

This validator is the normal fast release audit and does not rerun the expensive GB walk-forward.

## Unit Test Commands

Run Member B tests:

```bash
python -m pytest tests/test_release_reproducibility_b.py -q
```

Expected result:

```text
7 passed
```

Full repository gate:

```bash
python -m pytest -q
```

Any failure must be reported. Failures outside Member B scope must not be hidden, bypassed, or rewritten merely to obtain a green release gate.

## Frozen Artifact Checks

The release audit checks the frozen contract and canonical artifacts, including:

`configs/model_freeze.yaml`

`configs/final_evaluation.yaml`

`results/final_run_metadata.json`

`results/final_predictions.csv`

`results/final_metrics.csv`

It also produces:

`results/release_reproducibility_validation_b.csv`

`results/release_manifest_b.json`

The manifest records artifact path, SHA256, row count and schema where applicable, role, and canonical/derived status.

Do not edit frozen artifacts merely to make validation pass.

## Optional Full Model Reproduction

### EXPENSIVE FULL MODEL REPRODUCTION

Full reproduction requires the approved canonical processed dataset.

Run:

```bash
python scripts/run_final_walk_forward.py
```

The Gradient Boosting walk-forward may take approximately 100+ seconds.

Full model reproduction is optional for routine release validation when the frozen artifacts and cross-layer validator are sufficient. Do not rerun the full model solely to manufacture newer Git metadata or make the runtime HEAD appear newer.

## Expected Outputs

Fast validation should produce a PASS for B1, B2, and B3 and refresh:

`results/release_reproducibility_validation_b.csv`

`results/release_manifest_b.json`

Task-specific tests should pass independently.

A full reproduction, when intentionally requested with canonical data available, regenerates the canonical final evaluation outputs according to the frozen contract.

## Known Provenance Limitation

final metadata runtime HEAD predates the later Day21 source commit, while canonical prediction artifact content remained unchanged.

This limitation is documented rather than rewritten. Historical commit metadata must not be changed, and the full model must not be rerun merely to manufacture a newer HEAD.

## Troubleshooting

If validation fails, inspect the reported contract or artifact mismatch before changing anything.

If canonical processed data is unavailable, FAST RELEASE VALIDATION can still audit the frozen release artifacts where supported, but EXPENSIVE FULL MODEL REPRODUCTION requires the approved canonical dataset.

Do not alter frozen model parameters, canonical predictions, historical metadata, or validation rules to force PASS.

Local notebook modifications unrelated to Day 24 are not required for execution and must not be staged as part of Member B's release package.