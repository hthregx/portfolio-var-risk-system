# Day 24 — Release Packaging + QA Checklist — Member B

## Release Scope

- [x] Working branch is `release/day24-reproducibility-b`.
- [x] Day 24 package contains only Member B release-reproducibility and cross-review work.
- [x] No frozen model contract is modified by Member B.
- [x] No canonical result artifact is rewritten to manufacture a PASS.
- [x] A-owned implementation, tests, frozen artifacts, and historical metadata remain unchanged.

## Required Release Artifacts

- [x] `configs/model_freeze.yaml` is present.
- [x] `configs/final_evaluation.yaml` is present.
- [x] `results/final_predictions.csv` is present.
- [x] `results/final_metrics.csv` is present.
- [x] `results/final_metric_comparison.csv` is present.
- [x] `results/final_pairwise_diagnostics.csv` is present.
- [x] `results/final_pairwise_summary.csv` is present.
- [x] `results/model_freeze_validation_a.csv` is present.
- [x] `results/release_reproducibility_validation_b.csv` is present.
- [x] `results/release_manifest_b.json` is present.
- [x] `docs/day24-reproducibility-b.md` is present.
- [x] `docs/day24-cross-review-a-by-b.md` is present.

## Member B Validation

Run:

```bash
python scripts/audit_release_reproducibility_b.py
```

Observed Day 24 result:

```text
B1: 22/22 PASS
B2: 61/61 PASS
B1_FROZEN_MODEL_SPECIFICATION_PASS
B2_CROSS_LAYER_CONTRACT_PASS
B3_ARTIFACT_PROVENANCE_PASS
EWMA 0.94 retained for continuity; 0.90 had stronger sensitivity evidence.
```

Release checks:

- [x] B1 frozen model specification audit: `22/22 PASS`.
- [x] B2 cross-layer model contract validator: `61/61 PASS`.
- [x] B3 artifact provenance audit: `PASS`.
- [x] Frozen Historical Simulation contract matches implementation.
- [x] Frozen EWMA contract matches implementation.
- [x] Frozen Gradient Boosting G04 contract matches implementation.
- [x] Canonical G04 does not consume market-feature additions.
- [x] Prediction and metric schemas satisfy the release contract.
- [x] Strict violation and VaR sign conventions are validated.
- [x] Artifact provenance manifest is generated.

## Task-Specific Tests

Run:

```bash
python -m pytest tests/test_release_reproducibility_b.py -q
```

Observed:

```text
7 passed
```

- [x] Member B task-specific tests PASS.

## Cross-Review QA

Member B independently reviewed Member A's Day 24 final evidence package.

Review artifact:

`docs/day24-cross-review-a-by-b.md`

A evidence tests:

```bash
python -m pytest tests/test_final_evidence_a.py -q
```

Observed:

```text
10 passed
```

Reviewed A commits include:

- `b71e8e8` — final frozen evidence pack.
- `628dfdf` — final quantitative evidence documentation.
- `0d191d0` — merged Day 24 A evidence package.

Cross-review result:

```text
CROSS_REVIEW_A_BY_B_PASS
```

- [x] Member A Day 24 evidence reviewed by Member B.
- [x] A evidence tests: `10 passed`.
- [x] Frozen evaluation interpretation is consistent with the release contract.
- [x] No universal model winner is incorrectly declared.
- [x] Known provenance limitation is disclosed.
- [x] B6 cross-review evidence is present and reviewable.

## Repository-Level QA

The repository provides an explicit CI-safe surrogate-data mode for environments where the canonical processed dataset is intentionally not version-controlled.

Portable release/CI command:

```bash
MODEL_FREEZE_USE_SURROGATE_DATA=1 python -m pytest -q
```

Observed:

```text
384 passed, 1 skipped
```

There are zero failed tests in this mode.

The single skipped test is the canonical-data-dependent model-freeze test. The repository explicitly supports this behavior through `MODEL_FREEZE_USE_SURROGATE_DATA=1` because canonical processed data is intentionally not version-controlled.

This is not implemented by modifying or disabling an A-owned test. Member B does not change:

- `tests/test_model_freeze_a.py`;
- `scripts/validate_model_freeze_a.py`;
- frozen SHA values;
- canonical prediction artifacts;
- historical metadata.

The local canonical-data SHA/provenance validation remains a separate full-reproduction check when the approved canonical dataset is available.

- [x] Full repository pytest has zero failures in repository-provided CI-safe mode.
- [x] Result: `384 passed, 1 skipped`.
- [x] Expected canonical-data-dependent skip is documented.
- [x] Member B-specific release validator PASS.
- [x] Member B-specific tests PASS.
- [x] Member A evidence tests PASS.
- [x] No A-owned test or validator was modified to obtain the release result.

## Frozen Artifact and Provenance QA

Member B's release manifest records release artifact provenance including path, SHA256, row count and schema where applicable, role, and canonical/derived status.

Known provenance limitation:

> final metadata runtime HEAD predates the later Day21 source commit, while canonical prediction artifact content remained unchanged.

Handling:

- [x] Limitation is explicitly documented.
- [x] Historical commit metadata is not rewritten.
- [x] Full models are not rerun merely to manufacture a newer HEAD.
- [x] Frozen prediction content is not changed.
- [x] Provenance limitation is separated from model-contract drift.

## Packaging Hygiene

- [x] No credentials or secrets are part of the Member B package.
- [x] Canonical private/local processed data is not staged.
- [x] No large temporary outputs are included.
- [x] No stale notebook is required for release execution.
- [x] No model contract has changed after freeze.
- [x] No canonical prediction artifact has been regenerated merely for packaging.
- [x] Expensive full model reproduction is not required for the fast release-validation path.

The following unrelated local notebook modifications must not be staged:

```text
notebooks/05_ewma_evaluation.ipynb
notebooks/07_baseline_error_analysis.ipynb
```

## Staging Policy

Never use:

```bash
git add .
```

Stage only Member B Day 24 files:

```bash
git add docs/day24-cross-review-a-by-b.md
git add docs/day24-release-checklist-b.md
git add docs/day24-reproducibility-b.md
git add results/release_manifest_b.json
git add results/release_reproducibility_validation_b.csv
git add scripts/audit_release_reproducibility_b.py
git add tests/test_release_reproducibility_b.py
```

Do not stage:

```text
notebooks/05_ewma_evaluation.ipynb
notebooks/07_baseline_error_analysis.ipynb
data/processed/portfolio_returns.csv
```

## Pre-Commit Review

Run:

```bash
python scripts/audit_release_reproducibility_b.py
python -m pytest tests/test_release_reproducibility_b.py -q
python -m pytest tests/test_final_evidence_a.py -q
MODEL_FREEZE_USE_SURROGATE_DATA=1 python -m pytest -q

git diff --check
git diff --cached --check
git status --short
git diff --cached --stat
git diff --cached
```

Required release evidence:

```text
B1: 22/22 PASS
B2: 61/61 PASS
B3_ARTIFACT_PROVENANCE_PASS

7 passed
10 passed
384 passed, 1 skipped
```

`git diff --check` and `git diff --cached --check` must produce no whitespace errors.

The staged diff must contain only the seven Member B Day 24 files listed above.

## Release Decision

Member B's Day 24 release package satisfies the B-side reproducibility and QA contract when all of the following hold:

- [x] B1 frozen specification audit: `22/22 PASS`.
- [x] B2 cross-layer contract audit: `61/61 PASS`.
- [x] B3 artifact provenance audit: `PASS`.
- [x] B4 reproducibility runbook is complete.
- [x] B-specific tests: `7 passed`.
- [x] B6 Member A cross-review evidence is present.
- [x] Member A evidence tests: `10 passed`.
- [x] Repository-provided CI-safe full suite: `384 passed, 1 skipped, 0 failed`.
- [x] Canonical data is treated as an external prerequisite rather than committed release content.
- [x] No A-owned code, test, frozen contract, or historical metadata is modified to manufacture PASS.
- [x] No unrelated notebook or private/local dataset is included in the staged package.

**Member B Day 24 release packaging and QA: PASS.**

The expected canonical-data-dependent skip in portable CI does not represent a failed release test. Full canonical-data SHA reproduction remains a separate environment-dependent provenance check requiring the approved untracked canonical dataset.