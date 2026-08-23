# Day 23 — Model Freeze and A-Side Release QA

## Objective

Day 23 freezes the canonical one-day 95% portfolio VaR specifications and
hardens the existing final evaluation for reproducibility and release use.

The A-side scope covers:

- the canonical model-freeze specification;
- automated freeze-contract validation;
- regression tests for configuration drift;
- a persisted validation record;
- Historical Simulation and EWMA model cards;
- reproducibility and release-QA evidence.

Day 23 is a release-hardening milestone. It does not reopen model selection,
introduce new algorithms, add new features, retune parameters, change the
evaluation period, or regenerate canonical final predictions.

## Freeze Scope

The frozen portfolio is the equal-weight portfolio of:

- HPG
- FPT
- MWG

The forecast target is:

`portfolio_simple_return`

The frozen forecasting contract is:

- Confidence level: 95%
- Alpha: 0.05
- Forecast horizon: 1 trading day
- Evaluation start: 2024-12-18
- Evaluation end: 2026-07-28
- Common target dates: 398
- Methods: 3
- Canonical prediction rows: 1,194

A violation is defined strictly as:

`actual_return < quantile_return`

VaR is represented as:

`VaR = max(0, -quantile_return)`

The evaluation period was reserved from the documented validation-based
parameter selection, but it must not be described as a pristine untouched
test set because the broader evaluation sample had been inspected previously.

## Frozen Models

### Historical Simulation

Frozen config ID:

`historical_w250`

Frozen specification:

- Rolling window: 250 observations
- Alpha: 0.05
- Forecast horizon: 1 trading day

Historical Simulation remains the transparent non-parametric baseline.

### EWMA

Frozen config ID:

`ewma_d094`

Frozen specification:

- Decay factor: 0.94
- Updating mode: expanding
- Alpha: 0.05
- Variance initialization: first squared return
- Distribution: Normal
- Mean assumption: zero
- Forecast horizon: 1 trading day

Decay factor 0.90 produced stronger validation evidence, but it was not used
for the canonical final evaluation.

Decay factor 0.94 is frozen because it is the retained canonical baseline
used by the existing final evaluation. This freeze must not be interpreted as
evidence that 0.94 outperformed 0.90 during validation.

### Gradient Boosting Quantile Regression

Frozen config ID:

`gb_G04`

Frozen specification:

- Experiment ID: G04
- Loss: quantile
- Alpha: 0.05
- Estimators: 100
- Learning rate: 0.03
- Maximum depth: 2
- Minimum samples per leaf: 5
- Subsample: 1.0
- Random state: 42
- Training protocol: expanding refit for each target

Canonical G04 uses seven return-history features:

- `return_lag_1`
- `return_lag_2`
- `return_lag_5`
- `rolling_vol_5`
- `rolling_vol_20`
- `rolling_vol_60`
- `drawdown`

The canonical final runner does not use the integrated market-feature builder
for G04.

G04 is frozen as the configuration already used by the canonical final
predictions. This is a reproducibility and release decision, not a claim that
G04 is the confirmed overall winner across all G01-G07 validation
experiments.

## G05-G07 Reconciliation

Integrated B-side tuning artifacts contain:

- G01
- G05
- G06
- G07

Those experiments use a 13-feature specification combining return-history
and market/liquidity features.

Within that B-side validation slice, G05 has the lowest Pinball Loss.

The B-side results are validation evidence only and are not like-for-like
continuations of the seven-feature canonical G04 final-evaluation
specification.

Day 23 therefore does not use those results to retroactively replace,
retune, or regenerate the canonical G04 final predictions.

Model selection is not reopened on freeze day.

## Criterion-Specific Evaluation Facts

Historical Simulation produced:

- Forecasts: 398
- Violations: 27
- Violation rate: 6.7839%
- Pinball Loss: 0.001896464233
- Average VaR: 2.1330%

EWMA produced:

- Forecasts: 398
- Violations: 22
- Violation rate: 5.5276%
- Pinball Loss: 0.001968321253
- Average VaR: 2.5084%

Criterion-specific interpretation:

- EWMA is closest to the nominal 5% violation rate on the frozen evaluation
  sample.
- Historical Simulation has the lowest Average VaR among the three frozen
  methods on the frozen evaluation sample.
- EWMA has the highest aggregate Pinball Loss among the three frozen methods
  on the frozen evaluation sample.
- No overall winner is declared.

A lower Average VaR does not by itself establish better model performance.
Likewise, closeness to the nominal violation frequency does not by itself
establish overall model superiority.

## Canonical Data Freeze

Canonical processed data:

`data/processed/portfolio_returns.csv`

The file contains:

- Rows: 1,637
- Start date: 2020-01-03
- End date: 2026-07-28
- Duplicate dates: none
- Missing values: none

Canonical data SHA256:

`88a6dbac04349e40d84001b06c6a7e876ec660ea9eb87267a22aeb00c165f2fe`

The processed data file is not version-controlled in the audited repository
state, so the Day-23 freeze uses its validated semantic contract together
with its SHA256 digest.

## Automated Freeze Validation

Reusable validator:

`scripts/validate_model_freeze_a.py`

Regression tests:

`tests/test_model_freeze_a.py`

Persisted validation record:

`results/model_freeze_validation_a.csv`

The validator reconciles the frozen specification against:

- canonical processed data;
- `configs/final_evaluation.yaml`;
- canonical final predictions;
- configuration IDs;
- Historical Simulation parameters;
- EWMA parameters and the 0.90 validation caveat;
- G04 hyperparameters;
- G04 feature specification;
- canonical runner implementation wiring;
- canonical artifact hashes;
- integrated G05-G07 validation evidence.

Day-23 validation result:

- Checks: 84
- Passed: 84
- Failed: 0

Result:

`MODEL_FREEZE_VALIDATION_PASS`

The persisted validation artifact was independently reconciled with a fresh
in-memory validation run.

Its output was also reproduced through an external temporary output path with
an identical SHA256 digest.

Validation artifact SHA256:

`e55fc4c4d885ef21d0605ce0a979dafb2e59e92ed99a0bc7f3d466c28f6cc82e`

## Regression and Repository QA

Targeted Day-23 regression tests:

`7 passed`

Full repository test suite:

`330 passed in 28.16s`

No tracked canonical file was modified during the A-side release-QA
preflight.

The validated branch and integration-base HEAD were:

- Branch: `release/model-freeze-a`
- Integration base: `e17ae67b0bae5f3a9e30fedb0ea5a0eb34c40394`

## A-Side Artifact Integrity

The A-side artifacts validated before this release note were:

| Artifact | SHA256 |
| --- | --- |
| `configs/model_freeze.yaml` | `839317c17dfd3ac3657519105da67c5ffb1c36681822b9c345e26243d3f9be15` |
| `docs/model-cards/historical-simulation.md` | `23b61c9b287958825c21e8360ea57146e1bf6b3f03c1ff34a2012078c865ed7f` |
| `docs/model-cards/ewma.md` | `2b8e823a3699358a8c8bfcfbaa9fba0d0de450da3d9786885e5d97f72b0998ec` |
| `results/model_freeze_validation_a.csv` | `e55fc4c4d885ef21d0605ce0a979dafb2e59e92ed99a0bc7f3d466c28f6cc82e` |
| `scripts/validate_model_freeze_a.py` | `bd10b69b8e5eb444d85af2cbe31cd3656231c6042c1063622971402ff9455bd0` |
| `tests/test_model_freeze_a.py` | `e0e575afdb739655c005173cc0d858d62c9131e4c45f036aa707b8de10f02d38` |

The SHA256 of this release note is intentionally not self-recorded inside the
document because doing so would create a recursive hash dependency.

## Reproducibility Sequence

The frozen contract can be checked with:

```powershell
python scripts/validate_model_freeze_a.py
python -m pytest -q tests/test_model_freeze_a.py
python -m pytest -q
```

Expected Day-23 A-side results are:

- Freeze validation: 84 PASS, 0 FAIL
- Targeted regression tests: 7 passed
- Full suite: 330 passed

The persisted validation CSV can be regenerated explicitly with:

```powershell
python scripts/validate_model_freeze_a.py --output results/model_freeze_validation_a.csv
```

Regeneration should only be performed when intentionally reproducing the
validated contract. It must not be used to hide or overwrite an unexplained
freeze-contract failure.

## Release Interpretation

Day-23 model freeze means that subsequent release work should focus on:

- reproducibility;
- integration;
- documentation;
- testing;
- packaging;
- explanation;
- defect correction that does not alter the frozen modeling contract.

It does not authorize:

- new algorithms;
- new feature families;
- parameter retuning;
- post-freeze model reselection;
- evaluation-period changes;
- retroactive rewriting of canonical final predictions.

## A-Side Release QA Status

A-side model-freeze specification: **PASS**

Automated freeze validation: **PASS**

Historical Simulation model card: **PASS**

EWMA model card: **PASS**

Targeted Day-23 regression tests: **PASS**

Full repository test suite: **PASS**

Tracked canonical-file safety check: **PASS**

Member A's planned B-side cross-review is **DEFERRED** because no Day-23
B-side integration changes were available on `feature/eda-analysis` at the
A6 readiness check.
