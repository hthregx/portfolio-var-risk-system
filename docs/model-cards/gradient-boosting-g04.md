# Gradient Boosting G04 Model Card

## Purpose

G04 is the frozen Gradient Boosting model used for the final 5% VaR
evaluation. It forecasts the lower-tail portfolio return quantile for
one-step-ahead risk estimation.

## Frozen Specification

- Config: G04
- Alpha: 0.05
- Estimators: 100
- Learning rate: 0.03
- Maximum depth: 2
- Minimum samples per leaf: 5
- Subsample: 1.0
- Random seed: 42

This specification is frozen for the v0.6 release.

## Inputs

The model uses historical portfolio simple returns available before the
target return.

No target-date return is used as an input feature.

## Features

The implemented frozen feature set is:

- `return_lag_1`
- `return_lag_2`
- `return_lag_5`
- `rolling_vol_5`
- `rolling_vol_20`
- `rolling_vol_60`
- `drawdown`

Only implemented return-history features are used.

The frozen G04 model does not use:

- volume features;
- price-range features;
- VNINDEX or other market features.

## Training Procedure

G04 uses Gradient Boosting quantile regression at `alpha = 0.05`.

The frozen hyperparameters are those listed in the Frozen Specification.
The release process does not rerun tuning or perform additional model
selection.

## Walk-Forward Forecasting

Evaluation is chronological.

For each target date, features are constructed from information available
before the target return. Forecasting therefore preserves temporal ordering
rather than randomly shuffling evaluation observations.

## Evaluation Contract

The final common evaluation universe contains:

- 398 target dates;
- evaluation start: 2024-12-18;
- evaluation end: 2026-07-28.

A VaR violation is defined strictly as:

`actual_return < quantile_return`

Reported VaR uses:

`VaR = max(0, -quantile_return)`

All three final models use the same target-date universe and the same
actual portfolio return on each target date.

## Results

For G04 on the final evaluation universe:

- Forecasts: 398
- Violations: 24
- Violation rate: 6.03%
- Pinball Loss: 0.0017581309509574873
- Average VaR: 2.33%

G04 has the lowest aggregate Pinball Loss among the three frozen final
models on this evaluation.

Day-22 paired analysis provides additional context:

- GB vs Historical: 151 dates favor GB and 247 favor Historical.
- Mean GB-minus-Historical loss difference is negative.
- The aggregate GB advantage over Historical is therefore
  magnitude-driven/concentrated rather than a majority-of-dates advantage.
- GB vs EWMA: 231 dates favor GB and 167 favor EWMA, indicating a broader
  paired advantage on that comparison.

These results are criterion-specific and do not establish a universal
overall winner.

## Strengths

- Lowest aggregate Pinball Loss among the three frozen final models.
- Chronological walk-forward evaluation.
- Deterministic frozen specification.
- Explicit return-history feature construction.
- Reproducible random seed.

## Limitations

G04 is not established as a global winner across the G01-G07 tuning search.

Only the implemented return-history features are used. The frozen model
does not contain volume, price-range, VNINDEX, or other market features.

The final evaluation should not be described as a pristine untouched test
set. Results should be interpreted as final project evaluation evidence,
not as proof of universal future superiority.

Aggregate metrics can hide date-level differences. In particular, the
paired Historical comparison shows that lower aggregate GB Pinball Loss
does not mean GB performs better on most dates.

## Leakage Controls

The release audit checks that:

- lag 1 uses prior-return information;
- lag 2 and lag 5 use appropriately older observations;
- rolling-volatility features are backward-looking;
- drawdown is constructed from historical information;
- target return is absent from feature inputs;
- future-value perturbations do not alter the target-date feature row;
- evaluation remains chronological;
- shuffle is disabled or absent.

No direct target leakage was detected in the implemented feature
construction.

This statement does not claim that the model is guaranteed leakage-free
under every possible use or future implementation.

## Reproducibility

The frozen model uses:

- Config: G04
- Alpha: 0.05
- Seed: 42

Release audits are provided by:

- `scripts/audit_gb_freeze_b.py`
- `tests/test_gb_freeze_b.py`
- `results/gb_freeze_audit_b.csv`
- `scripts/audit_release_artifacts_b.py`
- `tests/test_release_artifacts_b.py`
- `results/release_artifact_audit_b.csv`

## Intended Use

G04 is intended for project-level comparison and one-step-ahead portfolio
5% VaR analysis under the documented data, feature, and evaluation
contracts.

## Non-Intended Use

The model should not be interpreted as:

- a universally superior VaR model;
- a guarantee of future loss coverage;
- a causal model of market risk;
- a production trading recommendation;
- evidence that omitted market or liquidity features are unnecessary.

## Freeze Status

**FROZEN — v0.6**

No new feature engineering, tuning, or model selection should be introduced
after this freeze without explicitly reopening the model-development
process.