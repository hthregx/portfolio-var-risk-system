# Leakage and Reproducibility Audit

## Scope

This audit verifies that the Historical Simulation and EWMA VaR
evaluation pipelines respect one-day-ahead forecasting and do not use
future target returns when producing forecasts.

The Day-14 audit covers:

- forecast-origin and target-date alignment;
- training-data inclusion rules;
- Historical Simulation;
- EWMA;
- future-data perturbation;
- cross-method target alignment;
- strict violation semantics;
- dependency locking;
- random-seed policy;
- reproducible pipeline execution;
- basic CI.

## Forecast origin definition

For each one-day-ahead forecast, `forecast_date` is the final date
available when the forecast is produced.

The corresponding `target_date` is the next evaluation date.

Required invariant:

`forecast_date < target_date`

The return observed on `target_date` must not be available to the model
when the forecast is generated.

## Target definition

The target observation is the portfolio return observed on
`target_date`.

Therefore:

- `actual_return` must correspond exactly to `target_date`;
- the target return is used only for evaluation;
- the target observation must not be included in training history;
- each method must have at most one forecast per target date.

## Training inclusion rule

For every forecast, training observations must satisfy:

`training_date <= forecast_date`

and:

`training_date < target_date`

The maximum training date must never reach the target date.

Historical Simulation uses its configured historical rolling window.

EWMA uses historical observations available before the target date.

The common walk-forward layer selects the training history and dates.
It must not reimplement Historical quantile logic or EWMA recursion.

## Historical Simulation audit

Historical Simulation was tested through the common walk-forward
evaluation path.

The tests verify:

- `forecast_date < target_date`;
- training data ends before the target;
- target dates are unique;
- target returns are correctly aligned;
- future observations do not affect earlier forecasts;
- violation classification uses strict `<`.

Result:

`PASS`

## EWMA audit

EWMA was tested through the same common walk-forward path using the
production `ewma_var_forecast` implementation.

The tests verify:

- `forecast_date < target_date`;
- training data ends before the target;
- target dates are unique;
- target returns are correctly aligned;
- future observations do not affect earlier forecasts;
- violation classification uses strict `<`.

The walk-forward layer does not duplicate EWMA recursion.

Result:

`PASS`

## Future perturbation test

The future perturbation test was applied to both Historical Simulation
and EWMA.

Procedure:

1. Produce a forecast for a selected target date.
2. Copy the input dataset.
3. Replace returns strictly after that target date with extreme values.
4. Re-run the forecast.
5. Compare the original and perturbed forecasts for the selected target.

The earlier `quantile_return` and `var` remain unchanged within
numerical tolerance.

Result:

`PASS`

This provides direct evidence that observations occurring after the
target do not influence the tested earlier forecast.

## Target alignment audit

Historical Simulation and EWMA were checked using aligned target dates.

The audit verifies:

- no duplicate `target_date`;
- no missing expected target;
- `actual_return` matches the target observation;
- both methods use the same target return for the same date;
- target-date sequences are consistent between methods.

Result:

`PASS`

## Violation semantics

A violation is defined strictly as:

`actual_return < quantile_return`

Equality is not a violation.

The boundary condition:

`actual_return == quantile_return`

must therefore produce:

`violation == False`

This case is tested explicitly.

Result:

`PASS`

## Random seed policy

Random seed is not applicable to the Historical Simulation and EWMA
baseline stage.

Both methods are deterministic for fixed input data, configuration, and
walk-forward ordering.

This stage does not use:

- stochastic model training;
- random sampling;
- randomized initialization;
- randomized parameter search.

Therefore, an arbitrary random seed is not required for the baseline
stage.

Reproducibility instead depends on fixed data, configuration,
dependencies, preprocessing, and execution order.

If a later model introduces stochastic operations, that stage must
define and record an explicit random seed.

Result:

`N/A — deterministic baseline`

## Dependency lock

Project dependencies are defined in:

`requirements.txt`

The dependency specification contains the packages required by the
current project rather than an unrestricted environment-wide
`pip freeze`.

Pinned dependencies support reproducible installation, testing, and
pipeline execution.

Result:

`PASS`

## Reproducible pipeline

The explicit reproducibility entry point is:

`scripts/run_pipeline.py`

It can be executed with:

`python scripts/run_pipeline.py`

The script orchestrates the existing baseline components rather than
reimplementing Historical or EWMA model logic.

The validated pipeline executes:

1. `scripts/generate_ewma_comparison.py`
2. `scripts/generate_ewma_figures.py`
3. `scripts/audit_ewma_comparison.py`
4. `scripts/generate_ewma_metadata.py`

The pipeline reproduced the canonical baseline results:

- Historical forecasts: 1,387;
- Historical violations: 75;
- EWMA forecasts: 1,387;
- EWMA violations: 73;
- shared exceptions: 56;
- Historical-only exceptions: 19;
- EWMA-only exceptions: 17.

The consistency audit completed successfully.

Observed terminal result:

`B-14.4 RUN PIPELINE PASS`

Result:

`PASS`

## Basic CI

Basic GitHub Actions CI is defined in:

`.github/workflows/tests.yml`

The workflow performs:

1. repository checkout;
2. Python setup;
3. dependency installation;
4. pytest execution.

This provides a clean-environment regression check for committed code.

Result:

`PASS`

## Test evidence

Dedicated leakage and alignment tests:

`10 passed`

Full project regression suite:

`130 passed`

The evidence covers:

- Historical no-look-ahead;
- EWMA no-look-ahead;
- future perturbation;
- target alignment;
- same-target consistency;
- strict violation semantics.

The reproducible pipeline also completed successfully.

## Remaining risks

Remaining risks include:

- future changes to walk-forward slicing;
- future changes to date alignment;
- models bypassing the common runner contract;
- preprocessing using unavailable future information;
- externally generated artifacts outside the current tests;
- future stochastic models without an explicit seed;
- dependency changes not reflected in `requirements.txt`;
- future pipeline stages not included in `scripts/run_pipeline.py`.

The leakage tests and reproducibility checks should remain in the
regression suite and CI.

## Day-14 status

| Requirement | Artifact | Status |
| --- | --- | --- |
| B-14.1 No-look-ahead | `tests/test_no_lookahead.py` | PASS |
| B-14.2 Future perturbation | leakage tests | PASS |
| B-14.3 Target alignment | leakage tests | PASS |
| B-14.4 Dependency lock | `requirements.txt` | PASS |
| B-14.5 Basic CI | `.github/workflows/tests.yml` | PASS |
| B-14.6 Leakage audit | `docs/leakage-audit.md` | PASS |
| Random-seed policy | deterministic baseline | N/A |
| Reproducible run script | `scripts/run_pipeline.py` | PASS |

## Conclusion

Historical Simulation and EWMA pass the implemented no-look-ahead,
future-perturbation, target-alignment, and strict-violation checks.

The baseline methods are deterministic, so an explicit random seed is
not required for this stage.

Dependency locking and basic CI are present.

The explicit reproducibility entry point `scripts/run_pipeline.py`
successfully reproduces the canonical Historical and EWMA baseline
pipeline and passes the consistency audit.

All required Day-14 leakage and reproducibility controls are present
and validated.

Overall Day-14 audit status:

`PASS`