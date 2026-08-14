cat > docs/leakage-audit.md <<'EOF'
# Leakage Audit

## Scope

This audit verifies that the Historical Simulation and EWMA VaR
evaluation pipelines respect one-day-ahead forecasting and do not use
future target returns when producing forecasts.

The audit covers:

- forecast-origin and target-date alignment;
- training-data inclusion rules;
- Historical Simulation;
- EWMA;
- future-data perturbation;
- cross-method target alignment;
- strict violation semantics.

## Forecast origin definition

For each one-day-ahead forecast, `forecast_date` is the final date
available to the model when the forecast is produced.

The corresponding `target_date` is the next evaluation date.

The required invariant is:

`forecast_date < target_date`

## Target definition

The target observation is the portfolio return observed on
`target_date`.

Therefore:

- the target return is used only for evaluation;
- it must not be included in the training history used to produce the
  forecast for that target;
- `actual_return` must correspond exactly to the return on
  `target_date`.

## Training inclusion rule

For every forecast, training observations must satisfy:

`training_date <= forecast_date`

and:

`training_date < target_date`

The maximum training date must never reach the target date.

Historical Simulation uses its configured rolling history.

EWMA uses the historical observations available before the target date.

## Historical Simulation audit

Historical Simulation was tested through the common walk-forward
runner.

The tests verify:

- `forecast_date < target_date`;
- training data ends before the target;
- target dates are unique;
- target returns are aligned correctly;
- future observations do not affect an already-produced forecast.

Historical no-look-ahead checks passed.

## EWMA audit

EWMA was tested through the same common walk-forward runner using the
production `ewma_var_forecast` implementation.

The tests verify:

- `forecast_date < target_date`;
- training data ends before the target;
- target dates are unique;
- target returns are aligned correctly;
- future observations do not affect an already-produced forecast.

EWMA no-look-ahead checks passed.

## Future perturbation test

A future perturbation test was performed for both Historical Simulation
and EWMA.

Procedure:

1. Produce a forecast for a selected target date.
2. Copy the input dataset.
3. Replace returns strictly after that target date with extreme values.
4. Re-run the walk-forward forecast.
5. Compare the original and perturbed forecasts for the selected target.

The forecast quantile and VaR remained unchanged within numerical
tolerance.

Result:

`PASS`

This provides direct evidence that observations occurring after the
target date do not influence the earlier forecast.

## Target alignment and same-date audit

Historical Simulation and EWMA were evaluated using the same target
sequence in the alignment tests.

The audit verifies:

- no duplicate `target_date`;
- expected target dates are not skipped;
- `actual_return` corresponds to the target observation;
- Historical and EWMA use the same `actual_return` for the same target;
- target-date sequences match between methods.

Result:

`PASS`

## Violation semantics

A violation is defined strictly as:

`actual_return < quantile_return`

Equality is not a violation.

The equality boundary case was tested explicitly:

`actual_return == quantile_return`

Result:

`violation == False`

## Test evidence

The dedicated leakage and alignment suite produced:

`10 passed`

The full project regression suite produced:

`130 passed`

These results include Historical and EWMA no-look-ahead, future
perturbation, target alignment, cross-method consistency, and strict
violation checks.

## Remaining risks

The automated tests provide evidence against look-ahead leakage in the
current common walk-forward implementation and production model
interfaces.

Remaining risks include:

- future changes to runner slicing or date-alignment logic;
- model implementations bypassing the common runner contract;
- externally generated input artifacts whose construction is not
  covered by these tests;
- changes to preprocessing that introduce information unavailable at
  forecast time.

The leakage tests should remain part of the regression suite and CI so
that future changes are checked against the same invariants.

## Conclusion

The current Historical Simulation and EWMA walk-forward evaluation
passes the implemented no-look-ahead and target-alignment checks.

Future-data perturbation does not change earlier forecasts, target
returns are aligned consistently between methods, and violation
classification uses the required strict inequality.
EOF