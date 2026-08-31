# Final Results Independent QA

## Scope

This document records manual QA checkpoints for the independent
final-results contract fixture.

The sample fixture is used only to validate schema, sign, violation,
and metric semantics.

It is not presented as the final experimental result.

## Core Rules

VaR sign convention:

`var = max(0, -quantile_return)`

Violation rule:

`actual_return < quantile_return`

The comparison is strict.

Equality is not a violation.

## Checkpoint 1 — First Target Date

Target date:

`2024-01-03`

Method:

`historical_simulation`

Observed fixture values:

- actual return: `-0.010`
- quantile return: `-0.020`

Expected VaR:

`max(0, -(-0.020)) = 0.020`

Expected violation:

`-0.010 < -0.020`

Result:

`False`

Therefore:

- expected VaR = `0.020`
- expected violation = `False`

This is a normal non-violation observation.

## Checkpoint 2 — Middle Target Date

Target date:

`2024-01-04`

Method:

`gradient_boosting`

Observed fixture values:

- actual return: `-0.030`
- quantile return: `-0.030`

Expected VaR:

`max(0, -(-0.030)) = 0.030`

Expected violation:

`-0.030 < -0.030`

Result:

`False`

This checkpoint explicitly demonstrates the equality rule.

Equality does not count as a violation because the contract uses a
strict `<` comparison rather than `<=`.

Therefore:

- expected VaR = `0.030`
- expected violation = `False`

## Checkpoint 3 — Last Target Date

Target date:

`2024-01-05`

Method:

`historical_simulation`

Observed fixture values:

- actual return: `0.010`
- quantile return: `0.005`

Expected VaR:

`max(0, -(0.005)) = 0.000`

Expected violation:

`0.010 < 0.005`

Result:

`False`

Therefore:

- expected VaR = `0.000`
- expected violation = `False`

This checkpoint demonstrates the zero-floor VaR rule when the predicted
quantile is non-negative.

## Manual Metric Check

For the three `historical_simulation` fixture predictions:

Violation values:

- `False`
- `True`
- `False`

Violation count:

`1`

Forecast count:

`3`

Violation rate:

`1 / 3 = 0.3333333333333333`

VaR values:

- `0.020`
- `0.020`
- `0.000`

Average VaR:

`(0.020 + 0.020 + 0.000) / 3`

`= 0.013333333333333334`

Minimum VaR:

`0.000`

Maximum VaR:

`0.020`

At `alpha = 0.05`, the three Pinball Loss contributions are:

1. `0.0005`
2. `0.0095`
3. `0.00025`

Mean Pinball Loss:

`(0.0005 + 0.0095 + 0.00025) / 3`

`= 0.003416666666666667`

These values match the independent metric fixture and automated audit.

## QA Status

- first target checkpoint: PASS
- middle target checkpoint: PASS
- last target checkpoint: PASS
- equality non-violation explanation: PASS
- VaR sign convention: PASS
- zero-floor VaR case: PASS
- manual metric recomputation: PASS

The values in this document are contract-fixture QA values, not final
model-performance claims.