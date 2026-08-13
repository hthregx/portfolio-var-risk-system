# Date, Target and Look-Ahead Audit — 14/08/2026

## Objective

This audit verifies the one-day-ahead date, target, realized-return, and no-look-ahead contracts for the canonical Historical Simulation and EWMA VaR evaluations.

The two methods are audited on exactly the same 1,387 canonical target dates.

## Canonical contract

Historical Simulation uses:

- window size: 250 observations
- confidence level: 95%
- one trading day forecast horizon

EWMA uses:

- alpha: 0.05
- decay: 0.94
- one trading day forecast horizon

The EWMA decay remains 0.94 in this canonical audit because the purpose is date/target and leakage verification rather than the 13/08 validation-based parameter-selection experiment.

For both methods, a violation is defined strictly as:

`target_return < quantile_return`

Equality is not a violation.

## Date and target alignment

Both Historical Simulation and EWMA produced exactly 1,387 forecasts on the same target-date sequence.

The canonical target period is:

`2020-12-31` through `2026-07-28`

Audit results:

| Check | Result |
|---|---:|
| Historical rows | 1,387 |
| EWMA rows | 1,387 |
| Common target dates identical | True |
| Common realized target returns identical | True |
| Historical target-return maximum difference vs canonical | 0.000e+00 |
| All forecast dates strictly before target dates | True |

For Historical Simulation:

`forecast_date == window_end_date`

and for every forecast:

`forecast_date < target_date`

EWMA histories contain only portfolio returns dated on or before the corresponding `forecast_date`, and every history date is strictly earlier than its `target_date`.

## Historical canonical regression

The production rolling Historical orchestration was compared against the canonical Historical backtest.

Maximum absolute differences were:

| Field | Maximum absolute difference |
|---|---:|
| quantile_return | 9.714e-17 |
| Historical VaR | 9.714e-17 |
| target_return | 0.000e+00 |

These differences are below the locked numerical tolerance of `1e-12`.

The strict Historical violation count is 75.

## EWMA alignment

EWMA was rebuilt for each canonical Historical forecast date using the complete available portfolio-return history through that forecast date only.

The EWMA target-date sequence and realized target-return sequence are identical to Historical Simulation row by row.

The strict EWMA violation count is 73.

This audit does not reinterpret these counts as a formal coverage test.

## Future-perturbation leakage audit

A deterministic future-perturbation check was performed at three forecast positions:

- first row: target `2020-12-31`
- middle row: target `2023-10-13`
- last row: target `2026-07-28`

For each checkpoint, portfolio returns strictly after the selected `target_date` were replaced with extreme deterministic values.

The selected forecast was then recomputed while preserving the permitted estimation history.

Observed forecast differences were:

| Position | Historical q diff | Historical VaR diff | EWMA q diff | EWMA VaR diff |
|---:|---:|---:|---:|---:|
| 0 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 693 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 1386 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

The perturbation therefore did not affect any audited Historical or EWMA forecast.

## Interpretation and safeguards

The audit supports the implementation-level conclusion that the evaluated Historical and EWMA forecasts obey the locked one-day-ahead date and target contract on the canonical evaluation sequence.

The future-perturbation result is a targeted leakage check rather than a proof that every possible future implementation is leakage-free.

No Kupiec test, formal coverage hypothesis test, or statistical-significance claim is introduced here.

No parameter tuning is performed in this audit.

## Reproducibility

Inputs:

- `data/processed/portfolio_returns.csv`
- `data/processed/historical_var_backtest.csv`

Production code:

- `src/backtesting/historical.py`
- `src/models/historical_var.py`
- `src/models/ewma_var.py`

Locked numerical tolerance:

`1e-12`

The Historical production orchestration regression was also covered by the project test suite, which contained 90 passing tests after the 14/08 refactor.

## Prompt provenance

The AI-assisted audit was constrained to the canonical 1,387 target dates, identical realized target returns, one-day-ahead date ordering, strict violation semantics, fixed Historical window 250, fixed canonical EWMA decay 0.94, and deterministic numerical tolerance `1e-12`.

The audit explicitly prohibited parameter tuning, future information in estimation histories, file writes during numerical verification, automatic staging or committing, and claims of statistical or universal superiority.

Future-perturbation checks were specified before interpretation and independently asserted for the first, middle, and last canonical evaluation positions.
