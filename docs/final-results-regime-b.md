# Final Results — Exception Timing and Volatility Regimes

## Objective

This analysis examines when final VaR exceptions occurred and how model
performance varied across volatility regimes. It is post-hoc descriptive
analysis and does not modify model training or forecasts.

The evaluation contains 398 common target dates. Using the strict rule
`actual_return < quantile_return`, Historical has 27 violations, EWMA 22,
and Gradient Boosting (GB) 24.

## Exception Timing

Exceptions were classified deterministically as shared by all three models,
shared by exactly two models, or exclusive to one model.

Across 398 dates:

| Category | Dates |
|---|---:|
| No violation | 365 |
| Shared by all 3 | 16 |
| Shared by exactly 2 | 8 |
| Historical only | 5 |
| EWMA only | 1 |
| GB only | 3 |

The most severe shared exception occurred on 2025-04-03 with an actual
portfolio return of -6.9840%.

The deterministic near-consecutive clustering rule identified the largest
multi-method cluster from 2026-03-02 to 2026-03-09, containing 5 exception
dates and 11 method-level violations.

## Regime Definition

Volatility regimes use prior 20-day portfolio volatility. The target day's
return is excluded, so regime assignment uses forecast-time information.

Thresholds are fixed from observations before the evaluation start:

- LOW: volatility < 0.011557
- NORMAL: 0.011557 to 0.016755
- HIGH: volatility > 0.016755

These regimes are used only for descriptive analysis and are not GB
features or model inputs.

## Regime Results

| Regime | Model | Obs. | Violations | Rate | Pinball Loss |
|---|---|---:|---:|---:|---:|
| LOW | Historical | 123 | 9 | 7.32% | 0.001968 |
| LOW | EWMA | 123 | 12 | 9.76% | 0.002180 |
| LOW | GB | 123 | 9 | 7.32% | 0.001830 |
| NORMAL | Historical | 166 | 9 | 5.42% | 0.001506 |
| NORMAL | EWMA | 166 | 7 | 4.22% | 0.001595 |
| NORMAL | GB | 166 | 10 | 6.02% | 0.001472 |
| HIGH | Historical | 109 | 9 | 8.26% | 0.002411 |
| HIGH | EWMA | 109 | 3 | 2.75% | 0.002297 |
| HIGH | GB | 109 | 5 | 4.59% | 0.002113 |

## Figures

- `figures/final_var_forecast_timeline.png`
- `figures/final_exception_timing.png`
- `figures/final_regime_comparison.png`

The timeline compares actual portfolio returns with the three 5% quantile
forecasts. The exception figure shows shared and model-exclusive violations.
The regime figure compares violation rates with the nominal 5% reference.

## Interpretation

Performance varies across volatility regimes, but the pattern is not uniform
across models.

Historical has its highest violation rate in HIGH volatility (8.26%).
EWMA has its highest violation rate in LOW volatility (9.76%) and its lowest
in HIGH volatility (2.75%). GB is closest to the nominal 5% rate in HIGH
volatility (4.59%).

Therefore, the evidence does not support the general claim that higher
volatility caused all models to fail more often. These results describe
observed associations between regime and out-of-sample VaR performance.

## Limitations

Regime analysis is post-hoc and descriptive, not causal. Results depend on
the fixed 20-day volatility definition and pre-evaluation thresholds.
Regime sample sizes also differ, so violation rates should not be interpreted
as definitive model rankings.