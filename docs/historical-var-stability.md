# Historical Simulation Stability Review - 09/08/2026

## Scope

This note records the stability review of the Historical Simulation VaR baseline performed on 09/08/2026.

The canonical configuration remains:

- Portfolio: equal-weight HPG, FPT and MWG
- Return: portfolio_simple_return
- VaR horizon: one trading day
- Confidence level: 95%
- Alpha: 0.05
- Historical rolling window: 250 observations
- Violation rule: actual_return < quantile_return
- VaR convention: var = max(0, -quantile_return)

This review is descriptive. It does not replace the baseline configuration and does not perform a formal coverage hypothesis test.

## Canonical backtest audit

The canonical Historical Simulation backtest contains 1,387 one-day forecasts from 2020-12-31 to 2026-07-28.

Validation results:

- Missing values: 0
- Duplicate target dates: 0
- Observations per forecast: 250
- Violations: 75
- Violation rate: 5.407354%
- VaR sign-convention maximum difference: 0.000e+00
- Strict violation rule matched all rows

## Exception-day review

All 75 exception days satisfy the strict rule:

`actual_return < quantile_return`

No equality cases were present in the full backtest.

Exception severity is defined as:

`loss_excess = quantile_return - actual_return`

Summary:

- Mean loss excess: 1.319919%
- Median loss excess: 0.789465%
- 25th percentile: 0.288329%
- 75th percentile: 1.871503%
- 90th percentile: 3.345445%
- Minimum loss excess: 0.017995%
- Maximum loss excess: 5.204671%

The most severe observed exception occurred for target date 2025-04-03:

- Actual return: -6.983986%
- Quantile return: -1.779316%
- Historical VaR: 1.779316%
- Loss excess: 5.204671%

These values describe exception severity only. No market-cause attribution is made in this review.

## VaR level review

Historical VaR descriptive statistics over the full canonical backtest are:

- Average VaR: 2.623147%
- Median VaR: 2.429710%
- Standard deviation: 0.692870%
- Minimum VaR: 1.708839%
- 25th percentile: 2.232218%
- 75th percentile: 2.702020%
- Maximum VaR: 4.378557%

The minimum VaR forecast occurred on 2024-10-31 for target date 2024-11-01.

The maximum VaR forecast occurred on 2022-12-26 for target date 2022-12-27.

Around the minimum-VaR neighborhood, the maximum one-step absolute VaR change was 0.097987 percentage points.

Around the maximum-VaR neighborhood, the maximum one-step absolute VaR change was 0.277720 percentage points.

The step-like behavior is consistent with an empirical rolling quantile, which may remain unchanged across several observations and then move when the order statistics in the rolling sample change.

## Minimal rolling-window sensitivity

A descriptive sensitivity check was performed for rolling windows 125, 250 and 500.

To ensure a fair comparison, all three configurations were evaluated on the same 1,137 target dates from 2021-12-31 to 2026-07-28.

| Window | Forecasts | Violations | Violation Rate | Pinball Loss | Average VaR | Minimum VaR | Maximum VaR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 125 | 1,137 | 73 | 6.420405% | 0.002049883256 | 2.510271% | 1.239389% | 4.955767% |
| 250 | 1,137 | 64 | 5.628848% | 0.002057873600 | 2.600651% | 1.708839% | 4.378557% |
| 500 | 1,137 | 68 | 5.980651% | 0.002079136820 | 2.571271% | 1.949615% | 3.101157% |

Relative to the nominal 5% violation rate:

- Window 125: +1.420405 percentage points
- Window 250: +0.628848 percentage points
- Window 500: +0.980651 percentage points

The three metrics do not identify unanimous dominance by one window.

The 125-day window has the lowest Pinball Loss but the highest violation rate. The 250-day window has the violation rate closest to the nominal 5% level among the three configurations on the common evaluation period. The 500-day window lies between the two in violation rate but has the highest Pinball Loss.

The 250-day baseline remains unchanged. This sensitivity analysis is descriptive and is not used as a test-set tuning procedure.

## Canonical baseline table

| Metric | Value |
|---|---:|
| Confidence Level | 95.00% |
| Alpha | 0.05 |
| Forecasts | 1,387 |
| Violations | 75 |
| Violation Rate | 5.407354% |
| Expected Violation Rate | 5.000000% |
| Expected Violations | 69.35 |
| Rate Difference | +0.407354 pp |
| Pinball Loss | 0.002057488935 |
| Average VaR | 2.623147% |
| Minimum VaR | 1.708839% |
| Maximum VaR | 4.378557% |

## Baseline interpretation

Across 1,387 one-day forecasts, Historical Simulation produced 75 VaR violations, corresponding to a violation rate of 5.407354% versus the nominal 5% rate.

The observed rate is 0.407354 percentage points above the target, equivalent to approximately 5.65 more violations than the nominal expected count of 69.35.

This result indicates mild undercoverage, or a small tendency for the Historical Simulation baseline to underestimate one-day lower-tail risk.

The Pinball Loss is 0.002057488935 and the Average VaR is 2.623147%.

These findings are a descriptive calibration assessment. No formal coverage hypothesis test is performed here, so the result does not establish statistical significance and is not interpreted as a rejection or failure of VaR coverage.

## Walk-forward dependency

At the start of the 09/08 review, the common walk-forward runner owned by the Software/Product workstream was not yet present on the integration branch.

The Historical model, unit tests and canonical numerical baseline are ready for integration. Final closure of the Historical Simulation phase still requires the common runner to reproduce the canonical Historical results without look-ahead.
