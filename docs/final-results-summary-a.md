# Final Walk-Forward Results ? Member A Analysis

## Objective

This analysis compares the three Day-21 one-day 95% portfolio VaR methods on
the same 398 target dates from 2024-12-18 through 2026-07-28.

The analysis is descriptive and evaluation-focused. It does not retune model
parameters, add features, rerun model selection, or declare a final overall
model winner.

The three evaluated methods are:

- Historical Simulation with a 250-observation rolling window;
- EWMA with decay 0.94;
- Gradient Boosting Quantile Regression using the G04 A-side validation
  candidate.

## Evaluation Contract

All three methods are evaluated on the same target-date universe and realized
portfolio returns.

The VaR conventions are:

- confidence level: 95%;
- quantile level: alpha = 0.05;
- horizon: one trading day;
- violation rule: `actual_return < quantile_return`;
- VaR value: `max(0, -quantile_return)`.

The evaluation period was reserved from parameter selection. It is not
described as a pristine untouched test set because the broader evaluation
period had previously been inspected.

## Core Metric Comparison

| Method | Forecasts | Violations | Violation Rate | Distance from 5% | Pinball Loss | Average VaR |
|---|---:|---:|---:|---:|---:|---:|
| Historical Simulation | 398 | 27 | 6.78% | 1.78% | 0.001896464233 | 2.13% |
| EWMA | 398 | 22 | 5.53% | 0.53% | 0.001968321253 | 2.51% |
| Gradient Boosting G04 | 398 | 24 | 6.03% | 1.03% | 0.001758130951 | 2.33% |

## Criterion-Specific Results

### Calibration

EWMA is closest to the nominal 5% violation rate.

Its observed violation rate is 5.53%, corresponding to
an absolute calibration distance of 0.53%.

Gradient Boosting G04 has a violation rate of 6.03%,
while Historical Simulation has the highest violation rate at
6.78%.

This comparison concerns empirical violation frequency only. Being closest to
5% is not, by itself, sufficient to identify an overall best model.

### Quantile Forecast Accuracy

Gradient Boosting G04 has the lowest aggregate Pinball Loss:

- Gradient Boosting G04: 0.001758130951;
- Historical Simulation: 0.001896464233;
- EWMA: 0.001968321253.

Therefore, G04 provides the strongest aggregate quantile-loss result on this
evaluation sample.

The paired analysis shows, however, that this aggregate result must not be
interpreted as G04 producing the lowest loss on most dates in every pairwise
comparison.

## Paired Gradient Boosting versus Historical Analysis

Gradient Boosting has lower observation-level Pinball Loss than Historical
Simulation on 151 of 398 target dates, while
Historical Simulation has lower loss on
247 dates.

There are 0 ties.

The mean paired loss difference,

`GB loss - Historical loss`,

is -0.000138333282.

The negative mean difference confirms that Gradient Boosting has the lower
average loss despite winning on fewer individual dates.

This result is magnitude-driven rather than frequency-driven.

The concentration diagnostics further show that:

- the largest 5 Gradient Boosting improvements account for
  43.70% of its total positive improvement
  magnitude over Historical Simulation;
- the largest 20 improvements account for
  86.55%;
- only 7 dates are required to
  account for 50% of total Gradient Boosting improvement magnitude;
- 16 dates account for 80%.

The largest Gradient Boosting improvement relative to Historical Simulation
occurs on 2025-04-09.

These results indicate that the aggregate Gradient Boosting advantage over
Historical Simulation is concentrated in a comparatively small number of
high-impact dates.

## Paired Gradient Boosting versus EWMA Analysis

Gradient Boosting has lower observation-level Pinball Loss than EWMA on
231 of 398 dates, compared with
167 dates favoring EWMA.

The mean paired difference,

`GB loss - EWMA loss`,

is -0.000210190302.

In contrast with the Historical comparison, Gradient Boosting also wins the
majority of individual dates against EWMA.

Its improvement magnitude is less concentrated than in the
Gradient-Boosting-versus-Historical comparison:

- the largest 5 improvements account for
  29.72%;
- the largest 20 account for
  57.21%;
- 13 dates account for 50% of total
  improvement magnitude;
- 63 dates account for 80%.

This evidence supports a broader paired advantage of Gradient Boosting over
EWMA than over Historical Simulation.

## Historical Simulation versus EWMA

The pairwise comparison also helps explain why Historical Simulation has a
lower aggregate Pinball Loss than EWMA.

EWMA has lower loss on only 131 dates,
whereas Historical Simulation has lower loss on
267 dates.

The mean difference,

`EWMA loss - Historical loss`,

is 0.000071857020, which is positive and therefore favors
Historical Simulation on average.

## Risk Magnitude

Historical Simulation has the lowest Average VaR at
2.13%.

Gradient Boosting G04 has an Average VaR of 2.33%, while
EWMA has the highest Average VaR at 2.51%.

A lower Average VaR is not automatically superior. It represents a lower
average forecast risk magnitude and must be interpreted together with
calibration and quantile-loss performance.

Historical Simulation combines the lowest Average VaR with the highest
violation rate among the three methods, illustrating why risk magnitude alone
cannot serve as the model-selection criterion.

## Result Interpretation

The Day-22 evidence is criterion-specific:

1. **EWMA provides the closest empirical calibration to the nominal 5%
   violation rate.**
2. **Gradient Boosting G04 provides the lowest aggregate Pinball Loss.**
3. **Historical Simulation provides the lowest Average VaR.**
4. **Historical Simulation produces the lowest observation-level loss on more
   target dates than G04 in their direct comparison.**
5. **G04 nevertheless has lower mean loss than Historical Simulation because
   its improvements on selected dates are substantially larger.**
6. **G04's advantage over EWMA is broader, combining a lower mean loss with a
   majority of individually better target dates.**

These findings demonstrate that model comparison cannot be reduced to a single
frequency count or a single average risk level.

## Limitations

The evaluation sample was reserved from parameter selection but is not claimed
to be a pristine untouched test set.

The Gradient Boosting configuration is G04, which is an A-side validation
candidate rather than a confirmed final G01-G07 tuning winner.

The implemented Gradient Boosting feature set contains return-history features
only. Price-range, trading-volume and market-index feature families are not
part of this implementation.

The paired concentration analysis is descriptive. It identifies where loss
differences occur but does not establish causal explanations for those dates.

Exception-timing and volatility-regime analysis are handled separately and
should not be inferred from this Member-A metric analysis.

## Model-Selection Status

No overall model winner is declared in this analysis.

The results are retained as evidence for the Day-23 model-freeze decision,
where calibration, quantile-loss behavior, model specification,
reproducibility and documented limitations must be considered together.
