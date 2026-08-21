# Gradient Boosting Validation Tuning

## Objective

This experiment slice evaluates the ensemble-size and learning-rate
trade-off of Gradient Boosting quantile regression for portfolio VaR.

G01 is rerun as the internal reference. G05-G07 are validation tuning
experiments only. This analysis does not select or declare a final model.

## Validation Contract

Target:

`portfolio_simple_return`

Training period after feature warm-up:

`2020-04-06` through `2021-12-30`

Validation period:

`2021-12-31` through `2024-12-17`

Training rows: `440`

Validation rows per experiment: `739`

The validation boundary was fixed before reviewing G05-G07 results and
was not changed after observing experiment metrics.

Later evaluation data are not used for tuning or model selection.

## Feature Set

All four experiments use the same 13 features.

Return-history features:

- `return_lag_1`
- `return_lag_2`
- `return_lag_5`
- `rolling_vol_5`
- `rolling_vol_20`
- `rolling_vol_60`
- `drawdown`

Market/liquidity features:

- `portfolio_range`
- `volume_change`
- `relative_volume_20`
- `market_return_lag_1`
- `market_return_lag_5`
- `market_vol_20`

Rows with incomplete feature warm-up are excluded before training.

## Experiment Configuration

All experiments use:

- loss: `quantile`
- alpha: `0.05`
- min_samples_leaf: `5`
- random_state: `42`

| Experiment | max_depth | learning_rate | n_estimators |
| --- | ---: | ---: | ---: |
| G01 | 2 | 0.05 | 100 |
| G05 | 2 | 0.05 | 50 |
| G06 | 2 | 0.05 | 200 |
| G07 | 2 | 0.10 | 100 |

G01 is the internal reference for this tuning slice.

## Metric Semantics

Violation is defined strictly as:

`actual_return < quantile_return`

Equality is not a violation.

Pinball loss is evaluated at `alpha = 0.05`.

VaR is:

`max(0, -quantile_return)`

Average VaR is reported as a descriptive risk measure and is not used
alone for model selection.

Runtime covers model fitting and validation prediction.

## Results

| Experiment | Estimators | LR | Violations | Violation Rate | Pinball Loss | Average VaR | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G01 | 100 | 0.05 | 53 | 0.071719 | 0.002157 | 0.021885 | 0.089627 |
| G05 | 50 | 0.05 | 50 | 0.067659 | 0.002113 | 0.023124 | 0.034090 |
| G06 | 200 | 0.05 | 61 | 0.082544 | 0.002224 | 0.020990 | 0.139183 |
| G07 | 100 | 0.10 | 69 | 0.093369 | 0.002307 | 0.021267 | 0.067553 |

All experiments produced `739` validation predictions.

The combined prediction artifact contains `2956` prediction rows.

Aggregate metrics were independently recomputed from prediction-level
evidence and matched the values in `results/gb_tuning_b.csv`.

## Ensemble-Size Analysis

G05, G01, and G06 isolate the effect of using 50, 100, and 200
estimators while keeping learning rate at `0.05`.

G05 produced the lowest pinball loss (`0.002113`) and the lowest
violation rate (`6.77%`) among these three configurations.

G01 produced a violation rate of `7.17%` and pinball loss of `0.002157`.

G06 increased the ensemble to 200 estimators but produced a higher
violation rate (`8.25%`) and higher pinball loss (`0.002224`).

Therefore, increasing boosting length from 50 to 100 or 200 estimators
did not improve validation calibration or pinball loss in this
experiment slice.

This result is validation evidence only and does not establish a final
model choice.

## Runtime Scaling

Observed runtimes were approximately:

- G05, 50 estimators: `0.0341 s`
- G01, 100 estimators: `0.0896 s`
- G06, 200 estimators: `0.1392 s`

Runtime generally increased with ensemble size.

The measurements are small and environment-dependent, so they should be
interpreted as relative experiment evidence rather than production
latency benchmarks.

## Learning-Rate Analysis

G01 and G07 both use 100 estimators and depth 2.

The primary configuration difference is:

- G01: learning rate `0.05`
- G07: learning rate `0.10`

G07 produced a violation rate of `9.34%`, compared with `7.17%` for G01.

Its pinball loss was also higher:

- G01: `0.002157`
- G07: `0.002307`

The observed validation evidence therefore does not show an improvement
from increasing the learning rate to `0.10`.

## Tail Prediction Range

Observed quantile ranges were:

| Experiment | Quantile Min | Quantile Max |
| --- | ---: | ---: |
| G01 | -0.056837 | -0.014192 |
| G05 | -0.057861 | -0.017450 |
| G06 | -0.057408 | -0.010180 |
| G07 | -0.061842 | -0.008470 |

G07 produced the most negative minimum quantile and the least negative
maximum quantile among the four experiments, giving it the widest
observed prediction range.

Combined with its higher violation rate and pinball loss, the
learning-rate `0.10` configuration shows less favorable validation
behavior than G01 in this slice.

This evidence is not sufficient by itself to characterize the model as
generally unstable.

## Calibration

The target quantile level is `5%`.

Observed violation rates were:

- G05: `6.77%`
- G01: `7.17%`
- G06: `8.25%`
- G07: `9.34%`

All four observed violation rates are above the nominal 5% level.

Among these experiments, G05 is closest to the nominal violation rate,
while G07 deviates the most.

These values describe validation calibration only.

## Pinball Loss and Average VaR Trade-Off

G05 has the lowest pinball loss and the highest Average VaR among the
four experiments.

G06 has a lower Average VaR than G05 but worse pinball loss and a higher
violation rate.

G07 also has lower Average VaR than G05 while producing the highest
pinball loss and violation rate.

This demonstrates why Average VaR must not be used alone to select a
configuration. Quantile loss and calibration evidence must also be
considered.

## Prediction-Level Evidence

Raw validation predictions are stored in:

`results/gb_tuning_b_predictions.csv`

Each row contains:

- `experiment_id`
- `forecast_date`
- `target_date`
- `actual_return`
- `quantile_return`
- `var`
- `violation`

Aggregate results are stored in:

`results/gb_tuning_b.csv`

All aggregate violation, violation-rate, pinball-loss, and Average VaR
values can be recomputed from the raw predictions.

## No-Look-Ahead

Return-history features are target-date aligned and use historical
returns available before the target date.

Market/liquidity features follow the source-date contract:

`source_max_date <= forecast_date < target_date`

Every prediction also satisfies:

`forecast_date < target_date`

The same validation target dates are used for G01, G05, G06, and G07.

No later evaluation segment is used to select or modify these
configurations.

## Limitations

This experiment evaluates only the predefined G01, G05, G06, and G07
configurations.

The results are based on one fixed validation period.

Runtime measurements depend on the execution environment.

Validation results must not be interpreted as final out-of-sample model
performance.

No experiment in this report is declared the final model.

## Conclusion

Within this validation slice, G05 provides the lowest pinball loss and
the violation rate closest to the nominal 5% level.

Increasing ensemble size to 200 estimators did not improve the observed
validation metrics.

Increasing learning rate from `0.05` to `0.10` at 100 estimators
produced a higher violation rate, higher pinball loss, and wider
prediction range.

These findings are tuning evidence only and do not establish a final
Gradient Boosting model.

## Next Step

Combine the G05-G07 evidence with the other validation tuning slice
before making any model-selection decision.

Any later evaluation segment must remain excluded from tuning decisions.