# Gradient Boosting Walk-Forward Integration

## Objective

Integrate the production `GradientBoostingVaR` model into the common
walk-forward backtesting framework without changing Historical or EWMA
semantics.

This is an integration smoke step only. It is not the final three-model
comparison.

## Integration Design

The existing analytical walk-forward path is retained for Historical
and EWMA.

Gradient Boosting is integrated through the stateful model path using a
model factory. G01 experiment internals are not hard-coded into the
common walk-forward runner.

For each target date `T`, the Gradient Boosting path:

1. selects eligible historical training rows;
2. requires every training row to precede `T`;
3. excludes target date `T` from training;
4. fits the production model on historical features and targets;
5. predicts the feature row corresponding to `T`;
6. normalizes the prediction to common `quantile_return` and `var`
   semantics.

The configured rolling or expanding training policy is handled by the
common runner.

## No-Lookahead and Date Contract

For every target date `T`:

`training_date < T`

and:

`forecast_date < target_date`

The target observation is never included in its own training set.

Feature row `T` must use only information available before target date
`T`.

Warm-up rows with incomplete features are excluded before model fitting.

Dedicated tests perturb target/future source observations and verify
that the feature row for `T` does not change.

## Feature Schema

The integration records the canonical 13-feature schema.

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

Feature count:

`13`

## Common Model Semantics

Historical, EWMA, and Gradient Boosting are checked against common
walk-forward semantics.

The integration verifies:

- common prediction schema;
- same target-date alignment;
- `forecast_date < target_date`;
- finite `quantile_return`;
- finite `var`;
- non-negative `var`;
- common VaR sign convention;
- common strict violation semantics.

This integration does not perform the final three-model performance
comparison.

## VaR Sign Convention

The common VaR representation is:

`var = max(0, -quantile_return)`

Therefore VaR must always be finite and non-negative.

## Strict Violation Semantics

A violation occurs only when:

`actual_return < quantile_return`

The comparison is strict.

If:

`actual_return == quantile_return`

the observation is not a violation.

The equality case is explicitly covered by an integration test through
the walk-forward runner.

## Production Model and Config Audit

The smoke integration uses the production `GradientBoostingVaR`.

Runtime audit verifies the fitted model has:

- `alpha == 0.05`
- fitted sklearn estimator `loss == "quantile"`
- `random_state == 42`
- fitted feature names equal the recorded 13-feature schema

The `loss` audit is read from the fitted estimator rather than from a
display-only constant.

`configs/gradient_boosting.yaml` is read-only for B20 and is not modified
by this integration.

## Gradient Boosting Adapter Evidence

Dedicated integration tests verify that:

- the GB adapter can fit and predict through the common runner;
- every model training index is strictly earlier than its target date;
- target date is excluded from its own training set;
- target dates remain correctly aligned;
- quantile predictions are finite;
- VaR predictions are finite;
- VaR predictions are non-negative;
- strict violation semantics are applied by the runner;
- equality is not counted as a violation;
- fixed-seed execution is deterministic.

The common runner does not hard-code G01-specific experiment logic.

## Feature Timing Evidence

No-lookahead behavior is tested for both feature slices.

### Return-history features

The test perturbs returns at the target date and later dates.

The feature row for the target date must remain unchanged.

This verifies that target/future returns are not used to construct the
target feature row.

### Market/liquidity features

The test perturbs target/future stock OHLCV observations and target/future
market observations.

The market/liquidity feature row for the target date must remain
unchanged.

This verifies that target/future source observations do not leak into
the target feature row.

## Historical / EWMA Regression Status

Existing Historical and EWMA regression tests remain green.

Regression gate:

`60 passed`

Dedicated cross-model integration tests separately verify common schema,
target-date alignment, VaR-sign convention, date semantics, and strict
violation semantics across Historical, EWMA, and Gradient Boosting.

The Gradient Boosting integration therefore does not require changing
Historical or EWMA semantics.

## Reproducibility

The production Gradient Boosting integration uses:

`random_state = 42`

Dedicated integration tests execute the same fixed-seed configuration
more than once and verify identical Gradient Boosting quantile and VaR
predictions.

## Smoke Artifact

Artifact:

`results/gb_walk_forward_smoke.csv`

Required schema:

- `forecast_date`
- `target_date`
- `actual_return`
- `quantile_return`
- `var`
- `violation`
- `method`

For all smoke predictions:

`method = gradient_boosting`

Observed smoke run:

- prediction rows: `40`
- alpha: `0.05`
- loss: `quantile`
- random state: `42`
- feature count: `13`
- method: `gradient_boosting`

Smoke checks:

- production model fit/predict: PASS
- quantile predictions finite: PASS
- VaR finite: PASS
- VaR non-negative: PASS
- `forecast_date < target_date`: PASS
- strict violation semantics: PASS
- smoke CSV generated: PASS

The smoke artifact is intentionally a small integration subset. It is
not the final walk-forward evaluation scheduled for the later
three-model comparison.

## Automated Test Evidence

Dedicated B20 integration tests verify:

- GB adapter fit/predict and common output contract;
- every training row precedes its target date;
- target date is excluded from its own training set;
- target-date alignment is preserved;
- `forecast_date < target_date`;
- finite `quantile_return`;
- finite and non-negative `var`;
- strict violation through the runner:
  `actual_return < quantile_return`;
- equality is explicitly verified as a non-violation through the runner;
- return-history feature row `T` is unchanged after perturbing
  target/future returns;
- market/liquidity feature row `T` is unchanged after perturbing
  target/future OHLCV and market data;
- Historical, EWMA, and Gradient Boosting share common schema,
  target-date alignment, VaR-sign convention, and violation semantics;
- fitted production model uses `alpha = 0.05`,
  `loss = "quantile"`, and `random_state = 42`;
- fixed-seed Gradient Boosting predictions are deterministic.

Dedicated B20 integration suite:

`9 passed`

Existing Historical/EWMA regression suite:

`60 passed`

Full repository suite:

`280 passed`