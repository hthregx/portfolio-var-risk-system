# G01 Quantile Boosting Prototype Validation

## Objective

G01 is an independent reference Gradient Boosting Quantile Regression
experiment owned by Member B.

The experiment uses
`GradientBoostingRegressor(loss="quantile", alpha=0.05)` directly in the
experiment runner and does not depend on Member A's production Gradient
Boosting implementation.

## Feature Slice Used

G01 uses the B17 market/liquidity feature slice:

- `portfolio_range`
- `volume_change`
- `relative_volume_20`
- `market_return_lag_1`
- `market_return_lag_5`
- `market_vol_20`

Target:

`portfolio_simple_return`

This is a prototype feature configuration, not the final Gradient
Boosting feature set.

## Training / Validation Contract

Training:

- start: `2020-02-07`
- end: `2021-12-30`
- rows: `480`

Validation:

- start: `2021-12-31`
- end: `2024-12-17`
- rows: `739`

The later segment beginning `2024-12-18` is excluded from G01 model
selection and is not described as a pristine untouched test set.

## G01 Configuration

- estimator: `GradientBoostingRegressor`
- loss: `quantile`
- alpha: `0.05`
- n_estimators: `100`
- learning_rate: `0.1`
- max_depth: `3`
- min_samples_leaf: `1`
- random_state: `42`

This is a fixed reference configuration, not a tuned optimum.

## Validation Metrics

- violations: `60`
- violation rate: `0.08119079837618404`
- pinball loss: `0.002189761236169089`
- average VaR: `0.021171411684163786`

Average VaR is reported as a descriptive risk-magnitude statistic and
is not used alone to select or rank the model.

## Runtime

Runtime is recorded in:

- `results/gb_experiment_log.csv`
- `results/gb_g01_metadata.json`

Runtime is measured for model fit and validation prediction.

## Prediction Sanity Checks

The G01 validation run confirmed:

- all quantile predictions are finite;
- all VaR values are finite and non-negative;
- prediction count is `739`;
- violation count is `60`;
- target dates match the validation sequence;
- target dates are unique;
- `forecast_date < target_date`;
- quantile distribution diagnostics are recorded;
- minimum and maximum VaR are recorded;
- obvious-outlier diagnostics are recorded.

Observed diagnostic values:

- quantile min: `-0.06390380909568857`
- quantile max: `-0.0006457485223488194`
- VaR min: `0.0006457485223488194`
- VaR max: `0.06390380909568857`
- obvious outlier count: `43`

The outlier count is diagnostic only. Predictions are not automatically
removed or clipped.

## Exception Count

G01 produced `60` validation violations.

Violation uses the strict rule:

`actual_return < quantile_return`

Equality is not counted as a violation.

## No-Look-Ahead

Feature rows use only information available before their target dates.

The B17 market/liquidity feature implementation follows:

`source_max_date <= forecast_date < target_date`

The G01 validation predictions also satisfy:

`forecast_date < target_date`

The later evaluation segment is not used for G01 training or model
selection.

## Limitations

- G01 is one fixed reference configuration.
- It uses only the B17 market/liquidity feature slice.
- No hyperparameter comparison has yet been performed.
- Results are validation evidence only.
- G01 does not establish that Gradient Boosting is the best or superior
  model.

## Next Step

The next step is light tuning on the validation period.