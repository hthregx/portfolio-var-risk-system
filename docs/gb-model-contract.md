# Gradient Boosting Quantile VaR Model Contract

## Purpose

This module implements the Gradient Boosting Quantile Regression model used to forecast the lower 5% quantile of the one-day-ahead portfolio return distribution.

The model is the machine-learning VaR method required by the project and is designed to remain compatible with the existing Historical Simulation and EWMA VaR sign convention.

## Production Implementation

Primary implementation:

`src/models/gradient_boosting_var.py`

Primary class:

`GradientBoostingVaR`

Dependency:

`scikit-learn==1.9.0`

Canonical configuration:

`configs/gradient_boosting.yaml`

Automated tests:

`tests/test_gradient_boosting_var.py`

## Forecast Target

The project target is the one-day-ahead portfolio simple return:

`portfolio_simple_return`

The model estimates the conditional lower-tail quantile:

`q_0.05(R_{t+1} | information available at t)`

The estimator predicts this quantile directly.

It does not fit a conditional mean and then convert the mean to VaR using a Normal-distribution assumption.

## Quantile Objective

The underlying estimator is:

`sklearn.ensemble.GradientBoostingRegressor`

with:

`loss="quantile"`

and the canonical project quantile level:

`alpha=0.05`

This corresponds to the 5% lower quantile associated with 95% one-day VaR.

## G01 Prototype Configuration

Experiment `G01` is the initial Gradient Boosting prototype configuration.

Its hyperparameters were fixed before validation performance was inspected:

- loss: `quantile`;
- alpha: `0.05`;
- n_estimators: `100`;
- learning_rate: `0.05`;
- max_depth: `2`;
- min_samples_leaf: `5`;
- subsample: `1.0`;
- random_state: `42`;
- forecast horizon: `1` trading observation.

The configuration file marks G01 as:

`selection_status: prototype_pre_validation`

G01 is therefore a prototype configuration rather than a tuned or frozen final model.

Later validation-based tuning must be recorded as separate experiments and must not retroactively change the definition of G01.

## Model API

### Construction

The canonical prototype is created with:

`GradientBoostingVaR()`

which loads the G01 default hyperparameters.

Hyperparameter inputs are validated before fitting.

### Fit

The production fit interface is:

`fit(features, target)`

`features` must be a non-empty pandas DataFrame with:

- unique feature-column names;
- string feature-column names;
- unique row index;
- numeric values;
- finite values;
- no missing values.

The feature builder is responsible for removing intentional feature warm-up rows before they are passed to the estimator.

The target must be one-dimensional, numeric, finite, non-empty, and have the same number of observations as the feature matrix.

When the target is a pandas Series, its index must exactly equal the feature DataFrame index.

This prevents silent time-series misalignment between feature rows and realized returns.

## Feature Schema Contract

Feature names and their column order are recorded during fitting.

Prediction data must contain exactly the same feature names in exactly the same order.

A reordered, missing, renamed, or additional feature column is rejected rather than silently rearranged.

The production model does not perform feature engineering internally.

Return-history, market, range, volume, and other feature families remain upstream responsibilities.

## Prediction API

### Quantile Prediction

`predict_quantile(features)`

returns a one-dimensional NumPy array containing predicted lower-tail return quantiles.

All predictions must be finite.

### Batch VaR Prediction

`predict(features)`

returns a pandas DataFrame with the columns:

1. `quantile_return`
2. `var`

The prediction DataFrame preserves the input feature index.

### Single-Row Forecast

`forecast(features)`

requires exactly one feature row and returns the common VaR dictionary contract:

`{"quantile_return": q, "var": v}`

This allows the machine-learning model to use the same sign semantics as the Historical Simulation and EWMA baselines.

## VaR Sign Convention

For every predicted return quantile:

`VaR = max(0, -quantile_return)`

Therefore:

- negative lower-tail return quantiles become positive VaR values;
- non-negative lower-tail return quantiles produce zero VaR;
- VaR is never negative.

The model does not change the project-wide sign convention.

## Temporal and Leakage Boundary

The model assumes that every input feature row has already been constructed using information available before the corresponding target return.

The return-history feature family enforces this separately through target-date alignment, lagging, shifted rolling statistics, shifted drawdown, target exclusion, and future-perturbation tests.

The estimator itself must not receive the realized target return as an input feature.

When a pandas Series is used as the training target, exact feature/target index equality is required.

The Gradient Boosting model therefore preserves upstream no-look-ahead controls rather than attempting to reconstruct temporal alignment internally.

## Reproducibility

G01 fixes:

`random_state=42`

and:

`subsample=1.0`

Automated tests fit two independent model instances on identical data and require identical predictions.

The model does not mutate caller-provided feature or target objects.

The installed machine-learning dependency used during implementation is explicitly declared as:

`scikit-learn==1.9.0`

in `requirements.txt`.

## Validation and Failure Policy

The implementation fails explicitly for structurally invalid model inputs.

It rejects examples including:

- empty feature matrices;
- duplicate feature columns;
- duplicate feature index values;
- non-string feature-column names;
- missing feature values;
- infinite feature values;
- non-numeric feature values;
- multidimensional targets;
- non-numeric targets;
- non-finite targets;
- feature/target length mismatch;
- pandas target-index mismatch;
- prediction before fitting;
- prediction feature-schema mismatch;
- invalid numeric hyperparameters.

Boolean values are not accepted as numeric hyperparameter substitutes.

## G01 Scope Boundary

G01 establishes that the required Gradient Boosting Quantile model can be fitted and can produce reproducible one-day lower-tail forecasts under the project VaR contract.

G01 does not establish that Gradient Boosting is superior to Historical Simulation or EWMA.

G01 does not constitute final feature selection.

G01 does not constitute hyperparameter tuning.

G01 does not constitute model freeze.

Validation metrics, experiment comparison, and light hyperparameter tuning are separate subsequent activities.

## Relationship to Baseline Models

Historical Simulation and EWMA return a common forecast representation containing:

`quantile_return`

and:

`var`

The Gradient Boosting wrapper preserves those semantics for downstream comparison.

All three methods must ultimately be evaluated using the same:

- portfolio target;
- target dates;
- one-day horizon;
- VaR sign convention;
- strict violation rule;
- evaluation metrics.

The strict project violation rule remains:

`actual_return < quantile_return`

Equality is not a violation.

## Implementation Traceability

Model:

`src/models/gradient_boosting_var.py`

G01 configuration:

`configs/gradient_boosting.yaml`

Tests:

`tests/test_gradient_boosting_var.py`

Dependency manifest:

`requirements.txt`

Return-history feature contract:

`docs/gb-return-feature-contract.md`

## Prompt Provenance

The implementation task required a production Gradient Boosting Quantile Regression VaR wrapper using direct quantile loss at alpha 0.05.

G01 was locked before validation performance using 100 estimators, learning rate 0.05, maximum depth 2, minimum samples per leaf 5, full-sample boosting, and random state 42.

The implementation was required to provide fit, quantile prediction, batch VaR prediction, and single-row forecast interfaces; preserve the common `quantile_return` and `var` contract; enforce `VaR=max(0,-q)`; validate finite numeric inputs; reject X/y index misalignment; preserve fitted feature schema; and demonstrate fixed-seed reproducibility.
