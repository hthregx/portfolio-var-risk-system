# Gradient Boosting Return-History Feature Contract

## Purpose

This module provides the return-history feature family for the Gradient Boosting Quantile Regression VaR pipeline.

It consumes the canonical equal-weight portfolio simple-return series produced upstream. It does not reconstruct HPG, FPT, and MWG portfolio returns.

## Canonical Input

The production entry point is:

`build_return_features(portfolio_return)`

The input must be a non-empty pandas Series satisfying all of the following conditions:

- series name: `portfolio_simple_return`;
- index type: `pandas.DatetimeIndex`;
- dates are unique;
- dates are sorted in strictly non-decreasing chronological order with no duplicate timestamps;
- values are numeric and finite;
- no missing returns are allowed;
- simple returns must be greater than `-1.0`.

The function copies validated values and does not mutate the caller's input.

## Temporal Alignment

Every output row is indexed by a target date `T`.

Features attached to target date `T` may use only portfolio returns observed strictly before `T`.

Therefore:

`return_lag_1(T) = R(T-1 observation)`

and never the realized return at target date `T`.

This convention keeps the target return outside its own feature vector and is the primary no-look-ahead contract of this feature family.

Windows are defined in numbers of available return observations, not calendar days.

## Feature Set

The canonical A-17 return-history feature set is:

- `return_lag_1`;
- `return_lag_2`;
- `return_lag_5`;
- `rolling_vol_5`;
- `rolling_vol_20`;
- `rolling_vol_60`;
- `drawdown`.

### Lagged Returns

For lag `k`:

`return_lag_k(T) = portfolio_simple_return.shift(k)`

with canonical lags 1, 2, and 5 observations.

### Rolling Volatility

The return series is shifted by one observation before rolling statistics are calculated.

For a window of size `w`:

`rolling_vol_w(T) = std(R[T-w], ..., R[T-1])`

using:

- windows 5, 20, and 60 observations;
- `ddof=1`;
- `min_periods=window`.

Thus the target-date return is never included in its own rolling-volatility feature.

### Drawdown

A wealth index is constructed from simple returns:

`W_t = cumulative_product(1 + R_t)`

The contemporaneous drawdown is:

`DD_t = W_t / running_max(W_t) - 1`

The production feature is then shifted by one observation:

`drawdown(T) = DD(T-1 observation)`

so target-date return information is excluded.

## Warm-Up Policy

Warm-up observations remain `NaN` by design.

No zero filling, backward filling, forward filling, or other artificial imputation is performed inside this module.

Examples:

- the first `return_lag_1` value is missing;
- the first 5 `rolling_vol_5` rows are missing;
- the first 20 `rolling_vol_20` rows are missing;
- the first 60 `rolling_vol_60` rows are missing;
- the first `drawdown` value is missing.

Downstream training code is responsible for selecting rows with sufficient feature history.

## Output Contract

The function returns a pandas DataFrame:

- with the same DatetimeIndex as the input Series;
- with seven feature columns in canonical order;
- with `float64` feature dtype;
- with intentional warm-up missing values only.

Canonical column order:

1. `return_lag_1`
2. `return_lag_2`
3. `return_lag_5`
4. `rolling_vol_5`
5. `rolling_vol_20`
6. `rolling_vol_60`
7. `drawdown`

## No-Look-Ahead Controls

The module is protected by explicit tests covering:

- exact lag arithmetic;
- rolling-window arithmetic using only prior returns;
- shifted drawdown;
- target-return exclusion;
- future-data perturbation;
- input immutability;
- deterministic repeated execution.

The future-perturbation check changes future returns and verifies that feature rows at or before the perturbation boundary remain unchanged.

## Validation and Failure Policy

The module fails explicitly rather than silently correcting structurally invalid inputs.

It rejects:

- non-Series inputs;
- empty Series;
- incorrect target name;
- non-DatetimeIndex inputs;
- duplicate dates;
- unsorted dates;
- missing returns;
- infinite returns;
- non-numeric returns;
- simple returns less than or equal to `-1.0`.

Chronological sorting is not performed inside the feature builder because silently sorting malformed modeling input could hide an upstream alignment error.

## Determinism and Reproducibility

The feature transformation contains no stochastic operations.

For the same ordered input Series, repeated calls must produce the same DataFrame.

The accompanying audit artifact is stored at:

`results/gb_return_feature_audit.json`

and the implementation tests are stored at:

`tests/test_gb_return_features.py`.

## Scope Boundary

This A-17 module owns only portfolio return-history features.

It does not implement:

- price-range features;
- trading-volume features;
- VN-Index or other market features;
- the Gradient Boosting estimator;
- model tuning;
- final feature selection.

Those concerns remain outside this module so the feature families can be developed and tested independently.

## Implementation Traceability

Primary implementation:

`src/gb_return_features.py`

Canonical upstream portfolio-return implementation:

`src/portfolio.py`

Canonical sample return data:

`data/sample/portfolio_returns_sample.csv`

Tests:

`tests/test_gb_return_features.py`

## Prompt Provenance

The implementation task specified a Gradient Boosting return-history feature family built from canonical `portfolio_simple_return`, with target-date alignment, no use of target or future returns, lag features 1/2/5, rolling volatility windows 5/20/60 using `ddof=1`, shifted drawdown, intentional warm-up missing values, deterministic behavior, explicit validation, and future-perturbation leakage tests.
