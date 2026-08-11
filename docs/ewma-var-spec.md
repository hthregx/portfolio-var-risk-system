# EWMA VaR Specification

## Scope

This specification defines the one-day EWMA VaR model used for the portfolio VaR system. The model receives a one-dimensional history of portfolio returns observed before the target date and produces a one-day lower-tail return quantile and VaR forecast.

The model is responsible only for numerical forecasting and input validation. Dates, rolling-window selection, realized target returns, violations, runtime measurement and file I/O belong to the backtesting runner.

## Canonical parameters

- Forecast horizon: one trading day
- Confidence level: 95%
- Alpha: 0.05
- EWMA decay factor lambda: 0.94
- Conditional mean assumption: zero
- Conditional innovation distribution: standard Normal

The canonical decay factor is fixed at 0.94 for the baseline EWMA model. It is not tuned on the evaluation period.

## Variance recursion

Let r_t denote the observed portfolio return at time t.

The EWMA conditional variance recursion is

sigma_t^2 = lambda * sigma_(t-1)^2 + (1 - lambda) * r_t^2

with lambda = 0.94.

The initialization rule is

sigma_1^2 = r_1^2

For an input history r_1, ..., r_T, the one-day-ahead variance forecast is obtained by initializing with r_1^2 and recursively updating with each subsequent observed return through r_T.

Only returns available before the target date may enter the recursion.

## Quantile and VaR

Under the zero-mean Normal assumption,

q_alpha = z_alpha * sigma

where z_alpha is the alpha quantile of the standard Normal distribution and sigma is the EWMA one-day-ahead volatility forecast.

For alpha = 0.05, z_alpha is approximately -1.64485362695147.

The reported VaR follows the common project sign convention

VaR = max(0, -q_alpha)

Therefore `quantile_return` is a lower-tail return forecast and `var` is reported as a non-negative loss magnitude.

## Production function contract

The planned production function is

`ewma_var_forecast(returns, alpha=0.05, decay=0.94)`

Inputs:

- `returns`: one-dimensional, non-empty, numeric and finite return history
- `alpha`: numeric value strictly between 0 and 1
- `decay`: numeric value strictly between 0 and 1

Output:

`{"quantile_return": float, "var": float}`

The function must not mutate the caller's input.

## Validation requirements

The implementation must reject:

- empty return histories
- multidimensional return inputs
- non-numeric returns
- NaN or infinite returns
- non-numeric alpha
- alpha outside the open interval (0, 1)
- non-numeric decay
- decay outside the open interval (0, 1)

## No-look-ahead requirement

For a target return at date t, the EWMA forecast must be constructed exclusively from returns dated before t. The model itself does not manage dates; enforcement of the training window and target alignment belongs to the common walk-forward runner.

## Baseline status

The canonical EWMA baseline uses lambda = 0.94 and the zero-mean Normal conditional distribution defined above. Alternative decay factors or distributional assumptions are outside the baseline specification and must not replace the canonical configuration through evaluation-period tuning.
