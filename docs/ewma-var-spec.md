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


## API design

The baseline uses the stateless model interface:

`ewma_var_forecast(returns, alpha=0.05, decay=0.94)`

This is compatible with the common walk-forward contract:

`forecast_function(training_returns)`

A persistent `fit/update/predict` object is not required for the baseline.
If a stateful API is introduced later, it must remain numerically
consistent with this specification.

## State lifecycle

At each forecast origin:

1. the runner provides only returns available before the target date;
2. variance is initialized from the first squared return;
3. EWMA recursion is applied through the final training return;
4. the model produces `quantile_return` and `var`;
5. the target return is observed after the forecast.

The baseline therefore reconstructs EWMA state from the supplied training
history at each forecast origin and does not keep hidden state between
runner calls.

## Runner compatibility

The EWMA model provides:

- `quantile_return`
- `var`

The common runner adds:

- `forecast_date`
- `target_date`
- `actual_return`
- `violation`
- `method`
- evaluation metadata

The common violation rule remains:

`actual_return < quantile_return`

`sigma` may be added later as EWMA-specific metadata, but it is not
required by the core prediction schema.

## Hand-calculation example

The hand-calculation oracle uses:

```text
returns = [-0.020, 0.010, -0.030, 0.015]
lambda = 0.94
alpha = 0.05
z_0.05 = -1.64485362695147
```

Assumptions:

- first-squared-return initialization
- zero conditional mean
- standard Normal distribution

Initialization:

```text
sigma_1^2 = (-0.020)^2
          = 0.0004000000
```

EWMA recursion:

```text
sigma_t^2
= 0.94 * sigma_(t-1)^2
+ 0.06 * r_t^2
```

Hand calculation:

| Step | Return | Variance Before | Variance After |
|---|---:|---:|---:|
| 1 | -0.020 | - | 0.0004000000 |
| 2 | 0.010 | 0.0004000000 | 0.0003820000 |
| 3 | -0.030 | 0.0003820000 | 0.0004130800 |
| 4 | 0.015 | 0.0004130800 | 0.0004017952 |

For the final variance state:

```text
sigma
= sqrt(0.0004017952)
= 0.0200448298

quantile_return
= z_0.05 * sigma
= -1.64485362695147 * 0.0200448298
= -0.0329708109

var
= max(0, -quantile_return)
= 0.0329708109
```

Final one-day-ahead forecast:

```text
sigma           = 0.0200448298
quantile_return = -0.0329708109
var             = 0.0329708109
```

This example provides the numerical oracle for the EWMA implementation
tests. The final values should match the production implementation within
the project's numerical tolerance.

## Unit-test plan

Existing EWMA tests already cover:

- numerical forecast values;
- default `lambda = 0.94`;
- first-squared-return initialization;
- VaR zero-floor behavior;
- input immutability;
- empty, NaN and infinite returns;
- multidimensional and non-numeric returns;
- invalid alpha;
- invalid decay.

Additional tests planned for implementation review:

- explicit one-step recursion test;
- explicit multi-step recursion test;
- sigma non-negative behavior;
- VaR sign convention;
- zero-volatility behavior;
- no-look-ahead integration with the common runner;
- EWMA/common prediction-schema compatibility.

Serialization testing is required only if persistent model state is introduced.


## Runner compatibility

The current EWMA model is compatible with the common walk-forward runner
without changing the runner contract.

EWMA provides the required forecast fields:

- `quantile_return`
- `var`

The common runner then generates:

- `forecast_date`
- `target_date`
- `actual_return`
- `quantile_return`
- `var`
- `violation`
- `method`

The runner computes violations using:

`actual_return < quantile_return`

EWMA volatility (`sigma`) is method-specific metadata and is not required
by the core prediction schema. If exposed later, it should be added as
optional metadata without changing the required common fields.