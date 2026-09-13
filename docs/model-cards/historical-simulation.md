# Historical Simulation VaR Model Card

## Model Summary

Historical Simulation is the non-parametric baseline used in the frozen
portfolio VaR system.

The model estimates the one-day 95% Value at Risk of the equal-weight
HPG/FPT/MWG portfolio from the empirical distribution of recent portfolio
simple returns.

Frozen configuration:

- Method: Historical Simulation
- Config ID: `historical_w250`
- Confidence level: 95%
- Alpha: 0.05
- Forecast horizon: 1 trading day
- Rolling window: 250 observations
- Target: `portfolio_simple_return`
- Portfolio: equal-weight HPG/FPT/MWG

The Day-23 release freezes this specification. The model is not retuned or
reselected during release hardening.

## Intended Use

The model provides a transparent benchmark for one-day portfolio downside
risk forecasting.

It is intended to:

- provide an interpretable VaR baseline;
- support comparison with EWMA and Gradient Boosting Quantile Regression;
- produce daily one-step-ahead VaR estimates under the common evaluation
  contract;
- support backtesting and model-risk analysis.

It is not intended to be interpreted as a complete measure of portfolio risk
or as a stand-alone trading or investment decision system.

## Method

For each forecast target date, the model uses only information available
before that target date.

The canonical implementation applies a rolling window of 250 historical
portfolio simple returns and estimates the lower-tail empirical quantile at
alpha 0.05.

The corresponding positive VaR magnitude follows the project convention:

`VaR = max(0, -quantile_return)`

A violation is defined strictly as:

`actual_return < quantile_return`

The rolling construction prevents future target returns from entering the
forecast window.

## Data Contract

The canonical portfolio contains:

- HPG
- FPT
- MWG

Each constituent receives equal weight.

Asset simple returns are combined into the portfolio simple return by the
equal-weight weighted sum. The VaR target is the portfolio simple return, not
the portfolio log return.

The canonical processed dataset contains 1,637 observations from 2020-01-03
through 2026-07-28.

## Frozen Evaluation Contract

The common final evaluation period is:

- Start: 2024-12-18
- End: 2026-07-28
- Forecast targets: 398
- Confidence level: 95%
- Forecast horizon: 1 trading day

This period was reserved from the documented validation-based parameter
selection. However, it must not be described as a pristine untouched test set
because the broader evaluation sample had been inspected previously.

All three frozen methods use the same target dates and the same realized
portfolio returns.

## Evaluation Results

On the frozen 398-target evaluation sample, Historical Simulation produced:

| Metric | Value |
| --- | ---: |
| Forecast count | 398 |
| Violations | 27 |
| Violation rate | 6.7839% |
| Pinball Loss | 0.001896464233 |
| Average VaR | 2.1330% |

The observed violation rate is above the nominal 5% tail probability.

Historical Simulation has the lowest Average VaR among the three frozen
methods on this evaluation sample.

This is a risk-magnitude comparison only. A lower Average VaR does not by
itself imply better calibration, better quantile accuracy, lower model risk,
or overall model superiority.

No overall winner is declared across the frozen methods.

## Strengths

Historical Simulation is simple, transparent, and directly tied to observed
portfolio returns.

It does not require a parametric return distribution or a fitted volatility
model. Its forecast can therefore be explained directly in terms of the
recent empirical return distribution.

The method also provides a useful reference point for evaluating whether more
complex models deliver materially different risk estimates.

## Limitations

The model assumes that the most recent 250 observations provide a useful
empirical representation of near-term downside risk.

Important limitations include:

- all observations inside the rolling window receive equal empirical weight;
- abrupt regime changes may not be reflected immediately;
- the method cannot extrapolate beyond losses represented in the historical
  window;
- results depend on the selected 250-observation window;
- the one-day 95% VaR estimate describes a quantile threshold and does not
  measure loss severity beyond that threshold;
- the evaluation evidence in this project is specific to the equal-weight
  HPG/FPT/MWG portfolio and should not be generalized automatically to other
  portfolios or periods.

## Model Risk and Interpretation

Historical Simulation should be interpreted alongside calibration, quantile
loss, VaR magnitude, temporal behavior, and the results of the other frozen
methods.

The model's lower Average VaR in this evaluation is not sufficient evidence
that it is safer or more accurate.

Likewise, the observed 6.7839% violation rate should be interpreted as an
evaluation result rather than a guarantee of future violation frequency.

## Reproducibility

Frozen model specification:

`configs/model_freeze.yaml`

Canonical evaluation configuration:

`configs/final_evaluation.yaml`

Canonical final predictions:

`results/final_predictions.csv`

Canonical final metrics:

`results/final_metrics.csv`

The frozen configuration ID is:

`historical_w250`

Day-23 validation is recorded in:

`results/model_freeze_validation_a.csv`

## Freeze Status

Status: **FROZEN**

After the Day-23 model freeze, no new algorithms, features, parameter tuning,
evaluation-period changes, or retroactive rewrites of the canonical final
predictions are permitted as part of release hardening.