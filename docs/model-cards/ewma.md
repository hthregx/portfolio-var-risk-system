# EWMA VaR Model Card

## Model Summary

Exponentially Weighted Moving Average (EWMA) is the parametric volatility
baseline used in the frozen portfolio VaR system.

The model estimates one-day 95% Value at Risk for the equal-weight
HPG/FPT/MWG portfolio using an exponentially weighted conditional variance
process.

Frozen configuration:

- Method: EWMA
- Config ID: `ewma_d094`
- Confidence level: 95%
- Alpha: 0.05
- Forecast horizon: 1 trading day
- Decay factor: 0.94
- Updating mode: expanding
- Distribution: Normal
- Mean assumption: zero
- Variance initialization: first squared return
- Target: `portfolio_simple_return`
- Portfolio: equal-weight HPG/FPT/MWG

The Day-23 release freezes this already evaluated specification. Parameter
selection is not reopened during release hardening.

## Intended Use

EWMA provides a volatility-responsive baseline for one-day portfolio downside
risk forecasting.

It is intended to:

- provide an interpretable volatility-based VaR baseline;
- respond to recent changes in return volatility;
- support comparison with Historical Simulation and Gradient Boosting
  Quantile Regression;
- support common-date backtesting and model-risk analysis.

It is not a complete measure of portfolio risk and should not be used as a
stand-alone trading or investment decision system.

## Method

The frozen model uses decay factor:

`lambda = 0.94`

The conditional variance recursion is conceptually:

`sigma_t^2 = lambda * sigma_(t-1)^2 + (1 - lambda) * r_(t-1)^2`

The model assumes a zero conditional mean and a Normal forecast distribution.

Variance is initialized using the first squared portfolio simple return.

The forecast is one-step-ahead and uses only information available before the
target date.

The project converts the lower-tail quantile into positive VaR using:

`VaR = max(0, -quantile_return)`

A violation is defined strictly as:

`actual_return < quantile_return`

## Data Contract

The canonical portfolio contains:

- HPG
- FPT
- MWG

The constituents receive equal weights.

The VaR target is `portfolio_simple_return`.

The canonical processed dataset contains 1,637 observations from 2020-01-03
through 2026-07-28.

## Frozen Evaluation Contract

The common evaluation period is:

- Start: 2024-12-18
- End: 2026-07-28
- Forecast targets: 398
- Confidence level: 95%
- Forecast horizon: 1 trading day

This period was reserved from the documented validation-based parameter
selection, but it must not be described as a pristine untouched test set
because the broader evaluation sample had been inspected previously.

All frozen methods use the same target dates and realized portfolio returns.

## Evaluation Results

On the frozen 398-target evaluation sample, EWMA produced:

| Metric | Value |
| --- | ---: |
| Forecast count | 398 |
| Violations | 22 |
| Violation rate | 5.5276% |
| Pinball Loss | 0.001968321253 |
| Average VaR | 2.5084% |

EWMA has the violation rate closest to the nominal 5% tail probability among
the three frozen methods on this evaluation sample.

This is a calibration-specific observation and does not establish EWMA as the
overall best or most accurate method.

EWMA also has the highest aggregate Pinball Loss among the three frozen
methods on this evaluation sample.

No overall winner is declared across the frozen methods.

## Decay-Factor Selection Context

The frozen canonical decay factor is `0.94`.

A validation alternative using decay factor `0.90` produced stronger
validation evidence.

Decay factor `0.90` was not adopted for the canonical final evaluation.

The retained `0.94` value reflects continuity with the canonical baseline
used for the existing final evaluation. The Day-23 freeze must not be
interpreted as evidence that `0.94` outperformed `0.90` during validation.

Model freeze locks the already evaluated canonical configuration rather than
reopening parameter selection.

## Strengths

EWMA gives greater effective weight to recent squared returns and can respond
more quickly to changing volatility than an equally weighted historical
window.

The model is computationally simple and its recursive variance process is
straightforward to interpret and reproduce.

## Limitations

Important limitations include:

- the Normal assumption may not fully represent heavy tails or asymmetry;
- the conditional mean is fixed at zero;
- forecasts depend materially on the decay factor;
- volatility dynamics are represented by a single recursive variance process;
- the model does not directly use nonlinear or external market features;
- one-day 95% VaR does not measure loss severity beyond the quantile threshold;
- project evidence is specific to the equal-weight HPG/FPT/MWG portfolio and
  evaluation period.

## Model Risk and Interpretation

The observed 5.5276% violation rate is close to the nominal 5% level, but
violation frequency is only one dimension of model performance.

The higher aggregate Pinball Loss illustrates why calibration frequency alone
must not be used to declare model superiority.

Evaluation should consider calibration, quantile loss, VaR magnitude,
temporal behavior, assumptions, and limitations together.

## Reproducibility

Frozen model specification:

`configs/model_freeze.yaml`

Canonical evaluation configuration:

`configs/final_evaluation.yaml`

Canonical final predictions:

`results/final_predictions.csv`

Canonical final metrics:

`results/final_metrics.csv`

Frozen configuration ID:

`ewma_d094`

Day-23 validation record:

`results/model_freeze_validation_a.csv`

## Freeze Status

Status: **FROZEN**

After the Day-23 model freeze, no new algorithms, features, parameter tuning,
evaluation-period changes, or retroactive rewrites of canonical final
predictions are permitted as part of release hardening.