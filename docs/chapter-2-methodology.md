# Chapter 2 — Methodology

## 2.1 Portfolio Return and Forecast Target

The study uses an equal-weight portfolio consisting of HPG, FPT, and MWG. With three assets, each portfolio weight is \(w_i=1/3\), with \(\sum_i w_i=1\).

The simple return of asset \(i\) at trading date \(t\) is:

\[ r_{i,t}=\frac{P_{i,t}}{P_{i,t-1}}-1. \]

The portfolio simple return is:

\[ R_{p,t}=\sum_{i=1}^{3}w_i r_{i,t}=\frac{r_{HPG,t}+r_{FPT,t}+r_{MWG,t}}{3}. \]

The variable `portfolio_simple_return` is the common modeling target for Historical Simulation, EWMA, and Gradient Boosting Quantile Regression.

The forecasting horizon is one trading day. For target date \(t+1\), a model may use only information available up to forecast origin \(t\).

The project uses a 95% confidence level, corresponding to \(\alpha=0.05\). If \(\hat q_{t+1}(0.05)\) is the predicted lower-tail return quantile, VaR is reported as:

\[ \widehat{VaR}_{t+1}=\max(0,-\hat q_{t+1}(0.05)). \]

Therefore the return quantile may be negative while the reported VaR is non-negative.

## 2.2 Historical Simulation VaR

### 2.2.1 Principle

Historical Simulation estimates downside risk directly from the empirical distribution of past portfolio returns and does not assume a parametric return distribution.

The canonical baseline uses a rolling window of \(W=250\) observations.

At forecast origin \(t\), the estimation sample is:

\[ \mathcal{R}_t=\{R_{p,t-W+1},\ldots,R_{p,t}\}. \]

The one-day lower-tail return forecast is:

\[ \hat q_{t+1}^{HS}(\alpha)=Q_{\alpha}(\mathcal{R}_t). \]

For the project baseline, \(\alpha=0.05\) and \(W=250\).

The production implementation computes the empirical quantile using `np.quantile(values, alpha, method="linear")`.

The interpolation rule is stated explicitly so that the methodology reproduces the numerical behavior of the production model.

### 2.2.2 Quantile-to-VaR Transformation

Historical Simulation first produces a lower-tail return quantile rather than a positive loss value.

The project converts this forecast into VaR using:

\[ \widehat{VaR}_{t+1}^{HS}=\max(0,-\hat q_{t+1}^{HS}). \]

A negative lower-tail return quantile therefore produces a positive VaR estimate.

If the predicted quantile is zero or positive, the reported VaR is floored at zero.

This convention preserves the distinction between `quantile_return` and the non-negative `var` output used by the production model.

### 2.2.3 Rolling Window

Historical Simulation is evaluated with a fixed rolling window of 250 portfolio-return observations.

For each new target date, the oldest observation leaves the estimation sample and the newest available observation enters it.

Therefore, the training-sample size remains constant:

\[ |\mathcal{R}_t| = 250. \]

The target observation itself is excluded from the estimation window.

A large new loss does not necessarily change Historical VaR immediately. The VaR forecast changes only when the composition or ordering of returns around the empirical 5% tail changes the estimated quantile.

This behavior differs from volatility-recursion methods such as EWMA and is a mechanical consequence of the rolling empirical-quantile estimator rather than evidence of superior or inferior model performance.

### 2.2.4 Input Validation

Before quantile estimation, the production Historical Simulation model validates the supplied return history.

The input returns must:

- contain numeric values;
- be one-dimensional;
- contain at least one observation;
- contain only finite values.

The quantile level must satisfy:

\[ 0 < \alpha < 1. \]

These checks belong to the model-level numerical contract. Date ordering, training-window construction, and prevention of look-ahead bias are enforced separately by the common walk-forward evaluation framework.

## 2.3 EWMA VaR

### 2.3.1 Model Assumptions

The EWMA baseline models time-varying portfolio volatility by assigning exponentially decreasing influence to older squared returns.

The canonical production configuration uses the decay factor:

\[ \lambda = 0.94. \]

The conditional mean is assumed to be zero and standardized innovations are modeled with a standard Normal distribution.

\[ Z_{t+1} \sim \mathcal{N}(0,1). \]

Under this specification, the next portfolio return is represented as:

\[ R_{p,t+1} = \sigma_t Z_{t+1}. \]

The model therefore estimates the next lower-tail return quantile from the current EWMA volatility estimate and the Normal quantile at \(\alpha=0.05\).

These assumptions correspond to the canonical EWMA production configuration and are not changed using evaluation-period performance.

### 2.3.2 Initialization and Variance Recursion

For an input history \(R_{p,1},R_{p,2},\ldots,R_{p,T}\), the production EWMA implementation initializes the conditional variance using the first squared return:

\[ \sigma_1^2 = R_{p,1}^2. \]

For each subsequent return, the variance is updated recursively as:

\[ \sigma_t^2 = \lambda\sigma_{t-1}^2 + (1-\lambda)R_{p,t}^2,\qquad t=2,\ldots,T. \]

With the canonical decay factor \(\lambda=0.94\), the recursion becomes:

\[ \sigma_t^2 = 0.94\sigma_{t-1}^2 + 0.06R_{p,t}^2. \]

The recursion is applied through the final return available in the supplied training history.

The volatility estimate used for the next one-day forecast is:

\[ \sigma_T = \sqrt{\sigma_T^2}. \]

This definition matches the production implementation, which rebuilds the EWMA state from the complete return history supplied to each model call.

### 2.3.3 Lower-Tail Quantile Forecast

Let \(z_{\alpha}\) denote the alpha quantile of the standard Normal distribution:

\[ z_{\alpha} = \Phi^{-1}(\alpha). \]

For \(\alpha=0.05\):

\[ z_{0.05} \approx -1.64485362695147. \]

The EWMA lower-tail return forecast is:

\[ \hat q_{T+1}^{EWMA} = z_{\alpha}\sigma_T. \]

The corresponding VaR is:

\[ \widehat{VaR}_{T+1}^{EWMA} = \max(0,-\hat q_{T+1}^{EWMA}). \]

The production implementation obtains the Normal quantile using `NormalDist().inv_cdf(alpha)` rather than a hard-coded z-score.

Therefore, the model output remains consistent with the project-wide distinction between `quantile_return` and non-negative `var`.

### 2.3.4 Expanding-History Evaluation

The canonical EWMA evaluation uses expanding history within the common walk-forward framework.

Forecasting begins after 250 portfolio-return observations are available. For the first target date, the EWMA model therefore receives 250 historical returns.

For each subsequent target date, the training history begins at the start of the available portfolio-return series and ends at the forecast date.

If the target observation has index \(t\), the expanding training sample contains observations with indices:

\[ 0,\ldots,t-1. \]

The target observation at index \(t\) is excluded from model input.

Consequently, the EWMA training-history size increases by one observation as the walk-forward evaluation advances.

This differs from the Historical Simulation baseline, which keeps a fixed rolling window of 250 observations.

The expanding design is compatible with the production EWMA function because the EWMA state is reconstructed from the complete return history supplied before each target date.

### 2.3.5 Response to New Volatility

EWMA assigns greater influence to recent squared returns than to older observations.

In the variance recursion, the newest squared return enters with weight:

\[ 1-\lambda. \]

For the canonical decay factor \(\lambda=0.94\), this weight is:

\[ 1-\lambda = 0.06. \]

Therefore, a large recent absolute return can increase the EWMA variance estimate and affect the next one-day VaR forecast immediately.

Older observations remain in the volatility estimate, but their influence decreases exponentially through repeated multiplication by \(\lambda\).

This response mechanism differs from Historical Simulation, where a new observation changes VaR only when it changes the empirical lower-tail quantile of the rolling window.

The distinction describes model mechanics only and does not imply that either method is universally superior.

### 2.3.6 Input and Parameter Validation

Before applying the EWMA recursion, the production model validates the supplied return history and model parameters.

The input returns must:

- contain numeric values;
- be one-dimensional;
- contain at least one observation;
- contain only finite values.

The quantile level must satisfy:

\[ 0 < \alpha < 1. \]

The EWMA decay factor must satisfy:

\[ 0 < \lambda < 1. \]

The canonical production configuration uses \(\alpha=0.05\) and \(\lambda=0.94\).

These checks belong to the model-level numerical contract. Date alignment, training-history construction, and exclusion of the target observation are enforced by the common walk-forward framework.

## 2.4 Walk-Forward Backtesting Framework

Historical Simulation and EWMA are evaluated through a common one-day-ahead walk-forward framework.

For each target index \(t\), the training sample contains only observations strictly before the target observation.

In rolling mode, the training sample is:

\[ \{R_{p,t-W},\ldots,R_{p,t-1}\}, \]

where the canonical Historical Simulation window is \(W=250\).

In expanding mode, the training sample begins at the start of the available return series and ends at \(t-1\). This mode is used for the canonical EWMA evaluation.

In implementation terms, the common runner constructs the training frame using rows before `target_index`, while the realized target is stored separately at `target_index`.

The last date in the training sample is recorded as `forecast_date`, and the realized observation date is recorded as `target_date`.

The runner enforces the ordering:

\[ forecast\_date < target\_date. \]

Each model is called only with its training-return history and must return `quantile_return` and `var`.

The common output schema records the forecast date, target date, realized return, predicted quantile, VaR, violation indicator, model name, number of observations, and evaluation mode.

Using a shared runner keeps date alignment, target definition, and prediction schema consistent across baseline models while leaving model-specific forecasting logic inside each model implementation.

## 2.5 Evaluation Metrics

All forecasting methods are evaluated using the same target returns, target dates, sign convention, and metric definitions.

### 2.5.1 VaR Violation

Let \(R_{p,t}\) be the realized portfolio return and \(\hat q_t(\alpha)\) the predicted lower-tail return quantile for the same target date.

A VaR violation occurs only when:

\[ R_{p,t} < \hat q_t(\alpha). \]

The violation indicator is therefore:

\[ I_t = \mathbf{1}\{R_{p,t}<\hat q_t(\alpha)\}. \]

The inequality is strict. If the realized return is exactly equal to the predicted quantile, the observation is not counted as a violation.

### 2.5.2 Violation Rate

For \(N\) forecasts, Violation Rate is:

\[ \mathrm{Violation\ Rate}=\frac{1}{N}\sum_{t=1}^{N} I_t. \]

At \(\alpha=0.05\), the nominal expected violation rate is 5%. The observed rate is interpreted descriptively relative to this nominal level.

### 2.5.3 Pinball Loss

Quantile-forecast accuracy is evaluated using Pinball Loss at \(\alpha=0.05\). Let:

\[ u_t = R_{p,t}-\hat q_t(\alpha). \]

The quantile loss for one forecast is:

\[ \rho_{\alpha}(u_t)=(\alpha-\mathbf{1}\{u_t<0\})u_t. \]

Equivalently:

\[
\rho_{\alpha} =
\begin{cases}
\alpha(R_{p,t}-\hat q_t), & R_{p,t}\geq\hat q_t,\\
(1-\alpha)(\hat q_t-R_{p,t}), & R_{p,t}<\hat q_t.
\end{cases}
\]

The reported Pinball Loss is the mean loss across all forecast dates. Lower values indicate better quantile-forecast accuracy on the evaluated sample.

Pinball Loss is computed from `quantile_return`, not from the non-negative `var` field.

### 2.5.4 Average VaR

Average VaR summarizes the mean forecast risk level:

\[ \mathrm{Average\ VaR}=\frac{1}{N}\sum_{t=1}^{N}\widehat{VaR}_t. \]

Average VaR is a descriptive measure of forecast magnitude. A lower Average VaR alone is not treated as evidence of a better model.

### 2.5.5 Exception Days

Exception days are the target dates for which the strict violation indicator equals one.

These dates are retained for temporal inspection, overlap analysis, clustering analysis, and model case studies.

The primary comparison therefore reports Violation Rate, Pinball Loss, Average VaR, and exception-day behavior on the same evaluation dates.

## 2.6 Baseline Configuration and Parameter-Selection Policy

The baseline comparison uses a deliberately small parameter set rather than an unrestricted search over the evaluation period.

For Historical Simulation, the canonical configuration is:

- rolling window: 250 observations;
- alpha: 0.05;
- one-day forecast horizon.

Sensitivity analysis also considered Historical windows of 125 and 500 observations. The 250-observation configuration is retained as the canonical Historical baseline.

For EWMA, the canonical production configuration is:

- decay factor: 0.94;
- alpha: 0.05;
- zero conditional mean;
- standard Normal innovation distribution;
- first-squared-return variance initialization;
- one-day forecast horizon.

The sensitivity analysis compared decay values 0.90, 0.94, and 0.97 using validation evidence.

Among these candidates, decay 0.90 produced the strongest validation result according to the predefined calibration and Pinball Loss criteria.

However, this validation result does not imply that the production default was automatically changed. Decay 0.94 remains the canonical EWMA baseline pending a separate traceable configuration decision.

Average VaR and the observed VaR range are treated as secondary descriptive evidence and are not used alone for parameter selection.

The later evaluation subset is not described as a pristine untouched test set because the broader evaluation period had already been inspected during the earlier Historical-versus-EWMA comparison.

This distinction separates validation-based candidate assessment from the retained production baseline and avoids overstating test-set independence.

## 2.7 No-Look-Ahead and Reproducibility Controls

The evaluation framework is designed so that each one-day-ahead forecast uses only information available before its target observation.

For every forecast, the training sample ends at `forecast_date`, while the realized return belongs to the later `target_date`.

The common runner enforces:

\[ forecast\_date < target\_date. \]

The target row is excluded from the training frame before the model forecast is generated.

Historical Simulation and EWMA are compared on the same target-date sequence and use the same realized `portfolio_simple_return` values for corresponding dates.

The violation rule is applied only after the realized target return is observed:

\[ actual\_return < quantile\_return. \]

A future-perturbation audit is used as an additional leakage check. Returns occurring after an audited forecast origin are modified, and forecasts generated before those future observations must remain unchanged.

The baseline Historical Simulation and EWMA methods are deterministic for fixed inputs and fixed configuration values. Their reported results can therefore be reproduced from the production model implementations, the common walk-forward contract, and the recorded configuration artifacts.

Reproducibility evidence is maintained through source-controlled model code, configuration files, tests, canonical result artifacts, and methodology documentation.

These controls support implementation-level no-look-ahead verification. They do not by themselves establish statistical superiority of any forecasting method.

## Implementation Traceability

The methodology in this chapter is mapped directly to the production implementation and supporting technical evidence.

Primary implementation references are:

- `src/models/historical_var.py` for Historical Simulation quantile estimation and VaR conversion;
- `src/models/ewma_var.py` for EWMA initialization, variance recursion, Normal quantile estimation, and VaR conversion;
- `src/backtesting/walk_forward.py` for rolling/expanding training construction, date ordering, prediction schema, and strict violation semantics;
- `configs/ewma.yaml` for the canonical EWMA configuration.

Supporting methodological and validation references include:

- `docs/portfolio-methodology.md`;
- `docs/historical-simulation-notes.md`;
- `docs/historical-var-stability.md`;
- `docs/ewma-var-spec.md`;
- `docs/ewma-evaluation.md`;
- `docs/walk-forward-runner.md`;
- `docs/baseline-sensitivity-decision.md`;
- `docs/baseline-decision-log.md`;
- `docs/date-target-leakage-audit.md`;
- `docs/leakage-audit.md`.

If production model or evaluation contracts change later, the corresponding methodology statements must be reviewed so that the report remains synchronized with the implemented system.

## Prompt Provenance

This chapter was drafted with AI assistance under constraints derived from the verified production implementation and repository evidence.

The drafting prompt locked the common target `portfolio_simple_return`, one-day forecast horizon, alpha 0.05, Historical rolling window 250, NumPy linear quantile convention, EWMA decay 0.94, zero conditional mean, standard Normal distribution, first-squared-return initialization, and the project-wide `VaR=max(0,-q)` sign convention.

The prompt also required strict violation semantics, no-look-ahead alignment, separation between validation-selected candidates and canonical production settings, and avoidance of unsupported claims of statistical or universal model superiority.

The drafting step was restricted to methodology documentation and did not modify production source code, configuration files, tests, or canonical result artifacts.
