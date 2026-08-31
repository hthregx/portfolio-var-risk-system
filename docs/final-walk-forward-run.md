# Final Walk-Forward Evaluation

## Purpose

This document describes the reproducible Day-21 common evaluation for the
portfolio VaR project.

The evaluation compares three one-day 95% VaR forecasting methods on the same
target-date universe:

- Historical Simulation
- EWMA
- Gradient Boosting Quantile Regression

The target is the equal-weight HPG/FPT/MWG portfolio simple return.

This evaluation is intended for model comparison and robustness analysis. The
evaluation period was reserved from parameter selection, but it must not be
described as a pristine untouched test set because the broader evaluation
period had already been inspected earlier in the project.

---

## Evaluation Contract

The canonical contract is stored in:

```text
configs/final_evaluation.yaml
```

Core settings:

| Setting | Value |
|---|---|
| Confidence level | 95% |
| Quantile alpha | 0.05 |
| Forecast horizon | 1 trading day |
| Target | `portfolio_simple_return` |
| Portfolio | Equal-weight HPG/FPT/MWG |
| Evaluation start | 2024-12-18 |
| Evaluation end | 2026-07-28 |
| Evaluation targets | 398 |
| Violation rule | `actual_return < quantile_return` |
| VaR convention | `max(0, -quantile_return)` |

Every method must forecast the same 398 target dates.

For every forecast:

```text
forecast_date < target_date
```

No same-day target information may enter the forecast.

---

## Method Configurations

### Historical Simulation

Canonical configuration:

```text
window = 250
mode = rolling
alpha = 0.05
```

The forecast uses only the 250 observations available before each target date.

### EWMA

Canonical configuration:

```text
decay = 0.94
mode = expanding
alpha = 0.05
initialization = first_squared_return
distribution = normal
mean_assumption = zero
```

A decay of `0.90` produced stronger validation evidence in the earlier
sensitivity analysis, but it is not adopted for this Day-21 canonical run.

The retained `0.94` configuration represents the established canonical
baseline and should not be described as the validation winner.

### Gradient Boosting Quantile Regression

Day-21 configuration:

```text
experiment_id = G04
loss = quantile
alpha = 0.05
n_estimators = 100
learning_rate = 0.03
max_depth = 2
min_samples_leaf = 5
subsample = 1.0
random_state = 42
training_protocol = expanding_refit_each_target
```

Features:

```text
return_lag_1
return_lag_2
return_lag_5
rolling_vol_5
rolling_vol_20
rolling_vol_60
drawdown
```

G04 is the selected candidate from the A-side validation slice.

It must not be described as the final G01-G07 tuning winner unless the
independent B-side G05-G07 evidence has been integrated and reviewed.

The following feature families are not implemented in the current canonical
Gradient Boosting pipeline:

```text
price_range
trading_volume
market_index
```

They must not be claimed as model inputs.

---

## Walk-Forward Design

All three methods are evaluated chronologically.

The common evaluation period contains 398 target dates from:

```text
2024-12-18
```

through:

```text
2026-07-28
```

The portfolio return history is never shuffled.

For every target date, only information available strictly before that target
date may be used for forecasting.

The prediction contract therefore requires:

```text
forecast_date < target_date
```

### Historical Simulation

Historical Simulation uses a rolling window of 250 prior portfolio returns.

For target date \(t\), the forecast is based on the 250 observations ending at
the previous available trading date.

### EWMA

EWMA uses expanding historical information with the canonical decay parameter:

```text
lambda = 0.94
```

The conditional volatility forecast is converted to the lower 5% return
quantile under the configured Normal zero-mean assumption.

### Gradient Boosting

Gradient Boosting uses an expanding refit protocol.

For each target date:

1. construct the feature-target history;
2. keep only rows strictly earlier than the target date;
3. fit a fresh Gradient Boosting Quantile Regression model;
4. build the feature row for the target date from prior information;
5. generate the 5% conditional return quantile.

The first evaluation target uses:

```text
1179
```

feature-ready training observations.

The final evaluation target uses:

```text
1576
```

feature-ready training observations.

The training count increases chronologically by one observation per evaluation
target.

---

## Return and VaR Conventions

The evaluated target is:

```text
portfolio_simple_return
```

The one-day lower-tail quantile forecast is stored as:

```text
quantile_return
```

For example, a forecast of:

```text
quantile_return = -0.025
```

means the estimated 5% portfolio-return quantile is -2.5%.

Reported positive VaR is defined as:

```text
var = max(0, -quantile_return)
```

Therefore the corresponding VaR would be:

```text
var = 0.025
```

A VaR violation is defined strictly as:

```text
actual_return < quantile_return
```

Equality is not counted as a violation.

This convention is used consistently in the walk-forward runner, exported
predictions, final metrics and automated regression tests.

---

## Reproducible Execution

Run commands from the repository root.

### Contract and data dry run

```powershell
python -m scripts.run_final_walk_forward --dry-run
```

The dry run validates:

- configuration loading;
- canonical portfolio-return data;
- evaluation boundaries;
- required Gradient Boosting feature history;
- common target-date availability;
- core model settings.

It does not fit forecasting models.

Expected high-level result:

```text
Evaluation targets    : 398
Historical window     : 250
EWMA decay            : 0.94
GB experiment         : G04
GB feature count      : 7
GB random state       : 42
Common universe       : PASS
Model fitting executed: NO
```

### Canonical full run

```powershell
python -m scripts.run_final_walk_forward
```

The full command:

1. runs Historical Simulation;
2. runs EWMA;
3. performs 398 expanding Gradient Boosting refits;
4. verifies common target-date alignment;
5. computes final metrics;
6. builds reproducibility metadata;
7. writes the canonical artifacts.

The normal module invocation is intentionally:

```powershell
python -m scripts.run_final_walk_forward
```

rather than:

```powershell
python scripts/run_final_walk_forward.py
```

so that the repository root is available for project package imports such as
`src`.

---

## Canonical Artifacts

The canonical run writes:

```text
results/final_predictions.csv
results/final_metrics.csv
results/final_run_metadata.json
```

### `final_predictions.csv`

Schema:

```text
forecast_date
target_date
method
actual_return
quantile_return
var
violation
runtime_seconds
config_id
```

Expected row count:

```text
398 targets x 3 methods = 1194 rows
```

The logical key is:

```text
(method, target_date)
```

and must be unique.

Each method must contain exactly:

```text
398
```

rows.

All methods must use identical target dates and identical realized
`actual_return` values.

`runtime_seconds` represents the total runtime returned by the corresponding
method adapter for the complete run.

The same method-level runtime is therefore repeated on each prediction row for
that method.

It is not per-forecast latency.

### `final_metrics.csv`

Schema:

```text
method
forecast_count
violation_count
violation_rate
pinball_loss
average_var
minimum_var
maximum_var
total_runtime_seconds
test_start
test_end
config_id
```

There is one row per forecasting method.

Metrics are computed from the common prediction table rather than by
duplicating or rerunning forecasting logic inside the metrics layer.

### `final_run_metadata.json`

Metadata records reproducibility information including:

- generation timestamp;
- Git commit;
- Git branch;
- Python version;
- relevant package versions;
- canonical data path;
- canonical data SHA256;
- evaluation-config path;
- evaluation-config SHA256;
- confidence level;
- quantile alpha;
- forecast horizon;
- portfolio definition;
- evaluation boundaries;
- method parameters;
- Gradient Boosting feature list;
- Gradient Boosting random seed;
- runtime semantics;
- output row counts.

The metadata intentionally records:

```text
model_frozen = false
pristine_untouched_test_claim = false
```

for this Day-21 run.

---

## Canonical Evaluation Results

All methods use the same 398 target dates from 2024-12-18 through 2026-07-28.

| Method | Forecasts | Violations | Violation Rate | Pinball Loss | Average VaR |
|---|---:|---:|---:|---:|---:|
| Historical Simulation | 398 | 27 | 0.067839195980 | 0.001896464233 | 0.021330189279 |
| EWMA | 398 | 22 | 0.055276381910 | 0.001968321253 | 0.025084042490 |
| Gradient Boosting G04 | 398 | 24 | 0.060301507538 | 0.001758130951 | 0.023349057036 |

The nominal violation rate at 95% confidence is:

```text
0.05
```

Absolute distance from the nominal violation rate:

| Method | Distance from 5% |
|---|---:|
| Historical Simulation | 0.017839195980 |
| EWMA | 0.005276381910 |
| Gradient Boosting G04 | 0.010301507538 |

---

## Metric Definitions

### Violation Count

A violation occurs when:

```text
actual_return < quantile_return
```

The violation count is the number of such observations in the evaluation
period.

### Violation Rate

Violation Rate is:

```text
violation_count / forecast_count
```

For a correctly calibrated 95% VaR model, the empirical rate should be
interpreted relative to the nominal rate:

```text
0.05
```

A violation rate close to 5% is useful calibration evidence, but violation rate
alone is not sufficient for selecting a final model.

### Pinball Loss

Pinball Loss evaluates quantile forecasts directly.

For quantile level:

```text
alpha = 0.05
```

and forecast error:

```text
error = actual_return - quantile_return
```

the observation-level quantile loss is:

```text
max(
    alpha * error,
    (alpha - 1) * error
)
```

The reported Pinball Loss is the mean of this loss across all evaluation
targets.

Lower Pinball Loss is better under this scoring rule.

### Average VaR

Average VaR is the mean positive VaR magnitude:

```text
mean(var)
```

over the common evaluation period.

Average VaR measures the typical magnitude of estimated downside risk.

Lower Average VaR is not automatically better because less conservative VaR
estimates may also produce more violations.

---

## Interpretation

The three reported metrics measure different aspects of VaR behaviour and
should not be collapsed into a single conclusion without an explicit model
selection rule.

### Violation Rate

EWMA is closest to the nominal 5% violation rate in this evaluation:

```text
EWMA                 5.5276%
Gradient Boosting    6.0302%
Historical           6.7839%
```

Absolute calibration distances are:

```text
EWMA                 0.005276381910
Gradient Boosting    0.010301507538
Historical           0.017839195980
```

This is evidence about empirical calibration.

It is not sufficient evidence by itself to declare EWMA the best model.

### Pinball Loss

Gradient Boosting G04 has the lowest Pinball Loss:

```text
Gradient Boosting    0.001758130951
Historical           0.001896464233
EWMA                 0.001968321253
```

Lower Pinball Loss indicates better quantile forecasting performance under the
quantile scoring rule used in this project.

This result supports G04 on the Pinball Loss criterion.

It does not by itself make G04 the final model winner.

### Average VaR

Historical Simulation produces the lowest Average VaR:

```text
Historical           0.021330189279
Gradient Boosting    0.023349057036
EWMA                 0.025084042490
```

Lower Average VaR is not automatically desirable.

A smaller risk estimate may be less conservative and may result in more
violations.

Average VaR therefore needs to be interpreted jointly with empirical
calibration and quantile forecast loss.

---

## Current Cross-Method Trade-Off

The Day-21 evidence can be summarized as:

```text
Calibration criterion       -> EWMA strongest
Pinball Loss criterion      -> Gradient Boosting G04 strongest
Average VaR                 -> Historical lowest
```

This is a descriptive comparison, not a final ranking.

The methods exhibit a genuine trade-off:

- EWMA is closest to the nominal violation frequency;
- Gradient Boosting G04 provides the lowest quantile scoring loss;
- Historical Simulation produces the lowest average VaR magnitude.

A final recommendation therefore requires an explicit decision rule and the
remaining review/model-freeze process.

---

## Runtime

Runtime is stored for reproducibility and engineering comparison.

The canonical A5.3 run recorded approximately:

| Method | Runtime |
|---|---:|
| Historical Simulation | 0.282 seconds |
| EWMA | 0.446 seconds |
| Gradient Boosting G04 | 106.313 seconds |

These values are machine- and run-dependent.

Gradient Boosting is substantially more expensive because the current
evaluation protocol refits the model separately for every target date.

The runtime difference therefore reflects both algorithmic complexity and the
evaluation implementation.

Runtime must not be interpreted as a predictive-quality metric.

Exact runtime values may vary across machines and repeated runs without
changing the forecasts or model-quality conclusions.

---

## Reproducibility Metadata

The canonical run records source provenance through SHA256 hashes.

For the Day-21 A5.3 artifact generation, the exported artifacts had the
following hashes:

```text
final_predictions.csv
1484fba7245688b66c0ab5a4ab9b37a6de7d420d7c9aed26a44bd90e994651f8

final_metrics.csv
c5a1b2328468cfee8ce523a9dde43c5ed82345a515ea9ec76252403bf353c1f6

final_run_metadata.json
ccb9bb25f56af40297a07f06172e11fa83e94531a54dbbea7f1cdc522d5d94c1
```

The metadata file also records hashes for:

```text
data/processed/portfolio_returns.csv
configs/final_evaluation.yaml
```

These source hashes allow a reviewer to determine whether the canonical input
data or evaluation configuration changed after the run.

Because runtime and generation timestamp are run-specific, rerunning the
pipeline can legitimately change the metadata artifact hash even when the
forecast configuration and numerical predictions remain unchanged.

---

## Validation Guarantees

The canonical artifacts have been checked for:

- exactly 1194 prediction rows;
- exactly 398 forecasts per method;
- exactly three forecasting methods;
- unique `(method, target_date)` pairs;
- identical target dates across methods;
- identical actual returns across methods;
- finite numerical output;
- positive method runtime;
- strict `forecast_date < target_date`;
- VaR sign-convention consistency;
- strict violation-rule consistency;
- stable method configuration identifiers;
- final metric schema;
- expected violation counts;
- metric recomputation from exported predictions;
- evaluation-boundary consistency;
- Git provenance;
- data/config SHA256 provenance;
- Gradient Boosting configuration metadata;
- Gradient Boosting feature-count consistency;
- runtime metadata consistency.

The canonical violation counts are:

```text
Historical Simulation    27
EWMA                     22
Gradient Boosting G04    24
```

---

## Automated Regression Tests

The Day-21 regression checks are located in:

```text
tests/test_final_walk_forward.py
```

Run them with:

```powershell
python -m pytest tests/test_final_walk_forward.py -q
```

The current targeted suite contains five tests covering:

- stable configuration IDs;
- final prediction schema;
- VaR sign convention;
- invalid violation-flag rejection;
- metrics/export round-trip;
- canonical Day-21 artifact contract.

The verified result at creation time was:

```text
5 passed
```

These tests intentionally avoid rerunning the full 398-target Gradient
Boosting walk-forward.

They validate contracts and canonical artifacts without introducing unnecessary
model-fitting cost into normal regression testing.

---

## Leakage and Temporal-Safety Notes

The evaluation is chronological.

No random train/test shuffle is used.

Gradient Boosting feature construction uses return-history information that is
available strictly before each target date.

Implemented return-history features are:

```text
return_lag_1
return_lag_2
return_lag_5
rolling_vol_5
rolling_vol_20
rolling_vol_60
drawdown
```

Rolling volatility features are constructed from lagged historical returns
rather than including the current target return.

The prediction contract additionally requires:

```text
forecast_date < target_date
```

for every exported forecast.

These checks reduce the risk of look-ahead leakage in the final evaluation
pipeline.

---

## Scope Limitations

The current Gradient Boosting implementation is deliberately limited to the
implemented return-history feature family.

The following planned feature families are not part of the canonical Day-21
model:

```text
price_range
trading_volume
market_index
```

The project should therefore not claim that the current Gradient Boosting
results incorporate volume, intraday range or VNINDEX-derived predictors.

VNINDEX remains benchmark/context information rather than a constituent of the
equal-weight HPG/FPT/MWG portfolio.

---

## Evaluation-Set Caveat

The evaluation period is labeled:

```text
reserved_later_robustness
```

and:

```text
used_for_parameter_selection = false
```

This means the period was excluded from the parameter-selection process used
for the Day-21 configurations.

However, the project must also preserve:

```text
pristine_untouched_test_claim = false
```

because the broader evaluation period had previously been inspected during the
project.

Accordingly, documentation should use language such as:

> reserved-later evaluation period

or:

> common robustness evaluation period

rather than claiming a completely untouched or pristine test set.

---

## Gradient Boosting Selection Caveat

The canonical Gradient Boosting configuration is:

```text
G04
```

G04 was selected from the A-side validation experiments available to this
branch.

Its relevant configuration is:

```text
n_estimators = 100
learning_rate = 0.03
max_depth = 2
min_samples_leaf = 5
subsample = 1.0
random_state = 42
```

At this stage:

```text
selection_status = validation_candidate_a_slice
model_frozen = false
```

Therefore the correct interpretation is:

> G04 is the current A-side validation candidate used for the Day-21 common
> evaluation.

The incorrect interpretation would be:

> G04 is already the final G01-G07 tuning winner.

That stronger claim requires the independent B-side tuning evidence and the
subsequent integration/model-freeze decision.

---

## EWMA Selection Caveat

The canonical Day-21 EWMA decay remains:

```text
0.94
```

The earlier validation analysis identified:

```text
0.90
```

as the stronger validation alternative.

The Day-21 evaluation intentionally retains `0.94` as the established
canonical baseline for continuity.

Therefore the documentation should distinguish:

```text
validation-selected alternative -> 0.90
Day-21 canonical baseline       -> 0.94
```

The `0.94` configuration must not be described as having won the earlier
validation comparison.

---

## Current Model-Selection Status

This Day-21 evaluation does not declare a final model winner.

The current evidence is:

| Criterion | Strongest observed result |
|---|---|
| Violation-rate calibration | EWMA |
| Pinball Loss | Gradient Boosting G04 |
| Lowest Average VaR | Historical Simulation |

No single method dominates every reported criterion.

The final model decision should therefore wait for:

- result interpretation;
- independent review;
- integration of relevant B-side evidence;
- explicit model-selection reasoning;
- model-freeze decision.

---

## Post-Evaluation Discipline

Once the project reaches model freeze, the forecasting specification should
remain stable.

Post-freeze work should focus on:

- correctness fixes;
- reproducibility;
- integration;
- statistical interpretation;
- result communication;
- automated validation;
- documentation;
- packaging.

New algorithms or new feature families should not be introduced after the
model-freeze point unless the project explicitly reopens model development and
documents that decision.

---

## Day-21 Summary

The Day-21 common evaluation successfully established a reproducible
three-method forecasting pipeline over the same 398 target dates.

Canonical output size:

```text
1194 prediction rows
3 metric rows
1 metadata document
```

Canonical results:

```text
Historical Simulation
violations     = 27
violation_rate = 0.067839195980
pinball_loss   = 0.001896464233
average_var    = 0.021330189279

EWMA
violations     = 22
violation_rate = 0.055276381910
pinball_loss   = 0.001968321253
average_var    = 0.025084042490

Gradient Boosting G04
violations     = 24
violation_rate = 0.060301507538
pinball_loss   = 0.001758130951
average_var    = 0.023349057036
```

The main conclusion at this stage is not that one model has already won.

Instead, the common walk-forward run has produced a validated and reproducible
evidence base for the subsequent result-analysis and model-freeze decisions.