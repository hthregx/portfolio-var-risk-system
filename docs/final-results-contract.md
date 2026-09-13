# Final Results Contract

## Purpose

Track B independently validates final VaR outputs without retraining models
or importing `src/evaluation/final_metrics.py`.

QA fixtures:

- `data/sample/final_predictions_contract_sample.csv`
- `data/sample/final_metrics_contract_sample.csv`

Final integration artifacts:

- `results/final_predictions.csv`
- `results/final_metrics.csv`

Fixtures are QA data only, not final experimental results.

## Prediction Contract

Required fields:

- `forecast_date`
- `target_date`
- `actual_return`
- `quantile_return`
- `var`
- `violation`
- `method`
- `runtime_seconds`
- `config_id`

Unique key:

`(method, target_date)`

Allowed methods:

- `historical_simulation`
- `ewma`
- `gradient_boosting`

Required rules:

`forecast_date < target_date`

`var = max(0, -quantile_return)`

`violation = actual_return < quantile_return`

Equality is not a violation.

All required numeric values must be finite. Missing values and duplicate
keys are rejected.

All methods must have the same target-date universe and the same
`actual_return` for each target date.

## Metrics Contract

Required fields:

- `method`
- `forecast_count`
- `violation_count`
- `violation_rate`
- `pinball_loss`
- `average_var`
- `minimum_var`
- `maximum_var`
- `total_runtime_seconds`
- `test_start`
- `test_end`
- `config_id`

Unique key:

`method`

Metrics are independently recomputed from predictions.

At `alpha = 0.05`:

- violation count = count of `actual_return < quantile_return`
- violation rate = violation count / forecast count
- Pinball Loss is independently recomputed
- average/minimum/maximum VaR are recomputed from `var`
- forecast count is recomputed from prediction rows

`results/final_metrics.csv` is used only for comparison, not as the source
of truth.

## Units and Display

Returns, VaR, Pinball Loss, and violation rate use decimal units.

Source numeric values are not rounded.

Percentage formatting is performed only by the display/report layer.

VaR sign is never changed.

## Validation

`src/validation/final_results_contract.py` checks:

- required columns and dtypes
- finite values
- duplicate keys
- date ordering
- allowed methods
- VaR sign convention
- strict violation rule
- common target dates
- common actual returns

Invalid data is rejected and is never automatically repaired.

Gradient Boosting leakage tests also verify that target/future return,
OHLCV, and market observations do not change feature row `T`.

## Reporting Adapter

`src/reporting/final_results_adapter.py` produces:

- model comparison table
- exception table
- latest VaR summary
- dashboard-ready data
- report-ready summary

The adapter does not retrain models, modify predictions, round source
values, or change VaR sign.

## Independent Audit

Fixture audit:

```bash
python scripts/audit_final_results.py \
  --predictions data/sample/final_predictions_contract_sample.csv \
  --metrics data/sample/final_metrics_contract_sample.csv \
  --alpha 0.05
```

Final integration audit:

```bash
python scripts/audit_final_results.py \
  --predictions results/final_predictions.csv \
  --metrics results/final_metrics.csv \
  --alpha 0.05
```

Track B does not import `src/evaluation/final_metrics.py`.

Manual QA checkpoints are documented in:

`docs/final-results-qa.md`

## Completion Gate

B21 passes when:

- valid fixtures pass
- intentional invalid fixtures are rejected
- wrong violation rule is rejected
- wrong alpha is detected
- temporal/leakage checks pass
- reporting adapter tests pass
- final A artifacts are independently audited
- full repository tests pass
- only Track B files are staged

Passing this contract is an independent QA result, not a claim about
which model performs best.