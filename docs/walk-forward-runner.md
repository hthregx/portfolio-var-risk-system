# Common Walk-Forward Runner

## 1. Purpose

`src/backtesting/walk_forward.py` provides a model-independent framework for
one-day-ahead walk-forward backtesting.

The runner is responsible for:

- building training windows;
- calling the forecasting model;
- aligning forecast and target dates;
- generating violations;
- collecting predictions;
- recording runtime metadata.

Model-specific forecasting logic is kept in `src/models/`.


## 2. Evaluation Modes

### Rolling

For target index `i` and window size `w`:

```text
train  = rows[i-w:i]
target = row[i]
```

The training window always contains exactly `w` observations.

### Expanding

```text
train  = rows[0:i]
target = row[i]
```

The training window grows after each forecast.

Supported modes:

```text
rolling
expanding
```


## 3. Model Interface

The runner calls:

```python
forecast = forecast_function(train_returns)
```

The model must return:

```python
{
    "quantile_return": float,
    "var": float,
}
```

The runner does not calculate model-specific forecasts such as Historical
Simulation quantiles.


## 4. Prediction Schema

Each forecast produces:

```text
window_end_date
forecast_date
target_date
actual_return
quantile_return
var
violation
method
observations
evaluation_mode
```

`forecast_date` is the last date in the training window.

`target_date` is the date being forecast.

The required alignment is:

```text
forecast_date < target_date
```


## 5. Sign Convention and Violation

`quantile_return` is the lower-tail return threshold and is normally negative.

`var` is the non-negative VaR loss magnitude.

A violation is:

```python
actual_return < quantile_return
```

Equality is not a violation.

The runner must not compare `actual_return < var`.


## 6. No-Lookahead Guarantee

The target observation is never included in the training window.

For every forecast:

```text
training data ends at forecast_date
forecast_date < target_date
```

This prevents target leakage during walk-forward evaluation.


## 7. Runtime

Total execution time is recorded as:

```python
predictions.attrs["runtime_seconds"]
```

Runtime is run-level metadata and does not affect forecast values.


## 8. Saving Predictions

Predictions can be saved separately with:

```python
save_predictions(predictions, output_path)
```

Full processed prediction files should remain ignored by Git.

Only small reviewer-facing samples should be tracked when required.


## 9. Pinball Loss Compatibility

The common schema preserves:

```text
actual_return
quantile_return
```

so downstream evaluation can calculate Pinball Loss at `alpha = 0.05`.

Pinball Loss does not need to be calculated inside the runner.


## 10. Verification

The runner is tested for:

- rolling and expanding windows;
- forecast count and schema;
- no lookahead;
- date alignment;
- violation logic;
- runtime logging;
- prediction saving;
- invalid inputs;
- model independence;
- Historical Simulation compatibility.

Current verification:

```text
Walk-forward tests: 27 passed
Full test suite:    117 passed
Failures:           0
Errors:             0
```

## Conclusion

The common walk-forward runner provides a reusable evaluation framework for
Historical Simulation and future forecasting models while maintaining a
consistent prediction schema, date alignment, sign convention, and
no-lookahead guarantee.