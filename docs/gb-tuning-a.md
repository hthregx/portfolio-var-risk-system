# Gradient Boosting Tuning A — G01 to G04

## Purpose

This experiment slice evaluates four pre-registered Gradient Boosting Quantile configurations on the fixed chronological validation period.

The objective is light validation tuning rather than unrestricted hyperparameter search.

## Validation Contract

Validation target dates:

- start: 2021-12-31
- end: 2024-12-17
- target count: 739

Each target date is forecast using an expanding training window containing only feature-ready observations strictly before that target date.

The first validation forecast uses 440 training observations and the last uses 1,178.

The reserved-later period beginning 2024-12-18 is excluded from parameter selection and no reserved-later metrics are computed in this task.

## Target and Features

Target:

`portfolio_simple_return`

Return-history features:

- return_lag_1
- return_lag_2
- return_lag_5
- rolling_vol_5
- rolling_vol_20
- rolling_vol_60
- drawdown

All features follow the existing target-date no-look-ahead contract.

## Common Model Contract

All experiments use:

- loss: quantile
- alpha: 0.05
- min_samples_leaf: 5
- subsample: 1.0
- random_state: 42
- one-day-ahead target
- VaR = max(0, -quantile_return)

A violation is defined strictly as:

`actual_return < quantile_return`

Equality is not a violation.

## Pre-Registered Experiments

| Experiment | max_depth | learning_rate | n_estimators |
|---|---:|---:|---:|
| G01 | 2 | 0.05 | 100 |
| G02 | 1 | 0.05 | 100 |
| G03 | 3 | 0.05 | 100 |
| G04 | 2 | 0.03 | 100 |

G01 is the previously locked prototype reference configuration.

G02-G04 were fixed before their validation performance was inspected.

## Validation Results

| Experiment | Violations | Violation Rate | Pinball Loss | Average VaR |
|---|---:|---:|---:|---:|
| G01 | 43 | 0.0581867388 | 0.002065622435 | 0.0247593073 |
| G02 | 41 | 0.0554803789 | 0.002040482579 | 0.0260197235 |
| G03 | 51 | 0.0690121786 | 0.002120241571 | 0.0232998426 |
| G04 | 38 | 0.0514208390 | 0.002029749664 | 0.0255870042 |

## A-Slice Validation Decision

Pinball Loss at alpha 0.05 is the primary ranking metric for this tuning slice.

G04 has the lowest validation Pinball Loss among G01-G04.

Its Violation Rate is also closer to the nominal 5% level than G01.

Relative to G01, G04 reduces Pinball Loss by approximately 1.74%.

G04 is therefore selected as the A-slice validation candidate among G01-G04.

This is not a final G01-G07 model-selection decision because G05-G07 belong to the independent B tuning slice.

The result does not modify the production Gradient Boosting configuration in this task.

## Interpretation

G02 improves Pinball Loss relative to G01 while producing a slightly more conservative Average VaR.

G03 performs worst among the four configurations on both Pinball Loss and violation-rate calibration, so increasing maximum depth from 2 to 3 is not supported by this validation slice.

G04 improves the primary metric while keeping the architecture close to G01, changing only the learning rate from 0.05 to 0.03.

These results are validation-specific and are not evidence of universal superiority across future market regimes.

## Reproducibility

Experiment code:

`scripts/run_gb_tuning_a.py`

Automated tests:

`tests/test_gb_tuning_a.py`

Aggregate metrics:

`results/gb_tuning_a.csv`

Metadata:

`results/gb_tuning_a_metadata.json`

The metadata records the validation boundary, feature schema, expanding-refit protocol, experiment IDs, random seed, source base commit, and reserved-later exclusion.

## Prompt Provenance

PROMPT GỐC ĐỂ LƯU:

Run a pre-registered Gradient Boosting Quantile light-tuning slice for G01-G04 using the fixed 739-row chronological validation period. Refit on an expanding history strictly before every target date, preserve the alpha 0.05 quantile and VaR sign contracts, rank primarily by validation Pinball Loss, exclude reserved-later performance from parameter selection, and record the A-slice candidate without claiming a final G01-G07 winner or automatically modifying production configuration.
