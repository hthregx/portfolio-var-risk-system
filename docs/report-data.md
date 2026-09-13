# Data and Portfolio Construction

## Data

The project uses daily observations for three Vietnamese stocks: HPG, FPT,
and MWG. The processed portfolio-return dataset contains 1,637 daily return
observations from 03 January 2020 to 28 July 2026.

## Portfolio Construction

An equal-weight portfolio is constructed from HPG, FPT, and MWG. Daily simple
returns are used for each stock, and `portfolio_simple_return` is used as the
common modeling target.

Using the same portfolio-return target ensures that Historical Simulation,
EWMA, and Gradient Boosting are evaluated on a consistent basis.

## Data Validation

Before modeling, the data pipeline checks:

- date consistency;
- duplicate dates;
- missing values;
- numeric return values;
- finite observations.

## Forecast Horizon

The project uses a one-trading-day forecasting horizon. For each target date,
only information available before that target observation may be used for
model estimation and forecasting.