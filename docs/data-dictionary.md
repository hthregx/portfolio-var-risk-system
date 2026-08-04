# Data Dictionary

## 1. Purpose

This document defines the main variables used in the cleaned OHLCV datasets and the exploratory data analysis stage of the portfolio Value at Risk project.

The validated datasets cover HPG, FPT, and MWG from 2020-01-02 to 2026-07-28.

---

## 2. Processed OHLCV Variables

| Variable | Data Type | Unit | Description | Source / Calculation |
|---|---|---|---|---|
| `date` | datetime | Trading date | Trading date associated with each OHLCV observation. | Standardized from the raw `time` field by the validation pipeline. |
| `ticker` | string | - | Stock ticker identifying the security. | Added during the data validation and cleaning process. |
| `open` | float | Price | Opening price of the stock for the trading session. | Raw market data after validation. |
| `high` | float | Price | Highest traded price during the trading session. | Raw market data after validation. |
| `low` | float | Price | Lowest traded price during the trading session. | Raw market data after validation. |
| `close` | float | Price | Closing price of the stock for the trading session. | Raw market data after validation. |
| `volume` | integer | Shares | Trading volume recorded for the trading session. | Raw market data after validation. |

---

## 3. EDA-Derived Variables

| Variable | Data Type | Unit | Description | Calculation |
|---|---|---|---|---|
| `normalized_price` | float | Index | Closing price normalized to 100 at the first observation of each ticker. Used to compare relative price performance across stocks with different absolute price levels. | `close_t / close_0 * 100` |
| `previous_close` | float | Price | Closing price from the previous trading session for the same ticker. | One-period lag of `close`. |
| `log_return` | float | Decimal return | Daily continuously compounded return calculated from consecutive closing prices. | `ln(close_t / close_t-1)` |
| `abs_log_return` | float | Decimal return | Absolute magnitude of the daily log return. Used to identify large price movements during EDA. | `abs(log_return)` |
| `return_sign` | categorical | - | Direction of the daily log return. | `negative`, `zero`, or `positive`. |

---

## 4. Data Coverage

| Ticker | Observations | Valid Log Returns | Start Date | End Date |
|---|---:|---:|---|---|
| HPG | 1638 | 1637 | 2020-01-02 | 2026-07-28 |
| FPT | 1638 | 1637 | 2020-01-02 | 2026-07-28 |
| MWG | 1638 | 1637 | 2020-01-02 | 2026-07-28 |

All three securities share the same 1,638 trading dates in the validated dataset.

---

## 5. Return Convention

Daily log returns are defined as:

\[
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
\]

where:

- \(P_t\) is the closing price on trading day \(t\);
- \(P_{t-1}\) is the closing price on the previous trading day.

The first observation of each ticker has no previous closing price and therefore has an undefined log return (`NaN`).

These log returns are currently used for exploratory data analysis. The final return convention used for portfolio construction and VaR estimation will be formally defined in the portfolio-return stage.

---

## 6. Data Quality Rules

The processed datasets are generated only after validation of:

- required OHLCV columns;
- trading-date parsing;
- missing observations;
- duplicate trading dates;
- nonpositive prices;
- negative trading volume;
- OHLC consistency;
- chronological ordering.

Extreme price movements are flagged for review but are not automatically removed because they may represent genuine market tail events.

---

## 7. Notes

- Raw data must not be modified during exploratory analysis.
- EDA is performed only on files under `data/processed`.
- Zero-return observations are retained as valid market observations.
- Individual-stock 5th percentiles observed during EDA are not the final portfolio VaR estimates.
- Additional portfolio-level and model-related variables will be added to later versions of this data dictionary.

## Portfolio Return Dataset

### Canonical dataset

The canonical portfolio-return dataset is stored at:

`data/processed/portfolio_returns.csv`

A small review sample is stored at:

`data/sample/portfolio_returns_sample.csv`

The canonical dataset contains 1,637 valid daily return observations
from 2020-01-03 through 2026-07-28, with no duplicate dates and no
missing values.

### Portfolio construction

The default portfolio contains HPG, FPT, and MWG.

The portfolio is long-only, fully invested, and equally weighted:

- HPG weight = 1/3
- FPT weight = 1/3
- MWG weight = 1/3

Individual simple return is defined as:

`R_i,t = P_i,t / P_i,t-1 - 1`

The portfolio simple return is constructed cross-sectionally as:

`R_p,t = sum(w_i * R_i,t)`

The portfolio simple return is the primary portfolio return series
used by the Value at Risk pipeline.

Portfolio log return is derived from the constructed portfolio simple
return:

`r_p,t = ln(1 + R_p,t)`

A weighted average of individual asset log returns is not treated as
the exact portfolio log return. It is used only as a methodological
diagnostic and is not included in the canonical portfolio-return
dataset.

### Variable dictionary

| Variable | Type | Unit | Definition | Role |
| --- | --- | --- | --- | --- |
| `date` | datetime | trading date | Trading date associated with the return observation. | Time index |
| `HPG_simple_return` | float | decimal return | Daily simple return of HPG. | Portfolio input |
| `FPT_simple_return` | float | decimal return | Daily simple return of FPT. | Portfolio input |
| `MWG_simple_return` | float | decimal return | Daily simple return of MWG. | Portfolio input |
| `portfolio_simple_return` | float | decimal return | Equal-weight simple return of HPG, FPT, and MWG. | Primary VaR return series |
| `portfolio_log_return` | float | decimal log return | `ln(1 + portfolio_simple_return)`. | Derived analytical series |

### Missing-value convention

The first price observation for each asset has no previous closing
price and therefore has an undefined return.

This initial undefined return is retained during return construction
and validation, but it is excluded from the canonical portfolio-return
dataset.

As a result, 1,638 aligned price dates produce 1,637 valid portfolio
return observations.

A missing return must not be interpreted or filled as a zero return.

### VN-Index benchmark

VN-Index is designated as a market benchmark and is not a constituent
of the default portfolio.

Its portfolio weight is therefore zero.

The current raw benchmark file is:

`data/raw/VNINDEX.csv`

The raw VN-Index dataset contains 1,638 observations covering
2020-01-02 through 2026-07-28.

Its raw timestamp includes a `07:00:00` time component. For daily
trading-calendar comparison, the timestamp is normalized to midnight
without changing the calendar date.

After normalization, the VN-Index and portfolio price calendars contain
1,638 common trading dates with no missing dates in either direction.

`VNINDEX.csv` remains a raw dataset. No `VNINDEX_clean.csv` processed
benchmark dataset has been established at this stage, so raw VN-Index
data must not be silently treated as model-ready processed data.
