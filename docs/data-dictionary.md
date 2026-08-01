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