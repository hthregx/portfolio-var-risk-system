# Gradient Boosting Feature Specification

## B-17.1 — Canonical Data Audit

The market/liquidity feature slice uses the following canonical inputs:

| Series | Source | Date field | Required fields |
| --- | --- | --- | --- |
| HPG | `data/processed/HPG_clean.csv` | `date` | `high`, `low`, `close`, `volume` |
| FPT | `data/processed/FPT_clean.csv` | `date` | `high`, `low`, `close`, `volume` |
| MWG | `data/processed/MWG_clean.csv` | `date` | `high`, `low`, `close`, `volume` |
| VN-Index | `data/raw/VNINDEX.csv` | `time` | `close` |

Stock schema:

`date,ticker,open,high,low,close,volume`

VN-Index schema:

`time,open,high,low,close,volume`

VN-Index timestamps are normalized to trading dates before alignment.

Audit result:

- High: YES
- Low: YES
- Close: YES
- Volume: YES
- VN-Index: YES

B-17.1 canonical data audit: `PASS`

---

## B-17.2 — Calendar and Volume Audit

### Calendar

HPG, FPT, MWG, and VN-Index each contain 1,638 unique trading dates from
2020-01-02 through 2026-07-28.

Audit result:

- duplicate dates: 0
- stock dates missing from VN-Index: 0
- VN-Index dates outside the common stock calendar: 0

Canonical alignment policy:

`exact-date`

Future market backfill is forbidden.

For every target feature row:

`source_max_date <= forecast_date < target_date`

must hold.

### Volume

Canonical stock data contain:

| Asset | Missing | Zero | Negative | Maximum |
| --- | ---: | ---: | ---: | ---: |
| HPG | 0 | 0 | 0 | 215,999,100 |
| FPT | 0 | 0 | 0 | 47,733,100 |
| MWG | 0 | 0 | 0 | 29,616,700 |

Volume handling policy:

- missing volume -> invalid input; feature construction fails explicitly
- zero denominator -> affected ratio feature becomes `NaN`
- negative volume -> invalid input
- extreme positive volume -> preserved, not clipped
- corporate-event-like spike -> retained for audit; not automatically classified as a corporate event
- future volume -> never used to fill an earlier feature row

B-17.2 calendar and volume audit: `PASS`

---

## B-17.3 — Feature Timing and Formula Contract

Each output row is indexed by target date `T`.

The latest allowed source date is forecast date `F`, where:

`F < T`

All inputs used for the row must satisfy:

`source_date <= F`

### Range

For stock `i`:

`range_i(F) = (high_i(F) - low_i(F)) / close_i(F)`

Portfolio feature:

`portfolio_range = mean(range_HPG, range_FPT, range_MWG)`

Target-date High, Low, or Close values are not used in their own target
feature row.

### Volume

For stock `i`:

`volume_change_i(F) = volume_i(F) / volume_i(F-1) - 1`

`relative_volume_20_i(F) = volume_i(F) / mean(volume_i[F-19:F]) - 1`

Portfolio features are equal-weight averages across HPG, FPT, and MWG.

### Market

VN-Index features are:

- `market_return_lag_1`
- `market_return_lag_5`
- `market_vol_20`

All are computed using market data available no later than forecast date
`F`.

`market_vol_20` uses:

- window = 20
- `ddof = 1`
- `min_periods = 20`

### Warm-up and missing data

Warm-up values remain `NaN` until enough history exists.

Missing source data are never repaired using future observations.

For canonical data, exact-date stock/VN-Index alignment is used.

### No-look-ahead

The implementation is validated by:

- direct target-date perturbation
- future perturbation
- missing-market-date / no-future-backfill test

Changing target-date or future observations must not alter an already
valid earlier feature row.

B-17.3 feature timing and formula contract: `PASS`

---

## B-17.4 — Production Feature Interface

Implementation:

`src/gb_market_features.py`

Entry point:

`build_market_features(stock_frames, market_frame)`

Output columns:

1. `portfolio_range`
2. `volume_change`
3. `relative_volume_20`
4. `market_return_lag_1`
5. `market_return_lag_5`
6. `market_vol_20`

The output is a target-date-indexed pandas DataFrame.

The return-history feature family is owned separately. This document only
defines the expected interface: feature families expose target-date-indexed
DataFrames and independently preserve the no-look-ahead contract.

B-17.4 production feature interface: `PASS`

---

## B-17.5 — Canonical Market Feature Audit

The production builder was executed on:

- `data/processed/HPG_clean.csv`
- `data/processed/FPT_clean.csv`
- `data/processed/MWG_clean.csv`
- `data/raw/VNINDEX.csv`

Canonical run result:

- rows: 1,638
- finite rows after warm-up: 1,617
- warm-up rows: 21
- alignment policy: exact-date
- future backfill: disabled
- target-date no-look-ahead: PASS
- market alignment: PASS
- future perturbation: PASS
- determinism: PASS
- dedicated market-feature tests: 19 PASS

Audit artifact:

`results/gb_market_feature_audit.json`

Observed missing feature values are limited to expected lag and warm-up
requirements.

B-17.5 canonical market feature audit: `PASS`