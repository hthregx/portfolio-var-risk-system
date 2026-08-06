# Historical VaR Rolling QA

**Reviewer:** Người B  
**Implementation owner:** Người A  
**Notebook reviewed:** `notebooks/03_historical_simulation.ipynb`  
**Review date:** 05/08/2026  
**Rolling window:** 250 observations  
**Confidence level:** 95%  
**Quantile level:** 5%  
**Tolerance:** `1e-12`

---

# 1. Review Objective

The purpose of this review is to independently verify the rolling Historical VaR implementation.

The review covers:

- no look-ahead bias;
- rolling window size;
- forecast date and target date semantics;
- independent rolling cross-check;
- rolling invariants;
- rolling summary statistics.

---

# 2. Rolling Forecast Definition

For each `target_date`, the implementation uses exactly **250 historical portfolio returns** immediately preceding the target date.

```text
window_start_index = target_index - 250
window_end_index   = target_index - 1
target_index       = target_index
```

Therefore,

- the estimation window always contains 250 observations;
- the target-day return is excluded from the estimation window;
- Historical VaR is computed using only information available before the forecast date.

---

# 3. Independent No-lookahead Audit

A representative rolling forecast was selected for review.

| Item | Value |
|------|------|
| Forecast date | 2020-12-30 |
| Target date | 2020-12-31 |

### Review Findings

- `forecast_date < target_date`
- the estimation window ends before the target date;
- the target-day return is excluded from the rolling estimation window;
- no future information is used when computing Historical VaR.

**Result: PASS**

---

# 4. Forecast Date vs Target Date

The rolling Historical VaR output clearly distinguishes between the forecast generation date and the forecast evaluation date.

Example:

| Forecast Date | Target Date |
|---------------|-------------|
| 2020-12-30 | 2020-12-31 |

### Review Findings

- `forecast_date` represents the last date with available historical information.
- `target_date` represents the next trading day on which the forecast is evaluated.
- The implementation follows a next-day forecasting workflow.
- No ambiguity between forecast generation and forecast application was found.

**Result: PASS**

---

# 5. Independent Rolling Cross-check

Three representative forecasts were independently reviewed:

- First forecast
- Middle forecast
- Last forecast

For each forecast, the reviewer:

1. reconstructed the corresponding 250-observation estimation window;
2. recalculated the empirical 5% quantile using:

```python
np.quantile(window_returns, 0.05, method="linear")
```

3. calculated Historical VaR using

```text
VaR = max(0, -q05)
```

4. compared the independently calculated value with the implementation output.

Comparison tolerance:

```text
1e-12
```

### Review Result

| Forecast | Independent Review | Result |
|-----------|-------------------|--------|
| First | Matched | PASS |
| Middle | Matched | PASS |
| Last | Matched | PASS |

### Conclusion

The independently calculated Historical VaR values matched the implementation output for all three representative forecasts.

No difference exceeded the specified tolerance (`1e-12`).

**Overall Result: PASS**

---

# 6. Rolling Invariants

The reviewer verified the following rolling invariants.

| Check | Expected | Result |
|--------|----------|--------|
| Fixed rolling window | 250 observations | PASS |
| No missing Historical VaR | Yes | PASS |
| No duplicate target dates | Yes | PASS |
| Target dates strictly increasing | Yes | PASS |
| Historical VaR ≥ 0 | Yes | PASS |
| Quantile level | 0.05 | PASS |

### Conclusion

The rolling Historical VaR implementation satisfies all required rolling invariants.

**Overall Result: PASS**

---

# 7. QA Summary

Data source:

```text
data/processed/historical_var_rolling.csv
```

### Rolling Statistics

| Metric | Value |
|--------|-------:|
| Number of forecasts | 1387 |
| First forecast date | 2020-12-31 |
| Last forecast date | 2026-07-28 |
| Minimum Historical VaR | 0.017088 |
| Maximum Historical VaR | 0.043786 |
| Average Historical VaR | 0.026231 |

### Review Summary

The reviewer confirms that:

- the rolling window size remains fixed at 250 observations;
- no missing Historical VaR values exist after forecasting begins;
- no duplicate target dates were found;
- target dates are strictly increasing;
- Historical VaR values are always non-negative;
- the implementation consistently applies the 5% empirical quantile.

---

# 8. Final QA Result

The rolling Historical VaR implementation satisfies all independent review criteria.

The reviewer confirms:

- no look-ahead bias;
- correct forecast-date and target-date semantics;
- correct rolling Historical VaR calculation;
- correct rolling invariants;
- consistent rolling summary statistics.

**Overall QA Result: PASS**