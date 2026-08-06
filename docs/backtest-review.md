# Historical VaR Independent Backtest Review

**Reviewer:** Người B  
**Implementation owner:** Người A  
**Notebook reviewed:** `notebooks/03_historical_simulation.ipynb`  
**Review notebook:** `notebooks/05_backtest_review.ipynb`  
**Review date:** 06/08/2026

---

# 1. Review Scope

The reviewer independently validates:

- Historical VaR violation logic;
- sign convention;
- first, middle and last forecasts;
- violation table consistency;
- backtest chart;
- methodology documentation.

The reviewer does not modify or reimplement the original backtesting pipeline.

---

# 2. Independent Sign Audit

Four representative observations were independently reviewed:

| Case | Result |
|------|--------|
| Normal positive return | PASS |
| Normal negative return | PASS |
| Obvious violation | PASS |
| Near-threshold observation | PASS |

For each observation, the reviewer independently evaluated:

```text
target_return < quantile_return
```

and compared the expected result with the implementation.

All observations matched.

**Result: PASS**

---

# 3. First, Middle and Last Forecast Audit

Three representative forecasts were reviewed:

- First forecast
- Middle forecast
- Last forecast

For each forecast, the reviewer confirmed:

- target return;
- quantile threshold;
- expected violation;
- actual violation.

All reviewed forecasts matched the implementation.

**Result: PASS**

---

# 4. Full Violation-table Audit

The reviewer independently verified the complete backtest output.

| Metric | Value |
|---------|------:|
| Total observations | 1387 |
| Violation rows | 75 |
| Non-violation rows | 1312 |
| Incorrect violation flags | 0 |

Independent validation confirmed:

- every violation satisfies

```text
target_return < quantile_return
```

- every non-violation satisfies

```text
target_return >= quantile_return
```

No inconsistencies were found.

**Result: PASS**

---

# 5. Chart Audit

The reviewer verified that the backtest chart plots:

- actual portfolio return;
- negative historical quantile threshold;
- violation observations.

The chart does not use the positive Historical VaR magnitude as the threshold.

The plotted sign convention is correct.

**Result: PASS**

---

# 6. Review Summary

| Review Item | Result |
|-------------|--------|
| Independent sign audit | PASS |
| First / Middle / Last audit | PASS |
| Full violation-table audit | PASS |
| Chart audit | PASS |
| Methodology review | PASS |

---

# 7. Final Conclusion

The reviewer found no inconsistency between the expected Historical VaR violation logic and the implementation.

The violation rule, sign convention, violation table and chart are internally consistent.

**Overall Independent Backtest Validation: PASS**