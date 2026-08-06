# Historical VaR Backtest Methodology

## Reviewer

- **Reviewer:** Người B
- **Implementation owner:** Người A
- **Review date:** 06/08/2026

---

# 1. Objective

The purpose of the backtest is to independently verify whether Historical VaR violations are correctly identified by the implementation.

The reviewer validates the violation logic without modifying the implementation or reproducing the original backtesting pipeline.

---

# 2. Historical VaR Violation

A Historical VaR violation occurs when the realized portfolio return falls below the empirical left-tail threshold estimated from the rolling historical window.

The violation rule is:

```text
target_return < quantile_return
```

where:

- `target_return` is the realized portfolio return on the target date.
- `quantile_return` is the empirical 5% historical quantile estimated from the previous 250 observations.

---

# 3. Sign Convention

The empirical quantile represents a downside return threshold and is therefore typically negative.

Example:

```text
quantile_return = -0.0300
```

Historical VaR is reported as a positive loss magnitude:

```text
historical_var = max(0, -quantile_return)
```

Example:

```text
quantile_return = -0.0300
historical_var = 0.0300
```

Although both values have the same magnitude, they serve different purposes.

- `quantile_return` is used to evaluate violations.
- `historical_var` is used to report portfolio risk.

---

# 4. Violation Rule

The implementation uses the following decision rule:

```text
target_return < quantile_return
```

If the condition is satisfied:

```text
violation = True
```

Otherwise:

```text
violation = False
```

Equality is not considered a violation.

---

# 5. Chart Convention

The Historical VaR backtest chart compares:

- actual portfolio return;
- negative historical quantile threshold;
- violation observations.

The plotted threshold must be:

```text
quantile_return
```

rather than the positive Historical VaR magnitude.

This ensures that the threshold is displayed correctly on the return scale.

---

# 6. Independent Validation Result

The independent review confirms that:

- the violation rule is implemented correctly;
- the sign convention is internally consistent;
- the chart uses the negative quantile threshold instead of the positive VaR magnitude.

**Overall Methodology Review: PASS**