# Historical VaR Independent Review

**Reviewer:** Người B  
**Implementation owner:** Người A  
**Notebook:** `notebooks/03_historical_simulation.ipynb`

---

## 1. Review Scope

This review independently validates:

- Historical Simulation methodology
- Empirical quantile
- Sign convention
- Input selection
- Manual calculation
- Notebook reproducibility

The reviewer does not modify the implementation.

---

## 2. API Review

Reviewed function:

```text
calculate_historical_var(returns, confidence_level=0.95)
```

| Item | Result |
|------|--------|
| Default confidence level = 0.95 | PASS |
| Alpha = 0.05 | PASS |
| Linear empirical quantile | PASS |
| VaR = max(0, -q05) | PASS |

---

## 3. Manual Calculation

Independent return vector (20 observations) was used.

Results:

```text
q05  = -0.042
VaR95 = 0.042
```

NumPy verification produced the same result.

| Item | Result |
|------|--------|
| q05 | PASS |
| Historical VaR | PASS |

---

## 4. Sign Convention

Reviewed rule:

```text
VaR = max(0, -q05)
```

| Case | Result |
|------|--------|
| q05 < 0 → VaR > 0 | PASS |
| q05 = 0 → VaR = 0 | PASS |
| q05 > 0 → VaR = 0 | PASS |

---

## 5. Input Review

Historical VaR is calculated from:

```text
portfolio_simple_return
```

Not from:

- portfolio_log_return
- HPG return
- FPT return
- MWG return

| Item | Result |
|------|--------|
| portfolio_simple_return used | PASS |
| portfolio_log_return not used | PASS |
| Individual stock returns not used | PASS |

---

## 6. Notebook Reproducibility

Notebook:

```text
notebooks/03_historical_simulation.ipynb
```

Review steps:

- Restart Kernel
- Run All Cells

Initial execution failed because:

```text
data/processed/portfolio_returns.csv
```

was missing.

After generating the dataset from:

```text
notebooks/02_portfolio_returns.ipynb
```

the notebook executed successfully.

| Item | Result |
|------|--------|
| Restart Kernel | PASS |
| Run All | PASS |
| No traceback | PASS |
| Output generated | PASS |

---

# Final Result

| Review Item | Result |
|-------------|--------|
| Methodology | PASS |
| Manual calculation | PASS |
| Sign convention | PASS |
| Input review | PASS |
| Notebook reproducibility | PASS |

**Overall Review: PASS**