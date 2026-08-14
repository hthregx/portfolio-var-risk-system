# Historical VaR Backtest Result Summary

**Reviewer:** Người B  
**Implementation owner:** Người A  
**Review date:** 07/08/2026  

## 1. Backtest Metrics

Historical Simulation được thực hiện với:

- Confidence level: 95%
- Left-tail probability: 5%
- Rolling window: 250 observations
- Forecast horizon: one day ahead

Backtest dataset:

`data/processed/historical_var_backtest.csv`

Kết quả backtest:

| Metric | Value |
|---|---:|
| Number of forecasts | 1387 |
| Number of violations | 75 |
| Observed violation rate | 5.407354% |
| Expected violation rate | 5.000000% |
| Violation rate difference | +0.407354 percentage points |
| Average Historical VaR | 2.623147% |
| Minimum Historical VaR | 1.708839% |
| Maximum Historical VaR | 4.378557% |

Expected violation rate là 5%, tương ứng với confidence level 95%.

---

## 2. Independent Metric Verification

Người B independently recomputed các backtest metrics trực tiếp từ:

`data/processed/historical_var_backtest.csv`

Việc kiểm tra không sử dụng summary của Người A làm nguồn tính toán.

Các metrics được tính lại gồm:

- number of forecasts;
- number of violations;
- observed violation rate;
- average Historical VaR;
- minimum Historical VaR;
- maximum Historical VaR.

Sau khi tính độc lập, kết quả của Người B được so sánh với metrics do Người A
báo cáo trong:

`outputs/tables/historical_var_backtest_metrics.csv`

Numeric tolerance sử dụng cho comparison:

`1e-12`

Tất cả các metrics đều khớp trong tolerance yêu cầu.

**B-07.1 Independent metric recomputation: PASS**

---

## 3. Independent Violation-Date Verification

Một VaR violation được xác định khi:

`target_return < quantile_return`

Trong đó:

- `target_return` là actual portfolio return tại target date;
- `quantile_return` là negative 5% Historical Simulation threshold;
- `historical_var` là positive loss magnitude.

Người B independently kiểm tra 15 violation observations gồm:

- first 5 violations;
- middle 5 violations;
- last 5 violations.

### First 5 violations

| Target date | Target return | Quantile return |
|---|---:|---:|
| 2021-01-19 | -0.060883 | -0.037728 |
| 2021-01-28 | -0.069552 | -0.036194 |
| 2021-04-22 | -0.024475 | -0.024062 |
| 2021-04-26 | -0.026977 | -0.024782 |
| 2021-06-08 | -0.024962 | -0.024782 |

### Middle 5 violations

| Target date | Target return | Quantile return |
|---|---:|---:|
| 2023-10-03 | -0.039877 | -0.035094 |
| 2023-10-17 | -0.031041 | -0.030237 |
| 2023-10-26 | -0.043928 | -0.028450 |
| 2023-10-31 | -0.030813 | -0.030237 |
| 2023-11-23 | -0.045536 | -0.024505 |

### Last 5 violations

| Target date | Target return | Quantile return |
|---|---:|---:|
| 2026-03-23 | -0.034915 | -0.027020 |
| 2026-07-20 | -0.025060 | -0.023372 |
| 2026-07-22 | -0.025912 | -0.024635 |
| 2026-07-24 | -0.025561 | -0.025074 |
| 2026-07-27 | -0.033970 | -0.025347 |

Tất cả 15 observations đều thỏa:

`target_return < quantile_return`

Không phát hiện inconsistency giữa actual return, quantile threshold và
violation flag trong sample được kiểm tra.

**B-07.2 Independent violation-date verification: PASS**

---

## 4. Calibration Interpretation

Với confidence level 95%, expected violation rate là:

`5.000000%`

Observed violation rate được tính từ backtest là:

`5.407354%`

Observed rate cao hơn nominal rate:

`5.407354% - 5.000000% = +0.407354 percentage points`

Do observed violation rate cao hơn mức 5%, kết quả cho thấy
**mild undercoverage**.

Điều này có nghĩa Historical Simulation có xu hướng nhẹ đánh giá thấp
one-day downside tail risk trong giai đoạn backtest.

Vì observed violation rate cao hơn nominal rate, không phù hợp để diễn giải
mô hình là `overly conservative`.

Đây là descriptive calibration assessment. Formal coverage hypothesis test
chưa được thực hiện trong phạm vi review này.

**B-07.3 Interpretation review: PASS**

---

## 5. Review Summary

| Check | Result |
|---|---|
| Number of forecasts recomputation | PASS |
| Number of violations recomputation | PASS |
| Violation rate recomputation | PASS |
| Average VaR recomputation | PASS |
| Minimum VaR recomputation | PASS |
| Maximum VaR recomputation | PASS |
| Numeric tolerance `1e-12` | PASS |
| First 5 violations | PASS |
| Middle 5 violations | PASS |
| Last 5 violations | PASS |
| Violation threshold logic | PASS |
| Calibration interpretation | PASS |

No material inconsistency was identified between the independently recomputed
metrics and the Historical Simulation backtest outputs.

## 6. Conclusion

Người B independently verified the Historical Simulation backtest results.

The backtest contains **1,387 forecasts** and **75 violations**, corresponding
to an observed violation rate of **5.407354%** compared with the nominal
**5.000000%** rate.

Independent metric recomputation matched the reported implementation outputs
within the required tolerance of `1e-12`.

The first 5, middle 5 and last 5 sampled violations all satisfied the required
condition:

`target_return < quantile_return`

The observed violation rate is slightly above the nominal level and is
therefore consistent with **mild undercoverage**, not an overly conservative
model.

**Independent Historical Simulation Backtest Validation: PASS**