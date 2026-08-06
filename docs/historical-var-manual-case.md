# Independent Manual Historical VaR Case

**Reviewer:** Người B  
**Implementation owner:** Người A  
**Confidence level:** 95%  
**Left-tail alpha:** 5%  
**Quantile method:** Linear interpolation  
**Tolerance:** `1e-12`

---

## 1. Mục tiêu

Người B thực hiện một phép tính Historical VaR thủ công trên một vector lợi
suất nhỏ được lựa chọn độc lập.

Mục đích là xác định trước:

- sorted returns;
- empirical 5th percentile;
- VaR magnitude;
- expected result để Người A cross-check.

Phép tính này không sử dụng hàm `calculate_historical_var()` của Người A.

---

## 2. Vector lợi suất độc lập

Vector gồm 20 lợi suất danh mục:

```text
 0.010
-0.020
 0.030
-0.010
 0.000
 0.040
-0.030
 0.020
-0.080
 0.010
-0.005
 0.022
-0.040
 0.003
 0.014
-0.035
 0.018
-0.015
 0.025
 0.005
 ## Result

| Item | Value |
|------|------:|
| Observations | 20 |
| q05 | -0.042 |
| Historical VaR | 0.042 |

**Manual calculation result: PASS**