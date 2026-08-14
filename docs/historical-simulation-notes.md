# Historical Simulation Methodology Notes

**Reviewer:** Người B  
**Implementation owner:** Người A  
**Notebook:** `notebooks/03_historical_simulation.ipynb`  
**Review scope:** Independent validation  
**Confidence level:** 95%  
**Left-tail probability:** 5%  
**Rolling window:** 250 observations  

---

## 1. Historical Simulation

Historical Simulation ước lượng Value at Risk (VaR) trực tiếp từ phân
phối thực nghiệm của lợi suất lịch sử, không yêu cầu giả định lợi suất
tuân theo phân phối chuẩn.

Với confidence level 95%, mô hình sử dụng empirical quantile tại mức 5%:

```python
np.quantile(
    returns.to_numpy(),
    0.05,
    method="linear",
)
```

`quantile_return` là ngưỡng lợi suất tại đuôi trái và thường mang giá trị
âm. Historical VaR được biểu diễn dưới dạng độ lớn tổn thất không âm:

```text
historical_var = max(0, -quantile_return)
```

Do đó:

- `quantile_return`: negative return threshold;
- `historical_var`: positive VaR magnitude.

---

## 2. Rolling Historical VaR

Mỗi forecast sử dụng rolling window gồm **250 observations** trước
`target_date`.

Target return không nằm trong estimation window, giúp tránh look-ahead
bias.

Quy trình:

```text
250 historical returns
        ↓
5% empirical quantile
        ↓
quantile_return
        ↓
historical_var
        ↓
next-day forecast
```

---

## 3. Backtest và VaR Violation

Một VaR violation xảy ra khi:

```text
target_return < quantile_return
```

Ngược lại:

```text
target_return >= quantile_return
```

được xem là non-violation.

Backtest phải so sánh actual return với **negative quantile threshold**,
không phải với positive VaR magnitude.

---

## 4. Calibration

Với confidence level 95%, expected violation rate là **5%**.

Kết quả backtest:

| Metric | Value |
|---|---:|
| Number of forecasts | 1387 |
| Number of violations | 75 |
| Observed violation rate | 5.407354% |
| Expected violation rate | 5.000000% |

Observed violation rate cao hơn nominal level khoảng **0.407354
percentage points**.

Kết quả này phù hợp với **mild undercoverage**, tức mô hình có xu hướng
nhẹ đánh giá thấp one-day downside tail risk trong giai đoạn backtest.
Do violation rate cao hơn 5%, không phù hợp để diễn giải mô hình là
`overly conservative`.

Đây là descriptive calibration assessment; chưa thực hiện formal
coverage hypothesis test.

---

## 5. Independent Validation

Người B thực hiện kiểm tra độc lập mà không viết lại pipeline của Người A:

- tính lại number of forecasts, number of violations và violation rate;
- tính lại average, minimum và maximum Historical VaR;
- so sánh metrics với Người A bằng tolerance `1e-12`;
- kiểm tra first 5, middle 5 và last 5 violations;
- xác nhận các violation đều thỏa
  `target_return < quantile_return`;
- kiểm tra calibration interpretation có phù hợp với actual violation
  rate.

Các kiểm tra không phát hiện inconsistency trong metrics, violation logic
hoặc interpretation.

**Independent Validation Result: PASS**

Chi tiết kết quả được trình bày tại
`docs/backtest-result-summary.md`. Release-level QA được kiểm tra riêng
trong checklist `v0.1.0`.