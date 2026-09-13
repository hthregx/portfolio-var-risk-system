# Historical Simulation Methodology Notes

**Reviewer:** Người B  
**Implementation owner:** Người A  
**Notebook:** `notebooks/03_historical_simulation.ipynb`  
**Review scope:** Independent validation  
**Confidence level:** 95%  
**Alpha:** 0.05  
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

Phần tính empirical quantile thuộc model Historical Simulation.
Common walk-forward runner chỉ gọi model thông qua public forecast
interface và không tự triển khai lại phép tính quantile.

---

## 2. Rolling Historical VaR

Historical baseline sử dụng rolling window gồm **250 observations**
để tạo forecast one-day-ahead.

Với mỗi `target_date`, estimation window chỉ chứa các portfolio returns
xuất hiện trước target observation.

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
one-day-ahead forecast
        ↓
realized target return
```

`forecast_date` tương ứng với ngày cuối cùng của estimation window,
trong khi `target_date` là trading day tiếp theo cần được đánh giá.

Do đó:

```text
forecast_date < target_date
```

Target return không được đưa vào estimation window. Điều này đảm bảo
forecast chỉ sử dụng thông tin có sẵn trước `target_date` và tránh
look-ahead bias.

---

## 3. Backtest và VaR Violation

Sau khi tạo forecast, realized portfolio return tại `target_date`
được so sánh với `quantile_return`.

Một VaR violation xảy ra khi:

```text
target_return < quantile_return
```

Ngược lại:

```text
target_return >= quantile_return
```

được xem là non-violation.

Nếu:

```text
target_return == quantile_return
```

thì observation không được tính là violation.

Backtest phải so sánh actual return với **negative quantile threshold**,
không phải với positive VaR magnitude.

Không sử dụng:

```text
target_return < historical_var
```

vì `historical_var` được biểu diễn dưới dạng độ lớn tổn thất không âm.

---

## 4. Calibration

Với confidence level 95%, expected violation rate là **5%**.

Kết quả Historical baseline:

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

## 5. Evaluation Metrics

Historical Simulation được đánh giá bằng bộ metrics chung của dự án.

### Violation Rate

Violation Rate đo tỷ lệ realized returns thấp hơn forecast quantile:

```text
Violation Rate = Number of Violations / Number of Forecasts
```

Metric này được sử dụng để đánh giá calibration của VaR forecast so với
nominal tail probability `alpha = 0.05`.

### Pinball Loss

Pinball Loss đánh giá chất lượng của quantile forecast tại
`alpha = 0.05`.

Common prediction output phải giữ cả:

```text
actual_return
quantile_return
```

để Pinball Loss có thể được tính nhất quán cho Historical Simulation,
EWMA và Gradient Boosting.

Historical runner không cần tính Pinball Loss trực tiếp. Metric có thể
được tính ở evaluation layer từ canonical prediction output.

### Average VaR

Average VaR đo mức độ rủi ro trung bình được forecast trong toàn bộ
evaluation period.

VaR được giữ theo sign convention:

```text
var >= 0
```

trong khi `quantile_return` giữ negative return threshold.

### Exception Days

Exception days là các `target_date` thỏa:

```text
actual_return < quantile_return
```

Danh sách exception days được giữ để phục vụ kiểm tra tail events,
stability analysis và cross-model comparison.

Các metrics trên được sử dụng thống nhất để phục vụ so sánh Historical
Simulation với EWMA và Gradient Boosting.

---

## 6. Common Prediction Schema

Historical Simulation được đưa qua common walk-forward runner để tạo
prediction output có schema dùng chung cho các model.

Core fields gồm:

```text
forecast_date
target_date
actual_return
quantile_return
var
violation
method
```

Runner giữ thêm metadata:

```text
window_end_date
observations
evaluation_mode
runtime_seconds
```

Trong đó:

- `forecast_date`: ngày forecast được tạo;
- `target_date`: ngày chứa realized return cần dự báo;
- `actual_return`: realized portfolio return tại `target_date`;
- `quantile_return`: forecast quantile tại `alpha = 0.05`;
- `var`: non-negative VaR magnitude;
- `violation`: kết quả kiểm tra actual return với quantile threshold;
- `method`: model identifier;
- `observations`: số observations được sử dụng để estimation;
- `evaluation_mode`: rolling hoặc expanding;
- `runtime_seconds`: runtime metadata của walk-forward execution.

`quantile_return` được giữ dưới tên chung thay vì đổi thành field riêng
cho Historical Simulation vì field này còn được sử dụng cho Pinball Loss
và cross-model comparison.

---

## 7. No-Lookahead Guarantee

Common walk-forward evaluation tuân theo nguyên tắc:

```text
training observations < target observation
```

Đối với rolling Historical baseline:

```text
training window = previous 250 observations
target          = next observation
```

Target observation không được sử dụng để:

- xây dựng estimation window;
- tính empirical quantile;
- tính VaR forecast.

`forecast_date` là ngày cuối cùng của training window và phải đứng trước
`target_date`.

Synthetic no-lookahead tests và Historical regression tests được sử dụng
để kiểm tra invariant này.

---

## 8. Historical Runner Regression

Common walk-forward runner được kiểm tra độc lập bằng cách tái tạo
Historical baseline thông qua public Historical forecast interface.

Runner không triển khai lại:

```python
np.quantile(...)
```

mà chuyển training returns cho Historical model.

Regression validation kiểm tra:

- number of prediction rows;
- `quantile_return`;
- VaR;
- forecast dates;
- target dates;
- violation flags.

Acceptance criteria:

```text
Rows                 = 1387
Max quantile diff    <= 1e-12
Max VaR diff         <= 1e-12
Forecast dates       exact match
Target dates         exact match
Violations           exact match
```

Historical regression test trên common walk-forward runner đã PASS.

---

## 9. Independent Validation

Người B thực hiện kiểm tra độc lập mà không viết lại pipeline của Người A:

- tính lại number of forecasts, number of violations và violation rate;
- tính lại average, minimum và maximum Historical VaR;
- so sánh metrics với Người A bằng tolerance `1e-12`;
- kiểm tra first 5, middle 5 và last 5 violations;
- xác nhận các violation đều thỏa
  `target_return < quantile_return`;
- kiểm tra calibration interpretation có phù hợp với actual violation rate;
- kiểm tra common walk-forward output với Historical baseline;
- kiểm tra forecast/target date alignment;
- kiểm tra no-lookahead behavior;
- xác nhận common prediction schema giữ `actual_return` và
  `quantile_return` cho downstream evaluation.

Các kiểm tra không phát hiện inconsistency trong metrics, violation logic,
date alignment hoặc calibration interpretation.

**Independent Validation Result: PASS**

Chi tiết kết quả backtest được trình bày tại
`docs/backtest-result-summary.md`.

Common walk-forward behavior được mô tả tại
`docs/walk-forward-runner.md`.

Release-level QA được kiểm tra riêng trong checklist `v0.1.0`.