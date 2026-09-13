# Weekly Note #2

## Completed

Trong tuần, các thành phần chính sau đã được hoàn thành hoặc kiểm tra:

- Historical Simulation baseline và backtesting;
- Historical VaR stability analysis;
- EWMA VaR implementation và evaluation;
- common walk-forward runner;
- rolling và expanding evaluation support;
- common prediction schema;
- runtime logging và prediction saving;
- no-lookahead verification;
- Historical row-level regression;
- repository và artifact audit;
- Historical Simulation methodology documentation;
- Data section và Chapter 1 drafting.

---

## Evidence

Historical baseline:

| Metric | Result |
|---|---:|
| Number of forecasts | 1387 |
| Number of violations | 75 |
| Violation rate | 5.407354% |
| Expected violation rate | 5.000000% |

Walk-forward verification:

```text
Historical regression: PASS
Prediction rows: 1387
Rolling window: 250
Evaluation mode: rolling
```

Full test suite:

```text
117 passed
0 failed
0 errors
```

Common prediction output giữ các core fields:

```text
forecast_date
target_date
actual_return
quantile_return
var
violation
method
```

---

## Open Risks

- Gradient Boosting evaluation chưa hoàn tất.
- Cross-model comparison chưa thể kết luận.
- Final report metrics cần được review trước khi khóa báo cáo.
- Các model cần tiếp tục sử dụng cùng evaluation dates, sign convention
  và common metrics.

---

## Decisions

Các quyết định chính:

- sử dụng `portfolio_simple_return` làm common modeling target;
- giữ `quantile_return` trong common prediction schema;
- sử dụng one-day-ahead walk-forward evaluation;
- không cho target observation xuất hiện trong training window;
- tách model-specific logic khỏi generic walk-forward runner;
- full processed prediction artifacts không được Git track;
- Historical, EWMA và Gradient Boosting phải được đánh giá trên cùng
  framework trước khi kết luận relative performance.

---

## Git / PR References

Walk-forward implementation:

```text
Commit: 780d032
PR: #31
Merge commit: d93b47a
```

Common walk-forward runner đã được merge vào integration branch và
Historical regression test đã PASS.

---

## Next Week Focus

Các công việc tiếp theo:

- tiếp tục model evaluation;
- hoàn thiện Gradient Boosting Quantile Regression;
- chuẩn hóa cross-model comparison;
- kiểm tra Pinball Loss và các common metrics;
- review exception-day behavior;
- tiếp tục hoàn thiện report;
- chuẩn bị model comparison và final interpretation.