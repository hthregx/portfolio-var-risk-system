# Chapter 1 — Introduction

## 1.1 Background

Danh mục cổ phiếu chịu rủi ro thị trường do biến động giá tài sản.
Việc định lượng mức tổn thất tiềm năng là một phần quan trọng trong
quản trị rủi ro tài chính.

Value at Risk (VaR) cung cấp một thước đo định lượng cho downside risk
trong một khoảng thời gian và mức tin cậy xác định.

Dự án này tập trung xây dựng và đánh giá hệ thống dự báo VaR một ngày
cho danh mục cổ phiếu Việt Nam.

---

## 1.2 Problem Definition

Danh mục nghiên cứu gồm ba cổ phiếu:

- HPG;
- FPT;
- MWG.

Danh mục được xây dựng theo phương pháp equal-weight.

Biến mục tiêu chung của các mô hình là:

```text
portfolio_simple_return
```

Dự án thực hiện dự báo downside risk với horizon một trading day.

Mỗi forecast chỉ được sử dụng thông tin có sẵn trước `target_date`
nhằm tránh look-ahead bias.

---

## 1.3 Project Objectives

Mục tiêu của dự án là xây dựng một quy trình có thể tái lập để dự báo,
backtest và so sánh one-day portfolio VaR.

Các mục tiêu chính gồm:

1. xây dựng và kiểm tra dữ liệu portfolio return;
2. triển khai các phương pháp dự báo VaR;
3. chuẩn hóa walk-forward evaluation;
4. đảm bảo không xảy ra look-ahead bias;
5. đánh giá các mô hình bằng cùng bộ metrics;
6. hỗ trợ so sánh model trên cùng portfolio và evaluation period.

---

## 1.4 Forecasting Methods

Dự án nghiên cứu ba phương pháp:

1. Historical Simulation;
2. EWMA VaR;
3. Gradient Boosting Quantile Regression.

Historical Simulation sử dụng phân phối thực nghiệm của historical
returns.

EWMA sử dụng volatility estimate có trọng số giảm dần theo thời gian.

Gradient Boosting Quantile Regression được sử dụng để dự báo trực tiếp
conditional return quantile.

Ba phương pháp được thiết kế để sử dụng cùng portfolio target,
forecast horizon và evaluation framework.

---

## 1.5 Research Questions

Dự án tập trung trả lời các câu hỏi:

1. Các phương pháp dự báo one-day portfolio downside risk như thế nào?
2. Tần suất VaR violations của từng phương pháp có phù hợp với mức
   tail probability kỳ vọng hay không?
3. Chất lượng quantile forecast khác nhau như thế nào giữa các model?
4. Mức VaR trung bình và hành vi exception days khác nhau như thế nào?
5. Các model hoạt động như thế nào khi được đánh giá bằng cùng
   walk-forward framework?

---

## 1.6 Project Scope

Dự án sử dụng daily data của:

- HPG;
- FPT;
- MWG.

Ba tài sản được kết hợp thành equal-weight portfolio.

Phạm vi modeling tập trung vào:

```text
one-day-ahead VaR
alpha = 0.05
confidence level = 95%
```

Các model được đánh giá trên cùng portfolio-return target và sử dụng
common prediction schema để hỗ trợ cross-model comparison.

---

## 1.7 System Architecture

Pipeline tổng quát của dự án:

```text
Raw Data
    ↓
Data Validation
    ↓
Portfolio Construction
    ↓
Portfolio Returns
    ↓
Training Window
    ↓
VaR Model
    ↓
Walk-Forward Evaluation
    ↓
Predictions
    ↓
Backtesting Metrics
    ↓
Model Comparison
```

Model-specific forecasting logic được tách khỏi common walk-forward
runner.

Cách tổ chức này cho phép Historical Simulation, EWMA và Gradient
Boosting sử dụng cùng evaluation framework mà không đưa model-specific
logic vào generic runner.

---

## 1.8 Expected Contribution

Dự án hướng tới xây dựng một framework có thể tái lập cho việc dự báo
và đánh giá portfolio VaR.

Các đóng góp dự kiến gồm:

- common data and portfolio pipeline;
- nhiều phương pháp dự báo VaR;
- common walk-forward evaluation;
- no-lookahead verification;
- common prediction schema;
- consistent evaluation metrics;
- reproducible tests và documentation.

Các kết luận về relative model performance chỉ được đưa ra sau khi
Historical Simulation, EWMA và Gradient Boosting hoàn thành evaluation
trên cùng framework.