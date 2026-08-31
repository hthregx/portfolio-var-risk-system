# Risk-oriented EDA Notes

## 1. Ngày thực hiện

Ngày thực hiện: 01/08/2026

---

## 2. Mục tiêu

Thực hiện phân tích khám phá dữ liệu (Risk-oriented Exploratory Data Analysis) cho ba mã cổ phiếu:

- HPG
- FPT
- MWG

Mục tiêu là phân tích đặc điểm của lợi suất (daily log return), đánh giá outlier, mối tương quan giữa các cổ phiếu, rolling volatility và hành vi phần đuôi của phân phối nhằm chuẩn bị dữ liệu cho mô hình Value at Risk (VaR).

---

## 3. Outlier Analysis

Đã thực hiện:

- Tính daily log return cho HPG, FPT và MWG.
- Xác định:
  - Return lớn nhất.
  - Return nhỏ nhất.
  - Percentile.
  - Khoảng tứ phân vị (IQR).
  - Các quan sát vượt ngưỡng IQR.
- Không tự động xóa outlier.
- Các phiên biến động mạnh chỉ được gắn cờ để phục vụ phân tích rủi ro và kiểm tra Corporate Action.

Kết quả:

- HPG có 128 quan sát vượt ngưỡng IQR.
- FPT có 116 quan sát vượt ngưỡng IQR.
- MWG có 120 quan sát vượt ngưỡng IQR.

Hình tạo ra:

- figures/eda/05_return_boxplot.png

---

## 4. Correlation Analysis

Đã tính ma trận tương quan Pearson dựa trên daily log return của:

- HPG
- FPT
- MWG

Kết quả:

- Cặp có tương quan cao nhất: FPT – MWG (0.5423).
- Cặp có tương quan thấp nhất: HPG – FPT (0.4926).

Nhận xét:

- Ba cổ phiếu đều có tương quan dương mức trung bình.
- HPG và FPT có mức tương quan thấp nhất nên có khả năng mang lại lợi ích đa dạng hóa tốt hơn.

Hình tạo ra:

- figures/eda/06_return_correlation.png

---

## 5. Rolling Volatility

Đã tính rolling standard deviation với cửa sổ 20 phiên giao dịch.

Kết quả:

- MWG có rolling volatility trung bình cao nhất.
- MWG cũng có rolling volatility cực đại cao nhất.
- Biểu đồ cho thấy hiện tượng volatility clustering xuất hiện rõ ràng.

Hình tạo ra:

- figures/eda/07_rolling_volatility_20d.png

---

## 6. Tail Behavior

Đã phân tích phần đuôi phân phối của daily log return.

Các chỉ tiêu tính toán:

- Quantile 1%
- Quantile 5%
- Quantile 95%
- Quantile 99%
- Skewness
- Kurtosis

Kết quả:

- MWG có return_q05 âm nhất.
- Cả ba cổ phiếu đều có skewness âm.
- Cả ba cổ phiếu đều có excess kurtosis dương, cho thấy phân phối có đuôi dày hơn phân phối chuẩn.
- return_q05 được sử dụng làm cơ sở cho VaR 95%.

Hình tạo ra:

- figures/eda/08_left_tail_returns.png

---

## 7. Notebook

Notebook thực hiện:

- notebooks/01_eda_risk_analysis.ipynb

Notebook bao gồm:

- Outlier Analysis
- Correlation Analysis
- Rolling Volatility
- Tail Behavior
- Key Findings

---

## 8. Deliverables

Notebook:

- notebooks/01_eda_risk_analysis.ipynb

Figures:

- figures/eda/05_return_boxplot.png
- figures/eda/06_return_correlation.png
- figures/eda/07_rolling_volatility_20d.png
- figures/eda/08_left_tail_returns.png

---

## 9. Kết luận

Đã hoàn thành toàn bộ yêu cầu của phần Risk-oriented EDA:

- Hoàn thành Outlier Analysis.
- Hoàn thành Correlation Analysis.
- Hoàn thành Rolling Volatility Analysis.
- Hoàn thành Tail Behavior Analysis.
- Sinh đầy đủ 4 biểu đồ theo yêu cầu.
- Hoàn thiện notebook riêng để phục vụ merge vào notebook chính sau khi Pull Request được chấp nhận.

---

## 10. Git

Branch:

feature/eda-risk

Commit:

feat: add risk-oriented exploratory analysis

Pull Request:

feature/eda-risk → feature/eda-analysis