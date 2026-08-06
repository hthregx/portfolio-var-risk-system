# Risk EDA Notes

## Mục tiêu

Notebook **Risk EDA** được xây dựng nhằm phân tích các đặc điểm của dữ liệu có liên quan trực tiếp đến đo lường rủi ro tài chính và làm cơ sở cho việc xây dựng các mô hình Value at Risk (VaR).

Dữ liệu sử dụng:

- `data/processed/HPG_clean.csv`
- `data/processed/FPT_clean.csv`
- `data/processed/MWG_clean.csv`

---

## 1. Phân tích giá trị ngoại lệ (Return Outlier Analysis)

Phân tích các phiên giao dịch có lợi suất cực trị bằng thống kê mô tả kết hợp với biểu đồ Boxplot.

Các nội dung được đánh giá gồm:

- Giá trị lợi suất nhỏ nhất và lớn nhất.
- Phiên tăng mạnh nhất.
- Phiên giảm mạnh nhất.
- Các quan sát nằm ngoài khoảng IQR (Interquartile Range).

Các giá trị ngoại lệ **không bị loại bỏ** vì chúng phản ánh các biến động thực tế của thị trường và đóng vai trò quan trọng trong việc đánh giá rủi ro tài chính.

---

## 2. Phân tích tương quan lợi suất (Return Correlation)

Lợi suất ngày của HPG, FPT và MWG được ghép theo cùng ngày giao dịch trước khi tính hệ số tương quan Pearson.

Kết quả cho thấy:

- Các hệ số tương quan đều mang giá trị dương.
- Cặp **FPT – MWG** có mức tương quan cao nhất.
- Cặp **HPG – FPT** có mức tương quan thấp nhất.

Phân tích này nhằm đánh giá mức độ phụ thuộc giữa các tài sản và xem xét khả năng đa dạng hóa danh mục. Việc tối ưu danh mục chưa được thực hiện trong giai đoạn này.

---

## 3. Phân tích độ biến động theo thời gian (Rolling Volatility)

Rolling Volatility được tính với cửa sổ **20 phiên giao dịch** nhằm quan sát sự thay đổi của độ biến động theo thời gian.

Kết quả cho thấy:

- Xuất hiện hiện tượng **volatility clustering**.
- Có các giai đoạn thị trường biến động mạnh xen kẽ các giai đoạn ổn định.
- MWG có mức biến động lớn hơn trong nhiều giai đoạn so với HPG và FPT.

Điều này cho thấy giả định phương sai không đổi không còn phù hợp đối với dữ liệu tài chính.

---

## 4. Phân tích Tail Behavior

Đặc điểm đuôi phân phối của lợi suất được đánh giá thông qua các chỉ tiêu:

- q01
- q05
- q95
- q99
- Skewness
- Kurtosis

Trong đó, **q05** được đặc biệt quan tâm vì có liên hệ trực tiếp với phương pháp Historical Simulation VaR ở mức tin cậy 95%.

Tuy nhiên, q05 của từng cổ phiếu chỉ phản ánh mức rủi ro của từng tài sản riêng lẻ và **không phải** là VaR của toàn bộ danh mục đầu tư.

---

## Kết luận

Kết quả Risk EDA cho thấy:

- Dữ liệu xuất hiện nhiều phiên biến động mạnh.
- Phân phối lợi suất có đuôi trái và chứa các giá trị cực trị.
- Độ biến động thay đổi theo thời gian và tồn tại hiện tượng volatility clustering.
- Các cổ phiếu có tương quan dương, cần được xem xét khi xây dựng danh mục.
- Những kết quả này là cơ sở để triển khai các mô hình **Historical Simulation VaR**, **Parametric VaR**, **EWMA** và **GARCH VaR** trong các bước tiếp theo.