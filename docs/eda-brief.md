# EDA Brief

## 1. Dataset

Bộ dữ liệu sử dụng trong nghiên cứu gồm ba cổ phiếu:

- HPG
- FPT
- MWG

Đặc điểm dữ liệu:

- 1638 quan sát cho mỗi mã.
- Thời gian từ **2020-01-02 đến 2026-07-28**.
- Dữ liệu đã được đồng bộ ngày giao dịch.

---

## 2. Biến động giá

Giá cổ phiếu của ba doanh nghiệp có mức giá khác nhau nên cần chuẩn hóa khi so sánh.

Quan sát cho thấy:

- FPT có xu hướng tăng ổn định.
- HPG tăng mạnh nhưng biến động lớn hơn.
- MWG có mức dao động cao hơn trong nhiều giai đoạn.

---

## 3. Biến động lợi nhuận

Lợi nhuận ngày của cả ba cổ phiếu dao động quanh giá trị 0.

Đặc điểm nổi bật:

- Xuất hiện nhiều phiên tăng hoặc giảm mạnh.
- Có hiện tượng Volatility Clustering.
- Phân phối lợi nhuận không tuân theo phân phối chuẩn.

---

## 4. Volatility

Rolling Volatility cho thấy độ biến động thay đổi theo thời gian.

Mức độ biến động:

1. MWG (cao nhất)
2. HPG
3. FPT (thấp nhất)

Điều này cho thấy rủi ro thị trường không cố định.

---

## 5. Correlation

Ba cổ phiếu có tương quan dương ở mức trung bình.

- FPT – MWG có tương quan cao nhất.
- HPG – FPT có tương quan thấp nhất.

Mặc dù có tương quan nhưng vẫn tồn tại lợi ích đa dạng hóa danh mục.

---

## 6. Tail Behavior

Phân phối lợi nhuận xuất hiện đuôi dày.

Các phiên giảm mạnh xảy ra với tần suất cao hơn so với giả định phân phối chuẩn.

Điều này cho thấy rủi ro cực đoan cần được xem xét khi xây dựng mô hình VaR.

---

## 7. Ý nghĩa đối với VaR

Kết quả EDA cho thấy:

- Không nên giả định lợi nhuận tuân theo phân phối chuẩn.
- Cần sử dụng các mô hình phản ánh biến động thay đổi theo thời gian.
- Nên xem xét các mô hình Historical VaR, EWMA và GARCH.
- Khi tính VaR danh mục cần xét đến tương quan giữa các tài sản.