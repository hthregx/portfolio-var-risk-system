# Weekly Note 01

**Tuần:** 27/07 – 02/08/2026

---

# 1. Công việc đã hoàn thành

Trong tuần này đã hoàn thành các công việc chính sau:

- Chuẩn bị dữ liệu cho ba mã cổ phiếu HPG, FPT và MWG.
- Kiểm tra tính hợp lệ của dữ liệu.
- Đồng bộ ngày giao dịch giữa các mã cổ phiếu.
- Tính toán lợi nhuận ngày (Daily Log Return).
- Thực hiện phân tích dữ liệu khám phá (EDA).
- Phân tích:
  - Giá cổ phiếu
  - Phân phối lợi nhuận
  - Ngoại lệ (Outlier)
  - Tương quan
  - Rolling Volatility
  - Tail Behavior
- Chuẩn bị dữ liệu phục vụ các bước xây dựng mô hình VaR.

---

# 2. Nguồn dữ liệu

Bộ dữ liệu gồm ba mã cổ phiếu:

- HPG
- FPT
- MWG

Đặc điểm dữ liệu:

- **1638 quan sát cho mỗi mã cổ phiếu**
- **Khoảng thời gian:** 2020-01-02 → 2026-07-28

---

# 3. Kết quả kiểm tra dữ liệu

Đã thực hiện các bước kiểm tra:

- Đúng định dạng ngày.
- Không có ngày giao dịch trùng lặp.
- Không thiếu các cột dữ liệu quan trọng.
- Đồng bộ ngày giao dịch giữa các mã cổ phiếu.
- Không phát hiện lỗi trong dữ liệu đã xử lý.

**Kết quả kiểm tra:** **PASS**

Ngoài ra:

- **10 Unit Tests: PASS**

---

# 4. Kết quả EDA

Đã hoàn thành toàn bộ phân tích EDA gồm:

- Phân tích biến động giá
- Phân tích lợi nhuận ngày
- Phân phối lợi nhuận
- Outlier
- Rolling Volatility
- Correlation
- Tail Behavior

**EDA completed.**

---

# 5. Các quyết định kỹ thuật quan trọng

Trong quá trình xử lý dữ liệu đã thống nhất:

- Sử dụng **Log Return** thay cho Simple Return.
- Đồng bộ ngày giao dịch trước khi phân tích.
- Rolling Volatility sử dụng cửa sổ 20 phiên.
- Tail Risk được đánh giá bằng các phân vị thực nghiệm.
- Chỉ sử dụng dữ liệu đã được kiểm định.

---

# 6. Vấn đề gặp phải

Một số khó khăn trong quá trình thực hiện:

- Mức giá giữa các cổ phiếu khác nhau.
- Cần đồng bộ lịch giao dịch.
- Xuất hiện nhiều biến động mạnh.
- Độ biến động thay đổi theo thời gian.

---

# 7. Vấn đề đã giải quyết

Các vấn đề trên đã được xử lý bằng cách:

- Chuẩn hóa dữ liệu trước khi so sánh.
- Đồng bộ ngày giao dịch.
- Sử dụng Log Return.
- Kiểm tra dữ liệu trước khi thực hiện EDA.

---

# 8. Rủi ro còn tồn tại

Các rủi ro hiện tại gồm:

- Phân phối lợi nhuận có đuôi dày.
- Volatility thay đổi theo thời gian.
- Xuất hiện các phiên giảm mạnh.
- Tương quan giữa các cổ phiếu có thể thay đổi trong điều kiện thị trường biến động.

---

# 9. Kế hoạch tuần tới

Các công việc dự kiến:

- Xây dựng Portfolio Return.
- Tính Historical VaR.
- Xây dựng Parametric VaR.
- Áp dụng mô hình EWMA.
- Xây dựng GARCH VaR.
- So sánh kết quả các mô hình VaR.

---

# Trạng thái hiện tại

- HPG/FPT/MWG
- 1638 observations/ticker
- 2020-01-02 → 2026-07-28
- aligned trading dates
- validation PASS
- 10 unit tests PASS
- EDA completed

---
