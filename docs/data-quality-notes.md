# Data Quality Notes

## 1. Ngày thực hiện

Ngày thực hiện: 30/07/2026.

---

## 2. Danh sách ticker

Pipeline được chạy cho các mã:

- HPG
- FPT
- MWG

---

## 3. Kết quả của từng ticker

### HPG

- Raw rows: 1638
- Clean rows: 1638
- Rows removed: 0
- Duplicate date rows: 0
- Invalid OHLC rows: 0
- Extreme close move rows: 0
- Quality status: PASS

### FPT

- Raw rows: 1638
- Clean rows: 1638
- Rows removed: 0
- Duplicate date rows: 0
- Invalid OHLC rows: 0
- Extreme close move rows: 0
- Quality status: PASS

### MWG

- Raw rows: 1638
- Clean rows: 1638
- Rows removed: 0
- Duplicate date rows: 0
- Invalid OHLC rows: 0
- Extreme close move rows: 0
- Quality status: PASS

---

## 4. Dòng bị loại và lý do

Không có dòng dữ liệu nào bị loại trong cả ba ticker.

Kết quả validation cho thấy:

- Không có dữ liệu thiếu.
- Không có ngày giao dịch trùng.
- Không có giá âm hoặc bằng 0.
- Không có khối lượng giao dịch âm.
- Không có lỗi quan hệ OHLC.
- Không có ngày giao dịch sai thứ tự.

---

## 5. Danh sách biến động trên 15%

Không phát hiện phiên giao dịch nào có biến động tuyệt đối của giá đóng cửa lớn hơn 15%.

| Ticker | Extreme Move (>15%) |
|---------|--------------------:|
| HPG | 0 |
| FPT | 0 |
| MWG | 0 |

---

## 6. Nhận xét về Corporate Action

Trong dữ liệu hiện tại không phát hiện biến động giá lớn hơn 15%.

Trong các lần chạy khác, nếu xuất hiện biến động lớn thì cần kiểm tra các sự kiện như:

- Chia cổ tức.
- Chia tách cổ phiếu.
- Gộp cổ phiếu.
- Phát hành thêm.
- Điều chỉnh giá sau corporate action.

Do đó các phiên có biến động lớn chỉ được gắn cờ để kiểm tra, không tự động loại bỏ.

---

## 7. Giới hạn của nguồn dữ liệu

- Dữ liệu phụ thuộc nguồn tải về.
- Pipeline chỉ kiểm tra tính hợp lệ của dữ liệu OHLCV.
- Chưa kết nối trực tiếp với dữ liệu Corporate Action.
- Ngưỡng phát hiện biến động đang cố định ở 15%.

---

## 8. Công việc tiếp theo

- Bổ sung thêm các validation rule.
- Kiểm tra nhiều ticker hơn.
- Tự động sinh báo cáo chi tiết.
- Tích hợp pipeline vào CI/CD.
- Theo dõi Corporate Action để giảm cảnh báo giả.

---

## Kết luận

Pipeline validation đã chạy thành công cho HPG, FPT và MWG.

- Không có dòng dữ liệu bị loại.
- Không có lỗi OHLC.
- Không phát hiện biến động giá trên 15%.

Báo cáo chất lượng dữ liệu đã được tạo thành công.