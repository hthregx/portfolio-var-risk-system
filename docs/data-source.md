# Data Source

## 1. Dữ liệu cần thu thập

- HPG
- FPT
- MWG
- VN-Index hoặc VN30 nếu dữ liệu ổn định

## 2. Tần suất

Dữ liệu giao dịch theo ngày.

## 3. Các trường dữ liệu

- date
- open
- high
- low
- close
- volume

## 4. Khoảng thời gian dự kiến

Từ ngày 01/01/2020 đến thời điểm chốt dữ liệu.

## 5. Nguồn chính

Nguồn dữ liệu dự kiến: vnstock hoặc nguồn dữ liệu đã được
nhóm kiểm tra và giảng viên chấp thuận.

## 6. Cơ chế lưu trữ

- Dữ liệu tải lần đầu được lưu tại `data/raw/`.
- Không tải lại nếu file cache đã tồn tại.
- Dữ liệu đã làm sạch được lưu tại `data/processed/`.
- Repository chỉ lưu dữ liệu mẫu nhỏ tại `data/sample/`.

## 7. Kiểm tra chất lượng

- Kiểm tra ngày trùng.
- Kiểm tra giá trị thiếu.
- Kiểm tra giá OHLC không dương.
- Kiểm tra thứ tự thời gian.
- Kiểm tra các ngày giao dịch không đồng bộ.
- Kiểm tra điều chỉnh giá và corporate actions.

## 8. Rủi ro

- API thay đổi.
- Nguồn dữ liệu tạm thời không hoạt động.
- Dữ liệu chưa điều chỉnh.
- Các mã có ngày giao dịch không đồng bộ.