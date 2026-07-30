# Quy tắc kiểm tra và làm sạch dữ liệu OHLCV

## 1. Mục đích

Tài liệu này mô tả các quy tắc kiểm tra cấu trúc và làm sạch dữ liệu
giao dịch cổ phiếu theo ngày trong dự án:

**Xây dựng hệ thống dự báo và kiểm định Value at Risk một ngày cho danh mục cổ phiếu bằng Historical Simulation, EWMA và Gradient Boosting.**

Các quy tắc trong tài liệu được triển khai tại:

```text
src/validation_rules.py
```

Phần này tập trung vào việc đảm bảo dữ liệu OHLCV có cấu trúc hợp lệ
trước khi được sử dụng cho các bước:

- Đồng bộ ngày giao dịch.
- Tính lợi suất cổ phiếu.
- Tính lợi suất danh mục.
- Dự báo Value at Risk.
- Backtesting các mô hình VaR.

---

## 2. Phạm vi dữ liệu

Các quy tắc được thiết kế cho dữ liệu giao dịch cổ phiếu theo ngày với
cấu trúc OHLCV.

OHLCV gồm:

- `open`: giá mở cửa.
- `high`: giá cao nhất.
- `low`: giá thấp nhất.
- `close`: giá đóng cửa.
- `volume`: khối lượng giao dịch.

Các mã cổ phiếu được sử dụng trong dự án gồm:

- HPG.
- FPT.
- MWG.

Trong phạm vi công việc của thành viên A, các quy tắc đã được chạy thử
trên dữ liệu thực tế của HPG và FPT.

---

## 3. Cấu trúc dữ liệu bắt buộc

Mỗi bộ dữ liệu đầu vào phải có đầy đủ các cột sau:

| Tên cột | Kiểu dữ liệu dự kiến | Ý nghĩa |
|---|---|---|
| `date` | datetime | Ngày giao dịch |
| `open` | numeric | Giá mở cửa |
| `high` | numeric | Giá cao nhất |
| `low` | numeric | Giá thấp nhất |
| `close` | numeric | Giá đóng cửa |
| `volume` | numeric | Khối lượng giao dịch |

Danh sách cột bắt buộc trong code:

```python
REQUIRED_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
]
```

Nếu dữ liệu nguồn có cột `time` nhưng chưa có cột `date`, cột `time`
được đổi tên thành `date`.

Pipeline không được tiếp tục khi thiếu một hoặc nhiều cột bắt buộc.

---

## 4. Chuẩn hóa tên cột

Hàm phụ trách:

```python
normalize_column_names(data)
```

Tên cột được chuẩn hóa theo các quy tắc sau:

1. Chuyển tên cột về kiểu chuỗi.
2. Xóa khoảng trắng ở đầu và cuối.
3. Chuyển toàn bộ ký tự thành chữ thường.
4. Thay khoảng trắng bằng dấu gạch dưới.
5. Thay dấu gạch ngang bằng dấu gạch dưới.
6. Đổi tên `time` thành `date` khi dữ liệu chưa có cột `date`.

Ví dụ:

```text
" Time "          -> "date"
"Open"            -> "open"
"Adjusted-Close"  -> "adjusted_close"
```

Hàm không chỉnh sửa trực tiếp DataFrame đầu vào.

Một bản sao được tạo trước khi chuẩn hóa:

```python
normalized_data = data.copy()
```

Điều này giúp bảo vệ dữ liệu gốc và hạn chế các thay đổi ngoài ý muốn.

---

## 5. Kiểm tra các cột bắt buộc

Hàm phụ trách:

```python
validate_required_columns(data)
```

Hàm kiểm tra DataFrame có đầy đủ các cột trong `REQUIRED_COLUMNS` hay
không.

Nếu thiếu cột, hàm phát sinh `ValueError`.

Ví dụ dữ liệu thiếu cột `volume`:

```text
Missing required columns: ['volume']
```

Nếu thiếu nhiều cột, tất cả tên cột bị thiếu được báo cùng lúc.

Ví dụ:

```text
Missing required columns: ['high', 'volume']
```

Việc báo toàn bộ cột bị thiếu trong một lần giúp người phát triển sửa
dữ liệu đầu vào thuận tiện hơn.

---

## 6. Chuẩn hóa ngày giao dịch

Hàm phụ trách:

```python
parse_trading_dates(date_series)
```

Cột ngày giao dịch được chuyển sang kiểu datetime của Pandas.

Các quy tắc được áp dụng gồm:

1. Chấp nhận dữ liệu có nhiều định dạng ngày khác nhau.
2. Dùng `errors="coerce"` để chuyển ngày không hợp lệ thành `NaT`.
3. Dùng `format="mixed"` để hỗ trợ nhiều định dạng trong cùng một cột.
4. Chuẩn hóa dữ liệu thời gian về UTC.
5. Chuyển về múi giờ `Asia/Ho_Chi_Minh`.
6. Loại bỏ thông tin múi giờ sau khi chuyển đổi.
7. Đưa phần giờ về `00:00:00`.

Ví dụ:

```text
"2026-07-28"                  -> 2026-07-28
"2026-07-29 07:00:00+07:00"  -> 2026-07-29
"invalid-date"                -> NaT
None                          -> NaT
```

Các giá trị ngày không hợp lệ không được tự động thay thế bằng ngày
ước lượng.

`NaT` được giữ lại để quá trình kiểm tra cấu trúc có thể phát hiện và
ghi nhận lỗi.

---

## 7. Chuẩn hóa dữ liệu số

Các cột sau được chuyển sang kiểu dữ liệu số:

```text
open
high
low
close
volume
```

Việc chuyển đổi được thực hiện bằng:

```python
pd.to_numeric(
    prepared_data[column],
    errors="coerce",
)
```

Giá trị không thể chuyển sang dạng số sẽ trở thành `NaN`.

Ví dụ:

```text
"25.5"     -> 25.5
"1000000"  -> 1000000
"unknown"  -> NaN
```

Các giá trị `NaN` sau đó được phát hiện trong quá trình kiểm tra
`missing_rows`.

---

## 8. Kiểm tra lỗi cấu trúc

Hàm phụ trách:

```python
evaluate_structural_rules(data)
```

Hàm này chỉ đo lường và báo cáo các vấn đề trong dữ liệu.

Hàm không xóa hoặc chỉnh sửa dòng dữ liệu.

Các chỉ số trả về gồm:

| Chỉ số | Ý nghĩa |
|---|---|
| `date_parse_failures` | Số giá trị ngày bị chuyển thành `NaT` |
| `missing_rows` | Số dòng thiếu ít nhất một giá trị bắt buộc |
| `duplicate_date_rows` | Số dòng tham gia vào ngày giao dịch bị trùng |
| `nonpositive_price_rows` | Số dòng có ít nhất một giá OHLC nhỏ hơn hoặc bằng 0 |
| `negative_volume_rows` | Số dòng có khối lượng giao dịch âm |
| `dates_out_of_order` | Cho biết ngày giao dịch có đang sai thứ tự tăng dần hay không |

Ví dụ kết quả:

```python
{
    "date_parse_failures": 1,
    "missing_rows": 1,
    "duplicate_date_rows": 2,
    "nonpositive_price_rows": 1,
    "negative_volume_rows": 1,
    "dates_out_of_order": True,
}
```

Một dòng có thể đồng thời vi phạm nhiều quy tắc.

Ví dụ, một dòng có thể vừa:

- Có ngày không hợp lệ.
- Có giá mở cửa bằng 0.
- Có volume âm.

Do đó, không được cộng trực tiếp các chỉ số lỗi để suy ra chính xác số
dòng cần loại bỏ.

---

## 9. Quy tắc phát hiện ngày bị trùng

Một ngày được xem là bị trùng khi xuất hiện nhiều hơn một lần trong
cùng bộ dữ liệu của một mã cổ phiếu.

Ví dụ:

```text
2026-07-28
2026-07-28
```

Cả hai dòng đều tham gia vào lỗi trùng ngày.

Do đó:

```text
duplicate_date_rows = 2
```

không phải:

```text
duplicate_date_rows = 1
```

Khi làm sạch, bản ghi xuất hiện cuối cùng được giữ lại.

Quy tắc này được triển khai bằng:

```python
.drop_duplicates(
    subset="date",
    keep="last",
)
```

Việc giữ bản ghi cuối phải được ghi nhận rõ trong tài liệu để đảm bảo
quy trình có thể kiểm tra và tái lập.

---

## 10. Kiểm tra giá không dương

Các cột giá gồm:

```text
open
high
low
close
```

Mỗi giá trị phải lớn hơn 0.

Một dòng bị xem là không hợp lệ khi có ít nhất một giá trị:

```text
<= 0
```

Ví dụ:

| open | high | low | close | Kết quả |
|---:|---:|---:|---:|---|
| 25.0 | 25.8 | 24.8 | 25.5 | Hợp lệ |
| 0.0 | 25.8 | 24.8 | 25.5 | Không hợp lệ |
| 25.0 | -1.0 | 24.8 | 25.5 | Không hợp lệ |

Giá bằng 0 hoặc giá âm không phù hợp với dữ liệu giao dịch cổ phiếu và
bị loại trong bước làm sạch cấu trúc.

---

## 11. Kiểm tra khối lượng giao dịch

Cột `volume` phải lớn hơn hoặc bằng 0.

Điều kiện hợp lệ:

```text
volume >= 0
```

Volume âm được xem là lỗi dữ liệu.

Ví dụ:

| volume | Kết quả |
|---:|---|
| 1000000 | Hợp lệ |
| 0 | Hợp lệ về mặt cấu trúc |
| -100 | Không hợp lệ |

Volume bằng 0 chưa bị loại trong bước kiểm tra cấu trúc vì có thể xuất
hiện trong một số trường hợp dữ liệu hoặc phiên không phát sinh giao
dịch.

Các trường hợp volume bằng 0 cần được phân tích thêm ở bước kiểm tra
chất lượng dữ liệu chuyên sâu.

---

## 12. Làm sạch lỗi cấu trúc

Hàm phụ trách:

```python
clean_structural_issues(data, ticker)
```

Hàm thực hiện các bước sau:

1. Chuẩn hóa tên cột.
2. Kiểm tra các cột bắt buộc.
3. Chuẩn hóa ngày giao dịch.
4. Chuyển OHLCV sang kiểu dữ liệu số.
5. Loại dòng thiếu giá trị bắt buộc.
6. Loại dòng có giá OHLC nhỏ hơn hoặc bằng 0.
7. Loại dòng có volume âm.
8. Sắp xếp dữ liệu theo ngày tăng dần.
9. Khi trùng ngày, giữ bản ghi cuối cùng.
10. Chuẩn hóa mã cổ phiếu thành chữ in hoa.
11. Thêm cột `ticker`.
12. Đặt lại chỉ số dòng.

Cấu trúc dữ liệu sau khi làm sạch:

```text
date | ticker | open | high | low | close | volume
```

Ví dụ:

```text
ticker đầu vào: "hpg"
ticker đầu ra:  "HPG"
```

---

## 13. Điều kiện để một dòng được giữ lại

Một dòng chỉ được giữ khi đồng thời thỏa mãn ba nhóm điều kiện:

```text
Đầy đủ giá trị bắt buộc
AND
Tất cả giá OHLC lớn hơn 0
AND
Volume lớn hơn hoặc bằng 0
```

Trong code:

```python
valid_rows = (
    complete_rows
    & positive_prices
    & nonnegative_volume
)
```

Nếu một trong ba điều kiện không đạt, dòng đó bị loại khỏi DataFrame
đã làm sạch.

---

## 14. Những cách xử lý tự động không được phép

Quá trình làm sạch cấu trúc không được:

- Thay giá trị thiếu bằng 0.
- Dùng forward-fill để điền giá cổ phiếu bị thiếu.
- Dùng backward-fill để điền dữ liệu bị thiếu.
- Tự tạo ngày giao dịch mới.
- Sửa ngày không hợp lệ thành một ngày ước lượng.
- Chỉnh sửa trực tiếp dữ liệu trong `data/raw`.
- Ghi đè dữ liệu raw bằng dữ liệu đã làm sạch.
- Tự động thay đổi giá cổ phiếu.
- Loại một phiên chỉ vì giá biến động mạnh.
- Tự động điều chỉnh corporate action khi chưa có bằng chứng.
- Tự động sửa quan hệ OHLC khi phát hiện bất thường.

Các phiên biến động giá lớn chỉ được gắn cờ để kiểm tra thêm.

Việc loại bỏ dữ liệu chỉ được thực hiện đối với các lỗi cấu trúc bắt
buộc đã được định nghĩa rõ trong tài liệu này.

---

## 15. Các nội dung chưa được kiểm tra trong module này

File `src/validation_rules.py` chưa kiểm tra:

- `high` có lớn hơn hoặc bằng `open` hay không.
- `high` có lớn hơn hoặc bằng `close` hay không.
- `high` có lớn hơn hoặc bằng `low` hay không.
- `low` có nhỏ hơn hoặc bằng `open` hay không.
- `low` có nhỏ hơn hoặc bằng `close` hay không.
- Phiên có biến động giá lớn.
- Dấu hiệu chia cổ tức.
- Dấu hiệu chia tách cổ phiếu.
- Corporate action.
- Sự đồng bộ ngày giao dịch giữa HPG, FPT và MWG.
- Sự khác biệt về khoảng ngày giữa các mã.
- Các ngày giao dịch bị thiếu so với lịch thị trường.

Các nội dung trên thuộc pipeline kiểm tra chất lượng tổng hợp do thành
viên B phụ trách.

---

## 16. Kết quả kiểm tra trên dữ liệu mẫu

Dữ liệu mẫu được cố ý tạo với:

- Một ngày không hợp lệ.
- Một ngày xuất hiện hai lần.
- Không có giá không dương.
- Không có volume âm.

Kết quả:

```text
date_parse_failures: 1
missing_rows: 1
duplicate_date_rows: 2
nonpositive_price_rows: 0
negative_volume_rows: 0
dates_out_of_order: False
```

Kích thước dữ liệu:

```text
Original shape: (4, 6)
Cleaned shape: (2, 7)
```

Dữ liệu từ 4 dòng còn 2 dòng vì:

1. Dòng có ngày không hợp lệ bị loại.
2. Hai bản ghi trùng ngày được giữ lại một bản ghi cuối cùng.
3. Cột `ticker` được thêm vào nên số cột tăng từ 6 lên 7.

---

## 17. Kết quả kiểm tra trên dữ liệu HPG

Kết quả dữ liệu HPG:

```text
Ticker: HPG
Raw shape: (1638, 6)
Clean shape: (1638, 7)
Rows removed: 0
```

Chỉ số kiểm tra cấu trúc:

```text
date_parse_failures: 0
missing_rows: 0
duplicate_date_rows: 0
nonpositive_price_rows: 0
negative_volume_rows: 0
dates_out_of_order: False
```

Kiểm tra dữ liệu sau làm sạch:

```text
Dates sorted: True
Duplicate dates: 0
Missing values: 0
Start date: 2020-01-02
End date: 2026-07-28
```

Kết luận:

Dữ liệu HPG đạt các quy tắc kiểm tra cấu trúc được định nghĩa trong
module của thành viên A.

Không có dòng nào bị loại trong quá trình làm sạch cấu trúc.

---

## 18. Kết quả kiểm tra trên dữ liệu FPT

Kết quả dữ liệu FPT:

```text
Ticker: FPT
Raw shape: (1638, 6)
Clean shape: (1638, 7)
Rows removed: 0
```

Chỉ số kiểm tra cấu trúc:

```text
date_parse_failures: 0
missing_rows: 0
duplicate_date_rows: 0
nonpositive_price_rows: 0
negative_volume_rows: 0
dates_out_of_order: False
```

Kiểm tra dữ liệu sau làm sạch:

```text
Dates sorted: True
Duplicate dates: 0
Missing values: 0
Start date: 2020-01-02
End date: 2026-07-28
```

Kết luận:

Dữ liệu FPT đạt các quy tắc kiểm tra cấu trúc được định nghĩa trong
module của thành viên A.

Không có dòng nào bị loại trong quá trình làm sạch cấu trúc.

---

## 19. Bảng tổng hợp dữ liệu thực tế

| Mã cổ phiếu | Số dòng raw | Số dòng sạch | Số dòng bị loại | Ngày bắt đầu | Ngày kết thúc |
|---|---:|---:|---:|---|---|
| HPG | 1638 | 1638 | 0 | 2020-01-02 | 2026-07-28 |
| FPT | 1638 | 1638 | 0 | 2020-01-02 | 2026-07-28 |

Cả hai mã đều có:

```text
Date parse failures: 0
Missing rows: 0
Duplicate-date rows: 0
Non-positive price rows: 0
Negative-volume rows: 0
Dates out of order: False
```

Kết quả này chỉ xác nhận HPG và FPT đạt các điều kiện cấu trúc cơ bản.

Kết quả chưa khẳng định dữ liệu hoàn toàn không có:

- Quan hệ OHLC bất hợp lý.
- Biến động giá bất thường.
- Corporate action.
- Sai lệch do giá chưa điều chỉnh.
- Ngày giao dịch bị thiếu so với lịch thị trường.

---

## 20. Kiểm thử tự động

File unit test:

```text
tests/test_validation_rules.py
```

Có tổng cộng 6 unit test:

1. Kiểm tra chuẩn hóa tên cột.
2. Kiểm tra phát hiện cột bắt buộc bị thiếu.
3. Kiểm tra chuyển đổi nhiều định dạng ngày.
4. Kiểm tra phát hiện lỗi cấu trúc.
5. Kiểm tra loại bỏ dòng không hợp lệ.
6. Kiểm tra sắp xếp ngày và xử lý ngày bị trùng.

Lệnh chạy:

```powershell
python -m pytest tests\test_validation_rules.py -v
```

Kết quả thực tế:

```text
6 passed
```

Các test bao gồm cả:

- Trường hợp dữ liệu hợp lệ.
- Trường hợp thiếu cột.
- Ngày không hợp lệ.
- Ngày bị trùng.
- Giá không dương.
- Volume âm.
- Ngày sai thứ tự.
- Kiểm tra giữ bản ghi cuối khi trùng ngày.
- Kiểm tra mã cổ phiếu được chuyển thành chữ in hoa.
- Kiểm tra DataFrame đầu vào không bị chỉnh sửa trực tiếp.

---

## 21. Các hàm do thành viên A xây dựng

Thành viên A xây dựng 5 hàm:

### 21.1. `normalize_column_names()`

Chuẩn hóa tên cột về cùng một quy ước.

### 21.2. `validate_required_columns()`

Kiểm tra dữ liệu có đầy đủ các cột OHLCV bắt buộc.

### 21.3. `parse_trading_dates()`

Chuẩn hóa ngày giao dịch và chuyển ngày không hợp lệ thành `NaT`.

### 21.4. `evaluate_structural_rules()`

Đo lường các lỗi cấu trúc nhưng không xóa dữ liệu.

### 21.5. `clean_structural_issues()`

Loại các dòng không hợp lệ, sắp xếp ngày, xử lý duplicate và thêm
ticker.

---

## 22. Các file do thành viên A phụ trách

```text
src/validation_rules.py
tests/test_validation_rules.py
docs/data-validation-rules.md
```

Trong đó:

- `src/validation_rules.py`: chứa các quy tắc validation và cleaning.
- `tests/test_validation_rules.py`: chứa unit test.
- `docs/data-validation-rules.md`: mô tả quy tắc và kết quả kiểm tra.

---

## 23. Mối liên hệ với phần của thành viên B

Thành viên B sử dụng các hàm của thành viên A bằng cách import từ:

```python
from src.validation_rules import (
    clean_structural_issues,
    evaluate_structural_rules,
)
```

Luồng tích hợp:

```text
data_validation.py của thành viên B
                ↓
Gọi validation_rules.py của thành viên A
                ↓
Kiểm tra HPG, FPT và MWG
                ↓
Kiểm tra quan hệ OHLC
                ↓
Gắn cờ biến động bất thường
                ↓
Lưu dữ liệu sạch
                ↓
Xuất báo cáo chất lượng tổng hợp
```

Thành viên A xây dựng bộ quy tắc xử lý một DataFrame.

Thành viên B sử dụng bộ quy tắc đó để xây pipeline xử lý nhiều mã cổ
phiếu và xuất báo cáo tổng hợp.

---

## 24. Ngày thực hiện

Ngày thực hiện thực tế:

```text
30/07/2026
```

Công việc này được thực hiện trước nhiệm vụ dự kiến của ngày:

```text
31/07/2026
```

---