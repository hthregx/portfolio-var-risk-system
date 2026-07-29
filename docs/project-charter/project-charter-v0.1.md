# PROJECT CHARTER

## 1. Tên đề tài

### Tiếng Việt

Xây dựng hệ thống dự báo và kiểm định Value at Risk một ngày
cho danh mục cổ phiếu bằng Historical Simulation, EWMA
và Gradient Boosting.

### Tiếng Anh

Development of a One-Day Portfolio Value at Risk Forecasting
and Backtesting System Using Historical Simulation, EWMA
and Gradient Boosting.

## 2. Thời gian thực hiện

Từ ngày 27/07/2026 đến ngày 02/09/2026.

## 3. Bối cảnh

Nhà đầu tư và bộ phận quản trị rủi ro cần ước lượng mức lỗ tiềm
năng của danh mục trong điều kiện thị trường bất lợi. Value at Risk
là một chỉ số phổ biến để định lượng mức lỗ này.

Đề tài xây dựng một hệ thống dự báo và kiểm định VaR 95% cho
một phiên giao dịch tiếp theo của danh mục cổ phiếu Việt Nam.

## 4. Câu hỏi nghiên cứu

Historical Simulation, EWMA và Gradient Boosting Quantile
Regression khác nhau như thế nào về khả năng dự báo VaR 95%
một ngày cho danh mục cổ phiếu Việt Nam?

Các câu hỏi phụ:

1. Phương pháp nào có VaR Violation Rate gần mức kỳ vọng 5% nhất?
2. Phương pháp nào có Pinball Loss thấp nhất?
3. Phương pháp nào thích ứng tốt hơn trong giai đoạn thị trường biến động?
4. Gradient Boosting có cải thiện kết quả so với các phương pháp truyền thống không?

## 5. Mục tiêu tổng quát

Xây dựng một hệ thống Khoa học dữ liệu có khả năng thu thập,
xử lý dữ liệu, tính lợi suất danh mục, dự báo VaR 95% một ngày,
kiểm định kết quả và trình bày cảnh báo rủi ro trên giao diện.

## 6. Mục tiêu cụ thể

1. Xây dựng pipeline dữ liệu OHLCV theo ngày.
2. Tính lợi suất của từng cổ phiếu và lợi suất danh mục.
3. Triển khai Historical Simulation.
4. Triển khai EWMA.
5. Triển khai Gradient Boosting Quantile Regression tại alpha = 0.05.
6. Xây dựng quy trình walk-forward backtesting.
7. So sánh các mô hình bằng các chỉ số phù hợp.
8. Xây dựng dashboard Streamlit.
9. Kiểm thử và quản lý mã nguồn bằng GitHub.

## 7. Dữ liệu dự kiến

- HPG
- FPT
- MWG
- VN-Index hoặc VN30 nếu dữ liệu ổn định
- Tần suất: ngày
- Các cột chính: date, open, high, low, close, volume

Danh mục mặc định có trọng số bằng nhau.

## 8. Bộ phương pháp

### Historical Simulation

Ước lượng VaR bằng phân vị 5% của phân phối lợi suất lịch sử
trong một cửa sổ trượt.

### EWMA

Ước lượng phương sai có điều kiện, trong đó các quan sát gần
hiện tại có trọng số lớn hơn các quan sát xa.

### Gradient Boosting Quantile Regression

Sử dụng Gradient Boosting với quantile loss và alpha = 0.05
để dự báo trực tiếp phân vị 5% của lợi suất danh mục ngày tiếp theo.

## 9. Chỉ số đánh giá

- VaR Violation Rate
- Pinball Loss tại alpha = 0.05
- Average VaR
- Số lượng và thời điểm các ngày vi phạm
- Biểu đồ backtesting VaR

## 10. Sản phẩm phải bàn giao

1. Báo cáo thực tập tốt nghiệp.
2. Slide bảo vệ.
3. Toàn bộ mã nguồn và lịch sử GitHub.
4. Video demo dự phòng.
5. Nhật ký thực tập.
6. Nhật ký phát triển.
7. AI Development Log.

## 11. Phạm vi bắt buộc

- Ba cổ phiếu HPG, FPT và MWG.
- VaR 95% cho một ngày tiếp theo.
- Ba phương pháp chính thức.
- Walk-forward evaluation.
- Dashboard Streamlit chạy local.
- Unit test cho các công thức và pipeline quan trọng.

## 12. Không thuộc phạm vi bắt buộc

- GARCH.
- Deep Learning.
- Monte Carlo phức tạp.
- Tối ưu danh mục.
- Giao dịch tự động.
- Dữ liệu intraday.
- Dự báo giá cổ phiếu.
- Khuyến nghị mua bán.
- VaR nhiều kỳ hạn.

## 13. Phân công

### Thành viên A — Data Science và Modeling

- Dữ liệu.
- Lợi suất danh mục.
- Historical Simulation.
- EWMA.
- Feature engineering.
- Gradient Boosting.
- Backtesting và phân tích kết quả.

### Thành viên B — Software và Product

- Kiến trúc repository.
- Pipeline.
- Dashboard.
- Testing.
- Tài liệu cài đặt.
- Video demo.
- Release GitHub.

Cả hai thành viên phải review code và giải thích được toàn bộ hệ thống.

## 14. Rủi ro chính

- Nguồn dữ liệu không ổn định.
- Dữ liệu không được điều chỉnh đúng.
- Look-ahead bias.
- Phạm vi phát triển vượt kế hoạch.
- Mô hình ML không cải thiện so với baseline.
- Thành viên không giải thích được code do AI hỗ trợ.

## 15. Tiêu chí nghiệm thu

- Dự án chạy được từ README.
- Không sử dụng dữ liệu tương lai.
- Có kết quả backtesting ngoài mẫu.
- Có kiểm thử.
- Có Git history rõ ràng.
- Có đầy đủ nhật ký và AI Development Log.
- Hai thành viên giải thích được toàn bộ mã nguồn.