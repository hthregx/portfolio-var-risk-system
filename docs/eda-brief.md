# EDA Brief

## 1. Dataset

Phân tích sử dụng dữ liệu đã qua validation của ba mã:

- HPG
- FPT
- MWG

Mỗi ticker có 1.638 quan sát, trong khoảng thời gian từ
2020-01-02 đến 2026-07-28.

Các file đầu vào:

- `data/processed/HPG_clean.csv`
- `data/processed/FPT_clean.csv`
- `data/processed/MWG_clean.csv`

Dữ liệu được căn chỉnh theo ngày giao dịch trước khi thực hiện các phân tích
tương quan và so sánh rủi ro.

## 2. Price Behavior

Giá đóng cửa của ba ticker có xu hướng và mức tăng trưởng khác nhau.
Giá chuẩn hóa về 100 giúp so sánh hiệu suất tương đối mà không bị ảnh hưởng
bởi mức giá tuyệt đối ban đầu.

## 3. Individual Return Behavior

Daily log return của HPG, FPT và MWG chủ yếu dao động quanh 0 nhưng xuất hiện
một số phiên biến động mạnh.

Các quan sát cực đoan được giữ lại vì extreme return không đồng nghĩa với lỗi
dữ liệu. Chúng có thể phản ánh cú sốc thị trường, thông tin doanh nghiệp hoặc
corporate action.

Kết quả IQR:

- HPG: 128 quan sát được gắn cờ.
- FPT: 116 quan sát được gắn cờ.
- MWG: 120 quan sát được gắn cờ.

## 4. Correlation

Cặp có tương quan cao nhất:

- FPT–MWG: 0.5423

Cặp có tương quan thấp nhất:

- HPG–FPT: 0.4926

Cả ba cặp đều có tương quan dương ở mức trung bình. HPG–FPT có tương quan
thấp hơn nên có thể cung cấp lợi ích đa dạng hóa sơ bộ tốt hơn.

Phân tích này chưa thực hiện portfolio optimization.

## 5. Volatility

Rolling volatility được tính bằng độ lệch chuẩn của daily log return trong
20 phiên giao dịch và chưa annualize.

Kết quả:

- HPG: rolling volatility trung bình 1.9681%.
- FPT: rolling volatility trung bình 1.6808%.
- MWG: rolling volatility trung bình 2.0818%.

MWG có rolling volatility trung bình và cực đại cao nhất. Biểu đồ cho thấy
volatility clustering, nghĩa là các giai đoạn biến động cao thường xuất hiện
thành từng cụm.

## 6. Tail Behavior

Kết quả phân vị 5%:

- FPT: -2.6561%
- HPG: -3.3679%
- MWG: -3.6333%

MWG có `return_q05` âm nhất, cho thấy ngưỡng tổn thất tại đuôi trái 5% sâu
nhất trong ba ticker.

Skewness của cả ba ticker đều âm và excess kurtosis đều dương, cho thấy phân
phối lợi suất có đặc điểm bất đối xứng và đuôi dày hơn phân phối chuẩn.

## 7. Important Implications for VaR

`return_q05` là thống kê khám phá quan trọng cho VaR 95%, vì nó đại diện cho
ngưỡng của 5% lợi suất thấp nhất.

Tuy nhiên, q05 của từng cổ phiếu chỉ là thống kê EDA. Đây chưa phải portfolio
VaR cuối cùng vì chưa kết hợp trọng số danh mục và cấu trúc phụ thuộc giữa
các tài sản.