# B-01.3 — Kiểm tra nguồn dữ liệu của B

## 1. Nguồn và cách lấy dữ liệu

- Người thực hiện: B.
- Ngày kiểm tra: 05/11/2026.
- Nguồn theo manifest: KBS.
- Thư mục local: data/phase2/.
- Artifact ban đầu: phase2-eod-40634002113.
- GitHub Actions run ID: 40634002113.
- Cutoff artifact ban đầu: 27/10/2026.

B tự tải artifact bằng GitHub CLI, sau đó cập nhật bằng:

python -m scripts.run_phase2_daily_eod

Bộ dữ liệu sau cập nhật:
- Cutoff: 04/11/2026.
- Thời điểm tạo: 05/11/2026 lúc 16:30:10, giờ Việt Nam.
- Raw snapshot:
  data/phase2/raw_snapshots/20261105T163010636470+0700

## 2. Kết quả kiểm tra CSV sau cập nhật

| Chuỗi | Số dòng | Ngày đầu | Ngày cuối |
| --- | ---: | --- | --- |
| HPG | 1706 | 2020-01-02 | 2026-11-04 |
| FPT | 1706 | 2020-01-02 | 2026-11-04 |
| MWG | 1706 | 2020-01-02 | 2026-11-04 |
| VNINDEX | 1703 | 2020-01-02 | 2026-11-04 |
| Portfolio returns | 1705 | 2020-01-03 | 2026-11-04 |

Cả năm CSV:
- Số ô thiếu: 0.
- Số ngày không hợp lệ: 0.
- Số ngày trùng: 0.
- Ngày được sắp xếp tăng dần.

Portfolio simple return:
- Số giá trị không hữu hạn: 0.
- Số giá trị <= -1: 0.

Không có ô thiếu trong các hàng hiện hữu không đồng nghĩa
đã kiểm chứng đầy đủ mọi phiên giao dịch theo lịch thị trường.

## 3. Kiểm tra hash

SHA256 được tính trực tiếp từ portfolio_returns.csv:

f85bc33ce54d449e393cd81878e9de8627de88d0c309e7f821568b7177692f53

Kết quả đối chiếu với latest_manifest.json: KHỚP.

Hash này xác minh file lợi suất khớp manifest local;
chưa chứng minh toàn bộ dữ liệu giống bộ A đang sử dụng.

## 4. Đơn vị

Đối chiếu theo code pipeline:
- src/phase2/kbs_http.py gắn price_unit của cổ phiếu là vnd.
- src/phase2/daily_data.py chia OHLC cổ phiếu cho 1000.
- Giá OHLC cổ phiếu đầu ra: nghìn VND.
- VNINDEX: điểm chỉ số.
- Portfolio simple return: số thập phân;
  ví dụ 0.01 tương ứng 1%.

Raw snapshot hiện có CSV, không có metadata JSON riêng.
Đơn vị được xác minh theo quy tắc code;
chưa đối chiếu độc lập với nhà cung cấp.

## 5. Vấn đề chất lượng và giới hạn

VNINDEX:
- quality_status: REVIEW.
- index_features_eligible: false.
- Ba ngày OHLC đã bị loại:
  2021-02-17, 2021-12-10, 2023-05-15.

Chưa dùng VNINDEX làm features.
Baseline 7 features sử dụng lịch sử lợi suất danh mục.

Việc điều chỉnh giá do corporate actions chưa được
xác minh đầy đủ; không khẳng định toàn bộ giá đã adjusted.

Bộ hiện tại được cập nhật trên máy B.
A/B còn phải thống nhất snapshot, cutoff và hash
trước khi đối chiếu kết quả mô hình.

## 6. Kết luận B-01.3

B đã tự truy cập dữ liệu thật, kiểm tra số dòng,
phiên cuối, cấu trúc ngày, giá trị thiếu, tính hợp lệ
của lợi suất và hash portfolio.

Đơn vị đã được đối chiếu theo code pipeline.
Các giới hạn nguồn và trạng thái REVIEW được ghi rõ.

Hoàn thành phần kiểm tra nguồn thuộc phạm vi B-01.3.
Kết luận này không thay thế kiểm tra chất lượng dữ liệu
chuyên sâu hoặc nghiệm thu contract chung A/B.