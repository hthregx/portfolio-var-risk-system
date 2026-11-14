# B-08.4 — Danh mục hình báo cáo Phase 2

- Người tổng hợp: B.
- Công việc theo timeline: 12/11/2026.
- Ngày tổng hợp: 13/11/2026.
- Snapshot chung: cutoff 06/11/2026.
- Số hình dưới đây là dự kiến; cập nhật đồng thời khi đổi thứ tự trong Word.
- Giữ nguyên tiếng Anh trên ba hình EDA; caption báo cáo bằng tiếng Việt.
- EDA là phân tích mô tả toàn mẫu, không dùng để lựa chọn mô hình.

## 1. Hình 3.1 — Giá đóng cửa chuẩn hóa

- Ảnh: figures/phase2/b/eda_2026-11-06/normalized_trend.png
- Metadata: figures/phase2/b/eda_2026-11-06/normalized_trend_metadata.json
- Script: src/phase2/plot_trend_b.py
- Dữ liệu giá: data/phase2/reference_a/market/{HPG,FPT,MWG}.csv
- Đối chiếu lợi suất: data/phase2/reference_a/portfolio_returns.csv
- Manifest: data/phase2/reference_a/latest_manifest.json
- Phạm vi: 02/01/2020–06/11/2026; 1.708 hàng giá mỗi cổ phiếu.
- Trục X: ngày; trục Y: giá chuẩn hóa, ngày đầu bằng 100.
- VNINDEX không được đưa vào hình này.

Caption:

Diễn biến giá đóng cửa chuẩn hóa của HPG, FPT và MWG, giai đoạn
02/01/2020–06/11/2026. Giá mỗi cổ phiếu được quy về 100 tại ngày đầu kỳ.
Nguồn: bộ giá KBS tham chiếu của thực nghiệm. Trạng thái điều chỉnh giá
chưa được xác minh đầy đủ; hình không biểu diễn tổng lợi suất đầu tư.

## 2. Hình 3.2 — Phân phối lợi suất danh mục

- Ảnh: figures/phase2/b/eda_2026-11-06/return_distribution.png
- Metadata: figures/phase2/b/eda_2026-11-06/return_distribution_metadata.json
- Script: src/phase2/plot_distribution_b.py
- Dữ liệu: data/phase2/reference_a/portfolio_returns.csv
- Manifest: data/phase2/reference_a/latest_manifest.json
- Phạm vi: 03/01/2020–06/11/2026; 1.707 lợi suất.
- Trục X: lợi suất đơn hằng ngày (%); trục Y: số quan sát.

Caption:

Phân phối lợi suất đơn hằng ngày của danh mục HPG, FPT và MWG với
trọng số bằng nhau. Histogram gồm 1.707 quan sát; đường nét đứt biểu thị
phân vị thực nghiệm 5%, xấp xỉ −2,5935%. Nguồn: snapshot KBS cutoff
06/11/2026. Đây là phân vị mô tả toàn mẫu, không phải ngưỡng dự báo
VaR có điều kiện theo từng phiên.

## 3. Hình 3.3 — Biến động trượt và drawdown

- Ảnh: figures/phase2/b/eda_2026-11-06/vol_drawdown.png
- Metadata: figures/phase2/b/eda_2026-11-06/vol_drawdown_metadata.json
- Script: src/phase2/plot_vol_drawdown_b.py
- Dữ liệu: data/phase2/reference_a/portfolio_returns.csv
- Manifest: data/phase2/reference_a/latest_manifest.json
- Phạm vi: 03/01/2020–06/11/2026; 1.707 lợi suất.
- Trục X: ngày.
- Trục Y ô trên: độ lệch chuẩn lợi suất ngày (%).
- Trục Y ô dưới: drawdown (%).
- Volatility dùng cửa sổ 20/60 phiên, ddof=1, chưa thường niên hóa.
- Thống kê mô tả tại cuối phiên có sử dụng lợi suất phiên hiện tại;
  không đồng nhất với đặc trưng đã lag để dự báo.

Caption:

Biến động trượt và drawdown của danh mục, giai đoạn
03/01/2020–06/11/2026. Biến động được tính bằng độ lệch chuẩn mẫu
trong cửa sổ 20 và 60 phiên, chưa thường niên hóa. Drawdown được
tính từ chuỗi giá trị tích lũy của lợi suất danh mục, có xét vốn ban đầu.
Mức drawdown sâu nhất khoảng −49,76%, ngày 15/11/2022.
Nguồn: snapshot KBS cutoff 06/11/2026; trạng thái điều chỉnh giá
chưa được xác minh đầy đủ.

## 4. Hình 3.4 — Tương quan lợi suất

- Ảnh: figures/phase2/b/correlation_2026-11-06/correlation.png
- Metadata: figures/phase2/b/correlation_2026-11-06/correlation_metadata.json
- Bảng số liệu: figures/phase2/b/correlation_2026-11-06/correlation.csv
- Script: scripts/plot_correlation_b.py
- Dữ liệu giá: data/phase2/reference_a/market/{HPG,FPT,MWG}.csv
- Đối chiếu: data/phase2/reference_a/portfolio_returns.csv
- Phạm vi: 03/01/2020–06/11/2026; 1.707 quan sát chung.
- Hai trục: tên cổ phiếu và danh mục.
- Thang màu: hệ số tương quan Pearson từ −1 đến 1.

Caption:

Tương quan Pearson giữa lợi suất HPG, FPT, MWG và danh mục tỷ trọng
bằng nhau trên 1.707 ngày quan sát chung. Nguồn: giá KBS cutoff
06/11/2026; lợi suất danh mục tính lại khớp snapshot tham chiếu A.
Danh mục chứa cả ba cổ phiếu. Hình mô tả toàn mẫu, không dùng để
lựa chọn mô hình hoặc đặc trưng.

## 5. Hình 4.1 — Lợi suất thực tế và phân vị dự báo

- Ảnh: figures/phase2/b/backtesting_2026-11-06/actual_vs_var.png
- Metadata: figures/phase2/b/backtesting_2026-11-06/actual_vs_var_metadata.json
- Script: scripts/plot_holdout_var_b.py
- Nguồn: results/phase2/b/backtesting_2026-11-06/
- File đầu vào: violation_flags.csv, backtesting.csv, metadata.json.
- Phạm vi: 18/12/2024–06/11/2026; 468 ngày mỗi mô hình.
- Mô hình: HS, EWMA, GBv2 C1 với 12 đặc trưng.
- Trục X: ngày mục tiêu; trục Y: lợi suất (%).
- Ba ô dùng chung thang đo; không làm mượt.

Caption:

Lợi suất thực tế và phân vị dự báo 5% của HS, EWMA và GBv2 C1
trên 468 phiên holdout hồi cứu. Điểm đánh dấu thể hiện phiên có
lợi suất thực tế nhỏ hơn phân vị dự báo. Nguồn: kết quả backtesting
của B trên snapshot cutoff 06/11/2026. Holdout có giao với kỳ đánh giá
Phase 1; không dùng để lựa chọn lại cấu hình.

## 6. Hình 4.2 — Vi phạm theo tháng

- Ảnh: figures/phase2/b/backtesting_2026-11-06/exceptions_monthly_panels.png
- Metadata: figures/phase2/b/backtesting_2026-11-06/exceptions_monthly_panels_metadata.json
- Script: scripts/plot_exceptions_b.py
- Bảng tháng: figures/phase2/b/backtesting_2026-11-06/exceptions_monthly.csv
- Metadata nguồn trung gian:
  figures/phase2/b/backtesting_2026-11-06/exceptions_heatmap_metadata.json
- Nguồn gốc: results/phase2/b/backtesting_2026-11-06/
- Phạm vi: 18/12/2024–06/11/2026; 468 ngày mỗi mô hình.
- Mô hình: HS, EWMA, GBv2 C1 và G04.
- Trục X: tháng; trục Y: số phiên vi phạm.
- Đường kỳ vọng: 0,05 × số phiên dự báo thực tế trong tháng.
- Log: docs/phase2/qa_b/exceptions_monthly_check.txt

Caption:

Số phiên vi phạm ngưỡng dự báo theo tháng của bốn mô hình.
Đường nét đứt biểu thị số vi phạm kỳ vọng bằng 5% số phiên có dự báo
trong từng tháng. Tháng 12/2024 có 10 phiên và tháng 11/2026 có 5 phiên
trong phạm vi đánh giá. Các ô dùng cùng thang đo. Hình mang tính mô tả;
mốc kỳ vọng không phải ranh giới đạt/không đạt kiểm định.

## 7. Hình 4.3 — So sánh mô hình

- Ảnh: figures/phase2/b/model_comparison.png
- Metadata: figures/phase2/b/model_comparison_metadata.json
- Script: scripts/plot_model_comparison_b.py
- Nguồn: results/phase2/b/reproduce_day07_20261113_133340/
- File đầu vào: holdout_predictions.csv, holdout_metrics.csv, metadata.json.
- Phạm vi: 18/12/2024–06/11/2026; 468 ngày mỗi mô hình.
- Mô hình: HS, EWMA, GBv2 C1 với 12 đặc trưng.
- Trục X: mô hình.
- Trục Y trái: mean pinball loss × 1.000.
- Trục Y phải: tỷ lệ vi phạm (%).
- Log: docs/phase2/qa_b/model_comparison_check.txt

Caption:

So sánh mean pinball loss và tỷ lệ vi phạm của HS, EWMA và GBv2 C1
trên cùng 468 phiên holdout hồi cứu. Pinball loss được nhân 1.000
để hiển thị; đường nét đứt biểu thị tỷ lệ vi phạm danh nghĩa 5%.
Nguồn: lần chạy lại B reproduce_day07_20261113_133340, đã tính lại
metrics từ predictions và đối chiếu. Kết quả không dùng để lựa chọn
lại cấu hình hoặc khẳng định ưu thế trong mọi giai đoạn.

## 8. Hình bổ sung và tài liệu giữ lại

- Hình baseline CV:
  figures/phase2/b/aligned_2026-11-06/baseline_var_timeline.png
- Metadata:
  figures/phase2/b/aligned_2026-11-06/baseline_var_timeline_metadata.json
- Script: scripts/plot_cv_b.py
- Nguồn:
  results/phase2/b/aligned_2026-11-06_20261112_121348/baseline_cv.csv
- Phạm vi: 120 phiên CV, 01/07/2024–17/12/2024.
- Chỉ dùng ở phần CV hoặc phụ lục; không thay hình holdout.

Ba hình EDA tại thư mục figures/phase2/b/ gốc dùng cutoff 04/11/2026
được giữ làm bằng chứng cũ, không phải hình chính của bản báo cáo này.

Metadata heatmap được giữ vì script hình cột tháng sử dụng để kiểm tra
nguồn CSV. Không chọn ảnh heatmap làm hình chính.

Bản PNG tương quan là hình được chọn; PDF được ghi trong metadata
lịch sử không thuộc tập hình bàn giao hiện tại.

## 9. Quy tắc quản lý bằng chứng

- Hash ảnh và script được lưu trong metadata tương ứng.
- Khi sửa script hoặc hình, phải sinh lại metadata và kiểm tra hash.
- Không đổi riêng cutoff, số quan sát hoặc giá trị thống kê trong metadata.
- Không chỉnh đường dự báo hoặc làm mượt dữ liệu để thay đổi kết quả.
- Các hình EDA toàn mẫu không phải bằng chứng dự báo ngoài mẫu.
- Các hình holdout dùng đánh giá hồi cứu, không dùng lựa chọn lại mô hình.

## 10. Trạng thái

- [x] Đã chọn bảy hình chính và xác định nguồn, phạm vi, trục, script.
- [x] Ba hình EDA mới dùng cutoff 06/11/2026.
- [x] Kiểm tra hash ảnh và script của ba hình EDA mới: PASS.
- [x] Đã xem bố cục ba hình EDA; giữ nguyên tiếng Anh.
- [x] Hình holdout, hình vi phạm tháng và hình so sánh đã có bằng chứng kiểm tra trước đó.
- [ ] Cập nhật số hình thực tế và chèn đúng ảnh trong Word.
- [ ] A/B review tập hình trong bản báo cáo tổng hợp.
