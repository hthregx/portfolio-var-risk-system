# B-07.2 — Review metrics A bằng phép tính của B

- Người thực hiện: B.
- Timeline: 11/11/2026; thực hiện: 13/11/2026.
- Commit A: 1a593ed24f3b388faf4e9533c362e747bb21507a.
- Lần chạy B: reproduce_day07_20261113_133340.
- Trạng thái: PASS phía B; chưa thay thế xác nhận G06 hai chiều.

## 1. Phạm vi và phương pháp

B sử dụng `scripts/check_metrics_ab_b.py` để tính lại metrics
trực tiếp từ predictions A và B, rồi đối chiếu với các bảng công bố.

Trước khi so metrics, script kiểm tra tập mô hình, folds, số dự báo,
ngày mục tiêu, as_of, actual returns, dữ liệu hữu hạn và quy tắc VaR.

Các tên mô hình A có hậu tố `_a` được ánh xạ sang tên B.
CV gồm 5 folds, mỗi fold 196 phiên; holdout gồm 468 phiên/mô hình.

Sáu chỉ tiêu đối chiếu:

- Số dự báo và số vi phạm: yêu cầu bằng nhau chính xác.
- Violation rate, coverage, mean pinball loss và average VaR:
  dung sai tuyệt đối `1e-12`, alpha `0.05`.

Quy tắc tính:

- Vi phạm: `actual_return < quantile_return`.
- Coverage: `mean(actual_return >= quantile_return)`.
- Pinball: `mean(max(0.05 * e, -0.95 * e))`,
  với `e = actual_return - quantile_return`.
- Average VaR: `mean(max(0, -quantile_return))`.

Các bảng công bố được dùng không có cột coverage riêng.
Coverage tham chiếu được suy ra bằng `1 - violation_rate`.
Đây là kiểm tra tính nhất quán bổ sung, không phải kiểm định
thống kê độc lập.

## 2. Kết quả

| Phạm vi | Đối chiếu | Số dòng | Kết quả |
|---|---|---:|---|
| CV | A tính lại ↔ A công bố | 15 | PASS |
| CV | A tính lại ↔ B tính lại | 15 | PASS |
| CV | B tính lại ↔ B công bố | 25 | PASS |
| Holdout | A tính lại ↔ A công bố | 2 | PASS |
| Holdout | A tính lại ↔ B tính lại | 2 | PASS |
| Holdout | B tính lại ↔ B công bố | 4 | PASS |

Tổng cộng 63 dòng, mỗi dòng 6 chỉ tiêu:
378 phép so sánh chỉ tiêu, 0 lỗi vượt dung sai.

Sai lệch tuyệt đối từng chỉ tiêu được lưu trong `metric_table_b.csv`.
PASS không có nghĩa mọi sai lệch đều bằng 0.

Đối chiếu A/B chỉ áp dụng cho các mô hình chung:

- CV: G04, GBv2 C1 và GBv2 C2.
- Holdout: G04 và GBv2 C1.

HS/EWMA được kiểm tra giữa predictions B và metrics B công bố;
không tuyên bố đã đối chiếu với predictions HS/EWMA riêng của A.

## 3. Bằng chứng

Script:

`scripts/check_metrics_ab_b.py`

Nguồn A:

`reference_a/day07_1a593ed_20261113_125147_499370/features_a/results/phase2/a/training_2026-11-06/`

Nguồn B:

`results/phase2/b/reproduce_day07_20261113_133340/`

Kết quả kiểm tra:

`results/phase2/b/metrics_check_day07_20261113_133340/`

Các file:

- `metric_table_b.csv`: sai lệch và trạng thái từng đối chiếu.
- `recomputed_metrics.csv`: metrics B tính lại từ predictions A/B.
- `metadata.json`: nguồn, hash đầu vào, hash script, dung sai và trạng thái.

## 4. Kết luận và giới hạn

B xác nhận các metrics được đối chiếu nhất quán trong dung sai,
bao gồm pinball loss và coverage.

Kết quả này không thay thế kiểm tra toàn bộ leakage, không chứng minh
hiệu quả dự báo tương lai và không dùng để chọn lại cấu hình bằng holdout.

- [x] Tính lại metrics A từ predictions.
- [x] Tính lại metrics B từ lần chạy mới.
- [x] Đối chiếu metrics A/B trên ngày và folds tương ứng.
- [x] Ghi nhận sai lệch, dung sai và 0 lỗi.
- [ ] A hoàn tất review phía B và A/B xác nhận G06.