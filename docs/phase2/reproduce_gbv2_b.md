# B-07.1 — Tái lập GBv2 C1 và đối chiếu với A

- Người thực hiện: B.
- Timeline: 11/11/2026; thực hiện chạy lại: 13/11/2026.
- Nhánh: phase2/models-b.
- Commit A tham chiếu: 1a593ed24f3b388faf4e9533c362e747bb21507a.
- Lần chạy B: reproduce_day07_20261113_133340.
- Trạng thái: Tái lập phía B PASS; chờ review hai chiều và xác nhận G06.

## 1. Phạm vi

B tự chạy runner để tạo features, huấn luyện và sinh predictions,
sau đó đối chiếu với predictions A theo ngày.

Mô hình chính là GBv2 C1 với 12 đặc trưng. G04 7 đặc trưng là
baseline đối chứng; C2 được đối chiếu thêm trên CV.

A/B sử dụng chung builder bảy đặc trưng baseline
`src.gb_return_features`; B triển khai năm đặc trưng bổ sung riêng.
Không tuyên bố toàn bộ code A/B độc lập.

## 2. Dữ liệu và thiết kế

- Snapshot KBS cutoff 2026-11-06: 1707 lợi suất danh mục.
- Danh mục HPG, FPT, MWG; trọng số mỗi mã 1/3.
- Target: portfolio_simple_return; horizon: một phiên; alpha: 0.05.
- Sau warm-up 60 hàng: 1647 hàng features hợp lệ.
- Development: 1179 hàng, kết thúc 2024-12-17.
- Holdout: 468 hàng, từ 2024-12-18 đến 2026-11-06.

SHA256 snapshot:

`6389586ae8f1edbabb31760e0b555619186114b3f77d09212362fc2e1c40aadf`

| Fold | Train | Validation | Khoảng validation |
|---|---:|---:|---|
| 1 | 199 | 196 | 2021-01-14 – 2021-10-28 |
| 2 | 395 | 196 | 2021-10-29 – 2022-08-11 |
| 3 | 591 | 196 | 2022-08-12 – 2023-05-29 |
| 4 | 787 | 196 | 2023-05-30 – 2024-03-08 |
| 5 | 983 | 196 | 2024-03-11 – 2024-12-17 |

CV expanding, không shuffle, gap = 0; tổng 980 dự báo/mô hình.
GB fit một lần mỗi fold và một lần trên development trước holdout.
Features cập nhật bằng thông tin trước target; không refit GB mỗi phiên.

## 3. Cấu hình C1 — 12 đặc trưng

Bảy đặc trưng baseline:
`return_lag_1`, `return_lag_2`, `return_lag_5`,
`rolling_vol_5`, `rolling_vol_20`, `rolling_vol_60`, `drawdown`.

Năm đặc trưng bổ sung:
`momentum_5`, `momentum_20`, `downside_vol_20`,
`rolling_q05_60`, `rolling_range_20`.

Năm đặc trưng bổ sung dùng lịch sử đã shift(1).
Không sử dụng VNINDEX.

| Tham số | Giá trị |
|---|---|
| loss / alpha | quantile / 0.05 |
| n_estimators | 150 |
| learning_rate | 0.03 |
| max_depth | 2 |
| min_samples_leaf | 10 |
| subsample | 1.0 |
| random_state | 42 |

Tên A `gbv2_12f_c1_a` được ánh xạ sang tên B `gbv2_12f_c1`
khi đối chiếu.

## 4. Kết quả lần chạy mới

Runner hoàn thành với exit code 0 và `Reference check: PASS`.
SHA256 của cả bảy file đầu ra khớp metadata.

Dung sai tuyệt đối đối chiếu predictions: `1e-9`.

| Phạm vi | Mô hình B | Số dự báo | Max absolute difference | Kết quả |
|---|---|---:|---:|---|
| CV | gb_G04_7f | 980 | 9.3675067702747583e-17 | PASS |
| CV | gbv2_12f_c1 | 980 | 1.1102230246251565e-16 | PASS |
| CV | gbv2_12f_c2 | 980 | 1.5265566588595902e-16 | PASS |
| Holdout | gb_G04_7f | 468 | 9.7144514654701197e-17 | PASS |
| Holdout | gbv2_12f_c1 | 468 | 1.1796119636642288e-16 | PASS |

Nguồn: `metadata.json` của lần chạy B mới, không lấy số liệu
từ lần chạy cũ để thay thế bằng chứng.

## 5. Review code và giới hạn

Đã đọc code features, runner và tests tương ứng của A.
Code thể hiện kiểm tra schema, warm-up, thông tin trước target,
train/validation không chồng lấn và tính xác định trên fixture.
Đọc tests không đồng nghĩa đã chạy tests bản A trong môi trường riêng.

Các lưu ý:

- `best_overall_on_cv` của A chỉ xét G04, C1 và C2, không gồm HS/EWMA.
- A có nhãn `development_only_not_live_forecast` nhưng tạo cả
  CV và holdout; báo cáo dùng “CV development và holdout hồi cứu”.
- Commit tiếp nhận A là `1a593ed`; `git_sha` trong metadata A
  là `01aa58d`. Giữ riêng hai thông tin này.
- Holdout giao với kỳ đánh giá Phase 1; không phải test độc lập
  hoàn toàn và không dùng để lựa chọn lại cấu hình.
- G04 và C1 khác cả features lẫn hyperparameters.
- Tái lập PASS không chứng minh hiệu quả dự báo tương lai hoặc
  thay thế toàn bộ review leakage.

## 6. Đường dẫn bằng chứng

Kết quả lần chạy B mới:

`results/phase2/b/reproduce_day07_20261113_133340/`

Log:

`docs/phase2/qa_b/reproduce_day07_20261113_133340.txt`

Predictions và metadata A dùng trong lần chạy này:

`reference_a/day07_1a593ed_20261113_125147_499370/features_a/results/phase2/a/training_2026-11-06/`

Manifest tiếp nhận code A:

`reference_a/day07_1a593ed_20261113_125147_499370/RECEIVED_FILES.json`

## 7. Kết luận

- [x] B chạy lại GBv2 C1 đúng 12 đặc trưng.
- [x] Runner hoàn thành, file đầu ra khớp hash.
- [x] Đủ năm đối chiếu predictions A/B đạt dung sai.
- [ ] A review lần chạy B hiện tại.
- [ ] A/B hoàn tất đối chiếu hai chiều và xác nhận G06.

Tại thời điểm kiểm tra, chưa thấy `gbv2_crosscheck.md`,
`review_training_b.md` và `audit_b_metrics.csv` ở các đường dẫn
dự kiến trên hai nhánh đã kiểm tra. Audit baseline cũ không
tự động thay thế review lần chạy C1 hiện tại.

B-07.1 hoàn thành phần tái lập phía B; chưa tuyên bố G06 PASS.