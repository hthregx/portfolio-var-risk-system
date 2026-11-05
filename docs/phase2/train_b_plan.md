# B-01.2 — Kế hoạch runner độc lập của B

- Ngày lập: 05/11/2026.
- Người phụ trách: B.
- Nhánh: phase2/models-b.
- Trạng thái: bản thiết kế, chờ thống nhất contract với A.
- Phạm vi: Phase 2; không sửa cấu hình hoặc kết quả frozen Phase 1.

## 1. Mục tiêu và bài toán

B tự chạy HS, EWMA, GB G04 và tái lập GBv2 bằng runner riêng,
không sao chép predictions của A hoặc phụ thuộc file A chưa tạo.

- Danh mục: HPG/FPT/MWG, mỗi mã trọng số 1/3.
- Target: portfolio_simple_return.
- Horizon: một phiên giao dịch tiếp theo.
- VaR 95%, tương ứng quantile alpha = 0.05.
- Lợi suất dùng dạng thập phân: 0.01 = 1%.

## 2. Dữ liệu đầu vào

B tải artifact `phase2-eod-40634002113`, sau đó cập nhật KBS bằng:
`python -m scripts.run_phase2_daily_eod`.

Snapshot local đã kiểm tra:
- Dữ liệu: `data/phase2/portfolio_returns.csv`.
- Manifest: `data/phase2/latest_manifest.json`.
- Giai đoạn: 2020-01-03 đến 2026-11-04.
- Số dòng: 1705.
- SHA256:
  `f85bc33ce54d449e393cd81878e9de8627de88d0c309e7f821568b7177692f53`.

Dữ liệu không thiếu, ngày hợp lệ, tăng dần và không trùng.
Lợi suất hữu hạn, lớn hơn -1; hash khớp manifest.
Bằng chứng: `docs/phase2/b_source_check.md`.

Snapshot chung A/B: CHỜ CHỐT.
Phải cố định dữ liệu và hash trước khi chạy thực nghiệm.

Fixture xác định được dùng để kiểm thử schema và phép tính,
không dùng làm kết quả nghiên cứu.

## 3. Features và quy tắc thời gian

B dùng bảy features:
`return_lag_1`, `return_lag_2`, `return_lag_5`,
`rolling_vol_5`, `rolling_vol_20`, `rolling_vol_60`, `drawdown`.

Có thể dùng builder sẵn có theo
`docs/gb-return-feature-contract.md`; không chờ `features_a.py`.
Baseline này không sử dụng VNINDEX.

Với target_date T:
- X(T) chỉ dùng thông tin đã biết trước T.
- y(T) là lợi suất thực tế tại T.
- as_of < target_date.
- Mọi training_target_date < target_date.
- T+1 là phiên giao dịch tiếp theo, không phải ngày lịch.
- Không shift hai lần nếu builder đã lag.
- Loại warm-up không hợp lệ; không điền bằng dữ liệu tương lai.

## 4. Mô hình

| Mô hình | Cấu hình |
| --- | --- |
| HS | historical_w250; quantile 0.05 từ 250 lợi suất trước target |
| EWMA | ewma_d094; lambda 0.94; mean 0; Normal; khởi tạo bằng bình phương lợi suất đầu tiên |
| GB G04 | Quantile loss; alpha 0.05; 7 features; 100 trees; learning_rate 0.03; max_depth 2; min_samples_leaf 5; subsample 1.0; seed 42 |
| GBv2 | B tự chạy cấu hình và feature set đã thống nhất với A |

Giữ G04 cố định làm baseline.
Ngày 5 tái lập GBv2 7 features; ngày 7 tái lập đúng ứng viên
cuối, gồm cả 12 features nếu đó là cấu hình được chọn.

## 5. Validation và đánh giá

- Dùng 5 expanding time-series folds, không shuffle.
- Chọn cấu hình bằng validation mean pinball loss.
- Đánh giá cuối tách biệt, không dùng để tune.
- A/B dùng cùng folds, target dates và actual returns.
- Nếu cập nhật từ các phiên validation đã qua, phải mô tả rõ
  cách walk-forward và thống nhất quy tắc refit.

CHỜ CHỐT với A:
snapshot/hash, cutoff, lịch folds, minimum training size, gap,
tần suất refit, giai đoạn đánh giá cuối và dung sai đối chiếu.

## 6. Predictions và metrics

Schema đề xuất:

model,fold,as_of,target_date,quantile_return,var_return,actual_return

- Mỗi model có một dự báo/target_date trong từng bộ đánh giá.
- Tách OOF và final evaluation.
- Chuẩn hóa tên q_05 thành quantile_return nếu cần.
- actual_return chỉ phục vụ đánh giá, không làm đầu vào tại target.

Metrics:
- Mean pinball loss, alpha = 0.05.
- Violation rate: tỷ lệ actual_return < quantile_return.
- Average VaR, với var_return = max(0, -quantile_return).
- Số target và số violations.
- Kupiec triển khai ngày 6, nêu giới hạn lượng mẫu.

Không dùng VaR magnitude thay quantile trong pinball/violation.
Không âm thầm bỏ ngày thiếu forecast hoặc kết luận mô hình
tốt nhất chỉ dựa vào Average VaR.

## 7. Code, outputs và metadata

Code B:
- `src/phase2/training_b.py`
- `src/phase2/cv_b.py`
- `src/phase2/metrics_b.py`
- `src/phase2/backtesting_b.py`

Tests: `tests/test_phase2_*_b.py`.

Outputs trong `results/phase2/b/`:
`hs.csv`, `ewma.csv`, `gb_baseline.csv`, `baseline_cv.csv`,
`gbv2_retrain.csv`, `metric_table_b.csv`, `backtesting.csv`.

Log: `docs/phase2/qa_b/`.
Hình: `figures/phase2/b/`.

Mỗi lần chạy lưu commit SHA, phiên bản môi trường, data hash,
cutoff, cấu hình, feature list, seed, folds, quy tắc refit,
số mẫu và đường dẫn outputs.

Không ghi đè kết quả A hoặc `results/final_*` của Phase 1.

## 8. Tiến độ và nghiệm thu

- Ngày 1: thiết kế runner, kiểm tra nguồn.
- Ngày 2: HS/EWMA.
- Ngày 3: GB baseline và CV.
- Ngày 4: baseline 5-fold và metrics.
- Ngày 5: tái lập GBv2 7 features.
- Ngày 6: backtesting và biểu đồ.
- Ngày 7: tái lập ứng viên cuối, kiểm tra chéo A/B.

Checklist bản thiết kế:
- [ ] Đủ nguồn dữ liệu, fixture và bảy features.
- [ ] Rõ target T+1 và quy tắc chống leakage.
- [ ] Rõ mô hình, CV, metrics và outputs riêng của B.
- [ ] Không phụ thuộc file A chưa tạo.
- [ ] Liệt kê các quyết định chờ A/B thống nhất.

Hoàn thành B-01.2 là hoàn thành bản thiết kế.
Gate G01 chỉ PASS sau khi A/B thống nhất contract chung.