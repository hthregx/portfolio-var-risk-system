# B-04.5 — Review thiết kế thực nghiệm A/B

- Người thực hiện: B
- Ngày kiểm tra: 12/11/2026
- Công việc theo timeline: ngày 08/11/2026
- Nhánh B: phase2/models-b
- Commit A tham chiếu: 1a593ed24f3b388faf4e9533c362e747bb21507a
- Trạng thái: B đã kiểm tra; chờ A xác nhận thiết kế chung.

## 1. Phạm vi

Đối chiếu snapshot, target, features, lịch CV, holdout,
quy tắc refit và kết quả dự báo GB giữa A và B.

B tự chạy runner và sinh predictions trước khi đối chiếu
với predictions tham chiếu của A.

Review này không thay đổi cấu hình hoặc kết quả frozen Phase 1.

## 2. Snapshot

- Nguồn theo manifest: KBS.
- Cutoff: 2026-11-06.
- Số dòng portfolio returns: 1707.
- Giai đoạn: 2020-01-03 đến 2026-11-06.
- CSV local: data/phase2/reference_a/portfolio_returns.csv.
- Manifest local: data/phase2/reference_a/latest_manifest.json.
- SHA256 CSV khớp manifest và snapshot thực nghiệm A:

6389586ae8f1edbabb31760e0b555619186114b3f77d09212362fc2e1c40aadf

Hash xác nhận nội dung file, không chứng minh độc lập
tính chính xác của giá nguồn hoặc việc điều chỉnh giá.

## 3. Target và features

- Danh mục: HPG, FPT, MWG; trọng số mỗi mã 1/3.
- Target: portfolio_simple_return.
- Horizon: một phiên giao dịch tiếp theo.
- Alpha: 0.05.
- Điều kiện thời gian: as_of < target_date.
- Features chỉ sử dụng thông tin trước target_date.
- VNINDEX không được sử dụng trong bộ 12 features này.

Bảy features baseline:

1. return_lag_1
2. return_lag_2
3. return_lag_5
4. rolling_vol_5
5. rolling_vol_20
6. rolling_vol_60
7. drawdown

Năm features mở rộng:

8. momentum_5
9. momentum_20
10. downside_vol_20
11. rolling_q05_60
12. rolling_range_20

GB G04 giữ 7 features. GBv2 C1 và C2 dùng 12 features.

## 4. Chia dữ liệu và refit

Sau 60 hàng warm-up, còn 1647 hàng features hợp lệ:

- Development: 1179 hàng, kết thúc 2024-12-17.
- Holdout: 468 hàng, từ 2024-12-18 đến 2026-11-06.
- CV: expanding, 5 folds, không shuffle, gap = 0.
- Mỗi fold có 196 ngày validation; tổng 980 ngày.

| Fold | Số mẫu train | Số mẫu validation |
|---|---:|---:|
| 1 | 199 | 196 |
| 2 | 395 | 196 |
| 3 | 591 | 196 |
| 4 | 787 | 196 |
| 5 | 983 | 196 |

GB fit một lần đầu mỗi fold.
Trên holdout, GB fit bằng toàn bộ development rồi giữ cố định.

Features vẫn cập nhật theo thông tin trước từng target.
HS và EWMA sử dụng lịch sử returns trước từng target,
bao gồm lịch sử trước thời điểm bắt đầu validation.

Không dùng holdout để lựa chọn lại cấu hình.

## 5. Cấu hình GB

Các cấu hình đều dùng:

- loss: quantile
- alpha: 0.05
- max_depth: 2
- subsample: 1.0
- random_state: 42

| Cấu hình | Features | n_estimators | learning_rate | min_samples_leaf |
|---|---:|---:|---:|---:|
| G04 | 7 | 100 | 0.03 | 5 |
| GBv2 C1 | 12 | 150 | 0.03 | 10 |
| GBv2 C2 | 12 | 120 | 0.05 | 5 |

## 6. Quy tắc metrics

- Tiêu chí lựa chọn: mean pinball loss trên CV, alpha = 0.05.
- Violation: actual_return < quantile_return.
- VaR báo cáo: max(0, -quantile_return).
- So sánh trên cùng target dates và actual returns.
- Báo cáo số target, số violations, violation rate,
  mean pinball loss và average VaR.
- Không dùng VaR dương thay quantile_return để tính pinball.
- Không chọn mô hình chỉ dựa vào số violations hoặc average VaR.

## 7. Bằng chứng kiểm tra của B

### 7.1. Baseline và tests

- HS, EWMA, G04: mỗi mô hình có 980 dự báo CV.
- Mỗi mô hình có 5 folds, mỗi fold 196 dự báo.
- Ngày, folds và actual returns giữa ba baseline khớp.
- Tests metrics và aligned runner: 26 passed.
- Lệnh runner hoàn thành với exit code 0.
- Reference check: PASS.

Lệnh tests đã chạy:

python -m pytest tests/test_phase2_metrics_b.py tests/test_phase2_aligned_runner_b.py -q

### 7.2. Đối chiếu dự báo GB với A

Dung sai tuyệt đối: 1e-9.

| Phạm vi | Model B | Số dự báo | Max absolute difference | Trạng thái |
|---|---|---:|---:|---|
| CV | gb_G04_7f | 980 | 9.367506770274758e-17 | PASS |
| CV | gbv2_12f_c1 | 980 | 1.1102230246251565e-16 | PASS |
| CV | gbv2_12f_c2 | 980 | 1.5265566588595902e-16 | PASS |
| Holdout | gb_G04_7f | 468 | 9.71445146547012e-17 | PASS |
| Holdout | gbv2_12f_c1 | 468 | 1.1796119636642288e-16 | PASS |

Các phép kiểm trên xác nhận tái lập dự báo GB trong dung sai.
Chưa xác nhận HS/EWMA khớp một bộ predictions riêng của A
trên cùng kỳ đánh giá.

### 7.3. Biểu đồ baseline

- Đoạn vẽ: 2024-07-01 đến 2024-12-17.
- Số target: 120 cho mỗi baseline.
- HS: 5 violations, tương ứng 4.17%.
- EWMA: 8 violations, tương ứng 6.67%.
- G04: 2 violations, tương ứng 1.67%.
- CSV nguồn biểu đồ khớp baseline của lần chạy aligned.
- Các tỷ lệ trên chỉ đại diện đoạn vẽ, không phải toàn bộ CV.

## 8. Nhận xét về lựa chọn mô hình

| Model | Số target CV | Violations | Violation rate (%) | Mean pinball loss |
|---|---:|---:|---:|---:|
| ewma_d094 | 980 | 51 | 5.204081633 | 0.001963972 |
| gbv2_12f_c1 | 980 | 70 | 7.142857143 | 0.002080267 |
| gb_G04_7f | 980 | 62 | 6.326530612 | 0.002117906 |
| historical_w250 | 980 | 48 | 4.897959184 | 0.002119985 |
| gbv2_12f_c2 | 980 | 77 | 7.857142857 | 0.002178550 |

Giá trị trong bảng được làm tròn khi trình bày.

C1 có mean pinball loss thấp nhất trong nhóm GB.
EWMA có mean pinball loss thấp nhất khi xét cả năm cấu hình.

Vì vậy, "C1 là ứng viên GBv2 được chọn" không đồng nghĩa
"C1 tốt nhất trong toàn bộ các phương pháp".

A/B cần xác nhận phạm vi ứng viên và tiêu chí quyết định
trước khi chốt mô hình chính thức.

## 9. Giới hạn và nội dung còn mở

- Holdout trùng một phần giai đoạn đánh giá đã công bố ở Phase 1.
  Kết quả là đánh giá hồi cứu, không phải pristine independent test.
- Chuyển G04 sang C1 thay đổi cả features và hyperparameters.
  Không quy toàn bộ cải thiện cho việc tăng từ 7 lên 12 features.
- Tái lập dự báo không chứng minh hiệu quả dự báo tương lai.
- Cần bổ sung đối chiếu metrics được B tính lại trực tiếp
  từ predictions A với bảng metrics công bố của A.
- Cần A xác nhận thiết kế chung và phạm vi lựa chọn mô hình.

## 10. Đường dẫn bằng chứng

Lần chạy B:

results/phase2/b/aligned_2026-11-06_20261112_121348/

Các file chính:

- metadata.json
- features.csv
- baseline_cv.csv
- cv_predictions.csv
- cv_metrics.csv
- cv_metrics_summary_b.csv
- holdout_predictions.csv
- holdout_metrics.csv
- gbv2_retrain.csv

Log:

docs/phase2/qa_b/aligned_run_20261112_121348.txt

Tham chiếu A đã nhận từ commit nêu trên:

results/phase2/b/reference_a_1a593ed/results/phase2/a/training_2026-11-06/

Hình:

figures/phase2/b/aligned_2026-11-06/baseline_var_timeline.png
figures/phase2/b/aligned_2026-11-06/baseline_var_timeline.pdf

## 11. Kết luận review

- [x] B xác minh đúng snapshot A.
- [x] B đối chiếu thiết kế CV và holdout.
- [x] B chạy baseline trên cùng ngày đánh giá.
- [x] B tái lập dự báo G04 và GBv2 trong dung sai.
- [x] B kiểm tra dữ liệu và số violations trên hình baseline.
- [ ] Hoàn thiện đối chiếu metrics trực tiếp từ predictions A.
- [ ] A xác nhận thiết kế chung.
- [ ] A/B xác nhận phạm vi lựa chọn mô hình chính thức.

B hoàn thành bản review kỹ thuật để A xem xét.
Chưa ghi nhận phê duyệt chung A/B hoặc hoàn thành toàn bộ gate.