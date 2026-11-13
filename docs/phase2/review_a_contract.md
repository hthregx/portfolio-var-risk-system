# B-04.5 — Review thiết kế thực nghiệm A/B

- Người thực hiện: B.
- Ngày kiểm tra: 12/11/2026.
- Công việc theo timeline: B-04.5 ngày 08/11/2026.
- Bổ sung bằng chứng: kiểm tra metrics độc lập B-05.2 ngày 09/11/2026.
- Nhánh B: phase2/models-b.
- Commit A tham chiếu: 1a593ed24f3b388faf4e9533c362e747bb21507a.
- Trạng thái: B đã kiểm tra; chờ A xác nhận thiết kế chung.

## 1. Phạm vi

Đối chiếu snapshot, target, features, lịch CV, holdout,
quy tắc refit, dự báo GB và metrics giữa A và B.

B tự chạy runner và sinh predictions trước khi đối chiếu
với predictions tham chiếu của A.

GBv2 của B sử dụng 12 features để tái lập cấu hình A.
G04 giữ nguyên 7 features làm baseline.

Review này không thay đổi cấu hình hoặc kết quả frozen Phase 1.

## 2. Snapshot

- Nguồn theo manifest: KBS.
- Cutoff: 2026-11-06.
- Số dòng portfolio returns: 1707.
- Giai đoạn: 2020-01-03 đến 2026-11-06.
- CSV local: `data/phase2/reference_a/portfolio_returns.csv`.
- Manifest local: `data/phase2/reference_a/latest_manifest.json`.
- SHA256 CSV khớp manifest và snapshot thực nghiệm A:

`6389586ae8f1edbabb31760e0b555619186114b3f77d09212362fc2e1c40aadf`

Hash xác nhận nội dung file, không chứng minh độc lập
tính chính xác của giá nguồn hoặc việc điều chỉnh giá.

## 3. Target và features

- Danh mục: HPG, FPT, MWG; trọng số mỗi mã 1/3.
- Target: `portfolio_simple_return`.
- Horizon: một phiên giao dịch tiếp theo.
- Alpha: 0.05.
- Điều kiện thời gian: `as_of < target_date`.
- Training targets phải trước target đang dự báo.
- Features chỉ sử dụng thông tin trước target_date.
- VNINDEX không được sử dụng trong bộ 12 features này.

Bảy features baseline:

1. `return_lag_1`
2. `return_lag_2`
3. `return_lag_5`
4. `rolling_vol_5`
5. `rolling_vol_20`
6. `rolling_vol_60`
7. `drawdown`

Năm features mở rộng:

8. `momentum_5`
9. `momentum_20`
10. `downside_vol_20`
11. `rolling_q05_60`
12. `rolling_range_20`

G04 sử dụng 7 features.
GBv2 C1 và C2 sử dụng đủ 12 features theo thứ tự trên.

HS và EWMA sử dụng lịch sử lợi suất danh mục;
không sử dụng ma trận 12 features để thay thế lịch sử returns.

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
Lợi suất của các phiên validation hoặc holdout đã qua có thể
tham gia xây dựng features cho phiên tiếp theo.

HS và EWMA sử dụng lịch sử returns trước từng target,
bao gồm lịch sử trước thời điểm bắt đầu validation.

Đây là đánh giá dự báo một bước theo thời gian,
không phải dự báo toàn bộ fold cùng lúc tại đầu fold.

Không dùng holdout để lựa chọn lại cấu hình.

## 5. Cấu hình mô hình

### 5.1. HS và EWMA

- HS: cửa sổ 250 lợi suất trước target, phân vị 0.05,
  phương pháp nội suy linear.
- EWMA: lambda = 0.94, conditional mean = 0,
  giả định phân phối Normal.
- EWMA khởi tạo phương sai bằng bình phương lợi suất đầu tiên
  và cập nhật bằng thông tin trước target.

### 5.2. Gradient Boosting

Các cấu hình đều dùng:

- `loss`: quantile.
- `alpha`: 0.05.
- `max_depth`: 2.
- `subsample`: 1.0.
- `random_state`: 42.

| Cấu hình | Features | n_estimators | learning_rate | min_samples_leaf |
|---|---:|---:|---:|---:|
| G04 | 7 | 100 | 0.03 | 5 |
| GBv2 C1 | 12 | 150 | 0.03 | 10 |
| GBv2 C2 | 12 | 120 | 0.05 | 5 |

Cả ba cấu hình GB được đối chiếu trên CV.
G04 và GBv2 C1 được đối chiếu trên holdout.

## 6. Quy tắc metrics

- Tiêu chí lựa chọn: mean pinball loss trên CV, alpha = 0.05.
- Violation: `actual_return < quantile_return`.
- Coverage: `1 - violation_rate`.
- VaR báo cáo: `max(0, -quantile_return)`.
- So sánh trên cùng target dates và actual returns.
- Báo cáo số target, số violations, violation rate,
  coverage, mean pinball loss và average VaR.
- Không dùng VaR dương thay quantile_return để tính pinball.
- Không chọn mô hình chỉ dựa vào số violations hoặc average VaR.

Với residual = actual_return - quantile_return:

- Nếu residual >= 0: pinball = 0.05 * residual.
- Nếu residual < 0: pinball = -0.95 * residual.

Lợi suất thực tế bằng quantile_return không được tính là violation.

## 7. Bằng chứng kiểm tra của B

### 7.1. Baseline và tests

- HS, EWMA, G04: mỗi mô hình có 980 dự báo CV.
- Mỗi mô hình có 5 folds, mỗi fold 196 dự báo.
- Ngày, folds và actual returns giữa ba baseline khớp.
- Tests metrics và aligned runner: 26 passed.
- Lệnh runner hoàn thành với exit code 0.
- Reference check: PASS.

Lệnh tests đã chạy:

`python -m pytest tests/test_phase2_metrics_b.py tests/test_phase2_aligned_runner_b.py -q`

Sau khi bổ sung dependency scikit-learn vào workflow,
lượt kiểm tra local rộng hơn ghi nhận 75 passed với lệnh:

`python -m pytest tests/test_phase2*.py -q`

Kết quả local không thay thế trạng thái CI trên GitHub.
Trạng thái CI cần được kiểm tra trên commit tương ứng.

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
- Hình mới được xuất cùng metadata ghi nguồn và hash.
- Không làm mượt hoặc thay đổi giá trị dự báo.
- Các tỷ lệ trên chỉ đại diện đoạn vẽ, không phải toàn bộ CV.

### 7.4. Đối chiếu metrics độc lập — bổ sung B-05.2

B tính lại metrics trực tiếp từ predictions A và B,
không gọi hàm metrics của A hoặc aligned runner B.

Các chỉ tiêu đối chiếu gồm:

- Số target.
- Số violations.
- Violation rate.
- Mean pinball loss.
- Average VaR.

Coverage được tính bằng 1 - violation_rate.

Trước khi so metrics, script kiểm tra ngày, folds, actual returns,
giá trị hữu hạn, quy tắc VaR và cột violation trong predictions.

| Phạm vi | Phép đối chiếu | Số dòng PASS |
|---|---|---:|
| CV | Metrics tính lại từ predictions A so với bảng A | 15 |
| CV | Metrics tính lại từ predictions B so với bảng B | 25 |
| CV | Metrics tính lại A so với B trên các cấu hình GB chung | 15 |
| Holdout | Metrics tính lại từ predictions A so với bảng A | 2 |
| Holdout | Metrics tính lại từ predictions B so với bảng B | 4 |
| Holdout | Metrics tính lại A so với B trên các cấu hình GB chung | 2 |

Kết quả:

- Tổng cộng: 63 dòng đối chiếu PASS.
- Tổng errors: 0.
- Dung sai tuyệt đối cho chỉ tiêu số thực: 1e-12.
- Số target và số violations phải khớp chính xác.
- errors là số chỉ tiêu đối chiếu không đạt,
  không phải số phiên vi phạm VaR.

HS/EWMA được kiểm tra giữa predictions B và metrics B.
Đối chiếu A/B chỉ áp dụng cho các cấu hình GB chung.

Kết quả xác nhận tính nhất quán của metrics trong phạm vi kiểm tra;
không chứng minh mô hình dự báo tương lai tốt.

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

Tỷ lệ vi phạm CV của C1 là 7.14%, cao hơn mức danh nghĩa 5%;
cần ghi nhận khi đánh giá khả năng hiệu chỉnh rủi ro.

A/B cần xác nhận phạm vi ứng viên và tiêu chí quyết định
trước khi chốt mô hình chính thức.
Không dùng kết quả holdout để lựa chọn lại cấu hình.

## 9. Giới hạn và nội dung còn mở

- Holdout trùng một phần giai đoạn đánh giá đã công bố ở Phase 1.
  Kết quả là đánh giá hồi cứu, không phải tập kiểm định
  hoàn toàn chưa từng được quan sát.
- Chuyển G04 sang C1 thay đổi cả features và hyperparameters.
  Không quy toàn bộ cải thiện cho việc tăng từ 7 lên 12 features.
- Tái lập dự báo và metrics không chứng minh hiệu quả dự báo tương lai.
- Hash khớp không xác nhận độc lập độ chính xác của giá nguồn
  hoặc cách xử lý cổ tức và các sự kiện doanh nghiệp.
- Chưa đối chiếu HS/EWMA với một bộ predictions riêng của A
  trên cùng kỳ đánh giá.
- Cần A xác nhận thiết kế chung và phạm vi lựa chọn mô hình.

## 10. Đường dẫn bằng chứng

### 10.1. Lần chạy B

Thư mục:

`results/phase2/b/aligned_2026-11-06_20261112_121348/`

Các file chính:

- `metadata.json`
- `features.csv`
- `baseline_cv.csv`
- `cv_predictions.csv`
- `cv_metrics.csv`
- `cv_metrics_summary_b.csv`
- `holdout_predictions.csv`
- `holdout_metrics.csv`
- `gbv2_retrain.csv`

Trong bộ kết quả này, `gbv2_retrain.csv` chứa
468 dự báo holdout của GBv2 C1 12 features.

Log:

`docs/phase2/qa_b/aligned_run_20261112_121348.txt`

Code:

- `src/phase2/aligned_runner_b.py`
- `scripts/run_phase2_aligned_b.py`
- `tests/test_phase2_aligned_runner_b.py`

### 10.2. Tham chiếu A

Bản tham chiếu local đã nhận từ commit A nêu ở đầu tài liệu:

`results/phase2/b/reference_a_1a593ed/results/phase2/a/training_2026-11-06/`

Các file sử dụng:

- `metadata.json`
- `cv_predictions.csv`
- `holdout_predictions.csv`
- `cv_metrics.csv`
- `holdout_metrics.csv`

Bản tham chiếu local không được coi là kết quả do B tự sinh.
Có thể lấy lại từ cùng đường dẫn `results/phase2/a/training_2026-11-06/`
tại commit A đã khóa.

### 10.3. Biểu đồ baseline

Script:

`scripts/plot_cv_b.py`

Đầu ra:

- `figures/phase2/b/aligned_2026-11-06/baseline_var_timeline.png`
- `figures/phase2/b/aligned_2026-11-06/baseline_var_timeline.pdf`
- `figures/phase2/b/aligned_2026-11-06/baseline_var_timeline_metadata.json`

### 10.4. Kiểm tra metrics độc lập

Script:

`scripts/check_metrics_ab_b.py`

Đầu ra:

- `results/phase2/b/metrics_check_2026-11-06/metric_table_b.csv`
- `results/phase2/b/metrics_check_2026-11-06/recomputed_metrics.csv`
- `results/phase2/b/metrics_check_2026-11-06/metadata.json`

Lệnh đã chạy:

`python -m scripts.check_metrics_ab_b`

## 11. Kết luận review

- [x] B xác minh đúng snapshot A.
- [x] B đối chiếu thiết kế CV và holdout.
- [x] B chạy baseline trên cùng ngày đánh giá.
- [x] B tái lập dự báo G04 và GBv2 12 features trong dung sai.
- [x] B kiểm tra dữ liệu và số violations trên hình baseline.
- [x] Hoàn thiện đối chiếu metrics trực tiếp từ predictions A.
- [x] Metrics tính lại A/B đạt 63 dòng PASS, tổng errors = 0.
- [ ] A xác nhận thiết kế chung.
- [ ] A/B xác nhận phạm vi lựa chọn mô hình chính thức.

B hoàn thành bản review kỹ thuật và bổ sung kiểm tra metrics độc lập
để A xem xét.

Chưa ghi nhận phê duyệt chung A/B hoặc hoàn thành toàn bộ gate.