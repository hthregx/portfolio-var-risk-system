# Weekly Note #1

**Week:** 27/07–02/08/2026

## 1. Completed Work

- Tải dữ liệu HPG, FPT và MWG.
- Xây dựng và chạy data-validation pipeline.
- Kiểm tra missing values, duplicate dates, giá không dương, volume âm và
  quan hệ OHLC.
- Tạo dữ liệu processed.
- Hoàn thành EDA cơ bản.
- Hoàn thành Risk-oriented EDA:
  - Outlier analysis
  - Correlation analysis
  - Rolling volatility 20 phiên
  - Tail behavior
- Tạo đầy đủ các hình EDA từ 01 đến 08.
- Hoàn thành notebook riêng `01_eda_risk_analysis.ipynb`.

## 2. Data Source

Các ticker được sử dụng:

- HPG
- FPT
- MWG

Mỗi ticker có:

- 1.638 quan sát
- Ngày bắt đầu: 2020-01-02
- Ngày kết thúc: 2026-07-28

Dữ liệu được căn chỉnh theo ngày giao dịch.

## 3. Data Validation Results

- HPG: PASS
- FPT: PASS
- MWG: PASS
- Không có dòng bị loại trong lần chạy hiện tại.
- Không phát hiện lỗi quan hệ OHLC.
- Không phát hiện giá không dương hoặc volume âm.
- Data validation unit tests: PASS.

## 4. EDA Findings

- Daily log return chủ yếu dao động quanh 0 nhưng có các phiên cực đoan.
- Extreme returns không bị tự động xóa.
- Cặp có correlation cao nhất: FPT–MWG, 0.5423.
- Cặp có correlation thấp nhất: HPG–FPT, 0.4926.
- MWG có rolling volatility 20 phiên trung bình cao nhất.
- MWG có q05 âm nhất, đạt -3.6333%.
- Phân phối lợi suất có skewness âm và excess kurtosis dương.

## 5. Important Technical Decisions

- Sử dụng daily log return cho EDA rủi ro.
- Correlation được tính sau khi căn chỉnh theo ngày.
- Không annualize rolling volatility ở bước EDA đầu tiên.
- Không xóa outlier tự động.
- Sử dụng dữ liệu từ `data/processed`, không dùng trực tiếp `data/raw`.
- Tách notebook Risk EDA để tránh conflict Git.

## 6. Problems Encountered

- Cài đặt package gặp lỗi do `pywinpty` trên macOS.
- Jupyter từng sử dụng sai kernel nên không import được pandas.
- Notebook `.ipynb` có nguy cơ conflict khi nhiều người cùng sửa.
- Một số lệnh PowerShell không chạy được trên zsh của macOS.

## 7. Problems Resolved

- Loại `pywinpty` khỏi requirements trên macOS.
- Chọn lại kernel thuộc `.venv`.
- Tạo notebook riêng `01_eda_risk_analysis.ipynb`.
- Thay lệnh PowerShell bằng lệnh tương thích macOS.
- Tách branch `feature/eda-risk`.

## 8. Remaining Risks

- Chưa tích hợp Risk EDA vào notebook chính.
- Data Dictionary chưa được cập nhật đầy đủ sau integration.
- Chưa bổ sung kết quả portfolio return của Người A.
- Chưa hoàn thành full reproducibility test trên một bản clone sạch.
- Chưa thực hiện final QA cho package M1.

## 9. Next-Week Plan

- Tích hợp notebook EDA của Người A và Người B.
- Chạy lại notebook chính từ đầu.
- Kiểm tra tái tạo đủ 8 hình.
- Cập nhật Data Dictionary.
- Bổ sung kết quả portfolio return.
- Hoàn thiện README và logs.
- Chuẩn bị package M1.
- Chỉ tag `v0.2-data` sau khi full QA đạt yêu cầu.

