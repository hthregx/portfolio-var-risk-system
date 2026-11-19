# KẾT QUẢ CUỐI CÙNG — C1 gốc đã khóa

**Một mô hình duy nhất:** Gradient Boosting quantile regression C1 (12 đặc trưng; 150 cây; depth 2; learning rate 0.03; min_samples_leaf 10). Dự báo **phân vị 5% lợi suất danh mục một phiên tiếp theo**, tương ứng VaR 95%.

- Nguồn dữ liệu: KBS; HPG/FPT/MWG tỷ trọng 1/3, snapshot đến ngày **06/11/2026**.
- SHA256 bộ features thực: `14ac5dd4ce7dd6090420b9cee06c8ac40935c54691b4daf95d063b71a075eccc`.
- 1.179 quan sát development, 468 phiên holdout hồi cứu; CV 5 folds không xáo trộn thời gian.
- CV mean Pinball Loss: **0.002080267**.
- Holdout Pinball Loss: **0.001636961**.
- Vi phạm VaR: **25/468 = 5.34%** (kỳ vọng 5%).
- Kupiec p-value: **0.73703**. Christoffersen independence p-value: **0.53465**.
- Sai khác lớn nhất so với file dự báo C1 gốc đã công bố trong repository: **0**.

## Giải thích biểu đồ

`01_C1_VaR_with_Breaches.png`: đường xanh xám là lợi suất thực tế; đường cam là **ngưỡng phân vị 5% được dự báo trước phiên** (giá trị âm, tương ứng với ngưỡng VaR). Dấu X đỏ đánh dấu ngày có lợi suất thực tế thấp hơn ngưỡng. Biểu đồ dưới cho thấy tỷ lệ vi phạm trượt 60 phiên, đối chiếu mức danh nghĩa 5%. **Không đánh giá chất lượng VaR bằng việc đường cam bám sát lợi suất thực tế**.

`02_C1_Coverage.png`: phân phối VaR và mức vi phạm thực tế so với mức 5%.

`03_C1_Training_Validation_Loss.png`: đường loss theo 150 cây ở Fold 5 để quan sát quá trình học; không thay thế backtesting.

## Giới hạn

- Đây là một phép **tái tạo thực nghiệm cũ**, không phải dự báo thị trường trực tiếp ngày 19/11/2026.
- Holdout hồi cứu giao thoa giai đoạn Phase 1 đã phân tích, **không phải holdout hoàn toàn chưa từng được xem xét**.
- Tỷ lệ vi phạm gần 5% và p-value lớn **không chứng minh mô hình hoàn hảo**; chỉ cho biết kiểm định chưa bác bỏ giả thuyết ở mức ý nghĩa tương ứng.
- Không thực hiện tối ưu tham số hoặc chọn lại mô hình dựa vào holdout.
