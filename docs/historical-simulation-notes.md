# Historical Simulation Methodology Notes

**Reviewer:** Người B  
**Implementation owner:** Người A  
**Notebook:** `notebooks/03_historical_simulation.ipynb`  
**Review scope:** Independent validation  
**Confidence level:** 95%  
**Left-tail probability:** 5%

---

## 1. Historical Simulation là gì?

Historical Simulation là phương pháp ước lượng Value at Risk dựa trực tiếp
trên phân phối thực nghiệm của lợi suất lịch sử.

Phương pháp này không giả định lợi suất danh mục tuân theo phân phối chuẩn hoặc
một phân phối tham số cụ thể. Thay vào đó, Historical Simulation sử dụng các
quan sát lợi suất danh mục đã xảy ra trong quá khứ để xác định mức tổn thất tại
đuôi trái của phân phối.

Quy trình cơ bản gồm:

1. Thu thập chuỗi lợi suất danh mục.
2. Chọn cửa sổ dữ liệu lịch sử.
3. Tính phân vị thực nghiệm tại mức alpha tương ứng.
4. Chuyển phân vị lợi suất âm thành độ lớn VaR không âm.
5. Sử dụng VaR đó làm dự báo rủi ro cho phiên giao dịch tiếp theo.

Historical Simulation phù hợp với dự án vì:

- không yêu cầu giả định lợi suất tuân theo phân phối chuẩn;
- phản ánh trực tiếp dữ liệu lịch sử;
- giữ lại đặc điểm bất đối xứng và đuôi dày;
- dễ diễn giải và kiểm tra bằng tính toán độc lập;
- phù hợp làm mô hình benchmark cho các mô hình VaR khác.

---

## 2. Tại sao sử dụng empirical quantile?

Historical Simulation sử dụng empirical quantile vì phương pháp này dựa trên
phân phối thực nghiệm của dữ liệu quan sát.

Với chuỗi lợi suất danh mục:

\[
R_1,R_2,\ldots,R_n
\]

phân vị thực nghiệm mức alpha được xác định bởi:

\[
q_{\alpha}
=
\operatorname{Quantile}_{\alpha}(R)
\]

Notebook của Người A sử dụng:

```python
np.quantile(
    returns.to_numpy(),
    alpha,
    method="linear",
)