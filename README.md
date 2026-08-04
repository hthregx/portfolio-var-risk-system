# Portfolio VaR Risk System

## Tên đề tài

Xây dựng hệ thống dự báo và kiểm định Value at Risk một ngày
cho danh mục cổ phiếu bằng Historical Simulation, EWMA
và Gradient Boosting.

## Thời gian

27/07/2026 - 02/09/2026

## Phương pháp

1. Historical Simulation
2. EWMA
3. Gradient Boosting Quantile Regression

## Dữ liệu dự kiến

- HPG
- FPT
- MWG
- VN-Index hoặc VN30 nếu dữ liệu ổn định

## Công nghệ

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Streamlit
- pytest
- GitHub

## Cấu trúc dự án

- `data/`: dữ liệu raw, processed và sample
- `notebooks/`: khám phá và thử nghiệm
- `src/`: mã nguồn chính
- `tests/`: kiểm thử
- `app/`: ứng dụng Streamlit
- `docs/`: tài liệu và nhật ký
- `reports/`: báo cáo
- `slides/`: slide bảo vệ

## Trạng thái

Project initialization.
# Environment Setup

Tạo môi trường ảo:

```bash
python3 -m venv .venv
```

Kích hoạt:

macOS/Linux

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

Cài thư viện:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# Data Preparation

Dữ liệu được lưu trong:

```text
data/raw/
```

Sau khi validation:

```text
data/processed/
```

---

# Run Data Loader

```bash
python src/data_loader.py
```

---

# Run Validation Pipeline

```bash
python src/data_validation_pipeline.py
```

Kết quả:

- dữ liệu sạch trong `data/processed/`
- báo cáo chất lượng trong `docs/data-quality-report.csv`

---

# Run EDA Notebook

Khởi động Jupyter:

```bash
python -m jupyter notebook
```

Notebook EDA chính:

```text
notebooks/01_eda.ipynb
```

Notebook Risk EDA:

```text
notebooks/01_eda_risk_analysis.ipynb
```

Sau khi mở notebook:

```
Restart Kernel
↓
Run All
```

---

# Run Tests

```bash
python -m pytest -v
```

---

## Cấu trúc dự án

```text
portfolio-var-risk-system/
├── app/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── figures/
│   └── eda/
├── notebooks/
├── reports/
├── slides/
├── src/
├── tests/
├── requirements.txt
└── README.md
```

---

## Trạng thái

- Data loader completed
-  Data validation pipeline completed
-  Risk-oriented EDA completed
-  Portfolio return (in progress)
-  VaR modelling (upcoming)