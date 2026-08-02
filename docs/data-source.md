# Data Source

## 1. Dữ liệu cần thu thập

- HPG
- FPT
- MWG
- VN-Index hoặc VN30 nếu dữ liệu ổn định

## 2. Tần suất

Dữ liệu giao dịch theo ngày.

## 3. Các trường dữ liệu

- date
- open
- high
- low
- close
- volume

## 4. Khoảng thời gian dự kiến

Từ ngày 01/01/2020 đến thời điểm chốt dữ liệu.

## 5. Nguồn chính

Nguồn dữ liệu dự kiến: vnstock hoặc nguồn dữ liệu đã được
nhóm kiểm tra và giảng viên chấp thuận.

## 6. Cơ chế lưu trữ

- Dữ liệu tải lần đầu được lưu tại `data/raw/`.
- Không tải lại nếu file cache đã tồn tại.
- Dữ liệu đã làm sạch được lưu tại `data/processed/`.
- Repository chỉ lưu dữ liệu mẫu nhỏ tại `data/sample/`.

## 7. Kiểm tra chất lượng

- Kiểm tra ngày trùng.
- Kiểm tra giá trị thiếu.
- Kiểm tra giá OHLC không dương.
- Kiểm tra thứ tự thời gian.
- Kiểm tra các ngày giao dịch không đồng bộ.
- Kiểm tra điều chỉnh giá và corporate actions.

## 8. Rủi ro

- API thay đổi.
- Nguồn dữ liệu tạm thời không hoạt động.
- Dữ liệu chưa điều chỉnh.
- Các mã có ngày giao dịch không đồng bộ.

## Portfolio and Benchmark Data Usage

### Portfolio constituent data

The default portfolio consists of three equities:

- HPG
- FPT
- MWG

Validated processed datasets are stored at:

- `data/processed/HPG_clean.csv`
- `data/processed/FPT_clean.csv`
- `data/processed/MWG_clean.csv`

These three assets are the constituents used to construct the default
equal-weight portfolio.

### Derived portfolio data

The canonical derived portfolio-return dataset is stored at:

`data/processed/portfolio_returns.csv`

A compact review sample is stored at:

`data/sample/portfolio_returns_sample.csv`

The canonical portfolio-return dataset contains 1,637 valid daily
return observations from 2020-01-03 through 2026-07-28.

The primary return series used by the downstream Value at Risk
pipeline is:

`portfolio_simple_return`

This series is derived from the validated HPG, FPT, and MWG return
series and their portfolio weights.

### VN-Index benchmark

The raw VN-Index dataset is stored at:

`data/raw/VNINDEX.csv`

It contains 1,638 observations covering 2020-01-02 through
2026-07-28.

VN-Index is used only as a market benchmark and is not a constituent
of the default portfolio.

Its portfolio weight is therefore zero.

The raw VN-Index timestamp contains a `07:00:00` time component.
For daily trading-calendar comparison, this timestamp is normalized
to midnight. This operation removes only the time-of-day component
and does not change the trading date.

After trading-date normalization, the VN-Index and portfolio price
calendars contain 1,638 common dates and are fully aligned.

At this stage, no processed `data/processed/VNINDEX_clean.csv`
dataset has been established. Therefore, `data/raw/VNINDEX.csv`
must not be treated as model-ready processed data.

### Data lineage rule

Downstream portfolio-risk and Value at Risk components must use the
canonical `portfolio_simple_return` series from:

`data/processed/portfolio_returns.csv`

Portfolio returns must not be reconstructed from the raw VN-Index
dataset.

VN-Index remains a separate benchmark/context series unless the
project methodology is explicitly changed and revalidated.
