from pathlib import Path

import pandas as pd

from src.data_validation_pipeline import (
    detect_extreme_price_moves,
    evaluate_ohlc_relationships,
    run_validation_pipeline,
)


def create_valid_data() -> pd.DataFrame:
    """Tạo dữ liệu OHLCV hợp lệ dùng chung cho unit tests."""

    return pd.DataFrame(
        {
            "time": [
                "2026-01-02",
                "2026-01-05",
                "2026-01-06",
            ],
            "open": [
                100.0,
                102.0,
                104.0,
            ],
            "high": [
                105.0,
                108.0,
                110.0,
            ],
            "low": [
                98.0,
                100.0,
                103.0,
            ],
            "close": [
                102.0,
                105.0,
                108.0,
            ],
            "volume": [
                1000,
                1200,
                1500,
            ],
        }
    )


def test_valid_ohlc_relationship_passes() -> None:
    """Dữ liệu OHLC hợp lệ không được báo lỗi."""

    data = create_valid_data()

    result = evaluate_ohlc_relationships(data)

    assert result["invalid_ohlc_rows"] == 0
    assert result["invalid_ohlc_indices"] == []
    assert result["invalid_ohlc_dates"] == []


def test_invalid_high_row_is_detected() -> None:
    """Phát hiện dòng có high thấp hơn open và close."""

    data = create_valid_data()

    data.loc[1, "high"] = 99.0

    result = evaluate_ohlc_relationships(data)

    assert result["invalid_ohlc_rows"] == 1
    assert result["invalid_ohlc_indices"] == [1]
    assert result["invalid_ohlc_dates"] == [
        "2026-01-05",
    ]


def test_extreme_move_is_flagged_but_not_removed() -> None:
    """Biến động lớn chỉ được gắn cờ, không sửa dữ liệu gốc."""

    data = create_valid_data()

    # Giá đóng cửa tăng từ 100 lên 130, tương ứng 30%.
    data.loc[0, "close"] = 100.0
    data.loc[1, "close"] = 130.0
    data.loc[1, "high"] = 135.0

    original_data = data.copy(deep=True)
    original_row_count = len(data)

    result = detect_extreme_price_moves(
        data,
        threshold=0.15,
    )

    assert result["extreme_close_move_rows"] >= 1

    assert "2026-01-05" in (
        result["extreme_close_move_dates"]
    )

    # Hàm không được xóa dòng.
    assert len(data) == original_row_count

    # Hàm không được sửa DataFrame đầu vào.
    pd.testing.assert_frame_equal(
        data,
        original_data,
    )


def test_pipeline_creates_quality_report(
    tmp_path: Path,
) -> None:
    """Pipeline tạo dữ liệu sạch và báo cáo đúng schema."""

    raw_dir = tmp_path / "data" / "raw"

    processed_dir = (
        tmp_path / "data" / "processed"
    )

    report_file = (
        tmp_path
        / "docs"
        / "data-quality-report.csv"
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_data = create_valid_data()

    raw_file = raw_dir / "MWG.csv"

    raw_data.to_csv(
        raw_file,
        index=False,
    )

    report = run_validation_pipeline(
        tickers=["MWG"],
        raw_data_dir=raw_dir,
        processed_data_dir=processed_dir,
        report_file=report_file,
    )

    processed_file = (
        processed_dir / "MWG_clean.csv"
    )

    # Pipeline phải tạo đủ hai output.
    assert report_file.exists()
    assert processed_file.exists()

    # Báo cáo chỉ có một dòng cho MWG.
    assert len(report) == 1

    required_report_columns = {
        "ticker",
        "quality_status",
        "raw_rows",
        "clean_rows",
        "rows_removed",
        "date_parse_failures",
        "missing_rows",
        "duplicate_date_rows",
        "nonpositive_price_rows",
        "negative_volume_rows",
        "invalid_ohlc_rows",
        "dates_out_of_order",
        "extreme_close_move_rows",
        "start_date",
        "end_date",
        "processed_file",
        "pipeline_status",
    }

    assert required_report_columns.issubset(
        report.columns
    )

    report_row = report.iloc[0]

    assert report_row["ticker"] == "MWG"
    assert report_row["quality_status"] == "PASS"
    assert report_row["pipeline_status"] == "success"

    assert report_row["raw_rows"] == 3
    assert report_row["clean_rows"] == 3
    assert report_row["rows_removed"] == 0

    assert report_row["date_parse_failures"] == 0
    assert report_row["missing_rows"] == 0
    assert report_row["duplicate_date_rows"] == 0
    assert report_row["nonpositive_price_rows"] == 0
    assert report_row["negative_volume_rows"] == 0
    assert report_row["invalid_ohlc_rows"] == 0
    assert report_row["extreme_close_move_rows"] == 0

    assert report_row["start_date"] == "2026-01-02"
    assert report_row["end_date"] == "2026-01-06"

    # Kiểm tra file CSV báo cáo thực tế.
    saved_report = pd.read_csv(
        report_file
    )

    assert len(saved_report) == 1
    assert saved_report.loc[0, "ticker"] == "MWG"
    assert (
        saved_report.loc[0, "pipeline_status"]
        == "success"
    )

    # Kiểm tra file dữ liệu processed thực tế.
    processed_data = pd.read_csv(
        processed_file
    )

    expected_processed_columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    assert processed_data.columns.tolist() == (
        expected_processed_columns
    )

    # Schema cũ không được còn cột time.
    assert "time" not in processed_data.columns

    # Schema mới phải có date và ticker.
    assert "date" in processed_data.columns
    assert "ticker" in processed_data.columns

    # Tất cả dòng phải thuộc mã MWG.
    assert processed_data["ticker"].eq(
        "MWG"
    ).all()

    # Không làm mất dòng dữ liệu hợp lệ.
    assert len(processed_data) == 3

    # Không có missing value trong file sạch.
    assert processed_data.isna().sum().sum() == 0

    processed_dates = pd.to_datetime(
        processed_data["date"],
        errors="coerce",
    )

    # Ngày phải parse được và được sắp xếp tăng dần.
    assert processed_dates.notna().all()
    assert processed_dates.is_monotonic_increasing

    assert processed_dates.min() == pd.Timestamp(
        "2026-01-02"
    )

    assert processed_dates.max() == pd.Timestamp(
        "2026-01-06"
    )