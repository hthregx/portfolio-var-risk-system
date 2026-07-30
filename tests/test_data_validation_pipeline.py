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
            "open": [100.0, 102.0, 104.0],
            "high": [105.0, 108.0, 110.0],
            "low": [98.0, 100.0, 103.0],
            "close": [102.0, 105.0, 108.0],
            "volume": [1000, 1200, 1500],
        }
    )


def test_valid_ohlc_relationship_passes() -> None:
    data = create_valid_data()

    result = evaluate_ohlc_relationships(data)

    assert result["invalid_ohlc_rows"] == 0
    assert result["invalid_ohlc_indices"] == []


def test_invalid_high_row_is_detected() -> None:
    data = create_valid_data()

    # High thấp hơn open và close nên dòng này không hợp lệ.
    data.loc[1, "high"] = 99.0

    result = evaluate_ohlc_relationships(data)

    assert result["invalid_ohlc_rows"] == 1
    assert result["invalid_ohlc_indices"] == [1]


def test_extreme_move_is_flagged_but_not_removed() -> None:
    data = create_valid_data()

    # Từ 100 lên 130 tương ứng tăng 30%.
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
    assert "2026-01-05" in result["extreme_close_move_dates"]

    # Hàm chỉ gắn cờ, không xóa hay sửa dữ liệu gốc.
    assert len(data) == original_row_count

    pd.testing.assert_frame_equal(
        data,
        original_data,
    )


def test_pipeline_creates_quality_report(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    report_file = (
        tmp_path / "docs" / "data-quality-report.csv"
    )

    raw_dir.mkdir(parents=True)

    data = create_valid_data()
    data.to_csv(
        raw_dir / "MWG.csv",
        index=False,
    )

    report = run_validation_pipeline(
        tickers=["MWG"],
        raw_data_dir=raw_dir,
        processed_data_dir=processed_dir,
        report_file=report_file,
    )

    processed_file = processed_dir / "MWG_clean.csv"

    assert report_file.exists()
    assert processed_file.exists()
    assert len(report) == 1

    required_columns = {
        "ticker",
        "quality_status",
        "raw_rows",
        "clean_rows",
        "rows_removed",
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

    assert required_columns.issubset(report.columns)
    assert report.loc[0, "ticker"] == "MWG"
    assert report.loc[0, "pipeline_status"] == "success"
    assert report.loc[0, "raw_rows"] == 3
    assert report.loc[0, "clean_rows"] == 3
    assert report.loc[0, "rows_removed"] == 0