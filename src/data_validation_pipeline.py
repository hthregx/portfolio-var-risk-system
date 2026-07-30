"""OHLCV data validation pipeline.

Pipeline:
1. Đọc dữ liệu HPG, FPT, MWG từ data/raw.
2. Kiểm tra lỗi dữ liệu cơ bản.
3. Kiểm tra quan hệ OHLC.
4. Gắn cờ biến động giá đóng cửa tuyệt đối trên 15%.
5. Loại các dòng dữ liệu không hợp lệ.
6. Lưu dữ liệu sạch vào data/processed.
7. Xuất báo cáo vào docs/data-quality-report.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_FILE = PROJECT_ROOT / "docs" / "data-quality-report.csv"

DEFAULT_TICKERS = ("HPG", "FPT", "MWG")
PRICE_COLUMNS = ("open", "high", "low", "close")
REQUIRED_COLUMNS = ("time", "open", "high", "low", "close", "volume")

REPORT_COLUMNS = [
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
]


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột và tên cột thời gian."""

    frame = data.copy()

    frame.columns = [
        str(column).strip().lower()
        for column in frame.columns
    ]

    # Hỗ trợ trường hợp nguồn dữ liệu dùng cột date thay cho time.
    if "date" in frame.columns and "time" not in frame.columns:
        frame = frame.rename(columns={"date": "time"})

    return frame


def validate_required_columns(data: pd.DataFrame) -> None:
    """Kiểm tra các cột bắt buộc của dữ liệu OHLCV."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột và kiểu dữ liệu trước khi validation."""

    frame = normalize_columns(data)
    validate_required_columns(frame)

    frame["time"] = pd.to_datetime(
        frame["time"],
        errors="coerce",
    )

    for column in (*PRICE_COLUMNS, "volume"):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    return frame


def evaluate_ohlc_relationships(
    data: pd.DataFrame,
) -> dict[str, Any]:
    """Kiểm tra quan hệ giữa Open, High, Low và Close.

    Quy tắc:
    - high >= open
    - high >= low
    - high >= close
    - low <= open
    - low <= high
    - low <= close

    Returns
    -------
    dict
        Số dòng OHLC không hợp lệ, index và ngày tương ứng.
    """

    frame = normalize_columns(data)

    required_ohlc_columns = {"open", "high", "low", "close"}
    missing_columns = required_ohlc_columns.difference(frame.columns)

    if missing_columns:
        raise ValueError(
            "Missing OHLC columns: "
            + ", ".join(sorted(missing_columns))
        )

    invalid_mask = (
        (frame["high"] < frame["open"])
        | (frame["high"] < frame["low"])
        | (frame["high"] < frame["close"])
        | (frame["low"] > frame["open"])
        | (frame["low"] > frame["high"])
        | (frame["low"] > frame["close"])
    )

    invalid_indices = frame.index[invalid_mask].tolist()

    invalid_dates: list[str] = []

    if "time" in frame.columns:
        invalid_dates = (
            pd.to_datetime(
                frame.loc[invalid_mask, "time"],
                errors="coerce",
            )
            .dt.strftime("%Y-%m-%d")
            .dropna()
            .tolist()
        )

    return {
        "invalid_ohlc_rows": int(invalid_mask.sum()),
        "invalid_ohlc_indices": invalid_indices,
        "invalid_ohlc_dates": invalid_dates,
    }


def detect_extreme_price_moves(
    data: pd.DataFrame,
    threshold: float = 0.15,
) -> dict[str, Any]:
    """Phát hiện biến động bất thường của giá đóng cửa.

    Biến động tuyệt đối lớn hơn threshold chỉ được gắn cờ,
    không tự động xóa khỏi dữ liệu.

    Parameters
    ----------
    data:
        DataFrame chứa ít nhất cột close.
    threshold:
        Ngưỡng biến động. Mặc định 0.15 tương ứng 15%.

    Returns
    -------
    dict
        Số dòng biến động bất thường, ngày nghi ngờ và chi tiết.
    """

    if threshold <= 0:
        raise ValueError("threshold must be greater than zero")

    frame = normalize_columns(data)

    if "close" not in frame.columns:
        raise ValueError("Missing required column: close")

    close_values = pd.to_numeric(
        frame["close"],
        errors="coerce",
    )

    close_change = close_values.pct_change(fill_method=None)
    extreme_mask = close_change.abs() > threshold

    extreme_rows = frame.loc[extreme_mask].copy()
    extreme_rows["close_change"] = close_change.loc[extreme_mask]
    extreme_rows["close_change_pct"] = (
        close_change.loc[extreme_mask] * 100
    )

    extreme_dates: list[str] = []

    if "time" in frame.columns:
        extreme_dates = (
            pd.to_datetime(
                frame.loc[extreme_mask, "time"],
                errors="coerce",
            )
            .dt.strftime("%Y-%m-%d")
            .dropna()
            .tolist()
        )

    return {
        "extreme_close_move_rows": int(extreme_mask.sum()),
        "extreme_close_move_dates": extreme_dates,
        "extreme_move_indices": frame.index[extreme_mask].tolist(),
        "extreme_moves": extreme_rows,
    }


def run_base_validation_rules(
    data: pd.DataFrame,
) -> dict[str, Any]:
    """Chạy các validation rule cơ bản.

    Đây là lớp adapter cho phần validation rules của thành viên A.
    Khi nhóm đã có hàm riêng của A, có thể gọi hàm đó tại đây
    và giữ nguyên định dạng kết quả bên dưới.
    """

    missing_mask = data[list(REQUIRED_COLUMNS)].isna().any(axis=1)

    duplicate_date_mask = data.duplicated(
        subset=["time"],
        keep="first",
    )

    nonpositive_price_mask = (
        (data["open"] <= 0)
        | (data["high"] <= 0)
        | (data["low"] <= 0)
        | (data["close"] <= 0)
    )

    negative_volume_mask = data["volume"] < 0

    dates_out_of_order = int(
        (
            data["time"]
            .diff()
            .dropna()
            < pd.Timedelta(0)
        ).sum()
    )

    return {
        "missing_mask": missing_mask,
        "duplicate_date_mask": duplicate_date_mask,
        "nonpositive_price_mask": nonpositive_price_mask,
        "negative_volume_mask": negative_volume_mask,
        "missing_rows": int(missing_mask.sum()),
        "duplicate_date_rows": int(duplicate_date_mask.sum()),
        "nonpositive_price_rows": int(
            nonpositive_price_mask.sum()
        ),
        "negative_volume_rows": int(
            negative_volume_mask.sum()
        ),
        "dates_out_of_order": dates_out_of_order,
    }


def validate_ticker_data(
    ticker: str,
    input_file: Path,
    processed_data_dir: Path,
    threshold: float = 0.15,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Chạy validation và tạo dữ liệu sạch cho một ticker."""

    raw_data = pd.read_csv(input_file)
    raw_rows = len(raw_data)

    data = prepare_data(raw_data)

    base_result = run_base_validation_rules(data)
    ohlc_result = evaluate_ohlc_relationships(data)
    extreme_result = detect_extreme_price_moves(
        data,
        threshold=threshold,
    )

    invalid_ohlc_mask = pd.Series(
        False,
        index=data.index,
        dtype=bool,
    )

    invalid_ohlc_mask.loc[
        ohlc_result["invalid_ohlc_indices"]
    ] = True

    # Biến động trên 15% không nằm trong remove_mask.
    remove_mask = (
        base_result["missing_mask"]
        | base_result["duplicate_date_mask"]
        | base_result["nonpositive_price_mask"]
        | base_result["negative_volume_mask"]
        | invalid_ohlc_mask
    )

    clean_data = (
        data.loc[~remove_mask]
        .sort_values("time")
        .reset_index(drop=True)
    )

    processed_data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_file = (
        processed_data_dir / f"{ticker}_clean.csv"
    )

    clean_data.to_csv(
        processed_file,
        index=False,
        encoding="utf-8-sig",
    )

    clean_rows = len(clean_data)
    rows_removed = raw_rows - clean_rows

    if clean_data.empty:
        start_date = ""
        end_date = ""
    else:
        start_date = (
            clean_data["time"].min().strftime("%Y-%m-%d")
        )
        end_date = (
            clean_data["time"].max().strftime("%Y-%m-%d")
        )

    has_quality_issue = any(
        [
            base_result["missing_rows"] > 0,
            base_result["duplicate_date_rows"] > 0,
            base_result["nonpositive_price_rows"] > 0,
            base_result["negative_volume_rows"] > 0,
            ohlc_result["invalid_ohlc_rows"] > 0,
            base_result["dates_out_of_order"] > 0,
            extreme_result["extreme_close_move_rows"] > 0,
        ]
    )

    quality_status = (
        "REVIEW"
        if has_quality_issue
        else "PASS"
    )

    try:
        processed_file_display = str(
            processed_file.relative_to(PROJECT_ROOT)
        )
    except ValueError:
        processed_file_display = str(processed_file)

    report_row = {
        "ticker": ticker,
        "quality_status": quality_status,
        "raw_rows": raw_rows,
        "clean_rows": clean_rows,
        "rows_removed": rows_removed,
        "missing_rows": base_result["missing_rows"],
        "duplicate_date_rows": (
            base_result["duplicate_date_rows"]
        ),
        "nonpositive_price_rows": (
            base_result["nonpositive_price_rows"]
        ),
        "negative_volume_rows": (
            base_result["negative_volume_rows"]
        ),
        "invalid_ohlc_rows": (
            ohlc_result["invalid_ohlc_rows"]
        ),
        "dates_out_of_order": (
            base_result["dates_out_of_order"]
        ),
        "extreme_close_move_rows": (
            extreme_result["extreme_close_move_rows"]
        ),
        "start_date": start_date,
        "end_date": end_date,
        "processed_file": processed_file_display,
        "pipeline_status": "success",
    }

    details = {
        "ticker": ticker,
        "invalid_ohlc_dates": (
            ohlc_result["invalid_ohlc_dates"]
        ),
        "extreme_close_move_dates": (
            extreme_result["extreme_close_move_dates"]
        ),
        "extreme_moves": extreme_result["extreme_moves"],
    }

    return report_row, details


def failed_report_row(
    ticker: str,
    pipeline_status: str,
) -> dict[str, Any]:
    """Tạo dòng báo cáo khi pipeline thất bại."""

    return {
        "ticker": ticker,
        "quality_status": "FAIL",
        "raw_rows": 0,
        "clean_rows": 0,
        "rows_removed": 0,
        "missing_rows": 0,
        "duplicate_date_rows": 0,
        "nonpositive_price_rows": 0,
        "negative_volume_rows": 0,
        "invalid_ohlc_rows": 0,
        "dates_out_of_order": 0,
        "extreme_close_move_rows": 0,
        "start_date": "",
        "end_date": "",
        "processed_file": "",
        "pipeline_status": pipeline_status,
    }


def run_validation_pipeline(
    tickers: tuple[str, ...] | list[str] | None = None,
    raw_data_dir: Path | None = None,
    processed_data_dir: Path | None = None,
    report_file: Path | None = None,
    threshold: float = 0.15,
) -> pd.DataFrame:
    """Chạy validation pipeline cho toàn bộ ticker.

    Mặc định chạy HPG, FPT và MWG.
    """

    selected_tickers = (
        list(tickers)
        if tickers is not None
        else list(DEFAULT_TICKERS)
    )

    source_dir = (
        Path(raw_data_dir)
        if raw_data_dir is not None
        else RAW_DATA_DIR
    )

    output_dir = (
        Path(processed_data_dir)
        if processed_data_dir is not None
        else PROCESSED_DATA_DIR
    )

    output_report = (
        Path(report_file)
        if report_file is not None
        else REPORT_FILE
    )

    report_rows: list[dict[str, Any]] = []

    for ticker in selected_tickers:
        input_file = source_dir / f"{ticker}.csv"

        print(f"Validating {ticker}...")

        if not input_file.exists():
            report_rows.append(
                failed_report_row(
                    ticker=ticker,
                    pipeline_status="file_not_found",
                )
            )
            print(f"File not found: {input_file}")
            continue

        try:
            report_row, details = validate_ticker_data(
                ticker=ticker,
                input_file=input_file,
                processed_data_dir=output_dir,
                threshold=threshold,
            )

            report_rows.append(report_row)

            extreme_dates = details[
                "extreme_close_move_dates"
            ]

            print(
                f"Completed {ticker}: "
                f"{report_row['raw_rows']} raw rows, "
                f"{report_row['clean_rows']} clean rows, "
                f"{report_row['rows_removed']} removed, "
                f"{report_row['extreme_close_move_rows']} "
                "extreme moves."
            )

            if extreme_dates:
                print(
                    "Extreme move dates: "
                    + ", ".join(extreme_dates)
                )

        except Exception as error:
            report_rows.append(
                failed_report_row(
                    ticker=ticker,
                    pipeline_status=(
                        f"failed: {type(error).__name__}: "
                        f"{error}"
                    ),
                )
            )

            print(f"Validation failed for {ticker}: {error}")

    report = pd.DataFrame(
        report_rows,
        columns=REPORT_COLUMNS,
    )

    output_report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        output_report,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nData quality report:")
    print(report.to_string(index=False))
    print(f"\nReport saved to: {output_report}")

    return report


def main() -> None:
    """Điểm bắt đầu khi chạy trực tiếp file."""

    run_validation_pipeline()


if __name__ == "__main__":
    main()