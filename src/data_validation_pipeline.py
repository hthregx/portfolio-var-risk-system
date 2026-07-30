"""Pipeline kiểm tra và làm sạch dữ liệu OHLCV.

Luồng xử lý:
1. Đọc dữ liệu HPG, FPT và MWG từ ``data/raw``.
2. Dùng validation rules của thành viên A để kiểm tra lỗi cấu trúc.
3. Dùng validation rules của thành viên A để làm sạch cấu trúc và chuẩn hóa
   schema thành ``date + ticker + OHLCV``.
4. Kiểm tra quan hệ giữa open, high, low và close.
5. Loại các dòng vi phạm quan hệ OHLC.
6. Gắn cờ biến động giá đóng cửa tuyệt đối vượt ngưỡng cấu hình.
7. Lưu dữ liệu sạch vào ``data/processed``.
8. Xuất báo cáo vào ``docs/data-quality-report.csv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

# Hỗ trợ cả hai cách chạy:
# - python -m src.data_validation_pipeline
# - python src/data_validation_pipeline.py
if __package__:
    from .validation_rules import (
        clean_structural_issues,
        evaluate_structural_rules,
        normalize_column_names,
        parse_trading_dates,
    )
else:
    from validation_rules import (
        clean_structural_issues,
        evaluate_structural_rules,
        normalize_column_names,
        parse_trading_dates,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_FILE = PROJECT_ROOT / "docs" / "data-quality-report.csv"

DEFAULT_TICKERS = ("HPG", "FPT", "MWG")
PRICE_COLUMNS = ("open", "high", "low", "close")

REPORT_COLUMNS = [
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
]


def evaluate_ohlc_relationships(
    data: pd.DataFrame,
) -> dict[str, Any]:
    """Kiểm tra quan hệ hợp lệ giữa open, high, low và close.

    Quy tắc:
    - high >= open
    - high >= low
    - high >= close
    - low <= open
    - low <= high
    - low <= close

    Hàm chỉ phát hiện lỗi, không tự động chỉnh sửa giá.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Input data is empty.")

    frame = normalize_column_names(data)

    required_columns = {
        "date",
        "open",
        "high",
        "low",
        "close",
    }

    missing_columns = sorted(
        required_columns.difference(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing OHLC columns: {missing_columns}"
        )

    frame["date"] = parse_trading_dates(frame["date"])

    for column in PRICE_COLUMNS:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    invalid_mask = (
        frame["high"].lt(frame["open"])
        | frame["high"].lt(frame["low"])
        | frame["high"].lt(frame["close"])
        | frame["low"].gt(frame["open"])
        | frame["low"].gt(frame["high"])
        | frame["low"].gt(frame["close"])
    )

    invalid_indices = frame.index[
        invalid_mask
    ].tolist()

    invalid_dates = (
        frame.loc[invalid_mask, "date"]
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
    """Gắn cờ biến động tuyệt đối của close lớn hơn ``threshold``.

    Các dòng được gắn cờ để kiểm tra thủ công nhưng không tự động bị loại.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Input data is empty.")

    if threshold <= 0:
        raise ValueError(
            "threshold must be greater than zero."
        )

    frame = normalize_column_names(data)

    required_columns = {"date", "close"}
    missing_columns = sorted(
        required_columns.difference(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    frame["date"] = parse_trading_dates(frame["date"])
    frame["close"] = pd.to_numeric(
        frame["close"],
        errors="coerce",
    )

    frame = (
        frame
        .sort_values("date", kind="mergesort")
        .copy()
    )

    close_change = frame["close"].pct_change(
        fill_method=None
    )

    extreme_mask = close_change.abs().gt(threshold)

    extreme_rows = frame.loc[
        extreme_mask
    ].copy()

    extreme_rows["close_change"] = close_change.loc[
        extreme_mask
    ]

    extreme_rows["close_change_pct"] = (
        extreme_rows["close_change"] * 100
    )

    extreme_dates = (
        extreme_rows["date"]
        .dt.strftime("%Y-%m-%d")
        .dropna()
        .tolist()
    )

    return {
        "extreme_close_move_rows": int(
            extreme_mask.sum()
        ),
        "extreme_close_move_dates": extreme_dates,
        "extreme_move_indices": extreme_rows.index.tolist(),
        "extreme_moves": extreme_rows,
    }


def validate_ticker_data(
    ticker: str,
    input_file: Path,
    processed_data_dir: Path,
    threshold: float = 0.15,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Kiểm tra, làm sạch và lưu dữ liệu cho một mã cổ phiếu."""
    normalized_ticker = str(ticker).strip().upper()

    if not normalized_ticker:
        raise ValueError("ticker cannot be empty.")

    raw_data = pd.read_csv(input_file)
    raw_rows = len(raw_data)

    # Phần của thành viên A: đo lường lỗi cấu trúc trên dữ liệu raw.
    base_result = evaluate_structural_rules(raw_data)

    # Phần của thành viên A: làm sạch missing, duplicate, giá không dương,
    # volume âm; chuẩn hóa schema thành date + ticker + OHLCV.
    structurally_clean_data = clean_structural_issues(
        data=raw_data,
        ticker=normalized_ticker,
    )

    # Phần của thành viên B: kiểm tra quan hệ OHLC.
    ohlc_result = evaluate_ohlc_relationships(
        structurally_clean_data
    )

    invalid_ohlc_mask = pd.Series(
        False,
        index=structurally_clean_data.index,
        dtype=bool,
    )

    invalid_ohlc_mask.loc[
        ohlc_result["invalid_ohlc_indices"]
    ] = True

    # Chỉ loại thêm các dòng vi phạm quan hệ OHLC.
    clean_data = (
        structurally_clean_data
        .loc[~invalid_ohlc_mask]
        .sort_values("date", kind="mergesort")
        .reset_index(drop=True)
    )

    # Biến động lớn chỉ được gắn cờ, không nằm trong điều kiện loại dòng.
    extreme_result = detect_extreme_price_moves(
        clean_data,
        threshold=threshold,
    )

    processed_data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_file = (
        processed_data_dir
        / f"{normalized_ticker}_clean.csv"
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
            clean_data["date"]
            .min()
            .strftime("%Y-%m-%d")
        )

        end_date = (
            clean_data["date"]
            .max()
            .strftime("%Y-%m-%d")
        )

    has_quality_issue = any(
        [
            base_result["date_parse_failures"] > 0,
            base_result["missing_rows"] > 0,
            base_result["duplicate_date_rows"] > 0,
            base_result["nonpositive_price_rows"] > 0,
            base_result["negative_volume_rows"] > 0,
            bool(base_result["dates_out_of_order"]),
            ohlc_result["invalid_ohlc_rows"] > 0,
            extreme_result["extreme_close_move_rows"] > 0,
        ]
    )

    quality_status = (
        "REVIEW"
        if has_quality_issue
        else "PASS"
    )

    try:
        processed_file_display = (
            processed_file
            .relative_to(PROJECT_ROOT)
            .as_posix()
        )
    except ValueError:
        processed_file_display = str(processed_file)

    report_row = {
        "ticker": normalized_ticker,
        "quality_status": quality_status,
        "raw_rows": raw_rows,
        "clean_rows": clean_rows,
        "rows_removed": rows_removed,
        "date_parse_failures": (
            base_result["date_parse_failures"]
        ),
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
        "ticker": normalized_ticker,
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
    """Tạo dòng báo cáo khi pipeline không xử lý được một ticker."""
    return {
        "ticker": str(ticker).strip().upper(),
        "quality_status": "FAIL",
        "raw_rows": 0,
        "clean_rows": 0,
        "rows_removed": 0,
        "date_parse_failures": 0,
        "missing_rows": 0,
        "duplicate_date_rows": 0,
        "nonpositive_price_rows": 0,
        "negative_volume_rows": 0,
        "invalid_ohlc_rows": 0,
        "dates_out_of_order": False,
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

    Mặc định xử lý HPG, FPT và MWG.
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
        normalized_ticker = str(ticker).strip().upper()
        input_file = source_dir / f"{normalized_ticker}.csv"

        print(f"Validating {normalized_ticker}...")

        if not input_file.exists():
            report_rows.append(
                failed_report_row(
                    ticker=normalized_ticker,
                    pipeline_status="file_not_found",
                )
            )

            print(f"File not found: {input_file}")
            continue

        try:
            report_row, details = validate_ticker_data(
                ticker=normalized_ticker,
                input_file=input_file,
                processed_data_dir=output_dir,
                threshold=threshold,
            )

            report_rows.append(report_row)

            print(
                f"Completed {normalized_ticker}: "
                f"{report_row['raw_rows']} raw rows, "
                f"{report_row['clean_rows']} clean rows, "
                f"{report_row['rows_removed']} removed, "
                f"{report_row['extreme_close_move_rows']} "
                "extreme moves."
            )

            invalid_ohlc_dates = details[
                "invalid_ohlc_dates"
            ]

            if invalid_ohlc_dates:
                print(
                    "Invalid OHLC dates: "
                    + ", ".join(invalid_ohlc_dates)
                )

            extreme_dates = details[
                "extreme_close_move_dates"
            ]

            if extreme_dates:
                print(
                    "Extreme move dates: "
                    + ", ".join(extreme_dates)
                )

        except Exception as error:
            report_rows.append(
                failed_report_row(
                    ticker=normalized_ticker,
                    pipeline_status=(
                        f"failed: {type(error).__name__}: "
                        f"{error}"
                    ),
                )
            )

            print(
                f"Validation failed for "
                f"{normalized_ticker}: {error}"
            )

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
