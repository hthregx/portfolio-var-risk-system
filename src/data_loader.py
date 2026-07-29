from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from vnstock import Quote


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

REQUIRED_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """
    Read the YAML configuration file and return the project settings.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or "data" not in config:
        raise ValueError("The configuration file must contain a 'data' section.")

    return config


def normalize_ohlcv(
    raw_data: pd.DataFrame,
    symbol: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Standardize an OHLCV DataFrame while preserving diagnostic information.

    The raw source file remains unchanged in data/raw.
    The standardized result is used only for validation and sample generation.
    """
    if raw_data is None or raw_data.empty:
        raise ValueError(f"No data returned for symbol {symbol}.")

    data = raw_data.copy()

    data.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in data.columns
    ]

    if "date" not in data.columns and "time" in data.columns:
        data = data.rename(columns={"time": "date"})

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))

    if missing_columns:
        raise ValueError(
            f"{symbol} is missing required columns: {missing_columns}. "
            f"Available columns: {data.columns.tolist()}"
        )

    data = data[REQUIRED_COLUMNS].copy()

    parsed_dates = pd.to_datetime(
        data["date"],
        errors="coerce",
        utc=True,
    )

    data["date"] = (
        parsed_dates
        .dt.tz_convert("Asia/Ho_Chi_Minh")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    numeric_columns = ["open", "high", "low", "close", "volume"]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    missing_rows_before_cleaning = int(
        data[REQUIRED_COLUMNS].isna().any(axis=1).sum()
    )

    duplicate_dates_before_cleaning = int(
        data["date"].duplicated(keep=False).sum()
    )

    valid_positive_prices = (
        data[["open", "high", "low", "close"]] > 0
    ).all(axis=1)

    valid_volume = data["volume"].ge(0)

    maximum_reference_price = data[
        ["open", "low", "close"]
    ].max(axis=1)

    minimum_reference_price = data[
        ["open", "high", "close"]
    ].min(axis=1)

    valid_ohlc_relationship = (
        data["high"].ge(maximum_reference_price)
        & data["low"].le(minimum_reference_price)
    )

    invalid_ohlc_rows = int(
        (~valid_ohlc_relationship.fillna(False)).sum()
    )

    data = data.dropna(subset=REQUIRED_COLUMNS)

    data = data.loc[
        valid_positive_prices
        & valid_volume
        & valid_ohlc_relationship
    ].copy()

    data = (
        data
        .sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .reset_index(drop=True)
    )

    data.insert(1, "ticker", symbol)

    diagnostics = {
        "symbol": symbol,
        "raw_rows": len(raw_data),
        "clean_rows": len(data),
        "missing_rows_before_cleaning": missing_rows_before_cleaning,
        "duplicate_dates_before_cleaning": duplicate_dates_before_cleaning,
        "invalid_ohlc_rows": invalid_ohlc_rows,
        "start_date": (
            data["date"].min().strftime("%Y-%m-%d")
            if not data.empty
            else None
        ),
        "end_date": (
            data["date"].max().strftime("%Y-%m-%d")
            if not data.empty
            else None
        ),
    }

    return data, diagnostics


def fetch_symbol(
    symbol: str,
    data_config: dict[str, Any],
    refresh: bool = False,
) -> dict[str, Any]:
    """
    Download one symbol or load it from the local raw-data cache.
    """
    normalized_symbol = symbol.upper().strip()

    if not normalized_symbol:
        raise ValueError("Symbol cannot be empty.")

    raw_directory = PROJECT_ROOT / data_config["raw_directory"]
    sample_directory = PROJECT_ROOT / data_config["sample_directory"]

    raw_directory.mkdir(parents=True, exist_ok=True)
    sample_directory.mkdir(parents=True, exist_ok=True)

    raw_file = raw_directory / f"{normalized_symbol}.csv"
    sample_file = sample_directory / f"{normalized_symbol}_sample.csv"

    if raw_file.exists() and not refresh:
        raw_data = pd.read_csv(raw_file)
        retrieval_method = "cache"
    else:
        quote = Quote(
            symbol=normalized_symbol,
            source=data_config["source"],
        )

        raw_data = quote.history(
            start=data_config["start_date"],
            end=data_config["end_date"],
            interval=data_config["interval"],
        )

        if raw_data is None or raw_data.empty:
            raise ValueError(
                f"No market data returned for {normalized_symbol}."
            )

        raw_data.to_csv(raw_file, index=False)
        retrieval_method = "api"

    normalized_data, diagnostics = normalize_ohlcv(
        raw_data=raw_data,
        symbol=normalized_symbol,
    )

    sample_rows = int(data_config.get("sample_rows", 10))

    normalized_data.head(sample_rows).to_csv(
        sample_file,
        index=False,
    )

    diagnostics.update(
        {
            "required": True,
            "status": "success",
            "retrieval_method": retrieval_method,
            "raw_file": raw_file.relative_to(PROJECT_ROOT).as_posix(),
            "sample_file": sample_file.relative_to(PROJECT_ROOT).as_posix(),
        }
    )

    return diagnostics


def main() -> None:
    """
    Download required stock data and optionally attempt the market index.
    """
    config = load_config()
    data_config = config["data"]

    required_symbols = [
        str(symbol).upper().strip()
        for symbol in data_config["tickers"]
    ]

    symbol_plan: list[tuple[str, bool]] = [
        (symbol, True)
        for symbol in required_symbols
    ]

    if data_config.get("include_market_index", False):
        market_index = str(
            data_config.get("market_index", "")
        ).upper().strip()

        if market_index:
            symbol_plan.append((market_index, False))

    summary_rows: list[dict[str, Any]] = []
    required_failures: list[str] = []

    for symbol, is_required in symbol_plan:
        print(f"\nProcessing {symbol}...")

        try:
            diagnostics = fetch_symbol(
                symbol=symbol,
                data_config=data_config,
                refresh=False,
            )

            diagnostics["required"] = is_required
            summary_rows.append(diagnostics)

            print(
                f"Completed {symbol}: "
                f"{diagnostics['clean_rows']} rows, "
                f"{diagnostics['start_date']} -> "
                f"{diagnostics['end_date']}"
            )

        except Exception as error:
            summary_rows.append(
                {
                    "symbol": symbol,
                    "required": is_required,
                    "status": "failed",
                    "error": str(error),
                }
            )

            print(f"Failed {symbol}: {error}")

            if is_required:
                required_failures.append(symbol)

    summary = pd.DataFrame(summary_rows)

    summary_file = PROJECT_ROOT / "docs" / "data-download-summary.csv"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nData download summary:")
    print(summary.fillna("").to_string(index=False))

    print(f"\nSummary saved to: {summary_file}")

    if required_failures:
        raise RuntimeError(
            "Required symbols failed: "
            + ", ".join(required_failures)
        )


if __name__ == "__main__":
    main()