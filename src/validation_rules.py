import pandas as pd


REQUIRED_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def normalize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to lowercase and replace separators
    with underscores.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    normalized_data = data.copy()

    new_columns = []

    for column in normalized_data.columns:
        normalized_column = str(column).strip().lower()
        normalized_column = normalized_column.replace(" ", "_")
        normalized_column = normalized_column.replace("-", "_")

        new_columns.append(normalized_column)

    normalized_data.columns = new_columns

    if (
        "date" not in normalized_data.columns
        and "time" in normalized_data.columns
    ):
        normalized_data = normalized_data.rename(
            columns={"time": "date"}
        )

    return normalized_data


def validate_required_columns(data: pd.DataFrame) -> None:
    """
    Validate that the DataFrame contains all required columns.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    missing_columns = []

    for required_column in REQUIRED_COLUMNS:
        if required_column not in data.columns:
            missing_columns.append(required_column)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def parse_trading_dates(date_series: pd.Series) -> pd.Series:
    """
    Convert date values to normalized Vietnamese trading dates.

    Invalid date values are converted to NaT.
    """
    if not isinstance(date_series, pd.Series):
        raise TypeError("Input data must be a pandas Series.")

    parsed_dates = pd.to_datetime(
    date_series,
    errors="coerce",
    format="mixed",
    utc=True,
)

    vietnam_dates = (
        parsed_dates
        .dt.tz_convert("Asia/Ho_Chi_Minh")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    return vietnam_dates


def evaluate_structural_rules(
    data: pd.DataFrame,
) -> dict[str, int | bool]:
    """
    Evaluate structural problems without removing any rows.

    The function checks:
    - Invalid or missing values.
    - Duplicate trading dates.
    - Non-positive prices.
    - Negative volume.
    - Dates that are not sorted in ascending order.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Input data is empty.")

    prepared_data = normalize_column_names(data)

    validate_required_columns(prepared_data)

    prepared_data = prepared_data[
        REQUIRED_COLUMNS
    ].copy()

    prepared_data["date"] = parse_trading_dates(
        prepared_data["date"]
    )

    for column in NUMERIC_COLUMNS:
        prepared_data[column] = pd.to_numeric(
            prepared_data[column],
            errors="coerce",
        )

    date_parse_failures = int(
        prepared_data["date"].isna().sum()
    )

    missing_rows = int(
        prepared_data[REQUIRED_COLUMNS]
        .isna()
        .any(axis=1)
        .sum()
    )

    valid_dates = prepared_data.loc[
        prepared_data["date"].notna(),
        "date",
    ]

    duplicate_date_rows = int(
        valid_dates
        .duplicated(keep=False)
        .sum()
    )

    nonpositive_price_rows = int(
        prepared_data[PRICE_COLUMNS]
        .le(0)
        .any(axis=1)
        .sum()
    )

    negative_volume_rows = int(
        prepared_data["volume"]
        .lt(0)
        .sum()
    )

    dates_out_of_order = bool(
        not valid_dates.is_monotonic_increasing
    )

    diagnostics = {
        "date_parse_failures": date_parse_failures,
        "missing_rows": missing_rows,
        "duplicate_date_rows": duplicate_date_rows,
        "nonpositive_price_rows": nonpositive_price_rows,
        "negative_volume_rows": negative_volume_rows,
        "dates_out_of_order": dates_out_of_order,
    }

    return diagnostics


def clean_structural_issues(
    data: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """
    Clean mandatory structural problems in OHLCV data.

    Cleaning rules:
    - Remove rows containing missing mandatory values.
    - Remove rows containing non-positive prices.
    - Remove rows containing negative volume.
    - Sort observations by date.
    - Keep the last observation for duplicated dates.
    - Add the ticker column.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Input data is empty.")

    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("Ticker must be a non-empty string.")

    normalized_ticker = ticker.strip().upper()

    prepared_data = normalize_column_names(data)

    validate_required_columns(prepared_data)

    prepared_data = prepared_data[
        REQUIRED_COLUMNS
    ].copy()

    prepared_data["date"] = parse_trading_dates(
        prepared_data["date"]
    )

    for column in NUMERIC_COLUMNS:
        prepared_data[column] = pd.to_numeric(
            prepared_data[column],
            errors="coerce",
        )

    complete_rows = (
        prepared_data[REQUIRED_COLUMNS]
        .notna()
        .all(axis=1)
    )

    positive_prices = (
        prepared_data[PRICE_COLUMNS]
        .gt(0)
        .all(axis=1)
    )

    nonnegative_volume = (
        prepared_data["volume"] >= 0
    )

    valid_rows = (
        complete_rows
        & positive_prices
        & nonnegative_volume
    )

    cleaned_data = prepared_data.loc[
        valid_rows,
        REQUIRED_COLUMNS,
    ].copy()

    cleaned_data = (
        cleaned_data
        .sort_values(
            by="date",
            kind="mergesort",
        )
        .drop_duplicates(
            subset="date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    cleaned_data.insert(
        loc=1,
        column="ticker",
        value=normalized_ticker,
    )

    return cleaned_data


if __name__ == "__main__":
    sample_data = pd.DataFrame(
        {
            " Time ": [
                "2026-07-28",
                "2026-07-29",
                "2026-07-29",
                "invalid-date",
            ],
            "Open": [
                25.0,
                25.5,
                25.6,
                26.0,
            ],
            "High": [
                25.8,
                26.0,
                26.1,
                26.5,
            ],
            "Low": [
                24.8,
                25.2,
                25.3,
                25.8,
            ],
            "Close": [
                25.5,
                25.7,
                25.8,
                26.2,
            ],
            "Volume": [
                1_000_000,
                1_200_000,
                1_300_000,
                1_100_000,
            ],
        }
    )

    normalized_data = normalize_column_names(
        sample_data
    )

    print("Original columns:")
    print(sample_data.columns.tolist())

    print("\nNormalized columns:")
    print(normalized_data.columns.tolist())

    print("\nChecking complete data:")

    validate_required_columns(normalized_data)

    print("All required columns are available.")

    incomplete_data = normalized_data.drop(
        columns=["volume"]
    )

    print("\nChecking incomplete data:")

    try:
        validate_required_columns(incomplete_data)
    except ValueError as error:
        print(error)

    sample_dates = pd.Series(
        [
            "2026-07-28",
            "2026-07-29 07:00:00+07:00",
            "invalid-date",
            None,
        ]
    )

    parsed_dates = parse_trading_dates(
        sample_dates
    )

    print("\nParsed trading dates:")
    print(parsed_dates)

    print("\nNumber of invalid dates:")
    print(parsed_dates.isna().sum())

    diagnostics = evaluate_structural_rules(
        sample_data
    )

    print("\nStructural diagnostics:")

    for rule_name, result in diagnostics.items():
        print(f"{rule_name}: {result}")

    cleaned_data = clean_structural_issues(
        data=sample_data,
        ticker="HPG",
    )

    print("\nCleaned data:")
    print(cleaned_data)

    print("\nOriginal shape:", sample_data.shape)
    print("Cleaned shape:", cleaned_data.shape)