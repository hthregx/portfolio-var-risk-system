import pandas as pd
import pytest

from src.validation_rules import (
    clean_structural_issues,
    evaluate_structural_rules,
    normalize_column_names,
    parse_trading_dates,
    validate_required_columns,
)

def make_structural_test_data() -> pd.DataFrame:
    """
    Create test data containing several structural problems.
    """
    return pd.DataFrame(
        {
            "time": [
                "2026-07-29",
                "2026-07-28",
                "2026-07-28",
                "invalid-date",
            ],
            "open": [
                0.0,
                25.0,
                25.2,
                25.4,
            ],
            "high": [
                25.8,
                25.5,
                25.7,
                25.9,
            ],
            "low": [
                24.8,
                24.7,
                24.9,
                25.0,
            ],
            "close": [
                25.5,
                25.2,
                25.4,
                25.6,
            ],
            "volume": [
                1_000,
                -100,
                1_200,
                1_300,
            ],
        }
    )

def test_normalize_column_names() -> None:
    raw_data = pd.DataFrame(
        {
            " Time ": ["2026-07-28"],
            "Open": [25.0],
            "Adjusted-Close": [25.5],
        }
    )

    normalized_data = normalize_column_names(raw_data)

    assert normalized_data.columns.tolist() == [
        "date",
        "open",
        "adjusted_close",
    ]

    assert raw_data.columns.tolist() == [
        " Time ",
        "Open",
        "Adjusted-Close",
    ]


def test_missing_required_column_raises_error() -> None:
    incomplete_data = pd.DataFrame(
        {
            "date": ["2026-07-28"],
            "open": [25.0],
            "high": [25.8],
            "low": [24.8],
            "close": [25.5],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        validate_required_columns(incomplete_data)


def test_parse_trading_dates() -> None:
    date_series = pd.Series(
        [
            "2026-07-28",
            "2026-07-29 07:00:00+07:00",
            "invalid-date",
            None,
        ]
    )

    parsed_dates = parse_trading_dates(date_series)

    assert parsed_dates.iloc[0] == pd.Timestamp("2026-07-28")
    assert parsed_dates.iloc[1] == pd.Timestamp("2026-07-29")
    assert pd.isna(parsed_dates.iloc[2])
    assert pd.isna(parsed_dates.iloc[3])
    assert parsed_dates.isna().sum() == 2

def test_evaluate_structural_rules_detects_issues() -> None:
    raw_data = make_structural_test_data()

    diagnostics = evaluate_structural_rules(raw_data)

    assert diagnostics["date_parse_failures"] == 1
    assert diagnostics["missing_rows"] == 1
    assert diagnostics["duplicate_date_rows"] == 2
    assert diagnostics["nonpositive_price_rows"] == 1
    assert diagnostics["negative_volume_rows"] == 1
    assert diagnostics["dates_out_of_order"] is True

def test_clean_structural_issues_removes_invalid_rows() -> None:
    raw_data = make_structural_test_data()

    cleaned_data = clean_structural_issues(
        data=raw_data,
        ticker="HPG",
    )

    assert len(cleaned_data) == 1

    assert cleaned_data.iloc[0]["date"] == pd.Timestamp(
        "2026-07-28"
    )

    assert cleaned_data.iloc[0]["open"] == 25.2
    assert cleaned_data.iloc[0]["volume"] == 1_200

def test_clean_structural_issues_sorts_and_deduplicates() -> None:
    raw_data = pd.DataFrame(
        {
            "time": [
                "2026-07-30",
                "2026-07-28",
                "2026-07-29",
                "2026-07-29",
            ],
            "open": [
                26.0,
                24.5,
                25.0,
                25.4,
            ],
            "high": [
                26.5,
                25.0,
                25.6,
                25.9,
            ],
            "low": [
                25.7,
                24.2,
                24.8,
                25.1,
            ],
            "close": [
                26.2,
                24.8,
                25.2,
                25.7,
            ],
            "volume": [
                1_000,
                1_100,
                1_200,
                1_300,
            ],
        }
    )

    cleaned_data = clean_structural_issues(
        data=raw_data,
        ticker="hpg",
    )

    expected_dates = [
        pd.Timestamp("2026-07-28"),
        pd.Timestamp("2026-07-29"),
        pd.Timestamp("2026-07-30"),
    ]

    assert cleaned_data["date"].tolist() == expected_dates
    assert cleaned_data["date"].duplicated().sum() == 0
    assert cleaned_data["ticker"].eq("HPG").all()
    assert len(cleaned_data) == 3

    retained_duplicate = cleaned_data.loc[
        cleaned_data["date"] == pd.Timestamp("2026-07-29")
    ]

    assert retained_duplicate.iloc[0]["close"] == 25.7
    assert retained_duplicate.iloc[0]["volume"] == 1_300

    assert "ticker" not in raw_data.columns