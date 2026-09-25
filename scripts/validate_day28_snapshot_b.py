#!/usr/bin/env python3
"""Independent Day-28 fresh-snapshot validator (B28.3).

This validator is intentionally read-only and does not import A-owned project
implementation code. It independently checks the Day-28 market/portfolio
snapshot artifacts against the frozen canonical portfolio history.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_CUTOFF = pd.Timestamp("2026-08-28")
TOLERANCE = 1e-12

MARKET_SCHEMA = ["date", "ticker", "open", "high", "low", "close", "volume"]
PORTFOLIO_SCHEMA = [
    "date",
    "HPG_simple_return",
    "FPT_simple_return",
    "MWG_simple_return",
    "portfolio_simple_return",
    "portfolio_log_return",
]
ASSET_RETURN_COLUMNS = [
    "HPG_simple_return",
    "FPT_simple_return",
    "MWG_simple_return",
]
PORTFOLIO_NUMERIC_COLUMNS = PORTFOLIO_SCHEMA[1:]
MARKET_NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume"]
REQUIRED_PORTFOLIO_TICKERS = {"HPG", "FPT", "MWG"}


class ValidationError(RuntimeError):
    """Raised when an independent Day-28 validation check fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def parse_date_column(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    result = frame.copy()
    parsed = pd.to_datetime(result["date"], errors="coerce")
    require(parsed.notna().all(), f"{label}: date contains invalid/unparseable values")
    result["date"] = parsed
    return result


def require_exact_schema(frame: pd.DataFrame, expected: list[str], label: str) -> None:
    actual = list(frame.columns)
    require(actual == expected, f"{label}: schema mismatch; expected={expected}, actual={actual}")


def require_no_missing(frame: pd.DataFrame, label: str) -> None:
    missing = frame.isna().sum()
    bad = missing[missing > 0]
    require(bad.empty, f"{label}: missing values found: {bad.to_dict()}")


def require_finite(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    require(numeric.notna().all().all(), f"{label}: non-numeric values found in numeric columns")
    values = numeric.to_numpy(dtype=float)
    require(np.isfinite(values).all(), f"{label}: non-finite numeric values found")


def validate_market_snapshot(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path)
    require_exact_schema(frame, MARKET_SCHEMA, "market snapshot")
    require_no_missing(frame, "market snapshot")
    frame = parse_date_column(frame, "market snapshot")
    require_finite(frame, MARKET_NUMERIC_COLUMNS, "market snapshot")

    duplicate_count = int(frame.duplicated(subset=["date", "ticker"]).sum())
    require(duplicate_count == 0, f"market snapshot: {duplicate_count} duplicate (date,ticker) rows")

    require(frame["date"].is_monotonic_increasing, "market snapshot: dates are not chronological")
    sorted_frame = frame.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
    require(
        frame.reset_index(drop=True).equals(sorted_frame),
        "market snapshot: rows are not deterministically ordered by (date,ticker)",
    )

    tickers = set(frame["ticker"].astype(str))
    missing_tickers = sorted(REQUIRED_PORTFOLIO_TICKERS - tickers)
    require(not missing_tickers, f"market snapshot: missing portfolio tickers {missing_tickers}")

    actual_cutoff = frame["date"].max()
    require(
        actual_cutoff == EXPECTED_CUTOFF,
        f"market snapshot: cutoff {actual_cutoff.date()} != expected {EXPECTED_CUTOFF.date()}",
    )

    checks = [
        f"schema = {MARKET_SCHEMA}",
        "no duplicate (date,ticker) rows",
        "no missing values",
        "numeric fields are finite",
        "chronological (date,ticker) ordering",
        f"required portfolio tickers present: {sorted(REQUIRED_PORTFOLIO_TICKERS)}",
        f"data cutoff = {EXPECTED_CUTOFF.date()}",
    ]
    return frame, checks


def validate_portfolio_snapshot(path: Path) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    frame = pd.read_csv(path)
    require_exact_schema(frame, PORTFOLIO_SCHEMA, "portfolio snapshot")
    require_no_missing(frame, "portfolio snapshot")
    frame = parse_date_column(frame, "portfolio snapshot")
    require_finite(frame, PORTFOLIO_NUMERIC_COLUMNS, "portfolio snapshot")

    duplicate_count = int(frame["date"].duplicated().sum())
    require(duplicate_count == 0, f"portfolio snapshot: {duplicate_count} duplicate dates")
    require(frame["date"].is_monotonic_increasing, "portfolio snapshot: dates are not chronological")
    require(frame["date"].is_unique, "portfolio snapshot: dates are not unique")

    actual_cutoff = frame["date"].max()
    require(
        actual_cutoff == EXPECTED_CUTOFF,
        f"portfolio snapshot: cutoff {actual_cutoff.date()} != expected {EXPECTED_CUTOFF.date()}",
    )

    expected_portfolio = frame[ASSET_RETURN_COLUMNS].mean(axis=1)
    equal_weight_error = (expected_portfolio - frame["portfolio_simple_return"]).abs()
    max_equal_weight_error = float(equal_weight_error.max())
    require(
        max_equal_weight_error <= TOLERANCE,
        f"portfolio snapshot: equal-weight identity failed; max_abs_error={max_equal_weight_error:.3e}",
    )

    require(
        (frame["portfolio_simple_return"] > -1.0).all(),
        "portfolio snapshot: portfolio_simple_return <= -1 prevents log1p identity",
    )
    expected_log = np.log1p(frame["portfolio_simple_return"].to_numpy(dtype=float))
    log_error = np.abs(expected_log - frame["portfolio_log_return"].to_numpy(dtype=float))
    max_log_error = float(log_error.max())
    require(
        max_log_error <= TOLERANCE,
        f"portfolio snapshot: log/simple identity failed; max_abs_error={max_log_error:.3e}",
    )

    diagnostics = {
        "max_equal_weight_abs_error": max_equal_weight_error,
        "max_log_simple_abs_error": max_log_error,
    }
    checks = [
        f"schema = {PORTFOLIO_SCHEMA}",
        "no duplicate dates",
        "no missing values",
        "numeric fields are finite",
        "strict chronological ordering",
        "equal-weight identity: portfolio = mean(HPG,FPT,MWG)",
        "log/simple identity: portfolio_log_return = log1p(portfolio_simple_return)",
        f"data cutoff = {EXPECTED_CUTOFF.date()}",
    ]
    return frame, diagnostics, checks


def validate_historical_overlap(
    fresh: pd.DataFrame,
    canonical_path: Path,
) -> tuple[dict[str, object], list[str]]:
    canonical = pd.read_csv(canonical_path)
    require_exact_schema(canonical, PORTFOLIO_SCHEMA, "canonical portfolio history")
    require_no_missing(canonical, "canonical portfolio history")
    canonical = parse_date_column(canonical, "canonical portfolio history")
    require_finite(canonical, PORTFOLIO_NUMERIC_COLUMNS, "canonical portfolio history")
    require(canonical["date"].is_unique, "canonical portfolio history: duplicate dates")
    require(canonical["date"].is_monotonic_increasing, "canonical portfolio history: not chronological")

    canonical_cutoff = canonical["date"].max()
    fresh_historical = fresh.loc[fresh["date"] <= canonical_cutoff].reset_index(drop=True)
    canonical = canonical.reset_index(drop=True)

    require(
        len(fresh_historical) == len(canonical),
        "historical overlap: fresh prefix row count differs from canonical history "
        f"({len(fresh_historical)} vs {len(canonical)})",
    )
    require(
        fresh_historical["date"].equals(canonical["date"]),
        "historical overlap: date sequence through canonical cutoff differs",
    )

    max_errors: dict[str, float] = {}
    for column in PORTFOLIO_NUMERIC_COLUMNS:
        left = canonical[column].to_numpy(dtype=float)
        right = fresh_historical[column].to_numpy(dtype=float)
        error = np.abs(left - right)
        max_error = float(error.max()) if len(error) else 0.0
        max_errors[column] = max_error
        require(
            max_error <= TOLERANCE,
            f"historical overlap: {column} changed; max_abs_error={max_error:.3e}",
        )

    new_rows = fresh.loc[fresh["date"] > canonical_cutoff]
    require(len(new_rows) > 0, "historical overlap: no operational rows after canonical cutoff")
    require(
        new_rows["date"].min() > canonical_cutoff,
        "historical overlap: operational extension is not strictly after canonical cutoff",
    )

    diagnostics: dict[str, object] = {
        "canonical_rows": len(canonical),
        "canonical_cutoff": canonical_cutoff.date().isoformat(),
        "fresh_rows_through_canonical_cutoff": len(fresh_historical),
        "new_operational_rows": len(new_rows),
        "max_abs_error_by_column": max_errors,
    }
    checks = [
        f"canonical history preserved through {canonical_cutoff.date()}",
        "canonical date sequence unchanged",
        f"historical numeric overlap within tolerance {TOLERANCE:g}",
        "operational extension occurs strictly after canonical cutoff",
    ]
    return diagnostics, checks


def repo_root_from_script() -> Path:
    # When installed at repo/scripts/validate_day28_snapshot_b.py.
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(description="Independent B28.3 fresh-snapshot validator")
    parser.add_argument(
        "--market",
        type=Path,
        default=root / "data/snapshots/market_data_2026-08-28.csv",
        help="A-owned fresh market snapshot (read-only)",
    )
    parser.add_argument(
        "--portfolio",
        type=Path,
        default=root / "data/snapshots/portfolio_returns_2026-08-28.csv",
        help="A-owned fresh portfolio snapshot (read-only)",
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=root / "data/processed/portfolio_returns.csv",
        help="Frozen canonical portfolio history (read-only)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path, label in [
        (args.market, "market snapshot"),
        (args.portfolio, "portfolio snapshot"),
        (args.canonical, "canonical portfolio history"),
    ]:
        require(path.is_file(), f"{label} not found: {path}")

    market, market_checks = validate_market_snapshot(args.market)
    portfolio, portfolio_diag, portfolio_checks = validate_portfolio_snapshot(args.portfolio)
    overlap_diag, overlap_checks = validate_historical_overlap(portfolio, args.canonical)

    print("PASS - B28.3 INDEPENDENT FRESH-SNAPSHOT VALIDATOR")
    print()
    print("Market snapshot")
    print(f"  rows = {len(market)}")
    print(f"  start = {market['date'].min().date()}")
    print(f"  cutoff = {market['date'].max().date()}")
    print(f"  tickers = {', '.join(sorted(market['ticker'].astype(str).unique()))}")
    for item in market_checks:
        print(f"  [PASS] {item}")

    print()
    print("Fresh portfolio snapshot")
    print(f"  rows = {len(portfolio)}")
    print(f"  start = {portfolio['date'].min().date()}")
    print(f"  cutoff = {portfolio['date'].max().date()}")
    print(
        "  max equal-weight abs error = "
        f"{portfolio_diag['max_equal_weight_abs_error']:.3e}"
    )
    print(
        "  max log/simple abs error = "
        f"{portfolio_diag['max_log_simple_abs_error']:.3e}"
    )
    for item in portfolio_checks:
        print(f"  [PASS] {item}")

    print()
    print("Historical overlap")
    print(f"  canonical rows = {overlap_diag['canonical_rows']}")
    print(f"  canonical cutoff = {overlap_diag['canonical_cutoff']}")
    print(f"  new operational rows = {overlap_diag['new_operational_rows']}")
    max_errors = overlap_diag["max_abs_error_by_column"]
    assert isinstance(max_errors, dict)
    print(f"  maximum overlap abs error = {max(max_errors.values()):.3e}")
    for item in overlap_checks:
        print(f"  [PASS] {item}")

    print()
    print("[PASS] validator is read-only; no snapshot/canonical artifact is written")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"FAIL - B28.3 INDEPENDENT FRESH-SNAPSHOT VALIDATOR: {exc}", file=sys.stderr)
        raise SystemExit(1)
