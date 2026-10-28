"""Incremental updates must preserve published market history."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.phase2.daily_data import run_daily


VN = timezone(timedelta(hours=7))

DATES = [
    "2026-10-26",
    "2026-10-27",
    "2026-10-28",
    "2026-10-29",
]


def clock(day):
    return datetime(
        2026, 10, day, 18, 0,
        tzinfo=VN,
    )


def fake_fetcher_factory(
    *,
    revise_hpg=False,
    omit_mwg_new_date=False,
):
    def fetcher(symbol, start, cutoff):
        if symbol == "VNINDEX":
            base = 1760.0
            unit = "index_points"
        else:
            base = {
                "HPG": 21000.0,
                "FPT": 56000.0,
                "MWG": 75000.0,
            }[symbol]
            unit = "vnd"

        rows = []

        for index, day in enumerate(DATES):
            session = pd.Timestamp(day).date()

            if session < start or session > cutoff:
                continue

            if (
                omit_mwg_new_date
                and symbol == "MWG"
                and day == "2026-10-29"
            ):
                continue

            step = (
                1.0
                if symbol == "VNINDEX"
                else 100.0
            )

            opening = base + index * step

            closing = opening + (
                0.5 if symbol == "VNINDEX"
                else 50.0
            )

            if (
                revise_hpg
                and symbol == "HPG"
                and day == "2026-10-27"
            ):
                closing += 30.0

            buffer = (
                2.0 if symbol == "VNINDEX"
                else 200.0
            )

            rows.append({
                "date": day,
                "open": opening,
                "high": opening + buffer,
                "low": opening - buffer,
                "close": closing,
                "volume": 100000 + index,
            })

        frame = pd.DataFrame(rows)
        frame.attrs["price_unit"] = unit
        return frame

    return fetcher


def test_new_session_appends_without_rewriting_history(tmp_path):
    fetcher = fake_fetcher_factory()

    first = run_daily(tmp_path, clock(28), fetcher)

    market_path = (
        tmp_path / "data/phase2/market/HPG.csv"
    )

    returns_path = (
        tmp_path / "data/phase2/portfolio_returns.csv"
    )

    original_market = pd.read_csv(market_path)
    original_returns = pd.read_csv(returns_path)

    assert first["cutoff"] == "2026-10-28"
    assert len(original_market) == 3
    assert len(original_returns) == 2

    second = run_daily(tmp_path, clock(29), fetcher)

    updated_market = pd.read_csv(market_path)
    updated_returns = pd.read_csv(returns_path)

    assert second["cutoff"] == "2026-10-29"
    assert second["portfolio_last_date"] == "2026-10-29"

    assert len(updated_market) == 4
    assert len(updated_returns) == 3

    pd.testing.assert_frame_equal(
        original_market,
        updated_market.iloc[:3].reset_index(drop=True),
    )

    pd.testing.assert_frame_equal(
        original_returns,
        updated_returns.iloc[:2].reset_index(drop=True),
    )

    assert (
        updated_market["date"].duplicated().sum() == 0
    )

    assert (
        first["portfolio_sha256"]
        != second["portfolio_sha256"]
    )


def test_historical_revision_blocks_publication(tmp_path):
    run_daily(
        tmp_path,
        clock(28),
        fake_fetcher_factory(),
    )

    path = (
        tmp_path / "data/phase2/portfolio_returns.csv"
    )

    original = path.read_bytes()

    with pytest.raises(ValueError, match="revised"):
        run_daily(
            tmp_path,
            clock(29),
            fake_fetcher_factory(revise_hpg=True),
        )

    assert path.read_bytes() == original


def test_missing_new_stock_session_blocks_publication(tmp_path):
    run_daily(
        tmp_path,
        clock(28),
        fake_fetcher_factory(),
    )

    path = (
        tmp_path / "data/phase2/portfolio_returns.csv"
    )

    original = path.read_bytes()

    with pytest.raises(ValueError, match="different latest sessions"):
        run_daily(
            tmp_path,
            clock(29),
            fake_fetcher_factory(
                omit_mwg_new_date=True
            ),
        )

    assert path.read_bytes() == original
