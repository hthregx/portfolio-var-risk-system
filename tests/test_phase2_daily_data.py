from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.phase2.daily_data import (
    eligible_session,
    normalize_ohlcv,
    merge_history,
    build_portfolio,
    run_daily,
)


VN = timezone(timedelta(hours=7))


def quotes(days, scale=1):
    return pd.DataFrame({
        "date": days,
        "open": [100 * scale + i for i in range(len(days))],
        "high": [102 * scale + i for i in range(len(days))],
        "low": [99 * scale + i for i in range(len(days))],
        "close": [101 * scale + i for i in range(len(days))],
        "volume": [1000 + i for i in range(len(days))],
    })


def test_cutoff_excludes_unfinished_session():
    now = datetime(2026, 10, 28, 10, tzinfo=VN)
    assert eligible_session(now).isoformat() == "2026-10-27"

    monday = datetime(2026, 10, 26, 10, tzinfo=VN)
    assert eligible_session(monday).isoformat() == "2026-10-23"


def test_future_partial_data_is_excluded():
    raw = quotes(["2026-10-27", "2026-10-28"])

    cutoff = eligible_session(
        datetime(2026, 10, 28, 10, tzinfo=VN)
    )

    result = normalize_ohlcv(raw, "HPG", cutoff)

    assert result["date"].tolist() == ["2026-10-27"]


def test_invalid_ohlc_is_rejected():
    raw = quotes(["2026-10-27"])
    raw.loc[0, "high"] = 1

    with pytest.raises(ValueError, match="invalid OHLC"):
        normalize_ohlcv(
            raw, "HPG", datetime(2026, 10, 27).date()
        )


def test_historical_revision_is_not_silent():
    old = quotes(["2026-10-26", "2026-10-27"])
    new = old.copy()
    new["close"] = new["close"].astype(float)
    new.loc[0, "close"] += 0.5

    with pytest.raises(ValueError, match="revised"):
        merge_history(old, new, "HPG")


def test_missing_asset_date_is_rejected():
    complete = quotes(["2026-10-26", "2026-10-27"])
    incomplete = quotes(["2026-10-26"])

    with pytest.raises(ValueError, match="not aligned"):
        build_portfolio({
            "HPG": complete,
            "FPT": complete,
            "MWG": incomplete,
        })


def test_daily_update_is_idempotent(tmp_path):
    dates = ["2026-10-26", "2026-10-27"]

    scales = {
        "HPG": 1,
        "FPT": 2,
        "MWG": 3,
        "VNINDEX": 4,
    }

    def fake_fetcher(symbol, start, end):
        return quotes(dates, scale=scales[symbol])

    now = datetime(2026, 10, 28, 10, tzinfo=VN)

    first = run_daily(tmp_path, now, fake_fetcher)
    second = run_daily(tmp_path, now, fake_fetcher)

    assert first["portfolio_sha256"] == second["portfolio_sha256"]
    assert first["rows"] == second["rows"] == 1

    assert (
        tmp_path / "data/phase2/market/HPG.csv"
    ).exists()

    assert (
        tmp_path / "data/phase2/latest_manifest.json"
    ).exists()