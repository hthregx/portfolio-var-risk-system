"""Safety tests for unattended daily EOD execution."""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.phase2.daily_data import run_daily
from src.phase2.eod_automation import run_scheduled_eod


VN = timezone(timedelta(hours=7))


def clock(day):
    return datetime(2026, 10, day, 18, 0, tzinfo=VN)


def configure(tmp_path, closed_dates=None):
    directory = tmp_path / "configs"
    directory.mkdir(exist_ok=True)

    (directory / "phase2_market_calendar.json").write_text(
        json.dumps({
            "timezone": "Asia/Ho_Chi_Minh",
            "closed_dates": closed_dates or [],
        }),
        encoding="utf-8",
    )


def fake_fetch(symbol, start, cutoff):
    price = 1760.0 if symbol == "VNINDEX" else {
        "HPG": 21000.0,
        "FPT": 56000.0,
        "MWG": 75000.0,
    }[symbol]

    frame = pd.DataFrame([
        {
            "date": day,
            "open": price + i,
            "high": price + i + 2,
            "low": price + i - 2,
            "close": price + i + 0.5,
            "volume": 100000 + i,
        }
        for i, day in enumerate([
            "2026-10-26",
            "2026-10-27",
            "2026-10-28",
            "2026-10-29",
        ])
        if start <= pd.Timestamp(day).date() <= cutoff
    ])

    frame.attrs["price_unit"] = (
        "index_points"
        if symbol == "VNINDEX"
        else "vnd"
    )
    return frame


def stale_fetch(symbol, start, cutoff):
    return fake_fetch(
        symbol,
        start,
        min(cutoff, pd.Timestamp("2026-10-28").date()),
    )


def read_logs(tmp_path):
    path = tmp_path / "data/phase2/logs/eod_runs.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_all_symbols_stale_blocks_publication(tmp_path):
    run_daily(tmp_path, clock(28), fake_fetch)

    path = tmp_path / "data/phase2/portfolio_returns.csv"
    original = path.read_bytes()

    with pytest.raises(ValueError, match="stale"):
        run_daily(tmp_path, clock(29), stale_fetch)

    assert path.read_bytes() == original


def test_missing_latest_index_blocks_publication(tmp_path):
    def index_lags(symbol, start, cutoff):
        if symbol == "VNINDEX":
            cutoff -= timedelta(days=1)
        return fake_fetch(symbol, start, cutoff)

    with pytest.raises(ValueError, match="VNINDEX data stale"):
        run_daily(tmp_path, clock(28), index_lags)

    assert not (
        tmp_path / "data/phase2/portfolio_returns.csv"
    ).exists()


def test_configured_holiday_skips_without_network(tmp_path):
    configure(tmp_path, ["2026-10-28"])

    def must_not_fetch(*args):
        raise AssertionError("Network must not be called")

    result = run_scheduled_eod(
        tmp_path, now=clock(28), fetcher=must_not_fetch
    )

    assert result["status"] == "SKIPPED"
    assert read_logs(tmp_path)[0]["reason"] == (
        "configured_market_closure"
    )


def test_runner_logs_stale_failure(tmp_path):
    configure(tmp_path)

    with pytest.raises(ValueError, match="stale"):
        run_scheduled_eod(
            tmp_path, now=clock(29), fetcher=stale_fetch
        )

    assert read_logs(tmp_path)[0]["status"] == "FAILED"

    assert not (
        tmp_path / "data/phase2/portfolio_returns.csv"
    ).exists()


def test_runner_logs_success(tmp_path):
    configure(tmp_path)

    event = run_scheduled_eod(
        tmp_path, now=clock(28), fetcher=fake_fetch
    )

    assert event["status"] == "SUCCESS"
    assert event["cutoff"] == "2026-10-28"
    assert read_logs(tmp_path)[0]["status"] == "SUCCESS"

    assert (
        tmp_path / "data/phase2/latest_manifest.json"
    ).exists()
