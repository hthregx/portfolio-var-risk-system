import json
from datetime import date, datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from src.phase2.daily_data import (
    normalize_ohlcv,
    run_daily,
)
from src.phase2.kbs_http import (
    fetch_daily_ohlcv,
)


def candles():
    return [
        {
            "t": "2026-10-28 07:00",
            "o": 21100,
            "h": 21150,
            "l": 20950,
            "c": 21000,
            "v": 2831900,
        },
        {
            "t": "2026-10-27 07:00",
            "o": 20500,
            "h": 21000,
            "l": 20400,
            "c": 20900,
            "v": 4000000,
        },
    ]


class FakeHTTPResponse:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(
            {"symbol": "HPG", "data_day": self.data}
        ).encode("utf-8")


def test_http_adapter_and_provider_dates(monkeypatch):
    seen = []

    def fake_urlopen(request, timeout):
        seen.append(request.full_url)
        return FakeHTTPResponse(candles())

    monkeypatch.setattr(
        "urllib.request.urlopen",
        fake_urlopen,
    )

    raw = fetch_daily_ohlcv(
        "HPG",
        date(2026, 10, 6),
        date(2026, 10, 27),
    )

    assert raw.attrs["price_unit"] == "vnd"
    assert raw.loc[0, "close"] == 21000

    url = urlparse(seen[0])
    params = parse_qs(url.query)

    assert "/stocks/HPG/data_day" in url.path
    assert params["sdate"] == ["06-10-2026"]
    assert params["edate"] == ["28-10-2026"]

    eligible = normalize_ohlcv(
        raw,
        "HPG",
        date(2026, 10, 27),
    )

    assert eligible["date"].tolist() == [
        "2026-10-27"
    ]


def test_index_uses_index_endpoint(monkeypatch):
    urls = []

    def fake_urlopen(request, timeout):
        urls.append(request.full_url)
        return FakeHTTPResponse(candles())

    monkeypatch.setattr(
        "urllib.request.urlopen",
        fake_urlopen,
    )

    raw = fetch_daily_ohlcv(
        "VNINDEX",
        date(2026, 10, 6),
        date(2026, 10, 27),
    )

    assert "/index/VNINDEX/data_day" in urls[0]
    assert raw.attrs["price_unit"] == "index_points"


def test_unsupported_symbol_never_calls_network():
    with pytest.raises(ValueError, match="Unsupported"):
        fetch_daily_ohlcv(
            "UNKNOWN",
            date(2026, 10, 6),
            date(2026, 10, 27),
        )


def test_daily_pipeline_converts_stock_units(tmp_path):
    dates = ["2026-10-26", "2026-10-27"]

    def fake_fetcher(symbol, start, cutoff):
        if symbol == "VNINDEX":
            value = 1760
            unit = "index_points"
        else:
            value = 21000
            unit = "vnd"

        frame = pd.DataFrame({
            "date": dates,
            "open": [value, value],
            "high": [value + 2, value + 2],
            "low": [value - 2, value - 2],
            "close": [value, value],
            "volume": [10000, 11000],
        })

        frame.attrs["price_unit"] = unit
        return frame

    now = datetime(
        2026, 10, 28, 10,
        tzinfo=timezone(timedelta(hours=7)),
    )

    run_daily(tmp_path, now, fake_fetcher)

    hpg = pd.read_csv(
        tmp_path / "data/phase2/market/HPG.csv"
    )

    index = pd.read_csv(
        tmp_path / "data/phase2/market/VNINDEX.csv"
    )

    assert hpg["close"].tolist() == [21.0, 21.0]
    assert index["close"].tolist() == [1760, 1760]