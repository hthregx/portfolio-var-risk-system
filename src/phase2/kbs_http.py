"""Read KBS daily OHLCV through its public HTTP endpoint.

This adapter avoids the Vnstock native DLL dependency.
The upstream endpoint and schema must be monitored for changes.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import date, timedelta

import pandas as pd


BASE_URL = (
    "https://kbbuddywts.kbsec.com.vn"
    "/iis-server/investment"
)

STOCKS = {"HPG", "FPT", "MWG"}
SUPPORTED = STOCKS | {"VNINDEX"}

FIELD_MAP = {
    "t": "date",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "va": "value",
}


def fetch_daily_ohlcv(
    symbol: str,
    start: date,
    cutoff: date,
    *,
    timeout: int = 40,
) -> pd.DataFrame:
    """Return KBS-native prices; stocks are in VND."""

    if symbol not in SUPPORTED:
        raise ValueError(f"Unsupported symbol: {symbol}")

    if not isinstance(start, date) or not isinstance(cutoff, date):
        raise TypeError("start and cutoff must be dates")

    if start > cutoff:
        raise ValueError("start must not exceed cutoff")

    category = (
        "index" if symbol == "VNINDEX" else "stocks"
    )

    endpoint = (
        f"{BASE_URL}/{category}/{symbol}/data_day"
    )

    params = urllib.parse.urlencode({
        "sdate": start.strftime("%d-%m-%Y"),
        "edate": (
            cutoff + timedelta(days=1)
        ).strftime("%d-%m-%Y"),
    })

    request = urllib.request.Request(
        f"{endpoint}?{params}",
        headers={
            "Accept": "application/json",
            "User-Agent": (
                "portfolio-var-risk-system-phase2/0.1"
            ),
        },
        method="GET",
    )

    with urllib.request.urlopen(
        request,
        timeout=timeout,
    ) as response:
        payload = json.loads(response.read())

    if not isinstance(payload, dict):
        raise ValueError(
            f"{symbol}: invalid KBS response structure"
        )

    rows = payload.get("data_day")

    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"{symbol}: missing or empty data_day"
        )

    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(
            f"{symbol}: invalid candle structure"
        )

    frame = pd.DataFrame(rows).rename(
        columns=FIELD_MAP
    )

    required = {
        "date", "open", "high",
        "low", "close", "volume",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"{symbol}: missing OHLCV fields "
            f"{sorted(missing)}"
        )

    # Preserve provider-native price units here.
    # Conversion happens after quality validation.
    frame.attrs["source"] = "KBS_HTTP"
    frame.attrs["price_unit"] = (
        "index_points"
        if symbol == "VNINDEX"
        else "vnd"
    )

    return frame
