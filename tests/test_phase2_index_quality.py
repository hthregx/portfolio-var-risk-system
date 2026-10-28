from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.phase2.daily_data import run_daily
from src.phase2.index_quality import (
    filter_reviewed_vnindex,
    reconcile_index_audit,
)


def index_rows():
    return pd.DataFrame([
        {
            "date": "2021-02-17 07:00",
            "open": 1127.46, "high": 1151.54,
            "low": 1127.46, "close": 1155.78,
            "volume": 1000000,
        },
        {
            "date": "2021-12-10 07:00",
            "open": 1469.83, "high": 1475.85,
            "low": 1469.68, "close": 1463.54,
            "volume": 1000000,
        },
        {
            "date": "2023-05-15 07:00",
            "open": 1074.82, "high": 1076.32,
            "low": 1067.15, "close": 1065.71,
            "volume": 1000000,
        },
        {
            "date": "2026-10-27 07:00",
            "open": 1760.0, "high": 1765.0,
            "low": 1755.0, "close": 1761.0,
            "volume": 1000000,
        },
    ])


def test_exactly_three_known_anomalies_are_excluded():
    raw = index_rows()
    clean, issues = filter_reviewed_vnindex(
        raw, date(2026, 10, 27)
    )

    assert len(clean) == 1
    assert len(issues) == 3
    assert clean["close"].iloc[0] == 1761.0
    assert len(raw) == 4

    assert set(issues["date"]) == {
        "2021-02-17",
        "2021-12-10",
        "2023-05-15",
    }


def test_new_ohlc_anomaly_fails_closed():
    raw = index_rows()
    extra = raw.tail(1).copy()
    extra["date"] = "2026-10-26 07:00"
    extra["high"] = 1750.0

    raw = pd.concat([raw, extra], ignore_index=True)

    with pytest.raises(ValueError, match="unreviewed"):
        filter_reviewed_vnindex(
            raw, date(2026, 10, 27)
        )


def test_revised_audit_record_is_rejected():
    _, issues = filter_reviewed_vnindex(
        index_rows(), date(2026, 10, 27)
    )

    changed = issues.copy()
    changed.loc[0, "close"] += 1.0

    with pytest.raises(ValueError, match="changed"):
        reconcile_index_audit(issues, changed)


def test_daily_run_preserves_raw_and_audits_index(tmp_path):
    days = ["2026-10-26", "2026-10-27"]

    def fake_fetcher(symbol, start, cutoff):
        value = 1760 if symbol == "VNINDEX" else 21000

        frame = pd.DataFrame({
            "date": days,
            "open": [value, value],
            "high": [value + 2, value + 2],
            "low": [value - 2, value - 2],
            "close": [value, value],
            "volume": [10000, 11000],
        })

        if symbol == "VNINDEX":
            frame = pd.concat(
                [frame, index_rows().head(1)],
                ignore_index=True,
            )

        frame.attrs["price_unit"] = (
            "index_points"
            if symbol == "VNINDEX"
            else "vnd"
        )

        return frame

    now = datetime(
        2026, 10, 28, 10,
        tzinfo=timezone(timedelta(hours=7)),
    )

    result = run_daily(tmp_path, now, fake_fetcher)

    audit = pd.read_csv(
        tmp_path
        / "data/phase2/quality/vnindex_invalid_ohlc.csv"
    )

    index = pd.read_csv(
        tmp_path / "data/phase2/market/VNINDEX.csv"
    )

    original = pd.read_csv(
        tmp_path
        / result["raw_snapshot"]
        / "VNINDEX.csv"
    )

    assert len(audit) == 1
    assert len(index) == 2
    assert len(original) == 3

    assert result["symbols"]["VNINDEX"][
        "quality_status"
    ] == "REVIEW"

    assert result["symbols"]["VNINDEX"][
        "index_features_eligible"
    ] is False

    assert result["rows"] == 1