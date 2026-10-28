"""Fail-closed handling of three known VNINDEX OHLC anomalies."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


KNOWN_GAPS = {
    "2021-02-17": 4.24,
    "2021-12-10": 6.14,
    "2023-05-15": 1.44,
}

ISSUE_COLUMNS = [
    "date", "open", "high", "low", "close",
    "high_gap", "low_gap", "max_gap_points",
]


def filter_reviewed_vnindex(
    raw: pd.DataFrame,
    cutoff: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude only reviewed anomalies from derived index data."""

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("VNINDEX: empty provider response")

    data = raw.copy()
    data.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in data.columns
    ]

    if "date" not in data and "time" in data:
        data = data.rename(columns={"time": "date"})

    needed = {"date", "open", "high", "low", "close"}
    if not needed.issubset(data.columns):
        raise ValueError("VNINDEX: missing date or OHLC columns")

    parsed = pd.to_datetime(data["date"], format="mixed", errors="coerce")
    if parsed.isna().any():
        raise ValueError("VNINDEX: invalid trading dates")

    prices = data[
        ["open", "high", "low", "close"]
    ].apply(pd.to_numeric, errors="coerce")

    if (
        prices.isna().any().any()
        or not np.isfinite(prices.to_numpy(dtype=float)).all()
    ):
        raise ValueError("VNINDEX: invalid price values")

    high_gap = (
        prices[["open", "low", "close"]].max(axis=1)
        - prices["high"]
    ).clip(lower=0.0)

    low_gap = (
        prices["low"]
        - prices[["open", "high", "close"]].min(axis=1)
    ).clip(lower=0.0)

    invalid = (
        (high_gap.gt(1e-8) | low_gap.gt(1e-8))
        & parsed.dt.date.le(cutoff)
    )

    issues = []

    for idx in data.index[invalid]:
        day = parsed.at[idx].strftime("%Y-%m-%d")
        gap = max(
            float(high_gap.at[idx]),
            float(low_gap.at[idx]),
        )

        expected = KNOWN_GAPS.get(day)

        if expected is None or abs(gap - expected) > 0.02:
            raise ValueError(
                f"VNINDEX: unreviewed OHLC anomaly on {day} "
                f"(gap={gap:.4f} points); manual review required"
            )

        issues.append({
            "date": day,
            **{
                field: float(prices.at[idx, field])
                for field in ("open", "high", "low", "close")
            },
            "high_gap": float(high_gap.at[idx]),
            "low_gap": float(low_gap.at[idx]),
            "max_gap_points": gap,
        })

    clean = data.loc[~invalid].copy()

    return clean, pd.DataFrame(
        issues, columns=ISSUE_COLUMNS
    )


def reconcile_index_audit(
    old: pd.DataFrame,
    current: pd.DataFrame,
) -> pd.DataFrame:
    """Keep historical audit records; reject changed anomalies."""

    nonempty = [
        frame
        for frame in (old, current)
        if not frame.empty
    ]

    rows = (
        pd.concat(nonempty, ignore_index=True)
        if nonempty
        else pd.DataFrame(columns=ISSUE_COLUMNS)
    )

    if rows.empty:
        return pd.DataFrame(columns=ISSUE_COLUMNS)

    if not set(ISSUE_COLUMNS).issubset(rows.columns):
        raise ValueError("VNINDEX: audit schema changed")

    rows = rows[ISSUE_COLUMNS].copy()
    rows["date"] = rows["date"].astype(str)

    for _, group in rows.groupby("date", sort=False):
        if len(group) > 1:
            values = group[
                ISSUE_COLUMNS[1:]
            ].to_numpy(dtype=float)

            if not np.allclose(
                values, values[0],
                atol=1e-7, rtol=0,
            ):
                raise ValueError(
                    "VNINDEX: reviewed historical candle changed; "
                    "manual review required"
                )

    return (
        rows.drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )