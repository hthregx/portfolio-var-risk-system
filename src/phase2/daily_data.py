"""Phase 2 EOD data ingestion with strict quality checks."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .index_quality import (
    ISSUE_COLUMNS,
    filter_reviewed_vnindex,
    reconcile_index_audit,
)

import numpy as np
import pandas as pd


TZ = ZoneInfo("Asia/Ho_Chi_Minh")
OHLCV = ("open", "high", "low", "close", "volume")
STOCKS = ("HPG", "FPT", "MWG")
SYMBOLS = (*STOCKS, "VNINDEX")


def eligible_session(now: datetime | None = None) -> date:
    """Return the latest potentially completed weekday session.

    Weekends are handled; market holidays require separate validation.
    """
    current = now or datetime.now(TZ)

    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must be timezone-aware")

    local = current.astimezone(TZ)

    session = (
        local.date()
        if local.hour >= 17
        else local.date() - timedelta(days=1)
    )

    while session.weekday() >= 5:
        session -= timedelta(days=1)

    return session


def normalize_ohlcv(
    raw: pd.DataFrame,
    symbol: str,
    cutoff: date,
) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError(f"{symbol}: empty provider response")

    frame = raw.copy()

    frame.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in frame.columns
    ]

    if "date" not in frame and "time" in frame:
        frame.rename(columns={"time": "date"}, inplace=True)

    missing = set(("date", *OHLCV)) - set(frame.columns)

    if missing:
        raise ValueError(
            f"{symbol}: missing columns {sorted(missing)}"
        )

    frame = frame.loc[:, ["date", *OHLCV]].copy()

    parsed = pd.to_datetime(frame["date"], errors="coerce")

    if parsed.isna().any():
        raise ValueError(f"{symbol}: invalid date")

    frame["date"] = parsed.dt.strftime("%Y-%m-%d")

    # Never use an unfinished or future trading session.
    frame = frame.loc[
        frame["date"] <= cutoff.isoformat()
    ].copy()

    if frame.empty:
        raise ValueError(
            f"{symbol}: no completed session in response"
        )

    if frame["date"].duplicated().any():
        raise ValueError(f"{symbol}: duplicate trading dates")

    for column in OHLCV:
        frame[column] = pd.to_numeric(
            frame[column], errors="coerce"
        )

    if not np.isfinite(
        frame[list(OHLCV)].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{symbol}: NaN or infinite OHLCV values"
        )

    if (
        frame[["open", "high", "low", "close"]] <= 0
    ).any().any():
        raise ValueError(f"{symbol}: nonpositive price")

    if (frame["volume"] < 0).any():
        raise ValueError(f"{symbol}: negative volume")

    invalid_high = (
        frame["high"]
        < frame[["open", "low", "close"]].max(axis=1)
    )

    invalid_low = (
        frame["low"]
        > frame[["open", "high", "close"]].min(axis=1)
    )

    if invalid_high.any() or invalid_low.any():
        raise ValueError(
            f"{symbol}: invalid OHLC relationships"
        )

    return (
        frame.sort_values("date")
        .reset_index(drop=True)
    )


def merge_history(
    existing: pd.DataFrame | None,
    incoming: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    if existing is None or existing.empty:
        return incoming.reset_index(drop=True)

    overlap = existing.merge(
        incoming, on="date", suffixes=("_old", "_new")
    )

    # Do not silently overwrite revised historical prices.
    for field in OHLCV:
        old = overlap[f"{field}_old"].to_numpy(dtype=float)
        new = overlap[f"{field}_new"].to_numpy(dtype=float)

        if not np.allclose(
            old, new, atol=1e-8, rtol=1e-8
        ):
            raise ValueError(
                f"{symbol}: provider revised historical "
                f"{field}; manual review required"
            )

    new_rows = incoming.loc[
        ~incoming["date"].isin(existing["date"])
    ]

    merged = pd.concat(
        [existing, new_rows],
        ignore_index=True,
    )

    return (
        merged.sort_values("date")
        .reset_index(drop=True)
    )


def build_portfolio(
    stocks: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    for ticker in STOCKS:
        if ticker not in stocks:
            raise ValueError(
                f"missing portfolio ticker {ticker}"
            )

    calendars = {
        ticker: set(stocks[ticker]["date"])
        for ticker in STOCKS
    }

    if len({
        frozenset(value)
        for value in calendars.values()
    }) != 1:
        raise ValueError(
            "stock trading dates are not aligned; "
            "do not forward-fill"
        )

    result = pd.DataFrame({
        "date": stocks[STOCKS[0]]["date"]
    })

    for ticker in STOCKS:
        close = stocks[ticker].set_index("date")["close"]
        result[ticker] = (
            result["date"].map(close).astype(float)
        )

    returns = result[list(STOCKS)].pct_change(
        fill_method=None
    )

    # Phase 1 contract: equal-weight simple returns.
    result["portfolio_simple_return"] = (
        returns.mean(axis=1)
    )

    result = result.dropna(
        subset=["portfolio_simple_return"]
    ).copy()

    if (result["portfolio_simple_return"] <= -1).any():
        raise ValueError("portfolio return <= -1")

    result["portfolio_log_return"] = np.log1p(
        result["portfolio_simple_return"]
    )

    return result[
        [
            "date",
            "portfolio_simple_return",
            "portfolio_log_return",
        ]
    ].reset_index(drop=True)


def fetch_kbs(
    symbol: str,
    start: date,
    cutoff: date,
) -> pd.DataFrame:
    """Fetch KBS-native OHLCV without importing Vnstock."""
    from .kbs_http import fetch_daily_ohlcv

    return fetch_daily_ohlcv(
        symbol=symbol,
        start=start,
        cutoff=cutoff,
    )



def atomic_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def run_daily(
    root: Path,
    now: datetime | None = None,
    fetcher=fetch_kbs,
) -> dict:
    root = Path(root)
    cutoff = eligible_session(now)

    base = root / "data" / "phase2"
    market_dir = base / "market"
    snapshot_dir = base / "raw_snapshots"

    histories = {}
    raw_frames = {}
    report = {}
    index_issues = pd.DataFrame(columns=ISSUE_COLUMNS)

    for symbol in SYMBOLS:
        path = market_dir / f"{symbol}.csv"

        existing = (
            pd.read_csv(path, dtype={"date": str})
            if path.exists()
            else None
        )

        if existing is not None and not existing.empty:
            previous_last = date.fromisoformat(
                existing["date"].max()
            )
            start = previous_last - timedelta(days=21)
        else:
            start = date(2020, 1, 1)

        response = fetcher(symbol, start, cutoff)

        candidate = response

        if symbol == "VNINDEX":
            candidate, index_issues = filter_reviewed_vnindex(
                response, cutoff
            )

        incoming = normalize_ohlcv(
            candidate, symbol, cutoff
        )

        # KBS quotes stocks in VND; Phase 1 uses
        # thousand-VND stock prices. Index stays in points.
        unit = response.attrs.get("price_unit")

        if unit == "vnd":
            if symbol not in STOCKS:
                raise ValueError(
                    f"{symbol}: unexpected VND price unit"
                )

            price_cols = [
                "open", "high", "low", "close"
            ]

            incoming = incoming.astype({
                column: "float64"
                for column in price_cols
            })

            incoming[price_cols] = (
                incoming[price_cols] / 1000.0
            )

        elif unit not in (None, "thousand_vnd", "index_points"):
            raise ValueError(
                f"{symbol}: unsupported price unit {unit}"
            )

        # Conservative freshness guard.
        if incoming["date"].max() < (
            cutoff - timedelta(days=4)
        ).isoformat():
            raise ValueError(
                f"{symbol}: latest data stale"
            )

        if existing is not None and not existing.empty:
            existing = normalize_ohlcv(
                existing, symbol, cutoff
            )

        combined = merge_history(
            existing, incoming, symbol
        )

        histories[symbol] = combined
        raw_frames[symbol] = response

        extreme_moves = int(
            (
                combined["close"]
                .pct_change(fill_method=None)
                .abs() > 0.15
            ).sum()
        )

        report[symbol] = {
            "rows": len(combined),
            "last_date": combined["date"].max(),
            "extreme_moves_flagged": extreme_moves,
        }

    audit_path = (
        base / "quality" / "vnindex_invalid_ohlc.csv"
    )

    previous_audit = (
        pd.read_csv(audit_path)
        if audit_path.exists()
        else pd.DataFrame(columns=ISSUE_COLUMNS)
    )

    audit = reconcile_index_audit(
        previous_audit, index_issues
    )

    report["VNINDEX"]["quality_status"] = (
        "REVIEW" if not audit.empty else "PASS"
    )

    report["VNINDEX"]["index_features_eligible"] = (
        audit.empty
    )

    report["VNINDEX"]["excluded_known_ohlc_dates"] = (
        audit["date"].tolist()
    )

    latest_stock_dates = {
        report[symbol]["last_date"]
        for symbol in STOCKS
    }

    if len(latest_stock_dates) != 1:
        raise ValueError(
            "Required tickers have different latest sessions"
        )

    # Fail closed if KBS has not published the expected session.
    # Never publish yesterday's data as today's successful EOD.
    expected_date = cutoff.isoformat()
    latest_stock_date = next(iter(latest_stock_dates))

    if latest_stock_date != expected_date:
        raise ValueError(
            f"Required tickers stale: latest={latest_stock_date}, "
            f"expected={expected_date}"
        )

    if report["VNINDEX"]["last_date"] != expected_date:
        raise ValueError(
            "VNINDEX data stale: "
            f"latest={report['VNINDEX']['last_date']}, "
            f"expected={expected_date}"
        )

    portfolio = build_portfolio({
        symbol: histories[symbol]
        for symbol in STOCKS
    })

    # All four symbols are validated before publishing.
    stamp = datetime.now(TZ).strftime(
        "%Y%m%dT%H%M%S%f%z"
    )

    snapshot = snapshot_dir / stamp
    snapshot.mkdir(parents=True, exist_ok=False)

    for symbol, frame in raw_frames.items():
        frame.to_csv(
            snapshot / f"{symbol}.csv",
            index=False,
        )

    for symbol, frame in histories.items():
        atomic_csv(
            frame,
            market_dir / f"{symbol}.csv",
        )

    if not audit.empty:
        if not audit_path.exists() or not index_issues.empty:
            atomic_csv(audit, audit_path)

    portfolio_path = base / "portfolio_returns.csv"
    atomic_csv(portfolio, portfolio_path)

    digest = hashlib.sha256(
        portfolio_path.read_bytes()
    ).hexdigest()

    manifest = {
        "cutoff": cutoff.isoformat(),
        "portfolio_last_date": portfolio["date"].max(),
        "source": "KBS",
        "rows": len(portfolio),
        "portfolio_sha256": digest,
        "symbols": report,
        "raw_snapshot": str(
            snapshot.relative_to(root)
        ),
        "generated_at": datetime.now(TZ).isoformat(),
    }

    destination = base / "latest_manifest.json"
    temporary = destination.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    temporary.replace(destination)

    return manifest