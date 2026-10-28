"""Guarded EOD operations with persistent success/failure logs."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from .daily_data import TZ, fetch_kbs, run_daily


def read_closed_dates(root: Path) -> set[str]:
    path = root / "configs" / "phase2_market_calendar.json"

    config = json.loads(path.read_text(encoding="utf-8"))

    if config.get("timezone") != "Asia/Ho_Chi_Minh":
        raise ValueError("Unexpected market calendar timezone")

    values = config.get("closed_dates")

    if not isinstance(values, list):
        raise ValueError("closed_dates must be a list")

    dates = set()

    for item in values:
        if not isinstance(item, str):
            raise ValueError("Invalid market closure date")

        try:
            parsed = date.fromisoformat(item)
        except ValueError as error:
            raise ValueError(
                f"Invalid market closure date: {item}"
            ) from error

        if parsed.isoformat() != item:
            raise ValueError("Market dates require YYYY-MM-DD")

        dates.add(item)

    return dates


def write_event(root: Path, event: dict) -> None:
    path = root / "data" / "phase2" / "logs" / "eod_runs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as output:
        output.write(
            json.dumps(event, ensure_ascii=False) + "\n"
        )


def run_scheduled_eod(
    root: Path,
    *,
    now: datetime | None = None,
    fetcher=fetch_kbs,
) -> dict:
    root = Path(root)
    current = now or datetime.now(TZ)

    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("A timezone-aware clock is required")

    local = current.astimezone(TZ)
    day = local.date().isoformat()

    event = {
        "started_at": local.isoformat(),
        "scheduled_date": day,
        "status": None,
    }

    try:
        closed_dates = read_closed_dates(root)

        if local.weekday() >= 5:
            event.update(
                status="SKIPPED",
                reason="weekend",
            )

        elif day in closed_dates:
            event.update(
                status="SKIPPED",
                reason="configured_market_closure",
            )

        elif local.hour < 17:
            event.update(
                status="SKIPPED",
                reason="before_eod_cutoff",
            )

        else:
            result = run_daily(
                root=root,
                now=local,
                fetcher=fetcher,
            )

            event.update(
                status="SUCCESS",
                cutoff=result["cutoff"],
                portfolio_last_date=result["portfolio_last_date"],
                rows=result["rows"],
                portfolio_sha256=result["portfolio_sha256"],
            )

    except Exception as error:
        event.update(
            status="FAILED",
            error_type=type(error).__name__,
            error=str(error),
        )
        write_event(root, event)
        raise

    write_event(root, event)
    return event
