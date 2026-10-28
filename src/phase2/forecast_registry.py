"""Append-only forecast records for Phase 2.

Forecast rows cannot be updated or deleted through the registry API.
Actual observations will be stored separately in a later step.
"""

from __future__ import annotations

import math
import re
import sqlite3
import uuid
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


VIETNAM_TZ = timezone(timedelta(hours=7))

COLUMNS = (
    "forecast_id",
    "model_name",
    "model_version",
    "generated_at",
    "data_as_of_date",
    "target_date",
    "quantile_alpha",
    "quantile_return",
    "var_return",
    "training_start",
    "training_end",
    "feature_version",
    "config_hash",
    "data_hash",
    "git_commit",
)


def _validate_date(value: str, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date string.")

    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Invalid {field}: {value}") from error


def _validate_hash(value: str, field: str, length: int) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a hexadecimal hash.")

    if not re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value):
        raise ValueError(f"{field} must have {length} hex characters.")


def _initialize(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS forecasts (
            forecast_id TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            data_as_of_date TEXT NOT NULL,
            target_date TEXT NOT NULL,
            quantile_alpha REAL NOT NULL,
            quantile_return REAL NOT NULL,
            var_return REAL NOT NULL CHECK (var_return >= 0),
            training_start TEXT NOT NULL,
            training_end TEXT NOT NULL,
            feature_version TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            data_hash TEXT NOT NULL,
            git_commit TEXT NOT NULL,
            UNIQUE (model_name, model_version, target_date)
        )
        """
    )

    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS forecasts_no_update
        BEFORE UPDATE ON forecasts
        BEGIN
            SELECT RAISE(ABORT, 'forecast rows are immutable');
        END
        """
    )

    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS forecasts_no_delete
        BEFORE DELETE ON forecasts
        BEGIN
            SELECT RAISE(ABORT, 'forecast rows are immutable');
        END
        """
    )


def register_forecast(
    registry_path: str | Path,
    *,
    model_name: str,
    model_version: str,
    generated_at: str,
    data_as_of_date: str,
    target_date: str,
    quantile_return: float,
    training_start: str,
    training_end: str,
    feature_version: str,
    config_hash: str,
    data_hash: str,
    git_commit: str,
    alpha: float = 0.05,
) -> dict:
    """Register a forecast before its target trading session."""

    text_fields = {
        "model_name": model_name,
        "model_version": model_version,
        "feature_version": feature_version,
    }

    for field, value in text_fields.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} cannot be empty.")

    as_of = _validate_date(data_as_of_date, "data_as_of_date")
    target = _validate_date(target_date, "target_date")
    train_start = _validate_date(training_start, "training_start")
    train_end = _validate_date(training_end, "training_end")

    if not train_start <= train_end <= as_of < target:
        raise ValueError("Invalid training/as-of/target chronology.")

    try:
        generated = datetime.fromisoformat(generated_at)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid generated_at timestamp.") from error

    if generated.tzinfo is None or generated.utcoffset() is None:
        raise ValueError("generated_at must contain a timezone offset.")

    generated_date = generated.astimezone(VIETNAM_TZ).date()

    if not as_of <= generated_date < target:
        raise ValueError("Forecast must precede its target date.")

    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("Invalid quantile alpha.")

    if not math.isclose(float(alpha), 0.05, abs_tol=1e-12, rel_tol=0):
        raise ValueError("Phase 2 currently requires alpha = 0.05.")

    if isinstance(quantile_return, bool):
        raise ValueError("Invalid quantile return.")

    try:
        quantile = float(quantile_return)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid quantile return.") from error

    if not math.isfinite(quantile) or quantile < -1.0:
        raise ValueError("Invalid simple-return quantile.")

    _validate_hash(config_hash, "config_hash", 64)
    _validate_hash(data_hash, "data_hash", 64)
    _validate_hash(git_commit, "git_commit", 40)

    record = {
        "forecast_id": uuid.uuid4().hex,
        "model_name": model_name.strip(),
        "model_version": model_version.strip(),
        "generated_at": generated.isoformat(),
        "data_as_of_date": as_of.isoformat(),
        "target_date": target.isoformat(),
        "quantile_alpha": float(alpha),
        "quantile_return": quantile,
        "var_return": max(0.0, -quantile),
        "training_start": train_start.isoformat(),
        "training_end": train_end.isoformat(),
        "feature_version": feature_version.strip(),
        "config_hash": config_hash.lower(),
        "data_hash": data_hash.lower(),
        "git_commit": git_commit.lower(),
    }

    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(path, timeout=10)) as connection:
        with connection:
            _initialize(connection)

            columns = ", ".join(COLUMNS)
            placeholders = ", ".join("?" for _ in COLUMNS)

            try:
                connection.execute(
                    f"INSERT INTO forecasts ({columns}) "
                    f"VALUES ({placeholders})",
                    tuple(record[column] for column in COLUMNS),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(
                    "Forecast already exists for this "
                    "model/version/target, or violates a constraint."
                ) from error

    return {**record, "status": "PENDING"}


def get_forecast(
    registry_path: str | Path,
    forecast_id: str,
) -> dict:
    """Read an original forecast without modifying it."""

    path = Path(registry_path)

    if not path.exists():
        raise FileNotFoundError(path)

    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM forecasts WHERE forecast_id = ?",
            (forecast_id,),
        ).fetchone()

    if row is None:
        raise LookupError(f"Forecast not found: {forecast_id}")

    # Status is a projection. Verification will live in another table.
    return {**dict(row), "status": "PENDING"}