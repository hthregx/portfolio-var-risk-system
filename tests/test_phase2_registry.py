import sqlite3

import pytest

from src.phase2.forecast_registry import (
    get_forecast,
    register_forecast,
)


def example_forecast():
    return {
        "model_name": "gb_quantile_v2",
        "model_version": "v2.0.0",
        "generated_at": "2026-10-28T16:30:00+07:00",
        "data_as_of_date": "2026-10-28",
        "target_date": "2026-10-29",
        "quantile_return": -0.0272,
        "training_start": "2020-01-03",
        "training_end": "2026-10-28",
        "feature_version": "F07",
        "config_hash": "a" * 64,
        "data_hash": "b" * 64,
        "git_commit": "c" * 40,
    }


def test_register_and_read_forecast(tmp_path):
    database = tmp_path / "registry.sqlite3"

    saved = register_forecast(database, **example_forecast())

    assert saved["status"] == "PENDING"
    assert saved["var_return"] == pytest.approx(0.0272)
    assert len(saved["forecast_id"]) == 32

    loaded = get_forecast(database, saved["forecast_id"])

    assert loaded["forecast_id"] == saved["forecast_id"]
    assert loaded["quantile_return"] == pytest.approx(-0.0272)
    assert loaded["target_date"] == "2026-10-29"


def test_duplicate_forecast_is_rejected(tmp_path):
    database = tmp_path / "registry.sqlite3"
    payload = example_forecast()

    register_forecast(database, **payload)

    with pytest.raises(ValueError, match="already exists"):
        register_forecast(database, **payload)

    with sqlite3.connect(database) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM forecasts"
        ).fetchone()[0]

    assert count == 1


def test_forecast_cannot_be_updated_or_deleted(tmp_path):
    database = tmp_path / "registry.sqlite3"

    saved = register_forecast(database, **example_forecast())
    forecast_id = saved["forecast_id"]

    with sqlite3.connect(database) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE forecasts SET quantile_return = ? "
                "WHERE forecast_id = ?",
                (-0.01, forecast_id),
            )

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "DELETE FROM forecasts WHERE forecast_id = ?",
                (forecast_id,),
            )

    loaded = get_forecast(database, forecast_id)
    assert loaded["quantile_return"] == pytest.approx(-0.0272)


def test_future_information_is_rejected(tmp_path):
    database = tmp_path / "registry.sqlite3"
    payload = example_forecast()

    payload["generated_at"] = "2026-10-29T17:00:00+07:00"

    with pytest.raises(ValueError, match="precede"):
        register_forecast(database, **payload)

    assert not database.exists()