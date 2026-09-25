from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

EXPECTED_MODELS = [
    "historical_simulation",
    "ewma",
    "gradient_boosting",
]
EXPECTED_FEATURES = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_5",
    "rolling_vol_5",
    "rolling_vol_20",
    "rolling_vol_60",
    "drawdown",
]
EXPECTED_OPERATIONAL_CUTOFF = "2026-08-28"
EXPECTED_CANONICAL_CUTOFF = "2026-07-28"
TOL = 1e-12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "configs" / "model_freeze.yaml").exists() and (parent / "results").exists():
            return parent
    raise RuntimeError("Could not locate repository root from script path.")


def finite_series(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce")
    return values.notna().all() and values.map(math.isfinite).all()


def main() -> int:
    root = find_repo_root()
    forecast_path = root / "results" / "latest_forecast_2026-08-28.csv"
    metadata_path = root / "results" / "latest_forecast_metadata_2026-08-28.json"
    freeze_path = root / "configs" / "model_freeze.yaml"

    require(forecast_path.exists(), f"Missing forecast artifact: {forecast_path}")
    require(metadata_path.exists(), f"Missing forecast metadata: {metadata_path}")
    require(freeze_path.exists(), f"Missing model freeze: {freeze_path}")

    df = pd.read_csv(forecast_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    freeze = yaml.safe_load(freeze_path.read_text(encoding="utf-8"))

    expected_columns = [
        "cutoff_date",
        "forecast_date",
        "target_date",
        "model",
        "quantile_return",
        "var_return",
        "confidence_level",
        "horizon_trading_days",
    ]
    require(list(df.columns) == expected_columns, f"Unexpected forecast schema: {list(df.columns)}")
    require(len(df) == 3, f"Expected exactly 3 model rows, got {len(df)}")
    require(df["model"].tolist() == EXPECTED_MODELS, f"Unexpected model rows/order: {df['model'].tolist()}")
    require(df["model"].is_unique, "Model rows must be unique")

    for col in ["quantile_return", "var_return", "confidence_level", "horizon_trading_days"]:
        require(finite_series(df[col]), f"Column {col} contains missing/non-finite values")

    q = pd.to_numeric(df["quantile_return"])
    v = pd.to_numeric(df["var_return"])
    expected_v = (-q).clip(lower=0.0)
    max_var_error = float((v - expected_v).abs().max())
    require((v >= -TOL).all(), "VaR must be non-negative")
    require(max_var_error <= TOL, f"VaR identity failed; max abs error={max_var_error:.3e}")
    require((pd.to_numeric(df["confidence_level"]) - 0.95).abs().max() <= TOL, "confidence_level must be 0.95")
    require((pd.to_numeric(df["horizon_trading_days"]) == 1).all(), "horizon must be one trading day")

    for col in ["cutoff_date", "forecast_date", "target_date"]:
        parsed = pd.to_datetime(df[col], errors="coerce")
        require(parsed.notna().all(), f"Invalid dates in {col}")

    cutoff = pd.to_datetime(df["cutoff_date"])
    forecast = pd.to_datetime(df["forecast_date"])
    target = pd.to_datetime(df["target_date"])
    require(cutoff.nunique() == 1, "cutoff_date must be common across the 3 model rows")
    require(forecast.nunique() == 1, "forecast_date must be common across the 3 model rows")
    require(target.nunique() == 1, "target_date must be common across the 3 model rows")
    require((cutoff == forecast).all(), "forecast_date must equal data cutoff for this latest snapshot forecast")
    require((target > forecast).all(), "target_date must be strictly after forecast_date")
    require(df["cutoff_date"].iloc[0] == EXPECTED_OPERATIONAL_CUTOFF, "Operational cutoff must be 2026-08-28")

    research = metadata["research_contract"]
    operational = metadata["operational_contract"]
    models = metadata["models"]

    require(research["canonical_data_cutoff"] == EXPECTED_CANONICAL_CUTOFF, "Canonical cutoff changed")
    require(research["canonical_evaluation_extended"] is False, "Canonical evaluation must not be extended")
    require(research["canonical_predictions_rewritten"] is False, "Canonical predictions must not be rewritten")
    require(operational["data_cutoff"] == EXPECTED_OPERATIONAL_CUTOFF, "Metadata operational cutoff mismatch")
    require(operational["forecast_date"] == EXPECTED_OPERATIONAL_CUTOFF, "Metadata forecast_date mismatch")
    require(operational["target_semantics"] == "next_trading_session", "Unexpected target semantics")
    require(abs(float(operational["confidence_level"]) - 0.95) <= TOL, "Metadata confidence must be 0.95")
    require(abs(float(operational["alpha"]) - 0.05) <= TOL, "Metadata alpha must be 0.05")
    require(int(operational["horizon_trading_days"]) == 1, "Metadata horizon must be one trading day")

    require(models["historical_simulation"]["config_id"] == "historical_w250", "Historical config_id mismatch")
    require(int(models["historical_simulation"]["window"]) == 250, "Historical window must be 250")
    require(models["historical_simulation"]["mode"] == "rolling", "Historical mode must be rolling")

    require(models["ewma"]["config_id"] == "ewma_d094", "EWMA config_id mismatch")
    require(abs(float(models["ewma"]["decay"]) - 0.94) <= TOL, "EWMA decay must be 0.94")
    require(models["ewma"]["mode"] == "expanding", "EWMA mode must be expanding")

    gb = models["gradient_boosting"]
    require(gb["config_id"] == "gb_G04", "GB must be canonical G04")
    require(gb["features"] == EXPECTED_FEATURES, f"GB features mismatch: {gb['features']}")
    require(gb["market_features_used"] is False, "Canonical G04 must not use market features")

    freeze_eval = freeze["evaluation"]
    freeze_models = freeze["models"]
    require(freeze_eval["end_date"] == EXPECTED_CANONICAL_CUTOFF, "model_freeze canonical evaluation end changed")
    require(freeze_models["historical_simulation"]["window"] == 250, "Freeze Historical w250 mismatch")
    require(abs(float(freeze_models["ewma"]["decay"]) - 0.94) <= TOL, "Freeze EWMA decay mismatch")
    require(freeze_models["gradient_boosting"]["experiment_id"] == "G04", "Freeze GB experiment mismatch")

    # Cross-check row-level values against metadata.
    require((df["confidence_level"].astype(float) == float(operational["confidence_level"])).all(), "CSV/metadata confidence mismatch")
    require((df["horizon_trading_days"].astype(int) == int(operational["horizon_trading_days"])).all(), "CSV/metadata horizon mismatch")
    require((df["cutoff_date"] == operational["data_cutoff"]).all(), "CSV/metadata cutoff mismatch")
    require((df["forecast_date"] == operational["forecast_date"]).all(), "CSV/metadata forecast_date mismatch")
    require((df["target_date"] == operational["target_date"]).all(), "CSV/metadata target_date mismatch")

    print("PASS - B28.4 INDEPENDENT FORECAST-CONTRACT VALIDATOR")
    print()
    print("Forecast artifact")
    print(f"  rows = {len(df)}")
    print(f"  models = {', '.join(df['model'])}")
    print(f"  data cutoff = {df['cutoff_date'].iloc[0]}")
    print(f"  forecast date = {df['forecast_date'].iloc[0]}")
    print(f"  target date = {df['target_date'].iloc[0]}")
    print(f"  max VaR identity abs error = {max_var_error:.3e}")
    print("  [PASS] exactly 3 model rows")
    print("  [PASS] expected model names and unique rows")
    print("  [PASS] quantile_return values are finite")
    print("  [PASS] var_return values are finite and non-negative")
    print("  [PASS] VaR = max(0, -quantile_return)")
    print("  [PASS] confidence = 0.95")
    print("  [PASS] horizon = one trading day")
    print("  [PASS] cutoff_date = forecast_date = 2026-08-28")
    print("  [PASS] target_date is strictly after forecast_date")
    print()
    print("Metadata / frozen-contract cross-check")
    print("  [PASS] Historical = historical_w250 / rolling")
    print("  [PASS] EWMA = ewma_d094 / decay 0.94 / expanding")
    print("  [PASS] GB = gb_G04")
    print("  [PASS] exact 7 GB return-history features")
    print("  [PASS] market features = false")
    print("  [PASS] canonical evaluation cutoff remains 2026-07-28")
    print("  [PASS] canonical evaluation not extended")
    print("  [PASS] canonical predictions not rewritten")
    print("  [PASS] operational cutoff 2026-08-28 remains separate")
    print("  [PASS] CSV dates/confidence/horizon agree with metadata")
    print()
    print("[PASS] validator is read-only; no forecast/canonical artifact is written")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL - B28.4 FORECAST-CONTRACT VALIDATOR: {exc}", file=sys.stderr)
        raise
