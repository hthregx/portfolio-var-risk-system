"""Create versioned 18/25-feature experiments from A's immutable KBS snapshot.

Exports are NOT fitted models and cannot select a winner. Old holdout metrics,
Phase 2 predictions and Phase 1 frozen results are not read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.phase2.features_a import ALL_FEATURES
from src.phase3.features_a import (
    EXTRA_TO_18, EXTRA_TO_25, FEATURES_18, FEATURES_25,
    build_phase3_features, select_feature_set,
)

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = "2026-11-06"
DEV_END = "2024-12-17"
BASE_SOURCE_SHA = "6389586ae8f1edbabb31760e0b555619186114b3f77d09212362fc2e1c40aadf"
BASE_FEATURE_SHA = "14ac5dd4ce7dd6090420b9cee06c8ac40935c54691b4daf95d063b71a075eccc"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_source(snapshot: Path, feature_csv: Path, lock_json: Path):
    portfolio = snapshot / "data/phase2/portfolio_returns.csv"
    manifest = json.loads((snapshot / "data/phase2/latest_manifest.json").read_text(encoding="utf-8"))
    oldmeta = json.loads(feature_csv.with_name(f"features_{CUTOFF}_metadata.json").read_text(encoding="utf-8"))
    lock = json.loads(lock_json.read_text(encoding="utf-8-sig"))
    if not (sha256(portfolio) == BASE_SOURCE_SHA == manifest["portfolio_sha256"]
            == oldmeta["input_sha256"] == lock["source_sha256"]):
        raise ValueError("Phase 2 source snapshot hash changed")
    if not (sha256(feature_csv) == BASE_FEATURE_SHA == oldmeta["output_sha256"]
            == lock["features_sha256"]):
        raise ValueError("Phase 2 12-feature CSV hash changed")
    if (manifest["source"] != "KBS" or manifest["cutoff"] != CUTOFF
            or manifest["portfolio_last_date"] != CUTOFF):
        raise ValueError("Wrong snapshot source or cutoff")
    if tuple(oldmeta["feature_names"]) != tuple(ALL_FEATURES):
        raise ValueError("Phase 2 feature schema drift")
    if lock["training_cutoff"] != DEV_END or lock["selected_on_development_cv"] != "gbv2_12f_c1_a":
        raise ValueError("Phase 3 baseline lock not valid")
    raw = pd.read_csv(portfolio, dtype={"date": str}, float_precision="round_trip")
    previous = pd.read_csv(feature_csv, parse_dates=["target_date", "as_of"], float_precision="round_trip")
    if len(raw) != manifest["rows"] or len(previous) != oldmeta["feature_rows"]:
        raise ValueError("Source or baseline row count mismatch")
    date = pd.to_datetime(raw["date"], format="%Y-%m-%d", errors="raise")
    if date.hasnans or date.duplicated().any() or not date.is_monotonic_increasing:
        raise ValueError("Invalid trading date sequence")
    if raw["date"].iloc[-1] != CUTOFF:
        raise ValueError("Stale source")
    series = pd.Series(raw["portfolio_simple_return"].to_numpy(float), index=pd.DatetimeIndex(date), name="portfolio_simple_return")
    return series, previous


def export(snapshot: Path, output: Path, feature_csv: Path, lock_json: Path):
    if output.exists():
        raise FileExistsError("Phase 3 feature output already exists; never overwrite")
    series, old = _verify_source(snapshot, feature_csv, lock_json)
    full = build_phase3_features(series)
    if len(full) != 1707 or len(series) != 1707:
        raise ValueError("Unexpected source row count; verify snapshot")
    if full.iloc[:60].notna().all(axis=1).any():
        raise ValueError("Unexpected early fully valid rows")
    if not full.iloc[60:].notna().all(axis=None):
        bad = full.iloc[60:].columns[~full.iloc[60:].notna().all(axis=0)].tolist()
        raise ValueError(f"Missing feature after warmup: {bad}")
    if not np.isfinite(full.iloc[60:].to_numpy(float)).all():
        raise ValueError("Nonfinite values after 60-day warmup")
    historical = full.iloc[60:][list(ALL_FEATURES)]
    if old.columns.tolist() != ["target_date", "as_of", *ALL_FEATURES, "actual_return"]:
        raise ValueError("Old baseline columns not as expected")
    np.testing.assert_allclose(historical.to_numpy(float), old[list(ALL_FEATURES)].to_numpy(float), rtol=1e-10, atol=1e-12)
    if not np.array_equal(old.target_date.to_numpy(), series.index[60:].to_numpy()):
        raise ValueError("Phase 2 target dates changed")
    if not np.array_equal(old.as_of.to_numpy(), series.index[59:-1].to_numpy()):
        raise ValueError("Phase 2 as_of alignment changed")
    np.testing.assert_allclose(old.actual_return.to_numpy(float), series.iloc[60:].to_numpy(float), rtol=1e-10, atol=1e-12)
    valid = full.iloc[60:]
    result = []
    for count in (18, 25):
        values = select_feature_set(valid, count)
        table = values.copy()
        table.insert(0, "as_of", series.index[59:-1].to_numpy())
        table.insert(0, "target_date", series.index[60:].to_numpy())
        table["actual_return"] = series.iloc[60:].to_numpy(float)
        if not (table.as_of < table.target_date).all():
            raise ValueError("Date/lookahead contract violated")
        if len(table) != 1647 or len(table.columns) != count + 3:
            raise ValueError("Unexpected feature matrix shape")
        result.append((count, table))
    # All checks pass before writing any artifact.
    output.mkdir(parents=True, exist_ok=False)
    for count, table in result:
        csv = output / f"features_{count}_{CUTOFF}.csv"
        meta_path = output / f"features_{count}_{CUTOFF}_metadata.json"
        table.to_csv(csv, index=False, date_format="%Y-%m-%d", float_format="%.17g")
        meta = {
            "stage": "phase3_features_only_not_trained", "owner": "A",
            "cutoff": CUTOFF, "source": "KBS", "target": "portfolio_simple_return",
            "alpha": .05, "dev_end": DEV_END,
            "source_sha256": BASE_SOURCE_SHA, "phase2_feature_sha256": BASE_FEATURE_SHA,
            "baseline_lock_sha256": sha256(lock_json),
            "feature_count": count, "features": list(table.columns[2:-1]),
            "feature_rows": len(table), "warmup_rows": 60,
            "first_target": str(table.target_date.iloc[0].date()),
            "last_target": str(table.target_date.iloc[-1].date()),
            "development_rows": int((table.target_date <= DEV_END).sum()),
            "sha256": sha256(csv),
            "lag_policy": "all features of target t use returns strictly before t",
            "ewma_note": "EWMA vol uses alpha=0.06, adjust=False on lagged squared returns; initialization first observed square",
            "vol_ratio_note": "add 1e-12 to denominator to avoid divide-by-zero in near-zero volatility",
            "selection_warning": "NO model selection or fitting in this step; any later fitting/selection must use only target dates <= 2024-12-17. Old holdout has already been examined.",
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"FEATURE COUNT: {count}; ROWS: {len(table)}; DEV ROWS: {meta['development_rows']}", flush=True)
        print(f"{count}F SHA256: {meta['sha256']}", flush=True)
    print("PHASE 2 FIRST 12 FEATURES: VERIFIED UNCHANGED")
    print("LOOKAHEAD GUARD: TARGET-DATE FEATURES USE ONLY PAST RETURN")
    print("OLD HOLDOUT USED TO SELECT FEATURES: NO")
    print("OUTPUT:", output)
    print("STEP P3-02 FEATURE ENGINEERING PASS")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--feature-csv", required=True, type=Path)
    p.add_argument("--baseline-lock", required=True, type=Path)
    a = p.parse_args()
    export(a.snapshot.resolve(), a.out.resolve(), a.feature_csv.resolve(), a.baseline_lock.resolve())


if __name__ == "__main__":
    main()
