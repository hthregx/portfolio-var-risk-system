from __future__ import annotations

import csv
import fnmatch
import hashlib
import io
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/release_candidate_b.yaml"
MANIFEST_PATH = ROOT / "results/release_candidate_manifest_b.json"
PREDICTIONS_PATH = ROOT / "results/final_predictions.csv"
DAY24_VALIDATION_PATH = ROOT / "results/release_reproducibility_validation_b.csv"
OUTPUT_PATH = ROOT / "results/release_smoke_validation_b.csv"

EXPECTED_GB_FEATURES = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_5",
    "rolling_vol_5",
    "rolling_vol_20",
    "rolling_vol_60",
    "drawdown",
]


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def add(checks, check, expected, actual, ok, source):
    checks.append(
        {
            "check": check,
            "expected": str(expected),
            "actual": str(actual),
            "status": "PASS" if ok else "FAIL",
            "source": source,
        }
    )


def excluded(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    rc = config["release_candidate"]
    frozen = config["frozen_contract"]
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    base = rc["integration_base_sha"]
    artifacts = manifest["artifacts"]
    paths = [item["path"] for item in artifacts]
    patterns = rc["excluded_patterns"]
    checks = []

    try:
        git("cat-file", "-e", f"{base}^{{commit}}")
        base_ok = True
    except subprocess.CalledProcessError:
        base_ok = False

    add(checks, "release_base_resolvable", True, base_ok, base_ok, "git")

    required_manifest_keys = {
        "release_candidate",
        "integration_base_sha",
        "artifact_count",
        "excluded_patterns",
        "frozen_contract",
        "artifacts",
    }
    schema_ok = required_manifest_keys.issubset(manifest)
    add(
        checks,
        "manifest_schema",
        True,
        schema_ok,
        schema_ok,
        str(MANIFEST_PATH.relative_to(ROOT)),
    )

    expected_count = len(rc["required_artifacts"])
    add(
        checks,
        "manifest_artifact_count",
        expected_count,
        len(artifacts),
        len(artifacts) == expected_count,
        "manifest",
    )

    unique = len(paths) == len(set(paths))
    add(checks, "manifest_unique_paths", True, unique, unique, "manifest")

    tracked = True
    blobs_resolvable = True
    hashes_present = True
    hashes_correct = True

    for item in artifacts:
        path = item["path"]

        try:
            oid = git("rev-parse", f"{base}:{path}").decode().strip()
            blob = git("cat-file", "-p", oid)
        except subprocess.CalledProcessError:
            tracked = False
            blobs_resolvable = False
            continue

        if oid != item.get("git_blob_oid"):
            tracked = False

        digest = hashlib.sha256(blob).hexdigest()

        if not item.get("sha256_git_blob"):
            hashes_present = False
        if digest != item.get("sha256_git_blob"):
            hashes_correct = False

    add(checks, "all_manifest_files_tracked", True, tracked, tracked, "git")
    add(
        checks,
        "all_git_blobs_resolvable",
        True,
        blobs_resolvable,
        blobs_resolvable,
        "git",
    )
    add(
        checks,
        "all_sha256_present",
        True,
        hashes_present,
        hashes_present,
        "manifest",
    )
    add(
        checks,
        "all_sha256_match_git_blobs",
        True,
        hashes_correct,
        hashes_correct,
        "git+manifest",
    )

    private_ok = (
        "data/processed/portfolio_returns.csv" not in paths
        and not any(excluded(path, patterns) for path in paths)
    )
    add(
        checks,
        "private_data_excluded",
        True,
        private_ok,
        private_ok,
        "config+manifest",
    )

    required_paths = set(paths)

    def present(path):
        return path in required_paths

    add(
        checks,
        "model_freeze_present",
        True,
        present("configs/model_freeze.yaml"),
        present("configs/model_freeze.yaml"),
        "manifest",
    )

    add(
        checks,
        "historical_window_250",
        250,
        frozen["historical"]["window"],
        frozen["historical"]["window"] == 250,
        "release config",
    )
    add(
        checks,
        "ewma_decay_094",
        0.94,
        frozen["ewma"]["decay"],
        frozen["ewma"]["decay"] == 0.94,
        "release config",
    )
    add(
        checks,
        "gb_config_G04",
        "gb_G04",
        frozen["gradient_boosting"]["config_id"],
        frozen["gradient_boosting"]["config_id"] == "gb_G04",
        "release config",
    )

    features = frozen["gradient_boosting"]["features"]
    add(
        checks,
        "gb_feature_count_7",
        7,
        len(features),
        len(features) == 7,
        "release config",
    )
    add(
        checks,
        "gb_features_exact",
        "|".join(EXPECTED_GB_FEATURES),
        "|".join(features),
        features == EXPECTED_GB_FEATURES,
        "release config",
    )

    prediction_oid = git(
        "rev-parse",
        f"{base}:results/final_predictions.csv",
    ).decode().strip()
    prediction_blob = git("cat-file", "-p", prediction_oid)
    predictions = pd.read_csv(io.BytesIO(prediction_blob))

    add(
        checks,
        "prediction_rows_1194",
        1194,
        len(predictions),
        len(predictions) == 1194,
        "final_predictions",
    )

    methods = predictions["method"].nunique()
    add(
        checks,
        "method_count_3",
        3,
        methods,
        methods == 3,
        "final_predictions",
    )

    counts = predictions.groupby("method").size()
    forecast_ok = len(counts) == 3 and counts.eq(398).all()
    add(
        checks,
        "forecasts_per_method_398",
        398,
        counts.to_dict(),
        forecast_ok,
        "final_predictions",
    )

    start = str(predictions["target_date"].min())
    end = str(predictions["target_date"].max())

    add(
        checks,
        "evaluation_start",
        "2024-12-18",
        start,
        start == "2024-12-18",
        "final_predictions",
    )
    add(
        checks,
        "evaluation_end",
        "2026-07-28",
        end,
        end == "2026-07-28",
        "final_predictions",
    )

    actual = pd.to_numeric(predictions["actual_return"])
    quantile = pd.to_numeric(predictions["quantile_return"])
    var = pd.to_numeric(predictions["var"])

    stored = (
        predictions["violation"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

    violation_ok = stored.notna().all() and stored.equals(actual.lt(quantile))
    var_ok = ((var - (-quantile).clip(lower=0.0)).abs() <= 1e-12).all()

    add(
        checks,
        "strict_violation_contract",
        "actual_return < quantile_return",
        violation_ok,
        violation_ok,
        "final_predictions",
    )
    add(
        checks,
        "var_identity_contract",
        "max(0, -quantile_return)",
        var_ok,
        var_ok,
        "final_predictions",
    )

    add(
        checks,
        "final_metrics_present",
        True,
        present("results/final_metrics.csv"),
        present("results/final_metrics.csv"),
        "manifest",
    )
    add(
        checks,
        "pairwise_summary_present",
        True,
        present("results/final_pairwise_summary.csv"),
        present("results/final_pairwise_summary.csv"),
        "manifest",
    )

    day24_present = present(
        "results/release_reproducibility_validation_b.csv"
    )
    add(
        checks,
        "day24_validation_present",
        True,
        day24_present,
        day24_present,
        "manifest",
    )

    day24 = pd.read_csv(DAY24_VALIDATION_PATH)
    day24_pass = "status" in day24.columns and day24["status"].eq("PASS").all()

    add(
        checks,
        "day24_validation_all_pass",
        True,
        day24_pass,
        day24_pass,
        "day24 validation",
    )

    figure_paths = {
        "results/figures/final_average_var_a.png",
        "results/figures/final_pinball_loss_a.png",
        "results/figures/final_violation_rate_a.png",
    }
    figures_ok = figure_paths.issubset(required_paths)

    add(
        checks,
        "day24_figures_present",
        True,
        figures_ok,
        figures_ok,
        "manifest",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["check", "expected", "actual", "status", "source"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(checks)

    failures = sum(row["status"] == "FAIL" for row in checks)

    print(f"checks: {len(checks)}")
    print(f"PASS: {len(checks) - failures}")
    print(f"FAIL: {failures}")

    if failures:
        print("B3_RELEASE_SMOKE_VALIDATION_FAIL")
        return 1

    print("B3_RELEASE_SMOKE_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
