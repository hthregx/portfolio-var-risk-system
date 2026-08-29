from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/release_candidate_b.yaml"
MANIFEST = ROOT / "results/release_candidate_manifest_b.json"
LEDGER = ROOT / "results/release_artifact_ledger_b.csv"
SMOKE = ROOT / "results/release_smoke_validation_b.csv"
RUNBOOK = ROOT / "docs/day25-release-candidate-b.md"

EXPECTED_FEATURES = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_5",
    "rolling_vol_5",
    "rolling_vol_20",
    "rolling_vol_60",
    "drawdown",
]


def load_config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def test_release_config_schema():
    cfg = load_config()
    assert set(cfg) == {"release_candidate", "frozen_contract"}

    rc = cfg["release_candidate"]
    assert rc["base_ref"] == "feature/eda-analysis"
    assert len(rc["integration_base_sha"]) == 40
    assert rc["required_artifacts"]
    assert rc["excluded_patterns"]


def test_manifest_schema_and_unique_paths():
    manifest = load_manifest()

    required = {
        "release_candidate",
        "integration_base_sha",
        "artifact_count",
        "excluded_patterns",
        "frozen_contract",
        "artifacts",
    }
    assert required.issubset(manifest)

    paths = [item["path"] for item in manifest["artifacts"]]
    assert manifest["artifact_count"] == len(paths)
    assert len(paths) == len(set(paths))


def test_manifest_tracked_only_and_git_blobs():
    manifest = load_manifest()
    base = manifest["integration_base_sha"]

    for item in manifest["artifacts"]:
        path = item["path"]

        oid = git("rev-parse", f"{base}:{path}").decode().strip()
        assert oid == item["git_blob_oid"]

        blob = git("cat-file", "-p", oid)
        assert hashlib.sha256(blob).hexdigest() == item["sha256_git_blob"]
        assert len(blob) == int(item["blob_bytes"])
        assert item["status"] == "PASS"


def test_excluded_paths_and_private_data_absent():
    cfg = load_config()
    manifest = load_manifest()

    patterns = cfg["release_candidate"]["excluded_patterns"]
    paths = [item["path"] for item in manifest["artifacts"]]

    assert "data/**" in patterns
    assert ".venv/**" in patterns
    assert ".env" in patterns
    assert ".env.*" in patterns
    assert "**/__pycache__/**" in patterns
    assert ".pytest_cache/**" in patterns
    assert "**/.ipynb_checkpoints/**" in patterns

    assert "data/processed/portfolio_returns.csv" not in paths
    assert not any(path.startswith("data/") for path in paths)
    assert not any(".env" in path for path in paths)
    assert not any("credentials" in path.lower() for path in paths)


def test_frozen_model_contract():
    frozen = load_config()["frozen_contract"]

    assert frozen["portfolio"]["weighting"] == "equal_weight"
    assert frozen["portfolio"]["assets"] == ["HPG", "FPT", "MWG"]
    assert frozen["forecast_horizon_trading_days"] == 1
    assert frozen["confidence_level"] == 0.95
    assert frozen["alpha"] == 0.05

    assert frozen["historical"]["window"] == 250
    assert frozen["ewma"]["decay"] == 0.94

    gb = frozen["gradient_boosting"]
    assert gb["config_id"] == "gb_G04"
    assert gb["feature_count"] == 7
    assert gb["features"] == EXPECTED_FEATURES


def test_prediction_contract():
    predictions = pd.read_csv(ROOT / "results/final_predictions.csv")

    assert len(predictions) == 1194
    assert predictions["method"].nunique() == 3

    counts = predictions.groupby("method").size()
    assert counts.eq(398).all()

    assert predictions["target_date"].min() == "2024-12-18"
    assert predictions["target_date"].max() == "2026-07-28"


def test_release_validator_all_pass():
    validation = pd.read_csv(SMOKE)

    assert list(validation.columns) == [
        "check",
        "expected",
        "actual",
        "status",
        "source",
    ]
    assert len(validation) >= 24
    assert validation["check"].is_unique
    assert validation["status"].eq("PASS").all()


def test_manifest_is_deterministic():
    subprocess.run(
        [sys.executable, "scripts/build_release_candidate_b.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    first_ledger = LEDGER.read_bytes()
    first_manifest = MANIFEST.read_bytes()

    subprocess.run(
        [sys.executable, "scripts/build_release_candidate_b.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert LEDGER.read_bytes() == first_ledger
    assert MANIFEST.read_bytes() == first_manifest


def test_runbook_release_boundaries():
    text = RUNBOOK.read_text(encoding="utf-8").lower()

    assert "pristine, never-inspected" in text
    assert "not a" in text
    assert "full final model rerun" in text
    assert "not a\nday 25 release acceptance gate" in text
    assert "data/processed/portfolio_returns.csv" in text
    assert "canonical git blob bytes" in text


def test_day24_release_evidence_present():
    manifest = load_manifest()
    paths = {item["path"] for item in manifest["artifacts"]}

    required = {
        "results/release_reproducibility_validation_b.csv",
        "results/final_evidence_validation_a.csv",
        "results/figures/final_average_var_a.png",
        "results/figures/final_pinball_loss_a.png",
        "results/figures/final_violation_rate_a.png",
    }

    assert required.issubset(paths)
