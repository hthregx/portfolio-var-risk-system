"""Reproducible Phase 2 A submission preflight; read-only inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import yaml

EXPECTED_MODELS = ("gb_G04_7f_a", "gbv2_12f_c1_a")
EXPECTED_CUTOFF = "2026-11-06"
EXPECTED_INPUT_SHA256 = "6389586ae8f1edbabb31760e0b555619186114b3f77d09212362fc2e1c40aadf"
EXPECTED_FEATURE_SHA256 = "14ac5dd4ce7dd6090420b9cee06c8ac40935c54691b4daf95d063b71a075eccc"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_holdout(pred: pd.DataFrame, portfolio: pd.DataFrame) -> dict:
    columns = {"model", "as_of", "target_date", "quantile_return", "var_return", "actual_return"}
    require(columns.issubset(pred.columns), "Missing holdout prediction columns")
    require(len(pred) == 936, "Expected exactly 936 A holdout rows")
    require(not pred[list(columns)].isna().any().any(), "Null holdout values")

    models = set(pred["model"].astype(str))
    require(models == set(EXPECTED_MODELS), "Unexpected holdout model set")

    actual_map = portfolio.set_index("date")["portfolio_simple_return"]
    dates_by_model = []
    calculated = {}
    for name in EXPECTED_MODELS:
        section = pred.loc[pred["model"] == name].copy()
        require(len(section) == 468, f"{name}: expected 468 predictions")
        require(not section["target_date"].duplicated().any(), f"{name}: duplicate target dates")
        target = pd.to_datetime(section["target_date"], errors="raise")
        as_of = pd.to_datetime(section["as_of"], errors="raise")
        require(target.is_monotonic_increasing, f"{name}: unsorted target dates")
        require(bool((as_of < target).all()), f"{name}: information leaked from target or future")
        dates = target.dt.strftime("%Y-%m-%d").tolist()
        require(set(dates).issubset(set(actual_map.index)), f"{name}: target missing in original data")
        recorded = section["actual_return"].to_numpy(dtype=float)
        source = actual_map.loc[dates].to_numpy(dtype=float)
        require(np.isfinite(recorded).all(), f"{name}: nonfinite actual return")
        require(np.allclose(recorded, source, rtol=1e-11, atol=1e-12), f"{name}: actual return mismatch")
        quantile = section["quantile_return"].to_numpy(dtype=float)
        var = section["var_return"].to_numpy(dtype=float)
        require(np.isfinite(quantile).all() and np.isfinite(var).all(), f"{name}: nonfinite forecast")
        require(np.allclose(var, np.maximum(0.0, -quantile), atol=1e-12, rtol=0), f"{name}: VaR sign mismatch")
        require(np.min(quantile) > -1.0, f"{name}: invalid simple return quantile")
        dates_by_model.append(dates)
        exceptions = int((recorded < quantile).sum())
        errors = recorded - quantile
        pinball = float(np.mean(np.maximum(0.05 * errors, -0.95 * errors)))
        calculated[name] = {"n": len(section), "exceptions": exceptions, "violation_rate": exceptions / len(section), "pinball": pinball}

    require(dates_by_model[0] == dates_by_model[1], "A model holdout dates are not directly comparable")
    require(calculated["gb_G04_7f_a"]["exceptions"] == 25, "Unexpected G04 exception count")
    require(calculated["gbv2_12f_c1_a"]["exceptions"] == 25, "Unexpected GBv2 exception count")
    return calculated


class _ImageRefs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            values = dict(attrs)
            if values.get("src"):
                self.sources.append(values["src"])


def validate_offline_html(path: Path) -> int:
    require(path.is_file() and path.stat().st_size > 1000, "Missing or too-small offline dashboard")
    parser = _ImageRefs()
    html = path.read_text(encoding="utf-8")
    parser.feed(html)
    require("GB" in html or "gbv2" in html.lower(), "Dashboard missing model results")
    for raw_src in parser.sources:
        if raw_src.startswith("data:image/"):
            continue
        require(not raw_src.startswith(("http:", "https:", "//")), "Dashboard depends on an external image")
        src = unquote(raw_src.split("?", 1)[0].split("#", 1)[0]).replace("\\", "/")
        require(".." not in Path(src).parts, "Dashboard path escapes output directory")
        require((path.parent / src).is_file(), f"Dashboard broken image: {raw_src}")
    return len(parser.sources)


def inspect_phase1_freeze(root: Path) -> dict:
    freeze_path = root / "configs/model_freeze.yaml"
    freeze = yaml.safe_load(freeze_path.read_text(encoding="utf-8"))
    out = {}
    for label, entry in freeze["canonical_artifacts"].items():
        p = root / entry["path"]
        require(p.is_file(), f"Frozen canonical artifact missing: {entry['path']}")
        actual = sha256(p)
        require(actual == entry["sha256"], f"Phase 1 frozen file CHANGED: {entry['path']}")
        out[label] = actual
    return out


def audit(root: Path, data_root: Path, *, write: bool = True) -> dict:
    root, data_root = root.resolve(), data_root.resolve()
    source_dir = data_root / "data/phase2"
    portfolio_path = source_dir / "portfolio_returns.csv"
    manifest_path = source_dir / "latest_manifest.json"
    require(portfolio_path.is_file() and manifest_path.is_file(), "Missing locked A market snapshot")
    data_hash = sha256(portfolio_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(data_hash == manifest["portfolio_sha256"] == EXPECTED_INPUT_SHA256, "A input SHA256 changed")
    require(manifest["cutoff"] == manifest["portfolio_last_date"] == EXPECTED_CUTOFF, "Snapshot cutoff changed")
    require(manifest["source"] == "KBS", "Data source is not KBS")
    for ticker in ("HPG", "FPT", "MWG", "VNINDEX"):
        require(manifest["symbols"][ticker]["last_date"] == EXPECTED_CUTOFF, f"{ticker}: stale data")
    require(manifest["symbols"]["VNINDEX"]["quality_status"] == "REVIEW", "VNINDEX gate changed without review")
    require(not manifest["symbols"]["VNINDEX"]["index_features_eligible"], "Unreviewed VNINDEX features enabled")

    portfolio = pd.read_csv(portfolio_path, dtype={"date": "string"})
    require(len(portfolio) == 1707, "Unexpected source return count")
    require(portfolio["date"].is_monotonic_increasing and not portfolio["date"].duplicated().any(), "Invalid return dates")
    require(str(portfolio["date"].iloc[-1]) == EXPECTED_CUTOFF, "Unexpected last return date")

    out_base = root / "results/phase2/a"
    features = out_base / f"features_{EXPECTED_CUTOFF}.csv"
    feature_meta = out_base / f"features_{EXPECTED_CUTOFF}_metadata.json"
    require(sha256(features) == EXPECTED_FEATURE_SHA256, "Feature matrix SHA256 changed")
    feature_info = json.loads(feature_meta.read_text(encoding="utf-8"))
    require(feature_info["input_sha256"] == data_hash, "Features used another data snapshot")
    require(feature_info["output_sha256"] == sha256(features), "Feature metadata output SHA mismatch")
    require(feature_info["feature_count"] == 12 and feature_info["feature_rows"] == 1647, "Unexpected feature dimensions")

    train = out_base / f"training_{EXPECTED_CUTOFF}"
    backtest = out_base / f"backtesting_{EXPECTED_CUTOFF}"
    b_audit = out_base / "crosscheck_b_2026-11-04"
    integration = out_base / f"integration_{EXPECTED_CUTOFF}"
    for p in (train, backtest, b_audit, integration):
        require(p.is_dir() and any(p.iterdir()), f"Missing A output directory: {p}")

    predictions = train / "holdout_predictions.csv"
    ranking = train / "cv_ranking.csv"
    require(predictions.is_file() and ranking.is_file(), "Missing A3 key training CSV")
    rank = pd.read_csv(ranking)
    require({"model", "cv_pinball"}.issubset(rank.columns), "CV ranking schema changed")
    rank = rank.sort_values("cv_pinball", ascending=True)
    require(rank.iloc[0]["model"] == "gbv2_12f_c1_a", "CV selected model changed")
    metrics = validate_holdout(pd.read_csv(predictions), portfolio)
    require(metrics["gbv2_12f_c1_a"]["pinball"] < metrics["gb_G04_7f_a"]["pinball"], "Holdout result no longer matches A3; investigate")

    pngs = sorted((backtest / "figures").glob("*.png"))
    require(len(pngs) == 5, "Expected exactly five A4 PNG figures")
    for png in pngs:
        require(png.read_bytes()[:8] == bytes.fromhex("89504e470d0a1a0a"), f"Corrupted PNG: {png}")

    html = integration / "comparison.html"
    html_refs = validate_offline_html(html)
    frozen_hashes = inspect_phase1_freeze(root)

    # Read-only data, A-generated experiment outputs, and frozen artifacts.
    tracked_inputs = [portfolio_path, manifest_path, features, feature_meta, predictions, ranking, html]
    inventory_paths = sorted(set(tracked_inputs + [p for folder in (train, backtest, b_audit, integration) for p in folder.rglob("*") if p.is_file()]))
    inventory = [{"path": str(p.relative_to(root)) if p.is_relative_to(root) else str(p), "sha256": sha256(p), "bytes": p.stat().st_size} for p in inventory_paths]

    report = {
        "stage": "A7_pre_submission_technical_audit",
        "status": "PASS_WITH_DECLARED_LIMITATIONS",
        "cutoff": EXPECTED_CUTOFF,
        "input_sha256": data_hash,
        "feature_sha256": EXPECTED_FEATURE_SHA256,
        "feature_count": 12,
        "feature_rows": 1647,
        "selected_on_development_cv": "gbv2_12f_c1_a",
        "holdout_results": metrics,
        "backtesting_figures": len(pngs),
        "dashboard_image_refs": html_refs,
        "canonical_phase1_artifact_hashes": frozen_hashes,
        "inventory": inventory,
        "limitations": [
            "B's original 2026-11-04 source bytes/hash not independently verified",
            "A and B GB evaluation windows/protocols differ; no direct score comparison",
            "Phase 1 historical evaluation period overlaps A holdout; not a pristine untouched test",
            "VNINDEX has three unresolved OHLC audit anomalies; market features disabled",
            "Main Cloud EOD workflow needs updated model-test dependencies before merging A tests into phase2/develop",
        ],
    }
    if write:
        dest_dir = out_base / f"release_{EXPECTED_CUTOFF}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "technical_audit.json"
        require(not dest.exists(), "Existing A7 audit; never silently overwrite")
        dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("A7 AUDIT JSON:", dest)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    opts = p.parse_args()
    report = audit(opts.repo_root, opts.data_root)
    print("SOURCE SNAPSHOT:", report["cutoff"])
    print("SOURCE SHA256:", report["input_sha256"])
    print("FEATURE COUNT:", report["feature_count"])
    print("CV SELECTED:", report["selected_on_development_cv"])
    print("HOLDOUT MODELS:", list(report["holdout_results"]))
    print("PHASE 1 ARTIFACTS VERIFIED:", len(report["canonical_phase1_artifact_hashes"]))
    print("FIGURES VERIFIED:", report["backtesting_figures"])
    print("INVENTORY FILES:", len(report["inventory"]))
    print("DECLARED LIMITATIONS:", len(report["limitations"]))
    print("A7 RELEASE AUDIT PASS WITH LIMITATIONS")


if __name__ == "__main__":
    main()
