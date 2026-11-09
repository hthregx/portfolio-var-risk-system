"""A-only Phase 3: verify and freeze Phase 2 CV baselines before optimizing.

Strictly development CV only. Never reads Phase 2 holdout predictions or metrics.
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

ALPHA = 0.05
EXPECTED_DEV_END = "2024-12-17"
EXPECTED_CUTOFF = "2026-11-06"
EXPECTED_MODEL = "gbv2_12f_c1_a"
MODELS = ("gb_G04_7f_a", "gbv2_12f_c1_a", "gbv2_12f_c2_a")
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinball(y, q, alpha: float = ALPHA) -> np.ndarray:
    y, q = np.asarray(y, dtype=float), np.asarray(q, dtype=float)
    if y.ndim != 1 or q.shape != y.shape or y.size == 0:
        raise ValueError("Invalid return/quantile shapes")
    if not np.isfinite(y).all() or not np.isfinite(q).all():
        raise ValueError("Nonfinite return/quantile")
    if not 0 < alpha < 1:
        raise ValueError("Invalid alpha")
    return np.maximum(alpha * (y - q), (alpha - 1) * (y - q))


def evaluate(frame: pd.DataFrame) -> dict:
    if frame.empty:
        raise ValueError("Empty forecast frame")
    y, q = frame["actual_return"].to_numpy(float), frame["quantile_return"].to_numpy(float)
    v = y < q
    return dict(n=len(frame), violations=int(v.sum()), violation_rate=float(v.mean()),
                pinball_loss=float(pinball(y, q).mean()),
                average_var=float(np.maximum(0, -q).mean()))


def assert_forecast_integrity(p: pd.DataFrame, features: pd.DataFrame, folds: list[dict]) -> None:
    required = {"model", "fold", "as_of", "target_date", "quantile_return",
                "var_return", "actual_return", "violation"}
    if not required.issubset(p.columns):
        raise ValueError("CV predictions schema mismatch")
    if len(p) != 2940 or set(p.model) != set(MODELS):
        raise ValueError("Unexpected models/CV row count")
    if p.duplicated(["model", "target_date"]).any():
        raise ValueError("Duplicate target dates for model")
    if not (p.as_of < p.target_date).all():
        raise ValueError("Future information or invalid as_of")
    if not (p.target_date <= pd.Timestamp(EXPECTED_DEV_END)).all():
        raise ValueError("CV uses holdout or post-development dates")
    if not np.isfinite(p[["quantile_return", "var_return", "actual_return"]].to_numpy(float)).all():
        raise ValueError("Non-finite values")
    expected_var = np.maximum(0, -p.quantile_return.to_numpy(float))
    if not np.allclose(expected_var, p.var_return.to_numpy(float), rtol=0, atol=1e-12):
        raise ValueError("VaR sign/magnitude mismatch")
    flags = p.violation.astype(str).str.lower().map({"true": True, "false": False})
    if flags.isna().any() or not np.array_equal(flags.to_numpy(bool),
                                                  p.actual_return.to_numpy(float) < p.quantile_return.to_numpy(float)):
        raise ValueError("Violation rule mismatch")
    reference = features.set_index("target_date")
    observed = reference.reindex(p.target_date)
    if observed.isna().any().any():
        raise ValueError("Predictions missing target dates from source features")
    if not np.array_equal(p.as_of.to_numpy(), observed.as_of.to_numpy()):
        raise ValueError("Forecast origin does not match feature table")
    if not np.allclose(p.actual_return.to_numpy(float), observed.actual_return.to_numpy(float),
                       atol=1e-13, rtol=0):
        raise ValueError("Actual return not aligned with immutable feature table")
    for fold in folds:
        name = f"dev_{fold['fold']}"
        within = p[p.fold == name]
        if len(within) != 3 * fold["val_n"]:
            raise ValueError(f"Unexpected rows in {name}")
        if not fold["train_last"] < fold["validation_first"]:
            raise ValueError(f"Overlapping train/validation in {name}")
        for model in MODELS:
            piece = within[within.model == model]
            if len(piece) != fold["val_n"] or not piece.target_date.is_monotonic_increasing:
                raise ValueError(f"Invalid sequence in {name}/{model}")
            if (piece.target_date.min().strftime("%Y-%m-%d") != fold["validation_first"]
                    or piece.target_date.max().strftime("%Y-%m-%d") != fold["validation_last"]):
                raise ValueError(f"Fold boundary mismatch in {name}/{model}")
    if len(folds) != 5 or sum(x["val_n"] for x in folds) != 980:
        raise ValueError("Expected five expanding CV folds")


def run(root: Path, snapshot: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"Immutable Phase 3 baseline already exists: {output}")
    meta_file = root / "results/phase2/a/training_2026-11-06/metadata.json"
    train_dir = meta_file.parent
    feature_csv = root / "results/phase2/a/features_2026-11-06.csv"
    feature_meta_file = root / "results/phase2/a/features_2026-11-06_metadata.json"
    raw = snapshot / "data/phase2/portfolio_returns.csv"
    raw_meta_file = snapshot / "data/phase2/latest_manifest.json"
    for path in (meta_file, feature_csv, feature_meta_file, raw, raw_meta_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    meta = json.loads(meta_file.read_text(encoding="utf-8-sig"))
    feat_meta = json.loads(feature_meta_file.read_text(encoding="utf-8-sig"))
    raw_meta = json.loads(raw_meta_file.read_text(encoding="utf-8-sig"))
    if meta["training_cutoff"] != EXPECTED_DEV_END or meta["snapshot_cutoff"] != EXPECTED_CUTOFF:
        raise ValueError("Unexpected development/snapshot cutoff")
    if meta["selected_v2_on_cv"] != EXPECTED_MODEL or meta["best_overall_on_cv"] != EXPECTED_MODEL:
        raise ValueError("Phase 2 CV winner changed")
    if not (sha256(raw) == meta["source_sha256"] == feat_meta["input_sha256"] == raw_meta["portfolio_sha256"]):
        raise ValueError("Raw KBS snapshot SHA256 mismatch")
    if not (sha256(feature_csv) == meta["features_sha256"] == feat_meta["output_sha256"]):
        raise ValueError("Feature CSV SHA256 mismatch")
    if feat_meta["feature_count"] != 12 or len(feat_meta["feature_names"]) != 12:
        raise ValueError("Frozen 12-feature schema changed")
    for name in ("cv_predictions", "cv_metrics", "cv_ranking"):
        p = train_dir / f"{name}.csv"
        if sha256(p) != meta["artifacts"][name]["sha256"]:
            raise ValueError(f"Phase 2 artifact SHA256 mismatch: {name}")
    f = pd.read_csv(feature_csv, parse_dates=["as_of", "target_date"], float_precision="round_trip")
    if len(f) != 1647 or not f.target_date.is_monotonic_increasing or f.target_date.duplicated().any():
        raise ValueError("Feature table changed")
    if f.columns.tolist() != ["target_date", "as_of", *feat_meta["feature_names"], "actual_return"]:
        raise ValueError("Feature ordering/schema changed")
    if not (f.as_of < f.target_date).all():
        raise ValueError("Invalid feature as_of dates")
    p = pd.read_csv(train_dir / "cv_predictions.csv", parse_dates=["as_of", "target_date"],
                    float_precision="round_trip")
    assert_forecast_integrity(p, f, meta["five_folds"])
    published = pd.read_csv(train_dir / "cv_metrics.csv", float_precision="round_trip")
    rows = []
    for model in MODELS:
        for n in range(1, 6):
            part = p[(p.model == model) & (p.fold == f"dev_{n}")]
            metrics = evaluate(part)
            old = published[(published.model == model) & (published.fold.astype(str) == str(n))]
            if len(old) != 1:
                raise ValueError(f"Missing published CV metrics: {model}/fold {n}")
            for field, value in metrics.items():
                if not np.isclose(float(old.iloc[0][field]), float(value), atol=1e-12, rtol=1e-10):
                    raise ValueError(f"Incorrect archived CV metric: {model}, {n}, {field}")
            rows.append(dict(model=model, fold=n, **metrics,
                             coverage_gap_pp=100 * (metrics["violation_rate"] - ALPHA)))
    table = pd.DataFrame(rows)
    rank = (table.groupby("model").pinball_loss.mean().sort_values().rename("cv_pinball").reset_index())
    published_rank = pd.read_csv(train_dir / "cv_ranking.csv", float_precision="round_trip")
    if rank.model.tolist() != published_rank.model.tolist():
        raise ValueError("Published ranking order differs")
    if not np.allclose(rank.cv_pinball, published_rank.cv_pinball, atol=1e-12, rtol=1e-10):
        raise ValueError("Published CV loss differs")
    if rank.iloc[0].model != EXPECTED_MODEL:
        raise ValueError("CV winner changed; baseline cannot be locked")
    baseline = table[table.model == "gb_G04_7f_a"].sort_values("fold")
    c1 = table[table.model == EXPECTED_MODEL].sort_values("fold")
    win_count = int((c1.pinball_loss.to_numpy() < baseline.pinball_loss.to_numpy()).sum())
    report = {
        "stage": "phase3_baseline_frozen_cv_only", "owner": "A",
        "source": "KBS", "snapshot_cutoff": EXPECTED_CUTOFF,
        "training_cutoff": EXPECTED_DEV_END,
        "source_sha256": sha256(raw), "features_sha256": sha256(feature_csv),
        "phase2_cv_predictions_sha256": sha256(train_dir / "cv_predictions.csv"),
        "phase2_cv_metrics_sha256": sha256(train_dir / "cv_metrics.csv"),
        "phase2_cv_ranking_sha256": sha256(train_dir / "cv_ranking.csv"),
        "feature_count": 12, "feature_names": feat_meta["feature_names"],
        "selected_on_development_cv": EXPECTED_MODEL,
        "cv_mean_pinball": {str(r.model): float(r.cv_pinball) for r in rank.itertuples()},
        "gbv2_c1_wins_vs_7f_folds": win_count,
        "gbv2_c1_cv_violation_rate_by_fold": [float(x) for x in c1.violation_rate],
        "all_folds": meta["five_folds"],
        "validation_rule": "5 expanding CV folds, train target date strictly earlier than validation target",
        "selection_rule": "Phase3 features and hyperparameters must be selected on development only; no holdout tuning",
        "limitations": ["Old holdout already examined and overlaps Phase1 evaluation; not a fresh blind test",
                        "Phase3 initial baseline lock does not measure train-vs-validation overfit gap yet",
                        "No fresh independent B replication of 12-feature GBv2 at identical protocol yet"],
        "python": platform.python_version(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                           text=True).strip(),
    }
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "baseline_cv_diagnostics.csv"
    json_path = output / "baseline_lock.json"
    table.to_csv(csv_path, index=False, float_format="%.17g")
    report["diagnostics_sha256"] = sha256(csv_path)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SOURCE SHA256:", report["source_sha256"])
    print("FEATURE SHA256:", report["features_sha256"])
    print("CV MODELS:", len(MODELS), "FOLDS:", 5, "FORECAST ROWS:", len(p))
    print("CV RANKING (LOCKED):")
    print(rank.to_string(index=False))
    print("C1 FOLD WINS:", win_count, "/ 5")
    print("C1 VIOLATION RATES (%):", [round(100 * x, 2) for x in c1.violation_rate])
    print("OLD HOLDOUT USED FOR SELECTION: NO")
    print("OUTPUT:", output)
    print("STEP P3-01 BASELINE LOCK PASS")
    return json_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    run(ROOT, a.snapshot.resolve(), a.out.resolve())


if __name__ == "__main__":
    main()
