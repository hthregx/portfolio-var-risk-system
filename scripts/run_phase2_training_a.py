"""A: GBv2 5-fold research train, no tuning on Phase1 final evaluation."""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.phase2.features_a import ALL_FEATURES, BASELINE, build_features_a

ROOT = Path(__file__).resolve().parents[1]
DEV_END = "2024-12-17"
ALPHA = 0.05
FROZEN = dict(loss="quantile", alpha=.05, n_estimators=100,
              learning_rate=.03, max_depth=2, min_samples_leaf=5,
              subsample=1.0, random_state=42)
CANDIDATES = {
    "gb_G04_7f_a": (list(BASELINE), FROZEN),
    "gbv2_12f_c1_a": (list(ALL_FEATURES), dict(FROZEN, n_estimators=150, min_samples_leaf=10)),
    "gbv2_12f_c2_a": (list(ALL_FEATURES), dict(FROZEN, n_estimators=120, learning_rate=.05)),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinball(y, q):
    y, q = np.asarray(y, float), np.asarray(q, float)
    if y.ndim != 1 or y.size == 0 or y.shape != q.shape:
        raise ValueError("Invalid y/q dimensions")
    if not np.isfinite(y).all() or not np.isfinite(q).all():
        raise ValueError("Nonfinite values")
    return np.where(y < q, (1 - ALPHA) * (q - y), ALPHA * (y - q))


def evaluation(frame):
    y = frame.actual_return.to_numpy(float)
    q = frame.quantile_return.to_numpy(float)
    return dict(n=len(frame), violations=int((y < q).sum()),
                violation_rate=float(np.mean(y < q)),
                pinball_loss=float(pinball(y, q).mean()),
                average_var=float(np.mean(np.maximum(0, -q))))


def split_dev_holdout(frame):
    if frame.target_date.isna().any() or frame.target_date.duplicated().any():
        raise ValueError("Invalid target dates")
    if not frame.target_date.is_monotonic_increasing:
        raise ValueError("Unsorted target dates")
    dev = frame[frame.target_date <= pd.Timestamp(DEV_END)].copy()
    hold = frame[frame.target_date > pd.Timestamp(DEV_END)].copy()
    if len(dev) < 360 or len(hold) < 50:
        raise ValueError("Insufficient dev/holdout")
    return dev, hold


def five_expanding_folds(n):
    if n < 360:
        raise ValueError("Insufficient development history")
    width = n // 6
    first = n - 5 * width
    return [(np.arange(0, first + i * width),
             np.arange(first + i * width, first + (i + 1) * width))
            for i in range(5)]


def predict(model_id, train, valid, fold):
    if train.target_date.max() >= valid.target_date.min():
        raise ValueError("Train/validation leakage")
    from sklearn.ensemble import GradientBoostingRegressor
    names, params = CANDIDATES[model_id]
    model = GradientBoostingRegressor(**params)
    model.fit(train[names], train.actual_return.to_numpy(float))
    q = model.predict(valid[names])
    y = valid.actual_return.to_numpy(float)
    if not np.isfinite(q).all():
        raise ValueError("Nonfinite prediction")
    return pd.DataFrame(dict(model=model_id, fold=fold,
                             as_of=valid.as_of.to_numpy(),
                             target_date=valid.target_date.to_numpy(),
                             quantile_return=q, var_return=np.maximum(0, -q),
                             actual_return=y, violation=y < q))


def validate_input(data_root, feature_csv, expected_sha):
    portfolio = data_root / "data/phase2/portfolio_returns.csv"
    manifest_file = data_root / "data/phase2/latest_manifest.json"
    feature_meta_file = feature_csv.with_name(feature_csv.stem + "_metadata.json")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    feature_meta = json.loads(feature_meta_file.read_text(encoding="utf-8"))
    if not (digest(portfolio) == manifest["portfolio_sha256"] == feature_meta["input_sha256"]):
        raise ValueError("Portfolio snapshot hash mismatch")
    if not (digest(feature_csv) == feature_meta["output_sha256"] == expected_sha):
        raise ValueError("Feature snapshot hash mismatch")
    if manifest["source"] != "KBS" or manifest["cutoff"] != "2026-11-06":
        raise ValueError("Unexpected snapshot cutoff/source")
    if tuple(feature_meta["feature_names"]) != tuple(ALL_FEATURES):
        raise ValueError("Feature schema changed")
    freeze = yaml.safe_load((ROOT / "configs/model_freeze.yaml").read_text(encoding="utf-8"))
    if freeze["evaluation"]["start_date"] != "2024-12-18":
        raise ValueError("Phase1 final evaluation start changed")
    raw = pd.read_csv(portfolio, parse_dates=["date"], float_precision="round_trip")
    frame = pd.read_csv(feature_csv, parse_dates=["target_date", "as_of"], float_precision="round_trip")
    if raw.date.isna().any() or raw.date.duplicated().any() or not raw.date.is_monotonic_increasing:
        raise ValueError("Invalid raw trading dates")
    if frame.columns.tolist() != ["target_date", "as_of", *ALL_FEATURES, "actual_return"]:
        raise ValueError("Feature CSV columns mismatch")
    if len(raw) != manifest["rows"] or len(frame) != len(raw) - 60:
        raise ValueError("Unexpected number of rows")
    if raw.date.iloc[-1].date().isoformat() != manifest["cutoff"]:
        raise ValueError("Raw data stale")
    if not (frame.as_of < frame.target_date).all():
        raise ValueError("as_of must precede target")
    if not np.array_equal(frame.target_date.to_numpy(), raw.date.iloc[60:].to_numpy()):
        raise ValueError("Target dates not aligned")
    if not np.array_equal(frame.as_of.to_numpy(), raw.date.iloc[59:-1].to_numpy()):
        raise ValueError("Forecast origin is not previous trading session")
    series = pd.Series(raw.portfolio_simple_return.to_numpy(float),
                       index=pd.DatetimeIndex(raw.date), name="portfolio_simple_return")
    original = build_features_a(series).iloc[60:]
    np.testing.assert_allclose(frame[list(ALL_FEATURES)].to_numpy(float),
                               original.to_numpy(float), atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(frame.actual_return.to_numpy(float),
                               raw.portfolio_simple_return.iloc[60:].to_numpy(float), atol=1e-14)
    if not np.isfinite(frame[list(ALL_FEATURES) + ["actual_return"]].to_numpy(float)).all():
        raise ValueError("Nonfinite feature data")
    return frame, manifest, digest(portfolio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--feature-csv", type=Path, required=True)
    ap.add_argument("--expected-sha", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    output = a.out.resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError("Output exists; refusing to overwrite experiment")
    frame, manifest, source_sha = validate_input(a.data_root, a.feature_csv, a.expected_sha)
    dev, holdout = split_dev_holdout(frame)
    print("DATA PASS", len(frame), "DEV", len(dev), "HOLDOUT", len(holdout), flush=True)
    rows, scores, folds = [], [], []
    for number, (train_ix, val_ix) in enumerate(five_expanding_folds(len(dev)), 1):
        tr, va = dev.iloc[train_ix], dev.iloc[val_ix]
        folds.append(dict(fold=number, train_n=len(tr), val_n=len(va),
                          train_last=str(tr.target_date.iloc[-1].date()),
                          validation_first=str(va.target_date.iloc[0].date()),
                          validation_last=str(va.target_date.iloc[-1].date())))
        for key in CANDIDATES:
            p = predict(key, tr, va, f"dev_{number}")
            rows.append(p)
            scores.append(dict(model=key, fold=number, **evaluation(p)))
        print("FOLD", number, "TRAIN", len(tr), "VALID", len(va), "PASS", flush=True)
    cv_preds = pd.concat(rows, ignore_index=True)
    cv_metrics = pd.DataFrame(scores)
    ranking = (cv_metrics.groupby("model").pinball_loss.mean()
               .sort_values(kind="stable").rename("cv_pinball").reset_index())
    ranking["n_features"] = ranking.model.map(lambda k: len(CANDIDATES[k][0]))
    chosen_v2 = ranking.loc[ranking.model.str.startswith("gbv2_")].iloc[0]["model"]
    best_cv = ranking.iloc[0]["model"]
    # CV determines the two holdout models, never the other way around.
    final_models = ["gb_G04_7f_a", chosen_v2]
    hold_preds = pd.concat([predict(key, dev, holdout, "holdout_fixed_fit")
                            for key in final_models], ignore_index=True)
    hold_metrics = pd.DataFrame([dict(model=key, **evaluation(hold_preds[hold_preds.model == key]))
                                 for key in final_models])
    output.mkdir(parents=True, exist_ok=False)
    files = dict(cv_predictions=cv_preds, cv_metrics=cv_metrics, cv_ranking=ranking,
                 holdout_predictions=hold_preds, holdout_metrics=hold_metrics)
    outputs = {}
    for key, table in files.items():
        path = output / f"{key}.csv"
        table.to_csv(path, index=False, date_format="%Y-%m-%d", float_format="%.17g")
        outputs[key] = dict(sha256=digest(path), rows=len(table))
    report = dict(owner="A", stage="development_only_not_live_forecast",
                  training_cutoff=DEV_END, snapshot_cutoff=manifest["cutoff"],
                  source_sha256=source_sha, features_sha256=a.expected_sha,
                  selected_v2_on_cv=chosen_v2, best_overall_on_cv=best_cv,
                  five_folds=folds, params={k:v[1] for k,v in CANDIDATES.items()},
                  features={k:v[0] for k,v in CANDIDATES.items()}, artifacts=outputs,
                  selection_rule="five-fold mean pinball; holdout never used to choose",
                  protocol="refit once per fold, lagged prior returns; holdout frozen fit on dev only",
                  caveat="retrospective holdout overlaps Phase1 published final evaluation; not pristine",
                  code_sha256=digest(__file__),
                  git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  python=platform.python_version())
    (output / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nCV RANKING (SELECT USING THIS ONLY):\n", ranking.to_string(index=False))
    print("\nHOLDOUT METRICS (AUDIT ONLY):\n", hold_metrics.to_string(index=False))
    print("\nOUTPUT:", output)
    print("A3 TRAINING PASS")


if __name__ == "__main__":
    main()
