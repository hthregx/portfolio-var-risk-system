"""B-owned reproduction of A's locked Phase 2 experiment, not live forecasts."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

from src.gb_return_features import build_return_features
from src.phase2.cv_b import expanding_splits
from src.phase2.ewma_b import ewma_quantiles

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = "6389586ae8f1edbabb31760e0b555619186114b3f77d09212362fc2e1c40aadf"
BASELINE = ["return_lag_1", "return_lag_2", "return_lag_5", "rolling_vol_5",
            "rolling_vol_20", "rolling_vol_60", "drawdown"]
EXTRA = ["momentum_5", "momentum_20", "downside_vol_20", "rolling_q05_60",
         "rolling_range_20"]
FEATURES = BASELINE + EXTRA
PARAMS = dict(loss="quantile", alpha=.05, n_estimators=100, learning_rate=.03,
              max_depth=2, min_samples_leaf=5, subsample=1., random_state=42)
MODELS = {
    "gb_G04_7f": (BASELINE, PARAMS),
    "gbv2_12f_c1": (FEATURES, dict(PARAMS, n_estimators=150, min_samples_leaf=10)),
    "gbv2_12f_c2": (FEATURES, dict(PARAMS, n_estimators=120, learning_rate=.05)),
}
FINAL_MODELS = ["historical_w250", "ewma_d094", "gb_G04_7f", "gbv2_12f_c1"]
FOLD_DATES = [
    ("2021-01-13", "2021-01-14", "2021-10-28"),
    ("2021-10-28", "2021-10-29", "2022-08-11"),
    ("2022-08-11", "2022-08-12", "2023-05-29"),
    ("2023-05-29", "2023-05-30", "2024-03-08"),
    ("2024-03-08", "2024-03-11", "2024-12-17"),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_reference_hash(path, expected):
    """Git normalizes A's Windows CSV line endings; allow only LF/CRLF changes.

    Raw input snapshot verification remains byte-exact in load_snapshot.
    """
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() == expected:
        return "byte_exact"
    lf = data.replace(b"\r\n", b"\n")
    for label, normalized in (("LF", lf), ("CRLF", lf.replace(b"\n", b"\r\n"))):
        if hashlib.sha256(normalized).hexdigest() == expected:
            return f"git_line_endings_restored_{label}"
    raise ValueError("Reference predictions hash mismatch")


def build_features_b(returns):
    """Share the canonical seven-feature builder; independently implement +5."""
    base = build_return_features(returns)
    if list(base.columns) != BASELINE:
        raise ValueError("Seven-feature contract changed")
    past = returns.shift(1)
    for window in (5, 20):
        base[f"momentum_{window}"] = np.expm1(np.log1p(past).rolling(window).sum())
    base["downside_vol_20"] = np.sqrt(past.clip(upper=0).pow(2).rolling(20).mean())
    base["rolling_q05_60"] = past.rolling(60).quantile(.05)
    base["rolling_range_20"] = past.rolling(20).max() - past.rolling(20).min()
    return base[FEATURES]


def load_snapshot(portfolio_path, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    if sha256(portfolio_path) != SOURCE_SHA or manifest.get("portfolio_sha256") != SOURCE_SHA:
        raise ValueError("Snapshot hash differs from locked A snapshot")
    if (manifest.get("source") != "KBS" or manifest.get("cutoff") != "2026-11-06"
            or manifest.get("portfolio_last_date") != "2026-11-06" or manifest.get("rows") != 1707):
        raise ValueError("Snapshot metadata differs from locked A contract")
    raw = pd.read_csv(portfolio_path, parse_dates=["date"], float_precision="round_trip")
    dates = pd.DatetimeIndex(raw.date)
    if (len(raw) != 1707 or dates.hasnans or dates.has_duplicates
            or not dates.is_monotonic_increasing or str(dates[0].date()) != "2020-01-03"
            or str(dates[-1].date()) != "2026-11-06"):
        raise ValueError("Unexpected source dates/rows")
    values = raw.portfolio_simple_return.to_numpy(float)
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Invalid simple returns")
    series = pd.Series(values, index=dates, name="portfolio_simple_return")
    features = build_features_b(series)
    frame = features.iloc[60:].copy()
    if len(frame) != 1647 or not np.isfinite(frame.to_numpy(float)).all():
        raise ValueError("Unexpected feature warmup/nonfinite values")
    frame.insert(0, "as_of", dates[59:-1])
    frame.insert(0, "target_date", dates[60:])
    frame["actual_return"] = values[60:]
    return series, frame.reset_index(drop=True), manifest


def locked_split(frame):
    dev = frame.loc[frame.target_date <= "2024-12-17"].reset_index(drop=True)
    hold = frame.loc[frame.target_date > "2024-12-17"].reset_index(drop=True)
    if len(dev) != 1179 or len(hold) != 468:
        raise ValueError("Development/holdout sizes differ from A")
    folds = expanding_splits(dev.target_date, n_splits=5)
    records = []
    for i, (train, valid) in enumerate(folds):
        observed = tuple(str(x.date()) for x in (
            dev.target_date.iloc[train[-1]], dev.target_date.iloc[valid[0]],
            dev.target_date.iloc[valid[-1]]))
        if observed != FOLD_DATES[i] or len(valid) != 196 or len(train) != 199 + i * 196:
            raise ValueError("CV boundaries differ from locked A folds")
        records.append(dict(fold=i + 1, train_n=len(train), val_n=len(valid),
                            train_last=observed[0], validation_first=observed[1],
                            validation_last=observed[2]))
    return dev, hold, folds, records


def prediction_frame(model, fold, valid, quantiles):
    q = np.asarray(quantiles, float)
    if q.shape != (len(valid),) or not np.isfinite(q).all():
        raise ValueError("Invalid forecasts")
    y = valid.actual_return.to_numpy(float)
    return pd.DataFrame(dict(model=model, fold=fold, as_of=valid.as_of.to_numpy(),
                             target_date=valid.target_date.to_numpy(), quantile_return=q,
                             var_return=np.maximum(0, -q), actual_return=y, violation=y < q))


def fit_predict(model_id, train, valid, fold):
    if train.target_date.max() >= valid.target_date.min():
        raise ValueError("Training must end before validation/holdout")
    names, params = MODELS[model_id]
    model = GradientBoostingRegressor(**params)
    model.fit(train[names], train.actual_return.to_numpy(float))
    return prediction_frame(model_id, fold, valid, model.predict(valid[names]))


def baseline_quantiles(series):
    """HS/EWMA retain all raw prior returns, including feature warmup rows."""
    r = series.to_numpy(float)
    hs = [np.quantile(r[i-250:i], .05, method="linear") for i in range(250, len(r))]
    ewma = ewma_quantiles(r, warmup=250)
    return {"historical_w250": pd.Series(hs, index=series.index[250:]),
            "ewma_d094": pd.Series(ewma, index=series.index[250:])}


def summarize(predictions):
    rows = []
    for (model, fold), part in predictions.groupby(["model", "fold"], sort=False):
        error = part.actual_return.to_numpy(float) - part.quantile_return.to_numpy(float)
        rows.append(dict(model=model, fold=fold, n=len(part),
                         violations=int((error < 0).sum()), violation_rate=float((error < 0).mean()),
                         pinball_loss=float(np.maximum(.05 * error, -.95 * error).mean()),
                         average_var=float(part.var_return.mean())))
    return pd.DataFrame(rows)


def compare_reference(observed, reference, tolerance=1e-9):
    """Require every A key/date/actual to match before comparing quantiles."""
    ref = reference.copy()
    ref["model"] = ref.model.str.removesuffix("_a")
    if not set(ref.model).issubset(MODELS):
        raise ValueError("Unknown reference model")
    part = observed.loc[observed.model.isin(ref.model.unique())].copy()
    keys = ["model", "target_date"]
    for df in (part, ref):
        for col in ("as_of", "target_date"):
            df[col] = pd.to_datetime(df[col], errors="raise")
        if df[keys].duplicated().any():
            raise ValueError("Duplicate reference/comparison target")
    part = part.sort_values(keys).reset_index(drop=True)
    ref = ref.sort_values(keys).reset_index(drop=True)
    if len(part) != len(ref) or not part[keys].equals(ref[keys]):
        raise ValueError("A/B target universe differs")
    if not part.as_of.equals(ref.as_of) or not part.fold.equals(ref.fold):
        raise ValueError("A/B as_of or folds differ")
    if not np.allclose(part.actual_return, ref.actual_return, atol=1e-13, rtol=0):
        raise ValueError("A/B realized returns differ")
    records = []
    for model in ref.model.unique():
        mask = ref.model == model
        difference = np.abs(part.loc[mask, "quantile_return"].to_numpy(float)
                            - ref.loc[mask, "quantile_return"].to_numpy(float))
        records.append(dict(model=model, n=int(mask.sum()), max_abs_difference=float(difference.max()),
                            tolerance=tolerance, status="PASS" if difference.max() <= tolerance else "MISMATCH"))
    return records


def run(portfolio_path, manifest_path, output, reference_dir=None, *, cv_only=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Output already exists; choose a new folder")
    series, frame, manifest = load_snapshot(portfolio_path, manifest_path)
    dev, hold, folds, records = locked_split(frame)
    baselines = baseline_quantiles(series.loc[:"2024-12-17"] if cv_only else series)
    cv_parts = []
    for number, (tr, va) in enumerate(folds, 1):
        valid = dev.iloc[va]
        for name, q in baselines.items():
            cv_parts.append(prediction_frame(name, f"dev_{number}", valid, q.loc[valid.target_date]))
        for name in MODELS:
            cv_parts.append(fit_predict(name, dev.iloc[tr], valid, f"dev_{number}"))
        print(f"Fold {number}: train={len(tr)}, validation={len(va)}", flush=True)
    cv = pd.concat(cv_parts, ignore_index=True)
    cv_metrics = summarize(cv)
    tables = dict(features=dev if cv_only else frame,
                  baseline_cv=cv.loc[cv.model.isin(FINAL_MODELS[:3])],
                  cv_predictions=cv, cv_metrics=cv_metrics)
    comparisons = [("cv", cv, set(MODELS))]
    if not cv_only:
        hold_parts = [prediction_frame(name, "holdout_fixed_fit", hold, q.loc[hold.target_date])
                      for name, q in baselines.items()]
        hold_parts += [fit_predict(name, dev, hold, "holdout_fixed_fit") for name in FINAL_MODELS[2:]]
        final = pd.concat(hold_parts, ignore_index=True)
        tables.update(holdout_predictions=final, holdout_metrics=summarize(final),
                      gbv2_retrain=final.loc[final.model == "gbv2_12f_c1"])
        comparisons.append(("holdout", final, set(FINAL_MODELS[2:])))
    # C1 was selected by A before this reproduction; never reselect using holdout.
    crosscheck = []
    reference_hash_checks = {}
    if reference_dir is not None:
        refdir = Path(reference_dir)
        meta = json.loads((refdir / "metadata.json").read_text(encoding="utf-8-sig"))
        if (meta.get("source_sha256") != SOURCE_SHA or meta.get("training_cutoff") != "2024-12-17"
                or meta.get("selected_v2_on_cv") != "gbv2_12f_c1_a"
                or meta.get("five_folds") != records):
            raise ValueError("Reference metadata differs from locked experiment")
        for stage, observed, expected_names in comparisons:
            path = refdir / f"{stage}_predictions.csv"
            artifact = meta["artifacts"][f"{stage}_predictions"]
            reference_hash_checks[path.name] = verify_reference_hash(path, artifact["sha256"])
            reference = pd.read_csv(path, float_precision="round_trip")
            if set(reference.model.str.removesuffix("_a")) != expected_names:
                raise ValueError("Reference model set differs from locked experiment")
            crosscheck.extend(dict(stage=stage, **r) for r in compare_reference(observed, reference))
    status = ("NOT_RUN" if not crosscheck else
              "PASS" if all(r["status"] == "PASS" for r in crosscheck) else "MISMATCH")
    metadata = dict(owner="B", source_sha256=SOURCE_SHA, snapshot_cutoff=manifest["cutoff"],
                    training_cutoff="2024-12-17", input_rows=len(series), feature_rows=len(frame),
                    development_rows=len(dev), holdout_rows=len(hold), features=FEATURES,
                    run_mode="cv_only" if cv_only else "cv_and_holdout", holdout_evaluated=not cv_only,
                    five_folds=records, models={k: dict(features=v[0], parameters=v[1]) for k,v in MODELS.items()},
                    selected_v2="gbv2_12f_c1", selection="reproduce_A_locked_C1_no_new_selection",
                    gb_refit="once_per_fold" if cv_only else "once_per_fold; once_on_development_before_holdout",
                    hs_ewma_protocol="sequential_one_step_using_all_available_prior_returns",
                    baseline_feature_builder="shared src.gb_return_features; additional five implemented by B",
                    reference_check=status, crosscheck=crosscheck, reference_hash_checks=reference_hash_checks,
                    caveat="Retrospective holdout overlaps Phase1 published evaluation; not pristine or live forecasts",
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    scikit_learn=sklearn.__version__, runner_sha256=sha256(__file__),
                    dependency_code_sha256={name: sha256(ROOT / name) for name in (
                        "src/gb_return_features.py", "src/phase2/cv_b.py", "src/phase2/ewma_b.py")},
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    # Publish only complete runs. Never overwrite previous outputs or source data.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".aligned-b-", dir=output.parent) as tmp:
        folder = Path(tmp) / "complete"
        folder.mkdir()
        for name, table in tables.items():
            table.to_csv(folder / f"{name}.csv", index=False, date_format="%Y-%m-%d", float_format="%.17g")
        metadata["artifacts"] = {p.name: dict(sha256=sha256(p), rows=len(pd.read_csv(p)))
                                 for p in sorted(folder.glob("*.csv"))}
        (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        folder.rename(output)
    print((cv_metrics if cv_only else tables["holdout_metrics"]).to_string(index=False))
    print(f"Reference check: {status}; outputs: {output}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "results/phase2/b/aligned_2026-11-06")
    parser.add_argument("--reference-dir", type=Path, help="A training_2026-11-06 folder (optional)")
    parser.add_argument("--cv-only", action="store_true", help="Day 4: evaluate development CV only; do not predict/evaluate holdout")
    args = parser.parse_args()
    result = run(args.portfolio, args.manifest, args.out, args.reference_dir, cv_only=args.cv_only)
    if result["reference_check"] == "MISMATCH":
        raise SystemExit(2)


if __name__ == "__main__":
    main()