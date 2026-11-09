"""A-only exploratory 12/18/25-feature comparison: fixed C1 GB 5-fold CV.

Never reads Phase 2 holdout predictions/metrics. NEVER publishes live forecasts.
Performance on development CV is reused exploratory evidence, not an untouched test.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import GradientBoostingRegressor

from src.phase2.features_a import ALL_FEATURES
from src.phase3.features_a import FEATURES_18, FEATURES_25
from src.phase3.diagnostics_a import (
    sha256, diagnostics, expanding_fold_boundaries, assert_feature_frames,
    high_correlations, make_ranking,
)

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = "2026-11-06"
DEV_END = "2024-12-17"
SOURCE_SHA = "6389586ae8f1edbabb31760e0b555619186114b3f77d09212362fc2e1c40aadf"
# Exact source hashes verified in previous steps; these are the user-produced P3-02 export bytes.
EXPECTED = {
    18: "609926f3af7b2cea81fdd63d5a9e47050bee2128d81b11e83c03d7eb5b818288",
    25: "b45094fea0f8814d19b0505b26f8d6556c184190fa73dc6b953b14c1245874fa",
}
OLD_12_SHA = "14ac5dd4ce7dd6090420b9cee06c8ac40935c54691b4daf95d063b71a075eccc"
SIZES = (12, 18, 25)


def load_and_check(repo: Path, outputs: Path, baseline_lock: Path):
    """Authenticate prior outputs and return development-only tables."""
    old = repo / f"results/phase2/a/features_{CUTOFF}.csv"
    old_meta = repo / f"results/phase2/a/features_{CUTOFF}_metadata.json"
    train_dir = repo / f"results/phase2/a/training_{CUTOFF}"
    train_meta_path = train_dir / "metadata.json"
    ref_csv = train_dir / "cv_predictions.csv"
    if not all(p.is_file() for p in (old, old_meta, train_meta_path, ref_csv, baseline_lock)):
        raise FileNotFoundError("Missing Phase 2 baseline inputs")
    lock = json.loads(baseline_lock.read_text(encoding="utf-8-sig"))
    train_meta = json.loads(train_meta_path.read_text(encoding="utf-8-sig"))
    old_meta_data = json.loads(old_meta.read_text(encoding="utf-8-sig"))
    if (sha256(old) != OLD_12_SHA or sha256(old) != lock["features_sha256"] or
            sha256(old) != old_meta_data["output_sha256"] or
            train_meta["features_sha256"] != OLD_12_SHA or
            lock["source_sha256"] != SOURCE_SHA or train_meta["source_sha256"] != SOURCE_SHA or
            lock["training_cutoff"] != DEV_END or
            train_meta["training_cutoff"] != DEV_END or
            lock["selected_on_development_cv"] != "gbv2_12f_c1_a"):
        raise ValueError("Phase 2 baseline lock or hash mismatch")
    if (sha256(ref_csv) != train_meta["artifacts"]["cv_predictions"]["sha256"] or
            sha256(ref_csv) != lock["phase2_cv_predictions_sha256"]):
        raise ValueError("Phase 2 CV reference predictions changed")
    names_by_size = {12: list(ALL_FEATURES), 18: list(FEATURES_18), 25: list(FEATURES_25)}
    inputs = {12: old}
    frames = {12: pd.read_csv(old, parse_dates=["target_date", "as_of"], float_precision="round_trip")}
    for size in (18, 25):
        path = outputs / f"features_{size}_{CUTOFF}.csv"
        meta_file = outputs / f"features_{size}_{CUTOFF}_metadata.json"
        if not path.is_file() or not meta_file.is_file():
            raise FileNotFoundError(f"Missing Phase 3 {size}F export")
        meta = json.loads(meta_file.read_text(encoding="utf-8-sig"))
        if (sha256(path) != EXPECTED[size] or sha256(path) != meta["sha256"]
                or meta["source_sha256"] != SOURCE_SHA
                or meta["phase2_feature_sha256"] != OLD_12_SHA
                or meta["baseline_lock_sha256"] != sha256(baseline_lock)
                or meta["feature_count"] != size
                or meta["features"] != names_by_size[size]
                or meta["dev_end"] != DEV_END):
            raise ValueError(f"Phase 3 {size}F metadata/hash mismatch")
        inputs[size] = path
        frames[size] = pd.read_csv(path, parse_dates=["target_date", "as_of"], float_precision="round_trip")
    dev, n = assert_feature_frames(frames, names_by_size, DEV_END)
    if n != 1179:
        raise ValueError("Unexpected dev size")
    reference = pd.read_csv(ref_csv, parse_dates=["target_date", "as_of"], float_precision="round_trip")
    reference = reference[reference.model == "gbv2_12f_c1_a"].reset_index(drop=True)
    if len(reference) != 980:
        raise ValueError("Missing reference 12F CV predictions")
    c1 = dict(train_meta["params"]["gbv2_12f_c1_a"])
    if (c1 != dict(loss="quantile", alpha=.05, n_estimators=150,
                   learning_rate=.03, max_depth=2, min_samples_leaf=10,
                   subsample=1.0, random_state=42)):
        raise ValueError("C1 hyperparameters differ from frozen Phase 2")
    return dev, reference, c1, train_meta, inputs


def generate(repo: Path, features_dir: Path, baseline_lock: Path, output: Path) -> dict:
    output = output.resolve()
    output.relative_to(repo.resolve())
    if output.exists():
        raise FileExistsError("Phase 3 diagnostics already exist. Refusing to overwrite")
    dev, reference, config, train_meta, source_files = load_and_check(repo, features_dir, baseline_lock)
    folds = expanding_fold_boundaries(len(dev[12]))
    for i, (tr, va) in enumerate(folds):
        row = train_meta["five_folds"][i]
        d = dev[12]
        if (int(row["train_n"]) != len(tr) or int(row["val_n"]) != len(va)
                or row["train_last"] != d.target_date.iloc[tr[-1]].date().isoformat()
                or row["validation_first"] != d.target_date.iloc[va[0]].date().isoformat()
                or row["validation_last"] != d.target_date.iloc[va[-1]].date().isoformat()):
            raise ValueError("Training fold protocol changed")
    scores, predictions, importance = [], [], []
    maximum_12_diff = 0.0
    for i, (train_idx, valid_idx) in enumerate(folds, 1):
        for size in SIZES:
            frame = dev[size]
            X = frame.iloc[:, 2:-1]
            y = frame.actual_return.to_numpy(float)
            model = GradientBoostingRegressor(**config)
            model.fit(X.iloc[train_idx], y[train_idx])
            q_tr = model.predict(X.iloc[train_idx])
            q_va = model.predict(X.iloc[valid_idx])
            metric = diagnostics(y[train_idx], q_tr, y[valid_idx], q_va)
            scores.append({"feature_count": size, "fold": i, "train_n": len(train_idx),
                           "valid_n": len(valid_idx),
                           "train_end": frame.target_date.iloc[train_idx[-1]].date().isoformat(),
                           "valid_start": frame.target_date.iloc[valid_idx[0]].date().isoformat(),
                           "valid_end": frame.target_date.iloc[valid_idx[-1]].date().isoformat(),
                           **metric})
            if size == 12:
                ref_fold = reference[reference.fold == f"dev_{i}"].reset_index(drop=True)
                if (len(ref_fold) != len(valid_idx)
                        or not ref_fold.target_date.equals(frame.target_date.iloc[valid_idx].reset_index(drop=True))
                        or not ref_fold.as_of.equals(frame.as_of.iloc[valid_idx].reset_index(drop=True))):
                    raise ValueError("Published Phase 2 C1 fold alignment mismatch")
                np.testing.assert_allclose(ref_fold.actual_return.to_numpy(float), y[valid_idx], rtol=0, atol=1e-13)
                delta = float(np.max(np.abs(ref_fold.quantile_return.to_numpy(float) - q_va)))
                maximum_12_diff = max(maximum_12_diff, delta)
                if delta > 1e-9:
                    raise ValueError(f"Reproducing frozen C1 predictions failed on fold {i}: {delta}")
            predictions.extend({"feature_count":size, "fold":i,
                                "as_of":str(a.date()), "target_date":str(d.date()),
                                "quantile_return":float(q), "var_return":float(max(0, -q)),
                                "actual_return":float(actual), "violation":bool(actual < q)}
                               for a, d, q, actual in zip(frame.as_of.iloc[valid_idx], frame.target_date.iloc[valid_idx],
                                                           q_va, y[valid_idx]))
            for name, imp in zip(X.columns, model.feature_importances_):
                importance.append({"feature_count":size, "fold":i, "feature":name,
                                   "impurity_importance":float(imp)})
        print(f"FOLD {i}: {len(train_idx)} TRAIN / {len(valid_idx)} VALID / 12F 18F 25F PASS", flush=True)
    fold_df = pd.DataFrame(scores)
    ranking = make_ranking(fold_df)
    imp_df = pd.DataFrame(importance)
    preds_df = pd.DataFrame(predictions)
    flagged = high_correlations(dev[25], list(FEATURES_25))
    if (len(scores) != 15 or len(preds_df) != 2940 or len(imp_df) != 275
            or not (preds_df.as_of < preds_df.target_date).all()
            or preds_df.duplicated(["feature_count", "target_date"]).any()
            or len(ranking) != 3):
        raise RuntimeError("Unexpected Phase 3 diagnostics output size")
    original_cv_pinball = 0.0020802674466814082
    reproduced = float(ranking.loc[ranking.feature_count == 12, "cv_pinball"].iloc[0])
    if abs(original_cv_pinball - reproduced) > 1e-9:
        raise ValueError("Phase 2 12F CV score not reproduced")
    importances_order = imp_df.groupby(["feature_count", "feature"], as_index=False).impurity_importance.mean()
    best_count = int(ranking.iloc[0].feature_count)

    # Write only after all input and training checks pass.
    output.mkdir(parents=True, exist_ok=False)
    csvs = {"cv_fold_metrics": fold_df, "cv_predictions": preds_df,
            "cv_ranking": ranking, "high_feature_correlations": flagged,
            "feature_importances": imp_df, "feature_importances_mean": importances_order}
    outputs = {}
    for name, table in csvs.items():
        path = output / f"{name}.csv"
        table.to_csv(path, index=False, float_format="%.17g")
        outputs[name] = {"sha256": sha256(path), "rows": len(table)}

    figdir = output / "figures"
    figdir.mkdir()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for size in SIZES:
        d = fold_df[fold_df.feature_count == size]
        ax.plot(d.fold, d.val_pinball, marker="o", label=f"{size} features")
    ax.set(title="5-fold expanding CV - validation pinball loss", xlabel="Fold", ylabel="Pinball loss (alpha=0.05)")
    ax.set_xticks(range(1,6))
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "cv_pinball_by_fold.png", dpi=155)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    xx = np.arange(len(SIZES))
    train = [fold_df[fold_df.feature_count==k].train_pinball.mean() for k in SIZES]
    val = [fold_df[fold_df.feature_count==k].val_pinball.mean() for k in SIZES]
    ax.bar(xx - .17, train, .34, label="Train mean pinball")
    ax.bar(xx + .17, val, .34, label="Validation mean pinball")
    ax.set(xticks=xx, xticklabels=[str(k) for k in SIZES],
           xlabel="Number of features", ylabel="Pinball loss (alpha=0.05)",
           title="Train versus validation - diagnostic, not proof of overfit")
    ax.grid(axis="y", alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "train_validation_gap.png", dpi=155)
    plt.close(fig)
    for path in sorted(figdir.glob("*.png")):
        outputs[str(path.relative_to(output)).replace("\\", "/")] = {"sha256": sha256(path)}

    manifest = {
        "owner":"A", "experiment":"phase3_feature_count_screen_fixed_C1",
        "snapshot_cutoff":CUTOFF, "development_cutoff":DEV_END,
        "scope":"EXPLORATORY development CV ONLY; old holdout NOT used for fitting or selection",
        "old_phase2_holdout_untouched":True,
        "source_sha256":SOURCE_SHA, "baseline_lock_sha256":sha256(baseline_lock),
        "feature_csv_hashes":{str(k):sha256(v) for k, v in source_files.items()},
        "phase2_12f_cv_predictions_sha256":sha256(repo / f"results/phase2/a/training_{CUTOFF}/cv_predictions.csv"),
        "phase2_12f_reproduction_max_abs_diff":maximum_12_diff,
        "candidate_config_fixed_for_all_feature_counts":config,
        "five_folds":train_meta["five_folds"],
        "feature_counts":list(SIZES), "best_feature_count_exploratory_cv_only":best_count,
        "model_selection_warning":"Prior Phase 2 development CV already inspected. This is an exploratory repeated-selection result, not unbiased evidence of future performance. P3-04 should use nested/walk-forward or truly future data.",
        "mean_impurity_importance_warning":"Tree impurity importances are not causal and can be misleading with correlated features.",
        "correlation_flag_threshold":.95, "high_correlation_pairs":len(flagged),
        "artifacts":outputs,
        "versions":{"python":platform.python_version(),"numpy":np.__version__,
                    "pandas":pd.__version__,"sklearn":sklearn_version},
        "git_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nCV RANKING - EXPLORATORY (DO NOT USE OLD HOLDOUT):")
    print(ranking.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"\nREPRODUCED PHASE 2 C1 12F MAX ABS DIFFERENCE: {maximum_12_diff:.12g}")
    print("CORRELATED FEATURE PAIRS (ABS PEARSON >= .95):", len(flagged))
    print("P3-03 CV BEST FEATURE COUNT (EXPLORATORY):", best_count)
    print("CV PREDICTIONS:", len(preds_df), "FIGURES: 2")
    print("OLD HOLDOUT USED FOR TRAINING OR SELECTION: NO")
    print("OUTPUT:", output)
    print("STEP P3-03 DIAGNOSTICS PASS")
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--baseline-lock", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    generate(ROOT, args.features_dir.resolve(), args.baseline_lock.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
