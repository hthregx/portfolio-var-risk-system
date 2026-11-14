"""Independent A-side backtesting, QC, and essential VaR figures.

Consumes A3 predictions without retraining, and never uses holdout to
reselect candidates. Won't write files in B or frozen Phase-1 directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.phase2.metrics_a import (
    ALPHA, backtest_one, pinball, validate_paired, validate_predictions,
)

ROOT = Path(__file__).resolve().parents[1]
DEV_END = pd.Timestamp("2024-12-17")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_metadata(training: Path, names: list[str]) -> dict:
    meta_file = training / "metadata.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if meta.get("training_cutoff") != "2024-12-17":
        raise RuntimeError("Wrong development cutoff")
    if meta.get("snapshot_cutoff") != "2026-11-06":
        raise RuntimeError("Unexpected A3 data cutoff")
    if "not pristine" not in meta.get("caveat", "").lower():
        raise RuntimeError("Missing non-pristine holdout disclaimer")
    for key in names:
        fp = training / (key + ".csv")
        if sha256(fp) != meta["artifacts"][key]["sha256"]:
            raise RuntimeError("Input CSV SHA256 mismatch: " + key)
        if len(pd.read_csv(fp)) != meta["artifacts"][key]["rows"]:
            raise RuntimeError("Input CSV row count mismatch: " + key)
    return meta


def validate_experiment(training: Path, snapshot: Path):
    names = ["cv_predictions", "cv_metrics", "cv_ranking",
             "holdout_predictions", "holdout_metrics"]
    meta = require_metadata(training, names)
    portfolio = snapshot / "data/phase2/portfolio_returns.csv"
    manifest = json.loads(
        (snapshot / "data/phase2/latest_manifest.json").read_text(encoding="utf-8")
    )
    if sha256(portfolio) != manifest["portfolio_sha256"] or sha256(portfolio) != meta["source_sha256"]:
        raise RuntimeError("A3 source snapshot has changed")
    cv = validate_predictions(pd.read_csv(training / "cv_predictions.csv"))
    ho = validate_predictions(pd.read_csv(training / "holdout_predictions.csv"))
    cvs = pd.read_csv(training / "cv_metrics.csv")
    rank = pd.read_csv(training / "cv_ranking.csv")
    hos = pd.read_csv(training / "holdout_metrics.csv")
    if (cv.target_date > DEV_END).any() or (ho.target_date <= DEV_END).any():
        raise RuntimeError("Development and holdout overlap")
    if ho.model.nunique() != 2 or cv.model.nunique() != 3:
        raise RuntimeError("Unexpected number of candidate models")
    if cv.fold.nunique() != 5:
        raise RuntimeError("Unexpected CV fold count")
    if set(ho.model) != {"gb_G04_7f_a", meta["selected_v2_on_cv"]}:
        raise RuntimeError("Holdout model selection differs from CV")
    if rank.iloc[0].model != meta["best_overall_on_cv"]:
        raise RuntimeError("Ranking not consistent with A3 metadata")
    v2 = rank.loc[rank.model.str.startswith("gbv2_"), "model"].iloc[0]
    if v2 != meta["selected_v2_on_cv"]:
        raise RuntimeError("V2 selection not consistent with CV")
    groups = {k: f for k, f in ho.groupby("model")}
    ref, selected = validate_paired(groups["gb_G04_7f_a"], groups[v2])
    if not (ref["fold"] == "holdout_fixed_fit").all():
        raise RuntimeError("Unexpected holdout fitting protocol")
    if not (selected["fold"] == "holdout_fixed_fit").all():
        raise RuntimeError("Unexpected V2 fitting protocol")
    cv_baseline = cv[cv.model == "gb_G04_7f_a"]
    for key in cv.model.unique():
        m = cv[cv.model == key].sort_values("target_date")
        validate_paired(cv_baseline, m)
        if not m.target_date.reset_index(drop=True).equals(
            cv[cv.model == rank.iloc[0].model].sort_values("target_date").target_date.reset_index(drop=True)
        ):
            raise RuntimeError("CV candidate target dates do not align")
        for fold, f in m.groupby("fold"):
            if len(f) == 0:
                raise RuntimeError("Empty validation fold")
            fold_number = int(str(fold).rsplit("_", 1)[-1])
            row = cvs[(cvs.model == key) & (cvs.fold == fold_number)]
            if len(row) != 1:
                raise RuntimeError("Missing CV metric per model/fold")
            own_loss = float(pinball(f.actual_return, f.quantile_return).mean())
            if not np.isclose(own_loss, row.iloc[0].pinball_loss, atol=1e-12):
                raise RuntimeError("CV metric inconsistent with raw predictions")
    for key, f in groups.items():
        row = hos[hos.model == key]
        if len(row) != 1:
            raise RuntimeError("Missing holdout metric")
        checked = backtest_one(f)
        for field, target_field in [("n", "n"), ("violations", "violations"),
                                    ("violation_rate", "violation_rate"),
                                    ("mean_pinball_loss", "pinball_loss"),
                                    ("mean_var", "average_var")]:
            if not np.isclose(checked[field], row.iloc[0][target_field], atol=1e-12):
                raise RuntimeError("Holdout metric mismatch: " + field)
    return meta, cv, ho, cvs, rank, ref, selected


def generate_essential_figures(tmp: Path, cv: pd.DataFrame,
                               ref: pd.DataFrame, v2: pd.DataFrame,
                               fold_stats: pd.DataFrame, hold_stats: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = tmp / "figures"
    figs.mkdir()
    selected_id = v2.model.iloc[0]

    plt.figure(figsize=(13, 5))
    plt.plot(v2.target_date, 100 * v2.actual_return, linewidth=.9, label="Observed portfolio return")
    plt.plot(v2.target_date, 100 * v2.quantile_return, linewidth=1.2, label="GBv2 predicted 5% quantile")
    flags = v2.actual_return < v2.quantile_return
    plt.scatter(v2.loc[flags, "target_date"], 100 * v2.loc[flags, "actual_return"],
                s=15, color="tab:red", label="VaR exceptions", zorder=4)
    plt.ylabel("Daily simple return (%)")
    plt.xlabel("Target trading date")
    plt.title("Observed returns vs predicted lower-tail quantile (GBv2)")
    plt.legend()
    plt.grid(alpha=.2)
    plt.tight_layout()
    plt.savefig(figs / "actual_vs_quantile.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 4.5))
    flags = (v2.actual_return < v2.quantile_return).astype(int).to_numpy()
    plt.step(v2.target_date, np.cumsum(flags), where="post", label="Actual cumulative violations")
    plt.plot(v2.target_date, ALPHA * np.arange(1, len(v2) + 1), linestyle="--",
             label="Nominal expected count: 5% of sessions")
    plt.xlabel("Target trading date")
    plt.ylabel("Cumulative exception count")
    plt.title("VaR 95% cumulative exceptions (GBv2)")
    plt.legend()
    plt.grid(alpha=.2)
    plt.tight_layout()
    plt.savefig(figs / "cumulative_exceptions.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    pivot = fold_stats.pivot(index="fold", columns="model", values="pinball_loss")
    for model in pivot.columns:
        plt.plot(pivot.index, pivot[model], marker="o", label=model)
    plt.xlabel("Expanding time-series CV fold")
    plt.ylabel("Mean 5% pinball loss")
    plt.title("Development CV stability, no holdout used")
    plt.xticks(pivot.index)
    plt.legend(fontsize=8)
    plt.grid(alpha=.2)
    plt.tight_layout()
    plt.savefig(figs / "cv_fold_pinball.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    plt.bar(hold_stats.model, hold_stats.mean_pinball_loss)
    plt.ylabel("Mean 5% pinball loss")
    plt.title("Retrospective holdout loss (not model selection)")
    plt.xticks(rotation=10)
    plt.grid(axis="y", alpha=.2)
    plt.tight_layout()
    plt.savefig(figs / "holdout_pinball_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 4.5))
    for f in (ref, v2):
        rolling = f.violation.astype(float).rolling(60, min_periods=60).mean()
        plt.plot(f.target_date, 100 * rolling, label=f.model.iloc[0])
    plt.axhline(ALPHA * 100, linestyle="--", color="black", label="Nominal 5%")
    plt.xlabel("Target trading date")
    plt.ylabel("Trailing 60-trading-session violation rate (%)")
    plt.title("VaR exception-rate dynamics")
    plt.legend(fontsize=8)
    plt.grid(alpha=.2)
    plt.tight_layout()
    plt.savefig(figs / "rolling_60d_violation_rate.png", dpi=160)
    plt.close()

    return figs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-dir", required=True, type=Path)
    ap.add_argument("--snapshot-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()
    training = a.training_dir.resolve()
    destination = a.out_dir.resolve()
    destination.relative_to(ROOT)  # prevent output to arbitrary folder
    if destination.exists():
        raise FileExistsError("Backtesting output already exists; refusing overwrite")

    meta, cv, ho, cvs, ranking, ref, selected = validate_experiment(
        training, a.snapshot_root.resolve())
    stats = pd.DataFrame([backtest_one(f) for _, f in ho.groupby("model")])
    fold_stats = pd.DataFrame([
        {"model": model, "fold": fold, "n": len(f),
         "violations": int(f.violation.sum()),
         "violation_rate": float(f.violation.mean()),
         "pinball_loss": float(pinball(f.actual_return, f.quantile_return).mean())}
        for (model, fold), f in cv.groupby(["model", "fold"])
    ])

    paired = validate_paired(ref, selected)
    base_loss = pinball(paired[0].actual_return, paired[0].quantile_return)
    new_loss = pinball(paired[1].actual_return, paired[1].quantile_return)
    comparison = pd.DataFrame([{
        "baseline": ref.model.iloc[0], "candidate": selected.model.iloc[0],
        "n_common": len(ref), "baseline_loss": float(base_loss.mean()),
        "candidate_loss": float(new_loss.mean()),
        "paired_loss_improvement": float(base_loss.mean() - new_loss.mean()),
        "relative_loss_improvement_pct": float(100 * (base_loss.mean() - new_loss.mean()) / base_loss.mean()),
        "baseline_violations": int(ref.violation.sum()),
        "candidate_violations": int(selected.violation.sum()),
        "equal_target_dates": True,
        "equal_realized_returns": True,
        "scope": "retrospective_holdout_not_pristine",
    }])
    events = ho.loc[ho.violation.astype(bool)].sort_values(["model", "target_date"])
    if len(events) != int(stats.violations.sum()):
        raise RuntimeError("Exception count mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="a4_tmp_", dir=destination.parent) as td:
        tmp = Path(td)
        tables = {"holdout_backtests.csv": stats,
                  "cv_fold_metrics_verified.csv": fold_stats,
                  "holdout_paired_comparison.csv": comparison,
                  "holdout_exceptions.csv": events}
        for name, tbl in tables.items():
            tbl.to_csv(tmp / name, index=False, date_format="%Y-%m-%d", float_format="%.17g")
        figs = generate_essential_figures(tmp, cv, ref, selected, fold_stats, stats)
        output_files = [*tables.keys(), *[str(p.relative_to(tmp)) for p in figs.iterdir()]]
        manifest = {
            "owner": "A", "step": "A4_backtesting_independent",
            "source_training_metadata_sha256": sha256(training / "metadata.json"),
            "source_data_sha256": meta["source_sha256"],
            "development_end": "2024-12-17", "holdout_start": str(ref.target_date.min().date()),
            "holdout_end": str(ref.target_date.max().date()),
            "selected_v2_on_cv": meta["selected_v2_on_cv"],
            "alpha": ALPHA,
            "violation_rule": "actual_return < quantile_return",
            "var_return_rule": "max(0, -quantile_return)",
            "tests": "Kupiec UC chi-square(1) and Christoffersen IND chi-square(1); CC chi-square(2); asymptotic diagnostics",
            "evaluation_limit": "Historical ex-post, holdout overlaps published Phase 1 evaluation; not pristine independent test and not live forecast",
            "protocol_caveat": "A-side G04 retains frozen hyperparameters but uses fold-level and fixed-holdout fits, unlike Phase 1 canonical expanding-refit-each-target predictions",
            "do_not_use_for_selection": True,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": {name: sha256(tmp / name) for name in sorted(output_files)},
        }
        (tmp / "a4_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        shutil.move(str(tmp), str(destination))

    print("\nINDEPENDENT HOLDOUT BACKTEST:")
    print(stats[["model", "n", "violations", "violation_rate", "mean_pinball_loss", "p_uc", "p_ind", "p_cc"]].to_string(index=False))
    print("\nPAIRED PINBALL COMPARISON:")
    print(comparison[["baseline", "candidate", "n_common", "relative_loss_improvement_pct"]].to_string(index=False))
    print("\nEXCEPTION ROWS:", len(events))
    print("FIGURES:", len(list((destination / "figures").glob("*.png"))))
    print("OUTPUT:", destination)
    print("A4 BACKTESTING PASS")


if __name__ == "__main__":
    main()
