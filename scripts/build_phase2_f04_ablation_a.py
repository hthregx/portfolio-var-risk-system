import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path.cwd()
BASE = ROOT / "results/phase2/a"
TRAIN = BASE / "training_2026-11-06"
OUT = ROOT / "figures/phase2/a"
ALPHA = 0.05

FEATURES_7 = [
    "return_lag_1", "return_lag_2", "return_lag_5",
    "rolling_vol_5", "rolling_vol_20",
    "rolling_vol_60", "drawdown"
]

FEATURES_12 = FEATURES_7 + [
    "momentum_5", "momentum_20",
    "downside_vol_20", "rolling_q05_60",
    "rolling_range_20"
]

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pinball(y, q):
    return float(np.mean(
        np.maximum(ALPHA * (y - q),
                   (ALPHA - 1) * (y - q))
    ))

def main():
    meta_file = TRAIN / "metadata.json"
    feature_file = BASE / "features_2026-11-06.csv"
    prediction_file = TRAIN / "cv_predictions.csv"

    meta = json.loads(meta_file.read_text(encoding="utf-8-sig"))

    if meta["features_sha256"] != sha256(feature_file):
        raise RuntimeError("Feature source hash mismatch")

    if meta["training_cutoff"] != "2024-12-17":
        raise RuntimeError("Development cutoff changed")

    if meta["source_sha256"] != (
        "6389586ae8f1edbabb31760e0b555619186114b3f77d09212362fc2e1c40aadf"
    ):
        raise RuntimeError("KBS source mismatch")

    params = meta["params"]["gbv2_12f_c1_a"]

    if meta["features"]["gbv2_12f_c1_a"] != FEATURES_12:
        raise RuntimeError("C1 feature order changed")

    frame = pd.read_csv(
        feature_file,
        parse_dates=["target_date", "as_of"],
        float_precision="round_trip"
    )

    if len(frame) != 1647:
        raise RuntimeError("Unexpected feature rows")

    dev = frame.loc[
        frame["target_date"] <= pd.Timestamp("2024-12-17")
    ].reset_index(drop=True)

    if len(dev) != 1179:
        raise RuntimeError("Unexpected development rows")

    saved = pd.read_csv(
        prediction_file,
        parse_dates=["target_date"],
        float_precision="round_trip"
    )

    saved = saved.loc[
        saved["model"] == "gbv2_12f_c1_a"
    ]

    if len(saved) != 980:
        raise RuntimeError("Unexpected saved CV prediction count")

    rows = []
    max_reproduction_diff = 0.0

    for index, fold in enumerate(meta["five_folds"], start=1):
        n_train = int(fold["train_n"])
        n_valid = int(fold["val_n"])

        train = dev.iloc[:n_train]
        valid = dev.iloc[n_train:n_train + n_valid]

        if len(valid) != 196:
            raise RuntimeError("Validation fold size mismatch")

        if (
            train["target_date"].iloc[-1].strftime("%Y-%m-%d")
            != fold["train_last"]
        ):
            raise RuntimeError("Train date mismatch")

        if (
            valid["target_date"].iloc[0].strftime("%Y-%m-%d")
            != fold["validation_first"]
        ):
            raise RuntimeError("Validation start mismatch")

        if (
            valid["target_date"].iloc[-1].strftime("%Y-%m-%d")
            != fold["validation_last"]
        ):
            raise RuntimeError("Validation end mismatch")

        if not train["target_date"].max() < valid["target_date"].min():
            raise RuntimeError("Temporal leakage")

        y = valid["actual_return"].to_numpy(float)

        for name, columns in (
            ("C1_7_features", FEATURES_7),
            ("C1_12_features", FEATURES_12)
        ):
            model = GradientBoostingRegressor(**params)
            model.fit(
                train[columns],
                train["actual_return"].to_numpy(float)
            )
            q = model.predict(valid[columns])

            if not np.isfinite(q).all():
                raise RuntimeError("Nonfinite forecast")

            if name == "C1_12_features":
                old = saved.loc[
                    saved["fold"] == f"dev_{index}"
                ].sort_values("target_date")

                if len(old) != len(valid):
                    raise RuntimeError("Saved fold length mismatch")

                if not np.array_equal(
                    old["target_date"].to_numpy(),
                    valid["target_date"].to_numpy()
                ):
                    raise RuntimeError("Saved target dates differ")

                error = float(np.max(np.abs(
                    q - old["quantile_return"].to_numpy(float)
                )))

                if error > 1e-9:
                    raise RuntimeError(
                        f"C1 reproduction mismatch: {error}"
                    )

                max_reproduction_diff = max(
                    max_reproduction_diff, error
                )

            rows.append({
                "fold": index,
                "model": name,
                "n_features": len(columns),
                "train_n": n_train,
                "validation_n": len(valid),
                "pinball_loss": pinball(y, q),
                "violations": int(np.sum(y < q)),
                "violation_rate": float(np.mean(y < q)),
            })

        print(f"FOLD {index}: BOTH MODELS PASS")

    scores = pd.DataFrame(rows)

    wide = scores.pivot(
        index="fold", columns="model",
        values="pinball_loss"
    )

    loss7 = wide["C1_7_features"]
    loss12 = wide["C1_12_features"]
    gain = loss7 - loss12

    mean7 = float(loss7.mean())
    mean12 = float(loss12.mean())
    delta = mean7 - mean12

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 8),
        gridspec_kw={"height_ratios": [1.5, 1]}
    )

    x = np.arange(5)
    width = 0.36

    axes[0].bar(
        x - width/2, loss7.to_numpy()*1000,
        width, label="7 features - C1"
    )
    axes[0].bar(
        x + width/2, loss12.to_numpy()*1000,
        width, label="12 features - C1"
    )

    axes[0].set_xticks(x, [f"Fold {i}" for i in range(1, 6)])
    axes[0].set_ylabel("Pinball loss x 1000")
    axes[0].set_title("Same C1 hyperparameters and expanding CV")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(
        x, gain.to_numpy()*1000
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, [f"Fold {i}" for i in range(1, 6)])
    axes[1].set_ylabel("Loss(7f) - Loss(12f) x 1000")
    axes[1].set_title(
        "Positive means 12 features perform better"
    )
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle(
        "F04 | Controlled feature ablation - GBv2 C1",
        fontsize=15
    )
    fig.tight_layout()

    png = OUT / "F04.png"
    csv = OUT / "F04_fold_metrics.csv"
    metadata_file = OUT / "F04_metadata.json"

    for path in (png, csv, metadata_file):
        if path.exists():
            raise FileExistsError(path)

    OUT.mkdir(parents=True, exist_ok=True)

    fig.savefig(png, dpi=190, bbox_inches="tight")
    plt.close(fig)

    scores.to_csv(csv, index=False, float_format="%.17g")

    metadata = {
        "figure": "F04 controlled feature ablation",
        "source": "KBS",
        "snapshot_cutoff": "2026-11-06",
        "evaluation": "development_cv_only",
        "source_sha256": meta["source_sha256"],
        "features_sha256": sha256(feature_file),
        "training_metadata_sha256": sha256(meta_file),
        "same_hyperparameters": True,
        "params": params,
        "seven_features": FEATURES_7,
        "twelve_features": FEATURES_12,
        "fold_count": 5,
        "validation_rows_per_model": 980,
        "mean_pinball_7f": mean7,
        "mean_pinball_12f": mean12,
        "improvement_7f_minus_12f": delta,
        "reproduction_max_abs_diff_12f": max_reproduction_diff,
        "holdout_used": False,
        "limitation": (
            "C1 hyperparameters were previously chosen for "
            "the 12-feature specification. This is a controlled "
            "conditional comparison, not proof of causal feature "
            "benefit or future superiority."
        ),
        "image_sha256": sha256(png),
        "metrics_sha256": sha256(csv),
    }

    metadata_file.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print(f"MEAN PINBALL 7F:  {mean7:.12f}")
    print(f"MEAN PINBALL 12F: {mean12:.12f}")
    print(f"DELTA (7F-12F):  {delta:.12f}")
    print(
        "12F REPRODUCTION MAX DIFF:",
        f"{max_reproduction_diff:.3e}"
    )
    print("F04 CONTROLLED ABLATION PASS")
    print("OUTPUT:", png)

if __name__ == "__main__":
    main()
