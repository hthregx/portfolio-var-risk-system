import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.phase2.features_a import build_features_a, ALL_FEATURES

ROOT = Path.cwd()
EXPECTED_HASH = "6389586ae8f1edbabb31760e0b555619186114b3f77d09212362fc2e1c40aadf"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_figure(fig, folder, name, metadata):
    png = folder / f"{name}.png"
    meta = folder / f"{name}_metadata.json"

    if png.exists() or meta.exists():
        raise FileExistsError(f"{name} already exists")

    fig.savefig(png, dpi=190, bbox_inches="tight")
    plt.close(fig)

    metadata["image_sha256"] = sha(png)
    meta.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"{name}: SAVED - {png}")
    print(f"{name}: METADATA - {meta}")

def load_market(base):
    symbols = ("HPG", "FPT", "MWG", "VNINDEX")
    frames = {}

    for symbol in symbols:
        path = base / "market" / f"{symbol}.csv"
        df = pd.read_csv(path)

        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            raise ValueError(f"{symbol}: missing OHLCV columns")

        df["date"] = pd.to_datetime(df["date"], errors="raise")
        if df["date"].isna().any() or df["date"].duplicated().any():
            raise ValueError(f"{symbol}: invalid dates")

        if not df["date"].is_monotonic_increasing:
            raise ValueError(f"{symbol}: unsorted market dates")

        frames[symbol] = (df, path)

    return frames

def figure_f01(base, folder, source_hash):
    frames = load_market(base)

    equity_dates = pd.DatetimeIndex(sorted(set().union(
        *(set(frames[s][0]["date"]) for s in ("HPG", "FPT", "MWG"))
    )))

    if len(equity_dates) < 1000:
        raise ValueError("Unexpected equity date coverage")

    symbols = ("HPG", "FPT", "MWG", "VNINDEX")
    matrix = np.zeros((4, len(equity_dates)), dtype=int)

    details = {}

    for row, symbol in enumerate(symbols):
        df, path = frames[symbol]

        price = df[["open","high","low","close"]].apply(
            pd.to_numeric, errors="coerce"
        )
        volume = pd.to_numeric(df["volume"], errors="coerce")

        good = (
            np.isfinite(price.to_numpy()).all(axis=1)
            & np.isfinite(volume.to_numpy())
            & (price.to_numpy() > 0).all(axis=1)
            & (volume.to_numpy() >= 0)
            & (price["high"].to_numpy() >=
               price[["open","close","low"]].max(axis=1).to_numpy())
            & (price["low"].to_numpy() <=
               price[["open","close","high"]].min(axis=1).to_numpy())
        )

        locations = pd.Index(equity_dates).get_indexer(df["date"])
        for ix, ok in zip(locations, good):
            if ix >= 0:
                matrix[row, ix] = 1 if ok else 2

        details[symbol] = {
            "source_sha256": sha(path),
            "source_rows": len(df),
            "valid_days": int((matrix[row] == 1).sum()),
            "missing_days": int((matrix[row] == 0).sum()),
            "invalid_ohlcv_days": int((matrix[row] == 2).sum())
        }

    fig, ax = plt.subplots(figsize=(16, 3.5))
    colors = ListedColormap(["#ECEFF3", "#207451", "#D98B25"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], colors.N)

    ax.imshow(
        matrix, aspect="auto", interpolation="nearest",
        cmap=colors, norm=norm
    )

    ax.set_yticks(range(4), symbols)

    ticks = np.linspace(0, len(equity_dates)-1, 8, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [equity_dates[i].strftime("%Y-%m-%d") for i in ticks],
        rotation=20, ha="right"
    )

    ax.set_title("F01 | Daily market data coverage - KBS")
    ax.set_xlabel("Equity trading sessions")
    ax.legend(
        handles=[
            Patch(color="#207451", label="Valid OHLCV"),
            Patch(color="#ECEFF3", label="Missing session"),
            Patch(color="#D98B25", label="Invalid OHLCV")
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.46),
        ncol=3, frameon=False
    )

    fig.tight_layout()

    save_figure(fig, folder, "F01", {
        "figure": "Market data coverage",
        "cutoff": "2026-11-06",
        "source": "KBS",
        "portfolio_sha256": source_hash,
        "n_equity_dates": len(equity_dates),
        "symbols": details,
        "calendar_rule": "union_of_three_stock_trading_dates",
        "missing_policy": "no_imputation",
        "vnindex_note": "VNINDEX REVIEW; price adjustment unverified",
        "quality_scope": "OHLCV structure and numeric validity only"
    })

    print("F01 SOURCE COVERAGE:")
    for symbol, item in details.items():
        print(
            symbol,
            "valid=", item["valid_days"],
            "missing=", item["missing_days"],
            "invalid=", item["invalid_ohlcv_days"]
        )

def figure_f02(dates, returns, folder, source_hash):
    series = pd.Series(
        returns, index=pd.DatetimeIndex(dates),
        name="portfolio_simple_return"
    )

    features = build_features_a(series)

    if tuple(features.columns) != tuple(ALL_FEATURES):
        raise ValueError("Unexpected feature schema")

    complete = features.notna().all(axis=1)

    if len(features) != 1707:
        raise ValueError("Unexpected return history")

    if int((~complete).sum()) != 60:
        raise ValueError("Warmup is not exactly 60 sessions")

    if not complete.iloc[60:].all():
        raise ValueError("Unexpected missing features after warmup")

    matrix = features.notna().to_numpy(dtype=int).T

    fig, axes = plt.subplots(
        2, 1, figsize=(16, 8),
        gridspec_kw={"height_ratios": [1, 1.3]}
    )

    cmap = ListedColormap(["#DE8E45", "#267755"])

    axes[0].imshow(
        matrix, aspect="auto", interpolation="nearest",
        cmap=cmap, vmin=0, vmax=1
    )
    axes[0].set_yticks(range(12), features.columns)
    axes[0].set_title("All sessions - 12 feature availability")
    ticks = np.linspace(0, len(series)-1, 7, dtype=int)
    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels(
        [series.index[i].strftime("%Y-%m") for i in ticks]
    )

    zoom = 95
    axes[1].imshow(
        matrix[:, :zoom],
        aspect="auto",
        interpolation="nearest",
        cmap=cmap, vmin=0, vmax=1
    )
    axes[1].axvline(59.5, color="#333333", linestyle="--")
    axes[1].set_yticks(range(12), features.columns)
    axes[1].set_title(
        "First 95 sessions - dashed line marks 60-session warmup"
    )

    ticks2 = np.linspace(0, zoom-1, 6, dtype=int)
    axes[1].set_xticks(ticks2)
    axes[1].set_xticklabels(
        [series.index[i].strftime("%Y-%m-%d") for i in ticks2],
        rotation=20, ha="right"
    )

    fig.suptitle("F02 | Feature missingness and warmup")
    fig.subplots_adjust(hspace=0.5, top=0.91)

    save_figure(fig, folder, "F02", {
        "figure": "Feature missingness",
        "source_sha256": source_hash,
        "input_rows": 1707,
        "feature_count": 12,
        "warmup_rows": 60,
        "complete_feature_rows": 1647,
        "missing_after_warmup": 0,
        "lag_policy": "all_features_use_prior_returns_only"
    })

    print("F02 WARMUP: 60")
    print("F02 COMPLETE ROWS: 1647")

def figure_f03(folder, source_hash):
    metadata_path = ROOT / (
        "results/phase2/a/training_2026-11-06/metadata.json"
    )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8-sig")
    )

    folds = metadata["five_folds"]

    if len(folds) != 5:
        raise ValueError("Expected five folds")

    if metadata["training_cutoff"] != "2024-12-17":
        raise ValueError("Unexpected development cutoff")

    if metadata["source_sha256"] != source_hash:
        raise ValueError("Training metadata source differs")

    fig, ax = plt.subplots(figsize=(14, 5))

    start = pd.Timestamp("2020-04-06")
    hold_start = pd.Timestamp("2024-12-18")
    hold_end = pd.Timestamp("2026-11-06")

    def num(date):
        return mdates.date2num(pd.Timestamp(date).to_pydatetime())

    for i, fold in enumerate(folds):
        if fold["val_n"] != 196:
            raise ValueError("Unexpected validation size")

        t_end = pd.Timestamp(fold["train_last"])
        v_start = pd.Timestamp(fold["validation_first"])
        v_end = pd.Timestamp(fold["validation_last"])

        if not t_end < v_start <= v_end:
            raise ValueError("Invalid CV chronology")

        y = 5 - i

        ax.barh(
            y, num(t_end)-num(start)+1,
            left=num(start), height=0.52,
            color="#4775B7"
        )

        ax.barh(
            y, num(v_end)-num(v_start)+1,
            left=num(v_start), height=0.52,
            color="#E5A046"
        )

        ax.text(
            num(v_start), y + 0.35,
            f"{fold['train_n']} train / {fold['val_n']} val",
            fontsize=8
        )

    ax.axvspan(
        num(hold_start), num(hold_end),
        color="#A9B0BA", alpha=0.22
    )

    ax.set_yticks(range(1, 6), [
        "Fold 5", "Fold 4", "Fold 3", "Fold 2", "Fold 1"
    ])
    ax.set_xlim(num(start), num(hold_end)+80)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("F03 | Expanding time-series cross-validation")
    ax.set_xlabel("Target trading date")

    ax.legend(
        handles=[
            Patch(color="#4775B7", label="Training"),
            Patch(color="#E5A046", label="Validation"),
            Patch(color="#A9B0BA", alpha=0.35,
                  label="Retrospective holdout")
        ],
        loc="upper left",
        frameon=False
    )

    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    save_figure(fig, folder, "F03", {
        "figure": "Expanding CV timeline",
        "source_sha256": source_hash,
        "training_metadata_sha256": sha(metadata_path),
        "n_folds": 5,
        "folds": folds,
        "training_cutoff": "2024-12-17",
        "holdout_first": "2024-12-18",
        "holdout_last": "2026-11-06",
        "holdout_note": "retrospective; overlaps Phase1 evaluation"
    })

    print("F03 FOLDS: 5")
    print("F03 VALIDATION: 196 sessions per fold")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()

    base = args.snapshot / "data/phase2"
    source = base / "portfolio_returns.csv"
    manifest = base / "latest_manifest.json"

    if sha(source) != EXPECTED_HASH:
        raise ValueError("KBS snapshot hash mismatch")

    source_meta = json.loads(
        manifest.read_text(encoding="utf-8-sig")
    )

    if source_meta["cutoff"] != "2026-11-06":
        raise ValueError("Unexpected cutoff")

    data = pd.read_csv(source)
    data["date"] = pd.to_datetime(data["date"])

    if len(data) != 1707:
        raise ValueError("Expected 1707 portfolio returns")

    if data["date"].duplicated().any():
        raise ValueError("Duplicate portfolio dates")

    if not data["date"].is_monotonic_increasing:
        raise ValueError("Unsorted portfolio dates")

    returns = data["portfolio_simple_return"].to_numpy(float)

    if not np.isfinite(returns).all():
        raise ValueError("Nonfinite portfolio returns")

    folder = ROOT / "figures/phase2/a"

    for name in ("F01","F02","F03"):
        if (
            (folder / f"{name}.png").exists()
            or (folder / f"{name}_metadata.json").exists()
        ):
            raise FileExistsError(f"{name}: existing output")

    folder.mkdir(parents=True, exist_ok=True)

    figure_f01(base, folder, EXPECTED_HASH)
    figure_f02(data["date"], returns, folder, EXPECTED_HASH)
    figure_f03(folder, EXPECTED_HASH)

    print("FINAL-03C: THREE FIGURES GENERATED")

if __name__ == "__main__":
    main()
