"""Review-safe visual revisions: F01 audited exclusions and F03 CV legend."""
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

EXPECTED = "6389586ae8f1edbabb31760e0b555619186114b3f77d09212362fc2e1c40aadf"
SYMBOLS = ("HPG", "FPT", "MWG", "VNINDEX")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, folder, stem, metadata):
    png = folder / f"{stem}.png"
    meta = folder / f"{stem}_metadata.json"
    if png.exists() or meta.exists():
        raise FileExistsError(f"Output already exists: {stem}")
    fig.savefig(png, dpi=190, bbox_inches="tight")
    plt.close(fig)
    metadata["image_sha256"] = digest(png)
    meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{stem}: PNG + METADATA PASS")


def make_f01(base, folder):
    market = {}
    for symbol in SYMBOLS:
        path = base / "market" / f"{symbol}.csv"
        frame = pd.read_csv(path)
        if not {"date", "open", "high", "low", "close", "volume"}.issubset(frame):
            raise ValueError(f"{symbol}: incomplete OHLCV schema")
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        if frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
            raise ValueError(f"{symbol}: invalid date ordering")
        market[symbol] = (frame, path)

    dates = pd.DatetimeIndex(sorted(set().union(*(set(market[s][0].date) for s in SYMBOLS[:3]))))
    if len(dates) != 1708:
        raise ValueError(f"Unexpected equity calendar: {len(dates)}")
    matrix = np.zeros((4, len(dates)), dtype=np.int8)  # 0 absent, 1 valid, 2 excluded, 3 invalid present
    details = {}
    for row, symbol in enumerate(SYMBOLS):
        frame, path = market[symbol]
        indices = dates.get_indexer(frame.date)
        if (indices < 0).any():
            raise ValueError(f"{symbol}: dates outside equity calendar")
        price = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
        vol = frame["volume"].to_numpy(dtype=float)
        valid = (
            np.isfinite(price).all(axis=1)
            & np.isfinite(vol)
            & (price > 0).all(axis=1)
            & (vol >= 0)
            & (price[:, 1] >= np.max(price[:, [0, 2, 3]], axis=1))
            & (price[:, 2] <= np.min(price[:, [0, 1, 3]], axis=1))
        )
        matrix[row, indices] = np.where(valid, 1, 3)
        details[symbol] = {"source_sha256": digest(path), "raw_rows": len(frame)}

    quality_path = base / "quality" / "vnindex_invalid_ohlc.csv"
    quality = pd.read_csv(quality_path)
    excluded_dates = pd.DatetimeIndex(pd.to_datetime(quality["date"], errors="raise"))
    missing_dates = dates[matrix[3] == 0]
    if len(excluded_dates) != 3 or set(excluded_dates) != set(missing_dates):
        raise ValueError("VNINDEX missing dates do not exactly match quality exclusions")
    matrix[3, dates.get_indexer(excluded_dates)] = 2
    if (matrix == 3).any() or (matrix == 0).any():
        raise ValueError("Unexpected unclassified missing/invalid sessions")
    for row, symbol in enumerate(SYMBOLS):
        details[symbol]["valid_sessions"] = int((matrix[row] == 1).sum())
        details[symbol]["excluded_by_quality"] = int((matrix[row] == 2).sum())

    fig, ax = plt.subplots(figsize=(15.5, 4.3))
    palette = ListedColormap(["#E4E7EB", "#247655", "#E59C32", "#C84444"])
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=palette,
              norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4))
    ax.set_yticks(range(4), SYMBOLS)
    ticks = np.linspace(0, len(dates) - 1, 8, dtype=int)
    ax.set_xticks(ticks, [dates[i].strftime("%Y-%m-%d") for i in ticks], rotation=20, ha="right")
    ax.set_xlabel("Trading sessions (equity calendar)")
    ax.set_title("F01 | Cleaned KBS market-data coverage — cutoff 2026-11-06")
    fig.legend(handles=[
        Patch(color="#247655", label="Valid clean OHLCV"),
        Patch(color="#E59C32", label="Excluded by OHLCV quality check"),
        Patch(color="#E4E7EB", label="Other absent sessions (none)"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.13), ncol=3, frameon=False)
    fig.text(0.5, 0.035,
             "VNINDEX exclusions: 2021-02-17, 2021-12-10, 2023-05-15; raw anomalies recorded in quality audit",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.84, bottom=0.38)
    save(fig, folder, "F01_v2", {
        "source": "KBS", "cutoff": "2026-11-06", "source_sha256": EXPECTED,
        "quality_audit_sha256": digest(quality_path),
        "quality_exclusion_dates": [d.strftime("%Y-%m-%d") for d in excluded_dates],
        "interpretation": "Three VNINDEX records are excluded for OHLCV quality, not unobserved raw sessions",
        "calendar_days": len(dates), "symbols": details,
    })
    print("F01: HPG/FPT/MWG 1708 valid; VNINDEX 1705 valid + 3 quality exclusions")


def make_f03(folder):
    root = Path.cwd()
    path = root / "results/phase2/a/training_2026-11-06/metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8-sig"))
    folds = metadata["five_folds"]
    if metadata["source_sha256"] != EXPECTED or metadata["training_cutoff"] != "2024-12-17" or len(folds) != 5:
        raise ValueError("Training metadata contract differs")
    first = pd.Timestamp("2020-04-06")
    hold_start = pd.Timestamp("2024-12-18")
    hold_end = pd.Timestamp("2026-11-06")
    num = lambda d: mdates.date2num(pd.Timestamp(d).to_pydatetime())
    fig, ax = plt.subplots(figsize=(15, 5.5))
    for i, fold in enumerate(folds):
        y = 5 - i
        train_end = pd.Timestamp(fold["train_last"])
        valid_start = pd.Timestamp(fold["validation_first"])
        valid_end = pd.Timestamp(fold["validation_last"])
        if fold["val_n"] != 196 or not first < train_end < valid_start <= valid_end:
            raise ValueError(f"Invalid fold {i+1} chronology")
        ax.barh(y, num(train_end) - num(first) + 1, left=num(first), height=0.52, color="#4775B7")
        ax.barh(y, num(valid_end) - num(valid_start) + 1, left=num(valid_start), height=0.52, color="#E5A046")
        ax.text(num(valid_start), y + 0.34, f"{fold['train_n']} train / {fold['val_n']} val", fontsize=8)
    ax.axvspan(num(hold_start), num(hold_end), color="#A9B0BA", alpha=0.24)
    ax.set_yticks(range(1, 6), ["Fold 5", "Fold 4", "Fold 3", "Fold 2", "Fold 1"])
    ax.set_ylim(0.45, 5.70)
    ax.set_xlim(num(first) - 40, num(hold_end) + 45)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Target trading date")
    ax.set_title("F03 | Five expanding CV folds with retrospective holdout")
    ax.grid(axis="x", alpha=0.20)
    fig.legend(handles=[
        Patch(color="#4775B7", label="Training"),
        Patch(color="#E5A046", label="Validation"),
        Patch(color="#A9B0BA", alpha=0.24, label="Retrospective holdout (not used for selection)"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.89, bottom=0.24)
    save(fig, folder, "F03_v2", {
        "source_sha256": EXPECTED, "training_metadata_sha256": digest(path),
        "development_end": "2024-12-17", "holdout_start": "2024-12-18",
        "holdout_end": "2026-11-06", "folds": folds,
        "caveat": "Holdout is retrospective and overlaps Phase1 evaluation",
    })
    print("F03: 5 expanding folds, 196 validation sessions each")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    base = args.snapshot / "data" / "phase2"
    manifest = json.loads((base / "latest_manifest.json").read_text(encoding="utf-8-sig"))
    if manifest["cutoff"] != "2026-11-06" or digest(base / "portfolio_returns.csv") != EXPECTED:
        raise ValueError("Snapshot cutoff/hash mismatch")
    out = Path.cwd() / "figures" / "phase2" / "a"
    for stem in ("F01_v2", "F03_v2"):
        if (out / f"{stem}.png").exists() or (out / f"{stem}_metadata.json").exists():
            raise FileExistsError(f"Existing output: {stem}")
    make_f01(base, out)
    make_f03(out)
    print("FINAL-03G REVISED FIGURES PASS")


if __name__ == "__main__":
    main()
