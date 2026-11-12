"""Vẽ 120 phiên CV cuối của 3 baseline, không train lại."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("results/phase2/b/aligned_2026-11-06_20261112_121348/baseline_cv.csv"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("figures/phase2/b"),
    )
    args = parser.parse_args()
    # ALIGNED_PLOT_METADATA_V1
    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    input_hash = sha256(args.input)
    source_meta_path = args.input.parent / "metadata.json"
    source_meta = json.loads(
        source_meta_path.read_text(encoding="utf-8")
    )
    expected_source = (
        "6389586ae8f1edbabb31760e0b555619186114"
        "b3f77d09212362fc2e1c40aadf"
    )
    if source_meta.get("source_sha256") != expected_source:
        raise ValueError("Snapshot không khớp bộ A đã khóa.")
    if source_meta.get("snapshot_cutoff") != "2026-11-06":
        raise ValueError("Cutoff không đúng.")

    artifact = source_meta.get("artifacts", {}).get(args.input.name, {})
    if artifact.get("sha256") != input_hash:
        raise ValueError("Hash CSV không khớp metadata của runner.")

    df = pd.read_csv(
        args.input, parse_dates=["target_date", "as_of"],
        float_precision="round_trip",
    )
    required = {
        "model", "fold", "target_date", "as_of",
        "actual_return", "quantile_return", "var_return",
    }
    if not required.issubset(df.columns):
        raise ValueError("CSV thiếu cột bắt buộc.")
    if df[["target_date", "as_of", "fold"]].isna().any().any():
        raise ValueError("Ngày hoặc fold bị thiếu.")
    if df.duplicated(["model", "target_date"]).any():
        raise ValueError("Trùng ngày dự báo.")
    if not (df["as_of"] < df["target_date"]).all():
        raise ValueError("Có as_of không trước target_date.")
    numeric = df[["actual_return", "quantile_return", "var_return"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Có giá trị không hữu hạn.")
    np.testing.assert_allclose(
        df["var_return"],
        np.maximum(0, -df["quantile_return"]),
        rtol=0, atol=1e-12,
    )

    models = [
        ("historical_w250", "Historical Simulation", "Cửa sổ 250 phiên"),
        ("ewma_d094", "EWMA", "Hệ số suy giảm 0,94"),
        ("gb_G04_7f", "Gradient Boosting · G04", "Baseline 7 features"),
    ]

    # Kiểm tra cùng ngày và lợi suất thực tế trước khi chọn đoạn vẽ.
    parts = []
    for key, _, _ in models:
        part = df.loc[df.model == key].sort_values("target_date")
        part = part.reset_index(drop=True)
        if (
            part.empty
            or part.target_date.isna().any()
            or part.target_date.duplicated().any()
        ):
            raise ValueError(f"Thiếu hoặc trùng ngày: {key}")
        values = part[["actual_return", "quantile_return"]].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"Giá trị không hợp lệ: {key}")
        if parts:
            pd.testing.assert_frame_equal(
                part[["target_date", "actual_return"]],
                parts[0][["target_date", "actual_return"]],
            )
        parts.append(part)

    if set(df["model"]) != {model[0] for model in models}:
        raise ValueError("CSV phải chứa đúng ba baseline.")

    for part in parts:
        counts = part.groupby("fold").size()
        if (
            len(part) != 980
            or set(counts.index) != {f"dev_{i}" for i in range(1, 6)}
            or not counts.eq(196).all()
        ):
            raise ValueError("Không đúng lịch CV: 5 folds x 196 ngày.")
        pd.testing.assert_frame_equal(
            part[["target_date", "as_of", "fold"]],
            parts[0][["target_date", "as_of", "fold"]],
        )

    parts = [part.tail(120).copy() for part in parts]
    if (
        parts[0]["target_date"].iloc[0] != pd.Timestamp("2024-07-01")
        or parts[0]["target_date"].iloc[-1] != pd.Timestamp("2024-12-17")
    ):
        raise ValueError("Đoạn vẽ không đúng 120 phiên CV đã kiểm tra.")
    n = len(parts[0])
    start, end = parts[0].target_date.iloc[[0, -1]]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titleweight": "bold",
        "text.color": "#172B45",
        "axes.labelcolor": "#475569",
        "xtick.color": "#64748B",
        "ytick.color": "#64748B",
        "svg.fonttype": "none",
    })

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 8.7), sharex=True, sharey=True,
    )
    fig.subplots_adjust(
        left=.075, right=.97, top=.78, bottom=.15, hspace=.57,
    )
    fig.text(
        .075, .955, "Rủi ro danh mục · So sánh baseline",
        fontsize=21, weight="bold",
    )
    fig.text(
        .075, .917,
        f"CV phát triển | {n} phiên cuối | "
        f"{start:%d/%m/%Y} – {end:%d/%m/%Y} | α = 5%",
        color="#64748B",
    )

    handles = [
        Line2D(
            [], [], color="#64748B", lw=1.1,
            label="Lợi suất thực tế",
        ),
        Line2D(
            [], [], color="#007F82", lw=2,
            label="Ngưỡng q05",
        ),
        Line2D(
            [], [], color="#C63D4F", marker="o", lw=0,
            markersize=6, label="Vi phạm: lợi suất < q05",
        ),
    ]
    fig.legend(
        handles=handles, loc="upper left",
        bbox_to_anchor=(.067, .884),
        ncol=3, frameon=False, columnspacing=2.5,
    )

    for ax, part, (_, title, detail) in zip(axes, parts, models):
        x = part.target_date
        y = part.actual_return
        q = part.quantile_return
        breach = y < q

        ax.set_facecolor("#F8FAFC")
        ax.plot(x, y, color="#64748B", lw=1.05, zorder=2)
        ax.plot(x, q, color="#007F82", lw=1.9, zorder=3)
        ax.vlines(
            x[breach], y[breach], q[breach],
            color="#C63D4F", lw=1, alpha=.6,
        )
        ax.scatter(
            x[breach], y[breach], color="#C63D4F", s=35,
            edgecolors="white", linewidths=.7, zorder=4,
        )

        ax.set_title(title, loc="left", fontsize=12, pad=23)
        ax.text(
            0, 1.035, detail, transform=ax.transAxes,
            fontsize=9, color="#64748B",
        )
        ax.text(
            1, 1.16,
            f"{breach.sum()}/{n} vi phạm · {breach.mean():.1%}",
            transform=ax.transAxes, ha="right",
            fontsize=10, weight="bold",
        )

        ax.axhline(0, color="#CBD5E1", lw=.8)
        ax.grid(axis="y", color="#E2E8F0", lw=.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.tick_params(axis="both", length=0, pad=7)
        ax.yaxis.set_major_locator(MultipleLocator(.02))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_ylabel("Lợi suất", fontsize=9)
        ax.margins(x=.01, y=.17)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    axes[-1].set_xlabel("Ngày mục tiêu", labelpad=10)

    fig.text(
        .075, .055,
        "Tỷ lệ vi phạm chỉ tính trên đoạn đang vẽ; "
        "không đại diện toàn bộ 5 folds.",
        fontsize=9, color="#64748B",
    )
    fig.text(
        .075, .031,
        "Nguồn: baseline_cv.csv · Ngưỡng q05 là quantile lợi suất, "
        "không phải VaR dương.",
        fontsize=8, color="#64748B",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = args.out / f"baseline_var_timeline.{ext}"
        fig.savefig(path, dpi=220, facecolor="white")
        print(f"Saved: {path}")
    model_records = {}
    for part, (model_id, _, _) in zip(parts, models):
        violations = part["actual_return"] < part["quantile_return"]
        model_records[model_id] = {
            "n_targets": len(part),
            "n_violations": int(violations.sum()),
            "violation_rate": float(violations.mean()),
            "violation_dates": (
                part.loc[violations, "target_date"]
                .dt.strftime("%Y-%m-%d").tolist()
            ),
            "folds": sorted(part["fold"].unique().tolist()),
        }

    metadata = {
        "stage": "development_cv",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": args.input.as_posix(),
        "input_sha256": input_hash,
        "runner_metadata_file": source_meta_path.as_posix(),
        "runner_metadata_sha256": sha256(source_meta_path),
        "source_data_sha256": source_meta["source_sha256"],
        "cutoff": source_meta["snapshot_cutoff"],
        "selection_rule": "last_120_validation_target_dates",
        "first_target": start.strftime("%Y-%m-%d"),
        "last_target": end.strftime("%Y-%m-%d"),
        "n_targets_per_model": n,
        "plot_unit": "percent_labels_from_decimal_returns",
        "threshold_column": "quantile_return",
        "violation_rule": "actual_return < quantile_return",
        "shared_y_axis": True,
        "y_limits_decimal": [float(v) for v in axes[0].get_ylim()],
        "smoothing": False,
        "models": model_records,
        "script_sha256": sha256(__file__),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "image_sha256": sha256(args.out / "baseline_var_timeline.png"),
        "pdf_sha256": sha256(args.out / "baseline_var_timeline.pdf"),
    }
    metadata_path = args.out / "baseline_var_timeline_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved: {metadata_path}")
    for model_id, record in model_records.items():
        print(f"{model_id}: {record['n_violations']}/{n} violations")
    plt.close(fig)


if __name__ == "__main__":
    main()