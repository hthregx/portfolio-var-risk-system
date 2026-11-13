"""B-06.3: plot actual returns and q05 on the full aligned holdout."""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phase2/b/backtesting_2026-11-06"
OUT = ROOT / "figures/phase2/b/backtesting_2026-11-06"

MODELS = [
    (
        "historical_w250",
        "Historical Simulation",
        "Cửa sổ 250 phiên",
        "#2563EB",
    ),
    (
        "ewma_d094",
        "EWMA",
        "Hệ số suy giảm 0,94",
        "#007F82",
    ),
    (
        "gbv2_12f_c1",
        "Gradient Boosting · C1",
        "GBv2 · 12 đặc trưng",
        "#7950B8",
    ),
]

EXPECTED_MODELS = {
    "historical_w250",
    "ewma_d094",
    "gb_G04_7f",
    "gbv2_12f_c1",
}

FIRST = pd.Timestamp("2024-12-18")
LAST = pd.Timestamp("2026-11-06")
N_TARGETS = 468
TOLERANCE = 1e-12


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_verified(name, metadata, **kwargs):
    path = BASE / name
    record = metadata["artifacts"][name]

    require(
        sha256(path) == record["sha256"],
        f"Hash không khớp: {name}",
    )

    frame = pd.read_csv(
        path,
        float_precision="round_trip",
        **kwargs,
    )

    require(
        len(frame) == record["rows"],
        f"Số dòng không khớp metadata: {name}",
    )

    return frame


def main():
    image_path = OUT / "actual_vs_var.png"
    metadata_path = OUT / "actual_vs_var_metadata.json"

    for path in (image_path, metadata_path):
        if path.exists():
            raise FileExistsError(
                f"Đã có {path}. Giữ bản cũ để kiểm tra trước khi chạy lại."
            )

    source_metadata_path = BASE / "metadata.json"
    source_meta = json.loads(
        source_metadata_path.read_text(encoding="utf-8")
    )

    require(
        source_meta["evaluation"] == "retrospective_holdout",
        "Đầu vào không phải retrospective_holdout.",
    )
    require(
        source_meta["snapshot_cutoff"] == "2026-11-06",
        "Sai cutoff.",
    )
    require(
        source_meta["quantile_alpha"] == 0.05,
        "Alpha khác 0.05.",
    )
    require(
        source_meta["n_targets_per_model"] == N_TARGETS,
        "Sai số ngày đánh giá.",
    )
    require(
        source_meta["metrics_crosscheck"] == "PASS",
        "Đối chiếu metrics chưa PASS.",
    )

    df = read_verified(
        "violation_flags.csv",
        source_meta,
        parse_dates=["as_of", "target_date"],
    )
    metrics = read_verified("backtesting.csv", source_meta)

    required = [
        "model", "fold", "as_of", "target_date",
        "actual_return", "quantile_return",
        "var_return", "violation",
    ]

    require(
        set(required).issubset(df.columns),
        "Bảng flags thiếu cột.",
    )
    require(
        not df[required].isna().any().any(),
        "Bảng flags có giá trị thiếu.",
    )
    require(
        set(df["model"]) == EXPECTED_MODELS,
        "Danh sách model trong flags khác dự kiến.",
    )
    require(
        set(metrics["model"]) == EXPECTED_MODELS
        and not metrics["model"].duplicated().any(),
        "Danh sách model trong metrics không hợp lệ.",
    )
    require(
        not df.duplicated(["model", "target_date"]).any(),
        "Trùng model/target_date.",
    )
    require(
        set(df["fold"]) == {"holdout_fixed_fit"},
        "Sai phạm vi holdout.",
    )

    for column in ("as_of", "target_date"):
        require(
            pd.api.types.is_datetime64_any_dtype(df[column]),
            f"Ngày không hợp lệ: {column}",
        )

    require(
        (df["as_of"] < df["target_date"]).all(),
        "Có as_of không trước target_date.",
    )

    numeric = df[
        ["actual_return", "quantile_return", "var_return"]
    ].to_numpy(dtype=float)

    require(
        np.isfinite(numeric).all(),
        "Có NaN hoặc vô cực.",
    )
    require(
        df["violation"].isin([0, 1]).all(),
        "Violation phải bằng 0 hoặc 1.",
    )

    np.testing.assert_array_equal(
        df["violation"].to_numpy(),
        (df["actual_return"] < df["quantile_return"])
        .astype(int).to_numpy(),
    )
    np.testing.assert_allclose(
        df["var_return"],
        np.maximum(0.0, -df["quantile_return"]),
        rtol=0,
        atol=TOLERANCE,
    )

    metrics = metrics.set_index("model")
    parts = {}
    reference = None

    # Validate all four models even though only three are plotted.
    for model in sorted(EXPECTED_MODELS):
        part = (
            df.loc[df["model"] == model]
            .sort_values("target_date")
            .reset_index(drop=True)
        )

        require(
            len(part) == N_TARGETS,
            f"{model}: sai số ngày.",
        )
        require(
            part["target_date"].iloc[0] == FIRST
            and part["target_date"].iloc[-1] == LAST,
            f"{model}: sai ngày đầu/cuối.",
        )

        if reference is None:
            reference = part
        else:
            pd.testing.assert_frame_equal(
                part[["target_date", "as_of", "fold"]],
                reference[["target_date", "as_of", "fold"]],
            )
            np.testing.assert_allclose(
                part["actual_return"],
                reference["actual_return"],
                rtol=0,
                atol=TOLERANCE,
            )

        count = int(part["violation"].sum())

        require(
            metrics.loc[model, "n_targets"] == N_TARGETS,
            f"{model}: số target khác bảng metrics.",
        )
        require(
            count == metrics.loc[model, "n_violations"],
            f"{model}: số vi phạm khác bảng metrics.",
        )
        np.testing.assert_allclose(
            count / N_TARGETS,
            metrics.loc[model, "violation_rate"],
            rtol=0,
            atol=TOLERANCE,
        )

        parts[model] = part

    # Shared limits include all plotted returns and forecasts.
    plotted_values = np.concatenate([
        parts[model][["actual_return", "quantile_return"]]
        .to_numpy().ravel()
        for model, _, _, _ in MODELS
    ])

    step = 0.02
    lower = np.floor(
        (min(0.0, plotted_values.min()) - 0.005) / step
    ) * step
    upper = np.ceil(
        (max(0.0, plotted_values.max()) + 0.005) / step
    ) * step

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": "#172B45",
        "axes.labelcolor": "#475569",
        "xtick.color": "#64748B",
        "ytick.color": "#64748B",
        "axes.unicode_minus": True,
    })

    fig, axes = plt.subplots(
        3, 1,
        figsize=(13.5, 10.5),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.97,
        top=0.78,
        bottom=0.14,
        hspace=0.55,
    )

    fig.text(
        0.085, 0.95,
        "Lợi suất thực tế và phân vị dự báo 5%",
        fontsize=21,
        weight="bold",
    )
    fig.text(
        0.085, 0.91,
        "Holdout hồi cứu · 468 phiên · "
        "18/12/2024 – 06/11/2026",
        fontsize=12,
        color="#64748B",
    )

    handles = [
        Line2D(
            [], [], color="#8291A6", lw=0.9,
            label="Lợi suất thực tế",
        ),
        Line2D(
            [], [], color="#334155", lw=1.8,
            label="Phân vị 5% (màu theo mô hình)",
        ),
        Line2D(
            [], [], color="#C63D4F", marker="o",
            markeredgecolor="white", lw=0, markersize=6,
            label="Vi phạm: lợi suất < q05",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.078, 0.875),
        ncol=3,
        frameon=False,
        fontsize=10,
        columnspacing=2.0,
    )

    records = {}

    for ax, (model, title, detail, color) in zip(axes, MODELS):
        part = parts[model]
        x = part["target_date"]
        actual = part["actual_return"]
        quantile = part["quantile_return"]
        breach = part["violation"].eq(1)

        count = int(breach.sum())
        rate_text = f"{100 * count / len(part):.2f}".replace(".", ",")

        ax.set_facecolor("#F8FAFC")

        ax.plot(
            x, actual,
            color="#8291A6",
            lw=0.75,
            alpha=0.9,
            zorder=2,
        )
        ax.plot(
            x, quantile,
            color=color,
            lw=1.65,
            zorder=3,
        )
        ax.scatter(
            x[breach],
            actual[breach],
            s=23,
            color="#C63D4F",
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )
        ax.axhline(0, color="#AAB7C8", lw=0.7)

        ax.set_title(
            title,
            loc="left",
            fontsize=12,
            fontweight="bold",
            color=color,
            pad=24,
        )
        ax.text(
            0, 1.035, detail,
            transform=ax.transAxes,
            color="#64748B",
            fontsize=9,
        )
        ax.text(
            1, 1.17,
            f"{count}/{len(part)} vi phạm · {rate_text}%",
            transform=ax.transAxes,
            ha="right",
            fontsize=10,
            weight="bold",
        )

        ax.set_ylim(lower, upper)
        ax.set_ylabel("Lợi suất (%)", fontsize=9)
        ax.yaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_major_formatter(
            PercentFormatter(xmax=1.0, decimals=0)
        )
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.tick_params(axis="both", length=0, pad=6)

        records[model] = {
            "n_targets": len(part),
            "n_violations": count,
            "violation_rate": count / len(part),
        }

    axes[-1].set_xlim(
        FIRST - pd.Timedelta(days=5),
        LAST + pd.Timedelta(days=5),
    )
    axes[-1].xaxis.set_major_locator(
        mdates.MonthLocator(interval=3)
    )
    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%m/%Y")
    )
    axes[-1].set_xlabel("Ngày mục tiêu", labelpad=10)

    fig.text(
        0.085, 0.075,
        "Nguồn: kết quả backtesting của B · "
        "Snapshot KBS cutoff 06/11/2026.",
        fontsize=9,
        color="#64748B",
    )
    fig.text(
        0.085, 0.05,
        "Cùng 468 ngày và thang đo; giữ nguyên dữ liệu, không làm mượt. "
        "Đường dự báo là phân vị lợi suất.",
        fontsize=8.5,
        color="#64748B",
    )
    fig.text(
        0.085, 0.025,
        "Holdout giao với giai đoạn Phase 1 đã công bố; "
        "không phải kiểm định độc lập hoàn toàn hoặc dự báo trực tiếp.",
        fontsize=8.5,
        color="#64748B",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=300, facecolor="white")
    plt.close(fig)

    metadata = {
        "task": "B-06.3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": "retrospective_holdout",
        "selection_rule": "all_holdout_target_dates",
        "snapshot_cutoff": "2026-11-06",
        "first_target": str(FIRST.date()),
        "last_target": str(LAST.date()),
        "n_targets_per_model": N_TARGETS,
        "quantile_alpha": 0.05,
        "forecast_column": "quantile_return",
        "violation_rule": "actual_return < quantile_return",
        "input_unit": "decimal_return",
        "axis_display_unit": "percent",
        "shared_y_axis": True,
        "y_limits_decimal": [float(lower), float(upper)],
        "smoothing": False,
        "models": records,
        "gbv2_features": source_meta["gbv2_features"],
        "checks": {
            "input_hashes": "PASS",
            "common_dates_as_of_actual": "PASS",
            "violation_flags": "PASS",
            "counts_vs_backtesting_table": "PASS",
        },
        "input_sha256": {
            "violation_flags.csv": sha256(BASE / "violation_flags.csv"),
            "backtesting.csv": sha256(BASE / "backtesting.csv"),
            "metadata.json": sha256(source_metadata_path),
        },
        "source_directory": str(BASE.relative_to(ROOT)),
        "script_sha256": sha256(__file__),
        "image_sha256": sha256(image_path),
        "caveats": source_meta["caveats"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }

    with metadata_path.open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print(f"Đoạn vẽ: {FIRST.date()} đến {LAST.date()}")
    for model, record in records.items():
        print(
            f"{model}: "
            f"{record['n_violations']}/{record['n_targets']} violations"
        )
    print("Hash, ngày và số vi phạm đối chiếu: PASS")
    print("Đã lưu:", image_path.relative_to(ROOT))
    print("Đã lưu:", metadata_path.relative_to(ROOT))


if __name__ == "__main__":
    main()