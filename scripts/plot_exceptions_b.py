from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/phase2/b/backtesting_2026-11-06"

MODELS = [
    ("historical_w250", "Historical Simulation",
     "Cửa sổ 250 phiên", "#2563EB", 28),
    ("ewma_d094", "EWMA",
     "Hệ số suy giảm λ = 0,94", "#00878A", 24),
    ("gbv2_12f_c1", "Gradient Boosting · C1",
     "GBv2 · 12 đặc trưng", "#7950BE", 25),
    ("gb_G04_7f", "Gradient Boosting · G04",
     "Đối chứng · 7 đặc trưng", "#64748B", 25),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = OUT / "exceptions_monthly.csv"
    source_meta = OUT / "exceptions_heatmap_metadata.json"

    meta = json.loads(source_meta.read_text(encoding="utf-8"))
    source_hash = sha(source)

    if source_hash != meta["monthly_csv_sha256"]:
        raise ValueError("CSV tháng không khớp hash metadata.")
    if (
        meta["first_target"] != "2024-12-18"
        or meta["last_target"] != "2026-11-06"
        or meta["n_targets_per_model"] != 468
        or meta["quantile_alpha"] != .05
    ):
        raise ValueError("Metadata khác phạm vi holdout dự kiến.")

    df = pd.read_csv(source)
    required = {
        "model", "month", "n_violations",
        "n_targets", "violation_rate",
    }
    if not required.issubset(df.columns):
        raise ValueError("CSV tháng thiếu cột.")
    if df[list(required)].isna().any().any():
        raise ValueError("CSV tháng có dữ liệu thiếu.")
    if set(df["model"]) != {item[0] for item in MODELS}:
        raise ValueError("Sai danh sách mô hình.")

    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    if df.duplicated(["model", "month"]).any():
        raise ValueError("Trùng model/tháng.")

    months = pd.period_range("2024-12", "2026-11", freq="M")
    parts, records = {}, {}
    reference_sizes = None

    for model, _, _, _, expected_count in MODELS:
        part = df.loc[df["model"] == model].sort_values("month")
        if list(part["month"]) != list(months):
            raise ValueError(f"Thiếu hoặc thừa tháng: {model}")

        counts = part["n_violations"].to_numpy(float)
        sizes = part["n_targets"].to_numpy(float)
        rates = part["violation_rate"].to_numpy(float)

        if not np.isfinite(np.r_[counts, sizes, rates]).all():
            raise ValueError(f"Giá trị không hữu hạn: {model}")
        if (
            (sizes <= 0).any()
            or (counts < 0).any()
            or (counts > sizes).any()
            or (sizes != np.floor(sizes)).any()
            or (counts != np.floor(counts)).any()
        ):
            raise ValueError(f"Số đếm không hợp lệ: {model}")

        np.testing.assert_allclose(
            rates, counts / sizes, atol=1e-12, rtol=0
        )
        if sizes.sum() != 468 or counts.sum() != expected_count:
            raise ValueError(f"Tổng số không khớp backtesting: {model}")

        if reference_sizes is None:
            reference_sizes = sizes.copy()
        else:
            np.testing.assert_array_equal(sizes, reference_sizes)

        parts[model] = (counts.astype(int), sizes.astype(int))
        records[model] = {
            "n_targets": int(sizes.sum()),
            "n_violations": int(counts.sum()),
            "violation_rate": float(counts.sum() / sizes.sum()),
        }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": "#172B45",
        "axes.labelcolor": "#475569",
        "xtick.color": "#64748B",
        "ytick.color": "#64748B",
    })

    fig, axes = plt.subplots(
        2, 2, figsize=(15, 9),
        sharex=True, sharey=True,
    )
    fig.subplots_adjust(
        left=.07, right=.97, top=.73,
        bottom=.25, hspace=.75, wspace=.12,
    )

    fig.text(
        .07, .94, "Số phiên vi phạm theo tháng",
        fontsize=25, weight="bold",
    )
    fig.text(
        .07, .89,
        "Holdout hồi cứu · 468 phiên · 18/12/2024 – 06/11/2026",
        fontsize=11, color="#64748B",
    )
    fig.text(
        .07, .835,
        "Cột: số vi phạm thực tế     "
        "Nét đứt: kỳ vọng 5% × số phiên dự báo trong tháng",
        fontsize=10, color="#475569",
    )

    x = np.arange(len(months))
    labels = [
        month.strftime("%m/%y") + ("*" if i in (0, 23) else "")
        for i, month in enumerate(months)
    ]
    ymax = max(counts.max() for counts, _ in parts.values()) + 1.2

    for ax, (model, title, detail, color, _) in zip(
        axes.flat, MODELS
    ):
        counts, sizes = parts[model]
        record = records[model]

        ax.set_facecolor("#F8FAFC")
        ax.bar(
            x, counts, width=.72,
            color=color, alpha=.90, zorder=3,
        )
        ax.plot(
            x, .05 * sizes,
            color="#9A6700", linewidth=1.4,
            linestyle=(0, (4, 3)), zorder=4,
        )

        for position, count in zip(x, counts):
            if count:
                ax.text(
                    position, count + .12, str(count),
                    ha="center", va="bottom",
                    fontsize=9, color=color, weight="bold",
                )

        ax.set_title(
            title, loc="left", fontsize=13,
            color=color, weight="bold", pad=30,
        )
        ax.text(
            0, 1.035, detail,
            transform=ax.transAxes,
            fontsize=9, color="#64748B",
        )
        ax.text(
            1, 1.18,
            f"{record['n_violations']}/468 · "
            f"{record['violation_rate']:.2%}",
            transform=ax.transAxes,
            ha="right", fontsize=10, weight="bold",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            labels, rotation=65, ha="right", fontsize=8
        )
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="both", length=0, pad=6)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0, ymax)
        ax.set_xlim(-.7, len(months) - .3)
        ax.grid(axis="y", color="#E2E8F0", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#CBD5E1")

    for ax in axes[:, 0]:
        ax.set_ylabel("Số phiên vi phạm")

    # Không thêm nhãn "Tháng" để tránh sát phần chú thích.
    first_n = int(reference_sizes[0])
    last_n = int(reference_sizes[-1])
    notes = [
        "Vi phạm khi actual_return < q05. Các ô dùng cùng thang đo; "
        "tháng không có cột là không có vi phạm.",
        f"* 12/2024 có {first_n} phiên; 11/2026 có {last_n} phiên. "
        "Hai tháng chỉ bao gồm một phần kỳ; mốc kỳ vọng dùng số phiên thực tế.",
        "Nguồn: exceptions_monthly.csv từ backtesting của B. "
        "Mốc kỳ vọng không phải ranh giới đạt/không đạt kiểm định.",
        "Đánh giá hồi cứu có giao với kỳ đánh giá Phase 1; "
        "không dùng hình này để lựa chọn lại mô hình.",
    ]
    for y, note in zip((.135, .10, .065, .03), notes):
        fig.text(.07, y, note, fontsize=9, color="#64748B")

    OUT.mkdir(parents=True, exist_ok=True)
    image = OUT / "exceptions_monthly_panels.png"
    metadata_path = OUT / "exceptions_monthly_panels_metadata.json"

    fig.savefig(image, dpi=240, facecolor="white")
    plt.close(fig)

    result = {
        "task": "B-06.4",
        "chart": "monthly_exception_counts_four_panels",
        "first_target": meta["first_target"],
        "last_target": meta["last_target"],
        "n_targets_per_model": 468,
        "quantile_alpha": .05,
        "violation_rule": "actual_return < quantile_return",
        "expected_count_rule": "0.05 * monthly_n_targets",
        "shared_y_axis": True,
        "models": records,
        "input_sha256": source_hash,
        "input_metadata_sha256": sha(source_meta),
        "script_sha256": sha(__file__),
        "image_sha256": sha(image),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }
    metadata_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for model, record in records.items():
        print(f"{model}: {record['n_violations']}/468 — PASS")
    print("Đã lưu:", image.relative_to(ROOT))
    print("Đã lưu:", metadata_path.relative_to(ROOT))


if __name__ == "__main__":
    main()