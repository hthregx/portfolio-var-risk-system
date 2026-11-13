"""B-07.3: so sánh pinball loss và violation rate trên cùng holdout."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phase2/b/reproduce_day07_20261113_133340"
OUT = ROOT / "figures/phase2/b"

MODELS = [
    ("historical_w250", "HS\n250 phiên", "#2563EB"),
    ("ewma_d094", "EWMA\nλ = 0,94", "#00878A"),
    ("gbv2_12f_c1", "GBv2 C1\n12 đặc trưng", "#7950BE"),
]
TOL = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    meta_path = BASE / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    require(
        meta["snapshot_cutoff"] == "2026-11-06"
        and meta["training_cutoff"] == "2024-12-17",
        "Cutoff khác lần chạy đã thống nhất.",
    )
    require(
        len(meta["models"]["gbv2_12f_c1"]["features"]) == 12,
        "C1 không có đúng 12 đặc trưng.",
    )
    require(
        meta["models"]["gbv2_12f_c1"]["parameters"]["alpha"] == .05,
        "Alpha C1 khác 0.05.",
    )

    tables, hashes = {}, {}
    for name in ("holdout_predictions.csv", "holdout_metrics.csv"):
        path = BASE / name
        hashes[name] = sha(path)
        record = meta["artifacts"][name]

        require(hashes[name] == record["sha256"], f"Hash sai: {name}")
        tables[name] = pd.read_csv(path, float_precision="round_trip")
        require(
            len(tables[name]) == record["rows"],
            f"Số dòng sai: {name}",
        )

    predictions = tables["holdout_predictions.csv"]
    published = tables["holdout_metrics.csv"]
    required = {
        "model", "fold", "target_date", "as_of",
        "actual_return", "quantile_return", "var_return",
    }
    require(
        required.issubset(predictions.columns),
        "Predictions thiếu cột.",
    )
    require(
        not predictions[list(required)].isna().any().any(),
        "Predictions có dữ liệu thiếu.",
    )
    require(
        not predictions.duplicated(["model", "target_date"]).any(),
        "Trùng model/target_date.",
    )
    require(
        not published["model"].duplicated().any(),
        "Bảng metrics trùng model.",
    )

    for column in ("target_date", "as_of"):
        predictions[column] = pd.to_datetime(
            predictions[column], errors="raise"
        )
    require(
        (predictions["as_of"] < predictions["target_date"]).all(),
        "as_of phải trước target_date.",
    )
    require(
        np.isfinite(
            predictions[
                ["actual_return", "quantile_return", "var_return"]
            ].to_numpy(float)
        ).all(),
        "Có giá trị không hữu hạn.",
    )

    published = published.set_index("model")
    reference, records = None, {}

    for model, _, _ in MODELS:
        require(model in published.index, f"Thiếu metrics: {model}")
        part = predictions.loc[
            predictions["model"] == model
        ].sort_values("target_date").reset_index(drop=True)

        require(len(part) == 468, f"{model}: không đủ 468 phiên.")
        require(
            set(part["fold"]) == {"holdout_fixed_fit"},
            f"{model}: sai fold.",
        )
        require(
            part["target_date"].iloc[0] == pd.Timestamp("2024-12-18")
            and part["target_date"].iloc[-1] == pd.Timestamp("2026-11-06"),
            f"{model}: sai phạm vi ngày.",
        )

        keys = part[["target_date", "as_of"]]
        y = part["actual_return"].to_numpy(float)
        q = part["quantile_return"].to_numpy(float)

        if reference is None:
            reference = (keys.copy(), y.copy())
        else:
            pd.testing.assert_frame_equal(keys, reference[0])
            np.testing.assert_allclose(
                y, reference[1], atol=TOL, rtol=0
            )

        np.testing.assert_allclose(
            part["var_return"], np.maximum(0, -q),
            atol=TOL, rtol=0,
        )
        error = y - q
        count = int(np.sum(y < q))
        record = {
            "n": len(part),
            "violations": count,
            "violation_rate": count / len(part),
            "pinball_loss": float(
                np.maximum(.05 * error, -.95 * error).mean()
            ),
            "average_var": float(np.maximum(0, -q).mean()),
        }

        row = published.loc[model]
        for field in ("n", "violations"):
            require(
                record[field] == row[field],
                f"{model}: {field} không khớp.",
            )
        for field in ("violation_rate", "pinball_loss", "average_var"):
            np.testing.assert_allclose(
                record[field], row[field], atol=TOL, rtol=0,
                err_msg=f"{model}: {field} không khớp.",
            )

        records[model] = record

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": "#172B45",
        "axes.labelcolor": "#475569",
        "xtick.color": "#475569",
        "ytick.color": "#64748B",
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6))
    fig.subplots_adjust(
        left=.09, right=.96, top=.70,
        bottom=.30, wspace=.30,
    )
    fig.text(
        .09, .93, "So sánh dự báo rủi ro danh mục",
        fontsize=23, weight="bold",
    )
    fig.text(
        .09, .875,
        "Holdout hồi cứu · 468 phiên · 18/12/2024 – 06/11/2026",
        fontsize=11, color="#64748B",
    )

    x = np.arange(len(MODELS))
    labels = [label for _, label, _ in MODELS]
    colors = [color for _, _, color in MODELS]
    losses = np.array([
        records[model]["pinball_loss"] for model, _, _ in MODELS
    ])
    rates = np.array([
        records[model]["violation_rate"] for model, _, _ in MODELS
    ])

    left, right = axes

    # Nhân 1000 để trục dễ đọc; không thay đổi dữ liệu nguồn.
    scaled_losses = losses * 1000
    left.bar(x, scaled_losses, width=.55, color=colors, zorder=3)
    left.set_title(
        "Sai số dự báo phân vị", loc="left",
        fontsize=13, weight="bold", pad=30,
    )
    left.text(
        0, 1.035, "Pinball loss thấp hơn là tốt hơn",
        transform=left.transAxes, fontsize=9, color="#64748B",
    )
    left.set_ylabel("Mean pinball loss × 1.000")
    left.set_ylim(0, scaled_losses.max() * 1.30)

    for i, value in enumerate(scaled_losses):
        left.text(
            i, value + scaled_losses.max() * .035,
            f"{value:.4f}", ha="center",
            fontsize=11, weight="bold", color=colors[i],
        )

    right.bar(x, rates, width=.55, color=colors, zorder=3)
    right.axhline(
        .05, color="#9A6700", linewidth=1.6,
        linestyle=(0, (4, 3)), zorder=4,
    )
    right.set_title(
        "Tần suất vi phạm", loc="left",
        fontsize=13, weight="bold", pad=30,
    )
    right.text(
        0, 1.035, "Nét đứt: mức danh nghĩa 5%",
        transform=right.transAxes, fontsize=9, color="#64748B",
    )
    right.set_ylabel("Tỷ lệ vi phạm")
    right.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    right.set_ylim(0, max(.05, rates.max()) * 1.40)

    for i, (model, _, _) in enumerate(MODELS):
        record = records[model]
        right.text(
            i, rates[i] + .003,
            f"{rates[i]:.2%}\n{record['violations']}/468",
            ha="center", fontsize=10,
            weight="bold", color=colors[i],
        )

    for ax in axes:
        ax.set_facecolor("#F8FAFC")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.tick_params(axis="both", length=0, pad=9)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E2E8F0", linewidth=.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#CBD5E1")

    notes = [
        "Pinball loss được nhân 1.000 để hiển thị; "
        "ví dụ 1,637 tương ứng khoảng 0,001637 trên lợi suất thập phân.",
        "Vi phạm khi actual_return < q05. "
        "Tỷ lệ thấp hơn không tự động có nghĩa mô hình tốt hơn.",
        "Nguồn: lần chạy B reproduce_day07_20261113_133340; "
        "đã tính lại từ predictions và đối chiếu metrics.",
        "Holdout có giao với kỳ đánh giá Phase 1; "
        "không dùng kết quả này để lựa chọn lại cấu hình.",
    ]
    for y, note in zip((.17, .125, .08, .035), notes):
        fig.text(.09, y, note, fontsize=9, color="#64748B")

    OUT.mkdir(parents=True, exist_ok=True)
    image = OUT / "model_comparison.png"
    output_meta = OUT / "model_comparison_metadata.json"

    fig.savefig(image, dpi=240, facecolor="white")
    plt.close(fig)

    metadata = {
        "task": "B-07.3",
        "evaluation": "retrospective_holdout",
        "first_target": "2024-12-18",
        "last_target": "2026-11-06",
        "n_targets_per_model": 468,
        "alpha": .05,
        "pinball_display_multiplier": 1000,
        "violation_rule": "actual_return < quantile_return",
        "gbv2_feature_count": 12,
        "source_directory": str(BASE.relative_to(ROOT)),
        "models": records,
        "metrics_crosscheck": "PASS",
        "absolute_tolerance": TOL,
        "input_sha256": hashes,
        "source_metadata_sha256": sha(meta_path),
        "script_sha256": sha(__file__),
        "image_sha256": sha(image),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }
    output_meta.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Cùng ngày, actual returns, hash và metrics: PASS")
    for model, record in records.items():
        print(
            f"{model}: pinball={record['pinball_loss']:.9f}; "
            f"violations={record['violations']}/468 "
            f"({record['violation_rate']:.2%})"
        )
    print("Đã lưu:", image.relative_to(ROOT))
    print("Đã lưu:", output_meta.relative_to(ROOT))


if __name__ == "__main__":
    main()