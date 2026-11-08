from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_HASH = (
    "f85bc33ce54d449e393cd81878e9de8627"
    "de88d0c309e7f821568b7177692f53"
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    data_path = ROOT / "data/phase2/portfolio_returns.csv"
    manifest_path = ROOT / "data/phase2/latest_manifest.json"
    output_dir = ROOT / "figures/phase2/b"
    image_path = output_dir / "return_distribution.png"
    metadata_path = output_dir / "return_distribution_metadata.json"

    # Kiểm tra đầu ra trước khi vẽ.
    for path in (image_path, metadata_path):
        if path.exists():
            raise FileExistsError(
                f"Đã có {path.name}. Kiểm tra bản cũ trước khi chạy lại."
            )

    digest = sha256(data_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if digest != EXPECTED_HASH:
        raise ValueError("Snapshot khác bộ dữ liệu đã chọn.")
    if digest != manifest.get("portfolio_sha256"):
        raise ValueError("Hash dữ liệu không khớp manifest.")
    if manifest.get("source") != "KBS":
        raise ValueError("Nguồn dữ liệu khác KBS.")
    if manifest.get("cutoff") != "2026-11-04":
        raise ValueError("Cutoff khác 2026-11-04.")

    df = pd.read_csv(data_path)
    dates = pd.to_datetime(df["date"], errors="raise")
    returns = pd.to_numeric(
        df["portfolio_simple_return"], errors="raise"
    ).to_numpy(dtype=float)

    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Ngày bị thiếu hoặc trùng.")
    if not dates.is_monotonic_increasing:
        raise ValueError("Ngày chưa tăng dần.")
    if not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("Lợi suất không hợp lệ.")
    if len(returns) != 1705:
        raise ValueError("Số dòng khác snapshot đã chọn.")

    first_date = str(dates.iloc[0].date())
    last_date = str(dates.iloc[-1].date())

    if first_date != "2020-01-03" or last_date != "2026-11-04":
        raise ValueError("Khoảng ngày khác snapshot đã kiểm tra.")

    # Tính quantile ở đơn vị thập phân; nhân 100 khi vẽ.
    alpha = 0.05
    q05 = float(np.quantile(returns, alpha, method="linear"))
    returns_pct = returns * 100
    q05_pct = q05 * 100

    # Chọn bins theo dữ liệu, giữ toàn bộ quan sát.
    edges = np.histogram_bin_edges(returns_pct, bins="fd")
    counts, _ = np.histogram(returns_pct, bins=edges)

    if int(counts.sum()) != len(returns):
        raise ValueError("Histogram không chứa đủ quan sát.")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        returns_pct,
        bins=edges,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
        label=f"Daily portfolio returns (n={len(returns):,})",
    )
    ax.axvline(
        q05_pct,
        color="#C62828",
        linestyle="--",
        linewidth=2,
        label=f"Empirical 5% quantile = {q05_pct:.4f}%",
    )
    ax.axvline(
        0,
        color="#555555",
        linestyle=":",
        linewidth=1.2,
        label="Zero return",
    )

    ax.set_title(
        "Distribution of daily portfolio simple returns\n"
        f"{first_date} to {last_date}",
        fontsize=14,
    )
    ax.set_xlabel("Daily portfolio simple return (%)")
    ax.set_ylabel("Number of observations")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)

    fig.text(
        0.5,
        0.035,
        "Source: KBS | Cutoff: 2026-11-04 | HPG, FPT, MWG: equal weights\n"
        "Full-sample descriptive quantile; not a conditional VaR forecast.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))

    metadata = {
        "stage": "descriptive_eda",
        "source": manifest["source"],
        "input_path": str(data_path.relative_to(ROOT)),
        "input_sha256": digest,
        "cutoff": manifest["cutoff"],
        "first_date": first_date,
        "last_date": last_date,
        "rows": len(returns),
        "return_column": "portfolio_simple_return",
        "input_unit": "decimal",
        "plot_unit": "percent",
        "alpha": alpha,
        "quantile_method": "linear",
        "empirical_q05_decimal": q05,
        "empirical_q05_percent": q05_pct,
        "observations_below_q05": int((returns < q05).sum()),
        "bin_method": "freedman_diaconis",
        "bin_count": len(edges) - 1,
        "bin_edges_percent": edges.tolist(),
        "histogram_observations": int(counts.sum()),
        "script_sha256": sha256(__file__),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=300, facecolor="white")
    plt.close(fig)

    metadata["image_sha256"] = sha256(image_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Số quan sát: {len(returns)}")
    print(f"Giai đoạn: {first_date} đến {last_date}")
    print(f"Phân vị 5% dạng thập phân: {q05:.10f}")
    print(f"Phân vị 5% dạng phần trăm: {q05_pct:.4f}%")
    print(f"Số bins: {len(edges) - 1}")
    print(f"Đã lưu: {image_path.relative_to(ROOT)}")
    print(f"Đã lưu: {metadata_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()