"""B-05.3: tương quan returns HPG/FPT/MWG và portfolio."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/phase2/reference_a"
OUT = ROOT / "figures/phase2/b/correlation_2026-11-06"
SYMBOLS = ["HPG", "FPT", "MWG"]
EXPECTED_SHA = (
    "6389586ae8f1edbabb31760e0b555619186114"
    "b3f77d09212362fc2e1c40aadf"
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(f"Đã có kết quả; không ghi đè: {OUT}")

    portfolio_path = BASE / "portfolio_returns.csv"
    manifest_path = BASE / "latest_manifest.json"
    audit_path = BASE / "market/source_check.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    if (
        sha256(portfolio_path) != EXPECTED_SHA
        or manifest.get("portfolio_sha256") != EXPECTED_SHA
        or manifest.get("cutoff") != "2026-11-06"
    ):
        raise ValueError("Snapshot portfolio không đúng bộ A đã khóa.")

    if (
        audit.get("status") != "PASS"
        or audit.get("portfolio_reference_sha256") != EXPECTED_SHA
    ):
        raise ValueError("Biên bản kiểm tra nguồn chưa hợp lệ.")

    portfolio = pd.read_csv(
        portfolio_path, parse_dates=["date"],
        float_precision="round_trip",
    ).set_index("date")["portfolio_simple_return"]

    if (
        portfolio.index.hasnans
        or portfolio.index.has_duplicates
        or not portfolio.index.is_monotonic_increasing
        or len(portfolio) != 1707
    ):
        raise ValueError("Lịch portfolio không hợp lệ.")

    closes = {}
    input_hashes = {}

    for symbol in SYMBOLS:
        path = BASE / "market" / f"{symbol}.csv"
        digest = sha256(path)
        if digest != audit["market_file_sha256"][symbol]:
            raise ValueError(f"{symbol}: file đã thay đổi sau kiểm tra nguồn.")

        df = pd.read_csv(
            path, parse_dates=["date"], float_precision="round_trip"
        )
        close = df.set_index("date")["close"].astype(float)

        if (
            close.index.hasnans
            or close.index.has_duplicates
            or not close.index.is_monotonic_increasing
            or len(close) != 1708
            or close.index[0] != pd.Timestamp("2020-01-02")
            or close.index[-1] != pd.Timestamp("2026-11-06")
        ):
            raise ValueError(f"{symbol}: lịch giá không đúng.")

        if not np.isfinite(close).all() or (close <= 0).any():
            raise ValueError(f"{symbol}: giá không hợp lệ.")

        if closes and not close.index.equals(closes[SYMBOLS[0]].index):
            raise ValueError("Lịch giá ba mã khác nhau.")

        closes[symbol] = close
        input_hashes[symbol] = digest

    prices = pd.DataFrame(closes)
    returns = prices.pct_change(fill_method=None).iloc[1:].copy()

    if not returns.index.equals(portfolio.index):
        raise ValueError("Ngày returns không khớp portfolio A.")

    if (
        not np.isfinite(returns.to_numpy()).all()
        or not np.isfinite(portfolio.to_numpy()).all()
        or (returns <= -1).any().any()
        or (portfolio <= -1).any()
    ):
        raise ValueError("Returns không hợp lệ.")

    difference = (returns.mean(axis=1) - portfolio).abs()
    np.testing.assert_allclose(
        returns.mean(axis=1), portfolio, rtol=0, atol=1e-12
    )

    returns["Portfolio"] = portfolio
    if (returns.std() == 0).any():
        raise ValueError("Có chuỗi hằng, không tính được tương quan.")

    corr = returns.corr(method="pearson")
    matrix = corr.to_numpy()

    if not np.isfinite(matrix).all():
        raise ValueError("Ma trận tương quan có giá trị không hữu hạn.")
    np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.diag(matrix), 1, rtol=0, atol=1e-12)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": "#172B45",
        "axes.labelcolor": "#475569",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
    })

    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    fig.subplots_adjust(left=.17, right=.85, top=.79, bottom=.23)

    fig.text(
        .08, .94, "Tương quan lợi suất danh mục",
        fontsize=22, weight="bold",
    )
    fig.text(
        .08, .89,
        "HPG · FPT · MWG · Danh mục tỷ trọng bằng nhau",
        fontsize=11, color="#64748B",
    )
    fig.text(
        .08, .85,
        "03/01/2020 – 06/11/2026 | 1.707 quan sát chung | Pearson",
        fontsize=10, color="#64748B",
    )

    cmap = plt.get_cmap("RdBu_r")
    image = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)

    labels = ["HPG", "FPT", "MWG", "Danh mục"]
    ax.set_xticks(range(4), labels=labels)
    ax.set_yticks(range(4), labels=labels)
    ax.tick_params(length=0, pad=12)

    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(4):
        for j in range(4):
            value = matrix[i, j]
            rgb = cmap((value + 1) / 2)[:3]
            luminance = np.dot(rgb, [0.2126, 0.7152, 0.0722])
            ax.text(
                j, i, f"{value:.3f}",
                ha="center", va="center",
                fontsize=16, weight="bold",
                color="white" if luminance < .5 else "#172B45",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=.045, pad=.05)
    colorbar.set_ticks([-1, -.5, 0, .5, 1])
    colorbar.set_label("Hệ số tương quan r", fontsize=10)
    colorbar.outline.set_visible(False)

    fig.text(
        .08, .13,
        "Nguồn: giá KBS từ artifact 42566441307; cutoff 06/11/2026.\n"
        "Returns danh mục tính lại khớp snapshot A trên toàn bộ 1.707 ngày.",
        fontsize=9, color="#64748B", linespacing=1.6,
    )
    fig.text(
        .08, .055,
        "Tương quan trên lợi suất, không phải giá. Danh mục chứa cả ba cổ phiếu.\n"
        "Mô tả toàn mẫu; không dùng hình này để lựa chọn mô hình hoặc features.",
        fontsize=9, color="#64748B", linespacing=1.6,
    )

    OUT.mkdir(parents=True)
    corr.to_csv(OUT / "correlation.csv", float_format="%.17g")
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT / f"correlation.{ext}", dpi=300, facecolor="white"
        )
    plt.close(fig)

    metadata = {
        "task": "B-05.3",
        "purpose": "full_sample_descriptive_analysis",
        "method": "Pearson",
        "input_type": "daily_simple_returns_from_close",
        "symbols": SYMBOLS,
        "portfolio_weights": {s: 1 / 3 for s in SYMBOLS},
        "first_return_date": str(returns.index[0].date()),
        "last_return_date": str(returns.index[-1].date()),
        "n_observations": len(returns),
        "market_cutoff": "2026-11-06",
        "artifact_run": audit["artifact_run"],
        "market_sha256": input_hashes,
        "portfolio_sha256": EXPECTED_SHA,
        "manifest_sha256": sha256(manifest_path),
        "source_check_sha256": sha256(audit_path),
        "max_abs_portfolio_return_difference": float(difference.max()),
        "absolute_tolerance": 1e-12,
        "return_warmup_price_rows": 1,
        "missing_value_imputation": False,
        "correlation_color_limits": [-1, 1],
        "price_adjustment_status": "unverified",
        "scope_note": (
            "Aggregate returns match A; original A market-file "
            "byte identity is not established."
        ),
        "script_sha256": sha256(__file__),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "outputs_sha256": {
            name: sha256(OUT / name)
            for name in ("correlation.csv", "correlation.png", "correlation.pdf")
        },
    }
    (OUT / "correlation_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(corr.to_string(float_format=lambda x: f"{x:.6f}"))
    print("\nSố quan sát:", len(returns))
    print("Sai lệch portfolio lớn nhất:", float(difference.max()))
    print("Đã lưu:", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()