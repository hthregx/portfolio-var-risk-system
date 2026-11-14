from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[2]
    base = root / "data/phase2/reference_a"
    output = root / "figures/phase2/b/eda_2026-11-06"
    expected_hash = (
        "6389586ae8f1edbabb31760e0b555619186114b3f77d09212362fc2e1c40aadf"
    )

    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    manifest_path = base / "latest_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not (
        sha256(base / "portfolio_returns.csv")
        == manifest["portfolio_sha256"]
        == expected_hash
    ):
        raise ValueError("Snapshot khác bộ đã dùng chạy HS/EWMA.")

    if manifest["cutoff"] != "2026-11-06" or manifest["source"] != "KBS":
        raise ValueError("Nguồn hoặc cutoff không đúng.")

    # Đọc và kiểm tra giá đóng cửa.
    prices = {}
    input_hashes = {}

    for symbol in ["HPG", "FPT", "MWG"]:
        path = base / "market" / f"{symbol}.csv"
        df = pd.read_csv(path)
        dates = pd.to_datetime(df["date"], errors="raise")
        close = pd.to_numeric(df["close"], errors="raise").astype(float)

        if dates.isna().any() or dates.duplicated().any():
            raise ValueError(f"{symbol}: ngày thiếu hoặc trùng.")
        if not dates.is_monotonic_increasing:
            raise ValueError(f"{symbol}: ngày chưa tăng dần.")
        if not np.isfinite(close).all() or (close <= 0).any():
            raise ValueError(f"{symbol}: close không hợp lệ.")

        if (
            len(df) != 1708
            or dates.iloc[0] != pd.Timestamp("2020-01-02")
            or dates.iloc[-1] != pd.Timestamp("2026-11-06")
        ):
            raise ValueError(f"{symbol}: sai số dòng hoặc khoảng ngày.")

        prices[symbol] = pd.Series(
            close.to_numpy(), index=dates, name=symbol
        )
        input_hashes[symbol] = sha256(path)

    stocks = pd.concat(
        [prices[s] for s in ["HPG", "FPT", "MWG"]], axis=1
    )

    if stocks.isna().any().any():
        raise ValueError("Lịch ba cổ phiếu khác nhau; không tự điền giá.")

    # Xác nhận giá cổ phiếu tái tạo đúng returns đã dùng cho HS/EWMA.
    portfolio = pd.read_csv(base / "portfolio_returns.csv")
    actual = pd.Series(
        portfolio["portfolio_simple_return"].to_numpy(),
        index=pd.to_datetime(portfolio["date"]),
    )
    rebuilt = stocks.pct_change(fill_method=None).iloc[1:].mean(axis=1)

    if not rebuilt.index.equals(actual.index):
        raise ValueError("Lịch returns không khớp giá cổ phiếu.")

    np.testing.assert_allclose(
        rebuilt, actual, rtol=0, atol=1e-12
    )

    if len(actual) != 1707 or not np.isfinite(actual).all():
        raise ValueError("Chuỗi lợi suất tham chiếu không hợp lệ.")

    normalized = stocks.div(stocks.iloc[0]).mul(100)
    base_date = stocks.index[0].strftime("%Y-%m-%d")

    figure_path = output / "normalized_trend.png"
    metadata_path = output / "normalized_trend_metadata.json"

    if figure_path.exists() or metadata_path.exists():
        raise FileExistsError("Đã có biểu đồ; kiểm tra trước khi chạy lại.")

    fig, ax = plt.subplots(figsize=(12, 6))
    axes = [ax]

    colors = {"HPG": "#2563eb", "FPT": "#059669", "MWG": "#d97706"}

    for symbol, color in colors.items():
        axes[0].plot(
            normalized.index, normalized[symbol],
            label=symbol, color=color, linewidth=1.3,
        )

    axes[0].axhline(100, color="#888888", linestyle="--", linewidth=0.8)
    axes[0].set_title(f"Stock closing prices — base 100 on {base_date}")
    axes[0].set_ylabel("Normalized close")
    axes[0].legend(loc="upper left", ncol=3)

    axes[0].set_xlabel("Date")
    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].set_xlim(normalized.index[0], normalized.index[-1])

    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.text(
        0.08, 0.035,
        f"Source: KBS | Cutoff: {manifest['cutoff']} | "
        "Price adjustment status unverified; not a total-return chart.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    metadata = {
        "task": "B-02.3",
        "source": manifest["source"],
        "cutoff": manifest["cutoff"],
        "base_date": base_date,
        "normalization": "100 * close / first_close",
        "symbols": ["HPG", "FPT", "MWG"],
        "vnindex_included": False,
        "price_rows_per_symbol": len(stocks),
        "n_returns_compared": len(actual),
        "last_date": str(stocks.index[-1].date()),
        "max_abs_portfolio_return_difference": float(
            np.max(np.abs(rebuilt.to_numpy() - actual.to_numpy()))
        ),
        "absolute_tolerance": 1e-12,
        "image_sha256": sha256(figure_path),
        "missing_price_policy": "no_fill",
        "adjusted_price_status": "unverified",
        "portfolio_sha256": expected_hash,
        "market_sha256": input_hashes,
        "manifest_sha256": sha256(manifest_path),
        "script_sha256": sha256(__file__),
        "matplotlib": matplotlib.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Đã tạo:", figure_path)
    print("Metadata:", metadata_path)
    print("Ngày gốc:", base_date)
    print("Cutoff:", manifest["cutoff"])
    print("\nGiá chuẩn hóa ngày cuối:")
    print(normalized.iloc[-1].round(2))
    print("Returns đối chiếu:", len(actual), "— PASS")


if __name__ == "__main__":
    main()