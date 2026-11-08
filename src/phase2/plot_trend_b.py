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
    base = Path("data/phase2")
    output = Path("figures/phase2/b")
    expected_hash = (
        "f85bc33ce54d449e393cd81878e9de8627de88d0c309e7f821568b7177692f53"
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

    if manifest["cutoff"] != "2026-11-04" or manifest["source"] != "KBS":
        raise ValueError("Nguồn hoặc cutoff không đúng.")

    # Đọc và kiểm tra giá đóng cửa.
    prices = {}
    input_hashes = {}

    for symbol in ["HPG", "FPT", "MWG", "VNINDEX"]:
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

        info = manifest["symbols"][symbol]
        if len(df) != info["rows"]:
            raise ValueError(f"{symbol}: số dòng khác manifest.")
        if not (
            dates.iloc[-1].strftime("%Y-%m-%d")
            == info["last_date"]
            == manifest["cutoff"]
        ):
            raise ValueError(f"{symbol}: ngày cuối khác cutoff.")

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
        rebuilt, actual, rtol=1e-10, atol=1e-12
    )

    normalized = stocks.div(stocks.iloc[0]).mul(100)
    base_date = stocks.index[0].strftime("%Y-%m-%d")

    # Thêm NaN tại ngày bị loại để đường VNINDEX bị ngắt.
    info = manifest["symbols"]["VNINDEX"]
    excluded = pd.to_datetime(info.get("excluded_known_ohlc_dates", []))
    calendar = (
        stocks.index.union(prices["VNINDEX"].index)
        .union(excluded)
        .sort_values()
    )
    index_series = prices["VNINDEX"].reindex(calendar)
    index_series.loc[excluded] = np.nan

    figure_path = output / "normalized_trend.png"
    metadata_path = output / "normalized_trend_metadata.json"

    if figure_path.exists() or metadata_path.exists():
        raise FileExistsError("Đã có biểu đồ; kiểm tra trước khi chạy lại.")

    # Vẽ hai panel chung trục thời gian.
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

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

    axes[1].plot(
        index_series.index, index_series,
        color="#475569", linewidth=1.2, label="VNINDEX",
    )

    for i, day in enumerate(excluded):
        axes[1].axvline(
            day, color="#b91c1c", linestyle=":", linewidth=0.9,
            label="Excluded OHLC date" if i == 0 else None,
        )

    axes[1].set_title(
        f"VNINDEX — quality status: {info.get('quality_status', 'UNKNOWN')}"
    )
    axes[1].set_ylabel("Index points")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="upper left", fontsize=9)

    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

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
        "vnindex_unit": "index_points",
        "vnindex_quality": info.get("quality_status"),
        "excluded_known_ohlc_dates": [
            d.strftime("%Y-%m-%d") for d in excluded
        ],
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
    print("\nVNINDEX ngày cuối:", prices["VNINDEX"].iloc[-1])


if __name__ == "__main__":
    main()