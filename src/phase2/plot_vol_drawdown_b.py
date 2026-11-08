from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
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
    image_path = output_dir / "vol_drawdown.png"
    metadata_path = output_dir / "vol_drawdown_metadata.json"

    for path in (image_path, metadata_path):
        if path.exists():
            raise FileExistsError(
                f"Đã có {path.name}; kiểm tra bản cũ trước khi chạy lại."
            )

    digest = sha256(data_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if digest != EXPECTED_HASH:
        raise ValueError("Snapshot khác bộ dữ liệu đã chọn.")
    if digest != manifest.get("portfolio_sha256"):
        raise ValueError("Hash không khớp manifest.")
    if manifest.get("cutoff") != "2026-11-04":
        raise ValueError("Cutoff khác 2026-11-04.")
    if manifest.get("source") != "KBS":
        raise ValueError("Nguồn dữ liệu khác KBS.")

    df = pd.read_csv(data_path)
    dates = pd.to_datetime(df["date"], errors="raise")
    values = pd.to_numeric(
        df["portfolio_simple_return"], errors="raise"
    ).to_numpy(dtype=float)

    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Ngày thiếu hoặc trùng.")
    if not dates.is_monotonic_increasing:
        raise ValueError("Ngày chưa tăng dần.")
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Lợi suất không hợp lệ.")

    returns = pd.Series(values, index=pd.DatetimeIndex(dates))
    first_date = str(returns.index[0].date())
    last_date = str(returns.index[-1].date())

    if (
        len(returns) != 1705
        or first_date != "2020-01-03"
        or last_date != "2026-11-04"
    ):
        raise ValueError("Số mẫu hoặc khoảng ngày khác snapshot.")

    # Volatility mô tả tại cuối phiên, không shift, không điền NaN.
    vol20 = returns.rolling(20, min_periods=20).std(ddof=1)
    vol60 = returns.rolling(60, min_periods=60).std(ddof=1)

    # W0 = 1; tính đỉnh bao gồm cả vốn ban đầu.
    wealth = (1.0 + returns).cumprod()
    running_peak = wealth.cummax().clip(lower=1.0)
    drawdown = wealth / running_peak - 1.0

    if vol20.isna().sum() != 19 or vol60.isna().sum() != 59:
        raise ValueError("Warm-up volatility không đúng.")
    if not np.isfinite(wealth.to_numpy()).all():
        raise ValueError("Giá trị tích lũy không hữu hạn.")
    if not ((drawdown >= -1) & (drawdown <= 0)).all():
        raise ValueError("Drawdown nằm ngoài khoảng hợp lệ.")

    trough_date = drawdown.idxmin()
    min_drawdown = float(drawdown.min())

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True
    )
    x = returns.index.to_pydatetime()

    axes[0].plot(
        x, vol20.to_numpy() * 100,
        color="#2563EB", linewidth=1.2, label="20 sessions",
    )
    axes[0].plot(
        x, vol60.to_numpy() * 100,
        color="#E67E00", linewidth=1.4, label="60 sessions",
    )
    axes[0].set_title(
        "Rolling volatility of daily portfolio returns"
    )
    axes[0].set_ylabel("Daily volatility (%)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend()

    dd_pct = drawdown.to_numpy() * 100
    axes[1].plot(
        x, dd_pct, color="#B91C1C", linewidth=1.2,
        label="Portfolio drawdown",
    )
    axes[1].fill_between(
        x, dd_pct, 0, color="#B91C1C", alpha=0.15
    )
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].scatter(
        [trough_date], [min_drawdown * 100],
        color="#7F1D1D", zorder=3,
        label=(
            f"Deepest drawdown: {min_drawdown * 100:.2f}% "
            f"({trough_date.date()})"
        ),
    )
    axes[1].set_title("Drawdown from cumulative portfolio wealth")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")
    axes[1].set_ylim(top=1)
    axes[1].legend(loc="lower left", fontsize=9)

    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)

    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].set_xlim(x[0], x[-1])

    fig.suptitle(
        f"Equal-weight HPG / FPT / MWG portfolio\n"
        f"{first_date} to {last_date}",
        fontsize=14,
    )
    fig.text(
        0.5, 0.025,
        "Source: KBS | Cutoff: 2026-11-04 | "
        "Volatility: sample standard deviation, not annualized\n"
        "Wealth compounded from supplied returns; "
        "price-adjustment status unverified.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))

    metadata = {
        "stage": "descriptive_eda",
        "source": "KBS",
        "input_sha256": digest,
        "cutoff": manifest["cutoff"],
        "rows": len(returns),
        "first_date": first_date,
        "last_date": last_date,
        "volatility_windows": [20, 60],
        "ddof": 1,
        "annualized": False,
        "include_current_session": True,
        "fill_missing": False,
        "warmup_missing": {"vol20": 19, "vol60": 59},
        "initial_wealth": 1.0,
        "drawdown_formula": "W_t / max(1, W_1, ..., W_t) - 1",
        "min_drawdown_decimal": min_drawdown,
        "min_drawdown_date": str(trough_date.date()),
        "last_drawdown_decimal": float(drawdown.iloc[-1]),
        "last_vol20_decimal": float(vol20.iloc[-1]),
        "last_vol60_decimal": float(vol60.iloc[-1]),
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

    print(f"Số lợi suất: {len(returns)}")
    print(f"Vol20 cuối kỳ: {vol20.iloc[-1] * 100:.4f}%")
    print(f"Vol60 cuối kỳ: {vol60.iloc[-1] * 100:.4f}%")
    print(f"Drawdown sâu nhất: {min_drawdown * 100:.4f}%")
    print(f"Ngày đáy drawdown: {trough_date.date()}")
    print(f"Drawdown cuối kỳ: {drawdown.iloc[-1] * 100:.4f}%")
    print(f"Đã lưu: {image_path.relative_to(ROOT)}")
    print(f"Đã lưu: {metadata_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()