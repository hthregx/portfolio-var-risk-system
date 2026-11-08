from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


def main():
    base = Path("data/phase2")
    output = Path("results/phase2/b")
    expected_hash = (
        "f85bc33ce54d449e393cd81878e9de8627de88d0c309e7f821568b7177692f53"
    )

    # 1. Kiểm tra đúng snapshot đã xác nhận.
    path = base / "portfolio_returns.csv"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads(
        (base / "latest_manifest.json").read_text(encoding="utf-8")
    )

    if digest != expected_hash or digest != manifest["portfolio_sha256"]:
        raise ValueError("Hash không khớp snapshot B hoặc manifest.")

    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"], errors="raise")
    returns = pd.to_numeric(
        df["portfolio_simple_return"], errors="raise"
    ).to_numpy(dtype=float)

    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Ngày thiếu hoặc trùng.")
    if not dates.is_monotonic_increasing:
        raise ValueError("Ngày chưa tăng dần.")
    if not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("Lợi suất không hợp lệ.")
    if len(df) != 1705 or len(df) != manifest["rows"]:
        raise ValueError("Số dòng không khớp snapshot.")
    if not (
        dates.iloc[-1].strftime("%Y-%m-%d")
        == manifest["portfolio_last_date"]
        == manifest["cutoff"]
        == "2026-11-04"
    ):
        raise ValueError("Cutoff không khớp snapshot.")

    # 2. HS: chỉ dùng 250 lợi suất trước ngày target.
    window, alpha = 250, 0.05
    quantiles = np.array([
        np.quantile(returns[i - window:i], alpha, method="linear")
        for i in range(window, len(returns))
    ])

    result = pd.DataFrame({
        "model": "historical_w250",
        "fold": "development",
        "as_of": dates.iloc[window - 1:-1].dt.strftime("%Y-%m-%d").to_numpy(),
        "target_date": dates.iloc[window:].dt.strftime("%Y-%m-%d").to_numpy(),
        "quantile_return": quantiles,
        "var_return": np.maximum(0, -quantiles),
        "actual_return": returns[window:],
    })

    # 3. Lưu kết quả; không ghi đè file đã tồn tại.
    csv_path = output / "hs.csv"
    meta_path = output / "hs_metadata.json"

    if csv_path.exists() or meta_path.exists():
        raise FileExistsError("Đã có kết quả HS. Dừng để kiểm tra file cũ.")

    metadata = {
        "stage": "development_only",
        "input_sha256": digest,
        "cutoff": manifest["cutoff"],
        "window": window,
        "alpha": alpha,
        "quantile_method": "linear",
        "prediction_rows": len(result),
        "first_target": result["target_date"].iloc[0],
        "last_target": result["target_date"].iloc[-1],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(csv_path, index=False)
    meta_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(json.dumps(metadata, indent=2))
    print("\n5 dự báo cuối:")
    print(result.tail().to_string(index=False))


if __name__ == "__main__":
    main()