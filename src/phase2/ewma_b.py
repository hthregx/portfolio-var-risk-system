from pathlib import Path
from statistics import NormalDist
import hashlib
import json

import numpy as np
import pandas as pd


def ewma_quantiles(returns, decay=0.94, alpha=0.05, warmup=250):
    """Dự báo trước khi cập nhật variance bằng return của target."""
    r = np.asarray(returns, dtype=float)

    if r.ndim != 1 or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError("Lợi suất phải hữu hạn, một chiều và lớn hơn -1.")
    if not isinstance(warmup, int) or warmup < 1 or len(r) <= warmup:
        raise ValueError("Warmup không hợp lệ hoặc không đủ dữ liệu.")
    if not 0 < decay < 1 or not 0 < alpha < 1:
        raise ValueError("Decay và alpha phải nằm trong (0, 1).")

    variance = r[0] ** 2
    z = NormalDist().inv_cdf(alpha)
    quantiles = []

    for i in range(1, len(r)):
        # Tại đây variance chỉ sử dụng r[0:i], không có r[i].
        if i >= warmup:
            quantiles.append(z * np.sqrt(variance))

        # Sau dự báo mới cập nhật bằng lợi suất phiên i.
        variance = decay * variance + (1 - decay) * r[i] ** 2

    return np.asarray(quantiles)


def main():
    base = Path("data/phase2")
    output = Path("results/phase2/b")
    expected_hash = (
        "f85bc33ce54d449e393cd81878e9de8627de88d0c309e7f821568b7177692f53"
    )

    path = base / "portfolio_returns.csv"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads(
        (base / "latest_manifest.json").read_text(encoding="utf-8")
    )

    if digest != expected_hash or digest != manifest["portfolio_sha256"]:
        raise ValueError("Hash không khớp snapshot đã dùng cho HS.")

    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"], errors="raise")
    returns = pd.to_numeric(
        df["portfolio_simple_return"], errors="raise"
    ).to_numpy(dtype=float)

    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Ngày thiếu hoặc trùng.")
    if not dates.is_monotonic_increasing:
        raise ValueError("Ngày chưa tăng dần.")
    if len(df) != 1705 or len(df) != manifest["rows"]:
        raise ValueError("Số dòng không khớp snapshot.")
    if not (
        dates.iloc[-1].strftime("%Y-%m-%d")
        == manifest["cutoff"]
        == manifest["portfolio_last_date"]
        == "2026-11-04"
    ):
        raise ValueError("Cutoff không khớp snapshot.")

    warmup = 250
    q = ewma_quantiles(returns)

    result = pd.DataFrame({
        "model": "ewma_d094",
        "fold": "development",
        "as_of": dates.iloc[warmup - 1:-1].dt.strftime("%Y-%m-%d").to_numpy(),
        "target_date": dates.iloc[warmup:].dt.strftime("%Y-%m-%d").to_numpy(),
        "quantile_return": q,
        "var_return": np.maximum(0, -q),
        "actual_return": returns[warmup:],
    })

    # Chỉ đối chiếu ngày và actual với HS, không dùng forecasts HS.
    hs = pd.read_csv(output / "hs.csv")
    for column in ["as_of", "target_date"]:
        if hs[column].tolist() != result[column].tolist():
            raise ValueError(f"{column} không khớp HS.")

    np.testing.assert_allclose(
        hs["actual_return"], result["actual_return"],
        rtol=1e-10, atol=1e-12,
    )

    csv_path = output / "ewma.csv"
    meta_path = output / "ewma_metadata.json"

    if csv_path.exists() or meta_path.exists():
        raise FileExistsError("Đã có kết quả EWMA; kiểm tra trước khi chạy lại.")

    metadata = {
        "stage": "development_only",
        "input_sha256": digest,
        "cutoff": manifest["cutoff"],
        "model": "ewma_d094",
        "decay": 0.94,
        "alpha": 0.05,
        "conditional_mean": 0.0,
        "distribution": "Normal",
        "initialization": "first_squared_return",
        "history": "expanding",
        "warmup": warmup,
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
    print("\nĐã khớp ngày và actual_return với HS.")
    print("\n5 dự báo cuối:")
    print(result.tail().to_string(index=False))


if __name__ == "__main__":
    main()