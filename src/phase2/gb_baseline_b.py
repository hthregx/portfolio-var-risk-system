from pathlib import Path
import hashlib
import json
import platform
import subprocess

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

from src.gb_return_features import build_return_features
from src.phase2.cv_b import expanding_splits


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_HASH = (
    "f85bc33ce54d449e393cd81878e9de8627"
    "de88d0c309e7f821568b7177692f53"
)
FEATURES = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_5",
    "rolling_vol_5",
    "rolling_vol_20",
    "rolling_vol_60",
    "drawdown",
]
PARAMS = {
    "loss": "quantile",
    "alpha": 0.05,
    "n_estimators": 100,
    "learning_rate": 0.03,
    "max_depth": 2,
    "min_samples_leaf": 5,
    "subsample": 1.0,
    "random_state": 42,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    data_path = ROOT / "data/phase2/portfolio_returns.csv"
    manifest_path = ROOT / "data/phase2/latest_manifest.json"
    output_dir = ROOT / "results/phase2/b"
    csv_path = output_dir / "gb_baseline.csv"
    metadata_path = output_dir / "gb_baseline_metadata.json"

    # Kiểm tra trước khi train, tránh ghi đè kết quả đã có.
    for path in (csv_path, metadata_path):
        if path.exists():
            raise FileExistsError(
                f"Đã có {path.name}; kiểm tra bản cũ trước khi chạy lại."
            )

    digest = sha256(data_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if digest != EXPECTED_HASH:
        raise ValueError("Snapshot khác bộ HS/EWMA đã kiểm tra.")
    if digest != manifest.get("portfolio_sha256"):
        raise ValueError("Hash dữ liệu không khớp manifest.")
    if manifest.get("cutoff") != "2026-11-04":
        raise ValueError("Cutoff khác snapshot đã chọn.")

    df = pd.read_csv(data_path, parse_dates=["date"])
    returns = df.set_index("date")["portfolio_simple_return"]

    # Builder kiểm tra ngày, giá trị và tạo features đã lag.
    features = build_return_features(returns)
    if features.columns.tolist() != FEATURES:
        raise ValueError("Tên hoặc thứ tự 7 features không đúng.")

    X = features.dropna()
    y = returns.loc[X.index]

    if len(returns) != 1705 or len(X) != 1645:
        raise ValueError("Số mẫu khác snapshot đã kiểm tra.")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Features chứa giá trị không hữu hạn.")

    # Phiên dữ liệu ngay trước từng target.
    as_of = pd.Series(
        returns.index, index=returns.index
    ).shift(1)

    predictions = []
    fold_records = []

    for fold, (train, val) in enumerate(
        expanding_splits(X.index, n_splits=5), start=1
    ):
        train_dates = X.index[train]
        target_dates = X.index[val]

        if train_dates.max() >= target_dates.min():
            raise ValueError("Train chạm hoặc vượt ngày validation.")

        # Mỗi fold khởi tạo model mới và chỉ fit trên train.
        model = GradientBoostingRegressor(**PARAMS)
        model.fit(X.iloc[train], y.iloc[train])

        # Features của mỗi target chỉ dùng lịch sử trước target đó.
        q = model.predict(X.iloc[val])
        if not np.isfinite(q).all():
            raise ValueError("Dự báo chứa giá trị không hữu hạn.")

        predictions.append(pd.DataFrame({
            "model": "gb_G04",
            "fold": f"development_fold_{fold}",
            "as_of": as_of.loc[target_dates].to_numpy(),
            "target_date": target_dates,
            "quantile_return": q,
            "var_return": np.maximum(0.0, -q),
            "actual_return": y.iloc[val].to_numpy(),
        }))

        fold_records.append({
            "fold": fold,
            "n_train": len(train),
            "train_start": str(train_dates[0].date()),
            "train_end": str(train_dates[-1].date()),
            "n_val": len(val),
            "val_start": str(target_dates[0].date()),
            "val_end": str(target_dates[-1].date()),
        })
        print(
            f"Fold {fold}: train={len(train)}, "
            f"validation={len(val)} — đã train"
        )

    result = pd.concat(predictions, ignore_index=True)

    if len(result) != 1370 or result["target_date"].duplicated().any():
        raise ValueError("Số lượng hoặc ngày dự báo không đúng.")
    if not (result["as_of"] < result["target_date"]).all():
        raise ValueError("as_of phải trước target_date.")

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()

    code_paths = [
        Path(__file__),
        ROOT / "src/phase2/cv_b.py",
        ROOT / "src/gb_return_features.py",
    ]
    metadata = {
        "stage": "development_only",
        "model": "gb_G04",
        "input_path": str(data_path.relative_to(ROOT)),
        "input_sha256": digest,
        "cutoff": manifest["cutoff"],
        "features": FEATURES,
        "parameters": PARAMS,
        "split_status": "provisional_not_agreed_with_A",
        "gap": 0,
        "refit": "once_at_start_of_each_fold",
        "forecast_protocol": (
            "sequential_one_step_with_observed_prior_returns"
        ),
        "input_rows": len(returns),
        "warmup_rows": len(features) - len(X),
        "valid_feature_rows": len(X),
        "prediction_rows": len(result),
        "first_target": str(result["target_date"].min().date()),
        "last_target": str(result["target_date"].max().date()),
        "folds": fold_records,
        "git_commit": git_commit,
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in code_paths
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(csv_path, index=False, date_format="%Y-%m-%d")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nĐã lưu: {csv_path.relative_to(ROOT)}")
    print(f"Đã lưu: {metadata_path.relative_to(ROOT)}")
    print(f"Số dự báo: {len(result)}")
    print("\n5 dự báo cuối:")
    print(result.tail().to_string(index=False))


if __name__ == "__main__":
    main()