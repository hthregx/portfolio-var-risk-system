from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ALPHA = 0.05
MODELS = {"historical_w250", "ewma_d094", "gb_G04"}
FOLDS = {f"development_fold_{i}" for i in range(1, 6)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _vector(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Đầu vào phải là vector một chiều, không rỗng.")
    if not np.isfinite(values).all():
        raise ValueError("Đầu vào chứa NaN hoặc vô cực.")
    return values


def _pair(actual, quantile):
    actual = _vector(actual)
    quantile = _vector(quantile)
    if actual.shape != quantile.shape:
        raise ValueError("Actual và quantile phải có cùng số phần tử.")
    return actual, quantile


def pinball_loss(actual, quantile, alpha=ALPHA):
    if (
        isinstance(alpha, (bool, np.bool_))
        or not np.isscalar(alpha)
    ):
        raise ValueError("Alpha phải là một số trong khoảng (0, 1).")

    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Alpha phải trong khoảng (0, 1).")

    actual, quantile = _pair(actual, quantile)
    error = actual - quantile
    return np.maximum(alpha * error, (alpha - 1) * error)


def violation_rate(actual, quantile):
    actual, quantile = _pair(actual, quantile)
    return float(np.mean(actual < quantile))


def average_var(quantile):
    quantile = _vector(quantile)
    return float(np.maximum(0.0, -quantile).mean())


def summarize_metrics(actual, quantile, alpha=ALPHA):
    actual, quantile = _pair(actual, quantile)
    losses = pinball_loss(actual, quantile, alpha)

    return {
        "n_targets": len(actual),
        "n_violations": int(np.sum(actual < quantile)),
        "mean_pinball_loss": float(losses.mean()),
        "violation_rate": violation_rate(actual, quantile),
        "average_var": average_var(quantile),
    }


def validate_table(df):
    required = {
        "model", "fold", "as_of", "target_date",
        "actual_return", "quantile_return", "var_return",
    }
    if not required.issubset(df.columns):
        raise ValueError("Bảng dự báo thiếu cột.")

    if df[list(required)].isna().any().any():
        raise ValueError("Bảng dự báo có ô thiếu.")
    if set(df["model"]) != MODELS or set(df["fold"]) != FOLDS:
        raise ValueError("Model hoặc fold khác lịch baseline.")
    if len(df) != 4110:
        raise ValueError("Bảng dự báo phải có 4110 dòng.")
    if df.duplicated(["model", "target_date"]).any():
        raise ValueError("Trùng dự báo model/target_date.")
    if not (df["as_of"] < df["target_date"]).all():
        raise ValueError("as_of phải trước target_date.")

    numeric = df[
        ["actual_return", "quantile_return", "var_return"]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError("Có giá trị không hữu hạn.")
    if (df["actual_return"] <= -1).any():
        raise ValueError("Lợi suất thực tế không hợp lệ.")

    np.testing.assert_allclose(
        df["var_return"],
        np.maximum(0.0, -df["quantile_return"]),
        rtol=0, atol=1e-12,
        err_msg="Cột var_return không khớp quantile_return.",
    )

    counts = df.groupby(["model", "fold"]).size()
    if len(counts) != 15 or not counts.eq(274).all():
        raise ValueError("Mỗi model/fold phải có 274 dự báo.")

    reference = None
    for model in sorted(MODELS):
        part = df.loc[df["model"] == model].sort_values("target_date")
        part = part.reset_index(drop=True)

        if reference is None:
            reference = part
            continue

        pd.testing.assert_frame_equal(
            part[["target_date", "as_of", "fold"]],
            reference[["target_date", "as_of", "fold"]],
        )
        np.testing.assert_allclose(
            part["actual_return"],
            reference["actual_return"],
            rtol=0, atol=1e-12,
            err_msg=f"{model}: actual_return khác mô hình còn lại.",
        )


def main():
    base = ROOT / "results/phase2/b"
    input_path = base / "baseline_cv.csv"
    input_meta_path = base / "baseline_cv_metadata.json"
    output_path = base / "baseline_metrics_b.csv"
    output_meta_path = base / "baseline_metrics_b_metadata.json"

    for path in (output_path, output_meta_path):
        if path.exists():
            raise FileExistsError(
                f"Đã có {path.name}; kiểm tra bản cũ trước khi chạy lại."
            )

    input_hash = sha256(input_path)
    input_meta = json.loads(
        input_meta_path.read_text(encoding="utf-8")
    )

    if input_hash != input_meta.get("output_sha256"):
        raise ValueError("Hash baseline_cv.csv không khớp metadata.")
    if input_meta.get("stage") != "development_only":
        raise ValueError("Đầu vào không phải bảng phát triển dự kiến.")

    df = pd.read_csv(
        input_path, parse_dates=["as_of", "target_date"]
    )
    validate_table(df)

    rows = []
    for model in sorted(MODELS):
        model_data = df.loc[df["model"] == model]

        groups = [
            (fold, model_data.loc[model_data["fold"] == fold])
            for fold in sorted(FOLDS)
        ]
        groups.append(("all_validation", model_data))

        for fold, part in groups:
            metrics = summarize_metrics(
                part["actual_return"].to_numpy(),
                part["quantile_return"].to_numpy(),
            )
            rows.append({
                "model": model,
                "fold": fold,
                "alpha": ALPHA,
                "first_target": str(part["target_date"].min().date()),
                "last_target": str(part["target_date"].max().date()),
                **metrics,
            })

    result = pd.DataFrame(rows)

    # Kiểm tra tổng số violations giữa từng fold và toàn bộ.
    for model in sorted(MODELS):
        part = result.loc[result["model"] == model]
        total = part.loc[part["fold"] == "all_validation"].iloc[0]
        folds = part.loc[part["fold"] != "all_validation"]

        if int(folds["n_violations"].sum()) != int(total["n_violations"]):
            raise ValueError("Tổng violations giữa các folds không khớp.")

    result.to_csv(output_path, index=False)

    metadata = {
        "stage": "development_only",
        "input_file": str(input_path.relative_to(ROOT)),
        "input_sha256": input_hash,
        "source_data_sha256": input_meta["input_sha256"],
        "cutoff": input_meta["cutoff"],
        "split_status": input_meta["split_status"],
        "alpha": ALPHA,
        "return_unit": "decimal",
        "violation_rate_unit": "fraction",
        "violation_rule": "actual_return < quantile_return",
        "var_rule": "max(0, -quantile_return)",
        "aggregation": "mean_over_observations",
        "rows": len(result),
        "script_sha256": sha256(__file__),
        "output_sha256": sha256(output_path),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    output_meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    overall = result.loc[
        result["fold"] == "all_validation"
    ].sort_values("mean_pinball_loss")

    display = overall[
        ["model", "n_targets", "n_violations", "mean_pinball_loss"]
    ].copy()
    display["violation_rate_pct"] = (
        overall["violation_rate"] * 100
    )
    display["average_var_pct"] = overall["average_var"] * 100

    print("\nMETRICS TRÊN TOÀN BỘ VALIDATION")
    print(display.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSắp xếp theo pinball loss; chưa chốt mô hình Phase 2.")
    print(f"Đã lưu {len(result)} dòng: {output_path.relative_to(ROOT)}")
    print(f"Đã lưu: {output_meta_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()