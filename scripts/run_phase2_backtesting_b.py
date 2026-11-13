"""Backtest B's aligned holdout predictions without retraining."""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import platform

import numpy as np
import pandas as pd

from src.phase2.backtesting_b import backtest_summary


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SOURCE_HASH = (
    "6389586ae8f1edbabb31760e0b555619186"
    "114b3f77d09212362fc2e1c40aadf"
)

MODELS = [
    "historical_w250",
    "ewma_d094",
    "gb_G04_7f",
    "gbv2_12f_c1",
]

FEATURES_12 = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_5",
    "rolling_vol_5",
    "rolling_vol_20",
    "rolling_vol_60",
    "drawdown",
    "momentum_5",
    "momentum_20",
    "downside_vol_20",
    "rolling_q05_60",
    "rolling_range_20",
]

ALPHA = 0.05
SIGNIFICANCE = 0.05
TOLERANCE = 1e-12


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def display_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_verified_csv(base, name, metadata, **kwargs):
    path = base / name
    record = metadata["artifacts"][name]

    require(
        sha256(path) == record["sha256"],
        f"Hash không khớp metadata: {path}",
    )

    frame = pd.read_csv(
        path,
        float_precision="round_trip",
        **kwargs,
    )
    require(
        len(frame) == record["rows"],
        f"Số dòng không khớp metadata: {name}",
    )
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "results/phase2/b/"
            "aligned_2026-11-06_20261112_121348"
        ),
    )
    parser.add_argument(
        "--portfolio",
        type=Path,
        default=Path(
            "data/phase2/reference_a/portfolio_returns.csv"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/phase2/b/backtesting_2026-11-06"
        ),
    )
    args = parser.parse_args()

    base = resolve_path(args.input_dir)
    portfolio_path = resolve_path(args.portfolio)
    out = resolve_path(args.out)

    if out.exists():
        raise FileExistsError(
            f"Đã có thư mục {out}. "
            "Giữ nguyên bản cũ; dùng --out với tên mới nếu cần chạy lại."
        )

    metadata_path = base / "metadata.json"
    meta = json.loads(
        metadata_path.read_text(encoding="utf-8-sig")
    )

    require(meta["owner"] == "B", "Đầu vào không phải kết quả B.")
    require(
        meta["source_sha256"] == EXPECTED_SOURCE_HASH,
        "Snapshot khác bộ A/B đã đối chiếu.",
    )
    require(
        meta["snapshot_cutoff"] == "2026-11-06",
        "Cutoff không đúng.",
    )
    require(
        meta["training_cutoff"] == "2024-12-17",
        "Training cutoff không đúng.",
    )
    require(
        meta["holdout_evaluated"] is True
        and meta["holdout_rows"] == 468,
        "Metadata holdout không đúng.",
    )
    require(
        meta["selected_v2"] == "gbv2_12f_c1",
        "Ứng viên GBv2 khác C1 đã khóa.",
    )
    require(
        meta["models"]["gbv2_12f_c1"]["features"] == FEATURES_12,
        "GBv2 C1 không có đúng bộ 12 đặc trưng.",
    )

    for model in ("gb_G04_7f", "gbv2_12f_c1"):
        require(
            meta["models"][model]["parameters"]["alpha"] == ALPHA,
            f"{model}: alpha khác 0.05.",
        )

    df = read_verified_csv(
        base,
        "holdout_predictions.csv",
        meta,
        parse_dates=["as_of", "target_date"],
    )

    required = [
        "model", "fold", "as_of", "target_date",
        "actual_return", "quantile_return", "var_return",
    ]
    require(
        set(required).issubset(df.columns),
        "Predictions thiếu cột.",
    )
    require(
        not df[required].isna().any().any(),
        "Predictions có giá trị thiếu.",
    )
    require(
        set(df["model"]) == set(MODELS),
        "Danh sách mô hình khác dự kiến.",
    )
    require(
        set(df["fold"]) == {"holdout_fixed_fit"},
        "Đầu vào không phải holdout_fixed_fit.",
    )
    require(
        not df.duplicated(["model", "target_date"]).any(),
        "Trùng model/target_date.",
    )

    for column in ("as_of", "target_date"):
        require(
            pd.api.types.is_datetime64_any_dtype(df[column]),
            f"{column}: ngày không hợp lệ.",
        )

    require(
        (df["as_of"] < df["target_date"]).all(),
        "Có as_of không trước target_date.",
    )

    numeric_columns = [
        "actual_return", "quantile_return", "var_return"
    ]
    require(
        np.isfinite(
            df[numeric_columns].to_numpy(dtype=float)
        ).all(),
        "Predictions chứa NaN hoặc vô cực.",
    )
    require(
        (df["actual_return"] > -1).all(),
        "Actual simple return không hợp lệ.",
    )
    np.testing.assert_allclose(
        df["var_return"],
        np.maximum(0.0, -df["quantile_return"]),
        rtol=0,
        atol=TOLERANCE,
    )

    # Independently check target dates, actual returns and prior session.
    require(
        sha256(portfolio_path) == EXPECTED_SOURCE_HASH,
        "CSV portfolio không khớp snapshot đã khóa.",
    )
    portfolio = pd.read_csv(
        portfolio_path,
        parse_dates=["date"],
        float_precision="round_trip",
    )
    require(
        {"date", "portfolio_simple_return"}.issubset(portfolio.columns),
        "Portfolio thiếu cột.",
    )
    dates = portfolio["date"]
    require(
        pd.api.types.is_datetime64_any_dtype(dates),
        "Ngày portfolio không hợp lệ.",
    )
    require(
        not dates.isna().any()
        and not dates.duplicated().any()
        and dates.is_monotonic_increasing,
        "Ngày portfolio thiếu, trùng hoặc chưa tăng dần.",
    )

    portfolio_values = portfolio["portfolio_simple_return"].to_numpy()
    require(
        np.isfinite(portfolio_values).all()
        and (portfolio_values > -1).all(),
        "Lợi suất portfolio không hợp lệ.",
    )

    truth = pd.DataFrame({
        "target_date": dates,
        "expected_as_of": dates.shift(1),
        "expected_actual": portfolio["portfolio_simple_return"],
    })
    truth = truth.loc[
        truth["target_date"].between(
            pd.Timestamp("2024-12-18"),
            pd.Timestamp("2026-11-06"),
        )
    ].reset_index(drop=True)

    require(len(truth) == 468, "Snapshot không có đúng 468 ngày holdout.")
    require(
        truth["target_date"].iloc[0] == pd.Timestamp("2024-12-18")
        and truth["target_date"].iloc[-1] == pd.Timestamp("2026-11-06"),
        "Sai biên thời gian holdout.",
    )

    summary_rows = []
    flag_tables = []

    for model in MODELS:
        part = (
            df.loc[df["model"] == model]
            .sort_values("target_date")
            .reset_index(drop=True)
        )

        require(len(part) == 468, f"{model}: khác 468 dự báo.")
        require(
            part["target_date"].equals(truth["target_date"]),
            f"{model}: ngày mục tiêu không khớp snapshot.",
        )
        require(
            part["as_of"].equals(truth["expected_as_of"]),
            f"{model}: as_of không phải phiên trước trong snapshot.",
        )
        np.testing.assert_allclose(
            part["actual_return"],
            truth["expected_actual"],
            rtol=0,
            atol=TOLERANCE,
        )

        actual = part["actual_return"].to_numpy(dtype=float)
        quantile = part["quantile_return"].to_numpy(dtype=float)
        stats = backtest_summary(
            actual,
            quantile,
            alpha=ALPHA,
            significance=SIGNIFICANCE,
        )

        summary_rows.append({
            "model": model,
            "evaluation": "retrospective_holdout",
            "first_target": "2024-12-18",
            "last_target": "2026-11-06",
            **stats,
        })

        flags = part[required].copy()
        flags["violation"] = (actual < quantile).astype(int)
        flags["covered"] = 1 - flags["violation"]
        flags["cumulative_violations"] = flags["violation"].cumsum()
        flag_tables.append(flags)

    summary = pd.DataFrame(summary_rows)
    all_flags = pd.concat(flag_tables, ignore_index=True)

    # Check independently recomputed metrics against the aligned run.
    published = read_verified_csv(
        base, "holdout_metrics.csv", meta
    )
    require(
        set(published["model"]) == set(MODELS)
        and not published["model"].duplicated().any(),
        "Bảng metrics tham chiếu thiếu hoặc trùng model.",
    )
    require(
        set(published["fold"]) == {"holdout_fixed_fit"},
        "Bảng metrics tham chiếu sai phạm vi.",
    )

    published = published.set_index("model")
    current = summary.set_index("model")

    metric_mapping = {
        "n_targets": "n",
        "n_violations": "violations",
        "violation_rate": "violation_rate",
        "mean_pinball_loss": "pinball_loss",
        "average_var": "average_var",
    }

    for model in MODELS:
        for computed_name, published_name in metric_mapping.items():
            actual_value = current.loc[model, computed_name]
            expected_value = published.loc[model, published_name]

            if computed_name in ("n_targets", "n_violations"):
                require(
                    actual_value == expected_value,
                    f"{model}: khác {computed_name}.",
                )
            else:
                np.testing.assert_allclose(
                    actual_value,
                    expected_value,
                    rtol=0,
                    atol=TOLERANCE,
                    err_msg=f"{model}: khác {computed_name}",
                )

    # All checks complete before creating output files.
    out.mkdir(parents=True, exist_ok=False)

    summary_path = out / "backtesting.csv"
    flags_path = out / "violation_flags.csv"

    summary.to_csv(
        summary_path, index=False, float_format="%.17g", mode="x"
    )
    all_flags.to_csv(
        flags_path,
        index=False,
        float_format="%.17g",
        date_format="%Y-%m-%d",
        mode="x",
    )

    output_meta = {
        "owner": "B",
        "task": "B-06.2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": "retrospective_holdout",
        "snapshot_cutoff": meta["snapshot_cutoff"],
        "training_cutoff": meta["training_cutoff"],
        "first_target": "2024-12-18",
        "last_target": "2026-11-06",
        "n_targets_per_model": 468,
        "models": MODELS,
        "primary_plot_models": [
            "historical_w250", "ewma_d094", "gbv2_12f_c1"
        ],
        "gbv2_features": FEATURES_12,
        "quantile_alpha": ALPHA,
        "test_significance": SIGNIFICANCE,
        "violation_rule": "actual_return < quantile_return",
        "coverage_rule": "1 - violation_rate",
        "var_rule": "max(0, -quantile_return)",
        "kupiec_distribution": "chi_square_df1_asymptotic",
        "metrics_crosscheck": "PASS",
        "metrics_absolute_tolerance": TOLERANCE,
        "actual_returns_and_dates_check": "PASS",
        "as_of_check": "previous_observed_session_in_locked_snapshot",
        "selection": meta["selection"],
        "gb_refit": meta["gb_refit"],
        "hs_ewma_protocol": meta["hs_ewma_protocol"],
        "caveats": [
            meta["caveat"],
            "No model selection or tuning performed on this holdout.",
            "Kupiec does not test independence of violations.",
            "Failure to reject H0 does not establish model correctness.",
            "Date checks do not replace feature/training leakage tests.",
        ],
        "input_files": {
            display_path(base / "holdout_predictions.csv"):
                sha256(base / "holdout_predictions.csv"),
            display_path(base / "holdout_metrics.csv"):
                sha256(base / "holdout_metrics.csv"),
            display_path(metadata_path): sha256(metadata_path),
            display_path(portfolio_path): sha256(portfolio_path),
        },
        "code_sha256": {
            "scripts/run_phase2_backtesting_b.py": sha256(__file__),
            "src/phase2/backtesting_b.py": sha256(
                ROOT / "src/phase2/backtesting_b.py"
            ),
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "artifacts": {
            "backtesting.csv": {
                "rows": len(summary),
                "sha256": sha256(summary_path),
            },
            "violation_flags.csv": {
                "rows": len(all_flags),
                "sha256": sha256(flags_path),
            },
        },
    }

    with (out / "metadata.json").open("x", encoding="utf-8") as file:
        json.dump(output_meta, file, indent=2, ensure_ascii=False)

    columns = [
        "model", "n_targets", "n_violations",
        "violation_rate", "coverage", "mean_pinball_loss",
        "average_var", "kupiec_lr", "kupiec_p_value",
        "kupiec_decision",
    ]
    print("\nBACKTESTING — 468 NGÀY HOLDOUT")
    print(summary[columns].to_string(
        index=False, float_format=lambda value: f"{value:.9f}"
    ))
    print("\nHash, ngày, actual returns và as_of: PASS")
    print("Đối chiếu metrics với lần chạy aligned: PASS")
    print("Đã lưu:", display_path(out))
    print("Đánh giá hồi cứu; không dùng để lựa chọn lại mô hình.")


if __name__ == "__main__":
    main()