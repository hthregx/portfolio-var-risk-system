"""B tính lại metrics độc lập và đối chiếu bảng A/B."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / (
    "results/phase2/b/reference_a_1a593ed/"
    "results/phase2/a/training_2026-11-06"
)
B = ROOT / "results/phase2/b/aligned_2026-11-06_20261112_121348"
OUT = ROOT / "results/phase2/b/metrics_check_2026-11-06"
TOL = 1e-12
FIELDS = ["n", "violations", "violation_rate", "pinball_loss", "average_var"]
GB = {"gb_G04_7f", "gbv2_12f_c1", "gbv2_12f_c2"}
BASELINES = {"historical_w250", "ewma_d094"}
hashes = {}


def read(path):
    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    return pd.read_csv(path, float_precision="round_trip")


def normalize(df, stage):
    df = df.copy()
    df["model"] = df["model"].str.removesuffix("_a")
    if stage == "holdout":
        df["fold"] = "holdout_fixed_fit"
    else:
        df["fold"] = df["fold"].map(
            lambda x: x if str(x).startswith("dev_")
            else f"dev_{int(x)}"
        )
    return df


def calculate(df, stage, owner):
    df = normalize(df, stage)
    expected_models = (
        GB if stage == "cv" else GB - {"gbv2_12f_c2"}
    )
    if owner == "B":
        expected_models = expected_models | BASELINES
    if set(df["model"]) != expected_models:
        raise ValueError(f"{owner}/{stage}: sai tập mô hình")

    for col in ("target_date", "as_of"):
        df[col] = pd.to_datetime(df[col], errors="raise")
    if df[["target_date", "as_of", "fold"]].isna().any().any():
        raise ValueError("Thiếu ngày hoặc fold")
    if df.duplicated(["model", "target_date"]).any():
        raise ValueError("Trùng ngày dự báo")
    if not (df["as_of"] < df["target_date"]).all():
        raise ValueError("as_of không trước target_date")

    values = df[["actual_return", "quantile_return", "var_return"]]
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Có giá trị không hữu hạn")

    np.testing.assert_allclose(
        df["var_return"], np.maximum(0, -df["quantile_return"]),
        rtol=0, atol=TOL,
    )
    if "violation" in df:
        flags = df["violation"].astype(str).str.lower()
        if not flags.isin(["true", "false"]).all():
            raise ValueError("Cột violation không hợp lệ")
        if not np.array_equal(
            flags.eq("true"),
            df["actual_return"] < df["quantile_return"],
        ):
            raise ValueError("Cột violation sai công thức")

    reference = None
    for model, part in df.groupby("model"):
        part = part.sort_values("target_date")
        keys = part[["target_date", "as_of", "fold"]].reset_index(drop=True)
        actual = part["actual_return"].to_numpy()
        if reference is None:
            reference = (keys, actual)
        else:
            pd.testing.assert_frame_equal(keys, reference[0])
            np.testing.assert_allclose(
                actual, reference[1], rtol=0, atol=TOL
            )
        counts = part.groupby("fold").size()
        expected_folds = (
            {f"dev_{i}" for i in range(1, 6)}
            if stage == "cv" else {"holdout_fixed_fit"}
        )
        if set(counts.index) != expected_folds:
            raise ValueError(f"{model}: sai folds")
        if not counts.eq(196 if stage == "cv" else 468).all():
            raise ValueError(f"{model}: sai số dự báo")

    rows = []
    for (model, fold), part in df.groupby(["model", "fold"]):
        y = part["actual_return"].to_numpy()
        q = part["quantile_return"].to_numpy()
        residual = y - q
        loss = np.where(residual >= 0, 0.05 * residual, -0.95 * residual)
        rate = float(np.mean(y < q))
        rows.append({
            "model": model, "fold": fold,
            "n": len(part),
            "violations": int(np.sum(y < q)),
            "violation_rate": rate,
            "coverage": 1 - rate,
            "pinball_loss": float(loss.mean()),
            "average_var": float(np.maximum(0, -q).mean()),
        })
    return pd.DataFrame(rows), df


def compare(left, right, label, stage):
    keys = ["model", "fold"]
    joined = left.merge(
        right, on=keys, how="outer", suffixes=("_left", "_right"),
        indicator=True, validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"{label}: thiếu hoặc thừa model/fold")

    rows = []
    for _, row in joined.iterrows():
        errors = 0
        record = {
            "stage": stage, "comparison": label,
            "model": row["model"], "fold": row["fold"],
        }
        for field in FIELDS:
            x, y = row[f"{field}_left"], row[f"{field}_right"]
            if not np.isfinite([x, y]).all():
                raise ValueError(f"{label}: metrics không hữu hạn")
            difference = abs(float(x) - float(y))
            limit = 0 if field in ("n", "violations") else TOL
            errors += int(difference > limit)
            record[f"{field}_abs_difference"] = difference
        record["errors"] = errors
        record["status"] = "PASS" if errors == 0 else "FAIL"
        rows.append(record)
    return rows


def main():
    if OUT.exists():
        raise FileExistsError(f"Đã có kết quả, không ghi đè: {OUT}")

    checks, calculated = [], []
    for stage in ("cv", "holdout"):
        metrics, predictions = {}, {}
        for owner, folder in (("A", A), ("B", B)):
            result, pred = calculate(
                read(folder / f"{stage}_predictions.csv"), stage, owner
            )
            published = normalize(
                read(folder / f"{stage}_metrics.csv"), stage
            )
            checks.extend(compare(
                result, published, f"{owner}_recomputed_vs_published", stage
            ))
            metrics[owner], predictions[owner] = result, pred
            calculated.append(result.assign(owner=owner, stage=stage))

        # Kiểm actual, ngày và fold giữa A/B trước khi so metrics.
        for model in metrics["A"]["model"].unique():
            a = predictions["A"].query("model == @model").sort_values("target_date")
            b = predictions["B"].query("model == @model").sort_values("target_date")
            cols = ["target_date", "as_of", "fold"]
            pd.testing.assert_frame_equal(
                a[cols].reset_index(drop=True),
                b[cols].reset_index(drop=True),
            )
            np.testing.assert_allclose(
                a["actual_return"], b["actual_return"], rtol=0, atol=TOL
            )

        common_b = metrics["B"].loc[
            metrics["B"]["model"].isin(metrics["A"]["model"])
        ]
        checks.extend(compare(
            metrics["A"], common_b, "A_vs_B_recomputed", stage
        ))

    audit = pd.DataFrame(checks)
    OUT.mkdir(parents=True)
    audit.to_csv(OUT / "metric_table_b.csv", index=False, float_format="%.17g")
    pd.concat(calculated, ignore_index=True).to_csv(
        OUT / "recomputed_metrics.csv", index=False, float_format="%.17g"
    )
    metadata = {
        "alpha": 0.05,
        "tolerance": TOL,
        "errors_definition": "number of metric comparisons exceeding tolerance",
        "source_commit_a": "1a593ed24f3b388faf4e9533c362e747bb21507a",
        "input_sha256": hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "comparison_rows": len(audit),
        "total_errors": int(audit["errors"].sum()),
        "status": "PASS" if audit["errors"].eq(0).all() else "FAIL",
    }
    (OUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(audit.groupby(["stage", "comparison", "status"]).size().to_string())
    print("\nTổng lỗi:", metadata["total_errors"])
    print("Kết quả:", metadata["status"])
    print("Đã lưu:", OUT.relative_to(ROOT))
    if metadata["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()