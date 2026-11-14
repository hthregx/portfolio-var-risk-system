"""Build A's feature table from an explicitly selected KBS snapshot."""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.phase2.features_a import ALL_FEATURES, build_features_a


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root", type=Path, required=True
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/phase2/a"),
    )

    args = parser.parse_args()

    base = args.data_root / "data" / "phase2"
    source = base / "portfolio_returns.csv"
    manifest_path = base / "latest_manifest.json"

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    digest = sha256(source)

    if digest != manifest["portfolio_sha256"]:
        raise RuntimeError("Input checksum does not match manifest")

    if manifest["source"] != "KBS":
        raise RuntimeError("Unexpected data source")

    df = pd.read_csv(source, dtype={"date": str})

    if len(df) != manifest["rows"]:
        raise RuntimeError("Unexpected source row count")

    dates = pd.to_datetime(
        df["date"],
        format="%Y-%m-%d",
        errors="raise",
    )

    if (
        dates.isna().any()
        or dates.duplicated().any()
        or not dates.is_monotonic_increasing
    ):
        raise RuntimeError("Invalid trading date ordering")

    if (
        df["date"].iloc[-1]
        != manifest["portfolio_last_date"]
        or df["date"].iloc[-1] != manifest["cutoff"]
    ):
        raise RuntimeError("Data cutoff mismatch")

    r = pd.Series(
        df["portfolio_simple_return"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(dates),
        name="portfolio_simple_return",
    )

    features = build_features_a(r)

    mask = features.notna().all(axis=1)

    if mask.sum() != len(r) - 60:
        raise RuntimeError("Unexpected feature warmup")

    selected = features.loc[mask].copy()

    if not np.isfinite(selected.to_numpy()).all():
        raise RuntimeError("Feature matrix contains nonfinite values")

    previous_date = pd.Series(
        r.index, index=r.index
    ).shift(1)

    output = selected.copy()

    output.insert(
        0, "as_of",
        previous_date.loc[selected.index].to_numpy(),
    )

    output.insert(
        0, "target_date",
        selected.index,
    )

    output["actual_return"] = r.loc[
        selected.index
    ].to_numpy()

    if not (output["as_of"] < output["target_date"]).all():
        raise RuntimeError("Look-ahead alignment detected")

    name = f"features_{manifest['cutoff']}"
    output_dir = args.out_dir.resolve()

    csv_path = output_dir / f"{name}.csv"
    metadata_path = output_dir / f"{name}_metadata.json"

    if csv_path.exists() or metadata_path.exists():
        raise FileExistsError(
            "Feature snapshot already exists; never overwrite."
        )

    # Create only after all input and alignment checks pass.
    output_dir.mkdir(parents=True, exist_ok=True)

    output.to_csv(
        csv_path,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
    )

    if pd.read_csv(csv_path).shape != output.shape:
        raise RuntimeError("Feature export roundtrip failed")

    metadata = {
        "stage": "development_only",
        "owner": "A",
        "cutoff": manifest["cutoff"],
        "source": "KBS",
        "input_sha256": digest,
        "output_sha256": sha256(csv_path),
        "input_rows": len(r),
        "warmup_rows": 60,
        "feature_rows": len(output),
        "feature_count": len(ALL_FEATURES),
        "feature_names": list(ALL_FEATURES),
        "lag_policy": "all inputs strictly before target_date",
        "target": "portfolio_simple_return",
        "alpha": 0.05,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT, text=True,
        ).strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("FEATURE COUNT:", len(ALL_FEATURES))
    print("INPUT ROWS:", len(r))
    print("FEATURE ROWS:", len(output))
    print("FIRST TARGET:", output["target_date"].iloc[0].date())
    print("LAST TARGET:", output["target_date"].iloc[-1].date())
    print("INPUT SHA256:", digest)
    print("OUTPUT SHA256:", metadata["output_sha256"])
    print("CSV:", csv_path)
    print("METADATA:", metadata_path)
    print("A FEATURE ENGINEERING PASS")


if __name__ == "__main__":
    main()
