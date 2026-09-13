import json
import subprocess
import time
from pathlib import Path

import pandas as pd


SUMMARY_PATH = Path("results/ewma_vs_historical.csv")
EXCEPTION_PATH = Path("results/exception_dates.csv")

FIGURES = [
    Path("figures/var/10_historical_var_backtest_release.png"),
    Path("figures/var/11_ewma_var_backtest.png"),
    Path("figures/var/12_ewma_vs_historical.png"),
]

OUTPUT_PATH = Path(
    "results/ewma_comparison_metadata.json"
)


def get_commit_hash():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main():
    start = time.perf_counter()

    summary = pd.read_csv(SUMMARY_PATH)
    exceptions = pd.read_csv(EXCEPTION_PATH)

    assert len(summary) == 2
    assert len(exceptions) == 92

    for path in FIGURES:
        if not path.exists():
            raise FileNotFoundError(path)

    runtime_seconds = (
        time.perf_counter() - start
    )

    metadata = {
        "alpha": 0.05,
        "confidence_level": 0.95,
        "forecast_horizon": 1,
        "historical": {
            "window_size": 250,
            "evaluation_mode": "rolling",
        },
        "ewma": {
            "decay": 0.94,
            "evaluation_mode": "expanding",
            "initialization": (
                "first_squared_return"
            ),
            "distribution": "normal",
            "mean_assumption": "zero",
        },
        "rerun_commands": [
            (
                "python "
                "scripts/generate_ewma_comparison.py"
            ),
            (
                "python "
                "scripts/generate_ewma_figures.py"
            ),
            (
                "python "
                "scripts/audit_ewma_comparison.py"
            ),
            (
                "python "
                "scripts/generate_ewma_metadata.py"
            ),
        ],
        "commit_hash": get_commit_hash(),
        "runtime_seconds": runtime_seconds,
        "generated_paths": [
            "results/ewma_vs_historical.csv",
            "results/exception_dates.csv",
            (
                "figures/var/"
                "10_historical_var_backtest_release.png"
            ),
            (
                "figures/var/"
                "11_ewma_var_backtest.png"
            ),
            (
                "figures/var/"
                "12_ewma_vs_historical.png"
            ),
        ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()