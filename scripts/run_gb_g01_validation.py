from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

from src.gb_market_features import FEATURE_COLUMNS, build_market_features


EXPERIMENT_ID = "G01"
TARGET = "portfolio_simple_return"
ALPHA = 0.05
RANDOM_STATE = 42

PREDICTION_COLUMNS = [
    "forecast_date",
    "target_date",
    "actual_return",
    "quantile_return",
    "var",
    "violation",
    "experiment_id",
]

VALIDATION_START = pd.Timestamp("2021-12-31")
VALIDATION_END = pd.Timestamp("2024-12-17")


def build_model():
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=ALPHA,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
    )


def load_dataset():
    stocks = {
        ticker: pd.read_csv(
            ROOT / f"data/processed/{ticker}_clean.csv",
            parse_dates=["date"],
        )
        for ticker in ("HPG", "FPT", "MWG")
    }

    market = pd.read_csv(
        ROOT / "data/raw/VNINDEX.csv"
    )

    target = (
        pd.read_csv(
            ROOT / "data/processed/portfolio_returns.csv",
            parse_dates=["date"],
        )
        .set_index("date")[[TARGET]]
    )

    return (
        build_market_features(stocks, market)
        .join(target, how="inner")
        .dropna()
    )


def compute_metrics(pred):
    actual = pred["actual_return"].to_numpy()
    q = pred["quantile_return"].to_numpy()

    return {
        "violations": int(pred["violation"].sum()),
        "violation_rate": float(
            pred["violation"].mean()
        ),
        "pinball_loss": float(
            mean_pinball_loss(
                actual,
                q,
                alpha=ALPHA,
            )
        ),
        "average_var": float(
            pred["var"].mean()
        ),
    }


def sanity_checks(pred, expected_dates):
    q = pred["quantile_return"]
    var = pred["var"]

    assert np.isfinite(q).all()
    assert np.isfinite(var).all()
    assert (var >= 0).all()

    assert len(pred) == len(expected_dates)
    assert pred["target_date"].is_unique

    assert pd.DatetimeIndex(
        pred["target_date"]
    ).equals(
        pd.DatetimeIndex(expected_dates)
    )

    assert (
        pred["forecast_date"]
        < pred["target_date"]
    ).all()

    expected_violation = (
        pred["actual_return"]
        < pred["quantile_return"]
    )

    assert pred["violation"].equals(
        expected_violation
    )

    q1 = q.quantile(0.25)
    q3 = q.quantile(0.75)
    iqr = q3 - q1

    outliers = int(
        (
            (q < q1 - 3 * iqr)
            | (q > q3 + 3 * iqr)
        ).sum()
    )

    return {
        "prediction_count": len(pred),
        "violation_count": int(
            pred["violation"].sum()
        ),
        "quantile_min": float(q.min()),
        "quantile_max": float(q.max()),
        "quantile_mean": float(q.mean()),
        "quantile_median": float(q.median()),
        "quantile_p05": float(
            q.quantile(0.05)
        ),
        "quantile_p95": float(
            q.quantile(0.95)
        ),
        "var_min": float(var.min()),
        "var_max": float(var.max()),
        "obvious_outlier_count": outliers,
    }


def run_g01(write_outputs=True):
    data = load_dataset()

    train = data.loc[
        data.index < VALIDATION_START
    ]

    validation = data.loc[
        (data.index >= VALIDATION_START)
        & (data.index <= VALIDATION_END)
    ]

    if train.empty or validation.empty:
        raise RuntimeError(
            "Empty train/validation set."
        )

    model = build_model()

    start = time.perf_counter()

    model.fit(
        train[FEATURE_COLUMNS],
        train[TARGET],
    )

    q = model.predict(
        validation[FEATURE_COLUMNS]
    )

    runtime = time.perf_counter() - start

    forecast_dates = data.index[
        data.index.get_indexer(
            validation.index
        ) - 1
    ]

    pred = pd.DataFrame({
        "forecast_date": forecast_dates,
        "target_date": validation.index,
        "actual_return": (
            validation[TARGET].to_numpy()
        ),
        "quantile_return": q,
        "var": np.maximum(0.0, -q),
        "violation": (
            validation[TARGET].to_numpy()
            < q
        ),
        "experiment_id": EXPERIMENT_ID,
    })

    pred = pred[PREDICTION_COLUMNS]

    check = sanity_checks(
        pred,
        validation.index,
    )

    result = compute_metrics(pred)

    experiment = {
        "experiment_id": EXPERIMENT_ID,
        "method": "GradientBoostingRegressor_quantile",
        "alpha": ALPHA,
        "feature_set": "|".join(FEATURE_COLUMNS),
        "train_start":
            train.index.min().date().isoformat(),
        "train_end":
            train.index.max().date().isoformat(),
        "validation_start":
            validation.index.min().date().isoformat(),
        "validation_end":
            validation.index.max().date().isoformat(),
        "n_train": len(train),
        "n_validation": len(validation),
        "n_estimators": model.n_estimators,
        "learning_rate": model.learning_rate,
        "max_depth": model.max_depth,
        "min_samples_leaf": model.min_samples_leaf,
        "random_state": model.random_state,
        **result,
        "runtime_seconds": runtime,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "target": TARGET,
        "alpha": ALPHA,
        "horizon": 1,
        "feature_names": FEATURE_COLUMNS,
        "model_family": "GradientBoostingRegressor",
        "package_version": sklearn.__version__,
        "random_seed": RANDOM_STATE,
        "validation_boundary": {
            "start": experiment[
                "validation_start"
            ],
            "end": experiment[
                "validation_end"
            ],
        },
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "runtime_seconds": runtime,
        "reserved_later_used": False,
        "average_var_selection_policy":
            "descriptive_only_not_standalone_selection",
        **result,
        **check,
    }

    if write_outputs:
        pred.to_csv(
            ROOT / "results/gb_g01_predictions.csv",
            index=False,
        )

        pd.DataFrame(
            [experiment]
        ).to_csv(
            ROOT / "results/gb_experiment_log.csv",
            index=False,
        )

        (
            ROOT / "results/gb_g01_metadata.json"
        ).write_text(
            json.dumps(
                metadata,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return pred, experiment, metadata


if __name__ == "__main__":
    pred, exp, meta = run_g01()

    print("G01 end-to-end: PASS")
    print("n_train:", exp["n_train"])
    print("n_validation:", exp["n_validation"])
    print("violations:", exp["violations"])
    print("violation_rate:", exp["violation_rate"])
    print("pinball_loss:", exp["pinball_loss"])
    print("average_var:", exp["average_var"])
    print("runtime_seconds:", exp["runtime_seconds"])
    print(
        "prediction_count:",
        meta["prediction_count"],
    )
    print(
        "quantile_range:",
        meta["quantile_min"],
        meta["quantile_max"],
    )
    print(
        "var_range:",
        meta["var_min"],
        meta["var_max"],
    )
    print(
        "obvious_outliers:",
        meta["obvious_outlier_count"],
    )