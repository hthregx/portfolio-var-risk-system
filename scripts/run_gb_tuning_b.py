from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

from src.gb_market_features import (
    FEATURE_COLUMNS as MARKET_FEATURES,
    build_market_features,
)
from src.gb_return_features import build_return_features


ALPHA = 0.05
SEED = 42
TARGET = "portfolio_simple_return"

VALIDATION_START = pd.Timestamp("2021-12-31")
VALIDATION_END = pd.Timestamp("2024-12-17")

RETURN_FEATURES = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_5",
    "rolling_vol_5",
    "rolling_vol_20",
    "rolling_vol_60",
    "drawdown",
]

FEATURES = RETURN_FEATURES + MARKET_FEATURES

EXPERIMENTS = {
    "G01": (2, 0.05, 100),
    "G05": (2, 0.05, 50),
    "G06": (2, 0.05, 200),
    "G07": (2, 0.10, 100),
}

PREDICTION_COLUMNS = [
    "experiment_id",
    "forecast_date",
    "target_date",
    "actual_return",
    "quantile_return",
    "var",
    "violation",
]


def load_dataset():
    returns = (
        pd.read_csv(
            ROOT / "data/processed/portfolio_returns.csv",
            parse_dates=["date"],
        )
        .set_index("date")[TARGET]
    )

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

    return_features = build_return_features(returns)
    market_features = build_market_features(stocks, market)

    return (
        return_features
        .join(market_features, how="inner")
        .join(returns.rename(TARGET), how="inner")
        .dropna()
    )


def compute_metrics(pred):
    actual = pred["actual_return"].to_numpy()
    q = pred["quantile_return"].to_numpy()

    return {
        "violations": int(pred["violation"].sum()),
        "violation_rate": float(pred["violation"].mean()),
        "pinball_loss": float(
            mean_pinball_loss(actual, q, alpha=ALPHA)
        ),
        "average_var": float(pred["var"].mean()),
    }


def run_experiment(
    experiment_id,
    config,
    train,
    validation,
    all_dates,
):
    depth, learning_rate, n_estimators = config

    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=ALPHA,
        max_depth=depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_samples_leaf=5,
        random_state=SEED,
    )

    start = time.perf_counter()

    model.fit(
        train[FEATURES],
        train[TARGET],
    )

    q = model.predict(
        validation[FEATURES]
    )

    runtime = time.perf_counter() - start

    positions = all_dates.get_indexer(
        validation.index
    )

    if (positions <= 0).any():
        raise RuntimeError("Invalid forecast-date alignment.")

    forecast_dates = all_dates[positions - 1]

    pred = pd.DataFrame({
        "experiment_id": experiment_id,
        "forecast_date": forecast_dates,
        "target_date": validation.index,
        "actual_return": validation[TARGET].to_numpy(),
        "quantile_return": q,
        "var": np.maximum(0.0, -q),
        "violation": (
            validation[TARGET].to_numpy() < q
        ),
    })[PREDICTION_COLUMNS]

    assert np.isfinite(pred["quantile_return"]).all()
    assert np.isfinite(pred["var"]).all()
    assert (pred["var"] >= 0).all()
    assert pred["target_date"].is_unique
    assert (
        pred["forecast_date"] < pred["target_date"]
    ).all()

    result = compute_metrics(pred)

    summary = {
        "experiment_id": experiment_id,
        "alpha": ALPHA,
        "max_depth": depth,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "min_samples_leaf": 5,
        "random_state": SEED,
        "feature_count": len(FEATURES),
        "train_start": train.index.min().date().isoformat(),
        "train_end": train.index.max().date().isoformat(),
        "validation_start":
            validation.index.min().date().isoformat(),
        "validation_end":
            validation.index.max().date().isoformat(),
        "n_train": len(train),
        "n_validation": len(validation),
        **result,
        "quantile_min":
            float(pred["quantile_return"].min()),
        "quantile_max":
            float(pred["quantile_return"].max()),
        "var_min": float(pred["var"].min()),
        "var_max": float(pred["var"].max()),
        "runtime_seconds": runtime,
        "created_at":
            datetime.now(timezone.utc).isoformat(),
    }

    return pred, summary


def run_all(write_outputs=True):
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
            "Empty train or validation set."
        )

    predictions = []
    summaries = []

    for experiment_id, config in EXPERIMENTS.items():
        pred, summary = run_experiment(
            experiment_id,
            config,
            train,
            validation,
            data.index,
        )

        predictions.append(pred)
        summaries.append(summary)

    predictions = pd.concat(
        predictions,
        ignore_index=True,
    )

    summary = pd.DataFrame(summaries)

    if write_outputs:
        predictions.to_csv(
            ROOT / "results/gb_tuning_b_predictions.csv",
            index=False,
        )

        summary.to_csv(
            ROOT / "results/gb_tuning_b.csv",
            index=False,
        )

    return predictions, summary


if __name__ == "__main__":
    predictions, summary = run_all()

    columns = [
        "experiment_id",
        "n_estimators",
        "learning_rate",
        "violations",
        "violation_rate",
        "pinball_loss",
        "average_var",
        "runtime_seconds",
    ]

    print(summary[columns].to_string(index=False))
    print()
    print("prediction_rows:", len(predictions))
    print("B19 G01/G05/G06/G07 tuning: PASS")