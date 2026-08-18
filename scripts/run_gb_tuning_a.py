from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.gb_return_features import build_return_features
from src.models.gradient_boosting_var import GradientBoostingVaR


DATA_PATH = Path(
    "data/processed/portfolio_returns.csv"
)

VALIDATION_START = pd.Timestamp("2021-12-31")
VALIDATION_END = pd.Timestamp("2024-12-17")
ALPHA = 0.05


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    max_depth: int
    learning_rate: float
    n_estimators: int


EXPERIMENTS = (
    ExperimentConfig(
        experiment_id="G01",
        max_depth=2,
        learning_rate=0.05,
        n_estimators=100,
    ),
    ExperimentConfig(
        experiment_id="G02",
        max_depth=1,
        learning_rate=0.05,
        n_estimators=100,
    ),
    ExperimentConfig(
        experiment_id="G03",
        max_depth=3,
        learning_rate=0.05,
        n_estimators=100,
    ),
    ExperimentConfig(
        experiment_id="G04",
        max_depth=2,
        learning_rate=0.03,
        n_estimators=100,
    ),
)


def load_feature_target_frame() -> pd.DataFrame:
    """Load canonical returns and build target-aligned features."""
    raw = pd.read_csv(DATA_PATH)

    required = {
        "date",
        "portfolio_simple_return",
    }

    missing = required - set(raw.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    dates = pd.to_datetime(
        raw["date"],
        errors="raise",
    )

    if dates.duplicated().any():
        raise ValueError(
            "Portfolio return dates must be unique."
        )

    if not dates.is_monotonic_increasing:
        raise ValueError(
            "Portfolio return dates must be sorted."
        )

    target = pd.Series(
        pd.to_numeric(
            raw["portfolio_simple_return"],
            errors="raise",
        ).to_numpy(),
        index=pd.DatetimeIndex(dates),
        name="portfolio_simple_return",
        dtype="float64",
    )

    if not np.isfinite(target.to_numpy()).all():
        raise ValueError(
            "Portfolio returns must be finite."
        )

    features = build_return_features(target)

    frame = features.copy()
    frame["target_return"] = target

    return frame.dropna().copy()


def pinball_loss(
    actual: np.ndarray,
    predicted_quantile: np.ndarray,
    alpha: float = ALPHA,
) -> float:
    """Return mean quantile pinball loss."""
    error = actual - predicted_quantile

    loss = np.maximum(
        alpha * error,
        (alpha - 1.0) * error,
    )

    return float(np.mean(loss))


def run_experiment(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Run one expanding-window validation experiment."""
    feature_columns = [
        column
        for column in frame.columns
        if column != "target_return"
    ]

    validation_dates = frame.index[
        (frame.index >= VALIDATION_START)
        & (frame.index <= VALIDATION_END)
    ]

    if len(validation_dates) != 739:
        raise ValueError(
            "Expected exactly 739 validation target dates."
        )

    predictions = []

    start_time = time.perf_counter()

    total_targets = len(validation_dates)

    print(
        f"{config.experiment_id}: starting "
        f"{total_targets} validation targets",
        flush=True,
    )

    for step, target_date in enumerate(
        validation_dates,
        start=1,
    ):
        if (
            step == 1
            or step % 100 == 0
            or step == total_targets
        ):
            print(
                f"{config.experiment_id}: "
                f"{step}/{total_targets} "
                f"target={target_date.date()}",
                flush=True,
            )

        train = frame.loc[
            frame.index < target_date
        ]

        if train.empty:
            raise ValueError(
                "Training data cannot be empty."
            )

        target_row = frame.loc[
            [target_date]
        ]

        model = GradientBoostingVaR(
            alpha=ALPHA,
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            min_samples_leaf=5,
            subsample=1.0,
            random_state=42,
        )

        model.fit(
            train[feature_columns],
            train["target_return"],
        )

        forecast = model.forecast(
            target_row[feature_columns]
        )

        actual_return = float(
            target_row["target_return"].iloc[0]
        )

        quantile_return = float(
            forecast["quantile_return"]
        )

        var_value = float(
            forecast["var"]
        )

        predictions.append(
            {
                "experiment_id": config.experiment_id,
                "forecast_date": train.index[-1],
                "target_date": target_date,
                "n_train": int(len(train)),
                "actual_return": actual_return,
                "quantile_return": quantile_return,
                "var": var_value,
                "violation": bool(
                    actual_return < quantile_return
                ),
            }
        )

    runtime = (
        time.perf_counter()
        - start_time
    )

    prediction_frame = pd.DataFrame(
        predictions
    )

    if not (
        prediction_frame["forecast_date"]
        < prediction_frame["target_date"]
    ).all():
        raise RuntimeError(
            "Forecast dates must precede target dates."
        )

    actual = prediction_frame[
        "actual_return"
    ].to_numpy()

    quantile = prediction_frame[
        "quantile_return"
    ].to_numpy()

    metrics = {
        "experiment_id": config.experiment_id,
        "alpha": ALPHA,
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "n_estimators": config.n_estimators,
        "min_samples_leaf": 5,
        "subsample": 1.0,
        "random_state": 42,
        "validation_rows": int(
            len(prediction_frame)
        ),
        "validation_start": str(
            prediction_frame[
                "target_date"
            ].min().date()
        ),
        "validation_end": str(
            prediction_frame[
                "target_date"
            ].max().date()
        ),
        "first_train_rows": int(
            prediction_frame[
                "n_train"
            ].iloc[0]
        ),
        "last_train_rows": int(
            prediction_frame[
                "n_train"
            ].iloc[-1]
        ),
        "violation_count": int(
            prediction_frame[
                "violation"
            ].sum()
        ),
        "violation_rate": float(
            prediction_frame[
                "violation"
            ].mean()
        ),
        "pinball_loss": pinball_loss(
            actual,
            quantile,
        ),
        "average_var": float(
            prediction_frame[
                "var"
            ].mean()
        ),
        "minimum_var": float(
            prediction_frame[
                "var"
            ].min()
        ),
        "maximum_var": float(
            prediction_frame[
                "var"
            ].max()
        ),
        "runtime_seconds": float(runtime),
    }

    return prediction_frame, metrics


def run_all_experiments() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run the locked G01-G04 validation grid."""
    frame = load_feature_target_frame()

    all_predictions = []
    metrics = []

    for config in EXPERIMENTS:
        prediction_frame, experiment_metrics = (
            run_experiment(
                frame,
                config,
            )
        )

        all_predictions.append(
            prediction_frame
        )

        metrics.append(
            experiment_metrics
        )

    return (
        pd.concat(
            all_predictions,
            ignore_index=True,
        ),
        pd.DataFrame(metrics),
    )


if __name__ == "__main__":
    predictions, summary = run_all_experiments()

    pd.set_option(
        "display.max_columns",
        None,
    )

    print(summary.to_string(index=False))

    print(
        "\nPrediction rows:",
        len(predictions),
    )
