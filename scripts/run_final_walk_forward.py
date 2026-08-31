from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

from src.evaluation.final_metrics import compute_final_metrics

from src.backtesting.walk_forward import (
    PREDICTION_COLUMNS,
    walk_forward,
)
from src.gb_return_features import build_return_features
from src.models.ewma_var import ewma_var_forecast
from src.models.gradient_boosting_var import GradientBoostingVaR
from src.models.historical_var import historical_var_forecast


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "final_evaluation.yaml"
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "processed" / "portfolio_returns.csv"

RETURN_COLUMN = "portfolio_simple_return"
TARGET_COLUMN = "target_return"

METHOD_HISTORICAL = "historical_simulation"
METHOD_EWMA = "ewma"
METHOD_GRADIENT_BOOSTING = "gradient_boosting"


def require(condition: bool, message: str) -> None:
    """Raise a clear error when an evaluation invariant fails."""
    if not condition:
        raise ValueError(message)


def load_config(path: Path) -> dict:
    """Load the Day-21 evaluation contract."""
    require(path.is_file(), f"Config file not found: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    require(
        isinstance(config, dict),
        "Evaluation config must contain a YAML mapping.",
    )

    required_sections = {
        "contract",
        "evaluation_period",
        "rules",
        "historical_simulation",
        "ewma",
        "gradient_boosting",
        "reproducibility",
    }

    missing_sections = required_sections - set(config)

    require(
        not missing_sections,
        f"Missing config sections: {sorted(missing_sections)}",
    )

    return config


def load_portfolio_returns(path: Path) -> pd.DataFrame:
    """Load and validate canonical portfolio simple returns."""
    require(path.is_file(), f"Portfolio return file not found: {path}")

    raw = pd.read_csv(path)

    required_columns = {
        "date",
        RETURN_COLUMN,
    }

    missing_columns = required_columns - set(raw.columns)

    require(
        not missing_columns,
        f"Missing portfolio columns: {sorted(missing_columns)}",
    )

    frame = raw[
        [
            "date",
            RETURN_COLUMN,
        ]
    ].copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="raise",
    )

    require(
        not frame["date"].duplicated().any(),
        "Portfolio return dates must be unique.",
    )

    require(
        frame["date"].is_monotonic_increasing,
        "Portfolio return dates must be chronologically sorted.",
    )

    frame[RETURN_COLUMN] = pd.to_numeric(
        frame[RETURN_COLUMN],
        errors="raise",
    ).astype("float64")

    require(
        not frame[RETURN_COLUMN].isna().any(),
        "Portfolio returns cannot contain missing values.",
    )

    require(
        np.isfinite(frame[RETURN_COLUMN].to_numpy()).all(),
        "Portfolio returns must contain only finite values.",
    )

    return frame.reset_index(drop=True)


def build_gb_frame(portfolio_data: pd.DataFrame) -> pd.DataFrame:
    """Create target-date-aligned Gradient Boosting features."""
    returns = pd.Series(
        portfolio_data[RETURN_COLUMN].to_numpy(dtype="float64"),
        index=pd.DatetimeIndex(portfolio_data["date"]),
        name=RETURN_COLUMN,
        dtype="float64",
    )

    features = build_return_features(returns)

    frame = features.copy()
    frame[TARGET_COLUMN] = returns

    return frame.dropna().copy()


def get_evaluation_dates(
    portfolio_data: pd.DataFrame,
    config: dict,
) -> pd.DatetimeIndex:
    """Return the frozen common Day-21 evaluation target dates."""
    start = pd.Timestamp(
        config["evaluation_period"]["start"]
    )
    end = pd.Timestamp(
        config["evaluation_period"]["end"]
    )

    dates = pd.DatetimeIndex(
        portfolio_data.loc[
            portfolio_data["date"].between(start, end),
            "date",
        ]
    )

    require(
        len(dates) > 0,
        "Evaluation period contains no target dates.",
    )

    require(
        dates.is_unique,
        "Evaluation target dates must be unique.",
    )

    require(
        dates.is_monotonic_increasing,
        "Evaluation target dates must be sorted.",
    )

    return dates


def validate_common_universe(
    portfolio_data: pd.DataFrame,
    gb_frame: pd.DataFrame,
    evaluation_dates: pd.DatetimeIndex,
    config: dict,
) -> None:
    """Validate history and common target-date eligibility."""
    gb_evaluation_dates = gb_frame.index[
        gb_frame.index.isin(evaluation_dates)
    ]

    require(
        gb_evaluation_dates.equals(evaluation_dates),
        "Gradient Boosting target dates do not match the common universe.",
    )

    evaluation_start = evaluation_dates[0]

    return_history_rows = int(
        (portfolio_data["date"] < evaluation_start).sum()
    )

    gb_history_rows = int(
        (gb_frame.index < evaluation_start).sum()
    )

    historical_window = int(
        config["historical_simulation"]["window"]
    )

    require(
        return_history_rows >= historical_window,
        "Insufficient return history for Historical Simulation.",
    )

    require(
        gb_history_rows > 0,
        "Gradient Boosting has no training rows before evaluation.",
    )


def _filter_common_dates(
    predictions: pd.DataFrame,
    evaluation_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Restrict model predictions to the exact common evaluation dates."""
    filtered = predictions.loc[
        predictions["target_date"].isin(evaluation_dates)
    ].copy()

    filtered = (
        filtered
        .sort_values("target_date")
        .reset_index(drop=True)
    )

    require(
        pd.DatetimeIndex(filtered["target_date"]).equals(
            evaluation_dates
        ),
        "Model output does not cover the exact common target-date universe.",
    )

    return filtered


def run_historical(
    portfolio_data: pd.DataFrame,
    evaluation_dates: pd.DatetimeIndex,
    config: dict,
) -> tuple[pd.DataFrame, float]:
    """Run the retained Historical Simulation baseline."""
    model_config = config["historical_simulation"]

    alpha = float(model_config["alpha"])
    window = int(model_config["window"])
    mode = str(model_config["mode"])

    def forecast_function(train_returns: np.ndarray) -> dict[str, float]:
        return historical_var_forecast(
            train_returns,
            alpha=alpha,
        )

    started = perf_counter()

    predictions = walk_forward(
        data=portfolio_data,
        forecast_function=forecast_function,
        return_col=RETURN_COLUMN,
        window_size=window,
        mode=mode,
        method=METHOD_HISTORICAL,
    )

    runtime_seconds = perf_counter() - started

    return (
        _filter_common_dates(
            predictions,
            evaluation_dates,
        ),
        runtime_seconds,
    )


def run_ewma(
    portfolio_data: pd.DataFrame,
    evaluation_dates: pd.DatetimeIndex,
    config: dict,
) -> tuple[pd.DataFrame, float]:
    """Run the retained canonical EWMA baseline."""
    model_config = config["ewma"]

    alpha = float(model_config["alpha"])
    decay = float(model_config["decay"])
    mode = str(model_config["mode"])

    start_after = int(
        config["historical_simulation"]["window"]
    )

    def forecast_function(train_returns: np.ndarray) -> dict[str, float]:
        return ewma_var_forecast(
            train_returns,
            alpha=alpha,
            decay=decay,
        )

    started = perf_counter()

    predictions = walk_forward(
        data=portfolio_data,
        forecast_function=forecast_function,
        return_col=RETURN_COLUMN,
        window_size=start_after,
        mode=mode,
        method=METHOD_EWMA,
    )

    runtime_seconds = perf_counter() - started

    return (
        _filter_common_dates(
            predictions,
            evaluation_dates,
        ),
        runtime_seconds,
    )


def build_gb_model(config: dict) -> GradientBoostingVaR:
    """Construct the configured Gradient Boosting candidate."""
    model_config = config["gradient_boosting"]

    return GradientBoostingVaR(
        alpha=float(model_config["alpha"]),
        n_estimators=int(model_config["n_estimators"]),
        learning_rate=float(model_config["learning_rate"]),
        max_depth=int(model_config["max_depth"]),
        min_samples_leaf=int(model_config["min_samples_leaf"]),
        subsample=float(model_config["subsample"]),
        random_state=int(model_config["random_state"]),
    )


def run_gradient_boosting(
    gb_frame: pd.DataFrame,
    evaluation_dates: pd.DatetimeIndex,
    config: dict,
) -> tuple[pd.DataFrame, float]:
    """Run expanding-refit GB quantile forecasts on common dates."""
    model_config = config["gradient_boosting"]
    feature_columns = list(model_config["features"])

    require(
        all(column in gb_frame.columns for column in feature_columns),
        "Configured Gradient Boosting features are missing from the feature frame.",
    )

    records: list[dict] = []

    started = perf_counter()
    total_targets = len(evaluation_dates)

    for step, target_date in enumerate(
        evaluation_dates,
        start=1,
    ):
        if (
            step == 1
            or step % 50 == 0
            or step == total_targets
        ):
            print(
                f"Gradient Boosting: {step}/{total_targets} "
                f"target={target_date.date()}",
                flush=True,
            )

        train = gb_frame.loc[
            gb_frame.index < target_date
        ]

        target_row = gb_frame.loc[
            [target_date]
        ]

        require(
            not train.empty,
            f"No GB training rows before {target_date.date()}.",
        )

        require(
            len(target_row) == 1,
            f"Expected one GB target row for {target_date.date()}.",
        )

        model = build_gb_model(config)

        model.fit(
            train[feature_columns],
            train[TARGET_COLUMN],
        )

        forecast = model.forecast(
            target_row[feature_columns]
        )

        actual_return = float(
            target_row[TARGET_COLUMN].iloc[0]
        )

        quantile_return = float(
            forecast["quantile_return"]
        )

        var_value = float(
            forecast["var"]
        )

        require(
            np.isfinite(
                [
                    actual_return,
                    quantile_return,
                    var_value,
                ]
            ).all(),
            f"Non-finite GB output at {target_date.date()}.",
        )

        forecast_date = train.index[-1]

        require(
            forecast_date < target_date,
            "GB forecast date must be strictly before target date.",
        )

        records.append(
            {
                "window_end_date": forecast_date,
                "forecast_date": forecast_date,
                "target_date": target_date,
                "actual_return": actual_return,
                "quantile_return": quantile_return,
                "var": var_value,
                "violation": actual_return < quantile_return,
                "method": METHOD_GRADIENT_BOOSTING,
                "observations": len(train),
                "evaluation_mode": "expanding",
            }
        )

    runtime_seconds = perf_counter() - started

    predictions = pd.DataFrame(
        records,
        columns=PREDICTION_COLUMNS,
    )

    require(
        pd.DatetimeIndex(predictions["target_date"]).equals(
            evaluation_dates
        ),
        "GB predictions do not match the common target dates.",
    )

    return predictions, runtime_seconds


def validate_method_alignment(
    prediction_frames: list[pd.DataFrame],
    evaluation_dates: pd.DatetimeIndex,
) -> None:
    """Check basic cross-method date and realized-return alignment."""
    require(
        len(prediction_frames) == 3,
        "Exactly three model prediction frames are required.",
    )

    reference = prediction_frames[0].set_index("target_date")

    require(
        reference.index.equals(evaluation_dates),
        "Reference method target dates are incorrect.",
    )

    for frame in prediction_frames[1:]:
        indexed = frame.set_index("target_date")

        require(
            indexed.index.equals(reference.index),
            "Methods do not share identical target dates.",
        )

        require(
            np.allclose(
                indexed["actual_return"].to_numpy(dtype="float64"),
                reference["actual_return"].to_numpy(dtype="float64"),
                rtol=0.0,
                atol=1e-15,
            ),
            "Actual returns differ across methods.",
        )


def print_dry_run_summary(
    portfolio_data: pd.DataFrame,
    gb_frame: pd.DataFrame,
    evaluation_dates: pd.DatetimeIndex,
    config: dict,
) -> None:
    """Print the evaluation plan without fitting any model."""
    evaluation_start = evaluation_dates[0]

    raw_history = int(
        (portfolio_data["date"] < evaluation_start).sum()
    )

    gb_history = int(
        (gb_frame.index < evaluation_start).sum()
    )

    print("")
    print("=== DAY 21 FINAL WALK-FORWARD DRY RUN ===")
    print("Contract status       :", config["contract"]["status"])
    print("Evaluation label      :", config["evaluation_period"]["label"])
    print("Evaluation start      :", evaluation_dates[0].date())
    print("Evaluation end        :", evaluation_dates[-1].date())
    print("Evaluation targets    :", len(evaluation_dates))
    print("Raw history rows      :", raw_history)
    print("GB history rows       :", gb_history)
    print(
        "Historical window     :",
        config["historical_simulation"]["window"],
    )
    print(
        "EWMA decay            :",
        config["ewma"]["decay"],
    )
    print(
        "GB experiment         :",
        config["gradient_boosting"]["experiment_id"],
    )
    print(
        "GB feature count      :",
        len(config["gradient_boosting"]["features"]),
    )
    print(
        "GB random state       :",
        config["gradient_boosting"]["random_state"],
    )
    print("Common universe       : PASS")
    print("Model fitting executed: NO")
    print("")
    print("PASS - DRY RUN CONTRACT AND DATA CHECK COMPLETE")


def run_all(
    portfolio_data: pd.DataFrame,
    gb_frame: pd.DataFrame,
    evaluation_dates: pd.DatetimeIndex,
    config: dict,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run all three Day-21 evaluation methods."""
    historical, historical_runtime = run_historical(
        portfolio_data,
        evaluation_dates,
        config,
    )

    ewma, ewma_runtime = run_ewma(
        portfolio_data,
        evaluation_dates,
        config,
    )

    gradient_boosting, gb_runtime = run_gradient_boosting(
        gb_frame,
        evaluation_dates,
        config,
    )

    frames = [
        historical,
        ewma,
        gradient_boosting,
    ]

    validate_method_alignment(
        frames,
        evaluation_dates,
    )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    runtimes = {
        METHOD_HISTORICAL: historical_runtime,
        METHOD_EWMA: ewma_runtime,
        METHOD_GRADIENT_BOOSTING: gb_runtime,
    }

    return combined, runtimes



FINAL_PREDICTION_COLUMNS = [
    "forecast_date",
    "target_date",
    "method",
    "actual_return",
    "quantile_return",
    "var",
    "violation",
    "runtime_seconds",
    "config_id",
]


def build_config_ids(config: dict) -> dict[str, str]:
    """Return stable identifiers for the three evaluated configurations."""
    return {
        METHOD_HISTORICAL: (
            f"historical_w"
            f"{int(config['historical_simulation']['window'])}"
        ),
        METHOD_EWMA: (
            "ewma_d"
            f"{float(config['ewma']['decay']):.2f}"
            .replace(".", "")
        ),
        METHOD_GRADIENT_BOOSTING: (
            "gb_"
            f"{config['gradient_boosting']['experiment_id']}"
        ),
    }


def build_final_predictions(
    predictions: pd.DataFrame,
    *,
    runtimes: dict[str, float],
    config_ids: dict[str, str],
) -> pd.DataFrame:
    """Convert common walk-forward output into the final CSV contract."""
    required_columns = {
        "forecast_date",
        "target_date",
        "method",
        "actual_return",
        "quantile_return",
        "var",
        "violation",
    }

    missing = required_columns - set(predictions.columns)

    if missing:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing)}"
        )

    methods = set(
        predictions["method"].astype(str)
    )

    if set(runtimes) != methods:
        raise ValueError(
            "Runtime keys must exactly match prediction methods."
        )

    if set(config_ids) != methods:
        raise ValueError(
            "Config ID keys must exactly match prediction methods."
        )

    frame = predictions.copy()

    frame["forecast_date"] = pd.to_datetime(
        frame["forecast_date"],
        errors="raise",
    )

    frame["target_date"] = pd.to_datetime(
        frame["target_date"],
        errors="raise",
    )

    if frame[
        ["method", "target_date"]
    ].duplicated().any():
        raise ValueError(
            "Duplicate method/target-date pairs found."
        )

    if not (
        frame["forecast_date"]
        <
        frame["target_date"]
    ).all():
        raise ValueError(
            "forecast_date must be before target_date."
        )

    expected_var = np.maximum(
        0.0,
        -frame[
            "quantile_return"
        ].to_numpy(dtype="float64"),
    )

    if not np.allclose(
        frame["var"].to_numpy(dtype="float64"),
        expected_var,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError(
            "Prediction table violates VaR sign convention."
        )

    expected_violation = (
        frame[
            "actual_return"
        ].to_numpy(dtype="float64")
        <
        frame[
            "quantile_return"
        ].to_numpy(dtype="float64")
    )

    if not np.array_equal(
        frame["violation"].to_numpy(dtype=bool),
        expected_violation,
    ):
        raise ValueError(
            "Prediction table violates strict violation rule."
        )

    runtime_values = {}

    for method, runtime in runtimes.items():
        value = float(runtime)

        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Invalid runtime for {method}."
            )

        runtime_values[method] = value

    frame["runtime_seconds"] = (
        frame["method"]
        .astype(str)
        .map(runtime_values)
    )

    frame["config_id"] = (
        frame["method"]
        .astype(str)
        .map(config_ids)
    )

    if frame[
        ["runtime_seconds", "config_id"]
    ].isna().any().any():
        raise ValueError(
            "Failed to attach runtime/config metadata."
        )

    frame = (
        frame[
            FINAL_PREDICTION_COLUMNS
        ]
        .sort_values(
            ["target_date", "method"]
        )
        .reset_index(drop=True)
    )

    return frame


def _file_sha256(path: Path) -> str:
    """Return SHA256 for a local file."""
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for chunk in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _package_version(package_name: str) -> str | None:
    """Return installed package version when available."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _git_value(*args: str) -> str | None:
    """Return a Git value without making metadata generation fatal."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None

    value = completed.stdout.strip()

    return value or None


def build_run_metadata(
    *,
    config: dict,
    config_path: Path,
    data_path: Path,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    runtimes: dict[str, float],
    config_ids: dict[str, str],
) -> dict:
    """Build reproducibility metadata for the Day-21 final run."""
    evaluation = config["evaluation_period"]
    contract = config["contract"]
    historical = config["historical_simulation"]
    ewma = config["ewma"]
    gb = config["gradient_boosting"]

    return {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "git": {
            "commit": _git_value(
                "rev-parse",
                "HEAD",
            ),
            "branch": _git_value(
                "branch",
                "--show-current",
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "packages": {
                "numpy": _package_version("numpy"),
                "pandas": _package_version("pandas"),
                "PyYAML": _package_version("PyYAML"),
                "scikit-learn": _package_version(
                    "scikit-learn"
                ),
            },
        },
        "source": {
            "data_path": str(
                data_path.resolve().relative_to(
                    REPO_ROOT.resolve()
                )
            ),
            "data_sha256": _file_sha256(
                data_path
            ),
            "config_path": str(
                config_path.resolve().relative_to(
                    REPO_ROOT.resolve()
                )
            ),
            "config_sha256": _file_sha256(
                config_path
            ),
        },
        "contract": {
            "confidence_level": float(
                contract["confidence_level"]
            ),
            "alpha": float(
                contract["alpha"]
            ),
            "forecast_horizon": int(
                contract["forecast_horizon"]
            ),
            "target": contract["target"],
            "portfolio": contract["portfolio"],
            "violation_rule": config[
                "rules"
            ]["violation"],
            "var_rule": config[
                "rules"
            ]["var"],
        },
        "evaluation": {
            "start": evaluation["start"],
            "end": evaluation["end"],
            "label": evaluation["label"],
            "target_count": int(
                predictions[
                    "target_date"
                ].nunique()
            ),
            "used_for_parameter_selection": bool(
                evaluation[
                    "used_for_parameter_selection"
                ]
            ),
            "pristine_untouched_test_claim": bool(
                evaluation[
                    "pristine_untouched_test_claim"
                ]
            ),
        },
        "methods": {
            METHOD_HISTORICAL: {
                "config_id": config_ids[
                    METHOD_HISTORICAL
                ],
                "window": int(
                    historical["window"]
                ),
                "mode": historical["mode"],
                "alpha": float(
                    historical["alpha"]
                ),
            },
            METHOD_EWMA: {
                "config_id": config_ids[
                    METHOD_EWMA
                ],
                "decay": float(
                    ewma["decay"]
                ),
                "mode": ewma["mode"],
                "alpha": float(
                    ewma["alpha"]
                ),
                "initialization": ewma[
                    "initialization"
                ],
                "distribution": ewma[
                    "distribution"
                ],
                "mean_assumption": ewma[
                    "mean_assumption"
                ],
            },
            METHOD_GRADIENT_BOOSTING: {
                "config_id": config_ids[
                    METHOD_GRADIENT_BOOSTING
                ],
                "experiment_id": gb[
                    "experiment_id"
                ],
                "selection_status": gb[
                    "selection_status"
                ],
                "model_frozen": bool(
                    gb["model_frozen"]
                ),
                "loss": gb["loss"],
                "alpha": float(
                    gb["alpha"]
                ),
                "n_estimators": int(
                    gb["n_estimators"]
                ),
                "learning_rate": float(
                    gb["learning_rate"]
                ),
                "max_depth": int(
                    gb["max_depth"]
                ),
                "min_samples_leaf": int(
                    gb["min_samples_leaf"]
                ),
                "subsample": float(
                    gb["subsample"]
                ),
                "random_state": int(
                    gb["random_state"]
                ),
                "training_protocol": gb[
                    "training_protocol"
                ],
                "features": list(
                    gb["features"]
                ),
            },
        },
        "runtime": {
            "semantics": (
                "Total runtime_seconds returned by each "
                "method adapter for this run; repeated per "
                "method in final_predictions.csv. Runtime is "
                "reported for reproducibility/operations and "
                "is not a predictive-quality ranking metric."
            ),
            "seconds_by_method": {
                method: float(runtime)
                for method, runtime
                in runtimes.items()
            },
        },
        "artifacts": {
            "prediction_rows": int(
                len(predictions)
            ),
            "metric_rows": int(
                len(metrics)
            ),
        },
    }


def _atomic_write_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """Write a CSV via a temporary sibling file."""
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    frame.to_csv(
        temporary_path,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
    )

    temporary_path.replace(path)


def _atomic_write_json(
    payload: dict,
    path: Path,
) -> None:
    """Write JSON via a temporary sibling file."""
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(path)


def write_final_artifacts(
    *,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    metadata: dict,
    output_dir: Path,
) -> dict[str, Path]:
    """Write the three canonical Day-21 result artifacts."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "predictions": (
            output_dir
            / "final_predictions.csv"
        ),
        "metrics": (
            output_dir
            / "final_metrics.csv"
        ),
        "metadata": (
            output_dir
            / "final_run_metadata.json"
        ),
    }

    _atomic_write_csv(
        predictions,
        paths["predictions"],
    )

    _atomic_write_csv(
        metrics,
        paths["metrics"],
    )

    _atomic_write_json(
        metadata,
        paths["metadata"],
    )

    return paths

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Day-21 common walk-forward evaluation "
            "for Historical Simulation, EWMA, and Gradient Boosting."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Day-21 evaluation YAML config.",
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to canonical portfolio returns.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate config, data, features, and common dates "
            "without fitting models."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = load_config(
        args.config.resolve()
    )

    portfolio_data = load_portfolio_returns(
        args.data.resolve()
    )

    gb_frame = build_gb_frame(
        portfolio_data
    )

    evaluation_dates = get_evaluation_dates(
        portfolio_data,
        config,
    )

    validate_common_universe(
        portfolio_data,
        gb_frame,
        evaluation_dates,
        config,
    )

    if args.dry_run:
        print_dry_run_summary(
            portfolio_data,
            gb_frame,
            evaluation_dates,
            config,
        )
        return 0

    predictions, runtimes = run_all(
        portfolio_data,
        gb_frame,
        evaluation_dates,
        config,
    )

    print("")
    print("=== WALK-FORWARD RUN COMPLETE ===")
    print("Prediction rows:", len(predictions))

    for method, runtime in runtimes.items():
        print(
            f"{method} runtime_seconds: {runtime:.6f}"
        )

    config_ids = build_config_ids(
        config
    )

    final_predictions = build_final_predictions(
        predictions,
        runtimes=runtimes,
        config_ids=config_ids,
    )

    final_metrics = compute_final_metrics(
        predictions,
        runtimes=runtimes,
        config_ids=config_ids,
        alpha=float(config["contract"]["alpha"]),
    )

    metadata = build_run_metadata(
        config=config,
        config_path=args.config.resolve(),
        data_path=args.data.resolve(),
        predictions=final_predictions,
        metrics=final_metrics,
        runtimes=runtimes,
        config_ids=config_ids,
    )

    artifact_paths = write_final_artifacts(
        predictions=final_predictions,
        metrics=final_metrics,
        metadata=metadata,
        output_dir=REPO_ROOT / "results",
    )

    print("")
    print("=== FINAL ARTIFACTS WRITTEN ===")

    for name, path in artifact_paths.items():
        print(
            f"{name}: "
            f"{path.relative_to(REPO_ROOT)}"
        )

    print("")
    print(
        "PASS - FINAL WALK-FORWARD ARTIFACT EXPORT COMPLETE"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
