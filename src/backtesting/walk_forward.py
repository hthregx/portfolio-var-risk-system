from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


PREDICTION_COLUMNS = [
    "window_end_date",
    "forecast_date",
    "target_date",
    "actual_return",
    "quantile_return",
    "var",
    "violation",
    "method",
    "observations",
    "evaluation_mode",
]


def walk_forward(
    data,
    forecast_function,
    return_col="portfolio_simple_return",
    window_size=250,
    mode="rolling",
    method="model",
):
    """
    Generic one-day-ahead walk-forward runner.

    forecast_function(train_returns) must return:
    {
        "quantile_return": float,
        "var": float,
    }
    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            "Data must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "Data cannot be empty."
        )

    if (
        "date" not in data.columns
        or return_col not in data.columns
    ):
        raise ValueError(
            "Missing required columns."
        )

    if (
        not isinstance(window_size, int)
        or isinstance(window_size, bool)
        or window_size <= 0
    ):
        raise ValueError(
            "window_size must be positive."
        )

    if window_size >= len(data):
        raise ValueError(
            "window_size must be smaller than data length."
        )

    if mode not in {
        "rolling",
        "expanding",
    }:
        raise ValueError(
            "mode must be 'rolling' or 'expanding'."
        )

    if not callable(
        forecast_function
    ):
        raise ValueError(
            "forecast_function must be callable."
        )

    frame = data[
        [
            "date",
            return_col,
        ]
    ].copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="raise",
    )

    # Sort chronologically before walk-forward evaluation.
    frame = (
        frame
        .sort_values("date")
        .reset_index(drop=True)
    )

    if frame[
        "date"
    ].duplicated().any():
        raise ValueError(
            "Duplicate dates are not allowed."
        )

    try:
        frame[
            return_col
        ] = pd.to_numeric(
            frame[
                return_col
            ],
            errors="raise",
        ).astype(
            "float64"
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Returns must be numeric."
        ) from exc

    if frame[
        return_col
    ].isna().any():
        raise ValueError(
            "Returns cannot contain missing values."
        )

    if not np.isfinite(
        frame[
            return_col
        ].to_numpy()
    ).all():
        raise ValueError(
            "Returns must be finite."
        )

    records = []

    start_time = perf_counter()

    for target_index in range(
        window_size,
        len(frame),
    ):
        if mode == "rolling":
            start_index = (
                target_index
                - window_size
            )
        else:
            start_index = 0

        train = frame.iloc[
            start_index:
            target_index
        ]

        target = frame.iloc[
            target_index
        ]

        train_returns = (
            train[
                return_col
            ]
            .to_numpy(
                dtype="float64"
            )
        )

        forecast = (
            forecast_function(
                train_returns
            )
        )

        if not isinstance(
            forecast,
            dict,
        ):
            raise ValueError(
                "Model must return a dictionary."
            )

        if not {
            "quantile_return",
            "var",
        }.issubset(
            forecast
        ):
            raise ValueError(
                "Model output must contain "
                "quantile_return and var."
            )

        quantile_return = float(
            forecast[
                "quantile_return"
            ]
        )

        var_value = float(
            forecast[
                "var"
            ]
        )

        if not np.isfinite(
            [
                quantile_return,
                var_value,
            ]
        ).all():
            raise ValueError(
                "Model output must be finite."
            )

        actual_return = float(
            target[
                return_col
            ]
        )

        window_end_date = (
            train[
                "date"
            ].iloc[-1]
        )

        forecast_date = (
            window_end_date
        )

        target_date = (
            target[
                "date"
            ]
        )

        if not (
            forecast_date
            < target_date
        ):
            raise ValueError(
                "forecast_date must be before target_date."
            )

        records.append(
            {
                "window_end_date":
                    window_end_date,

                "forecast_date":
                    forecast_date,

                "target_date":
                    target_date,

                "actual_return":
                    actual_return,

                "quantile_return":
                    quantile_return,

                "var":
                    var_value,

                "violation":
                    (
                        actual_return
                        < quantile_return
                    ),

                "method":
                    method,

                "observations":
                    len(
                        train
                    ),

                "evaluation_mode":
                    mode,
            }
        )

    result = pd.DataFrame(
        records,
        columns=PREDICTION_COLUMNS,
    )

    result.attrs[
        "runtime_seconds"
    ] = (
        perf_counter()
        - start_time
    )

    return result


def save_predictions(
    predictions,
    output_path,
):
    """
    Save predictions without coupling
    the runner to filesystem logic.
    """

    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        path,
        index=False,
    )

    return path
def walk_forward_model(
    data,
    features,
    model_factory,
    return_col="portfolio_simple_return",
    window_size=250,
    mode="rolling",
    method="gradient_boosting",
):
    """Walk-forward adapter for stateful fit/predict models."""

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame.")

    if not isinstance(features, pd.DataFrame):
        raise ValueError("Features must be a pandas DataFrame.")

    if not callable(model_factory):
        raise ValueError("model_factory must be callable.")

    if mode not in {"rolling", "expanding"}:
        raise ValueError(
            "mode must be 'rolling' or 'expanding'."
        )

    if (
        not isinstance(window_size, int)
        or isinstance(window_size, bool)
        or window_size <= 0
    ):
        raise ValueError("window_size must be positive.")

    if "date" not in data.columns or return_col not in data.columns:
        raise ValueError("Missing required columns.")

    frame = data[["date", return_col]].copy()
    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="raise",
    )

    frame = (
        frame.sort_values("date")
        .set_index("date")
    )

    if frame.index.has_duplicates:
        raise ValueError("Duplicate dates are not allowed.")

    frame[return_col] = pd.to_numeric(
        frame[return_col],
        errors="raise",
    ).astype("float64")

    if (
        frame[return_col].isna().any()
        or not np.isfinite(frame[return_col]).all()
    ):
        raise ValueError(
            "Returns must be finite and non-missing."
        )

    feature_frame = features.copy()

    if not isinstance(
        feature_frame.index,
        pd.DatetimeIndex,
    ):
        feature_frame.index = pd.to_datetime(
            feature_frame.index,
            errors="raise",
        )

    if feature_frame.index.has_duplicates:
        raise ValueError(
            "Feature dates must be unique."
        )

    feature_frame = feature_frame.sort_index()

    common_dates = (
        frame.index
        .intersection(feature_frame.index)
        .sort_values()
    )

    records = []
    start_time = perf_counter()

    for target_date in common_dates:
        target_feature = feature_frame.loc[
            [target_date]
        ]

        if target_feature.isna().any().any():
            continue

        eligible_dates = common_dates[
            common_dates < target_date
        ]

        train_features = feature_frame.loc[
            eligible_dates
        ]

        train_target = frame.loc[
            eligible_dates,
            return_col,
        ]

        valid = (
            train_features.notna().all(axis=1)
            & train_target.notna()
        )

        train_features = train_features.loc[valid]
        train_target = train_target.loc[
            train_features.index
        ]

        if mode == "rolling":
            train_features = train_features.tail(
                window_size
            )
            train_target = train_target.loc[
                train_features.index
            ]

        if len(train_features) < window_size:
            continue

        if not (
            train_features.index < target_date
        ).all():
            raise ValueError(
                "Training rows must precede target date."
            )

        model = model_factory()

        model.fit(
            train_features,
            train_target,
        )

        prediction = model.predict(
            target_feature
        ).iloc[0]

        quantile_return = float(
            prediction["quantile_return"]
        )

        var_value = float(
            prediction["var"]
        )

        if not np.isfinite(
            [quantile_return, var_value]
        ).all():
            raise ValueError(
                "Model output must be finite."
            )

        if var_value < 0.0:
            raise ValueError(
                "VaR must be non-negative."
            )

        earlier_dates = frame.index[
            frame.index < target_date
        ]

        if len(earlier_dates) == 0:
            continue

        forecast_date = earlier_dates[-1]

        if not forecast_date < target_date:
            raise ValueError(
                "forecast_date must be before target_date."
            )

        actual_return = float(
            frame.loc[target_date, return_col]
        )

        records.append({
            "window_end_date": forecast_date,
            "forecast_date": forecast_date,
            "target_date": target_date,
            "actual_return": actual_return,
            "quantile_return": quantile_return,
            "var": var_value,
            "violation":
                actual_return < quantile_return,
            "method": method,
            "observations": len(train_features),
            "evaluation_mode": mode,
        })

    result = pd.DataFrame(
        records,
        columns=PREDICTION_COLUMNS,
    )

    result.attrs["runtime_seconds"] = (
        perf_counter() - start_time
    )

    return result