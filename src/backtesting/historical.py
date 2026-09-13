import numpy as np
import pandas as pd

from src.models.historical_var import historical_var_forecast


ROLLING_COLUMNS = [
    "window_start_date",
    "window_end_date",
    "forecast_date",
    "target_date",
    "observations",
    "quantile_return",
    "historical_var",
    "target_return",
]


def calculate_rolling_historical_var(
    data: pd.DataFrame,
    return_col: str = "portfolio_simple_return",
    window_size: int = 250,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Generate one-day-ahead rolling Historical Simulation VaR forecasts."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            "Data must be provided as a pandas DataFrame."
        )

    if data.empty:
        raise ValueError("Data cannot be empty.")

    required_columns = {"date", return_col}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if (
        not isinstance(window_size, int)
        or isinstance(window_size, bool)
        or window_size < 2
    ):
        raise ValueError(
            "Window size must be an integer greater than or equal to 2."
        )

    normalized_data = data[["date", return_col]].copy()

    normalized_data["date"] = pd.to_datetime(
        normalized_data["date"],
        errors="raise",
    )

    normalized_data = (
        normalized_data
        .sort_values("date")
        .reset_index(drop=True)
    )

    if normalized_data["date"].duplicated().any():
        raise ValueError("Duplicate dates are not allowed.")

    try:
        normalized_data[return_col] = pd.to_numeric(
            normalized_data[return_col],
            errors="raise",
        ).astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Returns must contain only numeric values."
        ) from exc

    if normalized_data[return_col].isna().any():
        raise ValueError(
            "Returns cannot contain missing values."
        )

    if not np.isfinite(
        normalized_data[return_col].to_numpy()
    ).all():
        raise ValueError(
            "Returns must contain only finite values."
        )

    if len(normalized_data) <= window_size:
        raise ValueError(
            "Data must contain more observations than the rolling window."
        )

    try:
        confidence_value = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Confidence level must be numeric."
        ) from exc

    if (
        not np.isfinite(confidence_value)
        or confidence_value <= 0.0
        or confidence_value >= 1.0
    ):
        raise ValueError(
            "Confidence level must be finite and strictly between 0 and 1."
        )

    alpha = 1.0 - confidence_value
    forecast_records = []

    for target_index in range(window_size, len(normalized_data)):
        window_start_index = target_index - window_size

        estimation_window = normalized_data.iloc[
            window_start_index:target_index
        ]
        target_row = normalized_data.iloc[target_index]

        if target_index in estimation_window.index:
            raise RuntimeError(
                "Look-ahead detected: target observation is inside the estimation window."
            )

        estimation_returns = estimation_window[return_col]

        forecast = historical_var_forecast(
            estimation_returns.to_numpy(dtype="float64"),
            alpha=alpha,
        )

        window_start_date = estimation_window["date"].iloc[0]
        window_end_date = estimation_window["date"].iloc[-1]

        forecast_records.append(
            {
                "window_start_date": window_start_date,
                "window_end_date": window_end_date,
                "forecast_date": window_end_date,
                "target_date": target_row["date"],
                "observations": len(estimation_window),
                "quantile_return": forecast["quantile_return"],
                "historical_var": forecast["var"],
                "target_return": float(target_row[return_col]),
            }
        )

    return pd.DataFrame(
        forecast_records,
        columns=ROLLING_COLUMNS,
    )
