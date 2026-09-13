import numpy as np


def _validate_returns(returns):
    try:
        values = np.asarray(returns, dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("Returns must contain numeric values.") from exc

    if values.ndim != 1:
        raise ValueError("Returns must be one-dimensional.")
    if values.size == 0:
        raise ValueError("Returns cannot be empty.")
    if not np.isfinite(values).all():
        raise ValueError("Returns must contain only finite values.")

    return values.copy()


def _validate_alpha(alpha):
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("Alpha must be numeric.") from exc

    if not 0.0 < alpha_value < 1.0:
        raise ValueError("Alpha must be between 0 and 1.")

    return alpha_value


def historical_var_forecast(returns, alpha=0.05):
    values = _validate_returns(returns)
    alpha_value = _validate_alpha(alpha)

    quantile_return = float(np.quantile(values, alpha_value, method="linear"))
    var_value = max(0.0, -quantile_return)

    return {"quantile_return": quantile_return, "var": var_value}
