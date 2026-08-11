from statistics import NormalDist

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


def _validate_decay(decay):
    try:
        decay_value = float(decay)
    except (TypeError, ValueError) as exc:
        raise ValueError("Decay must be numeric.") from exc

    if not 0.0 < decay_value < 1.0:
        raise ValueError("Decay must be between 0 and 1.")

    return decay_value


def ewma_var_forecast(returns, alpha=0.05, decay=0.94):
    values = _validate_returns(returns)
    alpha_value = _validate_alpha(alpha)
    decay_value = _validate_decay(decay)

    variance = float(values[0] ** 2)
    for value in values[1:]:
        variance = decay_value * variance + (1.0 - decay_value) * float(value ** 2)

    volatility = float(np.sqrt(variance))
    quantile_return = float(NormalDist().inv_cdf(alpha_value) * volatility)
    var_value = max(0.0, -quantile_return)

    return {"quantile_return": quantile_return, "var": var_value}
