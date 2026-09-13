from pathlib import Path

import numpy as np
import pandas as pd


ALPHA = 0.05

HISTORICAL_PATH = Path(
    "data/processed/historical_var_backtest.csv"
)

EWMA_PATH = Path(
    "data/processed/ewma_var_backtest.csv"
)

OUTPUT_PATH = Path(
    "results/ewma_vs_historical.csv"
)


def pinball_loss(actual, quantile, alpha):
    error = actual - quantile

    losses = np.maximum(
        alpha * error,
        (alpha - 1.0) * error,
    )

    return float(losses.mean())


def summarize(
    frame,
    method,
    var_column,
):
    actual = frame["target_return"].to_numpy()
    quantile = frame["quantile_return"].to_numpy()
    var = frame[var_column].to_numpy()
    violation = frame["violation"].astype(bool)

    return {
        "method": method,
        "forecasts": len(frame),
        "violations": int(violation.sum()),
        "violation_rate": float(violation.mean()),
        "pinball_loss": pinball_loss(
            actual,
            quantile,
            ALPHA,
        ),
        "average_var": float(var.mean()),
        "minimum_var": float(var.min()),
        "maximum_var": float(var.max()),
        "first_target_date": frame["target_date"].iloc[0],
        "last_target_date": frame["target_date"].iloc[-1],
    }


def main():
    historical = pd.read_csv(
        HISTORICAL_PATH,
        parse_dates=["target_date"],
    )

    ewma = pd.read_csv(
        EWMA_PATH,
        parse_dates=["target_date"],
    )

    if len(historical) != len(ewma):
        raise ValueError(
            "Historical and EWMA row counts differ."
        )

    if not historical["target_date"].equals(
        ewma["target_date"]
    ):
        raise ValueError(
            "Historical and EWMA target dates differ."
        )

    np.testing.assert_allclose(
        historical["target_return"],
        ewma["target_return"],
        atol=1e-12,
        rtol=0.0,
    )

    rows = [
        summarize(
            historical,
            method="historical",
            var_column="historical_var",
        ),
        summarize(
            ewma,
            method="ewma",
            var_column="ewma_var",
        ),
    ]

    comparison = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(comparison.to_string(index=False))
    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()