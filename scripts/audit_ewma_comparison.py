from pathlib import Path

import numpy as np
import pandas as pd


HISTORICAL_PATH = Path(
    "data/processed/historical_var_backtest.csv"
)

EWMA_PATH = Path(
    "data/processed/ewma_var_backtest.csv"
)

SUMMARY_PATH = Path(
    "results/ewma_vs_historical.csv"
)

EXCEPTION_PATH = Path(
    "results/exception_dates.csv"
)

TOLERANCE = 1e-12


def main():
    historical = pd.read_csv(
        HISTORICAL_PATH,
        parse_dates=["forecast_date", "target_date"],
    )

    ewma = pd.read_csv(
        EWMA_PATH,
        parse_dates=["forecast_date", "target_date"],
    )

    summary = pd.read_csv(
        SUMMARY_PATH,
    ).set_index("method")

    exceptions = pd.read_csv(
        EXCEPTION_PATH,
        parse_dates=["target_date"],
    )

    # 1. Same number of comparison rows
    assert len(historical) == len(ewma)
    assert len(historical) == 1387

    # 2. Same target-date sequence
    pd.testing.assert_series_equal(
        historical["target_date"].reset_index(drop=True),
        ewma["target_date"].reset_index(drop=True),
        check_names=False,
    )

    # 3. Same forecast-date sequence
    pd.testing.assert_series_equal(
        historical["forecast_date"].reset_index(drop=True),
        ewma["forecast_date"].reset_index(drop=True),
        check_names=False,
    )

    # 4. Same actual returns
    np.testing.assert_allclose(
        historical["target_return"],
        ewma["target_return"],
        atol=TOLERANCE,
        rtol=0.0,
    )

    # 5. No duplicate target dates
    assert not historical["target_date"].duplicated().any()
    assert not ewma["target_date"].duplicated().any()

    # 6. No NaN
    assert not historical.isna().any().any()
    assert not ewma.isna().any().any()
    assert not summary.isna().any().any()
    assert not exceptions.isna().any().any()

    # 7. VaR must be non-negative
    assert (historical["historical_var"] >= 0.0).all()
    assert (ewma["ewma_var"] >= 0.0).all()

    # 8. Strict violation rule
    historical_violation_check = (
        historical["target_return"]
        < historical["quantile_return"]
    )

    ewma_violation_check = (
        ewma["target_return"]
        < ewma["quantile_return"]
    )

    assert np.array_equal(
        historical["violation"].astype(bool).to_numpy(),
        historical_violation_check.to_numpy(),
    )

    assert np.array_equal(
        ewma["violation"].astype(bool).to_numpy(),
        ewma_violation_check.to_numpy(),
    )

    # 9. Method labels
    assert (ewma["method"] == "ewma").all()

    assert set(summary.index) == {
        "historical",
        "ewma",
    }

    # 10. Summary metrics must match prediction files
    assert int(
        summary.loc["historical", "forecasts"]
    ) == len(historical)

    assert int(
        summary.loc["ewma", "forecasts"]
    ) == len(ewma)

    assert int(
        summary.loc["historical", "violations"]
    ) == int(historical["violation"].sum())

    assert int(
        summary.loc["ewma", "violations"]
    ) == int(ewma["violation"].sum())

    assert abs(
        summary.loc["historical", "violation_rate"]
        - historical["violation"].mean()
    ) <= TOLERANCE

    assert abs(
        summary.loc["ewma", "violation_rate"]
        - ewma["violation"].mean()
    ) <= TOLERANCE

    assert abs(
        summary.loc["historical", "average_var"]
        - historical["historical_var"].mean()
    ) <= TOLERANCE

    assert abs(
        summary.loc["ewma", "average_var"]
        - ewma["ewma_var"].mean()
    ) <= TOLERANCE

    # 11. Canonical counts
    assert int(historical["violation"].sum()) == 75
    assert int(ewma["violation"].sum()) == 73

    # 12. Exception-date consistency
    counts = exceptions[
        "exception_type"
    ].value_counts()

    assert len(exceptions) == 92
    assert counts["both"] == 56
    assert counts["historical_only"] == 19
    assert counts["ewma_only"] == 17

    assert (
        counts["both"]
        + counts["historical_only"]
    ) == 75

    assert (
        counts["both"]
        + counts["ewma_only"]
    ) == 73

    print("Historical rows:", len(historical))
    print("EWMA rows:", len(ewma))
    print(
        "Historical violations:",
        int(historical["violation"].sum()),
    )
    print(
        "EWMA violations:",
        int(ewma["violation"].sum()),
    )
    print(
        "Exception types:",
        counts.to_dict(),
    )
    print()
    print("B-12.3 CONSISTENCY AUDIT PASS")


if __name__ == "__main__":
    main()