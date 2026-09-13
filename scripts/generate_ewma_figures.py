from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HISTORICAL_PATH = Path(
    "data/processed/historical_var_backtest.csv"
)

EWMA_PATH = Path(
    "data/processed/ewma_var_backtest.csv"
)

FIGURE_DIR = Path("figures/var")
EXCEPTION_PATH = Path("results/exception_dates.csv")


def load_data():
    historical = pd.read_csv(
        HISTORICAL_PATH,
        parse_dates=[
            "forecast_date",
            "target_date",
        ],
    )

    ewma = pd.read_csv(
        EWMA_PATH,
        parse_dates=[
            "forecast_date",
            "target_date",
        ],
    )

    return historical, ewma


def build_exception_summary(
    historical,
    ewma,
):
    comparison = pd.DataFrame(
        {
            "target_date": historical["target_date"],
            "actual_return": historical["target_return"],
            "historical_quantile": historical["quantile_return"],
            "ewma_quantile": ewma["quantile_return"],
            "historical_violation": (
                historical["violation"].astype(bool)
            ),
            "ewma_violation": (
                ewma["violation"].astype(bool)
            ),
        }
    )

    comparison = comparison.loc[
        comparison[
            [
                "historical_violation",
                "ewma_violation",
            ]
        ].any(axis=1)
    ].copy()

    def classify(row):
        if (
            row["historical_violation"]
            and row["ewma_violation"]
        ):
            return "both"

        if row["historical_violation"]:
            return "historical_only"

        return "ewma_only"

    comparison["exception_type"] = comparison.apply(
        classify,
        axis=1,
    )

    return comparison


def plot_backtest(
    frame,
    title,
    output_path,
):
    plt.figure(figsize=(14, 6))

    plt.plot(
        frame["target_date"],
        frame["target_return"],
        label="Actual return",
        linewidth=0.8,
    )

    plt.plot(
        frame["target_date"],
        frame["quantile_return"],
        label="VaR threshold",
        linewidth=1.0,
    )

    exceptions = frame.loc[
        frame["violation"].astype(bool)
    ]

    plt.scatter(
        exceptions["target_date"],
        exceptions["target_return"],
        label="Violation",
        s=20,
    )

    plt.axhline(
        0.0,
        linewidth=0.8,
    )

    plt.title(title)
    plt.xlabel("Target date")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def plot_overlay(
    historical,
    ewma,
    output_path,
):
    plt.figure(figsize=(14, 6))

    plt.plot(
        historical["target_date"],
        historical["target_return"],
        label="Actual return",
        linewidth=0.7,
    )

    plt.plot(
        historical["target_date"],
        historical["quantile_return"],
        label="Historical threshold",
        linewidth=1.0,
    )

    plt.plot(
        ewma["target_date"],
        ewma["quantile_return"],
        label="EWMA threshold",
        linewidth=1.0,
    )

    plt.axhline(
        0.0,
        linewidth=0.8,
    )

    plt.title(
        "Historical Simulation vs EWMA VaR"
    )

    plt.xlabel("Target date")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def main():
    historical, ewma = load_data()

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXCEPTION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exception_summary = build_exception_summary(
        historical,
        ewma,
    )

    exception_summary.to_csv(
        EXCEPTION_PATH,
        index=False,
    )

    plot_backtest(
        historical,
        "Historical Simulation VaR Backtest",
        FIGURE_DIR
        / "10_historical_var_backtest_release.png",
    )

    plot_backtest(
        ewma.rename(
            columns={
                "target_return": "target_return",
            }
        ),
        "EWMA VaR Backtest",
        FIGURE_DIR
        / "11_ewma_var_backtest.png",
    )

    plot_overlay(
        historical,
        ewma,
        FIGURE_DIR
        / "12_ewma_vs_historical.png",
    )

    print(
        "Exceptions:",
        len(exception_summary),
    )

    print(
        exception_summary[
            "exception_type"
        ].value_counts()
    )

    print(
        f"Saved: {EXCEPTION_PATH}"
    )


if __name__ == "__main__":
    main()