from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PRED = ROOT / "results/final_predictions.csv"
EXCEPTIONS = ROOT / "results/final_exception_analysis.csv"
REGIMES = ROOT / "results/final_regime_analysis.csv"
FIGURES = ROOT / "figures"

METHODS = {
    "historical_simulation": "Historical",
    "ewma": "EWMA",
    "gradient_boosting": "GB",
}


def save(name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close()


def plot_var_timeline():
    df = pd.read_csv(PRED, parse_dates=["target_date"])

    actual = (
        df.groupby("target_date")["actual_return"]
        .first()
        .sort_index()
    )

    q = df.pivot(
        index="target_date",
        columns="method",
        values="quantile_return",
    ).sort_index()

    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label="Actual return", linewidth=1)

    for method, label in METHODS.items():
        plt.plot(
            q.index,
            q[method],
            label=f"{label} q0.05",
            linewidth=1,
        )

    plt.axhline(0, linewidth=0.7)
    plt.title("Portfolio Return and 5% VaR Quantile Forecasts")
    plt.xlabel("Target date")
    plt.ylabel("Return")
    plt.legend()
    save("final_var_forecast_timeline.png")


def plot_exception_timing():
    df = pd.read_csv(
        EXCEPTIONS,
        parse_dates=["target_date"],
    )

    categories = [
        ("shared_all_3", "Shared all 3", 6),
        ("shared_exactly_2", "Shared exactly 2", 5),
        ("historical_only", "Historical only", 4),
        ("ewma_only", "EWMA only", 3),
        ("gb_only", "GB only", 2),
    ]

    plt.figure(figsize=(12, 5))

    for category, label, y in categories:
        dates = df.loc[
            df["exception_category"] == category,
            "target_date",
        ]

        plt.scatter(
            dates,
            [y] * len(dates),
            label=label,
            s=35,
        )

    plt.yticks(
        [2, 3, 4, 5, 6],
        [
            "GB only",
            "EWMA only",
            "Historical only",
            "Shared exactly 2",
            "Shared all 3",
        ],
    )

    plt.title("VaR Exception Timing: Shared and Exclusive Violations")
    plt.xlabel("Target date")
    plt.ylabel("Exception category")
    plt.legend()
    save("final_exception_timing.png")


def plot_regime_comparison():
    df = pd.read_csv(REGIMES)

    pivot = df.pivot(
        index="regime",
        columns="method",
        values="violation_rate",
    ).reindex(["LOW", "NORMAL", "HIGH"])

    ax = pivot.rename(columns=METHODS).plot(
        kind="bar",
        figsize=(9, 5),
    )

    ax.axhline(
        0.05,
        linestyle="--",
        linewidth=1,
        label="5% reference",
    )

    ax.set_title("Violation Rate by Volatility Regime")
    ax.set_xlabel("Volatility regime")
    ax.set_ylabel("Violation rate")
    ax.legend()
    plt.xticks(rotation=0)

    save("final_regime_comparison.png")


def main():
    plot_var_timeline()
    plot_exception_timing()
    plot_regime_comparison()

    print("B4 final figures: PASS")

    for name in (
        "final_var_forecast_timeline.png",
        "final_exception_timing.png",
        "final_regime_comparison.png",
    ):
        print("saved:", FIGURES / name)


if __name__ == "__main__":
    main()