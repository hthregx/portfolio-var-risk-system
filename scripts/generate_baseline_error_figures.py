from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path("results/baseline_error_analysis.csv")
CASE_PATH = Path("results/baseline_case_studies.csv")
OUTPUT_DIR = Path("figures/baseline_error_analysis")

EXPECTED_FIGURES = [
    "01_baseline_var_vs_returns.png",
    "02_exception_timeline.png",
    "03_exception_severity.png",
    "04_exception_clusters.png",
    "05_case_study_risk_response.png",
]


def save_figure(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")

# QA gates

def validate_data(
    df: pd.DataFrame,
    cases: pd.DataFrame,
) -> None:
    """Validate canonical B-15.4 inputs."""

    assert len(df) == 1387
    assert df["target_date"].is_unique
    assert not df.isna().any().any()

    historical_expected = (
        df["target_return"] < df["historical_quantile"]
    )
    ewma_expected = (
        df["target_return"] < df["ewma_quantile"]
    )

    assert np.array_equal(
        historical_expected.to_numpy(),
        df["historical_violation"].astype(bool).to_numpy(),
    )
    assert np.array_equal(
        ewma_expected.to_numpy(),
        df["ewma_violation"].astype(bool).to_numpy(),
    )

    assert df["historical_violation"].sum() == 75
    assert df["ewma_violation"].sum() == 73

    shared = (
        df["historical_violation"]
        & df["ewma_violation"]
    )
    assert shared.sum() == 56

    assert (df["historical_var"] >= 0).all()
    assert (df["ewma_var"] >= 0).all()

    assert 5 <= len(cases) <= 10
    assert cases["target_date"].is_unique
    assert set(cases["target_date"]).issubset(
        set(df["target_date"])
    )



# Figure 01
# VaR thresholds vs realized returns

def figure_01(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(
        df["target_date"],
        df["target_return"],
        linewidth=0.8,
        label="Realized portfolio return",
    )
    ax.plot(
        df["target_date"],
        df["historical_quantile"],
        linewidth=1.0,
        label="Historical quantile",
    )
    ax.plot(
        df["target_date"],
        df["ewma_quantile"],
        linewidth=1.0,
        label="EWMA quantile",
    )

    violations = df[
        df["historical_violation"]
        | df["ewma_violation"]
    ]

    ax.scatter(
        violations["target_date"],
        violations["target_return"],
        s=18,
        marker="x",
        label="At least one baseline violation",
        zorder=5,
    )

    ax.axhline(0.0, linewidth=0.7)
    ax.set_title(
        "Baseline one-day-ahead VaR thresholds "
        "vs realized returns"
    )
    ax.set_xlabel("Target date")
    ax.set_ylabel("Return")
    ax.legend(loc="best", fontsize=8)

    fig.autofmt_xdate()
    save_figure(
        fig,
        "01_baseline_var_vs_returns.png",
    )

# Figure 02
# Exception timeline
def figure_02(df: pd.DataFrame) -> None:
    historical = df[df["historical_violation"]]
    ewma = df[df["ewma_violation"]]

    fig, ax = plt.subplots(figsize=(13, 4))

    ax.scatter(
        historical["target_date"],
        np.ones(len(historical)),
        marker="|",
        s=100,
        label="Historical violation",
    )
    ax.scatter(
        ewma["target_date"],
        np.full(len(ewma), 2.0),
        marker="|",
        s=100,
        label="EWMA violation",
    )

    ax.set_yticks(
        [1, 2],
        labels=["Historical", "EWMA"],
    )
    ax.set_ylim(0.5, 2.5)
    ax.set_title("Baseline exception timeline")
    ax.set_xlabel("Target date")
    ax.legend(loc="best", fontsize=8)

    fig.autofmt_xdate()
    save_figure(
        fig,
        "02_exception_timeline.png",
    )



# Figure 03
# Exception severity
# severity = max(0, quantile - realized return)
def figure_03(df: pd.DataFrame) -> None:
    exceptions = df[
        df["exception_type"] != "none"
    ]

    historical = exceptions[
        exceptions["historical_violation"]
    ]
    ewma = exceptions[
        exceptions["ewma_violation"]
    ]

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.scatter(
        historical["target_date"],
        historical["historical_exceedance"],
        s=25,
        marker="o",
        label="Historical exception severity",
    )
    ax.scatter(
        ewma["target_date"],
        ewma["ewma_exceedance"],
        s=25,
        marker="x",
        label="EWMA exception severity",
    )

    ax.set_title("Baseline exception severity")
    ax.set_xlabel("Target date")
    ax.set_ylabel(
        "Exceedance below return quantile"
    )
    ax.legend(loc="best", fontsize=8)

    fig.autofmt_xdate()
    save_figure(
        fig,
        "03_exception_severity.png",
    )

# Figure 04
# Exception clusters
def cluster_lengths(
    violation: pd.Series,
) -> list[int]:
    """Lengths of consecutive violation positions."""

    positions = np.flatnonzero(
        violation.astype(bool).to_numpy()
    )

    if len(positions) == 0:
        return []

    groups = np.split(
        positions,
        np.where(np.diff(positions) != 1)[0] + 1,
    )

    return [len(group) for group in groups]


def figure_04(df: pd.DataFrame) -> None:
    historical_lengths = cluster_lengths(
        df["historical_violation"]
    )
    ewma_lengths = cluster_lengths(
        df["ewma_violation"]
    )

    max_length = max(
        historical_lengths + ewma_lengths
    )
    lengths = np.arange(1, max_length + 1)

    historical_counts = [
        historical_lengths.count(length)
        for length in lengths
    ]
    ewma_counts = [
        ewma_lengths.count(length)
        for length in lengths
    ]

    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        lengths - width / 2,
        historical_counts,
        width=width,
        label="Historical",
    )
    ax.bar(
        lengths + width / 2,
        ewma_counts,
        width=width,
        label="EWMA",
    )

    ax.set_xticks(lengths)
    ax.set_title(
        "Exception-cluster length distribution"
    )
    ax.set_xlabel(
        "Consecutive canonical trading-date positions"
    )
    ax.set_ylabel("Number of clusters")
    ax.legend()

    save_figure(
        fig,
        "04_exception_clusters.png",
    )

# Figure 05
# Case-study risk response
# Positive VaR only to avoid sign ambiguity
def figure_05(
    df: pd.DataFrame,
    cases: pd.DataFrame,
) -> None:
    selected_dates = pd.to_datetime(
        cases["target_date"]
    )

    start = (
        selected_dates.min()
        - pd.Timedelta(days=15)
    )
    end = (
        selected_dates.max()
        + pd.Timedelta(days=15)
    )

    view = df[
        df["target_date"].between(start, end)
    ].copy()

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(
        view["target_date"],
        view["historical_var"],
        linewidth=1.2,
        label="Historical VaR",
    )
    ax.plot(
        view["target_date"],
        view["ewma_var"],
        linewidth=1.2,
        label="EWMA VaR",
    )

    indexed = view.set_index("target_date")

    for _, case in cases.iterrows():
        target_date = pd.Timestamp(
            case["target_date"]
        )

        ax.axvline(
            target_date,
            linewidth=0.7,
            linestyle="--",
        )

        if target_date not in indexed.index:
            continue

        row = indexed.loc[target_date]

        y = max(
            float(row["historical_var"]),
            float(row["ewma_var"]),
        )

        ax.annotate(
            case["case_id"],
            xy=(target_date, y),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax.set_title(
        "Baseline risk response around "
        "selected case-study dates"
    )
    ax.set_xlabel("Target date")
    ax.set_ylabel("Positive VaR magnitude")
    ax.legend(loc="best", fontsize=8)

    fig.autofmt_xdate()
    save_figure(
        fig,
        "05_case_study_risk_response.png",
    )


# Artifact gate
def validate_artifacts() -> None:
    for filename in EXPECTED_FIGURES:
        path = OUTPUT_DIR / filename

        assert path.exists()
        assert path.stat().st_size > 0



# Main
def main() -> None:
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "forecast_date",
            "target_date",
        ],
    )

    cases = pd.read_csv(
        CASE_PATH,
        parse_dates=[
            "forecast_date",
            "target_date",
        ],
    )

    validate_data(df, cases)

    figure_01(df)
    figure_02(df)
    figure_03(df)
    figure_04(df)
    figure_05(df, cases)

    validate_artifacts()

    print()
    print(
        "Generated figures:",
        len(EXPECTED_FIGURES),
    )
    print(
        "B-15.4 STANDARDIZED FIGURE SET PASS"
    )


if __name__ == "__main__":
    main()