from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

PAIRED_PATH = (
    REPO_ROOT
    / "results"
    / "final_pairwise_diagnostics.csv"
)

OUTPUT_PATH = (
    REPO_ROOT
    / "results"
    / "final_pairwise_summary.csv"
)

TOLERANCE = 1e-15

COMPARISONS = [
    {
        "comparison": "gb_vs_historical",
        "left_method": "gradient_boosting",
        "right_method": "historical_simulation",
        "delta_column": "gb_minus_historical_pinball",
    },
    {
        "comparison": "gb_vs_ewma",
        "left_method": "gradient_boosting",
        "right_method": "ewma",
        "delta_column": "gb_minus_ewma_pinball",
    },
    {
        "comparison": "ewma_vs_historical",
        "left_method": "ewma",
        "right_method": "historical_simulation",
        "delta_column": "ewma_minus_historical_pinball",
    },
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def dates_to_reach_fraction(
    values: np.ndarray,
    fraction: float,
) -> int:
    """Return count of largest positive values needed for a fraction."""

    positive = np.asarray(
        values,
        dtype="float64",
    )

    positive = positive[
        positive > TOLERANCE
    ]

    if len(positive) == 0:
        return 0

    ordered = np.sort(positive)[::-1]

    cumulative = np.cumsum(ordered)
    target = fraction * float(ordered.sum())

    return int(
        np.searchsorted(
            cumulative,
            target,
            side="left",
        )
        + 1
    )


def top_k_share(
    values: np.ndarray,
    k: int,
) -> float:
    """Return top-k share of total positive magnitude."""

    positive = np.asarray(
        values,
        dtype="float64",
    )

    positive = positive[
        positive > TOLERANCE
    ]

    if len(positive) == 0:
        return 0.0

    ordered = np.sort(positive)[::-1]

    return float(
        ordered[:k].sum()
        / ordered.sum()
    )


def load_paired() -> pd.DataFrame:
    """Load the validated A3.1 paired diagnostic table."""

    frame = pd.read_csv(
        PAIRED_PATH,
        parse_dates=["target_date"],
    )

    require(
        len(frame) == 398,
        f"Expected 398 paired rows, found {len(frame)}.",
    )

    require(
        frame["target_date"].is_unique,
        "Target dates must be unique.",
    )

    required = {
        "target_date",
        "actual_return",
        "historical_pinball_loss",
        "ewma_pinball_loss",
        "gb_pinball_loss",
        "gb_minus_historical_pinball",
        "gb_minus_ewma_pinball",
        "ewma_minus_historical_pinball",
        "lowest_pinball_method",
    }

    missing = required.difference(
        frame.columns
    )

    require(
        not missing,
        f"Missing paired columns: {sorted(missing)}",
    )

    return frame


def summarize_comparison(
    paired: pd.DataFrame,
    *,
    comparison: str,
    left_method: str,
    right_method: str,
    delta_column: str,
) -> dict:
    """Summarize one paired loss-difference series."""

    delta = paired[
        delta_column
    ].to_numpy(dtype="float64")

    require(
        np.isfinite(delta).all(),
        f"{comparison}: non-finite deltas.",
    )

    left_better = delta < -TOLERANCE
    right_better = delta > TOLERANCE
    ties = np.abs(delta) <= TOLERANCE

    require(
        np.all(
            left_better
            | right_better
            | ties
        ),
        f"{comparison}: incomplete sign classification.",
    )

    require(
        int(
            left_better.sum()
            + right_better.sum()
            + ties.sum()
        )
        == len(delta),
        f"{comparison}: sign counts do not reconcile.",
    )

    left_improvement = np.where(
        left_better,
        -delta,
        0.0,
    )

    left_deterioration = np.where(
        right_better,
        delta,
        0.0,
    )

    gross_improvement = float(
        left_improvement.sum()
    )

    gross_deterioration = float(
        left_deterioration.sum()
    )

    net_total_delta = float(
        delta.sum()
    )

    require(
        np.isclose(
            net_total_delta,
            gross_deterioration
            - gross_improvement,
            rtol=0.0,
            atol=1e-15,
        ),
        f"{comparison}: gross/net decomposition failed.",
    )

    min_index = int(
        np.argmin(delta)
    )

    max_index = int(
        np.argmax(delta)
    )

    return {
        "comparison": comparison,
        "left_method": left_method,
        "right_method": right_method,
        "observation_count": int(len(delta)),
        "left_better_count": int(
            left_better.sum()
        ),
        "right_better_count": int(
            right_better.sum()
        ),
        "tie_count": int(
            ties.sum()
        ),
        "left_better_share": float(
            left_better.mean()
        ),
        "mean_delta": float(
            delta.mean()
        ),
        "median_delta": float(
            np.median(delta)
        ),
        "mean_absolute_delta": float(
            np.abs(delta).mean()
        ),
        "gross_left_improvement": gross_improvement,
        "gross_left_deterioration": gross_deterioration,
        "net_total_delta": net_total_delta,
        "top_1_improvement_share": top_k_share(
            left_improvement,
            1,
        ),
        "top_5_improvement_share": top_k_share(
            left_improvement,
            5,
        ),
        "top_10_improvement_share": top_k_share(
            left_improvement,
            10,
        ),
        "top_20_improvement_share": top_k_share(
            left_improvement,
            20,
        ),
        "improvement_dates_to_50pct": (
            dates_to_reach_fraction(
                left_improvement,
                0.50,
            )
        ),
        "improvement_dates_to_80pct": (
            dates_to_reach_fraction(
                left_improvement,
                0.80,
            )
        ),
        "largest_left_improvement": float(
            left_improvement.max()
        ),
        "largest_left_improvement_date": (
            paired.loc[
                min_index,
                "target_date",
            ].strftime("%Y-%m-%d")
        ),
        "largest_left_deterioration": float(
            left_deterioration.max()
        ),
        "largest_left_deterioration_date": (
            paired.loc[
                max_index,
                "target_date",
            ].strftime("%Y-%m-%d")
        ),
    }


def build_summary(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """Build deterministic comparison summaries."""

    records = [
        summarize_comparison(
            paired,
            **specification,
        )
        for specification in COMPARISONS
    ]

    frame = pd.DataFrame(records)

    require(
        len(frame) == 3,
        "Expected exactly three pairwise summaries.",
    )

    require(
        (frame["observation_count"] == 398).all(),
        "Every pairwise comparison must use 398 dates.",
    )

    return frame


def validate_expected_aggregate_signs(
    summary: pd.DataFrame,
) -> None:
    """Validate signs implied by the canonical aggregate Pinball metrics."""

    lookup = summary.set_index(
        "comparison"
    )

    require(
        float(
            lookup.loc[
                "gb_vs_historical",
                "mean_delta",
            ]
        )
        < 0.0,
        "GB should have lower mean loss than Historical.",
    )

    require(
        float(
            lookup.loc[
                "gb_vs_ewma",
                "mean_delta",
            ]
        )
        < 0.0,
        "GB should have lower mean loss than EWMA.",
    )

    require(
        float(
            lookup.loc[
                "ewma_vs_historical",
                "mean_delta",
            ]
        )
        > 0.0,
        "EWMA should have higher mean loss than Historical.",
    )


def print_summary(
    paired: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Print evidence needed for Day-22 interpretation."""

    print("")
    print(
        "=== DAY 22 A3.2 MAGNITUDE AND CONCENTRATION ==="
    )

    print("")
    print("[MULTI-METHOD LOWEST LOSS COUNTS]")

    counts = (
        paired[
            "lowest_pinball_method"
        ]
        .value_counts()
        .sort_index()
    )

    for label, count in counts.items():
        print(
            f"{label:24s}: {int(count)}"
        )

    print("")
    print("[PAIRWISE LOSS-DIFFERENCE SUMMARY]")
    print(
        "Sign convention: delta = left loss - right loss"
    )
    print(
        "Negative mean delta favors the left method."
    )

    display_columns = [
        "comparison",
        "left_better_count",
        "right_better_count",
        "tie_count",
        "mean_delta",
        "median_delta",
        "gross_left_improvement",
        "gross_left_deterioration",
        "top_5_improvement_share",
        "top_20_improvement_share",
        "improvement_dates_to_50pct",
        "improvement_dates_to_80pct",
    ]

    print("")
    print(
        summary[
            display_columns
        ].to_string(index=False)
    )

    print("")
    print("[EXTREME PAIRED CONTRIBUTIONS]")

    for row in summary.itertuples(
        index=False
    ):
        print("")
        print(row.comparison)
        print(
            "  largest left improvement :",
            f"{row.largest_left_improvement:.15f}",
            "on",
            row.largest_left_improvement_date,
        )
        print(
            "  largest left deterioration:",
            f"{row.largest_left_deterioration:.15f}",
            "on",
            row.largest_left_deterioration_date,
        )

    print("")
    print(
        "Interpretation rule: frequency and magnitude "
        "must be discussed separately."
    )
    print(
        "No overall model winner is declared."
    )


def main() -> None:
    paired = load_paired()

    summary = build_summary(
        paired
    )

    validate_expected_aggregate_signs(
        summary
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print_summary(
        paired,
        summary,
    )

    print("")
    print(
        "Wrote:",
        OUTPUT_PATH.relative_to(REPO_ROOT),
    )
    print(
        "PASS - DAY 22 A3.2 CONCENTRATION ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()
