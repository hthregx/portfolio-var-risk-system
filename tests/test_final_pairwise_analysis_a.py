from __future__ import annotations

import numpy as np

from scripts.analyze_pairwise_concentration_a import (
    COMPARISONS,
    build_summary,
    load_paired,
)


def test_pairwise_summary_uses_all_common_targets() -> None:
    paired = load_paired()
    summary = build_summary(paired)

    assert len(paired) == 398
    assert len(summary) == 3
    assert (
        summary["observation_count"]
        == 398
    ).all()


def test_pairwise_sign_counts_reconcile() -> None:
    summary = build_summary(
        load_paired()
    )

    total = (
        summary["left_better_count"]
        + summary["right_better_count"]
        + summary["tie_count"]
    )

    assert (total == 398).all()


def test_gross_and_net_decomposition_reconciles() -> None:
    summary = build_summary(
        load_paired()
    )

    expected = (
        summary["gross_left_deterioration"]
        - summary["gross_left_improvement"]
    )

    assert np.allclose(
        summary[
            "net_total_delta"
        ].to_numpy(dtype="float64"),
        expected.to_numpy(dtype="float64"),
        rtol=0.0,
        atol=1e-15,
    )


def test_concentration_shares_are_valid_and_monotonic() -> None:
    summary = build_summary(
        load_paired()
    )

    columns = [
        "top_1_improvement_share",
        "top_5_improvement_share",
        "top_10_improvement_share",
        "top_20_improvement_share",
    ]

    values = summary[
        columns
    ].to_numpy(dtype="float64")

    assert (values >= 0.0).all()
    assert (values <= 1.0).all()

    assert (
        values[:, 0]
        <= values[:, 1]
    ).all()

    assert (
        values[:, 1]
        <= values[:, 2]
    ).all()

    assert (
        values[:, 2]
        <= values[:, 3]
    ).all()


def test_dates_to_concentration_threshold_are_ordered() -> None:
    summary = build_summary(
        load_paired()
    )

    assert (
        summary[
            "improvement_dates_to_50pct"
        ]
        <=
        summary[
            "improvement_dates_to_80pct"
        ]
    ).all()

    assert (
        summary[
            "improvement_dates_to_80pct"
        ]
        <=
        summary[
            "left_better_count"
        ]
    ).all()


def test_aggregate_pairwise_signs_match_canonical_results() -> None:
    summary = (
        build_summary(
            load_paired()
        )
        .set_index("comparison")
    )

    assert (
        summary.loc[
            "gb_vs_historical",
            "mean_delta",
        ]
        < 0.0
    )

    assert (
        summary.loc[
            "gb_vs_ewma",
            "mean_delta",
        ]
        < 0.0
    )

    assert (
        summary.loc[
            "ewma_vs_historical",
            "mean_delta",
        ]
        > 0.0
    )


def test_comparison_contract_is_fixed() -> None:
    assert [
        item["comparison"]
        for item in COMPARISONS
    ] == [
        "gb_vs_historical",
        "gb_vs_ewma",
        "ewma_vs_historical",
    ]
