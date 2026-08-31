from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(
    "results/baseline_error_analysis.csv"
)

OUTPUT_PATH = Path(
    "results/baseline_case_studies.csv"
)


def add_next_var_changes(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add next-forecast VaR changes.

    The target return at row t becomes known only after
    the forecast for row t was produced. Therefore the
    immediate model response is measured using row t+1.
    """

    result = frame.copy()

    for method in [
        "historical",
        "ewma",
    ]:
        var_col = f"{method}_var"

        result[
            f"{method}_next_var"
        ] = result[var_col].shift(-1)

        result[
            f"{method}_next_var_change"
        ] = (
            result[
                f"{method}_next_var"
            ]
            - result[var_col]
        )

    return result


def add_preceding_var_changes(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add one-step VaR movement before each target date.
    """

    result = frame.copy()

    for method in [
        "historical",
        "ewma",
    ]:
        var_col = f"{method}_var"

        result[
            f"{method}_previous_var"
        ] = result[var_col].shift(1)

        result[
            f"{method}_preceding_var_change"
        ] = (
            result[var_col]
            - result[
                f"{method}_previous_var"
            ]
        )

    return result


def select_cases(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Deterministic case-study selection.

    Rule:
    1. top 2 shared exceptions by combined exceedance;
    2. top 1 Historical-only exception;
    3. top 1 EWMA-only exception;
    4. include all observations from the longest
       Historical multi-observation exception cluster;
    5. deduplicate by target_date;
    6. cap final set at 10 target dates.
    """

    exception_rows = frame.loc[
        frame["exception_type"] != "none"
    ].copy()

    exception_rows[
        "combined_exceedance"
    ] = exception_rows[
        [
            "historical_exceedance",
            "ewma_exceedance",
        ]
    ].max(axis=1)

    selections = []
    # 1. Top 2 shared exceptions
    shared = (
        exception_rows.loc[
            exception_rows["exception_type"]
            == "both"
        ]
        .sort_values(
            [
                "combined_exceedance",
                "target_date",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(2)
        .copy()
    )

    shared[
        "selection_reason"
    ] = "top_shared_severity"

    selections.append(shared)

    # 2. Historical-only representative

    historical_only = (
        exception_rows.loc[
            exception_rows["exception_type"]
            == "historical_only"
        ]
        .sort_values(
            [
                "historical_exceedance",
                "target_date",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(1)
        .copy()
    )

    historical_only[
        "selection_reason"
    ] = "top_historical_only_severity"

    selections.append(historical_only)
    # 3. EWMA-only representative
    ewma_only = (
        exception_rows.loc[
            exception_rows["exception_type"]
            == "ewma_only"
        ]
        .sort_values(
            [
                "ewma_exceedance",
                "target_date",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(1)
        .copy()
    )

    ewma_only[
        "selection_reason"
    ] = "top_ewma_only_severity"

    selections.append(ewma_only)

    # 4. Longest Historical exception cluster
    positions = np.flatnonzero(
        frame[
            "historical_violation"
        ].astype(bool).to_numpy()
    )

    clusters = []

    start = 0

    for i in range(
        1,
        len(positions) + 1,
    ):
        end_cluster = (
            i == len(positions)
            or positions[i]
            != positions[i - 1] + 1
        )

        if not end_cluster:
            continue

        group = positions[start:i]

        clusters.append(group)

        start = i

    longest = max(
        clusters,
        key=lambda group: (
            len(group),
            frame.iloc[group][
                "historical_exceedance"
            ].max(),
        ),
    )

    longest_cluster = (
        frame.iloc[longest]
        .copy()
    )

    longest_cluster[
        "selection_reason"
    ] = "longest_historical_exception_cluster"

    selections.append(
        longest_cluster
    )
    # Combine + deduplicate

    selected = pd.concat(
        selections,
        ignore_index=True,
    )

    selected = (
        selected
        .sort_values(
            [
                "target_date",
                "selection_reason",
            ]
        )
        .drop_duplicates(
            subset=["target_date"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    if len(selected) > 10:
        selected = selected.iloc[:10].copy()

    assert 5 <= len(selected) <= 10

    assert (
        selected["exception_type"]
        == "historical_only"
    ).any()

    assert (
        selected["exception_type"]
        == "ewma_only"
    ).any()

    assert (
        selected["exception_type"]
        == "both"
    ).any()

    selected.insert(
        0,
        "case_id",
        [
            f"C{i:02d}"
            for i in range(
                1,
                len(selected) + 1,
            )
        ],
    )

    return selected


def main() -> None:
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "forecast_date",
            "target_date",
        ],
    )

    assert len(df) == 1387
    assert df["target_date"].is_unique

    # Temporal alignment used by A's notebook.
    assert np.array_equal(
        df["target_date"]
        .iloc[:-1]
        .to_numpy(),
        df["forecast_date"]
        .iloc[1:]
        .to_numpy(),
    )

    df = add_preceding_var_changes(df)
    df = add_next_var_changes(df)

    cases = select_cases(df)

    columns = [
        "case_id",
        "forecast_date",
        "target_date",
        "target_return",
        "historical_quantile",
        "historical_var",
        "historical_violation",
        "historical_exceedance",
        "historical_preceding_var_change",
        "historical_next_var_change",
        "ewma_quantile",
        "ewma_var",
        "ewma_violation",
        "ewma_exceedance",
        "ewma_preceding_var_change",
        "ewma_next_var_change",
        "exception_type",
        "selection_reason",
    ]

    cases = cases[columns]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        cases.to_string(
            index=False
        )
    )

    print()
    print(
        "Selected case-study dates:",
        len(cases),
    )

    print(
        "Exception types:",
        cases[
            "exception_type"
        ].value_counts().to_dict(),
    )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print()
    print(
        "B-15.3 CASE STUDY SELECTION PASS"
    )


if __name__ == "__main__":
    main()