import numpy as np
import pandas as pd


DATA_PATH = "results/baseline_error_analysis.csv"


def build_clusters(
    frame: pd.DataFrame,
    violation_col: str,
    exceedance_col: str,
) -> pd.DataFrame:
    """
    Build exception clusters using consecutive positions
    in the canonical trading-date sequence.
    """

    mask = frame[violation_col].astype(bool).to_numpy()
    positions = np.flatnonzero(mask)

    clusters = []

    if len(positions) == 0:
        return pd.DataFrame(
            columns=[
                "start_date",
                "end_date",
                "length",
                "max_exceedance",
            ]
        )

    cluster_start = 0

    for i in range(1, len(positions) + 1):
        is_end = (
            i == len(positions)
            or positions[i] != positions[i - 1] + 1
        )

        if not is_end:
            continue

        group_positions = positions[cluster_start:i]

        block = frame.iloc[group_positions]

        clusters.append(
            {
                "start_date": block[
                    "target_date"
                ].iloc[0],
                "end_date": block[
                    "target_date"
                ].iloc[-1],
                "length": len(group_positions),
                "max_exceedance": block[
                    exceedance_col
                ].max(),
            }
        )

        cluster_start = i

    return pd.DataFrame(clusters)


def main() -> None:
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["target_date"],
    )

    assert len(df) == 1387
    assert df["target_date"].is_monotonic_increasing
    assert df["target_date"].is_unique

    historical = build_clusters(
        df,
        "historical_violation",
        "historical_exceedance",
    )

    ewma = build_clusters(
        df,
        "ewma_violation",
        "ewma_exceedance",
    )

    # Historical canonical cluster contract
    assert len(historical) == 63
    assert int((historical["length"] == 1).sum()) == 53
    assert int((historical["length"] >= 2).sum()) == 10
    assert int(historical["length"].max()) == 4

    # EWMA canonical cluster contract
    assert len(ewma) == 65
    assert int((ewma["length"] == 1).sum()) == 57
    assert int((ewma["length"] >= 2).sum()) == 8
    assert int(ewma["length"].max()) == 2

    print("Historical clusters:", len(historical))
    print(
        "Historical singleton clusters:",
        int((historical["length"] == 1).sum()),
    )
    print(
        "Historical multi-observation clusters:",
        int((historical["length"] >= 2).sum()),
    )
    print(
        "Historical longest cluster:",
        int(historical["length"].max()),
    )

    print()

    print("EWMA clusters:", len(ewma))
    print(
        "EWMA singleton clusters:",
        int((ewma["length"] == 1).sum()),
    )
    print(
        "EWMA multi-observation clusters:",
        int((ewma["length"] >= 2).sum()),
    )
    print(
        "EWMA longest cluster:",
        int(ewma["length"].max()),
    )

    print()
    print(
        "Cluster definition: consecutive canonical "
        "trading-date positions"
    )
    print("B-15.2 CLUSTER AUDIT PASS")


if __name__ == "__main__":
    main()