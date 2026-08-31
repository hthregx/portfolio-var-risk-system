from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PRED_PATH = ROOT / "results/final_predictions.csv"
RETURNS_PATH = ROOT / "data/processed/portfolio_returns.csv"

EXCEPTION_OUT = ROOT / "results/final_exception_analysis.csv"
CASES_OUT = ROOT / "results/final_exception_cases.csv"
REGIME_OUT = ROOT / "results/final_regime_analysis.csv"

ALPHA = 0.05
EXPECTED_DATES = 398

EXPECTED_VIOLATIONS = {
    "historical_simulation": 27,
    "ewma": 22,
    "gradient_boosting": 24,
}

METHODS = tuple(EXPECTED_VIOLATIONS)

REGIME_WINDOW = 20
LOW_Q = 1 / 3
HIGH_Q = 2 / 3

# Maximum position gap in common target dates
# for the same near-consecutive exception cluster.
NEAR_CONSECUTIVE_GAP = 2

# LOAD
def load_predictions():
    df = pd.read_csv(PRED_PATH)

    required = {
        "target_date",
        "method",
        "actual_return",
        "quantile_return",
        "var",
        "violation",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df["target_date"] = pd.to_datetime(
        df["target_date"],
        errors="raise",
    )

    return df


# B1 — LOCK / VERIFY CONTRACT
def verify_contract(df):
    if set(df["method"]) != set(METHODS):
        raise ValueError("Unexpected methods.")

    if df.duplicated(
        ["method", "target_date"]
    ).any():
        raise ValueError(
            "Duplicate method/target_date."
        )

    expected = (
        df["actual_return"]
        < df["quantile_return"]
    )

    if not np.array_equal(
        df["violation"].to_numpy(),
        expected.to_numpy(),
    ):
        raise ValueError(
            "Strict < violation rule failed."
        )

    date_sets = {
        method: set(
            df.loc[
                df["method"] == method,
                "target_date",
            ]
        )
        for method in METHODS
    }

    reference = date_sets[METHODS[0]]

    if len(reference) != EXPECTED_DATES:
        raise ValueError(
            f"Expected {EXPECTED_DATES} common dates."
        )

    if any(
        dates != reference
        for dates in date_sets.values()
    ):
        raise ValueError(
            "Methods have different target dates."
        )

    if (
        df.groupby("target_date")[
            "actual_return"
        ]
        .nunique(dropna=False)
        .ne(1)
        .any()
    ):
        raise ValueError(
            "actual_return differs across methods."
        )

    counts = (
        df.groupby("method")["violation"]
        .sum()
        .astype(int)
        .to_dict()
    )

    if counts != EXPECTED_VIOLATIONS:
        raise ValueError(
            f"Violation count mismatch: {counts}"
        )

    return counts



# B2 — EXCEPTION TIMING
def classify_exception(row):
    h = bool(row["historical_violation"])
    e = bool(row["ewma_violation"])
    g = bool(row["gb_violation"])

    count = int(h) + int(e) + int(g)

    if count == 3:
        return "shared_all_3"
    if count == 2:
        return "shared_exactly_2"
    if h:
        return "historical_only"
    if e:
        return "ewma_only"
    if g:
        return "gb_only"

    return "no_violation"


def build_exception_matrix(df):
    out = (
        df.pivot(
            index="target_date",
            columns="method",
            values="violation",
        )
        .rename(columns={
            "historical_simulation":
                "historical_violation",
            "ewma":
                "ewma_violation",
            "gradient_boosting":
                "gb_violation",
        })
        .reset_index()
    )

    actual = (
        df.groupby(
            "target_date",
            as_index=False,
        )["actual_return"]
        .first()
    )

    out = out.merge(
        actual,
        on="target_date",
        validate="one_to_one",
    )

    flags = [
        "historical_violation",
        "ewma_violation",
        "gb_violation",
    ]

    out["violation_method_count"] = (
        out[flags]
        .astype(int)
        .sum(axis=1)
    )

    out["exception_category"] = (
        out.apply(
            classify_exception,
            axis=1,
        )
    )

    out["calendar_month"] = (
        out["target_date"]
        .dt.to_period("M")
        .astype(str)
    )

    out = (
        out.sort_values("target_date")
        .reset_index(drop=True)
    )

    # Deterministic near-consecutive clusters.
    out["target_position"] = np.arange(
        len(out)
    )

    out["cluster_id"] = pd.Series(
        pd.NA,
        index=out.index,
        dtype="Int64",
    )

    previous = None
    cluster = 0

    for i in out.index[
        out["violation_method_count"] > 0
    ]:
        position = int(
            out.loc[i, "target_position"]
        )

        if (
            previous is None
            or position - previous
            > NEAR_CONSECUTIVE_GAP
        ):
            cluster += 1

        out.loc[i, "cluster_id"] = cluster
        previous = position

    return out


def timing_summary(matrix):
    exc = matrix[
        matrix["violation_method_count"] > 0
    ].copy()

    monthly = (
        exc.groupby("calendar_month")
        .agg(
            exception_dates=(
                "target_date",
                "size",
            ),
            method_level_violations=(
                "violation_method_count",
                "sum",
            ),
        )
        .sort_values(
            [
                "method_level_violations",
                "exception_dates",
            ],
            ascending=False,
        )
    )

    clusters = (
        exc.groupby("cluster_id")
        .agg(
            cluster_start=(
                "target_date",
                "min",
            ),
            cluster_end=(
                "target_date",
                "max",
            ),
            exception_dates=(
                "target_date",
                "size",
            ),
            method_level_violations=(
                "violation_method_count",
                "sum",
            ),
            minimum_actual_return=(
                "actual_return",
                "min",
            ),
        )
        .sort_values(
            [
                "method_level_violations",
                "exception_dates",
                "minimum_actual_return",
                "cluster_start",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
        )
    )

    return monthly, clusters


def severe_case(
    matrix,
    category,
    case_type,
):
    part = matrix[
        matrix["exception_category"]
        == category
    ]

    if part.empty:
        return {
            "case_type": case_type,
            "target_date": pd.NaT,
            "actual_return": np.nan,
            "exception_category":
                "not_available",
        }

    row = (
        part.sort_values(
            [
                "actual_return",
                "target_date",
            ]
        )
        .iloc[0]
    )

    return {
        "case_type": case_type,
        "target_date":
            row["target_date"],
        "actual_return":
            row["actual_return"],
        "exception_category":
            category,
    }


def largest_cluster_case(matrix):
    _, clusters = timing_summary(
        matrix
    )

    row = clusters.iloc[0]

    return {
        "case_type":
            "largest_multi_method_cluster",
        "target_date":
            pd.NaT,
        "actual_return":
            row["minimum_actual_return"],
        "exception_category":
            "cluster",
        "cluster_start":
            row["cluster_start"],
        "cluster_end":
            row["cluster_end"],
        "exception_dates":
            row["exception_dates"],
        "method_level_violations":
            row["method_level_violations"],
    }


def build_cases(matrix):
    return pd.DataFrame([
        severe_case(
            matrix,
            "shared_all_3",
            "most_severe_shared",
        ),
        severe_case(
            matrix,
            "historical_only",
            "most_severe_historical_only",
        ),
        severe_case(
            matrix,
            "ewma_only",
            "most_severe_ewma_only",
        ),
        severe_case(
            matrix,
            "gb_only",
            "most_severe_gb_only",
        ),
        largest_cluster_case(matrix),
    ])


# B3 — VOLATILITY REGIME
def pinball(actual, quantile):
    error = actual - quantile

    return float(
        np.maximum(
            ALPHA * error,
            (ALPHA - 1.0) * error,
        ).mean()
    )


def build_regime_analysis(predictions):
    returns = pd.read_csv(
        RETURNS_PATH
    )

    required = {
        "date",
        "portfolio_simple_return",
    }

    missing = required - set(
        returns.columns
    )

    if missing:
        raise ValueError(
            f"Missing return columns: "
            f"{sorted(missing)}"
        )

    returns["date"] = pd.to_datetime(
        returns["date"],
        errors="raise",
    )

    returns = returns.sort_values(
        "date"
    )

    # Target day's return excluded.
    returns["prior_20d_vol"] = (
        returns[
            "portfolio_simple_return"
        ]
        .rolling(REGIME_WINDOW)
        .std(ddof=1)
        .shift(1)
    )

    evaluation_start = (
        predictions["target_date"].min()
    )

    # Thresholds derived only from
    # pre-evaluation history.
    history = returns.loc[
        returns["date"]
        < evaluation_start,
        "prior_20d_vol",
    ].dropna()

    if history.empty:
        raise ValueError(
            "No pre-evaluation regime history."
        )

    low = float(
        history.quantile(LOW_Q)
    )

    high = float(
        history.quantile(HIGH_Q)
    )

    regime = (
        returns[
            ["date", "prior_20d_vol"]
        ]
        .rename(
            columns={
                "date": "target_date"
            }
        )
    )

    pred = predictions.merge(
        regime,
        on="target_date",
        how="left",
        validate="many_to_one",
    )

    if pred[
        "prior_20d_vol"
    ].isna().any():
        raise ValueError(
            "Missing target-date volatility."
        )

    pred["regime"] = np.select(
        [
            pred["prior_20d_vol"] < low,
            pred["prior_20d_vol"] > high,
        ],
        [
            "LOW",
            "HIGH",
        ],
        default="NORMAL",
    )

    records = []

    for (
        regime_name,
        method,
    ), part in pred.groupby(
        ["regime", "method"],
        sort=True,
    ):
        actual = part[
            "actual_return"
        ].to_numpy(float)

        quantile = part[
            "quantile_return"
        ].to_numpy(float)

        var = part[
            "var"
        ].to_numpy(float)

        violations = (
            actual < quantile
        )

        rate = float(
            violations.mean()
        )

        records.append({
            "regime":
                regime_name,
            "method":
                method,
            "observations":
                int(len(part)),
            "violation_count":
                int(violations.sum()),
            "violation_rate":
                rate,
            "pinball_loss":
                pinball(
                    actual,
                    quantile,
                ),
            "average_var":
                float(var.mean()),
            "average_actual_return":
                float(actual.mean()),
            "calibration_distance":
                abs(rate - ALPHA),
            "low_vol_threshold":
                low,
            "high_vol_threshold":
                high,
        })

    return pd.DataFrame(records)


# MAIN
def main():
    predictions = load_predictions()

    # B1
    counts = verify_contract(
        predictions
    )

    print(
        "common_dates:",
        EXPECTED_DATES,
    )

    for method in METHODS:
        print(
            method,
            "violations:",
            counts[method],
        )

    print("strict < rule: PASS")
    print("same actual returns: PASS")
    print("B1 analysis contract: PASS")

    # B2
    matrix = build_exception_matrix(
        predictions
    )

    if len(matrix) != EXPECTED_DATES:
        raise ValueError(
            "Exception matrix row count mismatch."
        )

    cases = build_cases(
        matrix
    )

    monthly, clusters = (
        timing_summary(matrix)
    )

    EXCEPTION_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix.to_csv(
        EXCEPTION_OUT,
        index=False,
    )

    cases.to_csv(
        CASES_OUT,
        index=False,
    )

    print("\nexception categories:")
    print(
        matrix[
            "exception_category"
        ]
        .value_counts()
        .to_string()
    )

    print("\ncalendar concentration:")
    print(
        monthly.head(10).to_string()
    )

    print(
        "\nconsecutive / "
        "near-consecutive clusters:"
    )
    print(
        clusters.head(10).to_string()
    )

    print("\nselected cases:")
    print(
        cases.to_string(
            index=False
        )
    )

    print(
        "\nB2 exception timing: PASS"
    )

    print(
        "saved:",
        EXCEPTION_OUT,
    )
    print(
        "saved:",
        CASES_OUT,
    )

    # B3
    regime = build_regime_analysis(
        predictions
    )

    regime.to_csv(
        REGIME_OUT,
        index=False,
    )

    print("\nregime analysis:")
    print(
        regime.to_string(
            index=False
        )
    )

    print(
        "\nB3 regime analysis: PASS"
    )
    print(
        "saved:",
        REGIME_OUT,
    )


if __name__ == "__main__":
    main()