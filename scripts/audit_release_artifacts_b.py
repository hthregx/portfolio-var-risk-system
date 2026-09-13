from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PRED_PATH = ROOT / "results/final_predictions.csv"
OUTPUT_PATH = ROOT / "results/release_artifact_audit_b.csv"

ALPHA = 0.05

METHODS = (
    "historical_simulation",
    "ewma",
    "gradient_boosting",
)

EXPECTED_ROWS = 1194
EXPECTED_DATES = 398
EXPECTED_START = pd.Timestamp("2024-12-18")
EXPECTED_END = pd.Timestamp("2026-07-28")

EXPECTED = {
    "historical_simulation": {
        "forecast_count": 398,
        "violation_count": 27,
        "violation_rate": 0.0678391959798995,
        "pinball_loss": 0.0018964642326659775,
        "average_var": 0.021330189279201585,
    },
    "ewma": {
        "forecast_count": 398,
        "violation_count": 22,
        "violation_rate": 0.055276381909547742,
        "pinball_loss": 0.0019683212526850121,
        "average_var": 0.025084042489536107,
    },
    "gradient_boosting": {
        "forecast_count": 398,
        "violation_count": 24,
        "violation_rate": 0.06030150753768844,
        "pinball_loss": 0.0017581309509574873,
        "average_var": 0.023349057036283791,
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_predictions():
    df = pd.read_csv(
        PRED_PATH,
        parse_dates=["target_date"],
    )

    required = {
        "target_date",
        "method",
        "actual_return",
        "quantile_return",
        "var",
        "violation",
    }

    missing = required - set(df.columns)

    require(
        not missing,
        f"Missing columns: {sorted(missing)}",
    )

    return df


def pinball_loss(actual, quantile):
    error = actual - quantile

    return float(
        np.maximum(
            ALPHA * error,
            (ALPHA - 1.0) * error,
        ).mean()
    )


def audit_contract(df):
    require(
        len(df) == EXPECTED_ROWS,
        "Unexpected prediction row count.",
    )

    require(
        set(df["method"]) == set(METHODS),
        "Unexpected method set.",
    )

    counts = df["method"].value_counts()

    for method in METHODS:
        require(
            int(counts.get(method, 0)) == EXPECTED_DATES,
            f"{method}: expected 398 rows.",
        )

    require(
        df["target_date"].nunique() == EXPECTED_DATES,
        "Unexpected unique target-date count.",
    )

    require(
        df["target_date"].min() == EXPECTED_START,
        "Unexpected evaluation start.",
    )

    require(
        df["target_date"].max() == EXPECTED_END,
        "Unexpected evaluation end.",
    )

    require(
        not df.duplicated(
            ["method", "target_date"]
        ).any(),
        "Duplicate method/date key.",
    )

    require(
        not df.isna().any().any(),
        "Missing value detected.",
    )

    for column in (
        "actual_return",
        "quantile_return",
        "var",
    ):
        require(
            np.isfinite(
                df[column].to_numpy(float)
            ).all(),
            f"Non-finite values in {column}.",
        )

    require(
        (df["var"] >= 0).all(),
        "Negative VaR detected.",
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

    require(
        all(
            dates == reference
            for dates in date_sets.values()
        ),
        "Different target-date universe.",
    )

    actual_counts = (
        df.groupby("target_date")[
            "actual_return"
        ]
        .nunique(dropna=False)
    )

    require(
        (actual_counts == 1).all(),
        "actual_return differs across methods.",
    )


def audit_semantics(df):
    actual = df[
        "actual_return"
    ].to_numpy(float)

    quantile = df[
        "quantile_return"
    ].to_numpy(float)

    var = df[
        "var"
    ].to_numpy(float)

    expected_violation = (
        actual < quantile
    )

    require(
        np.array_equal(
            df["violation"].to_numpy(),
            expected_violation,
        ),
        "Violation rule mismatch.",
    )

    expected_var = np.maximum(
        0.0,
        -quantile,
    )

    require(
        np.allclose(
            var,
            expected_var,
            rtol=0.0,
            atol=1e-12,
        ),
        "VaR rule mismatch.",
    )


def recompute_metrics(df):
    rows = []

    for method in METHODS:
        part = (
            df.loc[
                df["method"] == method
            ]
            .sort_values("target_date")
            .reset_index(drop=True)
        )

        actual = part[
            "actual_return"
        ].to_numpy(float)

        quantile = part[
            "quantile_return"
        ].to_numpy(float)

        var = part[
            "var"
        ].to_numpy(float)

        violation = (
            actual < quantile
        )

        rows.append({
            "method": method,
            "forecast_count":
                int(len(part)),
            "violation_count":
                int(violation.sum()),
            "violation_rate":
                float(violation.mean()),
            "pinball_loss":
                pinball_loss(
                    actual,
                    quantile,
                ),
            "average_var":
                float(var.mean()),
        })

    return pd.DataFrame(rows)


def audit_expected_metrics(metrics):
    for _, row in metrics.iterrows():
        method = row["method"]
        expected = EXPECTED[method]

        require(
            int(row["forecast_count"])
            == expected["forecast_count"],
            f"{method}: forecast_count mismatch.",
        )

        require(
            int(row["violation_count"])
            == expected["violation_count"],
            f"{method}: violation_count mismatch.",
        )

        for column in (
            "violation_rate",
            "pinball_loss",
            "average_var",
        ):
            require(
                np.isclose(
                    float(row[column]),
                    expected[column],
                    rtol=0.0,
                    atol=1e-12,
                ),
                f"{method}: {column} mismatch.",
            )


def write_audit(df, metrics):
    rows = [
        ["prediction_rows", "PASS", len(df)],
        [
            "unique_target_dates",
            "PASS",
            df["target_date"].nunique(),
        ],
        [
            "evaluation_start",
            "PASS",
            df["target_date"].min().date(),
        ],
        [
            "evaluation_end",
            "PASS",
            df["target_date"].max().date(),
        ],
        [
            "common_target_dates",
            "PASS",
            EXPECTED_DATES,
        ],
        [
            "common_actual_returns",
            "PASS",
            EXPECTED_DATES,
        ],
        [
            "no_duplicate_method_date",
            "PASS",
            0,
        ],
        [
            "no_missing_values",
            "PASS",
            0,
        ],
        [
            "finite_quantiles",
            "PASS",
            EXPECTED_ROWS,
        ],
        [
            "finite_var",
            "PASS",
            EXPECTED_ROWS,
        ],
        [
            "var_non_negative",
            "PASS",
            EXPECTED_ROWS,
        ],
        [
            "violation_rule",
            "PASS",
            "actual_return < quantile_return",
        ],
        [
            "var_rule",
            "PASS",
            "max(0, -quantile_return)",
        ],
    ]

    for _, row in metrics.iterrows():
        method = row["method"]

        for column in (
            "forecast_count",
            "violation_count",
            "violation_rate",
            "pinball_loss",
            "average_var",
        ):
            rows.append([
                f"{method}_{column}",
                "PASS",
                row[column],
            ])

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        rows,
        columns=[
            "check",
            "status",
            "observed",
        ],
    ).to_csv(
        OUTPUT_PATH,
        index=False,
    )


def main():
    df = load_predictions()

    audit_contract(df)
    audit_semantics(df)

    metrics = recompute_metrics(df)
    audit_expected_metrics(metrics)

    write_audit(
        df,
        metrics,
    )

    print(
        "=== RELEASE ARTIFACT METRICS ==="
    )
    print(
        metrics.to_string(
            index=False
        )
    )

    print("")
    print("evaluation contract: PASS")
    print("violation semantics: PASS")
    print("VaR semantics: PASS")
    print("missing values: PASS")
    print("duplicate keys: PASS")
    print("common actual returns: PASS")
    print(
        "canonical metrics recomputation: PASS"
    )
    print(
        "RELEASE_ARTIFACT_AUDIT_PASS"
    )
    print(
        "saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()