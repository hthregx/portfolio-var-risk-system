from __future__ import annotations

import pandas as pd

from src.validation.final_results_contract import (
    validate_final_metrics,
    validate_final_predictions,
)


def build_model_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = validate_final_metrics(metrics)

    return (
        frame[
            [
                "method",
                "forecast_count",
                "violation_count",
                "violation_rate",
                "pinball_loss",
                "average_var",
                "minimum_var",
                "maximum_var",
                "total_runtime_seconds",
            ]
        ]
        .sort_values("method")
        .reset_index(drop=True)
    )


def build_exception_table(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = validate_final_predictions(predictions)

    return (
        frame.loc[
            frame["violation"],
            [
                "forecast_date",
                "target_date",
                "method",
                "actual_return",
                "quantile_return",
                "var",
            ],
        ]
        .sort_values(["target_date", "method"])
        .reset_index(drop=True)
    )


def build_latest_var_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = validate_final_predictions(predictions)

    latest = (
        frame.sort_values("target_date")
        .groupby("method", as_index=False)
        .tail(1)
    )

    return (
        latest[
            [
                "method",
                "target_date",
                "quantile_return",
                "var",
            ]
        ]
        .sort_values("method")
        .reset_index(drop=True)
    )


def build_dashboard_data(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> dict:
    return {
        "model_comparison": build_model_comparison(metrics),
        "exceptions": build_exception_table(predictions),
        "latest_var": build_latest_var_summary(predictions),
    }


def build_report_summary(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    comparison = build_model_comparison(metrics)

    summary = comparison[
        [
            "method",
            "forecast_count",
            "violation_rate",
            "pinball_loss",
            "average_var",
        ]
    ].copy()

    summary["violation_rate_display"] = (
        summary["violation_rate"] * 100
    ).map(lambda value: f"{value:.2f}%")

    return summary