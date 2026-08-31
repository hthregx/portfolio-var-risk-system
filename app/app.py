from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st


def format_vnd(value: int | float) -> str:
    """Format a VND value using Vietnamese thousands separators."""
    return f"{int(round(value)):,}".replace(",", ".")


def format_vnd_short(value: int) -> str:
    """Format preset amounts using concise English labels."""
    if value >= 1_000_000_000:
        return f"{value // 1_000_000_000}B"
    if value >= 1_000_000:
        return f"{value // 1_000_000}M"
    if value >= 1_000:
        return f"{value // 1_000}K"
    return str(value)


def load_predictions() -> pd.DataFrame:
    """Load the frozen canonical walk-forward predictions."""
    if not PREDICTIONS_PATH.is_file():
        st.error(
            "Frozen prediction artifact is missing: "
            f"{PREDICTIONS_PATH.relative_to(REPO_ROOT)}"
        )
        st.stop()

    frame = pd.read_csv(PREDICTIONS_PATH)

    required = {
        "forecast_date",
        "target_date",
        "method",
        "actual_return",
        "quantile_return",
        "var",
        "violation",
        "config_id",
    }

    if not required.issubset(frame.columns):
        st.error(
            "Frozen prediction artifact has an invalid schema."
        )
        st.stop()

    if len(frame) != 1194:
        st.error(
            "Frozen prediction artifact must contain "
            "1,194 canonical rows."
        )
        st.stop()

    frame["target_date"] = pd.to_datetime(
        frame["target_date"],
        errors="raise",
    )

    violation = (
        frame["violation"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
            }
        )
    )

    if violation.isna().any():
        st.error(
            "Frozen prediction violation column is invalid."
        )
        st.stop()

    frame["violation"] = violation.astype(bool)

    return frame


def parse_portfolio_value(raw_value: str) -> int | None:
    """Parse non-negative integer VND while preserving validation."""
    raw = raw_value.strip()

    if not raw:
        st.error("Enter a portfolio value in VND.")
        return None

    if raw.startswith("-"):
        st.error("Portfolio value cannot be negative.")
        return None

    if not re.fullmatch(r"[0-9\s.,_]+", raw):
        st.error(
            "Portfolio value must contain only digits "
            "and thousands separators."
        )
        return None

    digits = re.sub(r"[\s.,_]", "", raw)

    if not digits:
        st.error("Enter a portfolio value in VND.")
        return None

    value = int(digits)

    if not np.isfinite(value):
        st.error("Portfolio value must be finite.")
        return None

    return value


def set_portfolio_value(value: int) -> None:
    """Apply a quick-select portfolio value."""
    st.session_state.portfolio_value_text = format_vnd(value)


def normalize_portfolio_value() -> None:
    """Normalize a valid manual entry after Enter/focus change."""
    raw = st.session_state.portfolio_value_text.strip()

    if not raw or raw.startswith("-"):
        return

    if not re.fullmatch(r"[0-9\s.,_]+", raw):
        return

    digits = re.sub(r"[\s.,_]", "", raw)

    if digits:
        st.session_state.portfolio_value_text = format_vnd(int(digits))


REPO_ROOT = Path(__file__).resolve().parents[1]

FORECAST_PATH = (
    REPO_ROOT / "results" / "latest_forecast_2026-08-28.csv"
)

METRICS_PATH = (
    REPO_ROOT / "results" / "final_metrics.csv"
)

PREDICTIONS_PATH = (
    REPO_ROOT / "results" / "final_predictions.csv"
)

MODEL_LABELS = {
    "historical_simulation": "Historical Simulation",
    "ewma": "EWMA",
    "gradient_boosting": "Gradient Boosting G04",
}


def load_forecast() -> pd.DataFrame:
    if not FORECAST_PATH.is_file():
        st.error(
            "Latest forecast artifact is missing: "
            f"{FORECAST_PATH.relative_to(REPO_ROOT)}"
        )
        st.stop()

    frame = pd.read_csv(FORECAST_PATH)

    required = {
        "cutoff_date",
        "forecast_date",
        "target_date",
        "model",
        "quantile_return",
        "var_return",
        "confidence_level",
        "horizon_trading_days",
    }

    if not required.issubset(frame.columns):
        st.error("Latest forecast artifact has an invalid schema.")
        st.stop()

    expected_models = set(MODEL_LABELS)

    if len(frame) != 3 or set(frame["model"]) != expected_models:
        st.error("Latest forecast must contain exactly three frozen models.")
        st.stop()

    return frame


def load_metrics() -> pd.DataFrame:
    if not METRICS_PATH.is_file():
        st.error(
            "Frozen evaluation artifact is missing: "
            f"{METRICS_PATH.relative_to(REPO_ROOT)}"
        )
        st.stop()

    frame = pd.read_csv(METRICS_PATH)

    required = {
        "method",
        "forecast_count",
        "violation_rate",
        "pinball_loss",
        "average_var",
        "test_start",
        "test_end",
        "config_id",
    }

    if not required.issubset(frame.columns):
        st.error("Frozen evaluation artifact has an invalid schema.")
        st.stop()

    return frame



st.set_page_config(
    page_title="Portfolio VaR Risk Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("Portfolio VaR Risk Dashboard")
st.caption(
    "One-day 95% VaR for an equal-weight HPG / FPT / MWG portfolio"
)

forecast = load_forecast()
metrics = load_metrics()
predictions = load_predictions()

(
    latest_tab,
    backtesting_tab,
    evaluation_tab,
    methodology_tab,
) = st.tabs(
    [
        "Risk Snapshot",
        "Historical Backtesting",
        "Model Comparison",
        "Methodology",
    ]
)


with latest_tab:
    cutoff = str(forecast["cutoff_date"].iloc[0])
    target = str(forecast["target_date"].iloc[0])

    st.subheader("Operational VaR Forecast")

    st.write(
        f"Market information through **{cutoff}** → "
        f"next trading-session target **{target}**."
    )

    overview_columns = st.columns(4)

    overview_columns[0].metric(
        "Portfolio",
        "HPG / FPT / MWG",
    )
    overview_columns[1].metric(
        "Confidence level",
        "95%",
    )
    overview_columns[2].metric(
        "Data cutoff",
        cutoff,
    )
    overview_columns[3].metric(
        "Target session",
        target,
    )

    st.markdown("### Portfolio value")

    if "portfolio_value_text" not in st.session_state:
        st.session_state.portfolio_value_text = format_vnd(
            100_000_000
        )

    preset_values = [
        10_000_000,
        50_000_000,
        100_000_000,
        500_000_000,
        1_000_000_000,
    ]

    preset_columns = st.columns(len(preset_values))

    for preset_column, preset_value in zip(
        preset_columns,
        preset_values,
    ):
        preset_column.button(
            format_vnd_short(preset_value),
            key=f"portfolio_preset_{preset_value}",
            on_click=set_portfolio_value,
            args=(preset_value,),
            width="stretch",
        )

    raw_value = st.text_input(
        "Portfolio value (VND)",
        key="portfolio_value_text",
        on_change=normalize_portfolio_value,
        placeholder="V? d?: 100.000.000",
        help=(
            "Enter a portfolio value manually or select "
            "one of the suggested amounts above."
        ),
    )

    portfolio_value = parse_portfolio_value(raw_value)

    if portfolio_value is not None:
        st.caption(
            f"Selected portfolio value: "
            f"**{format_vnd(portfolio_value)} VND**"
        )

        columns = st.columns(3)

        for column, (_, row) in zip(
            columns,
            forecast.iterrows(),
        ):
            model = str(row["model"])
            var_return = float(row["var_return"])
            quantile_return = float(row["quantile_return"])
            estimated_loss = portfolio_value * var_return

            with column:
                st.markdown(f"### {MODEL_LABELS[model]}")
                st.metric(
                    "95% one-day VaR",
                    f"{var_return * 100:.4f}%",
                )
                st.metric(
                    "Estimated VaR amount",
                    f"{format_vnd(estimated_loss)} VND",
                )
                st.caption(
                    f"5% return quantile: "
                    f"{quantile_return * 100:.4f}%"
                )

        chart_data = forecast[
            ["model", "var_return"]
        ].copy()

        chart_data["Model"] = chart_data["model"].map(
            MODEL_LABELS
        )
        chart_data["VaR (%)"] = (
            chart_data["var_return"] * 100.0
        )
        chart_data["Estimated VaR (million VND)"] = (
            portfolio_value
            * chart_data["var_return"]
            / 1_000_000.0
        )

        st.divider()
        st.subheader("Risk Comparison")

        chart_left, chart_right = st.columns(2)

        with chart_left:
            st.markdown("#### 95% one-day VaR")
            st.bar_chart(
                chart_data,
                x="Model",
                y="VaR (%)",
                height=240,
            )

        with chart_right:
            st.markdown("#### Estimated VaR amount")
            st.bar_chart(
                chart_data,
                x="Model",
                y="Estimated VaR (million VND)",
                height=240,
            )

        st.info(
            "VaR is a risk-threshold estimate under the model assumptions. "
            "It is not the maximum possible loss."
        )


with backtesting_tab:
    st.subheader("Historical VaR Backtesting")

    st.write(
        "Explore the frozen walk-forward evaluation and inspect "
        "when realized portfolio returns crossed the predicted "
        "5% lower-tail threshold."
    )

    selected_method = st.selectbox(
        "Model",
        options=list(MODEL_LABELS),
        format_func=lambda method: MODEL_LABELS[method],
        key="backtesting_model",
    )

    model_predictions = (
        predictions.loc[
            predictions["method"] == selected_method
        ]
        .sort_values("target_date")
        .copy()
    )

    if len(model_predictions) != 398:
        st.error(
            "Selected model does not contain "
            "398 frozen evaluation observations."
        )
        st.stop()

    violation_count = int(
        model_predictions["violation"].sum()
    )
    violation_rate = float(
        model_predictions["violation"].mean()
    )
    average_var = float(
        model_predictions["var"].mean()
    )

    backtest_metrics = st.columns(4)

    backtest_metrics[0].metric(
        "Evaluation observations",
        f"{len(model_predictions):,}",
    )

    backtest_metrics[1].metric(
        "VaR violations",
        f"{violation_count}",
    )

    backtest_metrics[2].metric(
        "Violation rate",
        f"{violation_rate * 100:.2f}%",
        help="Nominal violation rate at 95% confidence is 5%.",
    )

    backtest_metrics[3].metric(
        "Average VaR",
        f"{average_var * 100:.2f}%",
    )

    chart_frame = model_predictions[
        [
            "target_date",
            "actual_return",
            "quantile_return",
            "violation",
        ]
    ].copy()

    chart_frame["target_date"] = (
        chart_frame["target_date"]
        .dt.strftime("%Y-%m-%d")
    )

    chart_frame["Actual Return (%)"] = (
        chart_frame["actual_return"] * 100.0
    )

    chart_frame["5% Quantile Threshold (%)"] = (
        chart_frame["quantile_return"] * 100.0
    )

    chart_values = chart_frame[
        [
            "target_date",
            "Actual Return (%)",
            "5% Quantile Threshold (%)",
            "violation",
        ]
    ].to_dict(orient="records")

    st.markdown(
        f"#### {MODEL_LABELS[selected_method]} - "
        "realized returns vs. VaR threshold"
    )

    backtest_spec = {
        "height": 360,
        "data": {
            "values": chart_values,
        },
        "layer": [
            {
                "mark": {
                    "type": "line",
                    "strokeWidth": 1.5,
                    "color": "#2563EB",
                },
                "encoding": {
                    "x": {
                        "field": "target_date",
                        "type": "temporal",
                        "title": "Target date",
                    },
                    "y": {
                        "field": "Actual Return (%)",
                        "type": "quantitative",
                        "title": "Portfolio return (%)",
                    },
                    "tooltip": [
                        {
                            "field": "target_date",
                            "type": "temporal",
                            "title": "Date",
                        },
                        {
                            "field": "Actual Return (%)",
                            "type": "quantitative",
                            "format": ".3f",
                        },
                    ],
                },
            },
            {
                "mark": {
                    "type": "line",
                    "strokeWidth": 2,
                    "strokeDash": [6, 4],
                    "color": "#DC2626",
                },
                "encoding": {
                    "x": {
                        "field": "target_date",
                        "type": "temporal",
                    },
                    "y": {
                        "field": "5% Quantile Threshold (%)",
                        "type": "quantitative",
                    },
                    "tooltip": [
                        {
                            "field": "target_date",
                            "type": "temporal",
                            "title": "Date",
                        },
                        {
                            "field": "5% Quantile Threshold (%)",
                            "type": "quantitative",
                            "format": ".3f",
                            "title": "5% Quantile (%)",
                        },
                    ],
                },
            },
            {
                "transform": [
                    {
                        "filter": "datum.violation == true",
                    }
                ],
                "mark": {
                    "type": "point",
                    "filled": True,
                    "size": 70,
                    "color": "#DC2626",
                    "stroke": "#FFFFFF",
                    "strokeWidth": 1,
                },
                "encoding": {
                    "x": {
                        "field": "target_date",
                        "type": "temporal",
                    },
                    "y": {
                        "field": "Actual Return (%)",
                        "type": "quantitative",
                    },
                    "tooltip": [
                        {
                            "field": "target_date",
                            "type": "temporal",
                            "title": "Violation date",
                        },
                        {
                            "field": "Actual Return (%)",
                            "type": "quantitative",
                            "format": ".3f",
                        },
                        {
                            "field": "5% Quantile Threshold (%)",
                            "type": "quantitative",
                            "format": ".3f",
                            "title": "5% Quantile (%)",
                        },
                    ],
                },
            },
        ],
    }

    st.vega_lite_chart(
        backtest_spec,
        width="stretch",
    )

    st.caption(
        "Blue line: realized portfolio return. "
        "Dashed red line: predicted 5% return quantile. "
        "Red markers: VaR violations where the realized return "
        "fell below the predicted quantile threshold."
    )

    st.info(
        "At 95% confidence, a well-calibrated model is expected "
        "to produce a violation rate near 5% over a sufficiently "
        "representative evaluation period."
    )


with evaluation_tab:
    st.subheader("Frozen Canonical Evaluation")

    comparison = metrics[
        [
            "method",
            "forecast_count",
            "violation_rate",
            "pinball_loss",
            "average_var",
            "config_id",
        ]
    ].copy()

    comparison["model"] = comparison["method"].map(MODEL_LABELS)
    comparison["violation_rate"] *= 100.0
    comparison["average_var"] *= 100.0

    comparison = comparison[
        [
            "model",
            "forecast_count",
            "violation_rate",
            "pinball_loss",
            "average_var",
            "config_id",
        ]
    ].rename(
        columns={
            "model": "Model",
            "forecast_count": "Forecasts",
            "violation_rate": "Violation Rate (%)",
            "pinball_loss": "Pinball Loss",
            "average_var": "Average VaR (%)",
            "config_id": "Frozen Config",
        }
    )

    st.dataframe(
        comparison,
        hide_index=True,
        width="stretch",
    )

    st.markdown("### Visual Comparison")

    visual_comparison = metrics[
        [
            "method",
            "violation_rate",
            "pinball_loss",
            "average_var",
        ]
    ].copy()

    short_labels = {
        "historical_simulation": "Historical",
        "ewma": "EWMA",
        "gradient_boosting": "GB G04",
    }

    visual_comparison["Model"] = (
        visual_comparison["method"].map(short_labels)
    )

    visual_comparison["Violation Rate (%)"] = (
        visual_comparison["violation_rate"] * 100.0
    )

    visual_comparison["Pinball Loss"] = (
        visual_comparison["pinball_loss"]
    )

    visual_comparison["Average VaR (%)"] = (
        visual_comparison["average_var"] * 100.0
    )

    comparison_values = visual_comparison[
        [
            "Model",
            "Violation Rate (%)",
            "Pinball Loss",
            "Average VaR (%)",
        ]
    ].to_dict(orient="records")

    model_order = [
        "Historical",
        "EWMA",
        "GB G04",
    ]

    comparison_columns = st.columns(3)

    violation_spec = {
        "height": 220,
        "data": {
            "values": comparison_values,
        },
        "layer": [
            {
                "mark": {
                    "type": "bar",
                    "cornerRadiusEnd": 4,
                    "color": "#2563EB",
                },
                "encoding": {
                    "y": {
                        "field": "Model",
                        "type": "nominal",
                        "sort": model_order,
                        "title": None,
                    },
                    "x": {
                        "field": "Violation Rate (%)",
                        "type": "quantitative",
                        "title": "Violation rate (%)",
                        "scale": {
                            "domain": [0, 8],
                        },
                    },
                    "tooltip": [
                        {
                            "field": "Model",
                            "type": "nominal",
                        },
                        {
                            "field": "Violation Rate (%)",
                            "type": "quantitative",
                            "format": ".2f",
                        },
                    ],
                },
            },
            {
                "mark": {
                    "type": "rule",
                    "color": "#DC2626",
                    "strokeDash": [6, 4],
                    "strokeWidth": 2,
                },
                "encoding": {
                    "x": {
                        "datum": 5.0,
                    },
                },
            },
        ],
    }

    pinball_spec = {
        "height": 220,
        "data": {
            "values": comparison_values,
        },
        "mark": {
            "type": "bar",
            "cornerRadiusEnd": 4,
            "color": "#2563EB",
        },
        "encoding": {
            "y": {
                "field": "Model",
                "type": "nominal",
                "sort": model_order,
                "title": None,
            },
            "x": {
                "field": "Pinball Loss",
                "type": "quantitative",
                "title": "Pinball loss",
                "scale": {
                    "zero": True,
                },
            },
            "tooltip": [
                {
                    "field": "Model",
                    "type": "nominal",
                },
                {
                    "field": "Pinball Loss",
                    "type": "quantitative",
                    "format": ".6f",
                },
            ],
        },
    }

    average_var_spec = {
        "height": 220,
        "data": {
            "values": comparison_values,
        },
        "mark": {
            "type": "bar",
            "cornerRadiusEnd": 4,
            "color": "#2563EB",
        },
        "encoding": {
            "y": {
                "field": "Model",
                "type": "nominal",
                "sort": model_order,
                "title": None,
            },
            "x": {
                "field": "Average VaR (%)",
                "type": "quantitative",
                "title": "Average VaR (%)",
                "scale": {
                    "zero": True,
                },
            },
            "tooltip": [
                {
                    "field": "Model",
                    "type": "nominal",
                },
                {
                    "field": "Average VaR (%)",
                    "type": "quantitative",
                    "format": ".3f",
                },
            ],
        },
    }

    with comparison_columns[0]:
        st.markdown("#### Violation Rate")
        st.vega_lite_chart(
            violation_spec,
            width="stretch",
        )
        st.caption(
            "Dashed red line: nominal 5% violation rate."
        )

    with comparison_columns[1]:
        st.markdown("#### Pinball Loss")
        st.vega_lite_chart(
            pinball_spec,
            width="stretch",
        )
        st.caption(
            "Lower values indicate better quantile forecast accuracy."
        )

    with comparison_columns[2]:
        st.markdown("#### Average VaR")
        st.vega_lite_chart(
            average_var_spec,
            width="stretch",
        )
        st.caption(
            "Risk magnitude only; lower does not automatically mean better."
        )

    closest_violation = metrics.loc[
        (metrics["violation_rate"] - 0.05).abs().idxmin(),
        "method",
    ]

    lowest_pinball = metrics.loc[
        metrics["pinball_loss"].idxmin(),
        "method",
    ]

    lowest_average_var = metrics.loc[
        metrics["average_var"].idxmin(),
        "method",
    ]

    closest_violation_value = float(
        metrics.loc[
            metrics["method"] == closest_violation,
            "violation_rate",
        ].iloc[0]
    )

    lowest_pinball_value = float(
        metrics.loc[
            metrics["method"] == lowest_pinball,
            "pinball_loss",
        ].iloc[0]
    )

    lowest_average_var_value = float(
        metrics.loc[
            metrics["method"] == lowest_average_var,
            "average_var",
        ].iloc[0]
    )

    st.markdown("### Interpretation")

    interpretation_columns = st.columns(3)

    with interpretation_columns[0]:
        with st.container(border=True):
            st.markdown("**Calibration**")
            st.markdown(
                f"### {MODEL_LABELS[closest_violation]}"
            )
            st.caption(
                f"{closest_violation_value * 100:.2f}% violation rate "
                "- closest to the nominal 5% target."
            )

    with interpretation_columns[1]:
        with st.container(border=True):
            st.markdown("**Quantile Accuracy**")
            st.markdown(
                f"### {MODEL_LABELS[lowest_pinball]}"
            )
            st.caption(
                f"Lowest pinball loss: "
                f"{lowest_pinball_value:.6f}."
            )

    with interpretation_columns[2]:
        with st.container(border=True):
            st.markdown("**Risk Magnitude**")
            st.markdown(
                f"### {MODEL_LABELS[lowest_average_var]}"
            )
            st.caption(
                f"Lowest average VaR: "
                f"{lowest_average_var_value * 100:.2f}%. "
                "This is descriptive, not an accuracy ranking."
            )

    st.info(
        "The three criteria answer different questions. "
        "The frozen evaluation therefore does not define "
        "a single overall winning model."
    )

    st.caption(
        "Canonical evaluation period: "
        f"{metrics['test_start'].iloc[0]} to "
        f"{metrics['test_end'].iloc[0]}."
    )


with methodology_tab:
    st.subheader("Methodology & Research Contract")

    st.write(
        "The dashboard separates frozen research evidence from "
        "operational forecasting. Model specifications and the "
        "canonical evaluation remain fixed."
    )

    contract_columns = st.columns(3)

    with contract_columns[0]:
        with st.container(border=True):
            st.markdown("### Portfolio Contract")
            st.markdown(
                """
- Assets: **HPG / FPT / MWG**
- Weights: **1/3 each**
- Return: equal-weighted simple return
- Horizon: **one trading session**
- Confidence level: **95%**
"""
            )

    with contract_columns[1]:
        with st.container(border=True):
            st.markdown("### Frozen Models")
            st.markdown(
                """
- **Historical:** 250-day rolling window
- **EWMA:** decay 0.94, expanding history
- **GB G04:** 5% quantile regression
- GB inputs: **7 return-history features**
- No model retuning in the dashboard
"""
            )

    with contract_columns[2]:
        with st.container(border=True):
            st.markdown("### Evaluation Design")
            st.markdown(
                """
- Chronological walk-forward evaluation
- **398** targets per model
- Evaluation: **2024-12-18 to 2026-07-28**
- No random shuffling
- Violation: actual return below 5% quantile
"""
            )

    st.markdown("### Research vs. Operational Boundary")

    boundary_columns = st.columns(3)

    boundary_columns[0].metric(
        "Canonical research cutoff",
        "2026-07-28",
    )

    boundary_columns[1].metric(
        "Operational data cutoff",
        "2026-08-28",
    )

    boundary_columns[2].metric(
        "Operational target session",
        "2026-09-03",
    )

    control_left, control_right = st.columns(2)

    with control_left:
        with st.container(border=True):
            st.markdown("### Leakage Controls")
            st.markdown(
                """
- Time order is preserved throughout evaluation.
- Forecast features use information available before the target.
- Rolling and lagged features do not use future observations.
- Frozen configurations are reused for operational forecasting.
- Operational refreshes do not rewrite canonical evaluation results.
"""
            )

    with control_right:
        with st.container(border=True):
            st.markdown("### Scope & Limitations")
            st.markdown(
                """
- Equal-weight portfolio of three Vietnamese equities.
- One-day 95% VaR only.
- VaR is not the maximum possible loss.
- Expected Shortfall is outside the frozen model scope.
- Liquidity, transaction costs, and portfolio optimization are not modeled.
"""
            )

    st.info(
        "Operational forecasts extend the available market data while "
        "preserving the frozen research specification. The dashboard "
        "is a presentation and monitoring layer, not a model-training layer."
    )
