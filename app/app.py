from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]

FORECAST_PATH = (
    REPO_ROOT / "results" / "latest_forecast_2026-08-28.csv"
)

METRICS_PATH = (
    REPO_ROOT / "results" / "final_metrics.csv"
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


def parse_portfolio_value(raw_value: str) -> float | None:
    normalized = raw_value.strip().replace(",", "").replace("_", "")

    if not normalized:
        st.error("Enter a portfolio value in VND.")
        return None

    try:
        value = float(normalized)
    except ValueError:
        st.error("Portfolio value must be numeric.")
        return None

    if not np.isfinite(value):
        st.error("Portfolio value must be finite.")
        return None

    if value < 0:
        st.error("Portfolio value cannot be negative.")
        return None

    return value


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

latest_tab, evaluation_tab, methodology_tab = st.tabs(
    [
        "Latest Forecast",
        "Frozen Evaluation",
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

    raw_value = st.text_input(
        "Portfolio value (VND)",
        value="100000000",
        help="Example: 100000000 or 100,000,000",
    )

    portfolio_value = parse_portfolio_value(raw_value)

    if portfolio_value is not None:
        st.write(
            f"Portfolio value: **{portfolio_value:,.0f} VND**"
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
                    f"{estimated_loss:,.0f} VND",
                )
                st.caption(
                    f"5% return quantile: "
                    f"{quantile_return * 100:.4f}%"
                )

        st.info(
            "VaR is a risk-threshold estimate under the model assumptions. "
            "It is not the maximum possible loss."
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

    st.markdown(
        f"""
- **Closest violation rate to 5%:** {MODEL_LABELS[closest_violation]}
- **Lowest pinball loss:** {MODEL_LABELS[lowest_pinball]}
- **Lowest average VaR:** {MODEL_LABELS[lowest_average_var]}

These are criterion-specific results; the frozen evaluation does not
define a single overall winning model.
"""
    )

    st.caption(
        "Canonical evaluation period: "
        f"{metrics['test_start'].iloc[0]} to "
        f"{metrics['test_end'].iloc[0]}."
    )


with methodology_tab:
    st.subheader("Methodology")

    st.markdown(
        """
**Portfolio**
- HPG, FPT, MWG
- Equal weights: 1/3 each
- Portfolio simple return: equal-weighted asset simple returns
- Forecast horizon: one trading session
- Confidence level: 95%

**Frozen models**
- Historical Simulation: 250-day rolling window
- EWMA: decay 0.94, expanding history, zero-mean Normal assumption
- Gradient Boosting G04: quantile loss at alpha = 0.05 with seven
  return-history features

**Data semantics**
- Canonical research evaluation cutoff: 2026-07-28
- Operational data cutoff: 2026-08-28
- Operational target date: 2026-09-03

The operational forecast extends the data available to the frozen
models. It does not retune the models or rewrite the canonical
evaluation.
"""
    )
