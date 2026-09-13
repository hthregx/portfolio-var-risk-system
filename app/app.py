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

MARKET_SNAPSHOT_PATH = (
    REPO_ROOT
    / "data"
    / "snapshots"
    / "market_data_2026-08-28.csv"
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


def load_market_snapshot() -> pd.DataFrame:
    """Load the operational market snapshot used by Stock Explorer."""
    if not MARKET_SNAPSHOT_PATH.is_file():
        st.error(
            "Operational market snapshot is missing: "
            f"{MARKET_SNAPSHOT_PATH.relative_to(REPO_ROOT)}"
        )
        st.stop()

    frame = pd.read_csv(MARKET_SNAPSHOT_PATH)

    required = {
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    if not required.issubset(frame.columns):
        st.error(
            "Operational market snapshot has an invalid schema."
        )
        st.stop()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="raise",
    )

    frame["ticker"] = (
        frame["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    supported_tickers = {"HPG", "FPT", "MWG"}

    if not supported_tickers.issubset(
        set(frame["ticker"])
    ):
        st.error(
            "Operational market snapshot must contain "
            "HPG, FPT and MWG."
        )
        st.stop()

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        )

    if frame[numeric_columns].isna().any().any():
        st.error(
            "Operational market snapshot contains missing "
            "numeric values."
        )
        st.stop()

    return (
        frame
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )


def filter_stock_period(
    stock_data: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    """Filter one stock series to the selected display period."""
    if stock_data.empty:
        return stock_data.copy()

    latest_date = stock_data["date"].max()

    if period == "1M":
        start_date = latest_date - pd.DateOffset(months=1)
    elif period == "3M":
        start_date = latest_date - pd.DateOffset(months=3)
    elif period == "6M":
        start_date = latest_date - pd.DateOffset(months=6)
    elif period == "YTD":
        start_date = pd.Timestamp(
            year=latest_date.year,
            month=1,
            day=1,
        )
    elif period == "1Y":
        start_date = latest_date - pd.DateOffset(years=1)
    elif period == "All":
        return stock_data.copy()
    else:
        raise ValueError(
            f"Unsupported stock period: {period}"
        )

    return (
        stock_data.loc[
            stock_data["date"] >= start_date
        ]
        .copy()
        .reset_index(drop=True)
    )



st.set_page_config(
    page_title="Portfolio VaR Risk Dashboard",
    page_icon="📊",
    layout="wide",
)


st.markdown(
    """
<style>
/* D5.1a dashboard visual theme */

:root {
    --page: #f6f8fc;
    --card: #ffffff;
    --border: #e2e8f0;
    --text: #0f172a;
    --muted: #64748b;
    --navy: #10243d;
    --navy-deep: #0b1b2f;
    --blue: #2563eb;
}

[data-testid="stAppViewContainer"] {
    background: var(--page);
}

[data-testid="stHeader"] {
    background: rgba(246, 248, 252, 0.96);
}

.stMainBlockContainer {
    max-width: 1450px;
    padding-top: 1.4rem;
    padding-bottom: 4rem;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            var(--navy) 0%,
            var(--navy-deep) 100%
        );
    border-right: none;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] li {
    color: #e8eef6;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.14);
}

[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 0.9rem;
    padding: 0.85rem 1rem;
    box-shadow:
        0 3px 14px rgba(15, 23, 42, 0.035);
}

[data-testid="stMetricLabel"] p {
    color: var(--muted);
    font-weight: 550;
}

[data-testid="stMetricValue"] {
    color: var(--text);
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card);
    border-radius: 0.9rem;
}

div[data-testid="stAlert"] {
    border-radius: 0.8rem;
}

div.stButton > button {
    min-height: 2.5rem;
    border-radius: 0.7rem;
    border-color: #d7deea;
    font-weight: 600;
}

div.stButton > button:hover {
    border-color: var(--blue);
    color: var(--blue);
}

h1,
h2,
h3 {
    color: var(--text);
    letter-spacing: -0.025em;
}

hr {
    border-color: var(--border);
    margin-top: 1.6rem;
    margin-bottom: 1.6rem;
}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<style>
/* D5.1b polished navigation */
/* D5.3a compact dashboard layout */
/* D5.3c sidebar removed */
/* D5.4b colorful risk overview */
/* D5.4b.1 st.html rendering fix */

/* D5.5b colorful stock cards */

/* D5.6b compact analytics panels */

/* D5.7b colorful evaluation panels */

/* D5.8a final visual polish */

/* D5.8a.2 hero clipping fix */

.stMainBlockContainer {
    max-width: 1540px;
    padding-top: 2.2rem;
    padding-bottom: 2.2rem;
}

.app-hero {
    margin-top: 0.25rem;
}

.app-hero-title {
    padding-top: 0.08rem;
    line-height: 1.2;
}

/* ---------------------------------------------------------
   Hero
--------------------------------------------------------- */

.app-hero {
    position: relative;
    overflow: hidden;

    padding: 1.25rem 1.35rem;

    background:
        radial-gradient(
            circle at 92% 15%,
            rgba(124, 58, 237, 0.16),
            transparent 28%
        ),
        radial-gradient(
            circle at 72% 100%,
            rgba(14, 165, 233, 0.12),
            transparent 32%
        ),
        linear-gradient(
            120deg,
            #ffffff 0%,
            #f8fbff 46%,
            #f5f3ff 100%
        );

    border:
        1px solid rgba(99, 102, 241, 0.18);

    box-shadow:
        0 10px 32px rgba(37, 99, 235, 0.08);
}

.app-hero::before {
    content: "";
    position: absolute;

    width: 260px;
    height: 4px;

    left: 1.35rem;
    bottom: 0;

    border-radius: 999px;

    background:
        linear-gradient(
            90deg,
            #2563eb,
            #7c3aed,
            #06b6d4,
            #10b981
        );
}

.app-hero-title {
    background:
        linear-gradient(
            90deg,
            #0f172a 0%,
            #1d4ed8 72%,
            #6d28d9 100%
        );

    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;

    font-size: 2.08rem;
}

.operational-badge {
    box-shadow:
        0 4px 14px rgba(16, 185, 129, 0.14);
}

/* ---------------------------------------------------------
   Horizontal navigation
--------------------------------------------------------- */

.top-nav {
    position: relative;

    padding: 0.42rem;

    background:
        linear-gradient(
            90deg,
            rgba(255,255,255,0.98),
            rgba(248,250,252,0.98)
        );

    border:
        1px solid rgba(148, 163, 184, 0.22);

    box-shadow:
        0 4px 16px rgba(15, 23, 42, 0.045);
}

.top-nav a {
    min-height: 2.05rem;
    padding: 0.38rem 0.86rem;
}

.top-nav .nav-primary {
    background:
        linear-gradient(
            135deg,
            #2563eb,
            #4f46e5
        );

    box-shadow:
        0 5px 14px rgba(37, 99, 235, 0.18);
}

/* ---------------------------------------------------------
   Section headings
--------------------------------------------------------- */

.section-heading {
    margin-top: 0.25rem;
    margin-bottom: 0.68rem;
}

.section-index {
    background:
        linear-gradient(
            135deg,
            #dbeafe,
            #ede9fe
        );

    color: #3730a3;

    border:
        1px solid rgba(99, 102, 241, 0.16);
}

.section-title {
    font-size: 1.34rem;
}

/* ---------------------------------------------------------
   Controls
--------------------------------------------------------- */

div[data-baseweb="select"] > div {
    border-radius: 0.72rem;
}

div[data-testid="stSegmentedControl"] {
    border-radius: 0.8rem;
}

div.stButton > button {
    transition:
        transform 0.12s ease,
        box-shadow 0.12s ease,
        border-color 0.12s ease;
}

div.stButton > button:hover {
    transform: translateY(-1px);

    box-shadow:
        0 5px 13px rgba(37, 99, 235, 0.10);
}

/* ---------------------------------------------------------
   Charts
--------------------------------------------------------- */

[data-testid="stVegaLiteChart"] {
    overflow: hidden;

    box-shadow:
        0 5px 18px rgba(15, 23, 42, 0.045);
}

/* ---------------------------------------------------------
   Expanders / Methodology
--------------------------------------------------------- */

div[data-testid="stExpander"] details {
    overflow: hidden;

    border:
        1px solid rgba(99, 102, 241, 0.15);

    background:
        linear-gradient(
            135deg,
            #ffffff 0%,
            #fafaff 100%
        );

    box-shadow:
        0 4px 16px rgba(15, 23, 42, 0.035);
}

div[data-testid="stExpander"] summary {
    padding-top: 0.72rem;
    padding-bottom: 0.72rem;
}

/* ---------------------------------------------------------
   Alerts
--------------------------------------------------------- */

div[data-testid="stAlert"] {
    border:
        1px solid rgba(59, 130, 246, 0.13);

    box-shadow:
        0 3px 12px rgba(15, 23, 42, 0.025);
}

/* ---------------------------------------------------------
   Dividers / whitespace
--------------------------------------------------------- */

hr {
    margin-top: 0.72rem;
    margin-bottom: 0.72rem;
}

/* ---------------------------------------------------------
   Footer
--------------------------------------------------------- */

.dashboard-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;

    margin-top: 1.3rem;
    padding: 0.9rem 1rem;

    border-radius: 0.9rem;

    background:
        linear-gradient(
            90deg,
            #eff6ff,
            #f5f3ff,
            #ecfdf5
        );

    border:
        1px solid #e0e7ff;
}

.footer-brand {
    color: #0f172a;
    font-size: 0.78rem;
    font-weight: 800;
}

.footer-meta {
    color: #64748b;
    font-size: 0.68rem;
    text-align: right;
}

.footer-dots {
    display: inline-flex;
    gap: 0.25rem;
    margin-right: 0.5rem;
}

.footer-dot {
    width: 0.42rem;
    height: 0.42rem;
    border-radius: 50%;
    display: inline-block;
}

.footer-blue {
    background: #2563eb;
}

.footer-purple {
    background: #7c3aed;
}

.footer-green {
    background: #10b981;
}

@media (max-width: 800px) {

    .app-hero {
        flex-direction: column;
    }

    .dashboard-footer {
        flex-direction: column;
        align-items: flex-start;
    }

    .footer-meta {
        text-align: left;
    }
}



.backtest-kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
    margin: 0.55rem 0 0.9rem 0;
}

.backtest-kpi {
    padding: 0.8rem 0.9rem;
    border-radius: 0.85rem;
    border: 1px solid;
    background: #ffffff;
}

.bt-blue {
    background: linear-gradient(145deg, #eff6ff, #ffffff);
    border-color: #bfdbfe;
}

.bt-red {
    background: linear-gradient(145deg, #fef2f2, #ffffff);
    border-color: #fecaca;
}

.bt-orange {
    background: linear-gradient(145deg, #fff7ed, #ffffff);
    border-color: #fed7aa;
}

.bt-purple {
    background: linear-gradient(145deg, #f5f3ff, #ffffff);
    border-color: #ddd6fe;
}

.bt-label {
    color: #64748b;
    font-size: 0.66rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.bt-value {
    margin-top: 0.18rem;
    color: #0f172a;
    font-size: 1.45rem;
    font-weight: 820;
    letter-spacing: -0.03em;
}

.evaluation-chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;

    margin: 0.45rem 0 0.5rem 0;
    padding: 0.65rem 0.8rem;

    border: 1px solid #dbe4f0;
    border-radius: 0.8rem;

    background:
        linear-gradient(
            90deg,
            #f8fbff,
            #ffffff
        );
}

.evaluation-chart-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;

    color: #0f172a;
    font-size: 0.88rem;
    font-weight: 780;
}

.model-color-dot {
    display: inline-block;
    width: 0.62rem;
    height: 0.62rem;
    border-radius: 50%;
}

.evaluation-chip {
    padding: 0.24rem 0.5rem;
    border-radius: 999px;
    background: #f1f5f9;
    color: #475569;
    font-size: 0.62rem;
    font-weight: 760;
}

.comparison-intro {
    margin: 0.45rem 0 0.65rem 0;
    padding: 0.7rem 0.8rem;

    border-radius: 0.8rem;
    border: 1px solid #e0e7ff;

    background:
        linear-gradient(
            90deg,
            #eff6ff,
            #f5f3ff,
            #ecfdf5
        );
}

.comparison-intro-title {
    color: #0f172a;
    font-size: 0.9rem;
    font-weight: 780;
}

.comparison-intro-note {
    margin-top: 0.15rem;
    color: #64748b;
    font-size: 0.7rem;
}

@media (max-width: 900px) {
    .backtest-kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}



.analytics-label {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;

    margin: 0.7rem 0 0.45rem 0;
    padding: 0.68rem 0.85rem;

    border: 1px solid #dbe4f0;
    border-radius: 0.8rem;

    background:
        linear-gradient(
            90deg,
            #f8fbff 0%,
            #ffffff 100%
        );
}

.analytics-label-left {
    display: flex;
    align-items: center;
    gap: 0.55rem;
}

.analytics-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;

    width: 1.85rem;
    height: 1.85rem;

    border-radius: 0.55rem;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #4f46e5
        );

    color: white;
    font-size: 0.72rem;
    font-weight: 800;
}

.analytics-title {
    color: #0f172a;
    font-size: 0.92rem;
    font-weight: 780;
    letter-spacing: -0.015em;
}

.analytics-tag {
    padding: 0.22rem 0.48rem;
    border-radius: 999px;

    background: #eef2ff;
    color: #4f46e5;

    font-size: 0.62rem;
    font-weight: 780;
}

.analytics-divider {
    height: 1px;
    margin: 0.65rem 0;
    background:
        linear-gradient(
            90deg,
            transparent,
            #dbe4f0,
            transparent
        );
}

[data-testid="stVegaLiteChart"] {
    border: 1px solid #e5eaf1;
    border-radius: 0.9rem;
    padding: 0.35rem;
    background: #ffffff;

    box-shadow:
        0 4px 16px rgba(15, 23, 42, 0.035);
}

[data-testid="stArrowVegaLiteChart"] {
    border-radius: 0.9rem;
}



.stock-card-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.9rem;
    margin: 0.8rem 0 0.7rem 0;
}

.stock-card {
    position: relative;
    overflow: hidden;
    min-height: 205px;
    padding: 1rem 1.05rem;
    border: 1px solid;
    border-radius: 1rem;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.055);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.stock-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.09);
}

.stock-card-active {
    box-shadow:
        0 0 0 2px rgba(37, 99, 235, 0.12),
        0 10px 26px rgba(37, 99, 235, 0.10);
}

.stock-card-blue {
    background: linear-gradient(145deg, #eff6ff, #ffffff 72%);
    border-color: #bfdbfe;
}

.stock-card-purple {
    background: linear-gradient(145deg, #f5f3ff, #ffffff 72%);
    border-color: #ddd6fe;
}

.stock-card-green {
    background: linear-gradient(145deg, #ecfdf5, #ffffff 72%);
    border-color: #a7f3d0;
}

.stock-card-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.75rem;
}

.stock-name {
    color: #0f172a;
    font-size: 1.22rem;
    font-weight: 820;
    letter-spacing: -0.025em;
}

.stock-company {
    margin-top: 0.08rem;
    color: #64748b;
    font-size: 0.7rem;
}

.stock-focus {
    padding: 0.24rem 0.48rem;
    border-radius: 999px;
    background: #2563eb;
    color: white;
    font-size: 0.62rem;
    font-weight: 800;
    letter-spacing: 0.05em;
}

.stock-period {
    padding: 0.24rem 0.48rem;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.12);
    color: #64748b;
    font-size: 0.62rem;
    font-weight: 750;
}

.stock-price-label {
    margin-top: 0.95rem;
    color: #64748b;
    font-size: 0.68rem;
    font-weight: 650;
}

.stock-price {
    margin-top: 0.08rem;
    color: #0f172a;
    font-size: 1.65rem;
    line-height: 1.05;
    font-weight: 820;
    letter-spacing: -0.035em;
}

.stock-period-return {
    margin-top: 0.28rem;
    font-size: 0.82rem;
    font-weight: 780;
}

.stock-positive {
    color: #059669;
}

.stock-negative {
    color: #dc2626;
}

.stock-stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.45rem 0.8rem;
    margin-top: 0.9rem;
    padding-top: 0.75rem;
    border-top: 1px solid rgba(148, 163, 184, 0.25);
}

.stock-stat-label {
    color: #94a3b8;
    font-size: 0.61rem;
    font-weight: 650;
}

.stock-stat-value {
    margin-top: 0.05rem;
    color: #334155;
    font-size: 0.76rem;
    font-weight: 760;
}

@media (max-width: 1000px) {
    .stock-card-grid {
        grid-template-columns: 1fr;
    }
}



.overview-kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.8rem;
    margin: 0.35rem 0 1.05rem 0;
}

.overview-kpi {
    position: relative;
    overflow: hidden;
    min-height: 108px;
    padding: 1rem 1.05rem;
    border-radius: 1rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 5px 18px rgba(15, 23, 42, 0.055);
}

.overview-kpi::after {
    content: "";
    position: absolute;
    width: 90px;
    height: 90px;
    right: -28px;
    top: -30px;
    border-radius: 50%;
    opacity: 0.18;
}

.kpi-blue {
    background:
        linear-gradient(
            135deg,
            #eff6ff 0%,
            #ffffff 74%
        );
    border-color: #bfdbfe;
}

.kpi-blue::after {
    background: #2563eb;
}

.kpi-green {
    background:
        linear-gradient(
            135deg,
            #ecfdf5 0%,
            #ffffff 74%
        );
    border-color: #a7f3d0;
}

.kpi-green::after {
    background: #10b981;
}

.kpi-purple {
    background:
        linear-gradient(
            135deg,
            #f5f3ff 0%,
            #ffffff 74%
        );
    border-color: #ddd6fe;
}

.kpi-purple::after {
    background: #7c3aed;
}

.kpi-orange {
    background:
        linear-gradient(
            135deg,
            #fff7ed 0%,
            #ffffff 74%
        );
    border-color: #fed7aa;
}

.kpi-orange::after {
    background: #f97316;
}

.kpi-label {
    position: relative;
    z-index: 1;
    color: #64748b;
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.045em;
}

.kpi-value {
    position: relative;
    z-index: 1;
    margin-top: 0.38rem;
    color: #0f172a;
    font-size: 1.48rem;
    line-height: 1.15;
    font-weight: 780;
    letter-spacing: -0.025em;
}

.kpi-note {
    position: relative;
    z-index: 1;
    margin-top: 0.34rem;
    color: #64748b;
    font-size: 0.74rem;
}

.portfolio-input-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin: 0.2rem 0 0.55rem 0;
}

.portfolio-input-title {
    color: #0f172a;
    font-size: 1.15rem;
    font-weight: 750;
    letter-spacing: -0.02em;
}

.portfolio-input-tag {
    display: inline-flex;
    padding: 0.28rem 0.58rem;
    border-radius: 999px;
    background: #dbeafe;
    color: #1d4ed8;
    font-size: 0.7rem;
    font-weight: 750;
}

div[data-testid="stTextInput"] input {
    background: #ffffff;
    border: 1px solid #bfdbfe;
    border-radius: 0.75rem;
    min-height: 2.75rem;
    font-weight: 650;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #2563eb;
    box-shadow:
        0 0 0 3px rgba(37, 99, 235, 0.10);
}

.var-model-card {
    min-height: 215px;
    padding: 1rem 1.05rem 0.9rem 1.05rem;
    border-radius: 1rem;
    border: 1px solid;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.055);
}

.var-model-blue {
    background:
        linear-gradient(
            145deg,
            #eff6ff 0%,
            #ffffff 76%
        );
    border-color: #bfdbfe;
}

.var-model-purple {
    background:
        linear-gradient(
            145deg,
            #f5f3ff 0%,
            #ffffff 76%
        );
    border-color: #ddd6fe;
}

.var-model-green {
    background:
        linear-gradient(
            145deg,
            #ecfdf5 0%,
            #ffffff 76%
        );
    border-color: #a7f3d0;
}

.var-model-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.85rem;
}

.var-model-name {
    color: #0f172a;
    font-size: 1rem;
    font-weight: 760;
}

.var-model-chip {
    padding: 0.25rem 0.52rem;
    border-radius: 999px;
    font-size: 0.66rem;
    line-height: 1;
    font-weight: 800;
    letter-spacing: 0.035em;
}

.chip-blue {
    background: #dbeafe;
    color: #1d4ed8;
}

.chip-purple {
    background: #ede9fe;
    color: #6d28d9;
}

.chip-green {
    background: #d1fae5;
    color: #047857;
}

.var-label {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 650;
}

.var-value {
    margin-top: 0.12rem;
    font-size: 2rem;
    font-weight: 820;
    line-height: 1.05;
    letter-spacing: -0.035em;
}

.var-blue {
    color: #2563eb;
}

.var-purple {
    color: #7c3aed;
}

.var-green {
    color: #059669;
}

.var-amount {
    margin-top: 0.55rem;
    color: #0f172a;
    font-size: 1.05rem;
    font-weight: 720;
}

.var-foot {
    margin-top: 0.7rem;
    padding-top: 0.7rem;
    border-top: 1px solid rgba(148, 163, 184, 0.28);
    color: #64748b;
    font-size: 0.7rem;
}

.risk-summary-note {
    margin-top: 0.75rem;
    padding: 0.7rem 0.85rem;
    border-radius: 0.75rem;
    background:
        linear-gradient(
            90deg,
            #eff6ff,
            #f5f3ff
        );
    border: 1px solid #dbeafe;
    color: #475569;
    font-size: 0.76rem;
}

@media (max-width: 1000px) {
    .overview-kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}



[data-testid="stSidebar"],
[data-testid="collapsedControl"] {
    display: none !important;
}

.stMainBlockContainer {
    max-width: 1540px;
    padding-left: 2rem;
    padding-right: 2rem;
}

.top-nav {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    flex-wrap: wrap;
    margin: -0.25rem 0 1.1rem 0;
    padding: 0.5rem;
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid #dbe4f0;
    border-radius: 0.9rem;
    box-shadow: 0 3px 14px rgba(15, 23, 42, 0.04);
}

.top-nav a {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 2.15rem;
    padding: 0.42rem 0.9rem;
    border-radius: 0.65rem;
    color: #475569 !important;
    text-decoration: none !important;
    font-size: 0.84rem;
    font-weight: 650;
    transition: all 0.15s ease;
}

.top-nav a:hover {
    background: linear-gradient(
        135deg,
        #2563eb,
        #4f46e5
    );
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.20);
}

.top-nav .nav-primary {
    background: linear-gradient(
        135deg,
        #2563eb,
        #4f46e5
    );
    color: #ffffff !important;
}



.section-heading {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin: 0.15rem 0 0.85rem 0;
}

.section-index {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 2.15rem;
    height: 2.15rem;
    padding: 0 0.55rem;
    border-radius: 0.6rem;
    background: #eff6ff;
    color: #2563eb;
    border: 1px solid #dbeafe;
    font-size: 0.78rem;
    font-weight: 800;
}

.section-title {
    color: #0f172a;
    font-size: 1.42rem;
    font-weight: 760;
    letter-spacing: -0.025em;
}

[data-testid="stMetric"] {
    min-height: 0;
}

[data-testid="stMetricValue"] {
    font-size: 1.65rem;
}

div[data-testid="stExpander"] details {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 0.9rem;
}

div[data-testid="stExpander"] summary {
    font-weight: 700;
}

hr {
    margin-top: 0.9rem;
    margin-bottom: 0.9rem;
}

.stCaptionContainer {
    margin-top: -0.15rem;
}



.app-hero {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
    padding: 1.15rem 1.25rem;
    margin-bottom: 1.2rem;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 1rem;
    box-shadow: 0 6px 22px rgba(15, 23, 42, 0.05);
}

.app-hero-title {
    color: #0f172a;
    font-size: 2rem;
    font-weight: 780;
    line-height: 1.1;
    letter-spacing: -0.035em;
}

.app-hero-subtitle {
    margin-top: 0.45rem;
    color: #64748b;
    font-size: 0.94rem;
}

.operational-badge {
    display: inline-flex;
    align-items: center;
    white-space: nowrap;
    padding: 0.42rem 0.75rem;
    border-radius: 999px;
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
    font-size: 0.8rem;
    font-weight: 700;
}

.operational-dot {
    width: 0.46rem;
    height: 0.46rem;
    margin-right: 0.42rem;
    border-radius: 50%;
    background: #10b981;
}

.sidebar-brand-title {
    color: #ffffff;
    font-size: 1.32rem;
    font-weight: 750;
}

.sidebar-brand-subtitle {
    margin-top: 0.2rem;
    color: #9fb0c4;
    font-size: 0.82rem;
}

.sidebar-nav {
    margin-top: 1.4rem;
}

.sidebar-nav a {
    display: block;
    padding: 0.72rem 0.8rem;
    margin-bottom: 0.25rem;
    border-radius: 0.65rem;
    color: #dce6f2 !important;
    text-decoration: none !important;
    font-size: 0.92rem;
    font-weight: 550;
}

.sidebar-nav a:hover {
    background: rgba(37, 99, 235, 0.35);
    color: #ffffff !important;
}

.sidebar-note {
    margin-top: 1.6rem;
    padding-top: 1.1rem;
    border-top: 1px solid rgba(255, 255, 255, 0.14);
    color: #91a3b8;
    font-size: 0.78rem;
    line-height: 1.5;
}

.section-anchor {
    scroll-margin-top: 1rem;
}
</style>

<div class="app-hero"><div><div class="app-hero-title">Portfolio VaR Risk Dashboard</div><div class="app-hero-subtitle">One-day 95% VaR for an equal-weight HPG / FPT / MWG portfolio</div></div><div class="operational-badge"><span class="operational-dot"></span>Operational</div></div>
""",
    unsafe_allow_html=True,
)

forecast = load_forecast()
metrics = load_metrics()
predictions = load_predictions()
market_snapshot = load_market_snapshot()

st.markdown(
    """
<div class="top-nav">
    <a class="nav-primary" href="#risk-overview">
        Risk Overview
    </a>
    <a href="#stock-explorer">
        Stock Explorer
    </a>
    <a href="#historical-backtesting">
        Historical Backtesting
    </a>
    <a href="#model-comparison">
        Model Comparison
    </a>
    <a href="#methodology">
        Methodology
    </a>
</div>
""",
    unsafe_allow_html=True,
)


st.markdown(
    '<div id="risk-overview" class="section-anchor"></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-heading">'
    '<span class="section-index">01</span>'
    '<span class="section-title">Risk Overview</span>'
    '</div>',
    unsafe_allow_html=True,
)

with st.container():
    cutoff = str(forecast["cutoff_date"].iloc[0])
    target = str(forecast["target_date"].iloc[0])

    st.write(
        f"Market information through **{cutoff}** → "
        f"next trading-session target **{target}**."
    )

    st.html(
        f"""
<div class="overview-kpi-grid">
    <div class="overview-kpi kpi-blue">
        <div class="kpi-label">Portfolio</div>
        <div class="kpi-value">HPG / FPT / MWG</div>
        <div class="kpi-note">Equal weight - 33.3% each</div>
    </div>

    <div class="overview-kpi kpi-green">
        <div class="kpi-label">Confidence Level</div>
        <div class="kpi-value">95%</div>
        <div class="kpi-note">One-day Value at Risk</div>
    </div>

    <div class="overview-kpi kpi-purple">
        <div class="kpi-label">Data Cutoff</div>
        <div class="kpi-value">{cutoff}</div>
        <div class="kpi-note">Latest operational market snapshot</div>
    </div>

    <div class="overview-kpi kpi-orange">
        <div class="kpi-label">Target Session</div>
        <div class="kpi-value">{target}</div>
        <div class="kpi-note">Next trading-session forecast</div>
    </div>
</div>
"""
    )

    st.html(
        """
<div class="portfolio-input-heading">
    <div class="portfolio-input-title">
        Portfolio Value & VaR Forecast
    </div>
    <div class="portfolio-input-tag">
        Equal-weight portfolio
    </div>
</div>
"""
    )

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

        model_styles = {
            "historical_simulation": {
                "card": "var-model-blue",
                "chip": "chip-blue",
                "value": "var-blue",
                "tag": "HIST",
            },
            "ewma": {
                "card": "var-model-purple",
                "chip": "chip-purple",
                "value": "var-purple",
                "tag": "EWMA",
            },
            "gradient_boosting": {
                "card": "var-model-green",
                "chip": "chip-green",
                "value": "var-green",
                "tag": "GB G04",
            },
        }

        for column, (_, row) in zip(
            columns,
            forecast.iterrows(),
        ):
            model = str(row["model"])
            var_return = float(row["var_return"])
            quantile_return = float(row["quantile_return"])
            estimated_loss = portfolio_value * var_return

            style = model_styles[model]

            with column:
                st.html(
                    f"""
<div class="var-model-card {style["card"]}">
    <div class="var-model-top">
        <div class="var-model-name">
            {MODEL_LABELS[model]}
        </div>
        <div class="var-model-chip {style["chip"]}">
            {style["tag"]}
        </div>
    </div>

    <div class="var-label">
        95% one-day VaR
    </div>

    <div class="var-value {style["value"]}">
        {var_return * 100:.4f}%
    </div>

    <div class="var-label" style="margin-top: 0.65rem;">
        Estimated VaR amount
    </div>

    <div class="var-amount">
        {format_vnd(estimated_loss)} VND
    </div>

    <div class="var-foot">
        5% return quantile:
        {quantile_return * 100:.4f}%
    </div>
</div>
"""
                )

        st.html(
            """
<div class="risk-summary-note">
    Operational VaR is shown above using the frozen model
    specifications. Canonical backtesting and model comparison
    remain separated in the sections below.
</div>
"""
        )


st.divider()
st.markdown(
    '<div id="stock-explorer" class="section-anchor"></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-heading">'
    '<span class="section-index">02</span>'
    '<span class="section-title">Stock Explorer</span>'
    '</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Explore the three portfolio constituents using the "
    "operational market snapshot. This section is descriptive "
    "and does not alter the frozen canonical evaluation."
)

explorer_control_left, explorer_control_right = st.columns(
    [1, 2]
)

with explorer_control_left:
    selected_ticker = st.selectbox(
        "Ticker",
        options=["HPG", "FPT", "MWG"],
        key="stock_explorer_ticker",
    )

with explorer_control_right:
    selected_period = st.segmented_control(
        "Period",
        options=["1M", "3M", "6M", "YTD", "1Y", "All"],
        default="1Y",
        key="stock_explorer_period",
    )

if selected_period is None:
    selected_period = "1Y"

stock_data = (
    market_snapshot.loc[
        market_snapshot["ticker"] == selected_ticker
    ]
    .sort_values("date")
    .copy()
)

period_data = filter_stock_period(
    stock_data=stock_data,
    period=selected_period,
)

if len(period_data) < 2:
    st.warning(
        "The selected period does not contain enough "
        "observations for stock analytics."
    )
else:
    period_data["simple_return"] = (
        period_data["close"]
        .pct_change()
    )

    return_series = (
        period_data["simple_return"]
        .dropna()
    )

    first_close = float(
        period_data["close"].iloc[0]
    )
    latest_close = float(
        period_data["close"].iloc[-1]
    )

    period_return = (
        latest_close / first_close - 1.0
    )

    annualized_volatility = float(
        return_series.std(ddof=1)
        * np.sqrt(252.0)
    )

    running_peak = (
        period_data["close"]
        .cummax()
    )

    drawdown = (
        period_data["close"]
        / running_peak
        - 1.0
    )

    maximum_drawdown = float(
        drawdown.min()
    )

    best_day = float(
        return_series.max()
    )

    worst_day = float(
        return_series.min()
    )

    positive_day_ratio = float(
        (return_series > 0.0).mean()
    )

    stock_card_styles = {
        "HPG": {
            "class": "stock-card-blue",
            "company": "Hoa Phat Group",
        },
        "FPT": {
            "class": "stock-card-purple",
            "company": "FPT Corporation",
        },
        "MWG": {
            "class": "stock-card-green",
            "company": "Mobile World",
        },
    }

    stock_card_html = []

    for card_ticker in ["HPG", "FPT", "MWG"]:
        card_source = (
            market_snapshot.loc[
                market_snapshot["ticker"] == card_ticker
            ]
            .sort_values("date")
            .copy()
        )

        card_period = filter_stock_period(
            card_source,
            selected_period,
        )

        if card_period.empty:
            continue

        card_returns = (
            card_period["close"]
            .pct_change()
            .dropna()
        )

        card_latest_close = float(
            card_period["close"].iloc[-1]
        )

        if len(card_period) > 1:
            card_period_return = float(
                card_latest_close
                / float(card_period["close"].iloc[0])
                - 1.0
            )
        else:
            card_period_return = 0.0

        if len(card_returns) > 1:
            card_volatility = float(
                card_returns.std(ddof=1)
                * np.sqrt(252.0)
            )
        else:
            card_volatility = 0.0

        card_running_peak = (
            card_period["close"].cummax()
        )

        card_drawdown = (
            card_period["close"]
            / card_running_peak
            - 1.0
        )

        card_max_drawdown = float(
            card_drawdown.min()
        )

        card_best_day = (
            float(card_returns.max())
            if not card_returns.empty
            else 0.0
        )

        card_worst_day = (
            float(card_returns.min())
            if not card_returns.empty
            else 0.0
        )

        style = stock_card_styles[card_ticker]

        active_class = (
            " stock-card-active"
            if card_ticker == selected_ticker
            else ""
        )

        return_class = (
            "stock-positive"
            if card_period_return >= 0
            else "stock-negative"
        )

        focus_badge = (
            '<span class="stock-focus">FOCUS</span>'
            if card_ticker == selected_ticker
            else (
                f'<span class="stock-period">'
                f'{selected_period}</span>'
            )
        )

        stock_card_html.append(
            f"""
<div class="stock-card {style["class"]}{active_class}">
    <div class="stock-card-top">
        <div>
            <div class="stock-name">{card_ticker}</div>
            <div class="stock-company">
                {style["company"]}
            </div>
        </div>
        {focus_badge}
    </div>

    <div class="stock-price-label">
        Latest Close
    </div>

    <div class="stock-price">
        {card_latest_close:,.2f}
    </div>

    <div class="stock-period-return {return_class}">
        {selected_period} Return:
        {card_period_return * 100:+.2f}%
    </div>

    <div class="stock-stats">
        <div>
            <div class="stock-stat-label">
                Annualized Volatility
            </div>
            <div class="stock-stat-value">
                {card_volatility * 100:.2f}%
            </div>
        </div>

        <div>
            <div class="stock-stat-label">
                Max Drawdown
            </div>
            <div class="stock-stat-value stock-negative">
                {card_max_drawdown * 100:.2f}%
            </div>
        </div>

        <div>
            <div class="stock-stat-label">
                Best Day
            </div>
            <div class="stock-stat-value stock-positive">
                {card_best_day * 100:+.2f}%
            </div>
        </div>

        <div>
            <div class="stock-stat-label">
                Worst Day
            </div>
            <div class="stock-stat-value stock-negative">
                {card_worst_day * 100:.2f}%
            </div>
        </div>
    </div>
</div>
"""
        )

    st.html(
        '<div class="stock-card-grid">'
        + "".join(stock_card_html)
        + "</div>"
    )

    st.caption(
        f"Positive trading days: "
        f"{positive_day_ratio * 100:.1f}% | "
        f"Observations: {len(period_data):,} | "
        f"Range: "
        f"{period_data['date'].min().date()} to "
        f"{period_data['date'].max().date()}"
    )

    st.html(
        f"""
<div class="analytics-label">
    <div class="analytics-label-left">
        <span class="analytics-icon">PX</span>
        <span class="analytics-title">
            {selected_ticker} Closing Price
        </span>
    </div>
    <span class="analytics-tag">{selected_period}</span>
</div>
"""
    )

    price_chart = (
        period_data[
            ["date", "close"]
        ]
        .rename(
            columns={
                "date": "Date",
                "close": "Close",
            }
        )
        .set_index("Date")
    )

    st.line_chart(
        price_chart,
        y="Close",
        height=240,
    )

    st.caption(
        "Stock Explorer uses the operational snapshot only. "
        "Metrics shown here are descriptive and are not used "
        "to retune or re-evaluate the frozen VaR models."
    )

    st.html(
        """
<div class="analytics-label">
    <div class="analytics-label-left">
        <span class="analytics-icon">RT</span>
        <span class="analytics-title">
            Return Distribution & Drawdown
        </span>
    </div>
    <span class="analytics-tag">Tail Risk</span>
</div>
"""
    )

    visual_left, visual_right = st.columns(2)

    distribution_frame = pd.DataFrame(
        {
            "Return (%)": (
                return_series * 100.0
            )
        }
    )

    quantile_5 = float(
        return_series.quantile(0.05) * 100.0
    )

    distribution_values = (
        distribution_frame
        .to_dict(orient="records")
    )

    distribution_spec = {
        "height": 260,
        "data": {
            "values": distribution_values,
        },
        "layer": [
            {
                "mark": {
                    "type": "bar",
                    "opacity": 0.85,
                    "color": "#2563EB",
                },
                "encoding": {
                    "x": {
                        "field": "Return (%)",
                        "type": "quantitative",
                        "bin": {
                            "maxbins": 45,
                        },
                        "title": "Daily simple return (%)",
                    },
                    "y": {
                        "aggregate": "count",
                        "type": "quantitative",
                        "title": "Trading days",
                    },
                    "tooltip": [
                        {
                            "field": "Return (%)",
                            "type": "quantitative",
                            "bin": True,
                            "title": "Return range (%)",
                        },
                        {
                            "aggregate": "count",
                            "type": "quantitative",
                            "title": "Trading days",
                        },
                    ],
                },
            },
            {
                "mark": {
                    "type": "rule",
                    "color": "#DC2626",
                    "strokeWidth": 2,
                    "strokeDash": [6, 4],
                },
                "encoding": {
                    "x": {
                        "datum": quantile_5,
                    },
                },
            },
        ],
    }

    drawdown_frame = pd.DataFrame(
        {
            "Date": period_data["date"],
            "Drawdown (%)": drawdown * 100.0,
        }
    )

    drawdown_values = (
        drawdown_frame
        .assign(
            Date=lambda frame: (
                frame["Date"]
                .dt.strftime("%Y-%m-%d")
            )
        )
        .to_dict(orient="records")
    )

    drawdown_spec = {
        "height": 260,
        "data": {
            "values": drawdown_values,
        },
        "mark": {
            "type": "area",
            "color": "#DC2626",
            "opacity": 0.35,
            "line": {
                "color": "#DC2626",
                "strokeWidth": 1.5,
            },
        },
        "encoding": {
            "x": {
                "field": "Date",
                "type": "temporal",
                "title": "Date",
            },
            "y": {
                "field": "Drawdown (%)",
                "type": "quantitative",
                "title": "Drawdown (%)",
            },
            "tooltip": [
                {
                    "field": "Date",
                    "type": "temporal",
                    "title": "Date",
                },
                {
                    "field": "Drawdown (%)",
                    "type": "quantitative",
                    "format": ".2f",
                    "title": "Drawdown (%)",
                },
            ],
        },
    }

    with visual_left:
        st.markdown("##### Daily Return Distribution")
        st.vega_lite_chart(
            distribution_spec,
            width="stretch",
        )
        st.caption(
            f"Dashed red line: empirical 5% return "
            f"quantile = {quantile_5:.2f}%."
        )

    with visual_right:
        st.markdown("##### Drawdown from Running Peak")
        st.vega_lite_chart(
            drawdown_spec,
            width="stretch",
        )
        st.caption(
            "Drawdown measures the decline from the "
            "running closing-price peak."
        )

    st.markdown("#### Portfolio Context")

    period_start_date = (
        period_data["date"].min()
    )

    portfolio_market = (
        market_snapshot.loc[
            market_snapshot["ticker"].isin(
                ["HPG", "FPT", "MWG"]
            )
            & (
                market_snapshot["date"]
                >= period_start_date
            )
        ]
        .copy()
    )

    close_matrix = (
        portfolio_market
        .pivot(
            index="date",
            columns="ticker",
            values="close",
        )
        .sort_index()
        .loc[:, ["HPG", "FPT", "MWG"]]
        .dropna()
    )

    if len(close_matrix) < 2:
        st.warning(
            "Insufficient aligned observations for "
            "portfolio context analytics."
        )
    else:
        normalized_prices = (
            close_matrix
            .div(close_matrix.iloc[0])
            .mul(100.0)
        )

        normalized_chart = (
            normalized_prices
            .reset_index()
            .melt(
                id_vars="date",
                var_name="Ticker",
                value_name="Normalized Price",
            )
        )

        normalized_chart["date"] = (
            normalized_chart["date"]
            .dt.strftime("%Y-%m-%d")
        )

        normalized_values = (
            normalized_chart
            .to_dict(orient="records")
        )

        normalized_spec = {
            "height": 240,
            "data": {
                "values": normalized_values,
            },
            "mark": {
                "type": "line",
                "strokeWidth": 2,
            },
            "encoding": {
                "x": {
                    "field": "date",
                    "type": "temporal",
                    "title": "Date",
                },
                "y": {
                    "field": "Normalized Price",
                    "type": "quantitative",
                    "title": "Normalized price (base = 100)",
                },
                "color": {
                    "field": "Ticker",
                    "type": "nominal",
                    "title": "Ticker",
                },
                "tooltip": [
                    {
                        "field": "date",
                        "type": "temporal",
                        "title": "Date",
                    },
                    {
                        "field": "Ticker",
                        "type": "nominal",
                    },
                    {
                        "field": "Normalized Price",
                        "type": "quantitative",
                        "format": ".2f",
                    },
                ],
            },
        }

        aligned_returns = (
            close_matrix
            .pct_change()
            .dropna()
        )

        correlation = (
            aligned_returns
            .corr()
        )

        correlation_long = (
            correlation
            .rename_axis("Ticker")
            .reset_index()
            .melt(
                id_vars="Ticker",
                var_name="Compared With",
                value_name="Correlation",
            )
        )

        correlation_long["Correlation Label"] = (
            correlation_long["Correlation"]
            .map(lambda value: f"{value:.3f}")
        )

        correlation_values = (
            correlation_long
            .to_dict(orient="records")
        )

        ticker_order = [
            "HPG",
            "FPT",
            "MWG",
        ]

        correlation_spec = {
            "height": 240,
            "data": {
                "values": correlation_values,
            },
            "layer": [
                {
                    "mark": {
                        "type": "rect",
                        "cornerRadius": 4,
                    },
                    "encoding": {
                        "x": {
                            "field": "Compared With",
                            "type": "nominal",
                            "sort": ticker_order,
                            "title": None,
                        },
                        "y": {
                            "field": "Ticker",
                            "type": "nominal",
                            "sort": ticker_order,
                            "title": None,
                        },
                        "color": {
                            "field": "Correlation",
                            "type": "quantitative",
                            "scale": {
                                "domain": [-1, 1],
                                "scheme": "redblue",
                                "reverse": True,
                            },
                            "title": "Correlation",
                        },
                        "tooltip": [
                            {
                                "field": "Ticker",
                                "type": "nominal",
                            },
                            {
                                "field": "Compared With",
                                "type": "nominal",
                            },
                            {
                                "field": "Correlation",
                                "type": "quantitative",
                                "format": ".3f",
                            },
                        ],
                    },
                },
                {
                    "mark": {
                        "type": "text",
                        "fontSize": 13,
                        "fontWeight": "bold",
                    },
                    "encoding": {
                        "x": {
                            "field": "Compared With",
                            "type": "nominal",
                            "sort": ticker_order,
                        },
                        "y": {
                            "field": "Ticker",
                            "type": "nominal",
                            "sort": ticker_order,
                        },
                        "text": {
                            "field": "Correlation Label",
                            "type": "nominal",
                        },
                        "color": {
                            "condition": {
                                "test": (
                                    "abs(datum.Correlation) >= 0.65"
                                ),
                                "value": "white",
                            },
                            "value": "#111827",
                        },
                    },
                },
            ],
        }

        context_left, context_right = st.columns(
            [2, 1]
        )

        with context_left:
            st.html(
                """
<div class="analytics-label">
    <div class="analytics-label-left">
        <span class="analytics-icon">PF</span>
        <span class="analytics-title">
            Normalized Portfolio Constituents
        </span>
    </div>
    <span class="analytics-tag">Base 100</span>
</div>
"""
            )
            st.vega_lite_chart(
                normalized_spec,
                width="stretch",
            )
            st.caption(
                "Each closing-price series is rebased to "
                "100 at the beginning of the selected period."
            )

        with context_right:
            st.html(
                """
<div class="analytics-label">
    <div class="analytics-label-left">
        <span class="analytics-icon">CR</span>
        <span class="analytics-title">
            Return Correlation
        </span>
    </div>
    <span class="analytics-tag">Pearson</span>
</div>
"""
            )
            st.vega_lite_chart(
                correlation_spec,
                width="stretch",
            )
            st.caption(
                "Pearson correlation of aligned daily "
                "simple returns over the selected period."
            )

    st.info(
        "Stock Explorer is an operational descriptive layer. "
        "The selected ticker and period do not change the "
        "portfolio definition, frozen model configurations, "
        "or canonical backtesting evidence."
    )


st.divider()
st.markdown(
    '<div id="historical-backtesting" class="section-anchor"></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-heading">'
    '<span class="section-index">03</span>'
    '<span class="section-title">Historical Backtesting</span>'
    '</div>',
    unsafe_allow_html=True,
)

with st.container():
    st.markdown("**Frozen walk-forward backtesting**")

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

    st.html(
        f"""
<div class="backtest-kpi-grid">

    <div class="backtest-kpi bt-blue">
        <div class="bt-label">Observations</div>
        <div class="bt-value">
            {len(model_predictions):,}
        </div>
    </div>

    <div class="backtest-kpi bt-red">
        <div class="bt-label">VaR Violations</div>
        <div class="bt-value">
            {violation_count}
        </div>
    </div>

    <div class="backtest-kpi bt-orange">
        <div class="bt-label">Violation Rate</div>
        <div class="bt-value">
            {violation_rate * 100:.2f}%
        </div>
    </div>

    <div class="backtest-kpi bt-purple">
        <div class="bt-label">Average VaR</div>
        <div class="bt-value">
            {average_var * 100:.2f}%
        </div>
    </div>

</div>
"""
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

    backtest_model_colors = {
        "historical_simulation": "#2563EB",
        "ewma": "#7C3AED",
        "gradient_boosting": "#10B981",
    }

    selected_model_color = backtest_model_colors[
        selected_method
    ]

    st.html(
        f"""
<div class="evaluation-chart-header">
    <div class="evaluation-chart-title">
        <span
            class="model-color-dot"
            style="background:{selected_model_color};"
        ></span>
        {MODEL_LABELS[selected_method]}
        - realized returns vs. VaR threshold
    </div>

    <div class="evaluation-chip">
        Frozen walk-forward
    </div>
</div>
"""
    )

    backtest_spec = {
        "height": 290,
        "data": {
            "values": chart_values,
        },
        "layer": [
            {
                "mark": {
                    "type": "line",
                    "strokeWidth": 1.5,
                    "color": selected_model_color,
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


st.divider()
st.markdown(
    '<div id="model-comparison" class="section-anchor"></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-heading">'
    '<span class="section-index">04</span>'
    '<span class="section-title">Model Comparison</span>'
    '</div>',
    unsafe_allow_html=True,
)

with st.container():
    st.markdown("**Frozen canonical evaluation**")

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

    with st.expander(
        "Canonical evaluation table",
        expanded=False,
    ):
        st.dataframe(
            comparison,
            hide_index=True,
            width="stretch",
        )

    st.html(
        """
<div class="comparison-intro">
    <div class="comparison-intro-title">
        Model Performance Comparison
    </div>
    <div class="comparison-intro-note">
        Blue = Historical ? Purple = EWMA ? Green = GB G04
    </div>
</div>
"""
    )

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
        "height": 190,
        "data": {
            "values": comparison_values,
        },
        "layer": [
            {
                "mark": {
                    "type": "bar",
                    "cornerRadiusEnd": 4,
                },
                "encoding": {
                "color": {
                    "field": "Model",
                    "type": "nominal",
                    "scale": {
                        "domain": model_order,
                        "range": [
                            "#2563EB",
                            "#7C3AED",
                            "#10B981",
                        ],
                    },
                    "legend": None,
                },
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
        "height": 190,
        "data": {
            "values": comparison_values,
        },
        "mark": {
            "type": "bar",
            "cornerRadiusEnd": 4,
        },
        "encoding": {
            "color": {
                "field": "Model",
                "type": "nominal",
                "scale": {
                    "domain": model_order,
                    "range": [
                        "#2563EB",
                        "#7C3AED",
                        "#10B981",
                    ],
                },
                "legend": None,
            },
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
        "height": 190,
        "data": {
            "values": comparison_values,
        },
        "mark": {
            "type": "bar",
            "cornerRadiusEnd": 4,
        },
        "encoding": {
            "color": {
                "field": "Model",
                "type": "nominal",
                "scale": {
                    "domain": model_order,
                    "range": [
                        "#2563EB",
                        "#7C3AED",
                        "#10B981",
                    ],
                },
                "legend": None,
            },
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


st.divider()
st.markdown(
    '<div id="methodology" class="section-anchor"></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-heading">'
    '<span class="section-index">05</span>'
    '<span class="section-title">Methodology & Research Contract</span>'
    '</div>',
    unsafe_allow_html=True,
)

with st.expander("Research contract, controls & limitations", expanded=False):
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

st.html(
    """
<div class="dashboard-footer">

    <div>
        <div class="footer-brand">
            <span class="footer-dots">
                <span class="footer-dot footer-blue"></span>
                <span class="footer-dot footer-purple"></span>
                <span class="footer-dot footer-green"></span>
            </span>
            Portfolio VaR Risk Dashboard
        </div>
    </div>

    <div class="footer-meta">
        Quantitative Risk Management ?
        HPG / FPT / MWG ?
        One-day 95% VaR
    </div>

</div>
"""
)
