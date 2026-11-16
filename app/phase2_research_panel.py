"""Phase 2: frozen research evidence viewer; no training and no network requests."""
from pathlib import Path
import base64
import json
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phase2/b/reproduce_day07_20261113_133340"
SOURCE_SHA = "6389586ae8f1edbabb31760e0b555619186114b3f77d09212362fc2e1c40aadf"
LABELS = {
    "historical_w250": "Historical Simulation (250)",
    "ewma_d094": "EWMA (lambda=0.94)",
    "gb_G04_7f": "Gradient Boosting G04 (7F)",
    "gbv2_12f_c1": "Gradient Boosting v2 C1 (12F)",
    "gbv2_12f_c2": "Gradient Boosting v2 C2 (12F)",
}
FIGURES = [
    ("F01", "Market coverage & data quality", "figures/phase2/a/F01_v2.png"),
    ("F02", "Feature missingness", "figures/phase2/a/F02.png"),
    ("F03", "Expanding CV design", "figures/phase2/a/F03_v2.png"),
    ("F04", "Controlled 7 vs 12 feature ablation", "figures/phase2/a/F04.png"),
    ("F05", "Gradient Boosting CV fold pinball", "results/phase2/a/backtesting_2026-11-06/figures/cv_fold_pinball.png"),
    ("F06", "Normalized constituent trends", "figures/phase2/b/eda_2026-11-06/normalized_trend.png"),
    ("F07", "Portfolio return distribution", "figures/phase2/b/eda_2026-11-06/return_distribution.png"),
    ("F08", "Rolling volatility & drawdown", "figures/phase2/b/eda_2026-11-06/vol_drawdown.png"),
    ("F09", "Baseline CV final 120 dates", "figures/phase2/b/aligned_2026-11-06/baseline_var_timeline.png"),
    ("F10", "Return correlation matrix", "figures/phase2/b/correlation_2026-11-06/correlation.png"),
    ("F11", "Actual returns vs 5% forecast quantiles", "figures/phase2/b/backtesting_2026-11-06/actual_vs_var.png"),
    ("F12", "Monthly exceptions", "figures/phase2/b/backtesting_2026-11-06/exceptions_monthly_panels.png"),
    ("F13", "Retrospective holdout comparison", "figures/phase2/b/model_comparison.png"),
]


def load_evidence():
    metadata = json.loads((BASE / "metadata.json").read_text(encoding="utf-8-sig"))
    assert metadata["source_sha256"] == SOURCE_SHA, "Incorrect Phase 2 snapshot"
    assert metadata["snapshot_cutoff"] == "2026-11-06"
    assert metadata["training_cutoff"] == "2024-12-17"
    assert metadata["feature_rows"] == 1647
    assert metadata["holdout_rows"] == 468

    cv = pd.read_csv(BASE / "cv_metrics.csv")
    hold = pd.read_csv(BASE / "holdout_metrics.csv")
    predictions = pd.read_csv(BASE / "holdout_predictions.csv")
    expected_cv = set(LABELS)
    expected_hold = expected_cv - {"gbv2_12f_c2"}
    assert len(cv) == 25 and set(cv.model) == expected_cv
    assert len(hold) == 4 and set(hold.model) == expected_hold
    assert len(predictions) == 1872 and set(predictions.model) == expected_hold
    assert (cv.groupby("model")["n"].sum() == 980).all()
    assert (hold["n"] == 468).all()
    assert (predictions.groupby("model").size() == 468).all()
    assert set(predictions["fold"]) == {"holdout_fixed_fit"}

    predictions["target_date"] = pd.to_datetime(predictions["target_date"])
    predictions["as_of"] = pd.to_datetime(predictions["as_of"])
    assert (predictions["as_of"] < predictions["target_date"]).all()
    assert predictions["target_date"].min() == pd.Timestamp("2024-12-18")
    assert predictions["target_date"].max() == pd.Timestamp("2026-11-06")
    violations = predictions["violation"].astype(str).str.lower().map({"true":True,"false":False})
    assert violations.notna().all()
    predictions["violation"] = violations.astype(bool)
    assert (predictions["violation"] == (predictions["actual_return"] < predictions["quantile_return"])).all()
    counts = predictions.groupby("model")["violation"].sum().astype(int)
    reported = hold.set_index("model")["violations"].astype(int)
    assert (counts.sort_index() == reported.sort_index()).all()

    for path_id, _, path in FIGURES:
        assert (ROOT / path).is_file(), f"Missing figure {path_id}: {path}"

    return cv, hold, predictions, metadata


def make_summary(frame):
    return (frame.assign(
        Model=frame["model"].map(LABELS),
        **{"Pinball loss":frame["pinball_loss"],
           "Violation rate (%)": 100 * frame["violation_rate"],
           "Violations":frame["violations"]}
    )[["Model","Violations","Violation rate (%)","Pinball loss"]]
            .sort_values("Pinball loss"))


def render_phase2_research():
    st.divider()
    st.markdown('<div id="phase2-research" class="section-anchor"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-heading"><span class="section-index">06</span>'
        '<span class="section-title">Phase 2 Research & Validation</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("Historical research snapshot: KBS 06/11/2026 · 1-day 95% VaR · equal-weight HPG / FPT / MWG")
    st.info("Phase 2 results are **research-only**. They do not replace the frozen Phase 1 dashboard forecasts or operational dates above.")
    try:
        cv, hold, pred, metadata = load_evidence()
    except Exception as exc:
        st.error("Phase 2 evidence is missing or violates the frozen data contract. Phase 1 remains unchanged.")
        st.code(f"{type(exc).__name__}: {exc}")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Return observations", "1,707")
    k2.metric("Aligned feature rows", "1,647")
    k3.metric("Development CV", "5 × 196")
    k4.metric("Retrospective holdout", "468 sessions")

    overview, cvtab, holdtab, figures_tab, audit = st.tabs([
        "Key findings", "Development CV", "Retrospective holdout", "13 figures", "Evidence & caveats"
    ])

    with overview:
        st.markdown("#### What the data support")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.metric("Best overall CV pinball", "EWMA")
            st.caption("Mean pinball loss: 0.001963972")
        with c2:
            st.metric("Best GB development CV", "GBv2 C1")
            st.caption("Mean pinball loss: 0.002080267")
        with c3:
            st.metric("Lowest retrospective holdout pinball", "GBv2 C1")
            st.caption("Mean pinball loss: 0.001636961")
        st.markdown(
            "In a **controlled C1 feature ablation** over five folds, 12 features lower mean pinball "
            "loss by **1.4124%** relative to 7 features at identical model hyperparameters. "
            "This does not establish statistical significance or improved future performance."
        )
        st.warning("EWMA is best across all five CV models; GBv2 C1 is best **within GB models**. "
                   "Holdout is retrospective and overlaps Phase 1 evaluation.")

    with cvtab:
        st.markdown("#### Five expanding validation folds — 980 predictions per model")
        cvagg = cv.groupby("model", as_index=False).agg(
            n=("n", "sum"), violations=("violations", "sum"), pinball_loss=("pinball_loss", "mean")
        )
        cvagg["violation_rate"] = cvagg["violations"] / cvagg["n"]
        st.dataframe(make_summary(cvagg).style.format({"Violation rate (%)":"{:.2f}","Pinball loss":"{:.9f}"}), use_container_width=True, hide_index=True)
        fold = cv.copy()
        fold["Model"] = fold["model"].map(LABELS)
        fold["Fold"] = fold["fold"]
        chart = alt.Chart(fold).mark_line(point=True).encode(
            x=alt.X("Fold:N", sort=["dev_1","dev_2","dev_3","dev_4","dev_5"],title="Expanding CV fold"),
            y=alt.Y("pinball_loss:Q",title="Mean 5% pinball loss"),
            color="Model:N", tooltip=["Model:N","Fold:N",alt.Tooltip("pinball_loss:Q",format=".7f")]
        ).properties(height=340)
        st.altair_chart(chart, use_container_width=True)
        st.caption("CV ranks are calculated on the development period, not on the retrospective holdout.")

    with holdtab:
        st.markdown("#### Retrospective holdout: 18/12/2024–06/11/2026")
        st.dataframe(make_summary(hold).style.format({"Violation rate (%)":"{:.2f}","Pinball loss":"{:.9f}"}), use_container_width=True, hide_index=True)
        available = list(hold["model"])
        selected = st.selectbox("Inspect a retrospective model", available,
                               format_func=lambda x: LABELS[x], key="phase2_holdout_model")
        series = pred[pred["model"]==selected].sort_values("target_date").copy()
        actual = alt.Chart(series).mark_line(color="#64748b",strokeWidth=1).encode(
            x=alt.X("target_date:T",title="Target session"),
            y=alt.Y("actual_return:Q",title="Portfolio simple return / quantile"),
            tooltip=["target_date:T",alt.Tooltip("actual_return:Q",format=".3%")]
        )
        limit = alt.Chart(series).mark_line(color="#0891b2",strokeWidth=2).encode(
            x="target_date:T", y="quantile_return:Q",
            tooltip=["target_date:T",alt.Tooltip("quantile_return:Q",format=".3%")]
        )
        exceptions = alt.Chart(series[series["violation"]]).mark_circle(size=55,color="#e11d48").encode(
            x="target_date:T",y="actual_return:Q",
            tooltip=["target_date:T",alt.Tooltip("actual_return:Q",format=".3%")]
        )
        st.altair_chart((actual+limit+exceptions).properties(height=330).interactive(),use_container_width=True)
        st.caption("Gray: observed return · Teal: forecast 5% quantile · Red: violations (return < quantile). "
                   "This historical output is **not** a live next-session forecast.")
        selected_metric = hold[hold["model"]==selected].iloc[0]
        a,b,c=st.columns(3)
        a.metric("Exceptions", f"{int(selected_metric['violations'])}/468")
        b.metric("Violation rate", f"{selected_metric['violation_rate']*100:.2f}%")
        c.metric("Pinball loss", f"{selected_metric['pinball_loss']:.9f}")

    with figures_tab:
        st.markdown("#### Thirteen audited research figures")
        st.caption("F01_v2 and F03_v2 are the final presentation versions. F09 shows only the last 120 CV sessions.")
        for idx in range(0,len(FIGURES),2):
            left,right=st.columns(2)
            for col, entry in zip((left,right),FIGURES[idx:idx+2]):
                fig_id,title,path=entry
                with col:
                    with st.expander(f"{fig_id} · {title}",expanded=False):
                        png_b64 = base64.b64encode(
                            (ROOT / path).read_bytes()
                        ).decode("ascii")

                        st.html(
                            '<img alt="Phase 2 research figure" '
                            'style="display:block;width:100%;height:auto" '
                            f'src="data:image/png;base64,{png_b64}">'
                        )
                        st.caption(path)
        st.caption("F06 stock-price corporate-action adjustment is not fully verified. F07 quantile is descriptive, not conditional VaR.")

    with audit:
        st.markdown("#### Evidence contract and pending human sign-offs")
        st.code("snapshot SHA256: " + SOURCE_SHA)
        st.write("Phase 2: five expanding CV folds, 980 development validation forecasts/model; "
                 "468 retrospective holdout forecasts/model for HS/EWMA/G04/C1.")
        st.write("A/B GB numerical cross-check: 1,647 feature rows; common GB prediction comparisons PASS. "
                 "Baseline seven features share a builder; five extra features implemented separately by B.")
        st.warning("G06 A/B dual sign-off: PENDING · G07 release/figure acceptance: PENDING · G08 final handoff: PENDING")
        st.warning("The holdout overlaps previously evaluated Phase 1 history. Historical, exploratory comparisons "
                   "must not be advertised as a pristine out-of-sample or live success.")
        st.caption("Phase 3 is separate: exploratory 12/18/25 feature studies and refit-policy comparisons "
                   "did not confirm an improvement over C1; no promoted production model.")
