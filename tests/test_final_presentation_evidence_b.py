from pathlib import Path
import subprocess

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results/final_presentation_evidence_b.csv"
PRED = ROOT / "results/final_predictions.csv"
COMP = ROOT / "results/final_metric_comparison.csv"
PAIR = ROOT / "results/final_pairwise_summary.csv"
DOC = ROOT / "docs/day26-final-presentation-qa-b.md"
BUILDER = ROOT / "scripts/build_final_presentation_evidence_b.py"

COLS = [
    "evidence_id",
    "section",
    "claim",
    "expected",
    "actual",
    "tolerance",
    "status",
    "source",
]


def evidence():
    return pd.read_csv(CSV, dtype=str)


def test_csv_schema_exact():
    assert list(evidence().columns) == COLS


def test_exactly_28_rows():
    assert len(evidence()) == 28


def test_ids_01_to_28_unique():
    df = evidence()
    assert df["evidence_id"].tolist() == [
        f"{i:02d}" for i in range(1, 29)
    ]
    assert df["evidence_id"].is_unique


def test_all_status_pass():
    assert evidence()["status"].eq("PASS").all()


def test_all_tolerance_exact():
    assert evidence()["tolerance"].eq("1e-12").all()


def test_sources_are_tracked_and_allowed():
    allowed = {
        "results/final_predictions.csv",
        "results/final_metric_comparison.csv",
        "results/final_pairwise_summary.csv",
    }

    sources = set(evidence()["source"])
    assert sources <= allowed

    for source in sources:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", source],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, source


def test_prediction_panel_contract():
    p = pd.read_csv(PRED)

    assert len(p) == 1194
    assert p["method"].nunique() == 3
    assert p.groupby("method").size().eq(398).all()
    assert 1194 == 3 * 398
    assert p.duplicated(["method", "target_date"]).sum() == 0

    actuals = p.groupby("target_date")["actual_return"].nunique()
    assert actuals.eq(1).all()


def test_exact_date_range():
    p = pd.read_csv(PRED)

    assert p["target_date"].min() == "2024-12-18"
    assert p["target_date"].max() == "2026-07-28"
    assert p["target_date"].nunique() == 398


def test_exact_config_ids():
    p = pd.read_csv(PRED)

    expected = {
        "historical_simulation": "historical_w250",
        "ewma": "ewma_d094",
        "gradient_boosting": "gb_G04",
    }

    for method, config_id in expected.items():
        values = p.loc[
            p["method"] == method,
            "config_id",
        ].unique().tolist()

        assert values == [config_id]


def test_criterion_leaders():
    c = pd.read_csv(COMP)

    calibration = c.loc[
        c["calibration_leader"].astype(str).str.lower() == "true",
        "method",
    ].tolist()

    pinball = c.loc[
        c["pinball_leader"].astype(str).str.lower() == "true",
        "method",
    ].tolist()

    lowest_var = c.loc[
        c["lowest_average_var"].astype(str).str.lower() == "true",
        "method",
    ].tolist()

    assert calibration == ["ewma"]
    assert pinball == ["gradient_boosting"]
    assert lowest_var == ["historical_simulation"]


def test_exact_pairwise_evidence():
    p = pd.read_csv(PAIR).set_index("comparison")

    expected = {
        "gb_vs_historical": (
            151,
            247,
            -0.0001383332817084946,
        ),
        "gb_vs_ewma": (
            231,
            167,
            -0.00021019030172751563,
        ),
        "ewma_vs_historical": (
            131,
            267,
            0.00007185702001901683,
        ),
    }

    for name, (left, right, delta) in expected.items():
        row = p.loc[name]

        assert int(row["observation_count"]) == 398
        assert int(row["left_better_count"]) == left
        assert int(row["right_better_count"]) == right
        assert int(row["tie_count"]) == 0
        assert abs(float(row["mean_delta"]) - delta) <= 1e-12


def test_document_guardrails_and_builder_boundary():
    text = DOC.read_text(encoding="utf-8")
    builder = BUILDER.read_text(encoding="utf-8").lower()

    headings = [
        "## Objective",
        "## Frozen Presentation Contract",
        "## Aggregate Quantitative Evidence",
        "## Calibration Story",
        "## Quantile Accuracy Story",
        "## Risk Magnitude Story",
        "## Pairwise Evidence",
        "## Presentation Language Guardrails",
        "## Reproducibility Boundary",
        "## Evaluation Limitations",
        "## Provenance Boundary",
        "## Presentation Handoff",
    ]

    for heading in headings:
        assert heading in text

    required = [
        "There is no single overall winner",
        "Lower average VaR is not evidence of automatic model superiority.",
        "not a pristine never-inspected test set",
        "EWMA decay `0.94` is canonical but must not be described as the validation winner.",
        "known provenance completeness limitation",
        "Byte-for-byte figure reproducibility is limited to the tested environment.",
        "Gradient Boosting has the lowest mean pinball loss.",
        "Historical Simulation has the lowest average reported VaR",
        "different evaluation criteria answer different questions",
    ]

    for phrase in required:
        assert phrase in text

    forbidden = [
        "data/processed/portfolio_returns.csv",
        "run_final_walk_forward",
        "src.gb_market_features",
        "requests",
        "urllib",
        "socket",
        "http://",
        "https://",
    ]

    for token in forbidden:
        assert token not in builder