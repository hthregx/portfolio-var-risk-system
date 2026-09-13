from pathlib import Path
import subprocess

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FACTS = ROOT / "results/final_defense_facts_b.csv"
MANIFEST = ROOT / "results/final_submission_manifest_b.csv"
RUNBOOK = ROOT / "docs/day27-final-demo-defense-runbook-b.md"
BUILDER = ROOT / "scripts/build_final_submission_readiness_b.py"

FACT_COLUMNS = [
    "fact_id", "topic", "fact", "expected",
    "actual", "tolerance", "status", "source",
]

MANIFEST_COLUMNS = [
    "artifact_id", "category", "path",
    "required", "tracked", "git_blob_sha", "status",
]


def facts():
    return pd.read_csv(FACTS, dtype=str)


def manifest():
    return pd.read_csv(MANIFEST, dtype=str)


def test_defense_fact_schema_and_rows():
    df = facts()
    assert list(df.columns) == FACT_COLUMNS
    assert len(df) == 20


def test_all_defense_facts_pass():
    df = facts()
    assert df["status"].eq("PASS").all()
    assert df["tolerance"].eq("1e-12").all()


def test_fact_ids_unique():
    df = facts()
    assert df["fact_id"].tolist() == [f"{i:02d}" for i in range(1, 21)]
    assert df["fact_id"].is_unique


def test_frozen_quantitative_values_exact():
    df = facts().set_index("fact_id")

    assert df.loc["05", "actual"] == "398|398|398"
    assert df.loc["06", "actual"] == "1194"
    assert df.loc["07", "actual"] == "historical_w250"
    assert df.loc["08", "actual"] == "ewma_d094"
    assert df.loc["09", "actual"] == "gb_G04"

    assert df.loc["10", "status"] == "PASS"
    assert df.loc["11", "status"] == "PASS"
    assert df.loc["12", "status"] == "PASS"
    assert df.loc["13", "status"] == "PASS"
    assert df.loc["14", "status"] == "PASS"
    assert df.loc["15", "status"] == "PASS"

    assert df.loc["16", "actual"] == (
        "ewma|gradient_boosting|historical_simulation"
    )


def test_pairwise_facts_exact():
    df = facts().set_index("fact_id")

    expected = {
        "17": (151, 247, -0.0001383332817084946),
        "18": (231, 167, -0.00021019030172751563),
        "19": (131, 267, 0.00007185702001901683),
    }

    for fid, (left, right, delta) in expected.items():
        parts = df.loc[fid, "actual"].split("|")

        assert int(parts[0]) == left
        assert int(parts[1]) == right
        assert abs(float(parts[2]) - delta) <= 1e-12
        assert df.loc[fid, "status"] == "PASS"


def test_manifest_schema_and_rows():
    df = manifest()

    assert list(df.columns) == MANIFEST_COLUMNS
    assert len(df) == 23
    assert df["artifact_id"].tolist() == [
        f"{i:02d}" for i in range(1, 24)
    ]
    assert df["artifact_id"].is_unique


def test_all_manifest_artifacts_ready():
    df = manifest()

    assert df["required"].eq("true").all()
    assert df["tracked"].eq("true").all()
    assert df["status"].eq("PASS").all()

    for path in df["path"]:
        p = ROOT / path
        assert p.is_file()
        assert p.stat().st_size > 0


def test_all_git_blob_shas_resolve():
    df = manifest()

    for row in df.itertuples(index=False):
        assert row.git_blob_sha

        result = subprocess.run(
            ["git", "rev-parse", f"HEAD:{row.path}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert result.stdout.strip() == row.git_blob_sha


def test_runbook_sections_and_guardrails():
    text = RUNBOOK.read_text(encoding="utf-8")
    lower = text.lower()

    sections = [
        "## Objective",
        "## Project in 30 Seconds",
        "## Two-Minute Executive Summary",
        "## Five-Minute Technical Walkthrough",
        "## Frozen Quantitative Facts",
        "## Model-by-Model Explanation",
        "## Pairwise Interpretation",
        "## Expected Defense Questions",
        "## Claims We Must Not Make",
        "## Reproducibility Demonstration",
        "## Failure-Recovery Plan",
        "## Final Submission Checklist",
        "## Final Handoff",
    ]

    for section in sections:
        assert section in text

    assert "there is no single overall winner" in lower
    assert "0.94 won validation" in lower
    assert "pristine untouched test set" in lower
    assert "lower average var" in lower

    forbidden = [
        "gradient boosting is the overall winner.",
        "ewma is the overall best model.",
        "historical simulation is the safest model.",
    ]

    for claim in forbidden:
        assert claim not in lower


def test_builder_has_no_forbidden_dependencies():
    source = BUILDER.read_text(encoding="utf-8").lower()

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
        assert token not in source