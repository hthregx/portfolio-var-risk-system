from pathlib import Path
import json, subprocess, sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_release_reproducibility_b.py"
CSV = ROOT / "results/release_reproducibility_validation_b.csv"
MANIFEST = ROOT / "results/release_manifest_b.json"


def run():
    return subprocess.run(
        [sys.executable, str(SCRIPT)], cwd=ROOT,
        capture_output=True, text=True
    )


def data():
    r = run()
    assert r.returncode == 0, r.stdout + r.stderr
    return pd.read_csv(CSV)


def test_validator_passes():
    r = run()
    assert r.returncode == 0, r.stdout + r.stderr
    for text in [
        "B1_FROZEN_MODEL_SPECIFICATION_PASS",
        "B2_CROSS_LAYER_CONTRACT_PASS",
        "B3_ARTIFACT_PROVENANCE_PASS",
    ]:
        assert text in r.stdout


def test_all_checks_pass():
    df = data()
    assert set(df["group"]) == {"B1", "B2"}
    assert set(df["status"]) == {"PASS"}


def test_b2_cross_layer_coverage():
    names = set(data().query("group == 'B2'")["check"])
    required = {
        "Target freeze-final", "Target final-metadata",
        "Alpha freeze-final", "Alpha final-metadata",
        "Horizon freeze-final", "Horizon final-metadata",
        "Historical window freeze-final",
        "EWMA decay freeze-final",
        "GB features freeze-final", "GB features final-metadata",
        "Prediction config IDs", "Metric config IDs",
    }
    assert required <= names


def test_b2_source_and_no_leakage():
    names = set(data().query("group == 'B2'")["check"])
    assert {
        "Runner uses configured GB features",
        "Runner trains GB before target only",
        "Runner excludes market feature builder",
        "No duplicate method-date keys",
        "Common actual returns",
    } <= names


def test_b2_prediction_semantics():
    names = set(data().query("group == 'B2'")["check"])
    assert {
        "Prediction violation semantics",
        "Prediction VaR semantics",
        "Prediction schema",
        "Three methods per target",
    } <= names


def test_b2_metrics_consistency():
    names = set(data().query("group == 'B2'")["check"])
    assert {
        "Metric schema",
        "Metric forecast counts",
        "Metric violation counts",
        "Metric violation rates",
        "Metric config IDs",
    } <= names


def test_manifest():
    data()
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths = {x["path"]: x for x in m["artifacts"]}

    required = {
        "configs/model_freeze.yaml",
        "configs/final_evaluation.yaml",
        "results/final_predictions.csv",
        "results/final_metrics.csv",
        "results/final_metric_comparison.csv",
        "results/final_pairwise_diagnostics.csv",
        "results/final_pairwise_summary.csv",
        "results/model_freeze_validation_a.csv",
    }

    assert required <= set(paths)
    assert all(len(paths[p]["sha256"]) == 64 for p in required)
    assert m["provenance_limitation"] == (
        "final metadata runtime HEAD predates the later Day21 source commit, "
        "while canonical prediction artifact content remained unchanged."
    )