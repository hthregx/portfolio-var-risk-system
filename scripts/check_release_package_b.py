from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = ROOT / "results/final_predictions.csv"
METRICS_PATH = ROOT / "results/final_metrics.csv"
METADATA_PATH = ROOT / "results/final_run_metadata.json"
MODEL_CARD_PATH = (
    ROOT / "docs/model-cards/gradient-boosting-g04.md"
)
RELEASE_NOTES_PATH = ROOT / "docs/releases/v0.6.0.md"

REQUIRED_FILES = [
    "configs/final_evaluation.yaml",
    "results/final_predictions.csv",
    "results/final_metrics.csv",
    "results/final_run_metadata.json",
    "results/final_metric_comparison.csv",
    "results/final_pairwise_diagnostics.csv",
    "results/final_pairwise_summary.csv",
    "notebooks/08_final_results_analysis.ipynb",
    "docs/final-results-summary-a.md",
    "docs/model-cards/gradient-boosting-g04.md",
    "docs/releases/v0.6.0.md",
    "results/gb_freeze_audit_b.csv",
    "results/release_artifact_audit_b.csv",
    "docs/releases/v0.6.0-checklist.md",
]

OPTIONAL_DAY22_B = [
    "results/final_exception_analysis.csv",
    "results/final_exception_cases.csv",
    "results/final_regime_analysis.csv",
    "docs/final-results-regime-b.md",
]

EXPECTED_METHODS = {
    "historical_simulation",
    "ewma",
    "gradient_boosting",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_required_files():
    missing = [
        relative
        for relative in REQUIRED_FILES
        if not (ROOT / relative).is_file()
    ]

    require(
        not missing,
        f"Missing release assets: {missing}",
    )

    return True


def check_canonical_artifacts():
    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["target_date"],
    )
    metrics = pd.read_csv(METRICS_PATH)

    metadata = json.loads(
        METADATA_PATH.read_text()
    )

    require(
        len(predictions) == 1194,
        "final_predictions.csv must contain 1194 rows.",
    )

    require(
        predictions["target_date"].nunique() == 398,
        "Expected 398 unique target dates.",
    )

    require(
        set(predictions["method"]) == EXPECTED_METHODS,
        "Unexpected prediction methods.",
    )

    counts = (
        predictions.groupby("method")
        .size()
        .to_dict()
    )

    require(
        all(
            counts.get(method) == 398
            for method in EXPECTED_METHODS
        ),
        "Each method must contain 398 forecasts.",
    )

    require(
        not predictions.empty,
        "final_predictions.csv is empty.",
    )

    require(
        not metrics.empty,
        "final_metrics.csv is empty.",
    )

    require(
        isinstance(metadata, dict) and metadata,
        "final_run_metadata.json is empty or invalid.",
    )

    return True


def check_release_documentation():
    model_card = MODEL_CARD_PATH.read_text(
        encoding="utf-8"
    )
    release_notes = RELEASE_NOTES_PATH.read_text(
        encoding="utf-8"
    )

    model_card_required = (
        "G04",
        "Limitations",
        "Freeze Status",
        "No direct target leakage",
    )

    for text in model_card_required:
        require(
            text in model_card,
            f"Model card missing required text: {text}",
        )

    release_required = (
        "Known Limitations",
        "G01-G07",
        "pristine untouched test set",
        "universal overall winner",
        "Freeze Policy",
    )

    for text in release_required:
        require(
            text in release_notes,
            f"Release notes missing required text: {text}",
        )

    return True


def check_day22_b_assets():
    return {
        relative: (ROOT / relative).is_file()
        for relative in OPTIONAL_DAY22_B
    }


def main():
    check_required_files()
    print("required release assets: PASS")

    check_canonical_artifacts()
    print("canonical release artifacts: PASS")

    check_release_documentation()
    print("release documentation: PASS")

    day22 = check_day22_b_assets()

    for path, present in day22.items():
        print(
            f"{path}: "
            f"{'PRESENT' if present else 'OPTIONAL/NOT PRESENT'}"
        )

    print("RELEASE_PACKAGE_CHECK_PASS")


if __name__ == "__main__":
    main()