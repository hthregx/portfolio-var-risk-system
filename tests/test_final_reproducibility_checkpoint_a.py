from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = ROOT / "results" / "final_reproducibility_checkpoint_a.csv"
PREDICTIONS = ROOT / "results" / "final_predictions.csv"
COMPARISON = ROOT / "results" / "final_metric_comparison.csv"
PAIRWISE = ROOT / "results" / "final_pairwise_summary.csv"
FREEZE = ROOT / "configs" / "model_freeze.yaml"
VALIDATION = ROOT / "results" / "final_evidence_validation_a.csv"
TRACEABILITY = ROOT / "results" / "final_claim_traceability_a.csv"
ACCEPTANCE = ROOT / "results" / "final_acceptance_matrix_a.csv"
SUMMARY = ROOT / "results" / "final_quantitative_summary_a.csv"
BUILDER = ROOT / "scripts" / "build_final_reproducibility_checkpoint_a.py"
HANDOFF = ROOT / "docs" / "day27-final-release-handoff-a.md"

TOL = 1e-12
ALPHA = 0.05

METHODS = (
    "historical_simulation",
    "ewma",
    "gradient_boosting",
)

EXPECTED_PREDICTION_COLUMNS = (
    "forecast_date",
    "target_date",
    "method",
    "actual_return",
    "quantile_return",
    "var",
    "violation",
    "runtime_seconds",
    "config_id",
)

EXPECTED_GB_FEATURES = (
    "return_lag_1",
    "return_lag_2",
    "return_lag_5",
    "rolling_vol_5",
    "rolling_vol_20",
    "rolling_vol_60",
    "drawdown",
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), list(reader)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    assert normalized in {"true", "false"}
    return normalized == "true"


def pinball(actual: float, quantile: float) -> float:
    error = actual - quantile
    return ALPHA * error if error >= 0 else (1.0 - ALPHA) * (-error)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def recompute_metrics() -> dict[str, dict[str, float | int]]:
    _, predictions = read_csv(PREDICTIONS)
    output: dict[str, dict[str, float | int]] = {}

    for method in METHODS:
        rows = [row for row in predictions if row["method"] == method]

        actual = [float(row["actual_return"]) for row in rows]
        quantile = [float(row["quantile_return"]) for row in rows]
        var = [float(row["var"]) for row in rows]

        violations = sum(
            observed < forecast
            for observed, forecast in zip(actual, quantile)
        )

        losses = [
            pinball(observed, forecast)
            for observed, forecast in zip(actual, quantile)
        ]

        output[method] = {
            "forecast_count": len(rows),
            "violation_count": violations,
            "violation_rate": violations / len(rows),
            "pinball_loss": sum(losses) / len(losses),
            "average_var": sum(var) / len(var),
            "minimum_var": min(var),
            "maximum_var": max(var),
        }

    return output


def test_checkpoint_contract() -> None:
    columns, rows = read_csv(CHECKPOINT)

    assert columns == [
        "check_id",
        "category",
        "check",
        "expected",
        "actual",
        "tolerance",
        "status",
        "source",
    ]

    assert len(rows) == 28
    assert [row["check_id"] for row in rows] == [
        f"R{number:02d}" for number in range(1, 29)
    ]
    assert len({row["check_id"] for row in rows}) == 28
    assert all(row["status"] == "PASS" for row in rows)
    assert all(row["tolerance"] == "1e-12" for row in rows)


def test_checkpoint_category_distribution() -> None:
    _, rows = read_csv(CHECKPOINT)

    actual = Counter(row["category"] for row in rows)

    assert actual == {
        "evaluation_contract": 6,
        "freeze_contract": 1,
        "frozen_configuration": 4,
        "frozen_inputs": 1,
        "prior_evidence_gates": 3,
        "recomputed_pairwise_evidence": 3,
        "recomputed_quantitative_evidence": 7,
        "release_boundary": 1,
        "risk_semantics": 2,
    }


def test_builder_is_release_only() -> None:
    text = BUILDER.read_text(encoding="utf-8")

    forbidden = (
        "data/processed",
        "data/raw",
        "run_final_walk_forward",
        "build_return_features",
        "sklearn",
        "pandas",
        "numpy",
        "requests",
        "urllib",
        "http://",
        "https://",
    )

    assert all(token not in text for token in forbidden)
    assert 'final_predictions.csv' in text
    assert 'final_metric_comparison.csv' in text
    assert 'final_pairwise_summary.csv' in text
    assert 'final_reproducibility_checkpoint_a.csv' in text


def test_frozen_canonical_artifact_hashes() -> None:
    with FREEZE.open("r", encoding="utf-8") as handle:
        freeze = yaml.safe_load(handle)

    artifacts = freeze["canonical_artifacts"]

    assert len(artifacts) == 7

    for entry in artifacts.values():
        path = ROOT / entry["path"]
        assert path.is_file()
        assert sha256(path) == entry["sha256"]


def test_prediction_panel_contract() -> None:
    columns, rows = read_csv(PREDICTIONS)

    assert tuple(columns) == EXPECTED_PREDICTION_COLUMNS
    assert len(rows) == 1194

    method_counts = Counter(row["method"] for row in rows)

    assert method_counts == {
        "historical_simulation": 398,
        "ewma": 398,
        "gradient_boosting": 398,
    }

    target_dates = sorted({row["target_date"] for row in rows})

    assert len(target_dates) == 398
    assert target_dates[0] == "2024-12-18"
    assert target_dates[-1] == "2026-07-28"

    keys = [(row["method"], row["target_date"]) for row in rows]
    assert len(keys) == len(set(keys))

    by_date: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        by_date[row["target_date"]].append(row)

    for group in by_date.values():
        assert len(group) == 3
        assert {row["method"] for row in group} == set(METHODS)
        assert len({row["actual_return"] for row in group}) == 1


def test_risk_semantics() -> None:
    _, rows = read_csv(PREDICTIONS)

    for row in rows:
        actual = float(row["actual_return"])
        quantile = float(row["quantile_return"])
        var = float(row["var"])

        assert parse_bool(row["violation"]) == (actual < quantile)
        assert var >= 0.0
        assert abs(var - max(0.0, -quantile)) <= TOL


def test_recomputed_quantitative_metrics() -> None:
    metrics = recompute_metrics()
    _, comparison = read_csv(COMPARISON)

    source = {row["method"]: row for row in comparison}

    numeric_columns = (
        "violation_rate",
        "pinball_loss",
        "average_var",
        "minimum_var",
        "maximum_var",
    )

    for method in METHODS:
        assert metrics[method]["forecast_count"] == int(
            source[method]["forecast_count"]
        )
        assert metrics[method]["violation_count"] == int(
            source[method]["violation_count"]
        )

        for column in numeric_columns:
            assert abs(
                float(metrics[method][column])
                - float(source[method][column])
            ) <= TOL


def test_criterion_specific_leaders() -> None:
    metrics = recompute_metrics()
    _, comparison = read_csv(COMPARISON)

    calibration = min(
        METHODS,
        key=lambda method: abs(
            float(metrics[method]["violation_rate"]) - ALPHA
        ),
    )

    pinball_leader = min(
        METHODS,
        key=lambda method: float(metrics[method]["pinball_loss"]),
    )

    lowest_var = min(
        METHODS,
        key=lambda method: float(metrics[method]["average_var"]),
    )

    expected_calibration = next(
        row["method"]
        for row in comparison
        if parse_bool(row["calibration_leader"])
    )

    expected_pinball = next(
        row["method"]
        for row in comparison
        if parse_bool(row["pinball_leader"])
    )

    expected_lowest_var = next(
        row["method"]
        for row in comparison
        if parse_bool(row["lowest_average_var"])
    )

    assert calibration == expected_calibration == "ewma"
    assert pinball_leader == expected_pinball == "gradient_boosting"
    assert lowest_var == expected_lowest_var == "historical_simulation"


def test_pairwise_recomputation() -> None:
    _, predictions = read_csv(PREDICTIONS)
    _, pairwise = read_csv(PAIRWISE)

    source = {row["comparison"]: row for row in pairwise}

    loss_panel: dict[str, dict[str, float]] = defaultdict(dict)

    for row in predictions:
        loss_panel[row["target_date"]][row["method"]] = pinball(
            float(row["actual_return"]),
            float(row["quantile_return"]),
        )

    specs = (
        ("gb_vs_historical", "gradient_boosting", "historical_simulation"),
        ("gb_vs_ewma", "gradient_boosting", "ewma"),
        ("ewma_vs_historical", "ewma", "historical_simulation"),
    )

    for name, left, right in specs:
        deltas = [
            panel[left] - panel[right]
            for _, panel in sorted(loss_panel.items())
        ]

        assert len(deltas) == int(source[name]["observation_count"])
        assert sum(delta < 0 for delta in deltas) == int(
            source[name]["left_better_count"]
        )
        assert sum(delta > 0 for delta in deltas) == int(
            source[name]["right_better_count"]
        )
        assert sum(delta == 0 for delta in deltas) == int(
            source[name]["tie_count"]
        )
        assert abs(
            sum(deltas) / len(deltas)
            - float(source[name]["mean_delta"])
        ) <= TOL


def test_model_freeze_contract() -> None:
    with FREEZE.open("r", encoding="utf-8") as handle:
        freeze = yaml.safe_load(handle)

    policy = freeze["freeze"]["policy"]
    evaluation = freeze["evaluation"]
    models = freeze["models"]

    assert freeze["freeze"]["status"] == "frozen"
    assert all(value is False for value in policy.values())

    assert evaluation["confidence_level"] == 0.95
    assert evaluation["alpha"] == 0.05
    assert evaluation["forecast_horizon_trading_days"] == 1
    assert evaluation["target_date_count"] == 398
    assert evaluation["prediction_row_count"] == 1194
    assert evaluation["used_for_parameter_selection"] is False
    assert evaluation["pristine_untouched_test_claim"] is False

    historical = models["historical_simulation"]
    assert historical["config_id"] == "historical_w250"
    assert historical["window"] == 250
    assert historical["mode"] == "rolling"

    ewma = models["ewma"]
    assert ewma["config_id"] == "ewma_d094"
    assert ewma["decay"] == 0.94
    assert ewma["validation_selected_alternative"]["decay"] == 0.90
    assert (
        ewma["validation_selected_alternative"][
            "adopted_for_canonical_evaluation"
        ]
        is False
    )

    gb = models["gradient_boosting"]
    assert gb["config_id"] == "gb_G04"
    assert tuple(gb["features"]) == EXPECTED_GB_FEATURES
    assert gb["market_features_used_by_canonical_g04"] is False


def test_prior_evidence_gates() -> None:
    _, validation = read_csv(VALIDATION)
    _, traceability = read_csv(TRACEABILITY)
    _, acceptance = read_csv(ACCEPTANCE)
    _, summary = read_csv(SUMMARY)
    _, checkpoint = read_csv(CHECKPOINT)

    assert len(validation) == 26
    assert all(row["status"] == "PASS" for row in validation)

    assert len(traceability) == 24
    assert all(row["status"] == "PASS" for row in traceability)

    assert len(acceptance) == 30
    assert all(row["status"] == "PASS" for row in acceptance)

    assert len(summary) == 3

    assert len(checkpoint) == 28
    assert all(row["status"] == "PASS" for row in checkpoint)


def test_release_handoff_guardrails() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    lower = text.lower()

    required = (
        "28 / 28 PASS",
        "EWMA is closest to the nominal 5% violation rate.",
        "Gradient Boosting has the lowest mean pinball loss.",
        "Historical Simulation has the lowest average reported VaR.",
        "Lower average VaR is not evidence of automatic model superiority.",
        "The retained `0.94` configuration must not be described as the validation winner.",
        "current canonical modeling snapshot ends on `2026-07-28`",
        "final freshness snapshot after the `2026-08-28` market close",
        "Day 27 does not perform that final data refresh.",
        "pristine, never-inspected test set",
        "provenance completeness limitation",
        "No universal model winner is declared.",
    )

    assert all(item in text for item in required)

    forbidden = (
        "ewma is the overall winner",
        "gradient boosting is the overall winner",
        "historical simulation is the overall winner",
        "ewma is the best model",
        "gradient boosting is the best model",
        "historical simulation is the best model",
        "0.94 won validation",
        "pristine untouched test set",
    )

    assert all(item not in lower for item in forbidden)