from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]

ACCEPTANCE = ROOT / "results" / "final_acceptance_matrix_a.csv"
SUMMARY = ROOT / "results" / "final_quantitative_summary_a.csv"
PREDICTIONS = ROOT / "results" / "final_predictions.csv"
COMPARISON = ROOT / "results" / "final_metric_comparison.csv"
PAIRWISE = ROOT / "results" / "final_pairwise_summary.csv"
TRACEABILITY = ROOT / "results" / "final_claim_traceability_a.csv"
VALIDATION = ROOT / "results" / "final_evidence_validation_a.csv"
FREEZE = ROOT / "configs" / "model_freeze.yaml"
DOC = ROOT / "docs" / "day26-final-presentation-evidence-a.md"
BUILDER = ROOT / "scripts" / "build_final_acceptance_a.py"

FIGURES = (
    ROOT / "results" / "figures" / "final_violation_rate_a.png",
    ROOT / "results" / "figures" / "final_pinball_loss_a.png",
    ROOT / "results" / "figures" / "final_average_var_a.png",
)

METHODS = (
    "historical_simulation",
    "ewma",
    "gradient_boosting",
)

CONFIGS = {
    "historical_simulation": "historical_w250",
    "ewma": "ewma_d094",
    "gradient_boosting": "gb_G04",
}

METRICS = {
    "historical_simulation": {
        "forecast_count": 398,
        "violation_count": 27,
        "violation_rate": 0.0678391959798995,
        "calibration_distance": 0.017839195979899497,
        "pinball_loss": 0.0018964642326659775,
        "average_var": 0.02133018927920153,
        "minimum_var": 0.0170883852278305,
        "maximum_var": 0.0270201977469214,
    },
    "ewma": {
        "forecast_count": 398,
        "violation_count": 22,
        "violation_rate": 0.05527638190954774,
        "calibration_distance": 0.005276381909547739,
        "pinball_loss": 0.0019683212526850112,
        "average_var": 0.02508404248953606,
        "minimum_var": 0.0140056564625239,
        "maximum_var": 0.0564466305776629,
    },
    "gradient_boosting": {
        "forecast_count": 398,
        "violation_count": 24,
        "violation_rate": 0.06030150753768844,
        "calibration_distance": 0.010301507537688437,
        "pinball_loss": 0.001758130950957487,
        "average_var": 0.023349057036283743,
        "minimum_var": 0.0137424730703311,
        "maximum_var": 0.0481390156210439,
    },
}

TOLERANCE = 1e-12


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), list(reader)


def as_bool(value: str) -> bool:
    value = value.strip().lower()
    assert value in {"true", "false"}
    return value == "true"


def test_acceptance_matrix_contract():
    columns, rows = read_csv(ACCEPTANCE)

    assert columns == [
        "acceptance_id",
        "category",
        "check",
        "expected",
        "actual",
        "tolerance",
        "status",
        "source",
    ]

    assert len(rows) == 30
    assert [r["acceptance_id"] for r in rows] == [
        f"{i:02d}" for i in range(1, 31)
    ]
    assert len({r["acceptance_id"] for r in rows}) == 30
    assert all(r["status"] == "PASS" for r in rows)
    assert all(r["tolerance"] == "1e-12" for r in rows)
    assert all(r["source"].strip() for r in rows)


def test_acceptance_category_distribution():
    _, rows = read_csv(ACCEPTANCE)

    assert Counter(r["category"] for r in rows) == Counter(
        {
            "evaluation_contract": 8,
            "risk_semantics": 2,
            "frozen_configuration": 5,
            "quantitative_evidence": 7,
            "pairwise_evidence": 3,
            "release_integrity": 2,
            "release_boundary": 3,
        }
    )


def test_prediction_panel_and_risk_semantics():
    columns, rows = read_csv(PREDICTIONS)

    assert columns == [
        "forecast_date",
        "target_date",
        "method",
        "actual_return",
        "quantile_return",
        "var",
        "violation",
        "runtime_seconds",
        "config_id",
    ]

    assert len(rows) == 1194

    assert Counter(r["method"] for r in rows) == Counter(
        {
            "historical_simulation": 398,
            "ewma": 398,
            "gradient_boosting": 398,
        }
    )

    dates = sorted({r["target_date"] for r in rows})

    assert len(dates) == 398
    assert dates[0] == "2024-12-18"
    assert dates[-1] == "2026-07-28"

    keys = [(r["method"], r["target_date"]) for r in rows]
    assert len(keys) == len(set(keys))

    by_date = {}

    for row in rows:
        by_date.setdefault(row["target_date"], []).append(row)

    for group in by_date.values():
        assert len(group) == 3
        assert {r["method"] for r in group} == set(METHODS)
        assert len({float(r["actual_return"]) for r in group}) == 1

    for row in rows:
        actual = float(row["actual_return"])
        quantile = float(row["quantile_return"])
        var = float(row["var"])

        assert as_bool(row["violation"]) == (actual < quantile)
        assert var >= 0.0
        assert var == pytest.approx(
            max(0.0, -quantile),
            abs=TOLERANCE,
        )


def test_model_freeze_contract():
    freeze = yaml.safe_load(FREEZE.read_text(encoding="utf-8"))

    assert freeze["freeze"]["status"] == "frozen"

    assert freeze["freeze"]["policy"] == {
        "new_algorithms_allowed": False,
        "new_features_allowed": False,
        "parameter_retuning_allowed": False,
        "evaluation_period_change_allowed": False,
        "canonical_prediction_rewrite_allowed": False,
    }

    evaluation = freeze["evaluation"]

    assert evaluation["confidence_level"] == pytest.approx(0.95)
    assert evaluation["alpha"] == pytest.approx(0.05)
    assert evaluation["forecast_horizon_trading_days"] == 1
    assert evaluation["start_date"] == "2024-12-18"
    assert evaluation["end_date"] == "2026-07-28"
    assert evaluation["target_date_count"] == 398
    assert evaluation["method_count"] == 3
    assert evaluation["prediction_row_count"] == 1194
    assert evaluation["used_for_parameter_selection"] is False
    assert evaluation["pristine_untouched_test_claim"] is False
    assert evaluation["violation_rule"] == "actual_return < quantile_return"
    assert evaluation["var_definition"] == "max(0, -quantile_return)"

    models = freeze["models"]

    hist = models["historical_simulation"]
    assert hist["config_id"] == "historical_w250"
    assert hist["window"] == 250
    assert hist["mode"] == "rolling"

    ewma = models["ewma"]
    assert ewma["config_id"] == "ewma_d094"
    assert ewma["decay"] == pytest.approx(0.94)
    assert ewma["mode"] == "expanding"
    assert ewma["validation_selected_alternative"]["decay"] == pytest.approx(0.90)
    assert (
        ewma["validation_selected_alternative"][
            "adopted_for_canonical_evaluation"
        ]
        is False
    )

    gb = models["gradient_boosting"]
    assert gb["config_id"] == "gb_G04"
    assert gb["n_estimators"] == 100
    assert gb["learning_rate"] == pytest.approx(0.03)
    assert gb["max_depth"] == 2
    assert gb["min_samples_leaf"] == 5
    assert gb["subsample"] == pytest.approx(1.0)
    assert gb["random_state"] == 42
    assert gb["features"] == [
        "return_lag_1",
        "return_lag_2",
        "return_lag_5",
        "rolling_vol_5",
        "rolling_vol_20",
        "rolling_vol_60",
        "drawdown",
    ]
    assert gb["market_features_used_by_canonical_g04"] is False


def test_frozen_metrics_and_leaders():
    _, rows = read_csv(COMPARISON)

    assert len(rows) == 3

    by_method = {r["method"]: r for r in rows}
    assert set(by_method) == set(METHODS)

    for method, expected in METRICS.items():
        row = by_method[method]

        assert row["config_id"] == CONFIGS[method]
        assert int(row["forecast_count"]) == expected["forecast_count"]
        assert int(row["violation_count"]) == expected["violation_count"]

        for column in (
            "violation_rate",
            "calibration_distance",
            "pinball_loss",
            "average_var",
            "minimum_var",
            "maximum_var",
        ):
            assert float(row[column]) == pytest.approx(
                expected[column],
                abs=TOLERANCE,
            )

    assert [
        r["method"] for r in rows
        if as_bool(r["calibration_leader"])
    ] == ["ewma"]

    assert [
        r["method"] for r in rows
        if as_bool(r["pinball_leader"])
    ] == ["gradient_boosting"]

    assert [
        r["method"] for r in rows
        if as_bool(r["lowest_average_var"])
    ] == ["historical_simulation"]


def test_pairwise_evidence():
    _, rows = read_csv(PAIRWISE)

    assert len(rows) == 3

    by_name = {r["comparison"]: r for r in rows}

    expected = {
        "gb_vs_historical": (151, 247, 0, -0.0001383332817084946),
        "gb_vs_ewma": (231, 167, 0, -0.00021019030172751563),
        "ewma_vs_historical": (131, 267, 0, 7.185702001901683e-05),
    }

    assert set(by_name) == set(expected)

    for comparison, values in expected.items():
        left, right, ties, delta = values
        row = by_name[comparison]

        assert int(row["observation_count"]) == 398
        assert int(row["left_better_count"]) == left
        assert int(row["right_better_count"]) == right
        assert int(row["tie_count"]) == ties
        assert float(row["mean_delta"]) == pytest.approx(
            delta,
            abs=TOLERANCE,
        )


def test_quantitative_summary_is_exact_projection():
    summary_columns, summary = read_csv(SUMMARY)
    source_columns, source = read_csv(COMPARISON)

    assert summary_columns == [
        "method",
        "config_id",
        "forecast_count",
        "violation_count",
        "violation_rate",
        "nominal_violation_rate",
        "calibration_distance",
        "pinball_loss",
        "average_var",
        "minimum_var",
        "maximum_var",
        "calibration_leader",
        "pinball_leader",
        "lowest_average_var",
        "source",
    ]

    assert len(summary) == 3
    assert [r["method"] for r in summary] == list(METHODS)

    source_by_method = {r["method"]: r for r in source}
    projected = [c for c in summary_columns if c != "source"]

    assert all(c in source_columns for c in projected)

    for row in summary:
        source_row = source_by_method[row["method"]]

        for column in projected:
            assert row[column] == source_row[column]

        assert row["source"] == "results/final_metric_comparison.csv"


def test_summary_values_and_leaders():
    _, rows = read_csv(SUMMARY)
    by_method = {r["method"]: r for r in rows}

    for method, expected in METRICS.items():
        row = by_method[method]

        assert row["config_id"] == CONFIGS[method]
        assert int(row["forecast_count"]) == expected["forecast_count"]
        assert int(row["violation_count"]) == expected["violation_count"]

        for column in (
            "violation_rate",
            "calibration_distance",
            "pinball_loss",
            "average_var",
            "minimum_var",
            "maximum_var",
        ):
            assert float(row[column]) == pytest.approx(
                expected[column],
                abs=TOLERANCE,
            )

    assert [r["method"] for r in rows if as_bool(r["calibration_leader"])] == [
        "ewma"
    ]
    assert [r["method"] for r in rows if as_bool(r["pinball_leader"])] == [
        "gradient_boosting"
    ]
    assert [r["method"] for r in rows if as_bool(r["lowest_average_var"])] == [
        "historical_simulation"
    ]


def test_prior_evidence_gates():
    _, traceability = read_csv(TRACEABILITY)
    _, validation = read_csv(VALIDATION)

    assert len(traceability) == 24
    assert len({r["claim_id"] for r in traceability}) == 24
    assert all(r["status"] == "PASS" for r in traceability)

    assert len(validation) == 26
    assert all(r["status"] == "PASS" for r in validation)

    signature = b"\x89PNG\r\n\x1a\n"

    for path in FIGURES:
        assert path.is_file()
        assert path.stat().st_size > 5000

        with path.open("rb") as handle:
            assert handle.read(8) == signature


def test_acceptance_release_boundaries():
    _, rows = read_csv(ACCEPTANCE)
    by_check = {r["check"]: r for r in rows}

    assert by_check["canonical_prediction_rows"]["actual"] == "1194"
    assert by_check["common_target_date_count"]["actual"] == "398"
    assert by_check["strict_violation_identity"]["actual"] == "true"
    assert by_check["var_identity_and_nonnegative"]["actual"] == "true"
    assert by_check["calibration_leader"]["actual"] == "ewma"
    assert by_check["pinball_leader"]["actual"] == "gradient_boosting"
    assert (
        by_check["lowest_average_var_method"]["actual"]
        == "historical_simulation"
    )
    assert by_check["interpretation_boundaries"]["actual"] == (
        "no_overall_winner=true|"
        "lower_var_not_superiority=true|"
        "non_pristine=true"
    )
    assert by_check["provenance_limitation_documented"]["actual"] == "true"
    assert by_check["prior_evidence_gates"]["actual"] == (
        "traceability=24/24 PASS|"
        "validation=26/26 PASS|"
        "figures=3/3 valid"
    )


def test_presentation_guardrails():
    text = DOC.read_text(encoding="utf-8")

    for heading in (
        "## Objective",
        "## Frozen Release Contract",
        "## Final Acceptance Status",
        "## Final Quantitative Summary",
        "## Calibration Interpretation",
        "## Quantile Accuracy Interpretation",
        "## Risk Magnitude Interpretation",
        "## Pairwise Evidence",
        "## Decision Guidance",
        "## Presentation Guidance",
        "## Reproducibility Evidence",
        "## Limitations",
        "## Provenance",
        "## Release Acceptance Interpretation",
        "## Final Presentation Interpretation",
    ):
        assert heading in text

    for sentence in (
        "Lower average VaR is not evidence of automatic model superiority.",
        "There is no single overall winner declared by the frozen final evaluation.",
        "It does not establish EWMA as the universally best model.",
        "The retained `0.94` configuration must not be described as the validation winner.",
        "It is not a new statistical experiment and does not reopen model selection.",
        "30 PASS checks",
        "0 FAIL checks",
        "pristine, never-inspected test set",
        "provenance incompleteness",
    ):
        assert sentence in text

    lowered = text.lower()

    for forbidden in (
        "Gradient Boosting is the overall winner",
        "EWMA is the overall winner",
        "Historical Simulation is the overall winner",
        "Gradient Boosting is the best model",
        "EWMA is the best model",
        "Historical Simulation is the best model",
        "0.94 won validation",
        "pristine untouched test set",
    ):
        assert forbidden.lower() not in lowered


def test_acceptance_builder_is_release_only():
    text = BUILDER.read_text(encoding="utf-8")
    lowered = text.lower()

    for forbidden in (
        "data/processed/portfolio_returns.csv",
        "run_final_walk_forward",
        "src.gb_market_features",
        "import requests",
        "from requests",
        "import urllib",
        "from urllib",
        "import socket",
        "from socket",
        "http://",
        "https://",
    ):
        assert forbidden not in lowered

    for required in (
        "final_predictions.csv",
        "final_metric_comparison.csv",
        "final_pairwise_summary.csv",
        "final_claim_traceability_a.csv",
        "final_evidence_validation_a.csv",
        "model_freeze.yaml",
        "day25-final-quantitative-story-a.md",
        "final_run_metadata.json",
        "final_acceptance_matrix_a.csv",
    ):
        assert required in text

    assert "TOLERANCE = 1e-12" in text