from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]

TRACEABILITY_PATH = ROOT / "results" / "final_claim_traceability_a.csv"
PREDICTIONS_PATH = ROOT / "results" / "final_predictions.csv"
COMPARISON_PATH = ROOT / "results" / "final_metric_comparison.csv"
PAIRWISE_PATH = ROOT / "results" / "final_pairwise_summary.csv"
VALIDATION_PATH = ROOT / "results" / "final_evidence_validation_a.csv"
FREEZE_PATH = ROOT / "configs" / "model_freeze.yaml"
STORY_PATH = ROOT / "docs" / "day25-final-quantitative-story-a.md"
BUILDER_PATH = ROOT / "scripts" / "build_final_traceability_a.py"

FIGURE_PATHS = [
    ROOT / "results" / "figures" / "final_violation_rate_a.png",
    ROOT / "results" / "figures" / "final_pinball_loss_a.png",
    ROOT / "results" / "figures" / "final_average_var_a.png",
]

EXPECTED_METHODS = {
    "historical_simulation",
    "ewma",
    "gradient_boosting",
}

EXPECTED_COLUMNS = [
    "claim_id",
    "category",
    "claim",
    "expected",
    "actual",
    "tolerance",
    "status",
    "source",
]

EXPECTED_CATEGORIES = {
    "evaluation_contract": 7,
    "frozen_configuration": 3,
    "aggregate_evidence": 5,
    "pairwise_evidence": 6,
    "release_boundary": 3,
}

EXPECTED_GB_FEATURES = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_5",
    "rolling_vol_5",
    "rolling_vol_20",
    "rolling_vol_60",
    "drawdown",
]

EXPECTED_PAIRWISE = {
    "gb_vs_historical": {
        "left_better_count": 151,
        "right_better_count": 247,
        "mean_delta": -0.0001383332817084946,
    },
    "gb_vs_ewma": {
        "left_better_count": 231,
        "right_better_count": 167,
        "mean_delta": -0.00021019030172751563,
    },
    "ewma_vs_historical": {
        "left_better_count": 131,
        "right_better_count": 267,
        "mean_delta": 0.00007185702001901683,
    },
}

TOLERANCE = 1e-12
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def traceability() -> pd.DataFrame:
    return pd.read_csv(
        TRACEABILITY_PATH,
        dtype={
            "claim_id": "string",
            "expected": "string",
            "actual": "string",
            "tolerance": "string",
            "status": "string",
            "source": "string",
        },
    )


@pytest.fixture(scope="module")
def predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_PATH)


@pytest.fixture(scope="module")
def comparison() -> pd.DataFrame:
    return pd.read_csv(COMPARISON_PATH)


@pytest.fixture(scope="module")
def pairwise() -> pd.DataFrame:
    return pd.read_csv(PAIRWISE_PATH)


@pytest.fixture(scope="module")
def freeze() -> dict:
    with FREEZE_PATH.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    assert isinstance(loaded, dict)

    return loaded


def test_traceability_schema_status_and_categories(
    traceability: pd.DataFrame,
) -> None:
    assert list(traceability.columns) == EXPECTED_COLUMNS
    assert len(traceability) == 24

    expected_ids = [f"{value:02d}" for value in range(1, 25)]

    assert traceability["claim_id"].tolist() == expected_ids
    assert traceability["claim_id"].is_unique
    assert traceability["status"].eq("PASS").all()
    assert traceability["tolerance"].eq("1e-12").all()

    category_counts = traceability["category"].value_counts().to_dict()

    assert category_counts == EXPECTED_CATEGORIES


def test_traceability_values_are_independently_consistent(
    traceability: pd.DataFrame,
) -> None:
    numeric_claim_ids = {"17", "19", "21"}

    for row in traceability.itertuples(index=False):
        if row.claim_id in numeric_claim_ids:
            expected = float(row.expected)
            actual = float(row.actual)

            assert actual == pytest.approx(
                expected,
                abs=TOLERANCE,
            )
        else:
            assert row.actual == row.expected

    indexed = traceability.set_index("claim_id")

    assert indexed.loc["11", "actual"] == "ewma"
    assert indexed.loc["12", "actual"] == "gradient_boosting"
    assert indexed.loc["13", "actual"] == "historical_simulation"
    assert indexed.loc["14", "actual"] == "true"
    assert indexed.loc["15", "actual"] == "true"
    assert indexed.loc["22", "actual"] == "true"
    assert indexed.loc["23", "actual"] == "true"
    assert indexed.loc["24", "actual"] == (
        "validation=26/26 PASS;figures=3/3"
    )


def test_traceability_sources_resolve(
    traceability: pd.DataFrame,
) -> None:
    for source_value in traceability["source"]:
        sources = [
            value.strip()
            for value in source_value.split(";")
            if value.strip()
        ]

        assert sources

        for source in sources:
            if "*" in source:
                assert list(ROOT.glob(source))
            else:
                assert (ROOT / source).is_file()


def test_prediction_contract(
    predictions: pd.DataFrame,
) -> None:
    assert list(predictions.columns) == [
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

    assert len(predictions) == 1194
    assert set(predictions["method"]) == EXPECTED_METHODS

    counts = predictions.groupby("method").size().to_dict()

    assert counts == {
        "historical_simulation": 398,
        "ewma": 398,
        "gradient_boosting": 398,
    }

    assert not predictions.duplicated(
        subset=["method", "target_date"]
    ).any()

    target_dates = sorted(predictions["target_date"].unique())

    assert len(target_dates) == 398
    assert target_dates[0] == "2024-12-18"
    assert target_dates[-1] == "2026-07-28"


def test_prediction_alignment_and_quant_definitions(
    predictions: pd.DataFrame,
) -> None:
    actual_counts = (
        predictions.groupby("target_date")["actual_return"]
        .nunique(dropna=False)
    )

    method_counts = (
        predictions.groupby("target_date")["method"]
        .nunique(dropna=False)
    )

    assert len(actual_counts) == 398
    assert actual_counts.eq(1).all()
    assert method_counts.eq(3).all()

    actual = pd.to_numeric(
        predictions["actual_return"],
        errors="raise",
    )

    quantile = pd.to_numeric(
        predictions["quantile_return"],
        errors="raise",
    )

    reported_var = pd.to_numeric(
        predictions["var"],
        errors="raise",
    )

    reported_violation = (
        predictions["violation"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

    assert reported_violation.notna().all()

    expected_violation = actual.lt(quantile)
    expected_var = (-quantile).clip(lower=0.0)

    assert reported_violation.equals(expected_violation)
    assert reported_var.ge(0.0).all()
    assert (reported_var - expected_var).abs().le(TOLERANCE).all()


def test_frozen_configuration_contract(
    predictions: pd.DataFrame,
    freeze: dict,
) -> None:
    evaluation = freeze["evaluation"]
    models = freeze["models"]

    assert evaluation["alpha"] == pytest.approx(0.05)
    assert evaluation["confidence_level"] == pytest.approx(0.95)
    assert evaluation["forecast_horizon_trading_days"] == 1
    assert evaluation["target_date_count"] == 398
    assert evaluation["prediction_row_count"] == 1194
    assert evaluation["pristine_untouched_test_claim"] is False
    assert evaluation["violation_rule"] == (
        "actual_return < quantile_return"
    )
    assert evaluation["var_definition"] == (
        "max(0, -quantile_return)"
    )

    historical = models["historical_simulation"]
    ewma = models["ewma"]
    gb = models["gradient_boosting"]

    assert historical["config_id"] == "historical_w250"
    assert historical["window"] == 250

    assert ewma["config_id"] == "ewma_d094"
    assert ewma["decay"] == pytest.approx(0.94)
    assert ewma["validation_selected_alternative"]["decay"] == pytest.approx(
        0.90
    )
    assert (
        ewma["validation_selected_alternative"][
            "adopted_for_canonical_evaluation"
        ]
        is False
    )

    assert gb["config_id"] == "gb_G04"
    assert gb["feature_count"] == 7
    assert gb["features"] == EXPECTED_GB_FEATURES
    assert gb["market_features_used_by_canonical_g04"] is False

    configs = (
        predictions.groupby("method")["config_id"]
        .unique()
        .to_dict()
    )

    assert configs["historical_simulation"].tolist() == [
        "historical_w250"
    ]
    assert configs["ewma"].tolist() == ["ewma_d094"]
    assert configs["gradient_boosting"].tolist() == ["gb_G04"]


def test_criterion_specific_leaders(
    comparison: pd.DataFrame,
) -> None:
    assert len(comparison) == 3
    assert set(comparison["method"]) == EXPECTED_METHODS

    calibration_leader = comparison.loc[
        comparison["calibration_distance"].idxmin(),
        "method",
    ]

    pinball_leader = comparison.loc[
        comparison["pinball_loss"].idxmin(),
        "method",
    ]

    lowest_average_var = comparison.loc[
        comparison["average_var"].idxmin(),
        "method",
    ]

    assert calibration_leader == "ewma"
    assert pinball_leader == "gradient_boosting"
    assert lowest_average_var == "historical_simulation"


def test_pairwise_contract(
    pairwise: pd.DataFrame,
) -> None:
    assert len(pairwise) == 3
    assert set(pairwise["comparison"]) == set(EXPECTED_PAIRWISE)

    indexed = pairwise.set_index("comparison")

    for comparison_name, expected in EXPECTED_PAIRWISE.items():
        row = indexed.loc[comparison_name]

        assert int(row["observation_count"]) == 398
        assert int(row["left_better_count"]) == (
            expected["left_better_count"]
        )
        assert int(row["right_better_count"]) == (
            expected["right_better_count"]
        )
        assert int(row["tie_count"]) == 0

        assert float(row["mean_delta"]) == pytest.approx(
            expected["mean_delta"],
            abs=TOLERANCE,
        )


def test_day24_validation_and_figures_remain_valid() -> None:
    validation = pd.read_csv(VALIDATION_PATH)

    assert len(validation) == 26
    assert validation["check"].is_unique
    assert validation["status"].eq("PASS").all()

    assert len(FIGURE_PATHS) == 3

    for path in FIGURE_PATHS:
        assert path.is_file()
        assert path.stat().st_size >= 5000

        with path.open("rb") as handle:
            assert handle.read(8) == PNG_SIGNATURE


def test_final_story_preserves_interpretation_boundaries() -> None:
    story = STORY_PATH.read_text(encoding="utf-8")
    lower_story = story.lower()

    assert "there is no single overall winner" in lower_story
    assert (
        "lower average var must not be interpreted as automatic "
        "model superiority"
        in lower_story
    )
    assert "pristine, never-inspected test set" in lower_story
    assert "provenance incompleteness" in lower_story
    assert "current tested environment" in lower_story

    forbidden_claims = [
        "ewma is the best model",
        "gradient boosting is the best model",
        "gradient boosting is the overall winner",
        "historical simulation is the best model",
        "historical simulation is the safest model",
        "0.94 won validation",
    ]

    for claim in forbidden_claims:
        assert claim not in lower_story


def test_traceability_builder_scope_is_release_safe() -> None:
    source = BUILDER_PATH.read_text(encoding="utf-8")

    forbidden_references = [
        "data/processed/portfolio_returns.csv",
        "run_final_walk_forward",
        "src.gb_market_features",
        "requests",
        "urllib",
        "socket",
        "http://",
        "https://",
    ]

    for reference in forbidden_references:
        assert reference not in source