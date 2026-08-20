from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_final_walk_forward import (
    FINAL_PREDICTION_COLUMNS,
    METHOD_EWMA,
    METHOD_GRADIENT_BOOSTING,
    METHOD_HISTORICAL,
    build_config_ids,
    build_final_predictions,
    load_config,
    write_final_artifacts,
)
from src.evaluation.final_metrics import (
    METRIC_COLUMNS,
    compute_final_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "final_evaluation.yaml"


def _synthetic_predictions() -> pd.DataFrame:
    records = []

    values = {
        METHOD_HISTORICAL: (-0.03, 0.03),
        METHOD_EWMA: (-0.025, 0.025),
        METHOD_GRADIENT_BOOSTING: (-0.02, 0.02),
    }

    for method, (quantile, var_value) in values.items():
        records.extend(
            [
                {
                    "window_end_date": pd.Timestamp("2026-01-01"),
                    "forecast_date": pd.Timestamp("2026-01-01"),
                    "target_date": pd.Timestamp("2026-01-02"),
                    "actual_return": -0.04,
                    "quantile_return": quantile,
                    "var": var_value,
                    "violation": -0.04 < quantile,
                    "method": method,
                    "observations": 100,
                    "evaluation_mode": (
                        "rolling"
                        if method == METHOD_HISTORICAL
                        else "expanding"
                    ),
                },
                {
                    "window_end_date": pd.Timestamp("2026-01-02"),
                    "forecast_date": pd.Timestamp("2026-01-02"),
                    "target_date": pd.Timestamp("2026-01-05"),
                    "actual_return": 0.01,
                    "quantile_return": quantile,
                    "var": var_value,
                    "violation": False,
                    "method": method,
                    "observations": 101,
                    "evaluation_mode": (
                        "rolling"
                        if method == METHOD_HISTORICAL
                        else "expanding"
                    ),
                },
            ]
        )

    return pd.DataFrame(records)


def test_build_config_ids_matches_day21_contract():
    config = load_config(CONFIG_PATH)

    assert build_config_ids(config) == {
        METHOD_HISTORICAL: "historical_w250",
        METHOD_EWMA: "ewma_d094",
        METHOD_GRADIENT_BOOSTING: "gb_G04",
    }


def test_final_prediction_contract():
    predictions = _synthetic_predictions()

    runtimes = {
        METHOD_HISTORICAL: 0.1,
        METHOD_EWMA: 0.2,
        METHOD_GRADIENT_BOOSTING: 1.5,
    }

    config_ids = {
        METHOD_HISTORICAL: "historical_w250",
        METHOD_EWMA: "ewma_d094",
        METHOD_GRADIENT_BOOSTING: "gb_G04",
    }

    final_predictions = build_final_predictions(
        predictions,
        runtimes=runtimes,
        config_ids=config_ids,
    )

    assert list(final_predictions.columns) == FINAL_PREDICTION_COLUMNS
    assert len(final_predictions) == 6

    assert not final_predictions.duplicated(
        ["method", "target_date"]
    ).any()

    assert (
        final_predictions["forecast_date"]
        < final_predictions["target_date"]
    ).all()

    expected_var = np.maximum(
        0.0,
        -final_predictions["quantile_return"].to_numpy(),
    )

    np.testing.assert_allclose(
        final_predictions["var"].to_numpy(),
        expected_var,
        rtol=0.0,
        atol=1e-15,
    )


def test_final_prediction_contract_rejects_bad_violation():
    predictions = _synthetic_predictions()

    predictions.loc[0, "violation"] = False

    with pytest.raises(
        ValueError,
        match="strict violation rule",
    ):
        build_final_predictions(
            predictions,
            runtimes={
                METHOD_HISTORICAL: 0.1,
                METHOD_EWMA: 0.2,
                METHOD_GRADIENT_BOOSTING: 1.5,
            },
            config_ids={
                METHOD_HISTORICAL: "historical_w250",
                METHOD_EWMA: "ewma_d094",
                METHOD_GRADIENT_BOOSTING: "gb_G04",
            },
        )


def test_metrics_and_export_roundtrip(tmp_path):
    predictions = _synthetic_predictions()

    runtimes = {
        METHOD_HISTORICAL: 0.1,
        METHOD_EWMA: 0.2,
        METHOD_GRADIENT_BOOSTING: 1.5,
    }

    config_ids = {
        METHOD_HISTORICAL: "historical_w250",
        METHOD_EWMA: "ewma_d094",
        METHOD_GRADIENT_BOOSTING: "gb_G04",
    }

    final_predictions = build_final_predictions(
        predictions,
        runtimes=runtimes,
        config_ids=config_ids,
    )

    metrics = compute_final_metrics(
        predictions,
        runtimes=runtimes,
        config_ids=config_ids,
        alpha=0.05,
    )

    metadata = {
        "contract": {
            "alpha": 0.05,
        },
        "artifacts": {
            "prediction_rows": len(final_predictions),
            "metric_rows": len(metrics),
        },
    }

    paths = write_final_artifacts(
        predictions=final_predictions,
        metrics=metrics,
        metadata=metadata,
        output_dir=tmp_path,
    )

    prediction_roundtrip = pd.read_csv(
        paths["predictions"]
    )

    metric_roundtrip = pd.read_csv(
        paths["metrics"]
    )

    metadata_roundtrip = json.loads(
        paths["metadata"].read_text(
            encoding="utf-8"
        )
    )

    assert list(prediction_roundtrip.columns) == FINAL_PREDICTION_COLUMNS
    assert list(metric_roundtrip.columns) == METRIC_COLUMNS

    assert len(prediction_roundtrip) == 6
    assert len(metric_roundtrip) == 3

    assert metadata_roundtrip["contract"]["alpha"] == 0.05
    assert metadata_roundtrip["artifacts"]["prediction_rows"] == 6
    assert metadata_roundtrip["artifacts"]["metric_rows"] == 3


def test_canonical_day21_artifacts_match_locked_contract():
    prediction_path = (
        REPO_ROOT
        / "results"
        / "final_predictions.csv"
    )

    metric_path = (
        REPO_ROOT
        / "results"
        / "final_metrics.csv"
    )

    metadata_path = (
        REPO_ROOT
        / "results"
        / "final_run_metadata.json"
    )

    assert prediction_path.is_file()
    assert metric_path.is_file()
    assert metadata_path.is_file()

    predictions = pd.read_csv(
        prediction_path,
        parse_dates=[
            "forecast_date",
            "target_date",
        ],
    )

    metrics = pd.read_csv(
        metric_path,
        parse_dates=[
            "test_start",
            "test_end",
        ],
    )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    assert list(predictions.columns) == FINAL_PREDICTION_COLUMNS
    assert list(metrics.columns) == METRIC_COLUMNS

    assert len(predictions) == 1194
    assert len(metrics) == 3

    assert set(predictions["method"]) == {
        METHOD_HISTORICAL,
        METHOD_EWMA,
        METHOD_GRADIENT_BOOSTING,
    }

    for method in (
        METHOD_HISTORICAL,
        METHOD_EWMA,
        METHOD_GRADIENT_BOOSTING,
    ):
        method_predictions = (
            predictions.loc[
                predictions["method"] == method
            ]
            .sort_values("target_date")
            .reset_index(drop=True)
        )

        assert len(method_predictions) == 398

        assert (
            method_predictions["target_date"].iloc[0]
            == pd.Timestamp("2024-12-18")
        )

        assert (
            method_predictions["target_date"].iloc[-1]
            == pd.Timestamp("2026-07-28")
        )

    indexed_metrics = metrics.set_index("method")

    assert int(
        indexed_metrics.loc[
            METHOD_HISTORICAL,
            "violation_count",
        ]
    ) == 27

    assert int(
        indexed_metrics.loc[
            METHOD_EWMA,
            "violation_count",
        ]
    ) == 22

    assert int(
        indexed_metrics.loc[
            METHOD_GRADIENT_BOOSTING,
            "violation_count",
        ]
    ) == 24

    assert metadata["contract"]["alpha"] == 0.05
    assert metadata["contract"]["confidence_level"] == 0.95
    assert metadata["evaluation"]["target_count"] == 398

    assert (
        metadata["evaluation"][
            "used_for_parameter_selection"
        ]
        is False
    )

    assert (
        metadata["evaluation"][
            "pristine_untouched_test_claim"
        ]
        is False
    )

    gb_metadata = metadata["methods"][
        METHOD_GRADIENT_BOOSTING
    ]

    assert gb_metadata["experiment_id"] == "G04"
    assert gb_metadata["random_state"] == 42
    assert gb_metadata["learning_rate"] == 0.03
    assert gb_metadata["max_depth"] == 2
    assert len(gb_metadata["features"]) == 7
