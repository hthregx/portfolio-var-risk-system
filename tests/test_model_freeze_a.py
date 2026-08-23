from copy import deepcopy
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

import scripts.validate_model_freeze_a as freeze_validator
from scripts.validate_model_freeze_a import (
    REPO_ROOT,
    validate_model_freeze,
)


FREEZE_PATH = REPO_ROOT / "configs" / "model_freeze.yaml"

CANONICAL_DATA_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "portfolio_returns.csv"
)

CI_DATA_SHA256 = (
    "88a6dbac04349e40d84001b06c6a7e876"
    "ec660ea9eb87267a22aeb00c165f2fe"
)

CI_DATA_ROWS = 1637
CI_DATA_START = "2020-01-03"
CI_DATA_END = "2026-07-28"


def use_surrogate_data():
    return (
        os.environ.get(
            "MODEL_FREEZE_USE_SURROGATE_DATA"
        )
        == "1"
        or not CANONICAL_DATA_PATH.is_file()
    )


def load_freeze():
    with FREEZE_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(autouse=True)
def provide_untracked_canonical_data(monkeypatch):
    if not use_surrogate_data():
        return

    dates = pd.date_range(
        start=CI_DATA_START,
        end=CI_DATA_END,
        periods=CI_DATA_ROWS,
    ).strftime("%Y-%m-%d")

    frame = pd.DataFrame(
        {
            "date": dates,
            "HPG_simple_return": 0.0,
            "FPT_simple_return": 0.0,
            "MWG_simple_return": 0.0,
            "portfolio_simple_return": 0.0,
            "portfolio_log_return": 0.0,
        }
    )

    original_read_csv = (
        freeze_validator.pd.read_csv
    )
    original_sha256 = (
        freeze_validator.sha256
    )

    canonical_path = (
        CANONICAL_DATA_PATH.resolve()
    )

    def read_csv(path, *args, **kwargs):
        if Path(path).resolve() == canonical_path:
            return frame.copy()

        return original_read_csv(
            path,
            *args,
            **kwargs,
        )

    def sha256(path):
        if Path(path).resolve() == canonical_path:
            return CI_DATA_SHA256

        return original_sha256(path)

    monkeypatch.setattr(
        freeze_validator.pd,
        "read_csv",
        read_csv,
    )

    monkeypatch.setattr(
        freeze_validator,
        "sha256",
        sha256,
    )


def write_mutated_freeze(tmp_path, mutate):
    payload = deepcopy(load_freeze())
    mutate(payload)

    path = tmp_path / "model_freeze.yaml"

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            payload,
            handle,
            sort_keys=False,
        )

    return path


def failed_contracts(frame):
    failed = frame.loc[
        frame["status"] == "FAIL",
        ["component", "contract"],
    ]

    return {
        tuple(row)
        for row in failed.itertuples(
            index=False,
            name=None,
        )
    }


def validate_mutation(tmp_path, mutate):
    path = write_mutated_freeze(
        tmp_path,
        mutate,
    )

    return validate_model_freeze(
        freeze_path=path,
        root=REPO_ROOT,
    )


def test_canonical_model_freeze_passes():
    if use_surrogate_data():
        pytest.skip(
            "requires canonical processed data, "
            "which is intentionally not version-controlled"
        )

    frame = validate_model_freeze(
        freeze_path=FREEZE_PATH,
        root=REPO_ROOT,
    )

    assert not frame.empty
    assert set(frame["status"]) == {"PASS"}


def test_wrong_historical_window_is_detected(tmp_path):
    frame = validate_mutation(
        tmp_path,
        lambda freeze: freeze["models"][
            "historical_simulation"
        ].update({"window": 200}),
    )

    assert (
        "historical_simulation",
        "window",
    ) in failed_contracts(frame)


def test_wrong_ewma_decay_is_detected(tmp_path):
    frame = validate_mutation(
        tmp_path,
        lambda freeze: freeze["models"][
            "ewma"
        ].update({"decay": 0.90}),
    )

    assert (
        "ewma",
        "decay",
    ) in failed_contracts(frame)


def test_wrong_gb_learning_rate_is_detected(tmp_path):
    frame = validate_mutation(
        tmp_path,
        lambda freeze: freeze["models"][
            "gradient_boosting"
        ].update({"learning_rate": 0.05}),
    )

    assert (
        "gradient_boosting",
        "learning_rate",
    ) in failed_contracts(frame)


def test_missing_gb_feature_is_detected(tmp_path):
    def mutate(freeze):
        freeze["models"]["gradient_boosting"][
            "features"
        ].remove("drawdown")

    frame = validate_mutation(
        tmp_path,
        mutate,
    )

    assert (
        "gradient_boosting",
        "features",
    ) in failed_contracts(frame)


def test_unexpected_gb_feature_is_detected(tmp_path):
    def mutate(freeze):
        freeze["models"]["gradient_boosting"][
            "features"
        ].append("market_return_lag_1")

    frame = validate_mutation(
        tmp_path,
        mutate,
    )

    assert (
        "gradient_boosting",
        "features",
    ) in failed_contracts(frame)


def test_wrong_evaluation_alpha_is_detected(tmp_path):
    frame = validate_mutation(
        tmp_path,
        lambda freeze: freeze[
            "evaluation"
        ].update({"alpha": 0.10}),
    )

    assert (
        "evaluation",
        "alpha",
    ) in failed_contracts(frame)