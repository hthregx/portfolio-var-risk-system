from copy import deepcopy
from pathlib import Path

import yaml

from scripts.validate_model_freeze_a import (
    REPO_ROOT,
    validate_model_freeze,
)


FREEZE_PATH = REPO_ROOT / "configs" / "model_freeze.yaml"


def load_freeze():
    with FREEZE_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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