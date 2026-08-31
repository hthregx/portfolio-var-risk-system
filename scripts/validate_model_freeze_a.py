from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FREEZE_PATH = (
    REPO_ROOT
    / "configs"
    / "model_freeze.yaml"
)

RESULT_COLUMNS = [
    "component",
    "contract",
    "expected_value",
    "observed_value",
    "status",
    "source",
]

EXPECTED_METHODS = {
    "historical_simulation",
    "ewma",
    "gradient_boosting",
}

EXPECTED_CONFIG_IDS = {
    "historical_simulation": "historical_w250",
    "ewma": "ewma_d094",
    "gradient_boosting": "gb_G04",
}

EXPECTED_VIOLATIONS = {
    "historical_simulation": 27,
    "ewma": 22,
    "gradient_boosting": 24,
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

EXPECTED_GB_MODEL_CONFIG_KEYS = {
    "alpha": "alpha",
    "n_estimators": "n_estimators",
    "learning_rate": "learning_rate",
    "max_depth": "max_depth",
    "min_samples_leaf": "min_samples_leaf",
    "subsample": "subsample",
    "random_state": "random_state",
}


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"YAML file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected YAML mapping: {path}"
        )

    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def display_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()

    if isinstance(value, set):
        value = sorted(value)

    if isinstance(
        value,
        (dict, list, tuple),
    ):
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        )

    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()

    return str(value)


def bool_series(
    values: pd.Series,
) -> np.ndarray:
    if values.dtype == bool:
        return values.to_numpy(dtype=bool)

    normalized = (
        values
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
    }

    if not normalized.isin(mapping).all():
        raise ValueError(
            "Violation column contains "
            "non-boolean values."
        )

    return (
        normalized
        .map(mapping)
        .to_numpy(dtype=bool)
    )


class ValidationCollector:
    def __init__(self) -> None:
        self.records: list[
            dict[str, str]
        ] = []

    def check(
        self,
        component: str,
        contract: str,
        expected: Any,
        observed: Any,
        source: str,
        *,
        passed: bool | None = None,
    ) -> None:
        if passed is None:
            passed = observed == expected

        self.records.append(
            {
                "component": component,
                "contract": contract,
                "expected_value": display_value(
                    expected
                ),
                "observed_value": display_value(
                    observed
                ),
                "status": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
                "source": source,
            }
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.records,
            columns=RESULT_COLUMNS,
        )


def git_base_is_ancestor(
    root: Path,
    base_sha: str,
) -> bool:
    result = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            base_sha,
            "HEAD",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    return result.returncode == 0


def function_node(
    tree: ast.AST,
    name: str,
) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and node.name == name
        ):
            return node

    raise ValueError(
        f"Function not found in canonical runner: {name}"
    )


def called_names(
    node: ast.AST,
) -> set[str]:
    names: set[str] = set()

    for child in ast.walk(node):
        if not isinstance(
            child,
            ast.Call,
        ):
            continue

        if isinstance(
            child.func,
            ast.Name,
        ):
            names.add(
                child.func.id
            )

        elif isinstance(
            child.func,
            ast.Attribute,
        ):
            names.add(
                child.func.attr
            )

    return names


def imported_modules(
    tree: ast.AST,
) -> set[str]:
    modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.Import,
        ):
            for alias in node.names:
                modules.add(
                    alias.name
                )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            if node.module:
                modules.add(
                    node.module
                )

    return modules


def extract_model_config_key(
    expression: ast.AST,
) -> str | None:
    if (
        isinstance(
            expression,
            ast.Call,
        )
        and len(expression.args) == 1
    ):
        return extract_model_config_key(
            expression.args[0]
        )

    if not isinstance(
        expression,
        ast.Subscript,
    ):
        return None

    if not (
        isinstance(
            expression.value,
            ast.Name,
        )
        and expression.value.id
        == "model_config"
    ):
        return None

    slice_node = expression.slice

    if (
        isinstance(
            slice_node,
            ast.Constant,
        )
        and isinstance(
            slice_node.value,
            str,
        )
    ):
        return slice_node.value

    return None


def gb_constructor_mapping(
    tree: ast.AST,
) -> dict[str, str | None]:
    node = function_node(
        tree,
        "build_gb_model",
    )

    for child in ast.walk(node):
        if not isinstance(
            child,
            ast.Call,
        ):
            continue

        if not (
            isinstance(
                child.func,
                ast.Name,
            )
            and child.func.id
            == "GradientBoostingVaR"
        ):
            continue

        return {
            keyword.arg:
            extract_model_config_key(
                keyword.value
            )
            for keyword
            in child.keywords
            if keyword.arg is not None
        }

    raise ValueError(
        "GradientBoostingVaR constructor "
        "call not found."
    )


def gb_feature_config_wiring(
    tree: ast.AST,
) -> bool:
    node = function_node(
        tree,
        "run_gradient_boosting",
    )

    for child in ast.walk(node):
        if not isinstance(
            child,
            ast.Assign,
        ):
            continue

        if len(
            child.targets
        ) != 1:
            continue

        target = child.targets[0]

        if not (
            isinstance(
                target,
                ast.Name,
            )
            and target.id
            == "feature_columns"
        ):
            continue

        value = child.value

        if not (
            isinstance(
                value,
                ast.Call,
            )
            and isinstance(
                value.func,
                ast.Name,
            )
            and value.func.id
            == "list"
            and len(value.args) == 1
        ):
            continue

        key = extract_model_config_key(
            value.args[0]
        )

        return key == "features"

    return False


def validate_model_freeze(
    freeze_path: Path | str = DEFAULT_FREEZE_PATH,
    root: Path | str = REPO_ROOT,
) -> pd.DataFrame:
    root = Path(root).resolve()
    freeze_path = Path(
        freeze_path
    )

    if not freeze_path.is_absolute():
        freeze_path = (
            root
            / freeze_path
        )

    freeze = load_yaml(
        freeze_path
    )

    collector = (
        ValidationCollector()
    )

    freeze_block = freeze[
        "freeze"
    ]

    policy = freeze_block[
        "policy"
    ]

    collector.check(
        "freeze",
        "status",
        "frozen",
        freeze_block["status"],
        freeze_path.as_posix(),
    )

    expected_policy = {
        "new_algorithms_allowed": False,
        "new_features_allowed": False,
        "parameter_retuning_allowed": False,
        "evaluation_period_change_allowed": False,
        "canonical_prediction_rewrite_allowed": False,
    }

    collector.check(
        "freeze",
        "post_freeze_policy",
        expected_policy,
        policy,
        freeze_path.as_posix(),
    )

    base_sha = str(
        freeze_block[
            "integration_base_sha"
        ]
    )

    collector.check(
        "freeze",
        "integration_base_is_ancestor_of_head",
        True,
        git_base_is_ancestor(
            root,
            base_sha,
        ),
        "git history",
    )

    canonical_data = freeze[
        "canonical_data"
    ]

    data_path = (
        root
        / canonical_data["path"]
    )

    data = pd.read_csv(
        data_path
    )

    dates = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    collector.check(
        "data",
        "sha256",
        canonical_data["sha256"],
        sha256(data_path),
        canonical_data["path"],
    )

    collector.check(
        "data",
        "row_count",
        canonical_data["row_count"],
        len(data),
        canonical_data["path"],
    )

    collector.check(
        "data",
        "start_date",
        canonical_data["start_date"],
        dates.min().date().isoformat(),
        canonical_data["path"],
    )

    collector.check(
        "data",
        "end_date",
        canonical_data["end_date"],
        dates.max().date().isoformat(),
        canonical_data["path"],
    )

    collector.check(
        "data",
        "duplicate_dates",
        False,
        bool(
            dates
            .duplicated()
            .any()
        ),
        canonical_data["path"],
    )

    collector.check(
        "data",
        "missing_values",
        False,
        bool(
            data
            .isna()
            .any()
            .any()
        ),
        canonical_data["path"],
    )

    artifacts = freeze[
        "canonical_artifacts"
    ]

    final_config_path = (
        root
        / artifacts[
            "final_evaluation_config"
        ][
            "path"
        ]
    )

    final_config = load_yaml(
        final_config_path
    )

    evaluation = freeze[
        "evaluation"
    ]

    source_contract = final_config[
        "contract"
    ]

    source_period = final_config[
        "evaluation_period"
    ]

    source_rules = final_config[
        "rules"
    ]

    collector.check(
        "evaluation",
        "confidence_level",
        evaluation[
            "confidence_level"
        ],
        source_contract[
            "confidence_level"
        ],
        final_config_path.as_posix(),
    )

    collector.check(
        "evaluation",
        "alpha",
        evaluation["alpha"],
        source_contract["alpha"],
        final_config_path.as_posix(),
    )

    collector.check(
        "evaluation",
        "forecast_horizon",
        evaluation[
            "forecast_horizon_trading_days"
        ],
        source_contract[
            "forecast_horizon"
        ],
        final_config_path.as_posix(),
    )

    collector.check(
        "evaluation",
        "portfolio",
        "equal_weight_HPG_FPT_MWG",
        source_contract[
            "portfolio"
        ],
        final_config_path.as_posix(),
    )

    collector.check(
        "evaluation",
        "start_date",
        evaluation[
            "start_date"
        ],
        str(
            source_period["start"]
        ),
        final_config_path.as_posix(),
    )

    collector.check(
        "evaluation",
        "end_date",
        evaluation[
            "end_date"
        ],
        str(
            source_period["end"]
        ),
        final_config_path.as_posix(),
    )

    collector.check(
        "evaluation",
        "used_for_parameter_selection",
        evaluation[
            "used_for_parameter_selection"
        ],
        source_period[
            "used_for_parameter_selection"
        ],
        final_config_path.as_posix(),
    )

    collector.check(
        "evaluation",
        "pristine_untouched_test_claim",
        evaluation[
            "pristine_untouched_test_claim"
        ],
        source_period[
            "pristine_untouched_test_claim"
        ],
        final_config_path.as_posix(),
    )

    collector.check(
        "evaluation",
        "violation_rule",
        evaluation[
            "violation_rule"
        ],
        source_rules[
            "violation"
        ],
        final_config_path.as_posix(),
    )

    collector.check(
        "evaluation",
        "var_definition",
        evaluation[
            "var_definition"
        ],
        source_rules["var"],
        final_config_path.as_posix(),
    )

    frozen_historical = freeze[
        "models"
    ][
        "historical_simulation"
    ]

    source_historical = final_config[
        "historical_simulation"
    ]

    for field in (
        "window",
        "mode",
        "alpha",
    ):
        collector.check(
            "historical_simulation",
            field,
            frozen_historical[
                field
            ],
            source_historical[
                field
            ],
            final_config_path.as_posix(),
        )

    collector.check(
        "historical_simulation",
        "config_id",
        "historical_w250",
        frozen_historical[
            "config_id"
        ],
        freeze_path.as_posix(),
    )

    frozen_ewma = freeze[
        "models"
    ][
        "ewma"
    ]

    source_ewma = final_config[
        "ewma"
    ]

    for field in (
        "decay",
        "mode",
        "alpha",
        "initialization",
        "distribution",
        "mean_assumption",
    ):
        collector.check(
            "ewma",
            field,
            frozen_ewma[
                field
            ],
            source_ewma[
                field
            ],
            final_config_path.as_posix(),
        )

    collector.check(
        "ewma",
        "config_id",
        "ewma_d094",
        frozen_ewma[
            "config_id"
        ],
        freeze_path.as_posix(),
    )

    collector.check(
        "ewma",
        "validation_alternative_decay",
        0.90,
        frozen_ewma[
            "validation_selected_alternative"
        ][
            "decay"
        ],
        freeze_path.as_posix(),
    )

    collector.check(
        "ewma",
        "validation_alternative_adopted",
        False,
        frozen_ewma[
            "validation_selected_alternative"
        ][
            "adopted_for_canonical_evaluation"
        ],
        freeze_path.as_posix(),
    )

    frozen_gb = freeze[
        "models"
    ][
        "gradient_boosting"
    ]

    source_gb = final_config[
        "gradient_boosting"
    ]

    for field in (
        "experiment_id",
        "alpha",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_samples_leaf",
        "subsample",
        "random_state",
        "training_protocol",
    ):
        collector.check(
            "gradient_boosting",
            field,
            frozen_gb[
                field
            ],
            source_gb[
                field
            ],
            final_config_path.as_posix(),
        )

    collector.check(
        "gradient_boosting",
        "config_id",
        "gb_G04",
        frozen_gb[
            "config_id"
        ],
        freeze_path.as_posix(),
    )

    collector.check(
        "gradient_boosting",
        "feature_count",
        7,
        frozen_gb[
            "feature_count"
        ],
        freeze_path.as_posix(),
    )

    collector.check(
        "gradient_boosting",
        "features",
        EXPECTED_GB_FEATURES,
        frozen_gb[
            "features"
        ],
        freeze_path.as_posix(),
    )

    collector.check(
        "gradient_boosting",
        "features_match_final_config",
        frozen_gb[
            "features"
        ],
        source_gb[
            "features"
        ],
        final_config_path.as_posix(),
    )

    collector.check(
        "gradient_boosting",
        "market_features_used_by_canonical_g04",
        False,
        frozen_gb[
            "market_features_used_by_canonical_g04"
        ],
        freeze_path.as_posix(),
    )

    collector.check(
        "gradient_boosting",
        "overall_g01_g07_winner_claim",
        False,
        frozen_gb[
            "overall_g01_g07_winner_claim"
        ],
        freeze_path.as_posix(),
    )

    predictions_path = (
        root
        / artifacts[
            "final_predictions"
        ][
            "path"
        ]
    )

    predictions = pd.read_csv(
        predictions_path
    )

    predictions[
        "forecast_date"
    ] = pd.to_datetime(
        predictions[
            "forecast_date"
        ],
        errors="raise",
    )

    predictions[
        "target_date"
    ] = pd.to_datetime(
        predictions[
            "target_date"
        ],
        errors="raise",
    )

    collector.check(
        "predictions",
        "row_count",
        evaluation[
            "prediction_row_count"
        ],
        len(predictions),
        predictions_path.as_posix(),
    )

    collector.check(
        "predictions",
        "method_set",
        EXPECTED_METHODS,
        set(
            predictions[
                "method"
            ]
        ),
        predictions_path.as_posix(),
    )

    counts = (
        predictions
        .groupby(
            "method"
        )
        .size()
        .astype(int)
        .to_dict()
    )

    expected_counts = {
        method:
        evaluation[
            "target_date_count"
        ]
        for method
        in EXPECTED_METHODS
    }

    collector.check(
        "predictions",
        "forecast_count_per_method",
        expected_counts,
        counts,
        predictions_path.as_posix(),
    )

    collector.check(
        "predictions",
        "start_date",
        evaluation[
            "start_date"
        ],
        predictions[
            "target_date"
        ]
        .min()
        .date()
        .isoformat(),
        predictions_path.as_posix(),
    )

    collector.check(
        "predictions",
        "end_date",
        evaluation[
            "end_date"
        ],
        predictions[
            "target_date"
        ]
        .max()
        .date()
        .isoformat(),
        predictions_path.as_posix(),
    )

    collector.check(
        "predictions",
        "forecast_date_before_target_date",
        True,
        bool(
            (
                predictions[
                    "forecast_date"
                ]
                <
                predictions[
                    "target_date"
                ]
            ).all()
        ),
        predictions_path.as_posix(),
    )

    collector.check(
        "predictions",
        "duplicate_method_target_keys",
        False,
        bool(
            predictions
            .duplicated(
                [
                    "method",
                    "target_date",
                ]
            )
            .any()
        ),
        predictions_path.as_posix(),
    )

    target_sets = {
        method: set(
            part[
                "target_date"
            ]
        )
        for method, part
        in predictions.groupby(
            "method"
        )
    }

    reference_targets = (
        target_sets[
            "historical_simulation"
        ]
    )

    common_targets = all(
        targets
        == reference_targets
        for targets
        in target_sets.values()
    )

    collector.check(
        "predictions",
        "common_target_dates",
        True,
        common_targets,
        predictions_path.as_posix(),
    )

    actual_counts = (
        predictions
        .groupby(
            "target_date"
        )[
            "actual_return"
        ]
        .nunique(
            dropna=False
        )
    )

    collector.check(
        "predictions",
        "common_actual_return_per_target",
        True,
        bool(
            int(
                actual_counts.max()
            )
            == 1
        ),
        predictions_path.as_posix(),
    )

    expected_var = np.maximum(
        0.0,
        -predictions[
            "quantile_return"
        ].to_numpy(
            dtype=float
        ),
    )

    observed_var = (
        predictions[
            "var"
        ].to_numpy(
            dtype=float
        )
    )

    collector.check(
        "predictions",
        "var_sign_rule",
        True,
        bool(
            np.allclose(
                observed_var,
                expected_var,
                rtol=0.0,
                atol=1e-12,
            )
        ),
        predictions_path.as_posix(),
    )

    expected_violation = (
        predictions[
            "actual_return"
        ].to_numpy(
            dtype=float
        )
        <
        predictions[
            "quantile_return"
        ].to_numpy(
            dtype=float
        )
    )

    observed_violation = (
        bool_series(
            predictions[
                "violation"
            ]
        )
    )

    collector.check(
        "predictions",
        "strict_violation_rule",
        True,
        bool(
            np.array_equal(
                observed_violation,
                expected_violation,
            )
        ),
        predictions_path.as_posix(),
    )

    violation_counts = (
        pd.DataFrame(
            {
                "method":
                predictions[
                    "method"
                ],
                "violation":
                expected_violation,
            }
        )
        .groupby(
            "method"
        )[
            "violation"
        ]
        .sum()
        .astype(int)
        .to_dict()
    )

    collector.check(
        "predictions",
        "canonical_violation_counts",
        EXPECTED_VIOLATIONS,
        violation_counts,
        predictions_path.as_posix(),
    )

    observed_config_ids = {
        method: sorted(
            part[
                "config_id"
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        for method, part
        in predictions.groupby(
            "method"
        )
    }

    expected_config_ids = {
        method: [
            config_id
        ]
        for method, config_id
        in EXPECTED_CONFIG_IDS.items()
    }

    collector.check(
        "predictions",
        "canonical_config_ids",
        expected_config_ids,
        observed_config_ids,
        predictions_path.as_posix(),
    )

    for artifact_name, artifact in artifacts.items():
        artifact_path = (
            root
            / artifact[
                "path"
            ]
        )

        collector.check(
            "artifacts",
            f"{artifact_name}_exists",
            True,
            artifact_path.is_file(),
            artifact["path"],
        )

        observed_hash = (
            sha256(
                artifact_path
            )
            if artifact_path.is_file()
            else "<missing>"
        )

        collector.check(
            "artifacts",
            f"{artifact_name}_sha256",
            artifact[
                "sha256"
            ],
            observed_hash,
            artifact[
                "path"
            ],
        )

    runner_path = (
        root
        / freeze[
            "reproducibility"
        ][
            "canonical_runner"
        ]
    )

    runner_source = (
        runner_path
        .read_text(
            encoding="utf-8"
        )
    )

    runner_tree = ast.parse(
        runner_source,
        filename=str(
            runner_path
        ),
    )

    imports = imported_modules(
        runner_tree
    )

    collector.check(
        "implementation",
        "imports_return_feature_builder",
        True,
        (
            "src.gb_return_features"
            in imports
        ),
        runner_path.as_posix(),
    )

    collector.check(
        "implementation",
        "imports_market_feature_builder",
        False,
        (
            "src.gb_market_features"
            in imports
        ),
        runner_path.as_posix(),
    )

    gb_frame = function_node(
        runner_tree,
        "build_gb_frame",
    )

    gb_calls = called_names(
        gb_frame
    )

    collector.check(
        "implementation",
        "build_gb_frame_uses_return_features",
        True,
        (
            "build_return_features"
            in gb_calls
        ),
        runner_path.as_posix(),
    )

    collector.check(
        "implementation",
        "build_gb_frame_uses_market_features",
        False,
        (
            "build_market_features"
            in gb_calls
        ),
        runner_path.as_posix(),
    )

    collector.check(
        "implementation",
        "run_gradient_boosting_uses_configured_feature_list",
        True,
        gb_feature_config_wiring(
            runner_tree
        ),
        runner_path.as_posix(),
    )

    constructor_mapping = (
        gb_constructor_mapping(
            runner_tree
        )
    )

    collector.check(
        "implementation",
        "gb_constructor_config_wiring",
        EXPECTED_GB_MODEL_CONFIG_KEYS,
        constructor_mapping,
        runner_path.as_posix(),
    )

    reconciliation = freeze[
        "gradient_boosting_candidate_reconciliation"
    ]

    tuning_path = (
        root
        / "results"
        / "gb_tuning_b.csv"
    )

    tuning = pd.read_csv(
        tuning_path
    )

    observed_candidate_ids = sorted(
        tuning[
            "experiment_id"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    collector.check(
        "gb_candidate_reconciliation",
        "b_candidate_ids",
        reconciliation[
            "b_candidate_ids"
        ],
        observed_candidate_ids,
        tuning_path.as_posix(),
    )

    feature_counts = sorted(
        tuning[
            "feature_count"
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    collector.check(
        "gb_candidate_reconciliation",
        "b_feature_count",
        [
            reconciliation[
                "b_feature_count"
            ]
        ],
        feature_counts,
        tuning_path.as_posix(),
    )

    best_pinball_row = (
        tuning
        .sort_values(
            [
                "pinball_loss",
                "experiment_id",
            ]
        )
        .iloc[0]
    )

    collector.check(
        "gb_candidate_reconciliation",
        "lowest_pinball_candidate",
        reconciliation[
            "b_lowest_pinball_candidate_on_its_validation_slice"
        ],
        str(
            best_pinball_row[
                "experiment_id"
            ]
        ),
        tuning_path.as_posix(),
    )

    collector.check(
        "gb_candidate_reconciliation",
        "feature_specification_is_not_like_for_like",
        False,
        (
            int(
                reconciliation[
                    "b_feature_count"
                ]
            )
            ==
            int(
                frozen_gb[
                    "feature_count"
                ]
            )
        ),
        freeze_path.as_posix(),
    )

    return collector.frame()


def print_summary(
    frame: pd.DataFrame,
) -> None:
    pass_count = int(
        (
            frame[
                "status"
            ]
            == "PASS"
        ).sum()
    )

    fail_count = int(
        (
            frame[
                "status"
            ]
            == "FAIL"
        ).sum()
    )

    print(
        f"Validation checks : {len(frame)}"
    )

    print(
        f"PASS              : {pass_count}"
    )

    print(
        f"FAIL              : {fail_count}"
    )

    if fail_count:
        print("")
        print(
            "[FAILED CHECKS]"
        )

        print(
            frame.loc[
                frame[
                    "status"
                ]
                == "FAIL"
            ].to_string(
                index=False
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Day-23 frozen VaR model "
            "specification against canonical data, "
            "configuration, implementation, and "
            "result artifacts."
        )
    )

    parser.add_argument(
        "--freeze",
        type=Path,
        default=DEFAULT_FREEZE_PATH,
        help=(
            "Path to model_freeze.yaml."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output CSV. "
            "No CSV is written when omitted."
        ),
    )

    args = parser.parse_args()

    frame = validate_model_freeze(
        freeze_path=args.freeze,
        root=REPO_ROOT,
    )

    print_summary(
        frame
    )

    if args.output is not None:
        output = args.output

        if not output.is_absolute():
            output = (
                REPO_ROOT
                / output
            )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        frame.to_csv(
            output,
            index=False,
            lineterminator="\n",
        )

        print("")
        try:
            display_output = output.relative_to(
                REPO_ROOT
            )
        except ValueError:
            display_output = output

        print(
            "Validation artifact:",
            display_output,
        )

    fail_count = int(
        (
            frame[
                "status"
            ]
            == "FAIL"
        ).sum()
    )

    if fail_count:
        print("")
        print(
            "MODEL_FREEZE_VALIDATION_FAIL"
        )

        return 1

    print("")
    print(
        "MODEL_FREEZE_VALIDATION_PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )