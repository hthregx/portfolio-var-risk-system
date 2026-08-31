from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = ROOT / "results" / "final_predictions.csv"
COMPARISON_PATH = ROOT / "results" / "final_metric_comparison.csv"
PAIRWISE_PATH = ROOT / "results" / "final_pairwise_summary.csv"
TRACEABILITY_PATH = ROOT / "results" / "final_claim_traceability_a.csv"
VALIDATION_PATH = ROOT / "results" / "final_evidence_validation_a.csv"
FREEZE_PATH = ROOT / "configs" / "model_freeze.yaml"
STORY_PATH = ROOT / "docs" / "day25-final-quantitative-story-a.md"
METADATA_PATH = ROOT / "results" / "final_run_metadata.json"

FIGURE_PATHS = (
    ROOT / "results" / "figures" / "final_violation_rate_a.png",
    ROOT / "results" / "figures" / "final_pinball_loss_a.png",
    ROOT / "results" / "figures" / "final_average_var_a.png",
)

OUTPUT_PATH = ROOT / "results" / "final_acceptance_matrix_a.csv"

TOLERANCE = 1e-12
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

METHOD_ORDER = (
    "historical_simulation",
    "ewma",
    "gradient_boosting",
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

ACCEPTANCE_COLUMNS = (
    "acceptance_id",
    "category",
    "check",
    "expected",
    "actual",
    "tolerance",
    "status",
    "source",
)


@dataclass(frozen=True)
class AcceptanceRow:
    acceptance_id: str
    category: str
    check: str
    expected: str
    actual: str
    tolerance: str
    status: str
    source: str


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")

        return list(reader)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(f"Expected boolean text, received: {value!r}")


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def numeric_text(value: float) -> str:
    return format(value, ".17g")


def add_check(
    rows: list[AcceptanceRow],
    acceptance_id: str,
    category: str,
    check: str,
    expected: str,
    actual: str,
    passed: bool,
    source: str,
) -> None:
    rows.append(
        AcceptanceRow(
            acceptance_id=acceptance_id,
            category=category,
            check=check,
            expected=expected,
            actual=actual,
            tolerance="1e-12",
            status="PASS" if passed else "FAIL",
            source=source,
        )
    )


def add_exact(
    rows: list[AcceptanceRow],
    acceptance_id: str,
    category: str,
    check: str,
    expected: Any,
    actual: Any,
    source: str,
) -> None:
    expected_text = str(expected)
    actual_text = str(actual)

    add_check(
        rows,
        acceptance_id,
        category,
        check,
        expected_text,
        actual_text,
        actual_text == expected_text,
        source,
    )


def add_numeric_mapping(
    rows: list[AcceptanceRow],
    acceptance_id: str,
    category: str,
    check: str,
    expected: dict[str, float],
    actual: dict[str, float],
    source: str,
) -> None:
    expected_text = "|".join(
        f"{method}={numeric_text(expected[method])}"
        for method in METHOD_ORDER
    )
    actual_text = "|".join(
        f"{method}={numeric_text(actual[method])}"
        for method in METHOD_ORDER
    )

    passed = all(
        abs(actual[method] - expected[method]) <= TOLERANCE
        for method in METHOD_ORDER
    )

    add_check(
        rows,
        acceptance_id,
        category,
        check,
        expected_text,
        actual_text,
        passed,
        source,
    )


def unique_index(
    rows: list[dict[str, str]],
    key: str,
    source: Path,
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}

    for row in rows:
        value = row[key]

        if value in indexed:
            raise ValueError(
                f"Duplicate {key}={value!r} in {source}"
            )

        indexed[value] = row

    return indexed


def unique_prediction_config(
    predictions: list[dict[str, str]],
    method: str,
) -> str:
    values = sorted(
        {
            row["config_id"]
            for row in predictions
            if row["method"] == method
        }
    )

    return "|".join(values)


def unique_leader(
    comparison: list[dict[str, str]],
    flag_column: str,
) -> str:
    leaders = sorted(
        row["method"]
        for row in comparison
        if parse_bool(row[flag_column])
    )

    return "|".join(leaders)


def valid_png(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 5000:
        return False

    with path.open("rb") as handle:
        return handle.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE


def build_acceptance_rows() -> list[AcceptanceRow]:
    predictions = read_csv(PREDICTIONS_PATH)
    comparison = read_csv(COMPARISON_PATH)
    pairwise = read_csv(PAIRWISE_PATH)
    traceability = read_csv(TRACEABILITY_PATH)
    validation = read_csv(VALIDATION_PATH)

    with FREEZE_PATH.open("r", encoding="utf-8") as handle:
        freeze = yaml.safe_load(handle)

    with STORY_PATH.open("r", encoding="utf-8") as handle:
        story = handle.read()

    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if not isinstance(freeze, dict):
        raise ValueError("Model freeze must be a mapping.")

    rows: list[AcceptanceRow] = []

    method_counts = Counter(row["method"] for row in predictions)

    target_dates = sorted(
        {row["target_date"] for row in predictions}
    )

    target_groups: dict[str, list[dict[str, str]]] = {}

    for row in predictions:
        target_groups.setdefault(row["target_date"], []).append(row)

    # 01
    add_exact(
        rows,
        "01",
        "evaluation_contract",
        "canonical_prediction_rows",
        1194,
        len(predictions),
        "results/final_predictions.csv",
    )

    # 02
    expected_panel = (
        "historical_simulation=398|"
        "ewma=398|"
        "gradient_boosting=398"
    )

    actual_panel = "|".join(
        f"{method}={method_counts.get(method, 0)}"
        for method in METHOD_ORDER
    )

    add_exact(
        rows,
        "02",
        "evaluation_contract",
        "canonical_panel_shape",
        expected_panel,
        actual_panel,
        "results/final_predictions.csv",
    )

    # 03
    add_exact(
        rows,
        "03",
        "evaluation_contract",
        "common_target_date_count",
        398,
        len(target_dates),
        "results/final_predictions.csv",
    )

    # 04
    add_exact(
        rows,
        "04",
        "evaluation_contract",
        "evaluation_start",
        "2024-12-18",
        target_dates[0] if target_dates else "",
        "results/final_predictions.csv",
    )

    # 05
    add_exact(
        rows,
        "05",
        "evaluation_contract",
        "evaluation_end",
        "2026-07-28",
        target_dates[-1] if target_dates else "",
        "results/final_predictions.csv",
    )

    # 06
    expected_method_set = set(METHOD_ORDER)

    common_panel = all(
        len(group) == 3
        and {row["method"] for row in group} == expected_method_set
        for group in target_groups.values()
    )

    add_exact(
        rows,
        "06",
        "evaluation_contract",
        "common_target_dates_across_methods",
        "true",
        bool_text(common_panel),
        "results/final_predictions.csv",
    )

    # 07
    common_actuals = all(
        len({float(row["actual_return"]) for row in group}) == 1
        for group in target_groups.values()
    )

    add_exact(
        rows,
        "07",
        "evaluation_contract",
        "common_actual_return_per_target",
        "true",
        bool_text(common_actuals),
        "results/final_predictions.csv",
    )

    # 08
    key_counts = Counter(
        (row["method"], row["target_date"])
        for row in predictions
    )

    no_duplicate_keys = all(
        count == 1
        for count in key_counts.values()
    )

    add_exact(
        rows,
        "08",
        "evaluation_contract",
        "unique_method_target_keys",
        "true",
        bool_text(no_duplicate_keys),
        "results/final_predictions.csv",
    )

    # 09
    strict_violation_identity = all(
        parse_bool(row["violation"])
        == (
            float(row["actual_return"])
            < float(row["quantile_return"])
        )
        for row in predictions
    )

    add_exact(
        rows,
        "09",
        "risk_semantics",
        "strict_violation_identity",
        "true",
        bool_text(strict_violation_identity),
        "results/final_predictions.csv",
    )

    # 10
    var_identity = True

    for row in predictions:
        quantile_return = float(row["quantile_return"])
        reported_var = float(row["var"])
        expected_var = max(0.0, -quantile_return)

        if reported_var < 0.0:
            var_identity = False
            break

        if abs(reported_var - expected_var) > TOLERANCE:
            var_identity = False
            break

    add_exact(
        rows,
        "10",
        "risk_semantics",
        "var_identity_and_nonnegative",
        "true",
        bool_text(var_identity),
        "results/final_predictions.csv",
    )

    # 11
    canonical_artifacts = freeze["canonical_artifacts"]

    expected_artifact_keys = (
        "final_evaluation_config",
        "final_predictions",
        "final_metrics",
        "final_run_metadata",
        "final_metric_comparison",
        "final_pairwise_diagnostics",
        "final_pairwise_summary",
    )

    valid_artifact_declarations = 0

    for key in expected_artifact_keys:
        item = canonical_artifacts.get(key)

        if not isinstance(item, dict):
            continue

        path_text = item.get("path")
        sha256_text = item.get("sha256")

        if not isinstance(path_text, str):
            continue

        if not isinstance(sha256_text, str) or len(sha256_text) != 64:
            continue

        path = ROOT / path_text

        if not path.is_file() or path.stat().st_size == 0:
            continue

        valid_artifact_declarations += 1

    add_exact(
        rows,
        "11",
        "release_integrity",
        "canonical_artifact_declarations",
        "7/7",
        f"{valid_artifact_declarations}/7",
        "configs/model_freeze.yaml;canonical frozen artifacts",
    )

    models = freeze["models"]

    historical = models["historical_simulation"]
    ewma = models["ewma"]
    gb = models["gradient_boosting"]

    # 12
    expected_historical = (
        "freeze=historical_w250|"
        "prediction=historical_w250|"
        "window=250|mode=rolling"
    )

    actual_historical = (
        f"freeze={historical['config_id']}|"
        f"prediction={unique_prediction_config(predictions, 'historical_simulation')}|"
        f"window={historical['window']}|"
        f"mode={historical['mode']}"
    )

    add_exact(
        rows,
        "12",
        "frozen_configuration",
        "historical_configuration",
        expected_historical,
        actual_historical,
        "configs/model_freeze.yaml;results/final_predictions.csv",
    )

    # 13
    expected_ewma = (
        "freeze=ewma_d094|"
        "prediction=ewma_d094|"
        "decay=0.94|mode=expanding"
    )

    actual_ewma = (
        f"freeze={ewma['config_id']}|"
        f"prediction={unique_prediction_config(predictions, 'ewma')}|"
        f"decay={ewma['decay']}|"
        f"mode={ewma['mode']}"
    )

    add_exact(
        rows,
        "13",
        "frozen_configuration",
        "ewma_configuration",
        expected_ewma,
        actual_ewma,
        "configs/model_freeze.yaml;results/final_predictions.csv",
    )

    # 14
    ewma_alternative = ewma["validation_selected_alternative"]

    expected_alt = "decay=0.9|adopted=false"

    actual_alt = (
        f"decay={ewma_alternative['decay']}|"
        f"adopted={bool_text(ewma_alternative['adopted_for_canonical_evaluation'])}"
    )

    add_exact(
        rows,
        "14",
        "frozen_configuration",
        "ewma_validation_alternative_not_adopted",
        expected_alt,
        actual_alt,
        "configs/model_freeze.yaml",
    )

    # 15
    expected_gb_config = (
        "freeze=gb_G04|prediction=gb_G04|"
        "n_estimators=100|learning_rate=0.03|"
        "max_depth=2|min_samples_leaf=5|"
        "subsample=1.0|random_state=42"
    )

    actual_gb_config = (
        f"freeze={gb['config_id']}|"
        f"prediction={unique_prediction_config(predictions, 'gradient_boosting')}|"
        f"n_estimators={gb['n_estimators']}|"
        f"learning_rate={gb['learning_rate']}|"
        f"max_depth={gb['max_depth']}|"
        f"min_samples_leaf={gb['min_samples_leaf']}|"
        f"subsample={gb['subsample']}|"
        f"random_state={gb['random_state']}"
    )

    add_exact(
        rows,
        "15",
        "frozen_configuration",
        "gradient_boosting_configuration",
        expected_gb_config,
        actual_gb_config,
        "configs/model_freeze.yaml;results/final_predictions.csv",
    )

    # 16
    expected_features = (
        ",".join(EXPECTED_GB_FEATURES)
        + "|market_features=false"
    )

    actual_features = (
        ",".join(gb["features"])
        + "|market_features="
        + bool_text(gb["market_features_used_by_canonical_g04"])
    )

    add_exact(
        rows,
        "16",
        "frozen_configuration",
        "gradient_boosting_feature_contract",
        expected_features,
        actual_features,
        "configs/model_freeze.yaml",
    )

    comparison_by_method = unique_index(
        comparison,
        "method",
        COMPARISON_PATH,
    )

    # 17
    expected_violations = (
        "historical_simulation=27|"
        "ewma=22|"
        "gradient_boosting=24"
    )

    actual_violations = "|".join(
        f"{method}={int(comparison_by_method[method]['violation_count'])}"
        for method in METHOD_ORDER
    )

    add_exact(
        rows,
        "17",
        "quantitative_evidence",
        "violation_counts",
        expected_violations,
        actual_violations,
        "results/final_metric_comparison.csv",
    )

    # 18
    add_numeric_mapping(
        rows,
        "18",
        "quantitative_evidence",
        "violation_rates",
        {
            "historical_simulation": 0.0678391959798995,
            "ewma": 0.05527638190954774,
            "gradient_boosting": 0.06030150753768844,
        },
        {
            method: float(
                comparison_by_method[method]["violation_rate"]
            )
            for method in METHOD_ORDER
        },
        "results/final_metric_comparison.csv",
    )

    # 19
    add_numeric_mapping(
        rows,
        "19",
        "quantitative_evidence",
        "mean_pinball_losses",
        {
            "historical_simulation": 0.0018964642326659775,
            "ewma": 0.0019683212526850112,
            "gradient_boosting": 0.001758130950957487,
        },
        {
            method: float(
                comparison_by_method[method]["pinball_loss"]
            )
            for method in METHOD_ORDER
        },
        "results/final_metric_comparison.csv",
    )

    # 20
    add_numeric_mapping(
        rows,
        "20",
        "quantitative_evidence",
        "average_var",
        {
            "historical_simulation": 0.02133018927920153,
            "ewma": 0.02508404248953606,
            "gradient_boosting": 0.023349057036283743,
        },
        {
            method: float(
                comparison_by_method[method]["average_var"]
            )
            for method in METHOD_ORDER
        },
        "results/final_metric_comparison.csv",
    )

    # 21
    add_exact(
        rows,
        "21",
        "quantitative_evidence",
        "calibration_leader",
        "ewma",
        unique_leader(comparison, "calibration_leader"),
        "results/final_metric_comparison.csv",
    )

    # 22
    add_exact(
        rows,
        "22",
        "quantitative_evidence",
        "pinball_leader",
        "gradient_boosting",
        unique_leader(comparison, "pinball_leader"),
        "results/final_metric_comparison.csv",
    )

    # 23
    add_exact(
        rows,
        "23",
        "quantitative_evidence",
        "lowest_average_var_method",
        "historical_simulation",
        unique_leader(comparison, "lowest_average_var"),
        "results/final_metric_comparison.csv",
    )

    pairwise_by_name = unique_index(
        pairwise,
        "comparison",
        PAIRWISE_PATH,
    )

    pairwise_expectations = (
        (
            "24",
            "gb_vs_historical",
            151,
            247,
            -0.0001383332817084946,
        ),
        (
            "25",
            "gb_vs_ewma",
            231,
            167,
            -0.00021019030172751563,
        ),
        (
            "26",
            "ewma_vs_historical",
            131,
            267,
            0.00007185702001901683,
        ),
    )

    for (
        acceptance_id,
        comparison_name,
        expected_left,
        expected_right,
        expected_delta,
    ) in pairwise_expectations:
        row = pairwise_by_name[comparison_name]

        actual_left = int(row["left_better_count"])
        actual_right = int(row["right_better_count"])
        actual_delta = float(row["mean_delta"])

        expected_text = (
            f"left={expected_left}|right={expected_right}|"
            f"mean_delta={numeric_text(expected_delta)}"
        )

        actual_text = (
            f"left={actual_left}|right={actual_right}|"
            f"mean_delta={numeric_text(actual_delta)}"
        )

        passed = (
            actual_left == expected_left
            and actual_right == expected_right
            and abs(actual_delta - expected_delta) <= TOLERANCE
        )

        add_check(
            rows,
            acceptance_id,
            "pairwise_evidence",
            comparison_name,
            expected_text,
            actual_text,
            passed,
            "results/final_pairwise_summary.csv",
        )

    # 27
    policy = freeze["freeze"]["policy"]

    policy_keys = (
        "new_algorithms_allowed",
        "new_features_allowed",
        "parameter_retuning_allowed",
        "evaluation_period_change_allowed",
        "canonical_prediction_rewrite_allowed",
    )

    expected_policy = "|".join(
        f"{key}=false"
        for key in policy_keys
    )

    actual_policy = "|".join(
        f"{key}={bool_text(bool(policy[key]))}"
        for key in policy_keys
    )

    add_exact(
        rows,
        "27",
        "release_boundary",
        "model_freeze_policy",
        expected_policy,
        actual_policy,
        "configs/model_freeze.yaml",
    )

    # 28
    trace_pass_count = sum(
        row["status"] == "PASS"
        for row in traceability
    )

    trace_ids = {
        row["claim_id"]
        for row in traceability
    }

    validation_pass_count = sum(
        row["status"] == "PASS"
        for row in validation
    )

    valid_figure_count = sum(
        valid_png(path)
        for path in FIGURE_PATHS
    )

    prior_expected = (
        "traceability=24/24 PASS|"
        "validation=26/26 PASS|"
        "figures=3/3 valid"
    )

    prior_actual = (
        f"traceability={trace_pass_count}/{len(traceability)} PASS|"
        f"validation={validation_pass_count}/{len(validation)} PASS|"
        f"figures={valid_figure_count}/3 valid"
    )

    prior_passed = (
        len(traceability) == 24
        and trace_pass_count == 24
        and len(trace_ids) == 24
        and len(validation) == 26
        and validation_pass_count == 26
        and valid_figure_count == 3
    )

    add_check(
        rows,
        "28",
        "release_integrity",
        "prior_evidence_gates",
        prior_expected,
        prior_actual,
        prior_passed,
        (
            "results/final_claim_traceability_a.csv;"
            "results/final_evidence_validation_a.csv;"
            "results/figures/final_*_a.png"
        ),
    )

    # 29
    story_lower = story.lower()

    no_overall_winner = (
        "no single overall winner" in story_lower
    )

    lower_var_boundary = (
        "lower average var" in story_lower
        and "automatic model superiority" in story_lower
    )

    non_pristine = (
        freeze["evaluation"]["pristine_untouched_test_claim"] is False
        and "pristine" in story_lower
        and "never-inspected test set" in story_lower
    )

    interpretation_actual = (
        "no_overall_winner="
        + bool_text(no_overall_winner)
        + "|lower_var_not_superiority="
        + bool_text(lower_var_boundary)
        + "|non_pristine="
        + bool_text(non_pristine)
    )

    interpretation_expected = (
        "no_overall_winner=true|"
        "lower_var_not_superiority=true|"
        "non_pristine=true"
    )

    add_exact(
        rows,
        "29",
        "release_boundary",
        "interpretation_boundaries",
        interpretation_expected,
        interpretation_actual,
        (
            "docs/day25-final-quantitative-story-a.md;"
            "configs/model_freeze.yaml"
        ),
    )

    # 30
    metadata_git_commit = (
        metadata.get("git", {}).get("commit")
        if isinstance(metadata, dict)
        else None
    )

    provenance_documented = (
        "provenance incompleteness" in story_lower
        and (
            "recorded runtime head predates the later source commit"
            in story_lower
        )
        and (
            "canonical prediction content remained unchanged"
            in story_lower
        )
        and metadata_git_commit
        == "3d7f58c8eab7242d65d399bfea2dd565901d55e9"
    )

    add_exact(
        rows,
        "30",
        "release_boundary",
        "provenance_limitation_documented",
        "true",
        bool_text(provenance_documented),
        (
            "docs/day25-final-quantitative-story-a.md;"
            "results/final_run_metadata.json"
        ),
    )

    return rows


def write_acceptance(rows: list[AcceptanceRow]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ACCEPTANCE_COLUMNS,
            lineterminator="\n",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    rows = build_acceptance_rows()
    write_acceptance(rows)

    pass_count = sum(
        row.status == "PASS"
        for row in rows
    )

    fail_rows = [
        row
        for row in rows
        if row.status != "PASS"
    ]

    category_counts = Counter(
        row.category
        for row in rows
    )

    print(f"Output : {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Rows   : {len(rows)}")
    print(f"PASS   : {pass_count}")
    print(f"FAIL   : {len(fail_rows)}")

    print("Categories:")

    for category in sorted(category_counts):
        print(
            f"  {category}: "
            f"{category_counts[category]}"
        )

    if fail_rows:
        print("Failed checks:")

        for row in fail_rows:
            print(
                f"  {row.acceptance_id} "
                f"{row.check}: "
                f"expected={row.expected!r} "
                f"actual={row.actual!r}"
            )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())