from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS = ROOT / "results" / "final_predictions.csv"
COMPARISON = ROOT / "results" / "final_metric_comparison.csv"
PAIRWISE = ROOT / "results" / "final_pairwise_summary.csv"
VALIDATION = ROOT / "results" / "final_evidence_validation_a.csv"
TRACEABILITY = ROOT / "results" / "final_claim_traceability_a.csv"
ACCEPTANCE = ROOT / "results" / "final_acceptance_matrix_a.csv"
SUMMARY = ROOT / "results" / "final_quantitative_summary_a.csv"
FREEZE = ROOT / "configs" / "model_freeze.yaml"

OUTPUT = ROOT / "results" / "final_reproducibility_checkpoint_a.csv"

TOLERANCE = 1e-12
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

OUTPUT_COLUMNS = (
    "check_id",
    "category",
    "check",
    "expected",
    "actual",
    "tolerance",
    "status",
    "source",
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")

        return list(reader.fieldnames), list(reader)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(f"Invalid boolean value: {value!r}")


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def float_text(value: float) -> str:
    return format(value, ".17g")


def add(
    rows: list[dict[str, str]],
    check_id: str,
    category: str,
    check: str,
    expected: str,
    actual: str,
    passed: bool,
    source: str,
) -> None:
    rows.append(
        {
            "check_id": check_id,
            "category": category,
            "check": check,
            "expected": expected,
            "actual": actual,
            "tolerance": "1e-12",
            "status": "PASS" if passed else "FAIL",
            "source": source,
        }
    )


def add_exact(
    rows: list[dict[str, str]],
    check_id: str,
    category: str,
    check: str,
    expected: object,
    actual: object,
    source: str,
) -> None:
    expected_text = str(expected)
    actual_text = str(actual)

    add(
        rows,
        check_id,
        category,
        check,
        expected_text,
        actual_text,
        expected_text == actual_text,
        source,
    )


def numeric_mapping_text(values: dict[str, float]) -> str:
    return "|".join(
        f"{method}={float_text(values[method])}"
        for method in METHODS
    )


def integer_mapping_text(values: dict[str, int]) -> str:
    return "|".join(
        f"{method}={values[method]}"
        for method in METHODS
    )


def add_numeric_mapping(
    rows: list[dict[str, str]],
    check_id: str,
    category: str,
    check: str,
    expected: dict[str, float],
    actual: dict[str, float],
    source: str,
) -> None:
    passed = all(
        abs(expected[method] - actual[method]) <= TOLERANCE
        for method in METHODS
    )

    add(
        rows,
        check_id,
        category,
        check,
        numeric_mapping_text(expected),
        numeric_mapping_text(actual),
        passed,
        source,
    )


def pinball_loss(actual: float, quantile: float) -> float:
    error = actual - quantile

    if error >= 0:
        return ALPHA * error

    return (1.0 - ALPHA) * (-error)


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


def unique_flagged_method(
    rows: list[dict[str, str]],
    column: str,
) -> str:
    values = [
        row["method"]
        for row in rows
        if parse_bool(row[column])
    ]

    return "|".join(values)


def build_rows() -> list[dict[str, str]]:
    prediction_columns, predictions = read_csv(PREDICTIONS)
    _, comparison = read_csv(COMPARISON)
    _, pairwise = read_csv(PAIRWISE)
    _, validation = read_csv(VALIDATION)
    _, traceability = read_csv(TRACEABILITY)
    _, acceptance = read_csv(ACCEPTANCE)
    _, summary = read_csv(SUMMARY)

    with FREEZE.open("r", encoding="utf-8") as handle:
        freeze = yaml.safe_load(handle)

    rows: list[dict[str, str]] = []

    canonical_artifacts = freeze["canonical_artifacts"]
    declared_paths = [
        ROOT / entry["path"]
        for entry in canonical_artifacts.values()
    ]

    declarations_ok = (
        len(canonical_artifacts) == 7
        and all(path.is_file() for path in declared_paths)
        and all(entry["sha256"] for entry in canonical_artifacts.values())
    )

    add_exact(
        rows,
        "R01",
        "frozen_inputs",
        "canonical_artifact_declarations",
        "7/7",
        "7/7" if declarations_ok else "incomplete",
        "configs/model_freeze.yaml;canonical frozen artifacts",
    )

    add_exact(
        rows,
        "R02",
        "evaluation_contract",
        "prediction_schema",
        "|".join(EXPECTED_PREDICTION_COLUMNS),
        "|".join(prediction_columns),
        "results/final_predictions.csv",
    )

    add_exact(
        rows,
        "R03",
        "evaluation_contract",
        "prediction_row_count",
        1194,
        len(predictions),
        "results/final_predictions.csv",
    )

    method_counts = Counter(row["method"] for row in predictions)
    panel_actual = "|".join(
        f"{method}={method_counts.get(method, 0)}"
        for method in METHODS
    )
    panel_expected = (
        "historical_simulation=398|"
        "ewma=398|"
        "gradient_boosting=398"
    )

    add_exact(
        rows,
        "R04",
        "evaluation_contract",
        "method_panel",
        panel_expected,
        panel_actual,
        "results/final_predictions.csv",
    )

    target_dates = sorted({row["target_date"] for row in predictions})
    range_actual = (
        f"count={len(target_dates)}|"
        f"start={target_dates[0]}|"
        f"end={target_dates[-1]}"
    )

    add_exact(
        rows,
        "R05",
        "evaluation_contract",
        "evaluation_target_range",
        "count=398|start=2024-12-18|end=2026-07-28",
        range_actual,
        "results/final_predictions.csv",
    )

    key_counts = Counter(
        (row["method"], row["target_date"])
        for row in predictions
    )
    unique_keys = (
        len(key_counts) == len(predictions)
        and all(count == 1 for count in key_counts.values())
    )

    add_exact(
        rows,
        "R06",
        "evaluation_contract",
        "unique_method_target_keys",
        "true",
        bool_text(unique_keys),
        "results/final_predictions.csv",
    )

    by_date: dict[str, list[dict[str, str]]] = {}

    for row in predictions:
        by_date.setdefault(row["target_date"], []).append(row)

    common_panel = all(
        len(group) == 3
        and {row["method"] for row in group} == set(METHODS)
        and len({row["actual_return"] for row in group}) == 1
        for group in by_date.values()
    )

    add_exact(
        rows,
        "R07",
        "evaluation_contract",
        "common_panel_and_actual_return",
        "true",
        bool_text(common_panel),
        "results/final_predictions.csv",
    )

    violation_identity = all(
        parse_bool(row["violation"])
        == (
            float(row["actual_return"])
            < float(row["quantile_return"])
        )
        for row in predictions
    )

    add_exact(
        rows,
        "R08",
        "risk_semantics",
        "strict_violation_identity",
        "true",
        bool_text(violation_identity),
        "results/final_predictions.csv",
    )

    var_identity = all(
        float(row["var"]) >= 0.0
        and abs(
            float(row["var"])
            - max(0.0, -float(row["quantile_return"]))
        )
        <= TOLERANCE
        for row in predictions
    )

    add_exact(
        rows,
        "R09",
        "risk_semantics",
        "var_identity_and_nonnegative",
        "true",
        bool_text(var_identity),
        "results/final_predictions.csv",
    )

    policy = freeze["freeze"]["policy"]
    policy_actual = "|".join(
        (
            f"new_algorithms_allowed={bool_text(policy['new_algorithms_allowed'])}",
            f"new_features_allowed={bool_text(policy['new_features_allowed'])}",
            f"parameter_retuning_allowed={bool_text(policy['parameter_retuning_allowed'])}",
            f"evaluation_period_change_allowed={bool_text(policy['evaluation_period_change_allowed'])}",
            f"canonical_prediction_rewrite_allowed={bool_text(policy['canonical_prediction_rewrite_allowed'])}",
        )
    )

    policy_expected = (
        "new_algorithms_allowed=false|"
        "new_features_allowed=false|"
        "parameter_retuning_allowed=false|"
        "evaluation_period_change_allowed=false|"
        "canonical_prediction_rewrite_allowed=false"
    )

    add_exact(
        rows,
        "R10",
        "freeze_contract",
        "model_freeze_policy",
        policy_expected,
        policy_actual,
        "configs/model_freeze.yaml",
    )

    models = freeze["models"]

    hist = models["historical_simulation"]
    hist_actual = (
        f"freeze={hist['config_id']}|"
        f"prediction={unique_prediction_config(predictions, 'historical_simulation')}|"
        f"window={hist['window']}|"
        f"mode={hist['mode']}"
    )

    add_exact(
        rows,
        "R11",
        "frozen_configuration",
        "historical_configuration",
        "freeze=historical_w250|prediction=historical_w250|window=250|mode=rolling",
        hist_actual,
        "configs/model_freeze.yaml;results/final_predictions.csv",
    )

    ewma = models["ewma"]
    alternative = ewma["validation_selected_alternative"]

    ewma_actual = (
        f"freeze={ewma['config_id']}|"
        f"prediction={unique_prediction_config(predictions, 'ewma')}|"
        f"decay={ewma['decay']}|"
        f"mode={ewma['mode']}|"
        f"alt_decay={alternative['decay']}|"
        f"alt_adopted={bool_text(alternative['adopted_for_canonical_evaluation'])}"
    )

    add_exact(
        rows,
        "R12",
        "frozen_configuration",
        "ewma_configuration_and_validation_boundary",
        "freeze=ewma_d094|prediction=ewma_d094|decay=0.94|mode=expanding|alt_decay=0.9|alt_adopted=false",
        ewma_actual,
        "configs/model_freeze.yaml;results/final_predictions.csv",
    )

    gb = models["gradient_boosting"]
    gb_actual = (
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
        "R13",
        "frozen_configuration",
        "gradient_boosting_configuration",
        "freeze=gb_G04|prediction=gb_G04|n_estimators=100|learning_rate=0.03|max_depth=2|min_samples_leaf=5|subsample=1.0|random_state=42",
        gb_actual,
        "configs/model_freeze.yaml;results/final_predictions.csv",
    )

    feature_actual = (
        ",".join(gb["features"])
        + f"|market_features={bool_text(gb['market_features_used_by_canonical_g04'])}"
    )

    add_exact(
        rows,
        "R14",
        "frozen_configuration",
        "gradient_boosting_feature_contract",
        ",".join(EXPECTED_GB_FEATURES) + "|market_features=false",
        feature_actual,
        "configs/model_freeze.yaml",
    )

    comparison_by_method = {
        row["method"]: row
        for row in comparison
    }

    recomputed: dict[str, dict[str, float | int]] = {}

    for method in METHODS:
        method_rows = [
            row
            for row in predictions
            if row["method"] == method
        ]

        actuals = [float(row["actual_return"]) for row in method_rows]
        quantiles = [float(row["quantile_return"]) for row in method_rows]
        vars_ = [float(row["var"]) for row in method_rows]

        violation_count = sum(
            actual < quantile
            for actual, quantile in zip(actuals, quantiles)
        )

        losses = [
            pinball_loss(actual, quantile)
            for actual, quantile in zip(actuals, quantiles)
        ]

        recomputed[method] = {
            "violation_count": violation_count,
            "violation_rate": violation_count / len(method_rows),
            "pinball_loss": sum(losses) / len(losses),
            "average_var": sum(vars_) / len(vars_),
            "minimum_var": min(vars_),
            "maximum_var": max(vars_),
        }

    expected_counts = {
        method: int(comparison_by_method[method]["violation_count"])
        for method in METHODS
    }
    actual_counts = {
        method: int(recomputed[method]["violation_count"])
        for method in METHODS
    }

    add_exact(
        rows,
        "R15",
        "recomputed_quantitative_evidence",
        "violation_counts",
        integer_mapping_text(expected_counts),
        integer_mapping_text(actual_counts),
        "results/final_predictions.csv;results/final_metric_comparison.csv",
    )

    metric_columns = (
        ("R16", "violation_rates", "violation_rate"),
        ("R17", "mean_pinball_losses", "pinball_loss"),
        ("R18", "average_var", "average_var"),
        ("R19", "minimum_var", "minimum_var"),
        ("R20", "maximum_var", "maximum_var"),
    )

    for check_id, check, column in metric_columns:
        expected = {
            method: float(comparison_by_method[method][column])
            for method in METHODS
        }
        actual = {
            method: float(recomputed[method][column])
            for method in METHODS
        }

        add_numeric_mapping(
            rows,
            check_id,
            "recomputed_quantitative_evidence",
            check,
            expected,
            actual,
            "results/final_predictions.csv;results/final_metric_comparison.csv",
        )

    actual_calibration = min(
        METHODS,
        key=lambda method: abs(
            float(recomputed[method]["violation_rate"]) - ALPHA
        ),
    )
    actual_pinball = min(
        METHODS,
        key=lambda method: float(recomputed[method]["pinball_loss"]),
    )
    actual_average_var = min(
        METHODS,
        key=lambda method: float(recomputed[method]["average_var"]),
    )

    expected_leaders = (
        f"calibration={unique_flagged_method(comparison, 'calibration_leader')}|"
        f"pinball={unique_flagged_method(comparison, 'pinball_leader')}|"
        f"average_var={unique_flagged_method(comparison, 'lowest_average_var')}"
    )

    actual_leaders = (
        f"calibration={actual_calibration}|"
        f"pinball={actual_pinball}|"
        f"average_var={actual_average_var}"
    )

    add_exact(
        rows,
        "R21",
        "recomputed_quantitative_evidence",
        "criterion_specific_leaders",
        expected_leaders,
        actual_leaders,
        "results/final_predictions.csv;results/final_metric_comparison.csv",
    )

    loss_panel: dict[str, dict[str, float]] = {}

    for row in predictions:
        target_date = row["target_date"]
        method = row["method"]
        loss_panel.setdefault(target_date, {})[method] = pinball_loss(
            float(row["actual_return"]),
            float(row["quantile_return"]),
        )

    pairwise_by_name = {
        row["comparison"]: row
        for row in pairwise
    }

    pair_specs = (
        (
            "R22",
            "gb_vs_historical",
            "gradient_boosting",
            "historical_simulation",
        ),
        (
            "R23",
            "gb_vs_ewma",
            "gradient_boosting",
            "ewma",
        ),
        (
            "R24",
            "ewma_vs_historical",
            "ewma",
            "historical_simulation",
        ),
    )

    for check_id, name, left_method, right_method in pair_specs:
        deltas = [
            panel[left_method] - panel[right_method]
            for _, panel in sorted(loss_panel.items())
        ]

        left_better = sum(delta < 0.0 for delta in deltas)
        right_better = sum(delta > 0.0 for delta in deltas)
        tie_count = sum(delta == 0.0 for delta in deltas)
        mean_delta = sum(deltas) / len(deltas)

        expected_row = pairwise_by_name[name]

        expected_text = (
            f"left={expected_row['left_better_count']}|"
            f"right={expected_row['right_better_count']}|"
            f"ties={expected_row['tie_count']}|"
            f"mean_delta={float_text(float(expected_row['mean_delta']))}"
        )

        actual_text = (
            f"left={left_better}|"
            f"right={right_better}|"
            f"ties={tie_count}|"
            f"mean_delta={float_text(mean_delta)}"
        )

        passed = (
            left_better == int(expected_row["left_better_count"])
            and right_better == int(expected_row["right_better_count"])
            and tie_count == int(expected_row["tie_count"])
            and abs(mean_delta - float(expected_row["mean_delta"]))
            <= TOLERANCE
        )

        add(
            rows,
            check_id,
            "recomputed_pairwise_evidence",
            name,
            expected_text,
            actual_text,
            passed,
            "results/final_predictions.csv;results/final_pairwise_summary.csv",
        )

    validation_pass = sum(
        row["status"] == "PASS"
        for row in validation
    )

    add_exact(
        rows,
        "R25",
        "prior_evidence_gates",
        "day24_evidence_validation",
        "26/26 PASS",
        f"{validation_pass}/{len(validation)} PASS",
        "results/final_evidence_validation_a.csv",
    )

    traceability_pass = sum(
        row["status"] == "PASS"
        for row in traceability
    )

    add_exact(
        rows,
        "R26",
        "prior_evidence_gates",
        "day25_claim_traceability",
        "24/24 PASS",
        f"{traceability_pass}/{len(traceability)} PASS",
        "results/final_claim_traceability_a.csv",
    )

    acceptance_pass = sum(
        row["status"] == "PASS"
        for row in acceptance
    )

    gate_actual = (
        f"acceptance={acceptance_pass}/{len(acceptance)} PASS|"
        f"quantitative_summary_rows={len(summary)}"
    )

    add_exact(
        rows,
        "R27",
        "prior_evidence_gates",
        "day26_release_evidence",
        "acceptance=30/30 PASS|quantitative_summary_rows=3",
        gate_actual,
        "results/final_acceptance_matrix_a.csv;results/final_quantitative_summary_a.csv",
    )

    acceptance_by_check = {
        row["check"]: row
        for row in acceptance
    }

    interpretation_ok = (
        acceptance_by_check["interpretation_boundaries"]["actual"]
        == "no_overall_winner=true|lower_var_not_superiority=true|non_pristine=true"
    )
    provenance_ok = (
        acceptance_by_check[
            "provenance_limitation_documented"
        ]["actual"]
        == "true"
    )
    non_pristine = (
        freeze["evaluation"]["pristine_untouched_test_claim"] is False
    )

    boundary_actual = (
        f"interpretation={bool_text(interpretation_ok)}|"
        f"provenance={bool_text(provenance_ok)}|"
        f"non_pristine={bool_text(non_pristine)}"
    )

    add_exact(
        rows,
        "R28",
        "release_boundary",
        "interpretation_and_provenance_boundary",
        "interpretation=true|provenance=true|non_pristine=true",
        boundary_actual,
        "results/final_acceptance_matrix_a.csv;configs/model_freeze.yaml",
    )

    return rows


def write_rows(rows: list[dict[str, str]]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows = build_rows()

    expected_ids = [
        f"R{number:02d}"
        for number in range(1, 29)
    ]

    actual_ids = [
        row["check_id"]
        for row in rows
    ]

    if actual_ids != expected_ids:
        raise ValueError(
            f"Unexpected checkpoint IDs: {actual_ids}"
        )

    write_rows(rows)

    failures = [
        row
        for row in rows
        if row["status"] != "PASS"
    ]

    categories = Counter(
        row["category"]
        for row in rows
    )

    print(f"Output: {OUTPUT.relative_to(ROOT)}")
    print(f"Checks: {len(rows)}")
    print(f"PASS: {len(rows) - len(failures)}")
    print(f"FAIL: {len(failures)}")

    for category in sorted(categories):
        print(f"{category}: {categories[category]}")

    for row in failures:
        print(
            "FAIL "
            f"{row['check_id']} "
            f"{row['check']} "
            f"expected={row['expected']} "
            f"actual={row['actual']}"
        )

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())