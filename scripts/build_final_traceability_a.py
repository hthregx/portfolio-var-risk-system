from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = REPO_ROOT / "results" / "final_predictions.csv"
COMPARISON_PATH = REPO_ROOT / "results" / "final_metric_comparison.csv"
PAIRWISE_PATH = REPO_ROOT / "results" / "final_pairwise_summary.csv"
VALIDATION_PATH = REPO_ROOT / "results" / "final_evidence_validation_a.csv"
FREEZE_PATH = REPO_ROOT / "configs" / "model_freeze.yaml"
DAY24_DOC_PATH = REPO_ROOT / "docs" / "day24-final-evidence-a.md"

FIGURE_PATHS = (
    REPO_ROOT / "results" / "figures" / "final_violation_rate_a.png",
    REPO_ROOT / "results" / "figures" / "final_pinball_loss_a.png",
    REPO_ROOT / "results" / "figures" / "final_average_var_a.png",
)

OUTPUT_PATH = REPO_ROOT / "results" / "final_claim_traceability_a.csv"

TOLERANCE = 1e-12

TRACEABILITY_COLUMNS = (
    "claim_id",
    "category",
    "claim",
    "expected",
    "actual",
    "tolerance",
    "status",
    "source",
)


@dataclass(frozen=True)
class ClaimRow:
    claim_id: str
    category: str
    claim: str
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


def require_columns(
    rows: list[dict[str, str]],
    required: set[str],
    source: Path,
) -> None:
    if not rows:
        raise ValueError(f"Empty required artifact: {source}")

    columns = set(rows[0])

    missing = sorted(required - columns)

    if missing:
        raise ValueError(
            f"{source} is missing required columns: {', '.join(missing)}"
        )


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


def add_exact(
    rows: list[ClaimRow],
    claim_id: str,
    category: str,
    claim: str,
    expected: Any,
    actual: Any,
    source: str,
) -> None:
    expected_text = str(expected)
    actual_text = str(actual)

    rows.append(
        ClaimRow(
            claim_id=claim_id,
            category=category,
            claim=claim,
            expected=expected_text,
            actual=actual_text,
            tolerance="1e-12",
            status="PASS" if actual_text == expected_text else "FAIL",
            source=source,
        )
    )


def add_numeric(
    rows: list[ClaimRow],
    claim_id: str,
    category: str,
    claim: str,
    expected: float,
    actual: float,
    source: str,
) -> None:
    difference = abs(actual - expected)

    rows.append(
        ClaimRow(
            claim_id=claim_id,
            category=category,
            claim=claim,
            expected=numeric_text(expected),
            actual=numeric_text(actual),
            tolerance="1e-12",
            status="PASS" if difference <= TOLERANCE else "FAIL",
            source=source,
        )
    )


def unique_value(rows: list[dict[str, str]], column: str) -> str:
    values = sorted({row[column] for row in rows})

    if len(values) != 1:
        return "|".join(values)

    return values[0]


def unique_leader(
    rows: list[dict[str, str]],
    flag_column: str,
) -> str:
    leaders = [
        row["method"]
        for row in rows
        if parse_bool(row[flag_column])
    ]

    if len(leaders) != 1:
        return "|".join(sorted(leaders))

    return leaders[0]


def load_inputs() -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, Any],
    str,
]:
    predictions = read_csv(PREDICTIONS_PATH)
    comparison = read_csv(COMPARISON_PATH)
    pairwise = read_csv(PAIRWISE_PATH)
    validation = read_csv(VALIDATION_PATH)

    require_columns(
        predictions,
        {
            "target_date",
            "method",
            "actual_return",
            "quantile_return",
            "var",
            "violation",
            "config_id",
        },
        PREDICTIONS_PATH,
    )

    require_columns(
        comparison,
        {
            "method",
            "calibration_leader",
            "pinball_leader",
            "lowest_average_var",
        },
        COMPARISON_PATH,
    )

    require_columns(
        pairwise,
        {
            "comparison",
            "left_better_count",
            "right_better_count",
            "mean_delta",
        },
        PAIRWISE_PATH,
    )

    require_columns(
        validation,
        {"check", "status"},
        VALIDATION_PATH,
    )

    with FREEZE_PATH.open("r", encoding="utf-8") as handle:
        freeze = yaml.safe_load(handle)

    if not isinstance(freeze, dict):
        raise ValueError("Model-freeze configuration must be a mapping.")

    day24_doc = DAY24_DOC_PATH.read_text(encoding="utf-8")

    return (
        predictions,
        comparison,
        pairwise,
        validation,
        freeze,
        day24_doc,
    )


def build_traceability() -> list[ClaimRow]:
    (
        predictions,
        comparison,
        pairwise,
        validation,
        freeze,
        day24_doc,
    ) = load_inputs()

    rows: list[ClaimRow] = []

    methods = sorted({row["method"] for row in predictions})
    target_dates = sorted({row["target_date"] for row in predictions})

    method_counts = {
        method: sum(row["method"] == method for row in predictions)
        for method in methods
    }

    duplicate_count = len(predictions) - len(
        {
            (row["method"], row["target_date"])
            for row in predictions
        }
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Duplicate method-target rows prevent traceability: {duplicate_count}"
        )

    violation_mismatches = 0
    var_mismatches = 0

    for row in predictions:
        actual_return = float(row["actual_return"])
        quantile_return = float(row["quantile_return"])
        reported_var = float(row["var"])
        reported_violation = parse_bool(row["violation"])

        derived_violation = actual_return < quantile_return
        derived_var = max(0.0, -quantile_return)

        if reported_violation != derived_violation:
            violation_mismatches += 1

        if abs(reported_var - derived_var) > TOLERANCE:
            var_mismatches += 1

    evaluation = freeze["evaluation"]
    models = freeze["models"]

    add_exact(
        rows,
        "01",
        "evaluation_contract",
        "Canonical final prediction row count is 1,194.",
        1194,
        len(predictions),
        "results/final_predictions.csv",
    )

    add_exact(
        rows,
        "02",
        "evaluation_contract",
        "Canonical evaluation contains exactly three methods.",
        3,
        len(methods),
        "results/final_predictions.csv",
    )

    forecasts_per_method = (
        unique_value(
            [
                {"count": str(method_counts[method])}
                for method in methods
            ],
            "count",
        )
        if methods
        else ""
    )

    add_exact(
        rows,
        "03",
        "evaluation_contract",
        "Each canonical method has 398 forecasts.",
        "398",
        forecasts_per_method,
        "results/final_predictions.csv",
    )

    add_exact(
        rows,
        "04",
        "evaluation_contract",
        "Frozen evaluation starts on 2024-12-18.",
        "2024-12-18",
        target_dates[0] if target_dates else "",
        "results/final_predictions.csv",
    )

    add_exact(
        rows,
        "05",
        "evaluation_contract",
        "Frozen evaluation ends on 2026-07-28.",
        "2026-07-28",
        target_dates[-1] if target_dates else "",
        "results/final_predictions.csv",
    )

    add_exact(
        rows,
        "06",
        "evaluation_contract",
        "Violations use the strict actual_return < quantile_return rule.",
        0,
        violation_mismatches,
        "results/final_predictions.csv",
    )

    add_exact(
        rows,
        "07",
        "evaluation_contract",
        "Reported VaR equals max(0, -quantile_return).",
        0,
        var_mismatches,
        "results/final_predictions.csv",
    )

    config_expectations = {
        "historical_simulation": ("historical_w250", "08"),
        "ewma": ("ewma_d094", "09"),
        "gradient_boosting": ("gb_G04", "10"),
    }

    for method, (expected_config, claim_id) in config_expectations.items():
        method_rows = [
            row for row in predictions
            if row["method"] == method
        ]

        prediction_config = unique_value(method_rows, "config_id")
        freeze_config = models[method]["config_id"]

        actual_config = (
            prediction_config
            if prediction_config == freeze_config
            else f"prediction={prediction_config};freeze={freeze_config}"
        )

        add_exact(
            rows,
            claim_id,
            "frozen_configuration",
            f"{method} uses the frozen canonical configuration.",
            expected_config,
            actual_config,
            "results/final_predictions.csv;configs/model_freeze.yaml",
        )

    add_exact(
        rows,
        "11",
        "aggregate_evidence",
        "EWMA is the violation-rate calibration leader.",
        "ewma",
        unique_leader(comparison, "calibration_leader"),
        "results/final_metric_comparison.csv",
    )

    add_exact(
        rows,
        "12",
        "aggregate_evidence",
        "Gradient Boosting has the lowest mean pinball loss.",
        "gradient_boosting",
        unique_leader(comparison, "pinball_leader"),
        "results/final_metric_comparison.csv",
    )

    add_exact(
        rows,
        "13",
        "aggregate_evidence",
        "Historical Simulation has the lowest average reported VaR.",
        "historical_simulation",
        unique_leader(comparison, "lowest_average_var"),
        "results/final_metric_comparison.csv",
    )

    no_overall_winner = (
        "no single overall winner" in day24_doc.lower()
        and models["gradient_boosting"]["overall_g01_g07_winner_claim"] is False
    )

    add_exact(
        rows,
        "14",
        "aggregate_evidence",
        "The frozen final evidence declares no single overall model winner.",
        "true",
        bool_text(no_overall_winner),
        "docs/day24-final-evidence-a.md;configs/model_freeze.yaml",
    )

    lower_var_boundary = (
        "lower average var must not be interpreted as automatic model superiority"
        in day24_doc.lower()
    )

    add_exact(
        rows,
        "15",
        "aggregate_evidence",
        "Lower average VaR is not treated as automatic model superiority.",
        "true",
        bool_text(lower_var_boundary),
        "docs/day24-final-evidence-a.md",
    )

    pairwise_by_name = {
        row["comparison"]: row
        for row in pairwise
    }

    pairwise_specs = (
        (
            "16",
            "17",
            "gb_vs_historical",
            "151|247",
            -0.0001383332817084946,
        ),
        (
            "18",
            "19",
            "gb_vs_ewma",
            "231|167",
            -0.00021019030172751563,
        ),
        (
            "20",
            "21",
            "ewma_vs_historical",
            "131|267",
            0.00007185702001901683,
        ),
    )

    for count_id, delta_id, comparison_name, expected_counts, expected_delta in pairwise_specs:
        pairwise_row = pairwise_by_name.get(comparison_name)

        if pairwise_row is None:
            actual_counts = "missing"
            actual_delta = float("inf")
        else:
            actual_counts = (
                f'{pairwise_row["left_better_count"]}|'
                f'{pairwise_row["right_better_count"]}'
            )
            actual_delta = float(pairwise_row["mean_delta"])

        add_exact(
            rows,
            count_id,
            "pairwise_evidence",
            f"{comparison_name} daily lower-loss counts match frozen evidence.",
            expected_counts,
            actual_counts,
            "results/final_pairwise_summary.csv",
        )

        add_numeric(
            rows,
            delta_id,
            "pairwise_evidence",
            f"{comparison_name} mean pinball-loss delta matches frozen evidence.",
            expected_delta,
            actual_delta,
            "results/final_pairwise_summary.csv",
        )

    non_pristine = (
        evaluation["pristine_untouched_test_claim"] is False
        and "should not be described as a pristine" in day24_doc.lower()
    )

    add_exact(
        rows,
        "22",
        "release_boundary",
        "The evaluation is not described as a pristine untouched test set.",
        "true",
        bool_text(non_pristine),
        "configs/model_freeze.yaml;docs/day24-final-evidence-a.md",
    )

    provenance_caveat = (
        "provenance incompleteness" in day24_doc.lower()
        and "runtime head" in day24_doc.lower()
    )

    add_exact(
        rows,
        "23",
        "release_boundary",
        "Known final-run metadata provenance incompleteness is documented.",
        "true",
        bool_text(provenance_caveat),
        "docs/day24-final-evidence-a.md",
    )

    validation_passes = sum(
        row["status"] == "PASS"
        for row in validation
    )
    validation_failures = len(validation) - validation_passes
    figure_count = sum(path.is_file() for path in FIGURE_PATHS)

    expected_integrity = "validation=26/26 PASS;figures=3/3"
    actual_integrity = (
        f"validation={validation_passes}/{len(validation)} PASS;"
        f"figures={figure_count}/3"
    )

    if validation_failures:
        actual_integrity += f";failures={validation_failures}"

    add_exact(
        rows,
        "24",
        "release_boundary",
        "Day 24 final evidence package remains internally valid.",
        expected_integrity,
        actual_integrity,
        "results/final_evidence_validation_a.csv;results/figures/*_a.png",
    )

    rows.sort(key=lambda row: row.claim_id)

    if len(rows) != 24:
        raise RuntimeError(
            f"Expected exactly 24 traceability claims, built {len(rows)}."
        )

    claim_ids = [row.claim_id for row in rows]

    if len(claim_ids) != len(set(claim_ids)):
        raise RuntimeError("Duplicate traceability claim IDs detected.")

    return rows


def write_traceability(rows: list[ClaimRow]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=TRACEABILITY_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "claim_id": row.claim_id,
                    "category": row.category,
                    "claim": row.claim,
                    "expected": row.expected,
                    "actual": row.actual,
                    "tolerance": row.tolerance,
                    "status": row.status,
                    "source": row.source,
                }
            )


def main() -> None:
    rows = build_traceability()
    write_traceability(rows)

    pass_count = sum(row.status == "PASS" for row in rows)
    fail_count = len(rows) - pass_count

    print(f"output={OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"rows={len(rows)}")
    print(f"pass={pass_count}")
    print(f"fail={fail_count}")

    if fail_count:
        failed_ids = [
            row.claim_id
            for row in rows
            if row.status != "PASS"
        ]
        raise SystemExit(
            "Traceability validation failed for claim IDs: "
            + ", ".join(failed_ids)
        )


if __name__ == "__main__":
    main()