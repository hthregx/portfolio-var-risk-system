from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

EXPECTED_METHODS = (
    "historical_simulation",
    "ewma",
    "gradient_boosting",
)
EXPECTED_TARGETS_PER_METHOD = 398
EXPECTED_PREDICTION_ROWS = 1194
EXPECTED_CONFIDENCE = 0.95
EXPECTED_ALPHA = 0.05
TOL = 1e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise AssertionError(f"Invalid boolean value: {value!r}")
    return normalized == "true"


def pinball_loss(actual: float, quantile: float, alpha: float) -> float:
    # Independent implementation; deliberately does not import project metric code.
    error = actual - quantile
    if error >= 0.0:
        return alpha * error
    return (1.0 - alpha) * (-error)


def approx_equal(left: float, right: float, tol: float = TOL) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tol)


def audit(root: Path) -> dict[str, dict[str, float | int]]:
    predictions_path = root / "results" / "final_predictions.csv"
    metrics_path = root / "results" / "final_metrics.csv"
    metadata_path = root / "results" / "final_run_metadata.json"

    for path in (predictions_path, metrics_path, metadata_path):
        assert path.is_file(), f"Missing canonical artifact: {path}"

    predictions = read_csv(predictions_path)
    canonical_metrics = read_csv(metrics_path)
    metadata = read_json(metadata_path)

    # Contract-level checks.
    assert len(predictions) == EXPECTED_PREDICTION_ROWS
    assert metadata["artifacts"]["prediction_rows"] == EXPECTED_PREDICTION_ROWS
    assert metadata["artifacts"]["metric_rows"] == 3
    assert metadata["evaluation"]["target_count"] == EXPECTED_TARGETS_PER_METHOD
    assert approx_equal(float(metadata["contract"]["confidence_level"]), EXPECTED_CONFIDENCE)
    assert approx_equal(float(metadata["contract"]["alpha"]), EXPECTED_ALPHA)
    assert int(metadata["contract"]["forecast_horizon"]) == 1
    assert metadata["contract"]["violation_rule"] == "actual_return < quantile_return"
    assert metadata["contract"]["var_rule"] == "max(0, -quantile_return)"

    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_target: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in predictions:
        by_method[row["method"]].append(row)
        by_target[row["target_date"]].append(row)

    assert set(by_method) == set(EXPECTED_METHODS)
    assert len(by_method) == 3
    assert len(by_target) == EXPECTED_TARGETS_PER_METHOD

    common_target_set = set(by_target)
    for method in EXPECTED_METHODS:
        rows = by_method[method]
        assert len(rows) == EXPECTED_TARGETS_PER_METHOD
        assert {row["target_date"] for row in rows} == common_target_set

    # Same realized target return across all three models on each target date.
    for target_date, rows in by_target.items():
        assert len(rows) == 3, f"Expected 3 model rows for {target_date}"
        actuals = [float(row["actual_return"]) for row in rows]
        assert max(actuals) - min(actuals) <= TOL, (
            f"actual_return mismatch across methods on {target_date}: {actuals}"
        )

    recomputed: dict[str, dict[str, float | int]] = {}

    for method in EXPECTED_METHODS:
        rows = by_method[method]
        violations = 0
        losses: list[float] = []
        vars_: list[float] = []

        for row in rows:
            actual = float(row["actual_return"])
            quantile = float(row["quantile_return"])
            stored_var = float(row["var"])
            stored_violation = parse_bool(row["violation"])

            expected_violation = actual < quantile
            expected_var = max(0.0, -quantile)

            assert stored_violation == expected_violation
            assert approx_equal(stored_var, expected_var)

            violations += int(expected_violation)
            losses.append(pinball_loss(actual, quantile, EXPECTED_ALPHA))
            vars_.append(expected_var)

        count = len(rows)
        recomputed[method] = {
            "forecast_count": count,
            "violation_count": violations,
            "violation_rate": violations / count,
            "pinball_loss": sum(losses) / count,
            "average_var": sum(vars_) / count,
        }

    canonical_by_method = {row["method"]: row for row in canonical_metrics}
    assert set(canonical_by_method) == set(EXPECTED_METHODS)

    numeric_fields = (
        "violation_rate",
        "pinball_loss",
        "average_var",
    )

    for method, values in recomputed.items():
        canonical = canonical_by_method[method]
        assert int(canonical["forecast_count"]) == values["forecast_count"]
        assert int(canonical["violation_count"]) == values["violation_count"]
        for field in numeric_fields:
            assert approx_equal(float(canonical[field]), float(values[field])), (
                f"{method} {field} mismatch: recomputed={values[field]} "
                f"canonical={canonical[field]}"
            )

    nominal_violation_rate = 1.0 - EXPECTED_CONFIDENCE
    closest_to_nominal = min(
        EXPECTED_METHODS,
        key=lambda method: abs(
            float(recomputed[method]["violation_rate"]) - nominal_violation_rate
        ),
    )
    lowest_pinball = min(
        EXPECTED_METHODS,
        key=lambda method: float(recomputed[method]["pinball_loss"]),
    )
    lowest_average_var = min(
        EXPECTED_METHODS,
        key=lambda method: float(recomputed[method]["average_var"]),
    )

    assert closest_to_nominal == "ewma"
    assert lowest_pinball == "gradient_boosting"
    assert lowest_average_var == "historical_simulation"
    assert len({closest_to_nominal, lowest_pinball, lowest_average_var}) > 1

    print("PASS - B28.1 FROZEN QUANTITATIVE BASELINE AUDIT")
    print(f"Models: {len(EXPECTED_METHODS)}")
    print(f"Targets/model: {EXPECTED_TARGETS_PER_METHOD}")
    print(f"Prediction rows: {len(predictions)}")
    print(f"Confidence: {EXPECTED_CONFIDENCE:.0%}")
    print()
    print(
        f"{'method':24s} {'N':>5s} {'viol':>5s} "
        f"{'viol_rate':>12s} {'pinball':>12s} {'avg_var':>12s}"
    )
    for method in EXPECTED_METHODS:
        row = recomputed[method]
        print(
            f"{method:24s} "
            f"{int(row['forecast_count']):5d} "
            f"{int(row['violation_count']):5d} "
            f"{float(row['violation_rate']):12.9f} "
            f"{float(row['pinball_loss']):12.9f} "
            f"{float(row['average_var']):12.9f}"
        )

    print()
    print("[PASS] EWMA closest to 5%")
    print("[PASS] Gradient Boosting lowest Pinball Loss")
    print("[PASS] Historical Simulation lowest Average VaR")
    print("[PASS] No overall winner across the three stated criteria")

    return recomputed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent Day28 B28.1 audit of frozen canonical VaR metrics."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of scripts/.",
    )
    args = parser.parse_args()
    audit(args.root.resolve())


if __name__ == "__main__":
    main()
