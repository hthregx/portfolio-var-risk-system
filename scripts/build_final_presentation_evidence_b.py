from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]

FREEZE = ROOT / "configs/model_freeze.yaml"
PRED = ROOT / "results/final_predictions.csv"
COMP = ROOT / "results/final_metric_comparison.csv"
PAIR = ROOT / "results/final_pairwise_summary.csv"
OUT = ROOT / "results/final_presentation_evidence_b.csv"

TOL = 1e-12

METHODS = [
    "historical_simulation",
    "ewma",
    "gradient_boosting",
]


def add(rows, eid, section, claim, expected, actual, ok, source):
    rows.append(
        {
            "evidence_id": eid,
            "section": section,
            "claim": claim,
            "expected": str(expected),
            "actual": str(actual),
            "tolerance": "1e-12",
            "status": "PASS" if ok else "FAIL",
            "source": source,
        }
    )


def main():
    pred = pd.read_csv(PRED)
    comp = pd.read_csv(COMP)
    pair = pd.read_csv(PAIR)

    freeze = yaml.safe_load(FREEZE.read_text(encoding="utf-8"))

    rows = []

    counts = pred.groupby("method").size().to_dict()
    dates = sorted(pred["target_date"].unique())

    expected_violation = (
        pred["actual_return"] < pred["quantile_return"]
    )

    actual_violation = (
        pred["violation"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

    var_expected = (-pred["quantile_return"]).clip(lower=0)

    add(rows, "01", "panel", "total canonical predictions",
        1194, len(pred), len(pred) == 1194, "results/final_predictions.csv")

    add(rows, "02", "panel", "method count",
        3, pred["method"].nunique(),
        pred["method"].nunique() == 3,
        "results/final_predictions.csv")

    add(rows, "03", "panel", "forecasts per method",
        398, counts,
        all(counts.get(m) == 398 for m in METHODS),
        "results/final_predictions.csv")

    add(rows, "04", "panel", "evaluation start",
        "2024-12-18", dates[0],
        dates[0] == "2024-12-18",
        "results/final_predictions.csv")

    add(rows, "05", "panel", "evaluation end",
        "2026-07-28", dates[-1],
        dates[-1] == "2026-07-28",
        "results/final_predictions.csv")

    add(rows, "06", "semantics", "strict violation rule",
        "actual_return < quantile_return",
        bool(actual_violation.equals(expected_violation)),
        bool(actual_violation.equals(expected_violation)),
        "results/final_predictions.csv")

    var_ok = bool(
        ((pred["var"] - var_expected).abs() <= TOL).all()
    )

    add(rows, "07", "semantics", "VaR identity",
        "max(0, -quantile_return)",
        var_ok, var_ok,
        "results/final_predictions.csv")

    configs = {
        m: sorted(
            pred.loc[pred["method"] == m, "config_id"].unique()
        )
        for m in METHODS
    }

    add(rows, "08", "config", "Historical config",
        "historical_w250",
        configs["historical_simulation"][0],
        configs["historical_simulation"] == ["historical_w250"],
        "results/final_predictions.csv")

    add(rows, "09", "config", "EWMA config",
        "ewma_d094",
        configs["ewma"][0],
        configs["ewma"] == ["ewma_d094"],
        "results/final_predictions.csv")

    add(rows, "10", "config", "GB config",
        "gb_G04",
        configs["gradient_boosting"][0],
        configs["gradient_boosting"] == ["gb_G04"],
        "results/final_predictions.csv")

    by_method = comp.set_index("method")

    metrics = {
        "historical_simulation": {
            "violations": 27,
            "violation_rate": 0.0678391959798995,
            "pinball_loss": 0.0018964642326659775,
            "average_var": 0.02133018927920153,
        },
        "ewma": {
            "violations": 22,
            "violation_rate": 0.05527638190954774,
            "pinball_loss": 0.0019683212526850112,
            "average_var": 0.02508404248953606,
        },
        "gradient_boosting": {
            "violations": 24,
            "violation_rate": 0.06030150753768844,
            "pinball_loss": 0.001758130950957487,
            "average_var": 0.023349057036283743,
        },
    }

    eid = 11

    for method in METHODS:
        expected = metrics[method]["violations"]
        actual = int(by_method.loc[method, "violation_count"])
        add(rows, f"{eid:02d}", "metrics",
            f"{method} violations",
            expected, actual, actual == expected,
            "results/final_metric_comparison.csv")
        eid += 1

    for method in METHODS:
        expected = metrics[method]["violation_rate"]
        actual = float(by_method.loc[method, "violation_rate"])
        add(rows, f"{eid:02d}", "metrics",
            f"{method} violation rate",
            expected, actual,
            abs(actual - expected) <= TOL,
            "results/final_metric_comparison.csv")
        eid += 1

    for method in METHODS:
        expected = metrics[method]["pinball_loss"]
        actual = float(by_method.loc[method, "pinball_loss"])
        add(rows, f"{eid:02d}", "metrics",
            f"{method} pinball loss",
            expected, actual,
            abs(actual - expected) <= TOL,
            "results/final_metric_comparison.csv")
        eid += 1

    for method in METHODS:
        expected = metrics[method]["average_var"]
        actual = float(by_method.loc[method, "average_var"])
        add(rows, f"{eid:02d}", "metrics",
            f"{method} average VaR",
            expected, actual,
            abs(actual - expected) <= TOL,
            "results/final_metric_comparison.csv")
        eid += 1

    calibration_leader = comp.loc[
        comp["calibration_leader"].astype(str).str.lower() == "true",
        "method",
    ].tolist()

    pinball_leader = comp.loc[
        comp["pinball_leader"].astype(str).str.lower() == "true",
        "method",
    ].tolist()

    lowest_var = comp.loc[
        comp["lowest_average_var"].astype(str).str.lower() == "true",
        "method",
    ].tolist()

    add(rows, "23", "leaders", "calibration leader",
        "ewma",
        calibration_leader[0],
        calibration_leader == ["ewma"],
        "results/final_metric_comparison.csv")

    add(rows, "24", "leaders", "pinball leader",
        "gradient_boosting",
        pinball_leader[0],
        pinball_leader == ["gradient_boosting"],
        "results/final_metric_comparison.csv")

    add(rows, "25", "leaders", "lowest average VaR",
        "historical_simulation",
        lowest_var[0],
        lowest_var == ["historical_simulation"],
        "results/final_metric_comparison.csv")

    pair = pair.set_index("comparison")

    expected_pairs = {
        "gb_vs_historical": (
            151, 247, -0.0001383332817084946
        ),
        "gb_vs_ewma": (
            231, 167, -0.00021019030172751563
        ),
        "ewma_vs_historical": (
            131, 267, 0.00007185702001901683
        ),
    }

    for eid, name in zip(("26", "27", "28"), expected_pairs):
        left, right, delta = expected_pairs[name]
        row = pair.loc[name]

        actual = (
            int(row["left_better_count"]),
            int(row["right_better_count"]),
            float(row["mean_delta"]),
        )

        ok = (
            actual[0] == left
            and actual[1] == right
            and abs(actual[2] - delta) <= TOL
        )

        add(
            rows,
            eid,
            "pairwise",
            name,
            f"{left}|{right}|{delta}",
            f"{actual[0]}|{actual[1]}|{actual[2]}",
            ok,
            "results/final_pairwise_summary.csv",
        )

    assert len(rows) == 28
    assert [r["evidence_id"] for r in rows] == [
        f"{i:02d}" for i in range(1, 29)
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)

    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "evidence_id",
                "section",
                "claim",
                "expected",
                "actual",
                "tolerance",
                "status",
                "source",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    failed = [r for r in rows if r["status"] == "FAIL"]

    print("rows:", len(rows))
    print("PASS:", len(rows) - len(failed))
    print("FAIL:", len(failed))

    if failed:
        for row in failed:
            print("FAIL:", row["evidence_id"], row["claim"])
        raise SystemExit(1)

    print("B26_PRESENTATION_EVIDENCE_PASS")


if __name__ == "__main__":
    main()