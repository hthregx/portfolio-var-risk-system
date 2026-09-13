from pathlib import Path
import subprocess
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PRED = pd.read_csv(ROOT / "results/final_predictions.csv")
COMP = pd.read_csv(ROOT / "results/final_metric_comparison.csv")
PAIR = pd.read_csv(ROOT / "results/final_pairwise_summary.csv")
FREEZE = yaml.safe_load((ROOT / "configs/model_freeze.yaml").read_text())

DAY24 = (ROOT / "docs/day24-final-evidence-a.md").read_text().lower()
DAY25 = (ROOT / "docs/day25-final-quantitative-story-a.md").read_text().lower()

FACTS = ROOT / "results/final_defense_facts_b.csv"
MANIFEST = ROOT / "results/final_submission_manifest_b.csv"
TOL = 1e-12


def same(a, b):
    a, b = str(a).split("|"), str(b).split("|")
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        try:
            if abs(float(x) - float(y)) <= TOL:
                continue
        except ValueError:
            pass
        return False
    return True


rows = []


def add(i, topic, fact, expected, actual, source):
    rows.append({
        "fact_id": i,
        "topic": topic,
        "fact": fact,
        "expected": str(expected),
        "actual": str(actual),
        "tolerance": "1e-12",
        "status": "PASS" if same(expected, actual) else "FAIL",
        "source": source,
    })


count = PRED.groupby("method").size()
m = COMP.set_index("method")
p = PAIR.set_index("comparison")
models = FREEZE["models"]
ev = FREEZE["evaluation"]

add("01", "scope", "portfolio", "true",
    str("equal-weight portfolio of hpg, fpt, and mwg" in DAY25).lower(),
    "docs/day25-final-quantitative-story-a.md")

add("02", "scope", "horizon", 1, ev["forecast_horizon_trading_days"],
    "configs/model_freeze.yaml")
add("03", "scope", "confidence/alpha", "0.95|0.05",
    f'{ev["confidence_level"]}|{ev["alpha"]}', "configs/model_freeze.yaml")
add("04", "evaluation", "dates", "2024-12-18|2026-07-28",
    f'{PRED.target_date.min()}|{PRED.target_date.max()}', "results/final_predictions.csv")
add("05", "evaluation", "targets/method", "398|398|398",
    f'{count["historical_simulation"]}|{count["ewma"]}|{count["gradient_boosting"]}',
    "results/final_predictions.csv")
add("06", "evaluation", "predictions", 1194, len(PRED), "results/final_predictions.csv")

for i, method, cfg in [
    ("07", "historical_simulation", "historical_w250"),
    ("08", "ewma", "ewma_d094"),
    ("09", "gradient_boosting", "gb_G04"),
]:
    add(i, "config", method, cfg, models[method]["config_id"], "configs/model_freeze.yaml")

for i, method, n, rate in [
    ("10", "historical_simulation", 27, 0.0678391959798995),
    ("11", "ewma", 22, 0.05527638190954774),
    ("12", "gradient_boosting", 24, 0.06030150753768844),
]:
    r = m.loc[method]
    add(i, "metrics", method, f"{n}|{rate}",
        f'{int(r.violation_count)}|{float(r.violation_rate)}',
        "results/final_metric_comparison.csv")

for i, method, value in [
    ("13", "historical_simulation", 0.0018964642326659775),
    ("14", "ewma", 0.0019683212526850112),
    ("15", "gradient_boosting", 0.001758130950957487),
]:
    add(i, "pinball", method, value, float(m.loc[method, "pinball_loss"]),
        "results/final_metric_comparison.csv")

leaders = "|".join([
    COMP.loc[COMP.calibration_distance.idxmin(), "method"],
    COMP.loc[COMP.pinball_loss.idxmin(), "method"],
    COMP.loc[COMP.average_var.idxmin(), "method"],
])
add("16", "leaders", "criterion leaders",
    "ewma|gradient_boosting|historical_simulation",
    leaders, "results/final_metric_comparison.csv")

for i, name, left, right, delta in [
    ("17", "gb_vs_historical", 151, 247, -0.0001383332817084946),
    ("18", "gb_vs_ewma", 231, 167, -0.00021019030172751563),
    ("19", "ewma_vs_historical", 131, 267, 0.00007185702001901683),
]:
    r = p.loc[name]
    actual = f'{int(r.left_better_count)}|{int(r.right_better_count)}|{float(r.mean_delta)}'
    add(i, "pairwise", name, f"{left}|{right}|{delta}", actual,
        "results/final_pairwise_summary.csv")

guard = (
    "there is no single overall winner" in DAY24
    and "pristine, never-inspected test set" in DAY24
    and "provenance incompleteness" in DAY24
)
add("20", "boundary", "release guardrails", "true", str(guard).lower(),
    "docs/day24-final-evidence-a.md")

facts = pd.DataFrame(rows)
facts.to_csv(FACTS, index=False, lineterminator="\n")


ARTIFACTS = [
    "configs/final_evaluation.yaml", "configs/model_freeze.yaml",
    "results/final_predictions.csv", "results/final_metrics.csv",
    "results/final_run_metadata.json", "results/final_metric_comparison.csv",
    "results/final_pairwise_diagnostics.csv", "results/final_pairwise_summary.csv",
    "scripts/build_final_evidence_a.py", "results/final_evidence_validation_a.csv",
    "docs/day24-final-evidence-a.md", "tests/test_final_evidence_a.py",
    "results/figures/final_violation_rate_a.png",
    "results/figures/final_pinball_loss_a.png",
    "results/figures/final_average_var_a.png",
    "scripts/build_final_traceability_a.py",
    "results/final_claim_traceability_a.csv",
    "docs/day25-final-quantitative-story-a.md",
    "tests/test_final_traceability_a.py",
    "scripts/build_final_presentation_evidence_b.py",
    "results/final_presentation_evidence_b.csv",
    "docs/day26-final-presentation-qa-b.md",
    "tests/test_final_presentation_evidence_b.py",
]

out = []

for i, path in enumerate(ARTIFACTS, 1):
    r = subprocess.run(
        ["git", "rev-parse", f"HEAD:{path}"],
        cwd=ROOT, capture_output=True, text=True
    )
    tracked = r.returncode == 0
    blob = r.stdout.strip() if tracked else ""
    f = ROOT / path

    out.append({
        "artifact_id": f"{i:02d}",
        "category": "frozen_canonical" if i <= 8 else
                    "day24_a" if i <= 15 else
                    "day25_a" if i <= 19 else "day26_b",
        "path": path,
        "required": "true",
        "tracked": str(tracked).lower(),
        "git_blob_sha": blob,
        "status": "PASS" if f.is_file() and f.stat().st_size > 0 and blob else "FAIL",
    })

manifest = pd.DataFrame(out)
manifest.to_csv(MANIFEST, index=False, lineterminator="\n")

print("facts:", len(facts), "PASS:", sum(facts.status == "PASS"))
print("manifest:", len(manifest), "PASS:", sum(manifest.status == "PASS"))

if len(facts) != 20 or not facts.status.eq("PASS").all():
    raise SystemExit("Defense facts failed")

if len(manifest) != 23 or not manifest.status.eq("PASS").all():
    raise SystemExit("Manifest failed")

print("B27_SUBMISSION_READINESS_PASS")