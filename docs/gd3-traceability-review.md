# GĐ3 Traceability Review

## B-16.2 — Issues / Commits / Experiment Logs Review

### Scope

Member B reviewed GĐ3 repository traceability from 10/08 through
15/08.

The review checks:

- required artifacts actually exist;
- implementation and analysis work is traceable to Git history;
- relevant work was merged;
- open GitHub issues were reviewed;
- conclusions are supported by repository evidence;
- parameter changes are covered by decision records.

### Timeline traceability

| Date | Requirement | Main evidence | Git traceability | Status |
| --- | --- | --- | --- | --- |
| 10/08 | EWMA specification | `docs/ewma-var-spec.md`, `configs/ewma.yaml` | `466faa8`, PR #34 | PASS |
| 11/08 | EWMA implementation | `src/models/ewma_var.py` | `600e821`, PR #26 | PASS |
| 12/08 | Baseline evaluation | `results/ewma_vs_historical.csv` | `9ccbf76`, PR #36 | PASS |
| 13/08 | B01–B06 sensitivity | `results/sensitivity_experiments.csv`, `results/ewma_sensitivity.csv` | `97ba0ce`, `c723dbe`, PR #37 | PASS |
| 14/08 | Leakage / reproducibility | `tests/test_no_lookahead.py`, `docs/leakage-audit.md`, `scripts/run_pipeline.py` | `0bf478a`, `9fa4551`, PR #42 | PASS |
| 15/08 | Baseline error analysis | `results/baseline_error_analysis.csv`, `results/baseline_case_studies.csv`, error-analysis docs and figures | `ede01b4`, `b1f1333`, PR #39/#40 | PASS |

### Artifact review

Required GĐ3 artifacts for the reviewed timeline exist in the
repository.

No required Day-10 through Day-15 artifact was identified as missing
during this review.

### Issue review

GitHub Issue #33:

`Define EWMA VaR design and implementation contract`

was reviewed against its acceptance criteria.

All 10 acceptance criteria were verified against the merged repository
artifacts.

The canonical configuration path is:

`configs/ewma.yaml`

Issue #33 was closed after verification.

Open GitHub issues at this review:

`0`

Blocking GitHub issues:

`0`

### Experiment and decision traceability

Historical sensitivity:

- B01 = window 125
- B02 = window 250
- B03 = window 500

Canonical Historical baseline:

`window = 250`

EWMA sensitivity:

- B04 = lambda 0.90
- B05 = lambda 0.94
- B06 = lambda 0.97

Day-13 validation-selected EWMA candidate:

`lambda = 0.90`

Canonical/production EWMA baseline:

`lambda = 0.94`

The distinction between the validation-selected candidate and the
canonical production parameter is documented in the baseline decision
record.

No undocumented baseline parameter change was identified.

### Evidence review

The reviewed baseline conclusions are supported by committed result,
test, audit, or decision artifacts.

No blocking conclusion without repository evidence was identified.

The Day-13 decision record also documents the limitation that the later
398-row segment was excluded from parameter selection but must not be
described as a pristine untouched test set.

### Final status

Artifacts present: `PASS`

Commit/PR traceability: `PASS`

Open issue review: `PASS`

Evidence-backed conclusions: `PASS`

Parameter decision records: `PASS`

B-16.2 Review issues / commits / experiment logs: `PASS`

Blocking findings: `NONE`