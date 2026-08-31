# Weekly Note #3 — Baseline VaR Phase Closeout

Planned milestone date: 16/08/2026.

Repository closeout record completed on 17/08/2026.

## Scope

This weekly note records the closeout status of the baseline VaR phase
covering Historical Simulation, EWMA, common walk-forward evaluation,
baseline comparison, sensitivity analysis, leakage controls, error
analysis, and methodology documentation.

## What Was Completed

- Historical Simulation baseline implementation and evaluation.
- EWMA baseline specification, implementation, and evaluation.
- Common walk-forward evaluation for both baseline models.
- Same-date Historical Simulation versus EWMA comparison.
- B01-B06 EWMA sensitivity analysis and canonical-baseline decision.
- Leakage and reproducibility review.
- Baseline error analysis and case-study review.
- Code-to-report consistency review.
- GĐ3 artifact and repository traceability review.
- Data/methodology draft review and freeze.

## Canonical Historical Baseline

Historical Simulation is the canonical rolling-window baseline with:

- rolling window = 250 observations;
- alpha = 0.05;
- one-day-ahead forecasting;
- strict violation rule: realized return < forecast quantile;
- VaR magnitude = max(0, -q).

Status: `COMPLETE`

## Canonical EWMA Baseline

EWMA VaR is implemented with:

- canonical lambda = 0.94;
- alpha = 0.05;
- zero conditional mean;
- standard Normal innovations;
- first-squared-return variance initialization;
- one-day-ahead forecasting;
- strict violation rule: realized return < forecast quantile;
- VaR magnitude = max(0, -q).

Status: `COMPLETE`

## Common Evaluation Framework

Historical Simulation and EWMA use the common target
`portfolio_simple_return` and one-day forecasting horizon.

The common walk-forward framework supports rolling and expanding
evaluation while enforcing forecast-date and target-date ordering.

Both baseline methods are evaluated on the same target-date sequence.

The common evaluation set includes:

- Violation Rate;
- Pinball Loss at alpha 0.05;
- Average VaR;
- exception-day analysis.

## Day-12 — Baseline Comparison

Historical Simulation and EWMA were compared using the same canonical
target-date sequence and common evaluation framework.

Primary comparison evidence:

- `results/ewma_vs_historical.csv`

This ensures that differences between the baseline models are not caused
by different evaluation dates.

Status: `COMPLETE`

## Day-13 — Sensitivity Decision

The B01-B06 sensitivity experiments were reviewed.

Primary evidence:

- `results/sensitivity_experiments.csv`
- `results/ewma_sensitivity.csv`
- `docs/baseline-sensitivity-decision.md`
- `docs/baseline-decision-log.md`

Historical Simulation retains the 250-observation rolling window.

EWMA retains lambda = 0.94 as the canonical production baseline.

EWMA lambda = 0.90 is recorded as the strongest validation candidate in
the sensitivity review, but it does not automatically replace the
canonical production setting.

Status: `COMPLETE`

## Day-14 — Leakage and Reproducibility

Date, target, and future-perturbation leakage checks have been
documented.

Primary evidence:

- `tests/test_no_lookahead.py`
- `docs/date-target-leakage-audit.md`
- `docs/leakage-audit.md`
- `scripts/run_pipeline.py`

The walk-forward contract preserves forecast-date and target-date
ordering so that target-date observations are not used to construct
their own forecasts.

The Chapter 2 documentation was validated against the production model
and walk-forward contracts.

The full automated test suite passed with 130 tests.

Status: `COMPLETE`

## Day-15 — Error Analysis

Baseline error analysis and case-study artifacts were reviewed.

Primary evidence:

- `results/baseline_error_analysis.csv`
- `results/baseline_case_studies.csv`
- `docs/baseline-error-analysis-review.md`
- `docs/baseline-case-study-notes.md`
- `figures/baseline_error_analysis/*`

The analysis covers:

- baseline VaR thresholds versus realized returns;
- exception timing;
- exception severity;
- exception clustering;
- selected case-study risk responses.

The Day-15 figures are present and their references were reviewed during
the B-16.3 data/method freeze.

Status: `COMPLETE`

## Canonical Baseline Decisions

- Historical Simulation retains the 250-observation rolling window.
- EWMA retains decay 0.94 as the canonical production baseline.
- EWMA decay 0.90 is recorded as the strongest validation candidate in
  the sensitivity review, but it does not automatically replace the
  canonical production setting.
- The later evaluation subset is not described as a pristine untouched
  test set because the broader evaluation period had already been
  inspected during the earlier baseline comparison.

## Methodology Report Status

Chapter 2 methodology has been drafted in
`docs/chapter-2-methodology.md` and covers:

- portfolio return and common forecast target;
- Historical Simulation methodology;
- EWMA methodology;
- walk-forward backtesting;
- evaluation metrics;
- baseline parameter-selection policy;
- no-look-ahead and reproducibility controls;
- implementation traceability and prompt provenance.

B-16.3 review confirmed:

- Data chapter draft stable: `YES`
- Baseline methodology stable: `YES`
- Metrics definitions stable: `YES`
- Figures referenced correctly: `YES`
- No unresolved blocking issue: `YES`

Shared report ownership remains:

`Member A edits -> Member B reviews`

Only one member should edit the shared report at a time.

## Traceability

Primary baseline artifacts include:

- `src/models/historical_var.py`;
- `src/models/ewma_var.py`;
- `src/backtesting/walk_forward.py`;
- `configs/ewma.yaml`;
- `docs/chapter-2-methodology.md`;
- `docs/historical-var-stability.md`;
- `docs/ewma-evaluation.md`;
- `docs/baseline-sensitivity-decision.md`;
- `docs/baseline-decision-log.md`;
- `docs/date-target-leakage-audit.md`;
- `docs/leakage-audit.md`;
- `docs/baseline-error-analysis-review.md`;
- `docs/baseline-case-study-notes.md`.

GĐ3 traceability review covered:

- 10/08 — EWMA specification;
- 11/08 — EWMA implementation;
- 12/08 — baseline evaluation;
- 13/08 — B01-B06 sensitivity;
- 14/08 — leakage/reproducibility;
- 15/08 — baseline error analysis.

Status: `PASS`

## Known Limitations

- Historical Simulation uses a fixed 250-observation rolling window and
  therefore may react slowly to changes in volatility.
- EWMA responds more directly to recent volatility but depends on its
  decay parameter and Normal-distribution assumption.
- Both approaches remain baseline VaR models and may not fully represent
  extreme tail behaviour.
- Results depend on the available portfolio data and the implemented
  walk-forward evaluation period.
- Sensitivity experiments are diagnostic and do not replace the
  canonical EWMA lambda = 0.94 setting.
- The later evaluation subset is not treated as a pristine untouched
  test set because the broader evaluation period had previously been
  inspected.

## Open Issues

GitHub issue #33,
`Define EWMA VaR design and implementation contract`, was reviewed
against its acceptance criteria.

All acceptance criteria were checked and issue #33 was closed.

Open blocking GitHub issues at this review: `NONE`

## Readiness for Gradient Boosting Phase

The baseline phase now provides:

- canonical Historical Simulation benchmark;
- canonical EWMA benchmark;
- common target and one-day-ahead forecasting contract;
- same-date comparison framework;
- sensitivity-analysis evidence;
- leakage and reproducibility controls;
- baseline error-analysis evidence;
- reviewed data/methodology draft.

Gradient Boosting Quantile Regression can therefore proceed using the
same target and walk-forward evaluation framework.

Final model comparison must wait until Gradient Boosting is evaluated
under that common framework.

Readiness for Gradient Boosting phase: `YES`

## Remaining Work

- Independent review of the methodology/report draft remains a separate
  review responsibility.
- Gradient Boosting Quantile Regression begins in the next modeling
  phase.
- Final model comparison must wait until Gradient Boosting is evaluated
  on the same target and backtesting framework.

## GĐ3 Exit Review

| Exit criterion | Status |
| --- | --- |
| EWMA runs on same test dates | YES |
| Baseline comparison complete | YES |
| Error analysis complete | YES |
| Leakage audit complete | YES |
| Reproducibility infrastructure | YES |
| Report 45-50% | YES |
| Blocking issues | NONE |

GĐ3 must not be declared complete while any required exit criterion
remains `NO` or unconfirmed.

## Schedule Note

The methodology deliverable belongs to the 16/08 milestone. Its
repository commit was created on 17/08/2026, so the technical deliverable
is complete but its Git traceability is one day later than the planned
milestone.

The timeline target for 16/08 is approximately 45-50% report completion.

## Final Status

B-16.1 Code / report consistency review: `PASS`

B-16.2 Issues / commits / experiment logs review: `PASS`

B-16.3 Data / method draft freeze: `PASS`

B-16.4 Weekly note #3: `COMPLETE`

GĐ3 exit status: `PASS`