# Weekly Note #3 — Baseline VaR Phase Closeout

Planned milestone date: 16/08/2026.
Repository closeout record completed on 17/08/2026.

## Scope

This weekly note records the closeout status of the baseline VaR phase covering Historical Simulation, EWMA, common walk-forward evaluation, baseline comparison, sensitivity analysis, leakage controls, and methodology documentation.

## Completed Baseline Work

- Historical Simulation is implemented as the canonical rolling-window baseline with window size 250 and alpha 0.05.
- EWMA VaR is implemented with canonical decay 0.94, zero conditional mean, standard Normal innovations, and first-squared-return variance initialization.
- Historical Simulation and EWMA use the common target `portfolio_simple_return` and one-day forecasting horizon.
- The common walk-forward framework supports rolling and expanding evaluation while enforcing forecast-date and target-date ordering.
- Both baseline methods are evaluated on the same target-date sequence.
- The common evaluation set includes Violation Rate, Pinball Loss at alpha 0.05, Average VaR, and exception-day analysis.
- Date, target, and future-perturbation leakage checks have been documented.
- Baseline error analysis and case-study artifacts are available for downstream reporting.

## Canonical Baseline Decisions

- Historical Simulation retains the 250-observation rolling window.
- EWMA retains decay 0.94 as the canonical production baseline.
- EWMA decay 0.90 is recorded as the strongest validation candidate in the sensitivity review, but it does not automatically replace the canonical production setting.
- The later evaluation subset is not described as a pristine untouched test set because the broader evaluation period had already been inspected during the earlier baseline comparison.

## Methodology Report Status

Chapter 2 methodology has been drafted in `docs/chapter-2-methodology.md` and covers:

- portfolio return and common forecast target;
- Historical Simulation methodology;
- EWMA methodology;
- walk-forward backtesting;
- evaluation metrics;
- baseline parameter-selection policy;
- no-look-ahead and reproducibility controls;
- implementation traceability and prompt provenance.

The Chapter 2 documentation was validated against the production model and walk-forward contracts. The full automated test suite passed with 130 tests.

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

## Remaining Work

- Independent review of the methodology/report draft remains a separate review responsibility.
- Gradient Boosting Quantile Regression begins in the next modeling phase.
- Final model comparison must wait until Gradient Boosting is evaluated on the same target and backtesting framework.

## Schedule Note

The methodology deliverable belongs to the 16/08 milestone. Its repository commit was created on 17/08/2026, so the technical deliverable is complete but its Git traceability is one day later than the planned milestone.
