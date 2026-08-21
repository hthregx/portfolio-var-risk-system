# GĐ3 Code–Report Consistency Review

## B-16.1 — Code ↔ Report Consistency Review

### Scope

Member B reviewed the baseline methodology/report written by Member A
against the implemented Historical Simulation and EWMA code.

The purpose of this review is to ensure that the report describes the
actual production baseline configuration and evaluation semantics.

### Evidence reviewed

Code and configuration:

- `src/models/historical_var.py`
- `src/models/ewma_var.py`
- `src/backtesting/walk_forward.py`
- `configs/ewma.yaml`

Report and audit evidence:

- `docs/chapter-2-methodology.md`
- `docs/report-data.md`
- `docs/leakage-audit.md`
- `results/ewma_vs_historical.csv`

### Consistency checklist

| Requirement | Code ↔ report result | Status |
| --- | --- | --- |
| Historical window = 250 | Consistent | PASS |
| alpha = 0.05 | Consistent | PASS |
| EWMA canonical lambda = 0.94 | Consistent | PASS |
| EWMA conditional mean = 0 | Consistent | PASS |
| EWMA uses Normal quantile | Consistent | PASS |
| EWMA sigma initialization = first squared return | Consistent | PASS |
| Violation rule uses strict `<` | Consistent | PASS |
| VaR = `max(0, -q)` | Consistent | PASS |
| Historical/EWMA same-date comparison | Consistent | PASS |
| No-look-ahead semantics | Consistent | PASS |

### Date and leakage semantics

For every one-day-ahead forecast:

`forecast_date < target_date`

Training observations satisfy:

`training_date <= forecast_date`

and:

`training_date < target_date`

Historical Simulation and EWMA are compared on the same target dates
and use the same realized `actual_return` for each target date.

### Review finding

No blocking inconsistency was identified between the reviewed
Historical/EWMA implementation and Member A's baseline methodology
draft.

All required B-16.1 consistency checks pass.

If any report statement differs from the production code or
configuration in a future revision, the report must not be merged
until the inconsistency is reconciled.

### Final status

B-16.1 Code ↔ report consistency review: `PASS`

Blocking inconsistencies: `NONE`