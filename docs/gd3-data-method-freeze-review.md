# GĐ3 Data and Method Draft Freeze Review

## B-16.3 — Lock Data / Method Draft

### Scope

Member B reviewed the current data and baseline methodology draft for
consistency with the implemented and validated GĐ3 baseline artifacts.

This is a review/freeze step. Member A remains the owner of the
methodology text; Member B reviews the draft and records corrections
rather than rewriting A's work.

### Data chapter

The data chapter was reviewed against the project data documentation,
validation rules, and baseline evaluation inputs.

The definitions of the portfolio data, returns, forecast horizon, and
target alignment are consistent with the reviewed implementation.

Data chapter draft stable: `YES`

### Baseline methodology

The baseline methodology was reviewed against the canonical Historical
Simulation and EWMA implementations.

The reviewed methodology is consistent with:

- Historical rolling window = 250;
- alpha = 0.05;
- canonical EWMA lambda = 0.94;
- zero conditional mean;
- Normal quantile;
- documented EWMA variance initialization;
- strict violation rule: `actual_return < quantile_return`;
- VaR magnitude: `max(0, -quantile_return)`;
- same-date baseline comparison;
- no-look-ahead walk-forward evaluation.

Baseline methodology stable: `YES`

### Metrics definitions

The baseline evaluation metrics and their interpretation were reviewed
against the evaluation outputs and implementation.

The reviewed definitions include violation count, violation rate,
pinball loss, VaR magnitude, and exception analysis.

Metrics definitions stable: `YES`

### Figure references

The standardized baseline error-analysis figure set was reviewed:

1. `figures/baseline_error_analysis/01_baseline_var_vs_returns.png`
2. `figures/baseline_error_analysis/02_exception_timeline.png`
3. `figures/baseline_error_analysis/03_exception_severity.png`
4. `figures/baseline_error_analysis/04_exception_clusters.png`
5. `figures/baseline_error_analysis/05_case_study_risk_response.png`

The figure names and purposes are documented consistently in the
baseline error-analysis review and case-study material.

Figures referenced correctly: `YES`

### Blocking issues

The GĐ3 traceability review found no remaining open GitHub issue after
the completed EWMA design issue was reconciled and closed.

No unresolved blocking issue: `YES`

### Shared-report ownership

Member A remains the owner of the shared methodology/report text.

Member B reviews the text and provides corrections.

Editing rule:

`A edits -> B reviews`

Only one member should edit the shared report at a time. Member A and
Member B must not concurrently edit the same shared report content.

### B-16.3 freeze decision

| Requirement | Status |
| --- | --- |
| Data chapter draft stable | YES |
| Baseline methodology stable | YES |
| Metrics definitions stable | YES |
| Figures referenced correctly | YES |
| No unresolved blocking issue | YES |
| Single-editor shared-report rule established | YES |

B-16.3 Data / Method draft freeze: `PASS`

Blocking issues: `NONE`
