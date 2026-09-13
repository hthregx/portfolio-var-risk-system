# Baseline Error Analysis — Independent Review

## Purpose

This note records the independent review of the baseline error-analysis
artifact produced for the Historical Simulation and EWMA baseline
comparison.

The review is performed independently from the analysis notebook.
No changes to the source notebook are made as part of this review.

## Reviewed artifacts

Primary error-analysis artifact:

- `results/baseline_error_analysis.csv`

Supporting case-study artifact:

- `results/baseline_case_studies.csv`

Source analysis notebook:

- `notebooks/07_baseline_error_analysis.ipynb`

The notebook remains owned by A. Findings from this review should be
reported rather than corrected directly in the notebook by B.

## Sample alignment audit

The baseline error-analysis dataset was checked for common-date
alignment between Historical Simulation and EWMA.

Expected comparison sample:

- target rows: 1,387
- Historical violations: 75
- EWMA violations: 73
- shared violations: 56
- Historical-only violations: 19
- EWMA-only violations: 17

The reviewed artifact satisfies these expected counts.

Both methods are compared on the same target-date universe and use the
same realized target return for each comparison date.

### Result

**PASS — common-date baseline comparison is aligned.**

## Violation audit

The required violation definition is strict:

`actual_return < quantile_return`

Equality with the forecast quantile must not be classified as a
violation.

The error-analysis violation indicators were reviewed against this
definition.

### Result

**PASS — strict violation semantics are consistent with the baseline
backtesting contract.**

## Exception category audit

Each target date belongs to one of the following relevant exception
categories:

- shared: both Historical and EWMA violate;
- Historical-only: Historical violates and EWMA does not;
- EWMA-only: EWMA violates and Historical does not;
- neither: neither method violates.

Observed exception counts are:

| Exception category | Count |
|---|---:|
| Shared | 56 |
| Historical-only | 19 |
| EWMA-only | 17 |

These counts reconcile with the model-level totals:

- Historical: 56 + 19 = 75;
- EWMA: 56 + 17 = 73.

### Result

**PASS — exception categories reconcile with the baseline violation
counts.**

## Severity audit

Exception severity is interpreted as the amount by which the realized
return exceeds the forecast loss threshold on a violation date.

For a return quantile `q` and realized return `r`, the violation
severity is:

`max(0, q - r)`

Therefore:

- severity is positive only when `r < q`;
- severity is zero for a non-violation;
- a larger positive value represents a larger threshold exceedance.

Severity must not be confused with positive VaR.

### Result

**PASS — severity interpretation is suitable for the standardized
error-analysis figures.**

## Cluster audit

Exception clusters were reviewed independently using consecutive
positions in the canonical trading-date sequence.

A cluster continues when two violation observations occur on
consecutive canonical target-date positions.

The definition does not use calendar-day distance. Therefore weekends
and market holidays do not incorrectly break a sequence of consecutive
trading observations.

### Historical Simulation

- clusters: 63
- singleton clusters: 53
- multi-observation clusters: 10
- longest cluster: 4 observations

### EWMA

- clusters: 65
- singleton clusters: 57
- multi-observation clusters: 8
- longest cluster: 2 observations

### Result

**PASS — cluster construction is based on canonical trading-date
positions and is appropriate for the case-study selection rule.**

## Case-study selection review

Five target dates were selected through the deterministic case-study
selection process.

The selected set contains:

- 2 shared exceptions;
- 2 Historical-only exceptions;
- 1 EWMA-only exception.

The selection therefore includes representation of every required
exception category.

The April 2025 observations include the longest Historical exception
cluster selected for detailed examination.

The EWMA-only case was selected using exception severity rather than
manual visual inspection.

### Result

**PASS — five case-study dates satisfy the required 5–10 day range and
were selected using an explicit rule rather than subjective chart
selection.**

## Look-ahead interpretation boundary

Forecast quantities must only be interpreted using information
available at the corresponding forecast origin.

Fields describing subsequent VaR movement may be used to discuss how a
model responded after the realized target return became observable.

They must not be presented as information available to the original
forecast.

Figure captions and case-study text must preserve this distinction.

### Result

**PASS as a review requirement.**

This constraint remains part of figure and report QA.

## Figure requirements

The standardized figure set must preserve the following conventions:

- Historical Simulation and EWMA naming must be consistent;
- target dates must use the same date convention;
- return quantiles must retain their return-space sign;
- positive VaR must not be plotted or described as a negative return
  quantile;
- violation markers must be visually distinguishable;
- figures must use the same canonical comparison sample;
- captions must remain descriptive and must not infer external event
  causes.

Expected figure set:

1. `01_baseline_var_vs_returns.png`
2. `02_exception_timeline.png`
3. `03_exception_severity.png`
4. `04_exception_clusters.png`
5. `05_case_study_risk_response.png`

## Review findings

No blocking numerical inconsistency was identified in the reviewed
baseline error-analysis artifact based on the independent checks
performed by B.

The following interpretation constraints remain important for report
integration:

1. Do not interpret positive VaR and negative return quantiles as the
   same signed quantity.
2. Do not interpret subsequent model response as information available
   at the original forecast date.
3. Do not infer news, market-event, or causal explanations from the
   numerical error-analysis artifact alone.
4. Preserve the common 1,387-date comparison sample when reporting
   Historical versus EWMA results.
5. Preserve the strict `<` violation definition in figures, captions,
   and report text.

## B-15.2 review status

- common target-date sample: PASS
- Historical violation count: PASS
- EWMA violation count: PASS
- shared exception count: PASS
- Historical-only count: PASS
- EWMA-only count: PASS
- strict violation rule: PASS
- same-target-return requirement: PASS
- severity definition review: PASS
- cluster definition review: PASS
- deterministic case selection: PASS

**Overall review status: PASS — no blocking finding identified.**