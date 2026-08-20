# Baseline Decision Log

## Purpose

This decision record documents the parameter sensitivity review for
Historical Simulation and EWMA VaR before the final baseline decision.

The sensitivity exercise is intended to evaluate parameter robustness
rather than to force a parameter change.

This record distinguishes between:

- the parameter preferred by the Day-13 validation comparison; and
- the existing canonical/production parameter retained for continuity.

It also records an important limitation of the validation design:
although the later segment was excluded from the Day-13 parameter-selection
calculations, full-sample baseline metrics had already been inspected during
the earlier Day-12 baseline evaluation. Therefore, the later segment must not
be described as a pristine untouched test set.

## Validation Contract

The common eligible universe contains 1,137 target dates.

All Historical and EWMA sensitivity experiments use the same fixed
Day-13 validation subset of 739 target dates from 2021-12-31 through
2024-12-17.

The remaining 398 later dates run from 2024-12-18 through 2026-07-28.
These dates were excluded from the Day-13 parameter-selection calculations.

However, this later segment should not be described as a pristine untouched
test segment because full-sample baseline metrics had already been inspected
during the earlier baseline evaluation.

Therefore, the correct interpretation is:

- Day-13 parameter selection uses only the 739-row validation subset;
- the later 398 rows are excluded from the Day-13 selection calculations;
- the later 398 rows are not considered a pristine unseen holdout because
  earlier full-sample baseline metrics had already been inspected.

The common model contract is:

- alpha = 0.05
- confidence level = 0.95
- portfolio = equal-weight HPG/FPT/MWG
- forecast horizon = 1 trading day
- VaR = max(0, -q0.05)

## Historical Candidates

| Experiment | Window | Forecast Count | Violation Count | Violation Rate | Pinball Loss | Average VaR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B01 | 125 | 739 | 45 | 6.089310% | 0.002117106095 | 2.673517% |
| B02 | 250 | 739 | 37 | 5.006766% | 0.002144803137 | 2.852501% |
| B03 | 500 | 739 | 37 | 5.006766% | 0.002174145160 | 2.797091% |

### Historical Review

B01 has a slightly lower Pinball Loss than B02 and B03, but its
violation rate of 6.089310% is materially farther from the nominal 5%
level.

B02 and B03 both produce a violation rate of 5.006766%, which is very
close to the nominal 5% level. B02 also has a lower Pinball Loss than
B03.

Window 250 therefore provides a reasonable balance between calibration,
quantile accuracy, model responsiveness, and simplicity.

The Historical sensitivity review supports retaining window = 250.

## EWMA Candidates

| Experiment | Decay | Forecast Count | Violation Count | Violation Rate | Pinball Loss | Average VaR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B04 | 0.90 | 739 | 40 | 5.412720% | 0.001910811794 | 2.542613% |
| B05 | 0.94 | 739 | 43 | 5.818674% | 0.001929922933 | 2.580969% |
| B06 | 0.97 | 739 | 43 | 5.818674% | 0.002000327134 | 2.642960% |

### EWMA Review

B04 with decay = 0.90 produces the strongest observed validation metrics
among the tested EWMA candidates.

Its Pinball Loss is 0.001910811794, lower than both decay = 0.94
and decay = 0.97.

Its violation rate is 5.412720%, which is also closer to the nominal
5% level than the 5.818674% violation rates produced by decay = 0.94
and decay = 0.97.

Therefore:

**Validation-selected EWMA candidate = decay 0.90.**

B05 with decay = 0.94 is the existing canonical/production EWMA
configuration. It does not win the Day-13 sensitivity comparison.
It is retained separately as the canonical default for continuity
pending stronger evidence and a deliberate baseline-change decision.

Therefore:

**Canonical/production EWMA default = decay 0.94.**

These two statements represent different decisions and must not be
interpreted as meaning that decay = 0.94 won the validation sensitivity
comparison.

B06 with decay = 0.97 does not provide sufficient improvement over the
other candidates. Its violation rate is equal to B05 while its Pinball
Loss and Average VaR are higher.

## Multi-Metric Assessment

The sensitivity assessment is not based on a single metric.

The review considers:

- Violation Rate
- Pinball Loss
- Average VaR
- stability across parameter choices
- model simplicity
- continuity with the existing canonical baseline

For Historical Simulation, window = 250 provides strong calibration and
a lower Pinball Loss than window = 500. The validation evidence therefore
supports retaining window = 250.

For EWMA, decay = 0.90 is the validation-selected candidate. This
conclusion is not based only on Pinball Loss: decay = 0.90 also produces
better violation-rate calibration than decay = 0.94 on the fixed
Day-13 validation subset.

However, selection by the sensitivity exercise and replacement of the
existing canonical/production configuration are treated as separate
decisions.

Decay = 0.94 therefore remains the canonical/production EWMA default for
continuity pending stronger evidence and an explicit decision to change
the production baseline.

This retention must not be interpreted as evidence that decay = 0.94
outperformed decay = 0.90 on the Day-13 validation subset.

## Advantages and Disadvantages

### Historical window = 250

Advantages:

- violation rate is very close to the nominal 5% level;
- lower Pinball Loss than window = 500;
- less reactive than the shorter 125-day window;
- established canonical configuration;
- simple rolling-window interpretation.

Disadvantages:

- Average VaR is higher than the other Historical candidates;
- fixed rolling windows may adapt more slowly than volatility-weighted
  approaches.

### EWMA decay = 0.90

Advantages:

- lowest Pinball Loss among the tested EWMA candidates;
- violation rate closest to the nominal 5% level among the tested EWMA
  candidates;
- strongest observed Day-13 EWMA validation result.

Disadvantages:

- differs from the existing canonical/production decay = 0.94
  configuration;
- adopting it as the production default would constitute a baseline
  change and therefore requires an explicit decision rather than being
  implied automatically by the sensitivity exercise.

### EWMA decay = 0.94

Advantages:

- established canonical/production EWMA configuration;
- standard and simple decay specification;
- competitive Pinball Loss;
- maintains continuity with the existing production baseline.

Disadvantages:

- does not win the Day-13 validation sensitivity comparison;
- decay = 0.90 has lower Pinball Loss on the validation subset;
- decay = 0.90 also has better violation-rate calibration on the
  validation subset.

## Baseline Decision

| Method | Validation-Selected Candidate | Canonical/Production Default | Decision |
| --- | ---: | ---: | --- |
| Historical Simulation | window = 250 | window = 250 | Retain |
| EWMA | decay = 0.90 | decay = 0.94 | Retain 0.94 for continuity pending explicit baseline-change evidence |

Historical Simulation has no distinction between the validation-selected
candidate and the retained canonical baseline: window = 250 is supported
by the sensitivity evidence and remains the canonical Historical baseline.

EWMA requires an explicit distinction.

The Day-13 validation comparison selects:

**decay = 0.90**

as the strongest EWMA validation candidate.

The existing canonical/production default remains:

**decay = 0.94**

for continuity pending stronger evidence and an explicit decision to
replace the canonical parameter.

Accordingly, decay = 0.94 is not recorded as the winner of the Day-13
EWMA sensitivity exercise.

The sensitivity exercise demonstrates that decay = 0.90 deserves
consideration as a replacement candidate, while the production baseline
remains unchanged until a separate baseline-change decision is made.

## Validation and Later-Segment Separation

Parameter comparison on Day 13 was performed using only the fixed
739-row validation subset.

The later 398-row segment from 2024-12-18 through 2026-07-28 was excluded
from the Day-13 parameter-selection calculations.

However, the later segment was not pristine or completely unseen at that
point because full-sample baseline metrics had already been inspected
during the earlier Day-12 baseline evaluation.

For that reason, this record does not characterize the later 398 rows as
an untouched official test period.

The appropriate claim is:

**The later segment was excluded from Day-13 parameter selection, but it
should not be described as a pristine untouched test segment because
full-sample baseline metrics had already been inspected.**

This limitation should be considered when interpreting subsequent
out-of-sample claims.

## Traceability

The experiment identifiers are:

- B01: Historical Simulation, window = 125
- B02: Historical Simulation, window = 250
- B03: Historical Simulation, window = 500
- B04: EWMA, decay = 0.90
- B05: EWMA, decay = 0.94
- B06: EWMA, decay = 0.97

Source sensitivity artifact:

`results/sensitivity_experiments.csv`

EWMA-specific artifact:

`results/ewma_sensitivity.csv`

## Audit Clarification

The Day-13 numerical sensitivity results remain unchanged.

This clarification addresses two documentation issues identified during
the later audit:

1. The later 398-row segment was excluded from Day-13 parameter selection,
   but it was previously exposed through full-sample baseline metrics and
   therefore must not be characterized as a pristine untouched test set.

2. EWMA decay = 0.90 is the validation-selected candidate, while decay =
   0.94 remains the canonical/production default for continuity. Retaining
   decay = 0.94 does not mean that it won the Day-13 sensitivity comparison.

No B01-B06 sensitivity metric is changed by this clarification.