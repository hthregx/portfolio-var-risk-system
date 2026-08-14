# Baseline Decision Log

## Purpose

This decision record documents the parameter sensitivity review for
Historical Simulation and EWMA VaR before the final baseline decision.

The sensitivity exercise is intended to evaluate robustness rather than
to force a parameter change.

## Validation Contract

The common eligible universe contains 1,137 target dates.

All Historical and EWMA sensitivity experiments use the same fixed
validation subset of 739 target dates from 2021-12-31 through
2024-12-17.

The remaining 398 later dates from 2024-12-18 through 2026-07-28 were
reserved from parameter selection.

The official test period was not used for parameter selection.

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

The existing Historical baseline of window = 250 is retained.

## EWMA Candidates

| Experiment | Decay | Forecast Count | Violation Count | Violation Rate | Pinball Loss | Average VaR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B04 | 0.90 | 739 | 40 | 5.412720% | 0.001910811794 | 2.542613% |
| B05 | 0.94 | 739 | 43 | 5.818674% | 0.001929922933 | 2.580969% |
| B06 | 0.97 | 739 | 43 | 5.818674% | 0.002000327134 | 2.642960% |

### EWMA Review

B04 with decay = 0.90 produces the strongest validation metrics among
the tested EWMA candidates. It has the lowest Pinball Loss and a
violation rate closer to the nominal 5% level than B05 and B06.

B05 with decay = 0.94 is the existing canonical EWMA configuration.
Its validation performance remains competitive and its parameter value
is the established baseline used by the production implementation.

B06 with decay = 0.97 does not provide sufficient improvement over the
other candidates. Its violation rate is equal to B05 while its Pinball
Loss and Average VaR are higher.

## Multi-Metric Assessment

The baseline decision is not based on a single metric.

The review considers:

- Violation Rate
- Pinball Loss
- Average VaR
- stability across parameter choices
- model simplicity
- continuity with the existing canonical baseline

Decay = 0.90 is the strongest EWMA validation candidate based on the
observed validation metrics. However, the sensitivity exercise is a
robustness check and does not by itself require replacement of the
canonical parameter.

Without stronger evidence from the joint A/B review that a baseline
change is necessary, decay = 0.94 remains the official EWMA baseline.

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

### EWMA decay = 0.94

Advantages:

- established canonical EWMA configuration;
- standard and simple decay specification;
- competitive Pinball Loss;
- maintains continuity with the production baseline.

Disadvantages:

- decay = 0.90 performs better on the current validation subset on both
  Pinball Loss and violation-rate calibration;
- its violation rate is farther from 5% than decay = 0.90.

## Baseline Decision

| Method | Selected Baseline | Decision |
| --- | ---: | --- |
| Historical Simulation | window = 250 | Retain |
| EWMA | decay = 0.94 | Retain pending stronger evidence |

Historical window = 250 remains the selected Historical baseline.

EWMA decay = 0.94 remains the selected EWMA baseline unless the joint
A/B cross-review identifies sufficiently strong evidence to replace it
with decay = 0.90.

The sensitivity analysis is interpreted as a robustness check. A
parameter change is not required when the evidence for replacement is
not sufficiently strong.

## Test-Period Separation

Parameter comparison and baseline recommendations were based only on
the fixed validation subset.

The later reserved dates were excluded from parameter selection.

The official test period was not used for parameter selection.

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