# Phase 3 A â€” Exploratory VaR optimization summary

## Scope
- Objective: conditional 5th-percentile portfolio simple-return forecast (VaR 95% for T+1), not trend classification.
- KBS snapshot cutoff: 2026-11-06; development through 2024-12-17.
- Phase 2 baseline: GBv2 C1 12 features, frozen and SHA256-verified.
- Original historical Phase 2 holdout NOT used for selection or Phase 3 evaluation.

## Feature-count screen (exploratory development CV)
| feature_count | cv_pinball |
| --- | --- |
| 12.0 | 0.0020802674466814 |
| 18.0 | 0.0021028871532953 |
| 25.0 | 0.0021613464728058 |

## Policies on 600 paired development targets
| policy | n | violations | violation_rate | mean_pinball |
| --- | --- | --- | --- | --- |
| C1_frozen | 600 | 36 | 0.0599999999999999 | 0.0018547455981367 |
| nested_selected | 600 | 28 | 0.0466666666666666 | 0.0018568696748033 |
| expanding_refit20 | 600 | 34 | 0.0566666666666666 | 0.0018420940700091 |
| rolling504_refit20 | 600 | 32 | 0.0533333333333333 | 0.0018377553521267 |

## Result / promotion gate
- NO_CONFIRMED_IMPROVEMENT: All descriptive paired block-bootstrap intervals include zero; original inspected history is not a pristine test
- No claim that rolling refitting is significantly or prospectively better.
- Retain GBv2 C1 as the research reference; do not promote any experiment to the live forecast registry.

## Limitations and next evaluation
- Historical development data were examined during earlier experiments: all comparisons remain exploratory.
- A clean future, pre-registered timestamped evaluation period is needed for confirmation.
- Backtesting non-rejection is not evidence of perfect model calibration.
- Check original B snapshot hash, common A/B protocol, and unresolved VNINDEX flags separately.
- Retrospective predictions cannot be registered as genuine historical real-time forecasts.

## Reproducibility
- All P3-03..P3-05 artifacts and Figure SHA256 hashes audited against manifests.
- A 600-session paired prediction and 60-refit schedule audit was performed without any model fit here.
- See `release_manifest.json` for full evidence SHA256 and input provenance.
