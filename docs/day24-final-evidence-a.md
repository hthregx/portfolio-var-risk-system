# Day 24 Final Evaluation Evidence — Member A

## Objective

This document packages the frozen final evaluation evidence for the three canonical one-day 95% portfolio VaR forecasting methods:

- Historical Simulation
- EWMA
- Gradient Boosting Quantile Regression

The purpose is to communicate the final numerical evidence without changing the frozen model specifications, evaluation window, features, parameters, predictions, or model-selection decisions.

No new model, feature, tuning exercise, or evaluation period is introduced here.

## Frozen Evaluation Contract

The final evaluation contains 398 target dates per method, covering 2024-12-18 through 2026-07-28.

With three methods, the canonical prediction artifact contains 1,194 rows in total.

The evaluation uses:

- one-day forecast horizon;
- 95% VaR;
- lower-tail probability `alpha = 0.05`;
- strict violation rule: `actual_return < quantile_return`;
- reported VaR identity: `VaR = max(0, -quantile_return)`.

The frozen configuration identifiers are:

| Method | Frozen configuration |
| --- | --- |
| Historical Simulation | `historical_w250` |
| EWMA | `ewma_d094` |
| Gradient Boosting | `gb_G04` |

The final prediction rows were validated to have aligned target dates, aligned realized portfolio returns, no duplicate method-target keys, and consistent violation and VaR definitions.

## Calibration Evidence

The nominal violation rate is 5%.

| Method | Violations | Violation rate | Distance from 5% |
| --- | ---: | ---: | ---: |
| Historical Simulation | 27 / 398 | 6.7839% | 1.7839 percentage points |
| EWMA | 22 / 398 | 5.5276% | 0.5276 percentage points |
| Gradient Boosting | 24 / 398 | 6.0302% | 1.0302 percentage points |

EWMA is the criterion-specific calibration leader because its observed violation rate is closest to the nominal 5% tail probability.

This statement is limited to violation-rate calibration. It does not establish EWMA as the overall best model.

![Frozen violation-rate comparison](figures/final_violation_rate_a.png)

## Quantile Accuracy Evidence

Mean pinball loss evaluates the quality of the predicted lower-tail quantile. Lower values are better under this criterion.

| Method | Mean pinball loss |
| --- | ---: |
| Historical Simulation | 0.001896464233 |
| EWMA | 0.001968321253 |
| Gradient Boosting | 0.001758130951 |

Gradient Boosting has the lowest mean pinball loss and is therefore the criterion-specific pinball-loss leader over the frozen evaluation window.

This result does not imply that Gradient Boosting produces the lowest loss on every target date.

![Frozen pinball-loss comparison](figures/final_pinball_loss_a.png)

## Risk Magnitude Evidence

Average reported VaR differs across the methods:

| Method | Average VaR |
| --- | ---: |
| Historical Simulation | 2.1330% |
| EWMA | 2.5084% |
| Gradient Boosting | 2.3349% |

Historical Simulation has the lowest average VaR.

Lower average VaR must not be interpreted as automatic model superiority. A smaller risk estimate may reflect a less conservative forecast and must be considered together with calibration and quantile-loss evidence.

![Frozen average-VaR comparison](figures/final_average_var_a.png)

## Pairwise Evidence

Pairwise diagnostics compare daily pinball losses on the same 398 target dates.

### Gradient Boosting vs Historical Simulation

Gradient Boosting has lower daily pinball loss on 151 dates, while Historical Simulation has lower loss on 247 dates.

Despite winning on fewer individual dates, Gradient Boosting has a lower mean pinball loss over the complete evaluation window:

`GB - Historical mean pinball delta = -0.000138333282`.

This means that the magnitude of Gradient Boosting's improvements on some dates is sufficient to outweigh its larger number of daily losses. The result therefore illustrates why daily win count and mean loss answer different questions.

The earlier pairwise concentration diagnostics also show that the improvement is relatively concentrated: approximately 86.5% of the gross Gradient Boosting improvement is accumulated within the top 20% of its improvement dates, and 16 improvement dates account for approximately 80% of gross improvement.

### Gradient Boosting vs EWMA

Gradient Boosting records lower daily pinball loss on 231 dates versus 167 dates for EWMA.

The mean difference is:

`GB - EWMA mean pinball delta = -0.000210190302`.

Here, both the daily win count and the mean pinball-loss comparison favor Gradient Boosting.

The improvement is less concentrated than in the Gradient Boosting versus Historical Simulation comparison: 63 improvement dates are required to reach approximately 80% of gross Gradient Boosting improvement.

### EWMA vs Historical Simulation

EWMA records lower daily pinball loss on 131 dates versus 267 dates for Historical Simulation.

The mean difference is:

`EWMA - Historical mean pinball delta = +0.000071857020`.

The positive delta indicates that EWMA has a higher mean pinball loss than Historical Simulation over this evaluation window.

These pairwise results do not define a universal model ranking. They describe differences under the pinball-loss criterion on the frozen evaluation dates.

## Criterion-Specific Findings

The frozen final evidence supports three separate conclusions:

1. **Calibration:** EWMA is closest to the nominal 5% violation rate.
2. **Quantile accuracy:** Gradient Boosting has the lowest mean pinball loss.
3. **Average risk magnitude:** Historical Simulation has the lowest average VaR.

These are criterion-specific findings rather than an overall ranking.

There is no single overall winner declared by the final evaluation package.

## Limitations

The evaluation period was reserved for the documented final parameter-selection workflow, but it should not be described as a pristine, never-inspected test set because broader inspection occurred earlier in the project.

The results are specific to:

- the equal-weight HPG/FPT/MWG portfolio;
- the frozen one-day 95% VaR setup;
- the frozen evaluation period from 2024-12-18 through 2026-07-28;
- the frozen model configurations.

The evidence should not be generalized automatically to other portfolios, horizons, confidence levels, or market regimes.

Runtime values in the frozen artifacts describe the recorded final run and should not be interpreted as portable performance benchmarks across hardware or software environments.

The final-run metadata also has a known provenance limitation: the recorded runtime HEAD predates the later source commit that finalized the Day 21 package. The canonical prediction content remained unchanged, so this is treated as provenance incompleteness rather than a reason to rewrite historical metadata or rerun the frozen models.

## Final Interpretation

The three models exhibit different strengths under the frozen evaluation contract.

EWMA is strongest on violation-rate calibration, Gradient Boosting is strongest on mean pinball loss, and Historical Simulation produces the lowest average reported VaR.

The pairwise analysis adds important context to the aggregate metrics. In particular, Gradient Boosting's lower mean pinball loss relative to Historical Simulation is driven by the magnitude of improvements rather than by winning on the majority of individual dates.

For this reason, the final release should preserve the criterion-specific interpretation instead of collapsing the evidence into a single model ranking.

The Day 24 evidence package is explanatory and reproducibility-oriented. It does not modify the frozen models, parameters, features, canonical predictions, or evaluation period.