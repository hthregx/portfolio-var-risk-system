# Day 26 Final Presentation Evidence - Member A

## Objective

This document packages the frozen quantitative release evidence into a presentation-ready form for the canonical one-day 95% portfolio VaR system.

Day 26 is a release-acceptance and communication step, not a model-development step.

No new algorithm, feature, parameter search, evaluation window, canonical prediction, or model-selection decision is introduced here.

The quantitative statements below are projections and interpretations of already-frozen release artifacts.

## Frozen Release Contract

The final system evaluates an equal-weight portfolio of HPG, FPT, and MWG.

The frozen evaluation contract is:

- forecast target: portfolio simple return;
- forecast horizon: one trading day;
- confidence level: 95%;
- lower-tail probability: `alpha = 0.05`;
- evaluation period: 2024-12-18 through 2026-07-28;
- target dates per method: 398;
- canonical methods: 3;
- total canonical prediction rows: 1,194;
- strict violation rule: `actual_return < quantile_return`;
- reported VaR rule: `VaR = max(0, -quantile_return)`.

The frozen model configurations are:

| Method | Frozen configuration |
| --- | --- |
| Historical Simulation | `historical_w250` |
| EWMA | `ewma_d094` |
| Gradient Boosting Quantile Regression | `gb_G04` |

Historical Simulation uses a 250-observation rolling window.

EWMA uses the retained canonical decay factor `0.94`. Decay `0.90` produced stronger validation evidence, but it was not adopted for the canonical final evaluation. The retained `0.94` configuration must not be described as the validation winner.

Gradient Boosting uses the frozen G04 configuration with seven canonical return-history features:

1. `return_lag_1`
2. `return_lag_2`
3. `return_lag_5`
4. `rolling_vol_5`
5. `rolling_vol_20`
6. `rolling_vol_60`
7. `drawdown`

The canonical G04 final evaluation does not use the separate market-feature specification explored in other validation work.

## Final Acceptance Status

Day 26 introduces a machine-readable final release acceptance layer:

- builder: `scripts/build_final_acceptance_a.py`;
- output: `results/final_acceptance_matrix_a.csv`;
- acceptance checks: 30;
- PASS checks: 30;
- FAIL checks: 0;
- duplicate acceptance IDs: 0;
- numerical tolerance: `1e-12`.

The acceptance matrix covers seven evidence categories:

| Category | Checks |
| --- | ---: |
| Evaluation contract | 8 |
| Risk semantics | 2 |
| Frozen configuration | 5 |
| Quantitative evidence | 7 |
| Pairwise evidence | 3 |
| Release integrity | 2 |
| Release boundary | 3 |

The matrix verifies the canonical panel shape, common target dates, common realized returns, strict violation semantics, VaR identity, frozen model configurations, final quantitative evidence, criterion-specific leaders, pairwise results, prior evidence gates, model-freeze policy, interpretation boundaries, and the known provenance limitation.

The acceptance artifact is deterministic in the current tested environment.

The Day 26 acceptance builder SHA256 is:

`9f614239ea8a2073f482c89924a0dbc5a668b5138861d0ef24616a3bb8238a29`

The Day 26 acceptance matrix SHA256 is:

`070454a95aefdf8a6e9bae01a804a1c4b33aee4ad7ca1eaa912eba11e4cb02e8`

## Final Quantitative Summary

Day 26 also adds a presentation-ready three-row projection of the frozen metric-comparison artifact:

`results/final_quantitative_summary_a.csv`

The summary contains one row per canonical method and does not introduce any new metric.

Its source is:

`results/final_metric_comparison.csv`

The Day 26 quantitative summary SHA256 is:

`d7f7893a158ee6ff6a27e3160d48e268a847751cab154f95e31a4ba7f7f152ca`

The summary preserves the frozen values below.

| Method | Config | Forecasts | Violations | Violation rate | Mean pinball loss | Average VaR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Historical Simulation | `historical_w250` | 398 | 27 | 6.7839% | 0.001896464233 | 2.1330% |
| EWMA | `ewma_d094` | 398 | 22 | 5.5276% | 0.001968321253 | 2.5084% |
| Gradient Boosting | `gb_G04` | 398 | 24 | 6.0302% | 0.001758130951 | 2.3349% |

These values are release evidence, not a newly computed Day 26 experiment.

## Calibration Interpretation

The nominal lower-tail violation probability is 5%.

Observed violation rates are:

| Method | Violations | Violation rate | Distance from 5% |
| --- | ---: | ---: | ---: |
| Historical Simulation | 27 / 398 | 6.7839% | 1.7839 percentage points |
| EWMA | 22 / 398 | 5.5276% | 0.5276 percentage points |
| Gradient Boosting | 24 / 398 | 6.0302% | 1.0302 percentage points |

EWMA is the criterion-specific calibration leader because its observed violation rate is closest to the nominal 5% lower-tail probability.

This conclusion applies to observed violation-rate calibration only.

It does not establish EWMA as the universally best model.

## Quantile Accuracy Interpretation

Mean pinball loss evaluates lower-tail quantile forecast accuracy under the chosen quantile-loss criterion.

Lower mean pinball loss is preferred under this criterion.

| Method | Mean pinball loss |
| --- | ---: |
| Historical Simulation | 0.001896464233 |
| EWMA | 0.001968321253 |
| Gradient Boosting | 0.001758130951 |

Gradient Boosting has the lowest mean pinball loss over the frozen evaluation period.

It is therefore the criterion-specific leader for average quantile accuracy under mean pinball loss.

This result does not imply that Gradient Boosting has the lowest loss on every target date.

## Risk Magnitude Interpretation

Average reported VaR is:

| Method | Average VaR | Minimum VaR | Maximum VaR |
| --- | ---: | ---: | ---: |
| Historical Simulation | 2.1330% | 1.7088% | 2.7020% |
| EWMA | 2.5084% | 1.4006% | 5.6447% |
| Gradient Boosting | 2.3349% | 1.3742% | 4.8139% |

Historical Simulation has the lowest average reported VaR.

Lower average VaR is not evidence of automatic model superiority.

A lower average risk estimate may reflect a less conservative forecast magnitude and must be interpreted together with calibration and quantile-loss evidence.

Average VaR must therefore not be used alone to declare a preferred model.

## Pairwise Evidence

Pairwise analysis compares daily pinball losses on the same 398 target dates.

### Gradient Boosting vs Historical Simulation

Gradient Boosting has lower daily pinball loss on 151 dates.

Historical Simulation has lower daily pinball loss on 247 dates.

There are no ties.

The mean loss difference is:

`GB - Historical = -0.00013833328170849459`

The negative mean difference means Gradient Boosting has the lower mean pinball loss even though Historical Simulation wins on more individual dates.

This is not contradictory.

Daily win count measures frequency, while mean loss also reflects the magnitude of gains and losses.

The frozen pairwise evidence therefore shows that Gradient Boosting's average advantage over Historical Simulation is driven by sufficiently large improvements on a smaller subset of dates rather than by winning the majority of dates.

### Gradient Boosting vs EWMA

Gradient Boosting has lower daily pinball loss on 231 dates.

EWMA has lower daily pinball loss on 167 dates.

There are no ties.

The mean loss difference is:

`GB - EWMA = -0.00021019030172751563`

For this pair, both daily win count and mean pinball loss favor Gradient Boosting under the pinball-loss criterion.

### EWMA vs Historical Simulation

EWMA has lower daily pinball loss on 131 dates.

Historical Simulation has lower daily pinball loss on 267 dates.

There are no ties.

The mean loss difference is:

`EWMA - Historical = +0.00007185702001901683`

The positive mean difference means EWMA has higher mean pinball loss than Historical Simulation over the frozen evaluation period.

This pair illustrates why calibration and quantile-loss conclusions should not be collapsed into a single ranking.

EWMA is closest to the nominal 5% violation rate while Historical Simulation has lower mean pinball loss than EWMA.

## Decision Guidance

The frozen final evaluation supports criterion-specific conclusions.

| Decision criterion | Evidence-supported method |
| --- | --- |
| Closest observed violation rate to nominal 5% | EWMA |
| Lowest mean pinball loss | Gradient Boosting |
| Lowest average reported VaR | Historical Simulation |

If the analytical priority is observed violation-rate calibration, EWMA provides the strongest result.

If the analytical priority is average lower-tail quantile accuracy under pinball loss, Gradient Boosting provides the strongest result.

If the analytical question is strictly which method reports the lowest average VaR magnitude, Historical Simulation has the lowest value.

These are separate criteria that answer different questions.

There is no single overall winner declared by the frozen final evaluation.

## Presentation Guidance

A presentation of the final results should lead with the common evaluation contract before comparing the methods.

The recommended sequence is:

1. Explain the equal-weight HPG/FPT/MWG portfolio and one-day 95% VaR target.
2. State that all three methods share 398 evaluation target dates.
3. Present violation-rate calibration.
4. Present mean pinball loss.
5. Present average VaR as a risk-magnitude measure rather than a superiority score.
6. Use the pairwise evidence to explain why average performance and daily win counts can differ.
7. End with criterion-specific conclusions rather than a universal ranking.

The preferred final message is that the three methods exhibit different strengths under different evaluation criteria.

## Reproducibility Evidence

The final quantitative release is supported by multiple evidence layers.

Day 24 produced the frozen final evidence figures and a 26-check evidence-validation artifact.

Day 25 produced a 24-claim claim-to-evidence traceability matrix with 24 PASS claims and 0 FAIL claims.

Day 26 produced a 30-check final acceptance matrix with 30 PASS checks and 0 FAIL checks.

Day 26 also produced a deterministic three-row quantitative summary projected directly from the frozen metric-comparison artifact.

No model runner was required to create the Day 26 presentation artifacts.

The existing final evidence figures reproduced byte-for-byte in the tested environment:

- `results/figures/final_violation_rate_a.png`;
- `results/figures/final_pinball_loss_a.png`;
- `results/figures/final_average_var_a.png`.

The byte-for-byte figure claim is limited to the tested environment.

It should not automatically be generalized across operating systems, package versions, plotting-library versions, or rendering stacks.

## Limitations

The final evaluation period was reserved for the documented final workflow, but it should not be described as a pristine, never-inspected test set because broader inspection occurred earlier in the project.

The conclusions are specific to:

- the equal-weight HPG/FPT/MWG portfolio;
- the one-day forecast horizon;
- the 95% VaR definition;
- the evaluation period from 2024-12-18 through 2026-07-28;
- the three frozen model configurations;
- the observed market conditions in that period.

The results should not automatically be generalized to other portfolios, confidence levels, forecast horizons, parameterizations, or market regimes.

The evaluation contains 398 common target dates per method. This is the frozen release sample for the documented comparison, not evidence of universal out-of-sample superiority.

Runtime is an operational and reproducibility measurement, not a predictive-quality ranking criterion.

## Provenance

The final-run metadata contains a known provenance completeness limitation.

The runtime metadata records an earlier repository HEAD than the later source commit that finalized the release package.

The canonical prediction content remained unchanged.

This is documented as provenance incompleteness.

It is not evidence that the canonical predictions are incorrect, and Day 26 does not rewrite historical metadata solely to remove this limitation.

## Release Acceptance Interpretation

The Day 26 machine-readable acceptance matrix reports 30 PASS checks and 0 FAIL checks.

The accepted package preserves:

- the frozen evaluation contract;
- the three canonical configurations;
- the common 398-date evaluation panel;
- strict violation semantics;
- the VaR identity;
- the frozen aggregate metrics;
- the frozen pairwise evidence;
- prior Day 24 and Day 25 evidence gates;
- model-freeze boundaries;
- interpretation guardrails;
- the known provenance limitation.

This acceptance is a release-readiness statement.

It is not a new statistical experiment and does not reopen model selection.

## Final Presentation Interpretation

The final quantitative story is a trade-off story rather than a single-model ranking.

EWMA is strongest on observed violation-rate calibration.

Gradient Boosting is strongest on mean pinball loss.

Historical Simulation has the lowest average reported VaR.

The pairwise evidence provides additional context for these aggregate results, particularly the distinction between frequency of daily wins and magnitude of loss improvements.

The release package therefore supports presenting the models by criterion-specific strengths and limitations.

No universal winner is declared.