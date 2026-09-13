# Day 25 Final Quantitative Story — Member A

## Objective

This document converts the frozen final evaluation evidence into a concise quantitative release narrative for the canonical one-day 95% portfolio VaR system.

The purpose is interpretation and traceability, not model development.

No new model, feature, tuning exercise, evaluation window, canonical prediction, or model-selection decision is introduced on Day 25.

All final claims are tied to the frozen evidence package and the Day 25 claim-to-evidence traceability matrix.

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
- strict violation definition: `actual_return < quantile_return`;
- reported VaR definition: `VaR = max(0, -quantile_return)`.

The frozen model configurations are:

| Method | Frozen configuration |
| --- | --- |
| Historical Simulation | `historical_w250` |
| EWMA | `ewma_d094` |
| Gradient Boosting Quantile Regression | `gb_G04` |

Historical Simulation uses a 250-observation rolling window.

EWMA uses the retained canonical decay factor `0.94`. This must not be presented as evidence that 0.94 defeated the 0.90 alternative in validation; the freeze preserves the canonical configuration already used in the final evaluation.

Gradient Boosting uses the frozen G04 configuration with the canonical seven return-history features:

1. `return_lag_1`
2. `return_lag_2`
3. `return_lag_5`
4. `rolling_vol_5`
5. `rolling_vol_20`
6. `rolling_vol_60`
7. `drawdown`

The canonical G04 final evaluation does not use the separate market-feature specification explored in other validation work.

## Evidence Traceability

Day 25 adds an explicit claim-to-evidence layer:

- builder: `scripts/build_final_traceability_a.py`;
- output: `results/final_claim_traceability_a.csv`;
- traceability claims: 24;
- PASS claims: 24;
- FAIL claims: 0;
- duplicate claim IDs: 0;
- numerical tolerance: `1e-12`.

The 24 claims cover:

- seven evaluation-contract claims;
- three frozen-configuration claims;
- five aggregate-evidence claims;
- six pairwise-evidence claims;
- three release-boundary claims.

The traceability artifact was regenerated repeatedly with identical SHA256 output in the current tested environment.

This evidence layer does not recompute or replace the final model outputs. It validates that the release narrative remains consistent with the already-frozen artifacts.

## Calibration Interpretation

The nominal lower-tail violation probability is 5%.

| Method | Violations | Violation rate | Distance from 5% |
| --- | ---: | ---: | ---: |
| Historical Simulation | 27 / 398 | 6.7839% | 1.7839 percentage points |
| EWMA | 22 / 398 | 5.5276% | 0.5276 percentage points |
| Gradient Boosting | 24 / 398 | 6.0302% | 1.0302 percentage points |

EWMA is the criterion-specific calibration leader because its observed violation rate is closest to the nominal 5% tail probability.

This conclusion is limited to violation-rate calibration.

It does not establish EWMA as the best model under every criterion.

## Quantile Accuracy Interpretation

Mean pinball loss evaluates the accuracy of the predicted lower-tail quantile. Lower values are preferred under this criterion.

| Method | Mean pinball loss |
| --- | ---: |
| Historical Simulation | 0.001896464233 |
| EWMA | 0.001968321253 |
| Gradient Boosting | 0.001758130951 |

Gradient Boosting has the lowest mean pinball loss over the frozen evaluation window.

It is therefore the criterion-specific leader for quantile accuracy under mean pinball loss.

This result does not mean that Gradient Boosting has lower loss on every individual target date.

## Risk Magnitude Interpretation

Average reported VaR is:

| Method | Average VaR |
| --- | ---: |
| Historical Simulation | 2.1330% |
| EWMA | 2.5084% |
| Gradient Boosting | 2.3349% |

Historical Simulation has the lowest average reported VaR.

Lower average VaR must not be interpreted as automatic model superiority.

A smaller VaR estimate describes the average magnitude of the reported risk forecast. It must be interpreted together with calibration and quantile-loss evidence rather than treated as an isolated model-ranking criterion.

## Pairwise Interpretation

Pairwise diagnostics compare daily pinball losses using the same 398 target dates.

### Gradient Boosting vs Historical Simulation

Gradient Boosting has lower daily pinball loss on 151 dates, while Historical Simulation has lower daily loss on 247 dates.

The mean pinball-loss difference is:

`GB - Historical = -0.00013833328170849459`

The negative mean difference indicates that Gradient Boosting has lower mean pinball loss even though it wins on fewer individual dates.

This is an important distinction: daily win count and mean loss measure different aspects of performance.

The Gradient Boosting improvements against Historical Simulation are relatively concentrated. Approximately 86.5% of gross Gradient Boosting improvement is accumulated within the top 20% of its improvement dates, and 16 improvement dates account for approximately 80% of gross improvement.

### Gradient Boosting vs EWMA

Gradient Boosting has lower daily pinball loss on 231 dates, compared with 167 dates for EWMA.

The mean difference is:

`GB - EWMA = -0.00021019030172751563`

Both the daily win count and mean pinball-loss comparison favor Gradient Boosting under this criterion.

The improvement is less concentrated than in the Gradient Boosting versus Historical Simulation comparison: approximately 63 Gradient Boosting improvement dates are required to accumulate 80% of gross improvement.

### EWMA vs Historical Simulation

EWMA has lower daily pinball loss on 131 dates, while Historical Simulation has lower loss on 267 dates.

The mean difference is:

`EWMA - Historical = +0.00007185702001901683`

The positive mean difference indicates that EWMA has higher mean pinball loss than Historical Simulation over the frozen evaluation window.

These pairwise comparisons provide context for the aggregate pinball-loss results. They do not establish a universal model ranking.

## Decision Guidance

The three methods show different strengths:

| Decision criterion | Evidence-supported method |
| --- | --- |
| Closest observed violation rate to nominal 5% | EWMA |
| Lowest mean pinball loss | Gradient Boosting |
| Lowest average reported VaR | Historical Simulation |

The appropriate interpretation depends on the analytical objective.

If the primary question is calibration to the nominal tail probability, EWMA provides the strongest observed result.

If the primary question is average lower-tail quantile accuracy under pinball loss, Gradient Boosting provides the strongest observed result.

If the question is simply which method reports the lowest average VaR magnitude, Historical Simulation has the lowest value.

These statements are criterion-specific.

The frozen evidence does not support collapsing these three criteria into one universal ranking.

There is no single overall winner declared by the final evaluation package.

## Reproduction Evidence

The Day 24 evidence figures were regenerated on Day 25 from the frozen metric-comparison artifact using the existing evidence builder.

The three figures are:

- `results/figures/final_violation_rate_a.png`;
- `results/figures/final_pinball_loss_a.png`;
- `results/figures/final_average_var_a.png`.

All three reproduced byte-for-byte in the current tested environment.

Their SHA256 values remained:

- violation-rate figure: `eff0dabc0257b1b8ca5fdcb158b8feeb8efe35d978ffbc18af7c4f91fdb93e18`;
- pinball-loss figure: `4619013d00dac4b7f260f807204b93104beba72f056a38eb720ce7dcc5180d80`;
- average-VaR figure: `ab55fc50b2b8075eb87a57c498952dda5eb4dabee1fdc4e8b64b30927d8b5ef0`.

The frozen comparison artifact and the evidence builder remained unchanged during reproduction.

The byte-for-byte claim is limited to the current tested environment and should not be generalized automatically across operating systems, dependency versions, or rendering stacks.

## Limitations

The final evaluation period was reserved for the documented final parameter-selection workflow, but it should not be described as a pristine, never-inspected test set because broader inspection occurred earlier in the project.

The quantitative conclusions are specific to:

- the equal-weight HPG/FPT/MWG portfolio;
- the one-day forecast horizon;
- the 95% VaR definition;
- the frozen evaluation period from 2024-12-18 through 2026-07-28;
- the frozen model configurations;
- the observed market conditions in that period.

The results should not automatically be generalized to different portfolios, confidence levels, horizons, parameterizations, or market regimes.

Runtime values recorded in the frozen artifacts are execution measurements from the final run and should not be interpreted as portable hardware-independent performance benchmarks.

## Provenance

The final-run metadata contains a known provenance limitation.

The recorded runtime HEAD predates the later source commit that finalized the Day 21 release package.

The canonical prediction content remained unchanged.

This is therefore documented as provenance incompleteness rather than treated as a reason to rewrite historical metadata or rerun the full frozen model pipeline.

Day 25 does not retroactively modify this history.

## Final Release Interpretation

The final quantitative evidence supports a trade-off interpretation rather than a single model ranking.

EWMA is strongest on observed violation-rate calibration.

Gradient Boosting is strongest on mean pinball loss.

Historical Simulation produces the lowest average reported VaR.

The pairwise analysis adds additional context, especially for Gradient Boosting versus Historical Simulation: Gradient Boosting achieves the lower mean pinball loss through the magnitude of improvements on a smaller subset of dates rather than by winning on the majority of dates.

The Day 25 traceability layer confirms that these claims are consistent with the frozen final artifacts.

For the final report, release notes, or presentation, the preferred conclusion is therefore to describe the criterion-specific strengths and limitations of the three models instead of declaring a universal winner.