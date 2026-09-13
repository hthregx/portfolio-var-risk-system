# Day 26 Final Presentation QA — Member B

## Objective

Validate the final presentation narrative against frozen canonical release evidence without changing models, metrics, predictions, or evaluation scope.

## Frozen Presentation Preflight Evidence

Day 26 Member B preflight was verified before release packaging:

- Branch: `release/day26-presentation-evidence-b`
- Integration base: `feature/eda-analysis`
- Merge-base ancestry check: PASS
- Initial Day 26 working tree: clean before B-owned artifacts were created
- Day 25 traceability gate: 24/24 PASS
- Day 24 evidence-validation gate: 26/26 PASS
- Canonical model-freeze validation: PASS
- Canonical prediction rows: 1194
- Methods: 3
- Forecasts per method: 398
- Evaluation range: 2024-12-18 through 2026-07-28
- Duplicate `(method, target_date)` rows: 0
- Same-date `actual_return` consistency: PASS
- Strict violation semantics: PASS
- VaR identity semantics: PASS

The four untracked Day 26 B-owned artifacts appearing after implementation are expected working outputs and do not contradict the clean-tree preflight performed before Day 26 work began.

## Frozen Presentation Contract

- Portfolio: equal-weight HPG/FPT/MWG
- Horizon: 1 trading day
- Confidence: 95%
- Alpha: 0.05
- Historical: `historical_w250`
- EWMA: `ewma_d094`
- Gradient Boosting: `gb_G04`
- Prediction rows: 1194
- Methods: 3
- Forecasts per method: 398
- Evaluation: 2024-12-18 through 2026-07-28
- Violation: `actual_return < quantile_return`
- VaR: `max(0, -quantile_return)`

EWMA decay `0.94` is canonical but must not be described as the validation winner.

## Aggregate Quantitative Evidence

| Method | Violations | Violation rate | Mean pinball loss | Average VaR |
| --- | ---: | ---: | ---: | ---: |
| Historical Simulation | 27 | 6.7839% | 0.001896464233 | 2.1330% |
| EWMA | 22 | 5.5276% | 0.001968321253 | 2.5084% |
| Gradient Boosting | 24 | 6.0302% | 0.001758130951 | 2.3349% |

Different evaluation criteria answer different questions. There is no single overall winner.

## Calibration Story

EWMA records 22 violations from 398 forecasts, or 5.5276%.

It is closest to the nominal 5% violation probability and is therefore the criterion-specific calibration leader.

This does not mean EWMA is the best model overall.

## Quantile Accuracy Story

Gradient Boosting has the lowest mean pinball loss:

`0.001758130951`

Therefore, Gradient Boosting is the criterion-specific leader for mean pinball loss.

It must not be described as the overall winner.

## Risk Magnitude Story

Historical Simulation has the lowest average reported VaR at approximately 2.1330%.

Lower average VaR is not evidence of automatic model superiority.

It must not be described as the safest or best model.

## Pairwise Evidence

### Gradient Boosting vs Historical Simulation

- GB lower daily pinball loss: 151 dates
- Historical lower daily pinball loss: 247 dates
- Mean GB - Historical delta: -0.0001383332817084946

Gradient Boosting loses the majority daily-win count but still has lower mean pinball loss. The magnitude of improvements on some dates is large enough to offset losing on more individual dates.

It is incorrect to say that Gradient Boosting wins more days than Historical Simulation.

### Gradient Boosting vs EWMA

- GB lower daily loss: 231 dates
- EWMA lower daily loss: 167 dates
- Mean delta: -0.00021019030172751563

Both daily win count and mean pinball loss favor Gradient Boosting for this pair.

### EWMA vs Historical Simulation

- EWMA lower daily loss: 131 dates
- Historical lower daily loss: 267 dates
- Mean delta: +0.00007185702001901683

EWMA has better violation-rate calibration, while Historical Simulation has lower mean pinball loss than EWMA.

This demonstrates that different evaluation criteria answer different questions.

## Presentation Language Guardrails

Allowed:

- EWMA is closest to the nominal 5% violation rate.
- Gradient Boosting has the lowest mean pinball loss.
- Historical Simulation has the lowest average reported VaR.

Not allowed:

- EWMA is the best model.
- Gradient Boosting is the overall winner.
- Historical Simulation is the safest/best model.
- Lower average VaR proves superiority.

There is no single overall winner.

## Reproducibility Boundary

The Day 26 presentation registry is built only from frozen tracked release artifacts.

It does not run training, forecasting, feature engineering, retuning, or private processed data.

Byte-for-byte figure reproducibility is limited to the tested environment.

## Evaluation Limitations

The evaluation period is not a pristine never-inspected test set.

Results are specific to the frozen portfolio, model configurations, evaluation period, and observed market conditions.

They should not be generalized automatically to other portfolios, horizons, confidence levels, or regimes.

## Provenance Boundary

Final runtime metadata has a known provenance completeness limitation: the recorded runtime HEAD predates a later source commit, while canonical prediction content remained unchanged.

Historical metadata must not be rewritten to manufacture newer provenance.

## Presentation Handoff

Presentation claims should be sourced from:

- `results/final_presentation_evidence_b.csv`
- `results/final_metric_comparison.csv`
- `results/final_pairwise_summary.csv`
- `results/final_predictions.csv`

Only criterion-specific conclusions should be presented. The frozen evidence does not support a universal model ranking.