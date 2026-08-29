# Day 27 Final Demo and Defense Runbook — Member B

## Objective

Provide a concise, evidence-grounded guide for final submission, live demo, presentation, and defense without changing frozen models, configurations, metrics, or canonical outputs.

## Project in 30 Seconds

The project compares Historical Simulation, EWMA, and Gradient Boosting Quantile Regression for one-day 95% VaR forecasting on an equal-weight HPG/FPT/MWG portfolio.

The frozen evaluation contains 398 common target dates per method, 1,194 canonical predictions in total.

The methods show different strengths, so the project reports criterion-specific conclusions rather than declaring one universal winner.

## Two-Minute Executive Summary

The final system uses:

- one-day forecast horizon;
- 95% VaR;
- alpha = 0.05;
- equal-weight HPG/FPT/MWG portfolio;
- Historical Simulation: `historical_w250`;
- EWMA: `ewma_d094`;
- Gradient Boosting: `gb_G04`.

Frozen evaluation results:

- Historical Simulation: 27 violations, 6.7839%, pinball 0.001896464233, average VaR 2.1330%;
- EWMA: 22 violations, 5.5276%, pinball 0.001968321253, average VaR 2.5084%;
- Gradient Boosting: 24 violations, 6.0302%, pinball 0.001758130951, average VaR 2.3349%.

Criterion-specific findings:

- EWMA is closest to the nominal 5% violation rate;
- Gradient Boosting has the lowest mean pinball loss;
- Historical Simulation has the lowest average reported VaR.

There is no single overall winner.

## Five-Minute Technical Walkthrough

1. Explain the frozen evaluation contract and portfolio.
2. Show the three canonical configurations.
3. Show the shared 398-date evaluation panel.
4. Explain violation rate as a calibration criterion.
5. Explain pinball loss as lower-tail quantile accuracy.
6. Explain average VaR as reported risk magnitude.
7. Show pairwise evidence to explain daily-win count versus mean-loss magnitude.
8. Finish with limitations, reproducibility boundary, and criterion-specific conclusions.

## Frozen Quantitative Facts

- Portfolio: equal-weight HPG/FPT/MWG.
- Horizon: one trading day.
- Confidence: 95%.
- Alpha: 0.05.
- Evaluation: 2024-12-18 through 2026-07-28.
- Forecasts per method: 398.
- Total predictions: 1,194.
- Historical config: `historical_w250`.
- EWMA config: `ewma_d094`.
- GB config: `gb_G04`.

Aggregate evidence:

| Method | Violations | Violation Rate | Mean Pinball Loss | Average VaR |
| --- | ---: | ---: | ---: | ---: |
| Historical Simulation | 27 | 6.7839% | 0.001896464233 | 2.1330% |
| EWMA | 22 | 5.5276% | 0.001968321253 | 2.5084% |
| Gradient Boosting | 24 | 6.0302% | 0.001758130951 | 2.3349% |

## Model-by-Model Explanation

### Historical Simulation

Historical Simulation is the empirical, non-parametric baseline using the frozen 250-observation rolling window.

It has the lowest average reported VaR, but lower average VaR does not imply superior calibration or overall model superiority.

### EWMA

EWMA is a volatility-adaptive parametric approach using canonical decay `0.94`.

Its violation rate of 5.5276% is closest to the nominal 5% rate.

EWMA 0.94 must not be described as the validation winner. The 0.90 alternative had stronger validation evidence, while 0.94 remained the retained canonical baseline for final continuity.

### Gradient Boosting

Gradient Boosting is the nonlinear conditional quantile model using frozen configuration `gb_G04`.

It has the lowest mean pinball loss over the frozen evaluation window.

This makes it the criterion-specific pinball-loss leader, not an overall model winner.

## Pairwise Interpretation

### GB vs Historical Simulation

- GB lower daily loss: 151 dates
- Historical lower daily loss: 247 dates
- Mean delta: -0.0001383332817084946

GB loses on more individual dates but still has lower mean pinball loss because improvement magnitude and improvement frequency are different concepts.

### GB vs EWMA

- GB lower daily loss: 231 dates
- EWMA lower daily loss: 167 dates
- Mean delta: -0.00021019030172751563

Both daily-win count and mean pinball loss favor GB in this pair.

### EWMA vs Historical Simulation

- EWMA lower daily loss: 131 dates
- Historical lower daily loss: 267 dates
- Mean delta: +0.00007185702001901683

EWMA has better violation-rate calibration, while Historical Simulation has lower mean pinball loss than EWMA.

Different criteria answer different questions.

## Expected Defense Questions

### Q1 — Why one-day 95% VaR?

Because the project scope was frozen around one-day 95% VaR. It is the project evaluation contract, not a claim that this horizon and confidence level are universally optimal.

### Q2 — Why equal weights?

Equal weighting keeps portfolio construction fixed so the study focuses on comparing risk-forecasting methods rather than introducing a separate portfolio-optimization problem.

### Q3 — Why simple portfolio returns?

The canonical forecasting target is portfolio simple return, and the VaR sign convention is defined on that target.

### Q4 — Why not average asset log returns?

Exact portfolio log return is `log1p(portfolio_simple_return)`, not the weighted average of individual asset log returns.

### Q5 — Why these three models?

Historical Simulation is an empirical baseline, EWMA is volatility-adaptive and parametric, and Gradient Boosting represents a nonlinear conditional quantile approach.

A newer or more complex method is not automatically better.

### Q6 — Which model is best?

There is no single overall winner.

EWMA is closest to the nominal 5% violation rate, Gradient Boosting has the lowest mean pinball loss, and Historical Simulation has the lowest average reported VaR.

### Q7 — Why does GB beat Historical in mean pinball while losing more days?

GB wins 151 dates versus 247 for Historical, but its mean delta is negative.

The magnitude of GB improvements on some dates is sufficient to outweigh losing on a larger number of individual dates.

### Q8 — Why retain EWMA 0.94 when 0.90 had stronger validation evidence?

The 0.90 alternative had stronger validation evidence.

The 0.94 configuration remained the retained canonical baseline for final continuity.

Do not claim that 0.94 won validation.

### Q9 — Is the final evaluation a clean untouched test set?

No.

It was reserved for the documented final workflow but should not be described as a pristine never-inspected test set because broader inspection occurred earlier in the project.

### Q10 — Does lower average VaR mean safer or better?

No.

Lower average VaR describes lower reported risk magnitude and may also imply a less conservative estimate.

It must be interpreted together with calibration and pinball-loss evidence.

### Q11 — What is the provenance limitation?

The runtime metadata records a HEAD that predates a later source commit.

Canonical predictions remained unchanged.

This is a provenance completeness limitation, not evidence that the predictions are incorrect.

### Q12 — How reproducible is the project?

Frozen CSV and model evidence are deterministic under the project checks.

Day24 figures reproduced byte-for-byte in the tested environment.

Universal PNG byte identity across every operating system or rendering stack is not claimed.

## Claims We Must Not Make

Do not claim:

- there is one universally best model;
- EWMA is the overall best model;
- Gradient Boosting is the overall winner;
- Historical Simulation is the safest or best model;
- lower average VaR proves superiority;
- EWMA 0.94 won validation;
- the final evaluation is a pristine untouched test set;
- figure bytes are guaranteed identical across every environment.

## Reproducibility Demonstration

Preferred lightweight demo:

```bash
python scripts/build_final_traceability_a.py
python scripts/build_final_presentation_evidence_b.py
python scripts/build_final_submission_readiness_b.py

python -m pytest tests/test_final_traceability_a.py -q
python -m pytest tests/test_final_presentation_evidence_b.py -q
```
## Failure-Recovery Plan

If the live demo fails because of the environment:

1. do not modify the model during the presentation;
2. use frozen tracked artifacts;
3. show the validation and evidence CSVs;
4. show Git commit and PR history;
5. show previously recorded test results;
6. explain that an environment problem is different from a model-evidence problem.
## Final Submission Checklist

Before submission:

- confirm `final_defense_facts_b.csv` is 20/20 PASS;
- confirm `final_submission_manifest_b.csv` is 23/23 PASS;
- confirm Day26 evidence is 28/28 PASS;
- confirm Day25 traceability is 24/24 PASS;
- confirm Day24 evidence is 26/26 PASS;
- confirm canonical predictions remain 1,194 rows;
- confirm no frozen canonical artifact was modified;
- run Day27 task tests;
- run the full repository test suite;
- run `git diff --check`;
- confirm only B-owned Day27 files are changed;
- confirm PR CI passes before merge.

## Final Handoff

The final presentation and defense should use frozen tracked evidence rather than rerunning or changing the models.

The release interpretation is:

- EWMA leads calibration;
- Gradient Boosting leads mean pinball loss;
- Historical Simulation has the lowest average reported VaR;
- there is no single overall winner.

All final claims must remain traceable to frozen repository artifacts.
