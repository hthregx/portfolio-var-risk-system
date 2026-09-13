# Day 27 Final Release Handoff - Member A

## Objective

This document hands off the frozen quantitative release package for the canonical one-day 95% portfolio VaR system.

Day 27 is a reproducibility and release-handoff step. It does not reopen model development, model selection, feature engineering, parameter tuning, the evaluation period, or canonical prediction generation.

The Day 27 work verifies that the final conclusions can be reconstructed from the frozen release evidence without rerunning the forecasting models.

## Release Scope

The canonical system evaluates an equal-weight portfolio of HPG, FPT, and MWG.

The frozen contract is:

- forecast target: portfolio simple return;
- forecast horizon: one trading day;
- confidence level: 95%;
- lower-tail probability: `alpha = 0.05`;
- evaluation period: 2024-12-18 through 2026-07-28;
- common target dates per method: 398;
- canonical methods: 3;
- canonical prediction rows: 1,194;
- strict violation rule: `actual_return < quantile_return`;
- reported VaR rule: `VaR = max(0, -quantile_return)`.

The frozen configurations are:

| Method | Configuration | Core specification |
| --- | --- | --- |
| Historical Simulation | `historical_w250` | 250-observation rolling window |
| EWMA | `ewma_d094` | decay `0.94`, expanding estimation, Normal zero-mean assumption |
| Gradient Boosting Quantile Regression | `gb_G04` | 100 estimators, learning rate `0.03`, depth `2`, min leaf `5`, subsample `1.0`, seed `42` |

The canonical Gradient Boosting configuration uses exactly seven return-history features:

1. `return_lag_1`
2. `return_lag_2`
3. `return_lag_5`
4. `rolling_vol_5`
5. `rolling_vol_20`
6. `rolling_vol_60`
7. `drawdown`

The canonical G04 evaluation does not use the separate market/liquidity feature specification explored in other validation work.

## Model-Freeze Boundary

The model freeze remains authoritative.

The release workflow must not:

- add a new algorithm;
- add a new feature;
- retune a frozen parameter;
- change the frozen evaluation period;
- rewrite canonical final predictions solely for release packaging.

EWMA decay `0.90` produced stronger validation evidence than `0.94`, but `0.90` was not adopted for the canonical final evaluation. The retained `0.94` configuration must not be described as the validation winner.

The frozen G04 configuration is retained because it is the canonical evaluated Gradient Boosting configuration. This is not a claim that G04 is the global winner across every prior candidate experiment.

## Day 27 Reproducibility Checkpoint

Day 27 introduces a deterministic machine-readable checkpoint:

- builder: `scripts/build_final_reproducibility_checkpoint_a.py`;
- output: `results/final_reproducibility_checkpoint_a.csv`;
- checkpoint checks: 28;
- PASS checks: 28;
- FAIL checks: 0;
- duplicate check IDs: 0;
- numerical tolerance: `1e-12`.

The checkpoint covers:

| Category | Checks |
| --- | ---: |
| Evaluation contract | 6 |
| Freeze contract | 1 |
| Frozen configuration | 4 |
| Frozen inputs | 1 |
| Recomputed quantitative evidence | 7 |
| Recomputed pairwise evidence | 3 |
| Prior evidence gates | 3 |
| Risk semantics | 2 |
| Release boundary | 1 |

The checkpoint output was reproduced byte-for-byte across two consecutive runs in the current tested environment.

Current tested-environment SHA256 values are:

- builder: `ec0d26f3014f4c8b1a0f1343ee553685d3494bd728c6d35ca5f19885f3a4484b`;
- checkpoint CSV: `1e89bbdfa71f1d71aa41f86ce33e6aa8da24ae5022d6dc2e0f96f6570dd38b1d`.

These byte-level hashes are environment-specific release evidence. Repository history should be used as the durable source of identity after commit and merge.

## Independent Quantitative Integrity Audit

The Day 27 checkpoint was independently audited without invoking the Day 27 builder as the calculation engine.

The audit recomputed directly from `results/final_predictions.csv`:

- forecast counts;
- violation counts;
- violation rates;
- mean pinball losses;
- average VaR;
- minimum VaR;
- maximum VaR;
- criterion-specific leaders;
- three pairwise daily pinball-loss comparisons.

The independent audit reported zero failures.

Small floating-point differences at the final machine-precision digits were within the locked tolerance of `1e-12` and did not change any result, ranking, count, or interpretation.

## Frozen Quantitative Evidence

| Method | Forecasts | Violations | Violation rate | Mean pinball loss | Average VaR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical Simulation | 398 | 27 | 6.7839% | 0.001896464233 | 2.1330% |
| EWMA | 398 | 22 | 5.5276% | 0.001968321253 | 2.5084% |
| Gradient Boosting | 398 | 24 | 6.0302% | 0.001758130951 | 2.3349% |

Criterion-specific conclusions are:

- EWMA is closest to the nominal 5% violation rate.
- Gradient Boosting has the lowest mean pinball loss.
- Historical Simulation has the lowest average reported VaR.

These are separate criteria. The frozen final evaluation does not declare a single universal winner.

Lower average VaR is not evidence of automatic model superiority.

## Pairwise Reproducibility Evidence

| Comparison | Left better | Right better | Ties | Mean left-minus-right loss |
| --- | ---: | ---: | ---: | ---: |
| Gradient Boosting vs Historical Simulation | 151 | 247 | 0 | -0.0001383332817084946 |
| Gradient Boosting vs EWMA | 231 | 167 | 0 | -0.00021019030172751563 |
| EWMA vs Historical Simulation | 131 | 267 | 0 | +0.00007185702001901683 |

The Gradient Boosting versus Historical Simulation result demonstrates why daily win frequency and mean loss can point in different directions. Historical Simulation wins on more individual dates, while Gradient Boosting still has the lower mean pinball loss because the magnitudes of gains and losses differ.

Pairwise evidence provides context for the aggregate metrics rather than a separate universal ranking.

## Evidence Chain

| Layer | Artifact | Role |
| --- | --- | --- |
| Canonical forecasts | `results/final_predictions.csv` | Common prediction panel and realized returns |
| Frozen comparison | `results/final_metric_comparison.csv` | Aggregate quantitative metrics and criterion leaders |
| Pairwise evidence | `results/final_pairwise_summary.csv` | Daily pinball-loss comparison summary |
| Model freeze | `configs/model_freeze.yaml` | Frozen model, evaluation, and release policy |
| Day 24 | `results/final_evidence_validation_a.csv` | 26-check evidence validation gate |
| Day 25 | `results/final_claim_traceability_a.csv` | 24-claim traceability gate |
| Day 26 | `results/final_acceptance_matrix_a.csv` | 30-check release acceptance gate |
| Day 26 | `results/final_quantitative_summary_a.csv` | Presentation-ready projection of frozen metrics |
| Day 27 | `results/final_reproducibility_checkpoint_a.csv` | 28-check reproducibility checkpoint |

The prior Member A evidence gates remain:

- Day 24: 26 / 26 PASS;
- Day 25: 24 / 24 PASS;
- Day 26: 30 / 30 PASS;
- Day 27 checkpoint: 28 / 28 PASS.

## Source-of-Truth Order

For quantitative release questions, use the following precedence:

1. `configs/model_freeze.yaml` for the frozen release contract and configuration boundary.
2. `results/final_predictions.csv` for canonical target-level forecasts and realized returns.
3. `results/final_metric_comparison.csv` for frozen aggregate metrics.
4. `results/final_pairwise_summary.csv` for frozen pairwise summary evidence.
5. `results/final_reproducibility_checkpoint_a.csv` for Day 27 release verification.
6. Day 24-26 Member A evidence files for validation, traceability, presentation, and acceptance context.

Presentation documents summarize these sources; they do not replace the machine-readable artifacts.

## Reproduction Workflow

From the repository root, rebuild the Day 27 checkpoint with:

```powershell
python scripts/build_final_reproducibility_checkpoint_a.py
```

Expected result:

```text
Checks: 28
PASS: 28
FAIL: 0
```

After the Day 27 automated QA suite is present, use:

```powershell
python -m pytest tests/test_final_reproducibility_checkpoint_a.py -q
python -m pytest -q
```

No model runner is required to regenerate the Day 27 checkpoint.

## Presentation-Safe Interpretation

When presenting the final results:

- describe EWMA as the calibration leader only;
- describe Gradient Boosting as the mean-pinball-loss leader only;
- describe Historical Simulation as the method with the lowest average reported VaR only;
- do not convert these criterion-specific results into a universal model ranking;
- do not claim that lower average VaR means better predictive performance;
- do not claim that EWMA `0.94` won validation against `0.90`;
- do not describe the final evaluation as a pristine, never-inspected test set;
- do not treat runtime as a predictive-quality ranking criterion.

The defensible final message is that the three frozen methods exhibit different strengths under different criteria.

## Data Freshness Boundary

The current canonical modeling snapshot ends on `2026-07-28`.

That date is the end of the frozen dataset used by the canonical evaluation and must remain distinct from any later data-freshness update.

The project workflow plans a separate final freshness snapshot after the `2026-08-28` market close so that the submission can also contain the most recent complete market data available near the project deadline.

That freshness workflow must not silently overwrite or reinterpret the frozen `2026-07-28` canonical evaluation.

If later data are downloaded, they must be clearly labeled as a separate freshness or operational snapshot unless the project explicitly defines and documents a new evaluation workflow.

Day 27 does not perform that final data refresh.

## Evaluation Limitation

The frozen evaluation period is 2024-12-18 through 2026-07-28 with 398 common target dates per method.

The evaluation was reserved for the documented final workflow and was not used for parameter selection, but it must not be described as a pristine, never-inspected test set because broader inspection occurred earlier in the project.

The conclusions are specific to the equal-weight HPG/FPT/MWG portfolio, one-day horizon, 95% VaR definition, frozen evaluation period, three retained configurations, and observed market conditions represented in that sample.

## Provenance Limitation

`results/final_run_metadata.json` records an earlier repository HEAD from the runtime that generated the canonical final artifacts.

Later release work finalized source code, validation, documentation, acceptance, and reproducibility evidence without changing the canonical prediction content.

This is a provenance completeness limitation. It is not evidence that the frozen predictions are incorrect, and historical runtime metadata should not be rewritten solely to make the recorded Git commit appear newer.

## Handoff Checklist

At release handoff, Member A should verify:

- the Day 27 checkpoint remains 28 / 28 PASS;
- the Day 27 automated QA suite passes;
- the full repository suite passes;
- only Day 27 Member A-owned files are staged;
- no canonical model artifact is rewritten;
- no unrelated notebook change is included;
- the integration base is current before PR creation;
- the PR targets `feature/eda-analysis`;
- the release branch is deleted only after merge verification.

Cross-review by Member B is useful when available but is non-blocking for Member A's independent release workflow.

## Release Handoff Interpretation

The Day 27 reproducibility work supports a stronger claim than simple file existence: the principal quantitative conclusions can be reconstructed independently from the frozen target-level prediction panel and matched back to the release evidence within the locked numerical tolerance.

This is a reproducibility and release-readiness result, not a new model experiment.

The release remains frozen around the same three canonical methods and the same evaluation panel.

No universal model winner is declared.
