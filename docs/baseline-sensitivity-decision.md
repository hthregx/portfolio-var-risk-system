# Baseline Sensitivity Decision — 13/08/2026

## Scope and pre-registration

The 13/08 sensitivity analysis compares a deliberately small parameter grid for Historical Simulation and EWMA before downstream configuration is finalized.

The common eligible universe contains 1,137 target dates from 2021-12-31 through 2026-07-28. Eligibility requires at least 500 portfolio-return observations strictly before each target date so that Historical windows 125, 250, and 500 can be evaluated on identical dates.

The chronological validation subset was fixed before sensitivity metrics were calculated. It contains the first 739 common dates, from 2021-12-31 through 2024-12-17. The remaining 398 dates, from 2024-12-18 through 2026-07-28, were reserved from parameter selection.

Sensitivity decisions in this record use validation metrics only. The reserved-later subset is not described as a pristine untouched test set because the full evaluation period had already been inspected during the 12/08 canonical Historical-versus-EWMA comparison.

## Decision rule

The primary decision evidence is:

1. absolute calibration distance, defined as `abs(violation_rate - 0.05)`; and
2. Pinball Loss at `alpha = 0.05`.

Average VaR and the observed minimum-to-maximum VaR range are secondary descriptive evidence. A lower Average VaR alone is not a selection criterion.

For each method, the existing baseline is retained when alternatives show mixed trade-offs. A parameter change is recommended only when a candidate improves both primary criteria relative to the existing baseline. These validation comparisons are descriptive and are not evidence of statistical or universal superiority.

## Validation experiments

| Experiment | Method | Parameter | Violations | Violation Rate | Calibration Distance | Pinball Loss | Average VaR | Min VaR | Max VaR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B01 | Historical | window=125 | 45 | 6.089309878% | 1.089310 pp | 0.002117106095 | 2.673516639% | 1.239388829% | 4.955766686% |
| B02 | Historical | window=250 | 37 | 5.006765900% | 0.006766 pp | 0.002144803137 | 2.852501286% | 1.708838523% | 4.378556664% |
| B03 | Historical | window=500 | 37 | 5.006765900% | 0.006766 pp | 0.002174145160 | 2.797090791% | 2.141247354% | 3.101156816% |
| B04 | EWMA | decay=0.90 | 40 | 5.412719892% | 0.412720 pp | 0.001910811794 | 2.542613170% | 0.990086500% | 6.043165193% |
| B05 | EWMA | decay=0.94 | 43 | 5.818673884% | 0.818674 pp | 0.001929922933 | 2.580969253% | 1.143839905% | 5.666074912% |
| B06 | EWMA | decay=0.97 | 43 | 5.818673884% | 0.818674 pp | 0.002000327134 | 2.642960194% | 1.401685299% | 5.039815334% |

Every experiment contains exactly 739 forecasts on the same validation target-date sequence.

## Historical Simulation decision

Retain `window=250`.

B01 (`window=125`) produces a lower Pinball Loss than B02, but its violation rate is 6.089309878%, placing it 1.089310 percentage points from the nominal 5% rate versus only 0.006766 percentage points for B02.

B03 (`window=500`) has the same violation count and calibration distance as B02 but a higher Pinball Loss, 0.002174145160 versus 0.002144803137.

Neither B01 nor B03 improves both primary criteria relative to B02. The validation evidence therefore supports retaining the existing Historical baseline of 250 observations.

## EWMA decision

Recommend `decay=0.90` as the validation-selected candidate for downstream configuration.

Relative to the canonical-before B05 setting `decay=0.94`, B04 improves absolute calibration distance from 0.818674 percentage points to 0.412720 percentage points and reduces Pinball Loss from 0.001929922933 to 0.001910811794. The B04-minus-B05 Pinball Loss difference is -0.000019111139.

Average VaR decreases by 0.038356 percentage points under B04 relative to B05. This is secondary descriptive evidence rather than the basis for selection. B04 also exhibits a wider observed VaR range, from 0.990086500% to 6.043165193%, which should be retained as a robustness consideration.

B06 (`decay=0.97`) does not improve either primary criterion relative to B05.

This record recommends `decay=0.90` from validation evidence only. It does not assert that the production source default has already changed. `src/models/ewma_var.py` remains at its existing `decay=0.94` default in this analysis step; any production or configuration change is a separate traceable integration action.

## Safeguards and limitations

All B01-B06 sensitivity metrics use only the pre-registered 739-row validation subset. Reserved-later metrics were excluded from parameter selection.

Historical candidates use the same target dates and strict violation convention `target_return < quantile_return`. EWMA candidates use the same validation target dates and the production EWMA forecasting function.

No formal coverage test or statistical significance test is used in this decision. No claim is made that one parameter is universally superior across future regimes.

The reserved-later period may support post-selection robustness analysis, but it must not be presented as a pristine untouched test set because the broader evaluation sample was previously inspected.

## Reproducibility

The machine-readable experiment registry is:

`results/sensitivity_experiments.csv`

It contains exactly six experiments B01-B06, the common-universe and validation metadata, the verified metrics, each parameter's pre-analysis canonical status, and the validation recommendation.

The B02 `window=250` validation forecasts were independently checked against the canonical Historical backtest with maximum absolute quantile and VaR differences of `1.596e-16`.

## Prompt provenance

The AI-assisted workflow was constrained by a fixed B01-B06 parameter grid, a fixed 739-row chronological validation split established before sensitivity metrics, exact common-date and no-look-ahead contracts, exact deterministic numerical checkpoints, and exclusion of reserved-later results from parameter selection.

The workflow also required assertion-based independent verification before documentation and explicitly prohibited automatic staging or committing. Parameter recommendations were derived from the locked decision rule rather than from unrestricted post-hoc search.
