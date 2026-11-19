# Final Code Submission — Portfolio VaR Risk System

**Submission snapshot:** 2026-11-19  
**Release source branch:** `submission/c1-frozen-20261119`  
**Frozen C1 reference commit:** `a21a8556413d0c0c40fe03b3b6597389a5abf20d`  
**Release base / main integration commit:** `8784c56b69fd8076dac8b22f183de4acd6ebcdb1`  
**Prepared from repository commit:** `a21a8556413d0c0c40fe03b3b6597389a5abf20d`

## 1. Scope of the submission

One-day 95% Value at Risk (VaR) for an equal-weight portfolio of HPG, FPT and MWG. The production/research system includes data validation, VaR estimation, backtesting, dashboard presentation and frozen research artifacts.

**Final highlighted risk model: C1 Gradient Boosting Quantile Regression (12 features).** Historical Simulation, EWMA, and earlier G04 remain as **historical comparators**. The prior Phase 1 G04 research contract is preserved as historical evidence; it is not replaced or silently rewritten.

The later experiments involving 7/30-session features, calendar seasonality, volatility normalization or recency weighting are **research explorations, not the selected final model**. Those experimental branches must not be presented as a confirmed improvement.

## 2. Frozen C1 specification

| Field | Value |
|---|---|
| Horizon | 1 trading session |
| Confidence | 95% (5% lower-tail quantile) |
| Source data cutoff | 2026-11-06 |
| Portfolio | HPG/FPT/MWG, equal weight |
| Estimator | GradientBoostingRegressor(loss=quantile) |
| Features | 12, using historical returns/risk statistics |
| Estimators | 150 |
| Learning rate | 0.03 |
| Max depth | 2 |
| Min leaf | 10 |
| Development observations | 1,179 |
| Retrospective holdout observations | 468 |
| Input SHA-256 | `14ac5dd4ce7dd6090420b9cee06c8ac40935c54691b4daf95d063b71a075eccc` |

## 3. Reproduced evidence

| Metric | Result |
|---|---:|
| CV mean Pinball Loss | 0.002080267447 |
| Holdout Pinball Loss | 0.001636961294 |
| VaR breaches | 25 / 468 |
| Breach rate | 5.3419% |
| Kupiec p-value | 0.737029 |
| Christoffersen independence p-value | 0.534649 |

The canonical C1 predictions were reproduced byte-for-byte at the numeric-value level, with zero maximum absolute quantile difference in the recorded validation. **A VaR quantile is not a point prediction of next-day actual returns.** A breach occurs when the realized portfolio return falls below the forecasted 5% quantile.

## 4. Important files

- `app/app.py`: Streamlit dashboard.
- `configs/c1_final_frozen.json`: immutable C1 model/data contract.
- `scripts/final_c1/run_final_c1.py`: reproducibility runner.
- `results/final_C1_frozen_2026-11-19/01_C1_VaR_with_Breaches.png`: VaR chart with 25 breach markers.
- `results/final_C1_frozen_2026-11-19/02_C1_Coverage.png`: coverage diagnostics.
- `results/final_C1_frozen_2026-11-19/FINAL_C1_BACKTEST.json`: risk statistics.
- `results/final_C1_frozen_2026-11-19/FINAL_C1_HOLDOUT.csv`: forecast and observed returns.
- `tests/test_final_c1_frozen.py`: frozen-model verification tests.

## 5. Reproduce and review

Run from the repository root after installing `requirements.txt`:

```bash
python -m pytest -q
python scripts/final_c1/run_final_c1.py --root . --verify-only
python -m streamlit run app/app.py
```

The final submission process must also verify the branch from a **fresh GitHub clone**, not only the local working directory.

## 6. Scientific boundaries

- The source snapshot is from **November 6, 2026**, not current live market data.
- The C1 holdout is historical and overlaps data inspected during Phase 1; it must **not** be called pristine or never-seen.
- Kupiec and independence p-values above 0.05 indicate no rejection of the tested hypotheses; they are **not proof** of perfect VaR quality.
- Older Phase 1 documentation and frozen research contracts remain available and are historical context rather than the single C1 result being highlighted in the submission.
- Do not change any locked C1 metric, input hash or reference prediction merely to improve a graph.

## 7. Closing conditions

Only describe the **entire repository** as finalized after full test-suite PASS, C1 `--verify-only` PASS, Streamlit smoke PASS, a clean fresh GitHub clone replay and final GitHub branch/PR integration. Keep the verified release commit SHA and submit that exact ref or tag.
