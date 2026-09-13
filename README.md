# Portfolio VaR Risk System

A reproducible quantitative risk-management project for forecasting and
backtesting **one-day 95% Value at Risk (VaR)** for an equal-weight Vietnamese
equity portfolio containing **HPG, FPT, and MWG**.

The project compares three frozen VaR approaches:

- Historical Simulation
- EWMA
- Gradient Boosting Quantile Regression

The repository includes the modeling pipeline, walk-forward evaluation,
validation tests, frozen research evidence, operational forecasts, and a
single-page Streamlit risk dashboard.

---

## 1. Project Objective

The system estimates the next-trading-day downside risk of an equal-weight
portfolio of:

- HPG
- FPT
- MWG

The official portfolio target is the **portfolio simple return**.

For each asset:

`R_i,t = P_i,t / P_i,t-1 - 1`

The equal-weight portfolio return is:

`R_p,t = sum(w_i * R_i,t)`

with:

`w_HPG = w_FPT = w_MWG = 1/3`

The derived portfolio log return is:

`r_p,t = log(1 + R_p,t)`

Portfolio log return is retained for descriptive analysis. The canonical VaR
forecast target is the portfolio simple return.

VNINDEX is used as market context / benchmark information and is not a
constituent of the portfolio.

---

## 2. Frozen Research Contract

The final research configuration is frozen to prevent retrospective
model-selection changes after evaluation.

| Item | Frozen specification |
| --- | --- |
| Portfolio | Equal-weight HPG / FPT / MWG |
| Forecast horizon | 1 trading day |
| Confidence level | 95% |
| Lower-tail probability | 5% |
| Canonical data cutoff | 2026-07-28 |
| Evaluation window | 2024-12-18 to 2026-07-28 |
| Evaluation targets | 398 per model |
| Total prediction rows | 1,194 |
| Violation rule | `actual_return < quantile_return` |
| Reported VaR | `max(0, -quantile_return)` |

The evaluation period must not be described as a pristine, never-inspected
test set because broader inspection occurred earlier in the research process.

---

## 3. Models

### Historical Simulation

Canonical configuration:

- configuration: `historical_w250`
- rolling window: 250 trading observations
- alpha: 0.05
- one-day forecast horizon

Historical Simulation estimates the lower-tail empirical quantile from the
rolling historical return window.

### EWMA

Canonical configuration:

- configuration: `ewma_d094`
- decay: 0.94
- expanding estimation
- Normal distribution
- zero conditional mean
- first squared return initialization
- alpha: 0.05

A decay of 0.90 produced stronger validation evidence in an earlier comparison.
The canonical value 0.94 is retained for continuity with the frozen final
evaluation and must not be interpreted as having won that validation
comparison.

### Gradient Boosting Quantile Regression

Canonical configuration:

- configuration: `gb_G04`
- quantile loss
- alpha: 0.05
- `n_estimators = 100`
- `learning_rate = 0.03`
- `max_depth = 2`
- `min_samples_leaf = 5`
- `subsample = 1.0`
- `random_state = 42`
- expanding refit for each target date

Canonical G04 uses exactly seven return-history features:

1. `return_lag_1`
2. `return_lag_2`
3. `return_lag_5`
4. `rolling_vol_5`
5. `rolling_vol_20`
6. `rolling_vol_60`
7. `drawdown`

Market-index, price-range, and trading-volume features are not part of the
canonical G04 specification.

---

## 4. Final Walk-Forward Evaluation

All three methods are evaluated over the same 398 target dates.

| Model | Violations | Violation Rate | Mean Pinball Loss | Average VaR |
| --- | ---: | ---: | ---: | ---: |
| Historical Simulation | 27 / 398 | 6.7839% | 0.001896464233 | 2.1330% |
| EWMA | 22 / 398 | 5.5276% | 0.001968321253 | 2.5084% |
| Gradient Boosting G04 | 24 / 398 | 6.0302% | 0.001758130951 | 2.3349% |

### Interpretation

The results are criterion-specific:

- **Calibration:** EWMA is closest to the nominal 5% violation rate.
- **Quantile accuracy:** Gradient Boosting G04 has the lowest mean pinball loss.
- **Average risk magnitude:** Historical Simulation has the lowest average VaR.

There is **no single overall winning model**.

A lower average VaR is not automatically evidence of a better risk model.

---

## 5. Operational Forecast

Research evaluation and operational forecasting are deliberately separated.

The canonical research cutoff remains:

`2026-07-28`

The latest included operational snapshot uses market information through:

`2026-08-28 EOD`

with forecast target:

`2026-09-03`

Latest frozen-specification one-day 95% VaR forecasts:

| Model | VaR |
| --- | ---: |
| Historical Simulation | 2.5074% |
| EWMA | 2.5758% |
| Gradient Boosting G04 | 2.1494% |

Post-cutoff operational data may be used to produce current forecasts under
the already-frozen specifications. This does not reopen model selection,
retune hyperparameters, extend the canonical backtest, or alter the frozen
research conclusions.

---

## 6. Streamlit Risk Dashboard

The project includes a full-width single-page Streamlit dashboard:

`app/app.py`

Main sections:

- **Risk Overview**
  - portfolio value input
  - VND quick-value presets
  - model-specific VaR percentages
  - estimated monetary VaR

- **Stock Explorer**
  - HPG / FPT / MWG selection
  - multiple lookback periods
  - latest close and return statistics
  - annualized volatility
  - maximum drawdown
  - best / worst daily returns
  - price history
  - return distribution
  - drawdown analysis
  - normalized constituent comparison
  - return correlation

- **Historical Backtesting**
  - realized portfolio returns
  - model VaR threshold
  - violation events
  - model-level backtesting statistics

- **Model Comparison**
  - violation-rate comparison
  - pinball-loss comparison
  - average-VaR comparison
  - criterion-specific interpretation

- **Methodology & Research Contract**
  - frozen model assumptions
  - evaluation boundaries
  - research interpretation notes

The dashboard presents existing frozen research and operational artifacts. User
interaction does not trigger model selection or hyperparameter tuning.

---

## 7. Data

### Canonical portfolio-return dataset

Canonical private processed dataset:

`data/processed/portfolio_returns.csv`

Contract:

- 1,637 observations
- start: 2020-01-03
- end: 2026-07-28
- no duplicate dates
- no missing values

Columns:

```text
date
HPG_simple_return
FPT_simple_return
MWG_simple_return
portfolio_simple_return
portfolio_log_return
```

The canonical processed dataset is intentionally **not version-controlled**.

### Operational snapshots

Versioned operational snapshots include:

```text
data/snapshots/market_data_2026-08-28.csv
data/snapshots/portfolio_returns_2026-08-28.csv
```

These are used for the operational forecast layer without changing the frozen
canonical evaluation.

---

## 8. Key Reproducibility Artifacts

Frozen model contract:

```text
configs/model_freeze.yaml
```

Final evaluation configuration:

```text
configs/final_evaluation.yaml
```

Canonical evaluation outputs:

```text
results/final_predictions.csv
results/final_metrics.csv
results/final_run_metadata.json
```

Operational forecast:

```text
results/latest_forecast_2026-08-28.csv
results/latest_forecast_metadata_2026-08-28.json
```

Canonical walk-forward runner:

```text
scripts/run_final_walk_forward.py
```

---

## 9. Repository Structure

```text
portfolio-var-risk-system/
|
|-- app/
|   `-- app.py
|
|-- configs/
|   |-- final_evaluation.yaml
|   |-- model_freeze.yaml
|   `-- ...
|
|-- data/
|   |-- sample/
|   `-- snapshots/
|
|-- docs/
|   |-- model-cards/
|   |-- releases/
|   `-- ...
|
|-- figures/
|
|-- notebooks/
|
|-- results/
|
|-- scripts/
|
|-- src/
|   |-- backtesting/
|   |-- evaluation/
|   |-- models/
|   `-- validation/
|
|-- tests/
|
|-- requirements.txt
`-- README.md
```

---

## 10. Local Setup

### Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Core dependencies include:

- NumPy
- pandas
- Matplotlib
- PyYAML
- pytest
- vnstock
- scikit-learn
- Streamlit

---

## 11. Run the Dashboard

From the repository root:

```bash
python -m streamlit run app/app.py
```

Then open the local Streamlit URL shown in the terminal.

---

## 12. Run Tests

Run the complete repository regression suite:

```bash
python -m pytest -q
```

The test suite covers data validation, portfolio construction, no-look-ahead
behavior, VaR implementations, walk-forward evaluation, frozen model
contracts, release artifacts, and reproducibility checks.

Some release tests rebuild deterministic release evidence as part of their
validation. If those generated files appear modified locally after testing,
review them before staging; they are not automatically intended as research
changes.

---

## 13. Reproduce the Canonical Final Evaluation

The canonical runner is:

```bash
python scripts/run_final_walk_forward.py
```

Reproduction requires the canonical processed portfolio dataset to exist
locally at:

```text
data/processed/portfolio_returns.csv
```

Because that dataset is not version-controlled, a fresh public clone does not
by itself contain all inputs required to regenerate the private canonical
evaluation.

The committed final artifacts allow the frozen results and their contracts to
be inspected without silently substituting another dataset.

---

## 14. Research Boundaries

The frozen release does not retroactively:

- introduce new algorithms;
- add new canonical features;
- retune model parameters;
- change the evaluation period;
- rewrite canonical predictions;
- replace G04 with a later experimental candidate;
- claim EWMA decay 0.94 won validation over 0.90;
- declare one model universally superior.

The project separates:

1. model development and validation;
2. frozen canonical evaluation;
3. post-cutoff operational forecasting;
4. dashboard presentation.

This separation is intended to keep the final quantitative conclusions
traceable and reproducible.

---

## 15. Current Project Status

The project has completed:

- data validation and portfolio construction;
- risk-oriented exploratory analysis;
- Historical Simulation implementation;
- EWMA implementation;
- Gradient Boosting Quantile Regression implementation;
- leakage-aware walk-forward evaluation;
- model freeze and reproducibility checks;
- final quantitative comparison;
- operational post-cutoff VaR forecasting;
- single-page Streamlit dashboard;
- automated repository regression testing.

The current development line is release-ready from the modeling and dashboard
perspective. Final branch reconciliation and release packaging are handled
separately from the frozen quantitative research contract.
