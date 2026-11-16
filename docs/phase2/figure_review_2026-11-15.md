# Phase 2 - Figure Review and Traceability

Repository: portfolio-var-risk-system
Integration branch: phase2/integration-ab-20261114
Integration base commit: 9dabac2131ac6938cd14f438cd46e22e38e4d3e6
Data: KBS snapshot 2026-11-06
Portfolio: equal-weight HPG / FPT / MWG
VaR target: one-day-ahead 5% return quantile

## Accepted figures

| ID | Figure | Canonical artifact |
|---|---|---|
| F01 | Market coverage | figures/phase2/a/F01_v2.png |
| F02 | Feature missingness | figures/phase2/a/F02.png |
| F03 | Expanding CV timeline | figures/phase2/a/F03_v2.png |
| F04 | Controlled feature ablation | figures/phase2/a/F04.png |
| F05 | CV fold pinball | results/phase2/a/backtesting_2026-11-06/figures/cv_fold_pinball.png |
| F06 | Normalized price trend | figures/phase2/b/eda_2026-11-06/normalized_trend.png |
| F07 | Return distribution | figures/phase2/b/eda_2026-11-06/return_distribution.png |
| F08 | Rolling volatility and drawdown | figures/phase2/b/eda_2026-11-06/vol_drawdown.png |
| F09 | Baseline CV VaR timeline | figures/phase2/b/aligned_2026-11-06/baseline_var_timeline.png |
| F10 | Stock/portfolio correlation | figures/phase2/b/correlation_2026-11-06/correlation.png |
| F11 | Actual returns vs q05 | figures/phase2/b/backtesting_2026-11-06/actual_vs_var.png |
| F12 | Monthly VaR exceptions | figures/phase2/b/backtesting_2026-11-06/exceptions_monthly_panels.png |
| F13 | Holdout model comparison | figures/phase2/b/model_comparison.png |

## Review status

- F01-F13: numerical and visual review completed; F06 retains a source-provenance caveat.
- F10: numerical, independent correlation, Git hash and visual review PASS.
- F01_v2 and F03_v2 replace the original versions for presentation.
- Original F01 and F03 retained for traceability.
- F05 image hash is recorded in the A4 backtesting manifest.
- G07 and G08 formal human sign-off remains pending.

## Important interpretation notes

1. F01: three VNINDEX sessions excluded for raw OHLCV quality anomalies.
2. F04: 12 features reduced CV pinball loss by approximately 1.4124% compared
   with 7 features at identical C1 hyperparameters, across five expanding folds.
   This does not establish statistical significance or future superiority.
3. F06: stock-close price-adjustment status is unverified.
   A market CSV byte identity with B's source files is not established,
   despite successful reconstruction of the common portfolio returns.
4. F07: full-sample empirical 5% quantile is descriptive, not a VaR forecast.
5. F09: displays the final 120 development CV sessions, not all CV folds.
6. F11-F13: 468-session retrospective holdout overlaps earlier Phase 1
   evaluation and is not a pristine or live out-of-sample test.
7. F12: monthly counts, not a day-level exception timeline.
8. F13: presents HS, EWMA and GBv2 C1; additional model comparisons
   and development CV rankings must be reported separately.
9. F10: local Windows CRLF conversion changes raw CSV SHA256, while the
   original committed Git blob matches B's metadata.
10. EWMA has the lowest mean pinball loss in the full five-model
    development CV comparison; C1 is best within GB candidates.

## Sign-off

A reviewer: PENDING
B reviewer: PENDING
Final figure QA (G07): PENDING
Full release and CI acceptance (G08): PENDING
