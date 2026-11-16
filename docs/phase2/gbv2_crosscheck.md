# G06 ? A/B GBv2 cross-check

Status: numerical cross-check PASS; two-person sign-off pending.

- Snapshot: KBS, 2026-11-06.
- Target: 1-day-ahead 5% portfolio-return quantile.
- Features: 12, 1647 rows aligned.
- Maximum feature difference: 2.4980018054066022e-15.
- Common GB CV predictions matched: 2940.
- Common GB holdout predictions matched: 936.
- Model dates, forecast origins, and violation flags: matched.
- B metrics recomputed independently: 25 CV + 4 holdout groups PASS.
- Numerical tolerance: 1e-9 for predictions, 1e-12 for metrics.

## Limitations

- Shared original seven-feature builder; B independently implements five added features.
- Retrospective holdout overlaps Phase 1 evaluation and is not pristine.
- Numerical reproduction does not establish future predictive superiority.
- Code B and release evidence still require final human review.
- Both A and B must approve G06 before formal sign-off.
