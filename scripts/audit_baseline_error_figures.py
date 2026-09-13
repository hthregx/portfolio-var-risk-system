from pathlib import Path

import numpy as np
import pandas as pd


DATA = Path("results/baseline_error_analysis.csv")
CASES = Path("results/baseline_case_studies.csv")
FIGURES = Path("figures/baseline_error_analysis")


df = pd.read_csv(
    DATA,
    parse_dates=["forecast_date", "target_date"],
)

cases = pd.read_csv(
    CASES,
    parse_dates=["forecast_date", "target_date"],
)


# Canonical sample
assert len(df) == 1387
assert df["target_date"].is_unique
assert not df.isna().any().any()

# Counts
h = df["historical_violation"].astype(bool)
e = df["ewma_violation"].astype(bool)

assert int(h.sum()) == 75
assert int(e.sum()) == 73
assert int((h & e).sum()) == 56
assert int((h & ~e).sum()) == 19
assert int((~h & e).sum()) == 17

# Strict violation
assert np.array_equal(
    h.to_numpy(),
    (
        df["target_return"]
        < df["historical_quantile"]
    ).to_numpy(),
)

assert np.array_equal(
    e.to_numpy(),
    (
        df["target_return"]
        < df["ewma_quantile"]
    ).to_numpy(),
)

# Severity
np.testing.assert_allclose(
    df["historical_exceedance"],
    np.maximum(
        0.0,
        df["historical_quantile"]
        - df["target_return"],
    ),
    atol=1e-12,
    rtol=0.0,
)

np.testing.assert_allclose(
    df["ewma_exceedance"],
    np.maximum(
        0.0,
        df["ewma_quantile"]
        - df["target_return"],
    ),
    atol=1e-12,
    rtol=0.0,
)

# Sign convention
assert (df["historical_var"] >= 0).all()
assert (df["ewma_var"] >= 0).all()

# Case-study QA
assert 5 <= len(cases) <= 10
assert cases["target_date"].is_unique

assert set(cases["target_date"]).issubset(
    set(df["target_date"])
)

assert (
    cases["exception_type"]
    == "both"
).any()

assert (
    cases["exception_type"]
    == "historical_only"
).any()

assert (
    cases["exception_type"]
    == "ewma_only"
).any()

# Temporal response check
assert np.array_equal(
    df["target_date"].iloc[:-1].to_numpy(),
    df["forecast_date"].iloc[1:].to_numpy(),
)

# Figure artifacts
expected = [
    "01_baseline_var_vs_returns.png",
    "02_exception_timeline.png",
    "03_exception_severity.png",
    "04_exception_clusters.png",
    "05_case_study_risk_response.png",
]

for name in expected:
    path = FIGURES / name
    assert path.exists()
    assert path.stat().st_size > 0

print("Rows:", len(df))
print("Historical violations:", int(h.sum()))
print("EWMA violations:", int(e.sum()))
print("Shared:", int((h & e).sum()))
print("Historical-only:", int((h & ~e).sum()))
print("EWMA-only:", int((~h & e).sum()))
print("Case-study dates:", len(cases))
print("Figures:", len(expected))
print()
print("B-15.5 FIGURE QA PASS")