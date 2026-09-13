from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gb_return_features import build_return_features


METADATA_PATH = ROOT / "results/final_run_metadata.json"
FINAL_RUNNER = ROOT / "scripts/run_final_walk_forward.py"
OUTPUT_PATH = ROOT / "results/gb_freeze_audit_b.csv"

EXPECTED_CONFIG = {
    "experiment_id": "G04",
    "alpha": 0.05,
    "n_estimators": 100,
    "learning_rate": 0.03,
    "max_depth": 2,
    "min_samples_leaf": 5,
    "subsample": 1.0,
    "random_state": 42,
}

EXPECTED_FEATURES = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_5",
    "rolling_vol_5",
    "rolling_vol_20",
    "rolling_vol_60",
    "drawdown",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_config():
    metadata = json.loads(
        METADATA_PATH.read_text()
    )

    gb = metadata["methods"]["gradient_boosting"]

    for key, expected in EXPECTED_CONFIG.items():
        require(
            gb.get(key) == expected,
            f"G04 config mismatch: {key}",
        )

    require(
        gb.get("features") == EXPECTED_FEATURES,
        "Frozen GB feature schema mismatch.",
    )

    return gb


def make_returns():
    dates = pd.date_range(
        "2024-01-01",
        periods=100,
        freq="D",
    )

    return pd.Series(
        [i / 10000 for i in range(100)],
        index=dates,
        name="portfolio_simple_return",
        dtype=float,
    )


def audit_feature_semantics():
    returns = make_returns()
    features = build_return_features(returns)

    require(
        list(features.columns) == EXPECTED_FEATURES,
        "Feature schema mismatch.",
    )

    t = returns.index[70]

    for lag in (1, 2, 5):
        expected = returns.shift(lag).loc[t]

        require(
            features.loc[
                t,
                f"return_lag_{lag}",
            ] == expected,
            f"return_lag_{lag} has wrong timing.",
        )

    history = returns.shift(1)

    for window in (5, 20, 60):
        expected = (
            history
            .rolling(
                window,
                min_periods=window,
            )
            .std(ddof=1)
            .loc[t]
        )

        require(
            features.loc[
                t,
                f"rolling_vol_{window}",
            ] == expected,
            (
                f"rolling_vol_{window} "
                "is not backward-looking."
            ),
        )

    return returns, features


def audit_drawdown():
    returns = make_returns()
    features = build_return_features(returns)

    wealth = (
        1.0 + returns
    ).cumprod()

    expected = (
        wealth
        .div(wealth.cummax())
        .sub(1.0)
        .shift(1)
    )

    pd.testing.assert_series_equal(
        features["drawdown"],
        expected,
        check_names=False,
    )


def audit_target_absent(features):
    forbidden = {
        "portfolio_simple_return",
        "actual_return",
        "target_return",
    }

    require(
        forbidden.isdisjoint(features.columns),
        "Target return appears in feature inputs.",
    )


def audit_future_perturbation(
    returns,
    features,
):
    boundary = returns.index[80]

    changed = returns.copy()

    # Change target/future observations only.
    changed.loc[boundary:] += 0.5

    modified = build_return_features(changed)

    # Features available through the boundary must
    # remain unchanged.
    pd.testing.assert_frame_equal(
        features.loc[:boundary],
        modified.loc[:boundary],
        check_exact=True,
    )


def audit_evaluation_protocol():
    source = FINAL_RUNNER.read_text()

    required = (
        "for step, target_date in enumerate(",
        "gb_frame.index < target_date",
        "forecast_date < target_date",
        '"evaluation_mode": "expanding"',
    )

    for text in required:
        require(
            text in source,
            f"Missing chronological control: {text}",
        )

    forbidden = (
        "train_test_split(",
        "shuffle=True",
        "sample(",
    )

    for text in forbidden:
        require(
            text not in source,
            (
                "Unexpected randomized evaluation "
                f"control: {text}"
            ),
        )


def main():
    rows = []

    audit_config()

    rows.extend([
        {
            "check": "g04_frozen_config",
            "status": "PASS",
        },
        {
            "check": "feature_schema",
            "status": "PASS",
        },
    ])

    returns, features = audit_feature_semantics()

    rows.extend([
        {
            "check": "lag_1_t_minus_1",
            "status": "PASS",
        },
        {
            "check": "lag_2_t_minus_2",
            "status": "PASS",
        },
        {
            "check": "lag_5_t_minus_5",
            "status": "PASS",
        },
        {
            "check": "rolling_vol_backward",
            "status": "PASS",
        },
    ])

    audit_drawdown()

    rows.append({
        "check": "drawdown_historical",
        "status": "PASS",
    })

    audit_target_absent(features)

    rows.append({
        "check": "target_return_absent",
        "status": "PASS",
    })

    audit_future_perturbation(
        returns,
        features,
    )

    rows.append({
        "check": "future_perturbation",
        "status": "PASS",
    })

    audit_evaluation_protocol()

    rows.extend([
        {
            "check": "chronological_evaluation",
            "status": "PASS",
        },
        {
            "check": "shuffle_disabled_or_absent",
            "status": "PASS",
        },
    ])

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(rows).to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("G04 frozen config: PASS")
    print("feature schema: PASS")
    print("lag semantics: PASS")
    print(
        "rolling volatility backward-looking: PASS"
    )
    print("drawdown historical-only: PASS")
    print("target return absent: PASS")
    print("future perturbation: PASS")
    print("chronological evaluation: PASS")
    print("shuffle disabled/absent: PASS")
    print(
        "No direct target leakage detected in "
        "the implemented feature construction."
    )
    print("B2 GB freeze audit: PASS")
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()