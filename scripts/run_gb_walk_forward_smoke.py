from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtesting.walk_forward import walk_forward_model
from src.gb_market_features import (
    FEATURE_COLUMNS as MARKET_FEATURES,
    build_market_features,
)
from src.gb_return_features import build_return_features
from src.models.gradient_boosting_var import GradientBoostingVaR


TARGET = "portfolio_simple_return"

RETURN_FEATURES = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_5",
    "rolling_vol_5",
    "rolling_vol_20",
    "rolling_vol_60",
    "drawdown",
]

FEATURES = RETURN_FEATURES + MARKET_FEATURES
CREATED_MODELS = []


def load_inputs():
    returns_df = pd.read_csv(
        ROOT / "data/processed/portfolio_returns.csv",
        parse_dates=["date"],
    )

    returns = returns_df.set_index("date")[TARGET]

    stocks = {
        ticker: pd.read_csv(
            ROOT / f"data/processed/{ticker}_clean.csv",
            parse_dates=["date"],
        )
        for ticker in ("HPG", "FPT", "MWG")
    }

    market = pd.read_csv(
        ROOT / "data/raw/VNINDEX.csv"
    )

    features = (
        build_return_features(returns)
        .join(
            build_market_features(stocks, market),
            how="inner",
        )
    )

    return returns_df, features


def model_factory():
    model = GradientBoostingVaR(
        alpha=0.05,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=5,
        random_state=42,
    )

    CREATED_MODELS.append(model)
    return model


def main():
    data, features = load_inputs()

    valid_dates = features.dropna().index
    smoke_dates = valid_dates[-40:]

    smoke_data = data[
        data["date"].isin(valid_dates)
    ].copy()

    smoke_features = features.loc[
        valid_dates
    ].copy()

    result = walk_forward_model(
        smoke_data,
        smoke_features,
        model_factory,
        window_size=250,
        mode="rolling",
        method="gradient_boosting",
    )

    if not CREATED_MODELS:
        raise RuntimeError(
            "Smoke run did not fit a Gradient Boosting model."
        )

    audit_model = CREATED_MODELS[-1]

    assert audit_model.alpha == 0.05
    assert audit_model.random_state == 42
    assert audit_model._model is not None
    assert audit_model._model.loss == "quantile"
    assert audit_model.feature_names == tuple(FEATURES)

    smoke = (
        result[
            result["target_date"].isin(smoke_dates)
        ]
        [
            [
                "forecast_date",
                "target_date",
                "actual_return",
                "quantile_return",
                "var",
                "violation",
                "method",
            ]
        ]
        .copy()
    )

    if smoke.empty:
        raise RuntimeError("Smoke result is empty.")

    output = ROOT / "results/gb_walk_forward_smoke.csv"

    smoke.to_csv(
        output,
        index=False,
    )

    print("alpha:", audit_model.alpha)
    print("loss:", audit_model._model.loss)
    print("random_state:", audit_model.random_state)
    print(
        "feature_count:",
        len(audit_model.feature_names),
    )
    print(
        "feature_names:",
        list(audit_model.feature_names),
    )
    print("smoke_rows:", len(smoke))
    print("method:", "gradient_boosting")
    print("saved:", output)
    print("B20 GB walk-forward smoke: PASS")


if __name__ == "__main__":
    main()