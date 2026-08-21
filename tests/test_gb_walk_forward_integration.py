from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtesting.walk_forward import (
    PREDICTION_COLUMNS,
    walk_forward,
    walk_forward_model,
)
from src.gb_market_features import build_market_features
from src.gb_return_features import build_return_features
from src.models.ewma_var import ewma_var_forecast
from src.models.gradient_boosting_var import GradientBoostingVaR
from src.models.historical_var import historical_var_forecast


def sample_data(n=90):
    dates = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(42)

    data = pd.DataFrame({
        "date": dates,
        "portfolio_simple_return": rng.normal(0, 0.02, n),
    })

    features = pd.DataFrame(
        rng.normal(size=(n, 4)),
        index=dates,
        columns=["f1", "f2", "f3", "f4"],
    )

    return data, features


def gb_factory():
    return GradientBoostingVaR(
        alpha=0.05,
        n_estimators=20,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=5,
        random_state=42,
    )


def constant_factory(q):
    class Model:
        def fit(self, x, y):
            return self

        def predict(self, x):
            return pd.DataFrame(
                {
                    "quantile_return": [q],
                    "var": [max(0.0, -q)],
                },
                index=x.index,
            )

    return Model


def run_gb():
    data, features = sample_data()
    return walk_forward_model(
        data,
        features,
        gb_factory,
        window_size=30,
    )


def test_gb_adapter_common_contract():
    result = run_gb()

    assert not result.empty
    assert result.columns.tolist() == PREDICTION_COLUMNS
    assert set(result["method"]) == {"gradient_boosting"}
    assert (result["forecast_date"] < result["target_date"]).all()
    assert np.isfinite(result["quantile_return"]).all()
    assert np.isfinite(result["var"]).all()
    assert (result["var"] >= 0).all()


def test_training_excludes_target():
    data, features = sample_data()
    seen = []

    class Spy:
        def fit(self, x, y):
            seen.append(x.index.copy())
            return self

        def predict(self, x):
            return pd.DataFrame(
                {"quantile_return": [-0.02], "var": [0.02]},
                index=x.index,
            )

    result = walk_forward_model(
        data,
        features,
        Spy,
        window_size=30,
    )

    for train_index, target in zip(
        seen,
        result["target_date"],
    ):
        assert (train_index < target).all()
        assert target not in train_index


def test_target_index_alignment():
    data, features = sample_data()

    result = walk_forward_model(
        data,
        features,
        gb_factory,
        window_size=30,
    )

    assert pd.DatetimeIndex(
        result["target_date"]
    ).equals(
        pd.DatetimeIndex(features.index[30:])
    )


def test_runner_strict_violation_and_equality_non_violation():
    dates = pd.bdate_range("2024-01-02", periods=6)
    features = pd.DataFrame(
        {"f": np.arange(6, dtype=float)},
        index=dates,
    )

    data = pd.DataFrame({
        "date": dates,
        "portfolio_simple_return":
            [0.01, 0.01, 0.01, -0.03, -0.02, -0.01],
    })

    result = walk_forward_model(
        data,
        features,
        constant_factory(-0.02),
        window_size=3,
    )

    assert result["violation"].tolist() == [
        True,
        False,
        False,
    ]


def test_return_feature_target_future_perturbation():
    dates = pd.bdate_range("2023-01-02", periods=100)

    returns = pd.Series(
        np.linspace(-0.02, 0.02, 100),
        index=dates,
        name="portfolio_simple_return",
    )

    target = dates[70]

    baseline = build_return_features(returns)

    changed_returns = returns.copy()
    changed_returns.loc[target:] += 0.5

    changed = build_return_features(changed_returns)

    pd.testing.assert_series_equal(
        baseline.loc[target],
        changed.loc[target],
    )


def test_market_feature_target_future_perturbation():
    dates = pd.bdate_range("2023-01-02", periods=100)

    stocks = {}
    for i, ticker in enumerate(("HPG", "FPT", "MWG")):
        close = 50 + i * 10 + np.arange(100) * 0.1

        stocks[ticker] = pd.DataFrame({
            "date": dates,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": 1000 + i * 100 + np.arange(100) * 10,
        })

    market = pd.DataFrame({
        "date": dates,
        "close": 1000 + np.arange(100),
    })

    target = dates[70]

    baseline = build_market_features(stocks, market)

    changed_stocks = {
        k: v.copy()
        for k, v in stocks.items()
    }

    for frame in changed_stocks.values():
        mask = frame["date"] >= target
        frame.loc[mask, ["high", "low", "close"]] *= 2
        frame.loc[mask, "volume"] *= 10

    changed_market = market.copy()
    changed_market.loc[
        changed_market["date"] >= target,
        "close",
    ] *= 2

    changed = build_market_features(
        changed_stocks,
        changed_market,
    )

    pd.testing.assert_series_equal(
        baseline.loc[target],
        changed.loc[target],
    )


def test_common_historical_ewma_gb_semantics():
    data, features = sample_data()

    historical = walk_forward(
        data,
        lambda x: historical_var_forecast(x, alpha=0.05),
        window_size=30,
        method="historical",
    )

    ewma = walk_forward(
        data,
        lambda x: ewma_var_forecast(
            x,
            alpha=0.05,
            decay=0.94,
        ),
        window_size=30,
        method="ewma",
    )

    gb = walk_forward_model(
        data,
        features,
        gb_factory,
        window_size=30,
    )

    for result in (historical, ewma, gb):
        assert result.columns.tolist() == PREDICTION_COLUMNS
        assert (result["forecast_date"] < result["target_date"]).all()

        np.testing.assert_allclose(
            result["var"],
            np.maximum(
                0.0,
                -result["quantile_return"].to_numpy(),
            ),
        )

        assert result["violation"].equals(
            result["actual_return"]
            < result["quantile_return"]
        )

    assert historical["target_date"].equals(ewma["target_date"])
    assert historical["target_date"].equals(gb["target_date"])


def test_production_config_and_loss():
    data, features = sample_data()
    target = data.set_index("date")[
        "portfolio_simple_return"
    ]

    model = GradientBoostingVaR()

    model.fit(
        features.iloc[:40],
        target.iloc[:40],
    )

    assert model.alpha == 0.05
    assert model.random_state == 42
    assert model._model.loss == "quantile"
    assert model.feature_names == tuple(features.columns)


def test_fixed_seed_deterministic():
    first = run_gb()
    second = run_gb()

    np.testing.assert_allclose(
        first["quantile_return"],
        second["quantile_return"],
    )

    np.testing.assert_allclose(
        first["var"],
        second["var"],
    )

    assert first["violation"].equals(
        second["violation"]
    )