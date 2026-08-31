import numpy as np
import pandas as pd
import pytest

from src.models.gradient_boosting_var import (
    DEFAULT_ALPHA,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MIN_SAMPLES_LEAF,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SUBSAMPLE,
    GradientBoostingVaR,
)


def make_features(
    rows: int = 80,
) -> pd.DataFrame:
    """Create deterministic numeric features for model tests."""
    index = pd.bdate_range(
        "2025-01-02",
        periods=rows,
    )

    return pd.DataFrame(
        {
            "return_lag_1": np.linspace(
                -0.03,
                0.03,
                rows,
            ),
            "rolling_vol_20": np.linspace(
                0.01,
                0.04,
                rows,
            ),
            "drawdown": np.linspace(
                -0.20,
                0.00,
                rows,
            ),
        },
        index=index,
        dtype="float64",
    )


def make_target(
    features: pd.DataFrame,
) -> pd.Series:
    """Create a deterministic target aligned to feature rows."""
    values = (
        0.45 * features["return_lag_1"].to_numpy()
        - 0.20 * features["rolling_vol_20"].to_numpy()
        + 0.05 * features["drawdown"].to_numpy()
    )

    return pd.Series(
        values,
        index=features.index,
        name="portfolio_simple_return",
        dtype="float64",
    )


def test_g01_default_contract() -> None:
    model = GradientBoostingVaR()

    assert model.alpha == DEFAULT_ALPHA == 0.05
    assert (
        model.n_estimators
        == DEFAULT_N_ESTIMATORS
        == 100
    )
    assert (
        model.learning_rate
        == DEFAULT_LEARNING_RATE
        == 0.05
    )
    assert model.max_depth == DEFAULT_MAX_DEPTH == 2
    assert (
        model.min_samples_leaf
        == DEFAULT_MIN_SAMPLES_LEAF
        == 5
    )
    assert model.subsample == DEFAULT_SUBSAMPLE == 1.0
    assert (
        model.random_state
        == DEFAULT_RANDOM_STATE
        == 42
    )


def test_fit_builds_quantile_estimator() -> None:
    features = make_features()
    target = make_target(features)

    model = GradientBoostingVaR().fit(
        features,
        target,
    )

    estimator = model._model

    assert estimator is not None
    assert estimator.loss == "quantile"
    assert estimator.alpha == 0.05
    assert estimator.n_estimators == 100
    assert estimator.learning_rate == 0.05
    assert estimator.max_depth == 2
    assert estimator.min_samples_leaf == 5
    assert estimator.subsample == 1.0
    assert estimator.random_state == 42


def test_fit_returns_self() -> None:
    features = make_features()
    target = make_target(features)

    model = GradientBoostingVaR()

    result = model.fit(
        features,
        target,
    )

    assert result is model


def test_feature_names_are_preserved_after_fit() -> None:
    features = make_features()
    target = make_target(features)

    model = GradientBoostingVaR().fit(
        features,
        target,
    )

    assert model.feature_names == tuple(
        features.columns
    )


def test_predict_quantile_returns_finite_vector() -> None:
    features = make_features()
    target = make_target(features)

    model = GradientBoostingVaR().fit(
        features,
        target,
    )

    prediction = model.predict_quantile(
        features.iloc[-4:]
    )

    assert prediction.shape == (4,)
    assert np.isfinite(prediction).all()


def test_predict_preserves_index_and_schema() -> None:
    features = make_features()
    target = make_target(features)

    model = GradientBoostingVaR().fit(
        features,
        target,
    )

    prediction_features = features.iloc[-5:]

    prediction = model.predict(
        prediction_features
    )

    assert prediction.columns.tolist() == [
        "quantile_return",
        "var",
    ]

    pd.testing.assert_index_equal(
        prediction.index,
        prediction_features.index,
    )

    assert all(
        dtype == np.dtype("float64")
        for dtype in prediction.dtypes
    )


def test_var_sign_convention_is_exact() -> None:
    features = make_features()
    target = make_target(features)

    model = GradientBoostingVaR().fit(
        features,
        target,
    )

    prediction = model.predict(
        features.iloc[-10:]
    )

    expected_var = np.maximum(
        0.0,
        -prediction["quantile_return"].to_numpy(),
    )

    assert np.allclose(
        prediction["var"].to_numpy(),
        expected_var,
        rtol=0.0,
        atol=0.0,
    )

    assert (
        prediction["var"] >= 0.0
    ).all()


def test_forecast_returns_common_var_contract() -> None:
    features = make_features()
    target = make_target(features)

    model = GradientBoostingVaR().fit(
        features,
        target,
    )

    forecast = model.forecast(
        features.iloc[[-1]]
    )

    assert list(forecast) == [
        "quantile_return",
        "var",
    ]

    assert all(
        isinstance(value, float)
        for value in forecast.values()
    )

    assert np.isfinite(
        list(forecast.values())
    ).all()

    assert forecast["var"] == max(
        0.0,
        -forecast["quantile_return"],
    )


def test_reproducibility_with_fixed_seed() -> None:
    features = make_features()
    target = make_target(features)

    first = (
        GradientBoostingVaR()
        .fit(features, target)
        .predict_quantile(features)
    )

    second = (
        GradientBoostingVaR()
        .fit(features, target)
        .predict_quantile(features)
    )

    assert np.array_equal(
        first,
        second,
    )


def test_fit_does_not_mutate_inputs() -> None:
    features = make_features()
    target = make_target(features)

    features_before = features.copy(
        deep=True
    )
    target_before = target.copy(
        deep=True
    )

    GradientBoostingVaR().fit(
        features,
        target,
    )

    pd.testing.assert_frame_equal(
        features,
        features_before,
    )

    pd.testing.assert_series_equal(
        target,
        target_before,
    )


def test_predict_before_fit_is_rejected() -> None:
    features = make_features()

    with pytest.raises(
        ValueError,
        match="fitted before prediction",
    ):
        GradientBoostingVaR().predict(
            features.iloc[:1]
        )


def test_feature_names_before_fit_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="fitted before feature names",
    ):
        _ = GradientBoostingVaR().feature_names


def test_forecast_requires_exactly_one_row() -> None:
    features = make_features()
    target = make_target(features)

    model = GradientBoostingVaR().fit(
        features,
        target,
    )

    with pytest.raises(
        ValueError,
        match="exactly one feature row",
    ):
        model.forecast(
            features.iloc[-2:]
        )


def test_prediction_column_order_must_match_fit() -> None:
    features = make_features()
    target = make_target(features)

    model = GradientBoostingVaR().fit(
        features,
        target,
    )

    reordered = features[
        list(reversed(features.columns))
    ].iloc[-2:]

    with pytest.raises(
        ValueError,
        match="columns must match",
    ):
        model.predict(reordered)


def test_target_series_index_must_match_features() -> None:
    features = make_features()
    target = make_target(features)

    misaligned = target.copy()
    misaligned.index = misaligned.index[::-1]

    with pytest.raises(
        ValueError,
        match="Target index must exactly match",
    ):
        GradientBoostingVaR().fit(
            features,
            misaligned,
        )


def test_feature_and_target_row_counts_must_match() -> None:
    features = make_features()
    target = make_target(features).to_numpy()[:-1]

    with pytest.raises(
        ValueError,
        match="row counts must match",
    ):
        GradientBoostingVaR().fit(
            features,
            target,
        )


def test_duplicate_feature_index_is_rejected() -> None:
    features = make_features()
    target = make_target(features)

    duplicated = features.copy()
    duplicated.index = pd.Index(
        [
            features.index[0],
            *features.index[:-1],
        ]
    )

    with pytest.raises(
        ValueError,
        match="Feature index must be unique",
    ):
        GradientBoostingVaR().fit(
            duplicated,
            target.to_numpy(),
        )


def test_duplicate_feature_columns_are_rejected() -> None:
    features = make_features()
    features.columns = [
        "x",
        "x",
        "z",
    ]

    target = np.linspace(
        -0.02,
        0.02,
        len(features),
    )

    with pytest.raises(
        ValueError,
        match="Feature columns must be unique",
    ):
        GradientBoostingVaR().fit(
            features,
            target,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_non_finite_features_are_rejected(
    invalid_value: float,
) -> None:
    features = make_features()
    target = make_target(features)

    features.iloc[10, 0] = invalid_value

    with pytest.raises(ValueError):
        GradientBoostingVaR().fit(
            features,
            target,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_non_finite_target_is_rejected(
    invalid_value: float,
) -> None:
    features = make_features()
    target = make_target(features)

    target.iloc[10] = invalid_value

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        GradientBoostingVaR().fit(
            features,
            target,
        )


def test_non_numeric_features_are_rejected() -> None:
    features = make_features()
    target = make_target(features)

    features = features.astype(
        {"return_lag_1": "object"}
    )
    features.iloc[5, 0] = "invalid"

    with pytest.raises(
        ValueError,
        match="feature values must be numeric",
    ):
        GradientBoostingVaR().fit(
            features,
            target,
        )


def test_non_numeric_target_is_rejected() -> None:
    features = make_features()

    target = [
        "invalid"
        for _ in range(len(features))
    ]

    with pytest.raises(
        ValueError,
        match="Target values must be numeric",
    ):
        GradientBoostingVaR().fit(
            features,
            target,
        )


@pytest.mark.parametrize(
    "invalid_alpha",
    [
        0.0,
        1.0,
        -0.05,
        np.nan,
        np.inf,
        True,
    ],
)
def test_invalid_alpha_is_rejected(
    invalid_alpha,
) -> None:
    with pytest.raises(ValueError):
        GradientBoostingVaR(
            alpha=invalid_alpha
        )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("n_estimators", 0),
        ("n_estimators", True),
        ("learning_rate", 0.0),
        ("learning_rate", True),
        ("max_depth", 0),
        ("max_depth", True),
        ("min_samples_leaf", 0),
        ("min_samples_leaf", True),
        ("subsample", 0.0),
        ("subsample", 1.1),
        ("subsample", True),
        ("random_state", True),
        ("random_state", 1.5),
    ],
)
def test_invalid_hyperparameters_are_rejected(
    parameter: str,
    value,
) -> None:
    with pytest.raises(ValueError):
        GradientBoostingVaR(
            **{parameter: value}
        )
