import numpy as np
import pandas as pd
import pytest

from src.gb_return_features import build_return_features


EXPECTED_COLUMNS = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_5",
    "rolling_vol_5",
    "rolling_vol_20",
    "rolling_vol_60",
    "drawdown",
]


def make_portfolio_returns(
    periods: int = 80,
) -> pd.Series:
    """Create deterministic portfolio returns for feature tests."""
    index = pd.bdate_range(
        "2026-01-02",
        periods=periods,
    )

    values = np.linspace(
        -0.025,
        0.035,
        periods,
        dtype="float64",
    )

    return pd.Series(
        values,
        index=index,
        name="portfolio_simple_return",
        dtype="float64",
    )


def test_build_return_features_schema_and_index() -> None:
    returns = make_portfolio_returns()

    features = build_return_features(returns)

    assert features.columns.tolist() == EXPECTED_COLUMNS

    pd.testing.assert_index_equal(
        features.index,
        returns.index,
    )

    assert all(
        dtype == np.dtype("float64")
        for dtype in features.dtypes
    )


def test_build_return_features_does_not_mutate_input() -> None:
    returns = make_portfolio_returns()
    before = returns.copy(deep=True)

    build_return_features(returns)

    pd.testing.assert_series_equal(
        returns,
        before,
    )


def test_return_lags_are_target_date_aligned() -> None:
    returns = make_portfolio_returns()

    features = build_return_features(returns)

    for lag in (1, 2, 5):
        expected = returns.shift(lag)

        pd.testing.assert_series_equal(
            features[f"return_lag_{lag}"],
            expected.rename(f"return_lag_{lag}"),
        )


@pytest.mark.parametrize(
    ("window", "row"),
    [
        (5, 5),
        (20, 20),
        (60, 60),
    ],
)
def test_rolling_volatility_uses_only_prior_returns(
    window: int,
    row: int,
) -> None:
    returns = make_portfolio_returns()

    features = build_return_features(returns)

    expected = returns.iloc[:window].std(
        ddof=1
    )

    actual = features[
        f"rolling_vol_{window}"
    ].iloc[row]

    assert actual == pytest.approx(
        expected,
        abs=1e-15,
    )


@pytest.mark.parametrize(
    "window",
    [5, 20, 60],
)
def test_rolling_volatility_keeps_full_window_warmup(
    window: int,
) -> None:
    returns = make_portfolio_returns()

    features = build_return_features(returns)

    column = features[
        f"rolling_vol_{window}"
    ]

    assert column.iloc[:window].isna().all()
    assert np.isfinite(column.iloc[window])


def test_drawdown_is_shifted_before_target_date() -> None:
    index = pd.bdate_range(
        "2026-01-02",
        periods=5,
    )

    returns = pd.Series(
        [
            0.10,
            -0.20,
            0.05,
            0.10,
            -0.05,
        ],
        index=index,
        name="portfolio_simple_return",
        dtype="float64",
    )

    features = build_return_features(returns)

    wealth = (1.0 + returns).cumprod()
    expected = (
        wealth
        .div(wealth.cummax())
        .sub(1.0)
        .shift(1)
        .rename("drawdown")
    )

    pd.testing.assert_series_equal(
        features["drawdown"],
        expected,
    )


def test_warmup_values_are_not_zero_filled() -> None:
    returns = make_portfolio_returns()

    features = build_return_features(returns)

    assert pd.isna(
        features["return_lag_1"].iloc[0]
    )

    assert pd.isna(
        features["return_lag_5"].iloc[4]
    )

    assert pd.isna(
        features["rolling_vol_60"].iloc[59]
    )

    assert pd.isna(
        features["drawdown"].iloc[0]
    )


def test_future_perturbation_does_not_change_past_features() -> None:
    returns = make_portfolio_returns()
    cutoff_position = 65

    baseline = build_return_features(
        returns
    )

    perturbed_returns = returns.copy()

    perturbed_returns.iloc[
        cutoff_position:
    ] = np.linspace(
        0.40,
        0.70,
        len(returns) - cutoff_position,
    )

    perturbed = build_return_features(
        perturbed_returns
    )

    pd.testing.assert_frame_equal(
        baseline.iloc[: cutoff_position + 1],
        perturbed.iloc[: cutoff_position + 1],
    )


def test_target_return_is_excluded_from_target_features() -> None:
    returns = make_portfolio_returns()
    target_position = 65

    baseline = build_return_features(
        returns
    )

    changed = returns.copy()
    changed.iloc[target_position] = 0.90

    changed_features = build_return_features(
        changed
    )

    pd.testing.assert_series_equal(
        baseline.iloc[target_position],
        changed_features.iloc[target_position],
    )


def test_build_return_features_is_deterministic() -> None:
    returns = make_portfolio_returns()

    first = build_return_features(returns)
    second = build_return_features(returns)

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_rejects_non_series_input() -> None:
    with pytest.raises(
        ValueError,
        match="pandas Series",
    ):
        build_return_features(
            pd.DataFrame(
                {
                    "portfolio_simple_return": [
                        0.01,
                        0.02,
                    ]
                }
            )
        )


def test_rejects_empty_series() -> None:
    returns = pd.Series(
        dtype="float64",
        index=pd.DatetimeIndex([]),
        name="portfolio_simple_return",
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        build_return_features(returns)


def test_rejects_wrong_series_name() -> None:
    returns = make_portfolio_returns()
    returns.name = "wrong_target"

    with pytest.raises(
        ValueError,
        match="portfolio_simple_return",
    ):
        build_return_features(returns)


def test_rejects_non_datetime_index() -> None:
    returns = pd.Series(
        [0.01, 0.02],
        index=[0, 1],
        name="portfolio_simple_return",
    )

    with pytest.raises(
        ValueError,
        match="DatetimeIndex",
    ):
        build_return_features(returns)


def test_rejects_duplicate_dates() -> None:
    returns = pd.Series(
        [0.01, 0.02],
        index=pd.to_datetime(
            [
                "2026-01-02",
                "2026-01-02",
            ]
        ),
        name="portfolio_simple_return",
    )

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        build_return_features(returns)


def test_rejects_unsorted_dates() -> None:
    returns = pd.Series(
        [0.01, 0.02],
        index=pd.to_datetime(
            [
                "2026-01-05",
                "2026-01-02",
            ]
        ),
        name="portfolio_simple_return",
    )

    with pytest.raises(
        ValueError,
        match="in increasing order",
    ):
        build_return_features(returns)


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_rejects_non_finite_returns(
    invalid_value: float,
) -> None:
    returns = make_portfolio_returns()
    returns.iloc[10] = invalid_value

    with pytest.raises(ValueError):
        build_return_features(returns)


@pytest.mark.parametrize(
    "invalid_value",
    [
        -1.0,
        -1.25,
    ],
)
def test_rejects_invalid_wealth_domain(
    invalid_value: float,
) -> None:
    returns = make_portfolio_returns()
    returns.iloc[10] = invalid_value

    with pytest.raises(
        ValueError,
        match="greater than -1.0",
    ):
        build_return_features(returns)
