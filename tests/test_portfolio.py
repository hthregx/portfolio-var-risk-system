import numpy as np
import pandas as pd
import pytest

from src.portfolio import (
    calculate_portfolio_log_return,
    calculate_portfolio_return,
    create_equal_weights,
    validate_weights,
)


def make_return_matrix() -> pd.DataFrame:
    """
    Create synthetic daily simple returns for portfolio unit tests.
    """
    return pd.DataFrame(
        {
            "HPG": [
                0.02,
                -0.01,
                0.03,
            ],
            "FPT": [
                -0.01,
                0.02,
                0.00,
            ],
            "MWG": [
                0.03,
                0.01,
                -0.015,
            ],
        },
        index=pd.to_datetime(
            [
                "2026-01-02",
                "2026-01-05",
                "2026-01-06",
            ]
        ),
    )


def test_create_equal_weights() -> None:
    weights = create_equal_weights(
        [" hpg ", "fpt", "MWG"]
    )

    assert weights.index.tolist() == [
        "HPG",
        "FPT",
        "MWG",
    ]

    assert weights.name == "weight"
    assert str(weights.dtype) == "float64"

    assert np.allclose(
        weights.to_numpy(),
        np.array(
            [
                1 / 3,
                1 / 3,
                1 / 3,
            ]
        ),
        rtol=0.0,
        atol=1e-12,
    )

    assert np.isclose(
        weights.sum(),
        1.0,
        rtol=0.0,
        atol=1e-12,
    )


def test_create_equal_weights_rejects_duplicate_tickers() -> None:
    with pytest.raises(
        ValueError,
        match="Ticker symbols must be unique",
    ):
        create_equal_weights(
            ["HPG", " hpg "]
        )


def test_validate_weights() -> None:
    input_weights = pd.Series(
        [0.50, 0.30, 0.20],
        index=[
            " hpg ",
            "fpt",
            "MWG",
        ],
    )

    validated_weights = validate_weights(
        input_weights
    )

    assert validated_weights.index.tolist() == [
        "HPG",
        "FPT",
        "MWG",
    ]

    assert validated_weights.name == "weight"
    assert str(validated_weights.dtype) == "float64"

    assert np.allclose(
        validated_weights.to_numpy(),
        np.array(
            [
                0.50,
                0.30,
                0.20,
            ]
        ),
        rtol=0.0,
        atol=1e-12,
    )


def test_validate_weights_rejects_invalid_total() -> None:
    invalid_weights = pd.Series(
        [0.50, 0.40, 0.30],
        index=[
            "HPG",
            "FPT",
            "MWG",
        ],
    )

    with pytest.raises(
        ValueError,
        match="Portfolio weights must sum to 1.0",
    ):
        validate_weights(
            invalid_weights
        )


def test_validate_weights_rejects_negative_weight() -> None:
    invalid_weights = pd.Series(
        [0.60, 0.50, -0.10],
        index=[
            "HPG",
            "FPT",
            "MWG",
        ],
    )

    with pytest.raises(
        ValueError,
        match="Portfolio weights cannot be negative",
    ):
        validate_weights(
            invalid_weights
        )


def test_calculate_portfolio_return() -> None:
    returns = make_return_matrix()

    weights = pd.Series(
        [
            1 / 3,
            1 / 3,
            1 / 3,
        ],
        index=[
            "HPG",
            "FPT",
            "MWG",
        ],
    )

    result = calculate_portfolio_return(
        returns,
        weights,
    )

    expected = np.array(
        [
            (0.02 - 0.01 + 0.03) / 3,
            (-0.01 + 0.02 + 0.01) / 3,
            (0.03 + 0.00 - 0.015) / 3,
        ]
    )

    assert np.allclose(
        result.to_numpy(),
        expected,
        rtol=0.0,
        atol=1e-12,
    )

    assert result.index.equals(
        returns.index
    )

    assert result.name == (
        "portfolio_simple_return"
    )

    assert str(result.dtype) == "float64"


def test_portfolio_return_is_independent_of_column_order() -> None:
    returns = make_return_matrix()

    weights = pd.Series(
        [0.50, 0.30, 0.20],
        index=[
            "HPG",
            "FPT",
            "MWG",
        ],
    )

    standard_result = calculate_portfolio_return(
        returns,
        weights,
    )

    reordered_result = calculate_portfolio_return(
        returns[
            [
                "MWG",
                "HPG",
                "FPT",
            ]
        ],
        weights,
    )

    assert np.allclose(
        standard_result.to_numpy(),
        reordered_result.to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )


def test_portfolio_return_rejects_ticker_mismatch() -> None:
    returns = make_return_matrix()

    incomplete_weights = pd.Series(
        [0.50, 0.50],
        index=[
            "HPG",
            "FPT",
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Return tickers and weight tickers "
            "must match exactly"
        ),
    ):
        calculate_portfolio_return(
            returns,
            incomplete_weights,
        )


def test_portfolio_return_rejects_missing_values() -> None:
    returns = make_return_matrix()

    returns.loc[
        pd.Timestamp("2026-01-02"),
        "HPG",
    ] = np.nan

    weights = create_equal_weights(
        [
            "HPG",
            "FPT",
            "MWG",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Asset returns cannot contain missing values",
    ):
        calculate_portfolio_return(
            returns,
            weights,
        )


def test_portfolio_return_does_not_mutate_inputs() -> None:
    returns = make_return_matrix()

    weights = pd.Series(
        [0.50, 0.30, 0.20],
        index=[
            "HPG",
            "FPT",
            "MWG",
        ],
        name="original_weights",
    )

    returns_before = returns.copy(
        deep=True
    )

    weights_before = weights.copy(
        deep=True
    )

    calculate_portfolio_return(
        returns,
        weights,
    )

    pd.testing.assert_frame_equal(
        returns,
        returns_before,
    )

    pd.testing.assert_series_equal(
        weights,
        weights_before,
    )


def test_calculate_portfolio_log_return() -> None:
    dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-05",
            "2026-01-06",
        ]
    )

    simple_returns = pd.Series(
        [
            0.02,
            -0.03,
            0.00,
        ],
        index=dates,
        name="portfolio_simple_return",
    )

    log_returns = calculate_portfolio_log_return(
        simple_returns
    )

    expected = np.log1p(
        simple_returns.to_numpy()
    )

    assert np.allclose(
        log_returns.to_numpy(),
        expected,
        rtol=0.0,
        atol=1e-15,
    )

    assert log_returns.index.equals(
        simple_returns.index
    )

    assert log_returns.name == (
        "portfolio_log_return"
    )

    assert str(log_returns.dtype) == "float64"


def test_portfolio_log_return_rejects_invalid_domain() -> None:
    simple_returns = pd.Series(
        [
            0.02,
            -1.00,
        ]
    )

    with pytest.raises(
        ValueError,
        match=(
            "Portfolio simple returns must be "
            "greater than -1.0"
        ),
    ):
        calculate_portfolio_log_return(
            simple_returns
        )