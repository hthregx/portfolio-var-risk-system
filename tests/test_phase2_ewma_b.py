from statistics import NormalDist

import numpy as np
import pytest

from src.phase2.ewma_b import ewma_quantiles


def test_ewma_matches_hand_calculation():
    # Sau hai returns 0.10 và -0.20:
    # variance = 0.94 * 0.01 + 0.06 * 0.04 = 0.0118
    q = ewma_quantiles([0.10, -0.20, 0.30, -0.40], warmup=2)
    z = NormalDist().inv_cdf(0.05)

    v2 = 0.0118
    v3 = 0.94 * v2 + 0.06 * 0.30 ** 2

    np.testing.assert_allclose(q, z * np.sqrt([v2, v3]))


def test_first_forecast_uses_first_squared_return():
    q = ewma_quantiles([0.10, -0.20], warmup=1)
    expected = NormalDist().inv_cdf(0.05) * 0.10

    np.testing.assert_allclose(q, [expected])


def test_target_and_future_do_not_change_current_forecast():
    returns = np.linspace(-0.02, 0.03, 280)
    before = ewma_quantiles(returns)

    changed = returns.copy()
    changed[260:] = 0.50
    after = ewma_quantiles(changed)

    # Các target 250..260 phải giữ nguyên, kể cả target 260.
    np.testing.assert_array_equal(before[:11], after[:11])

    # Return 260 chỉ được ảnh hưởng dự báo từ phiên 261.
    assert not np.isclose(before[11], after[11])


def test_prediction_count_and_zero_returns():
    q = ewma_quantiles(np.zeros(1705))

    assert len(q) == 1455
    np.testing.assert_array_equal(q, np.zeros(1455))


@pytest.mark.parametrize("bad", [
    [0.01, np.nan, 0.02],
    [0.01, np.inf, 0.02],
    [0.01, -1.00, 0.02],
])
def test_reject_invalid_returns(bad):
    with pytest.raises(ValueError):
        ewma_quantiles(bad, warmup=1)


def test_reject_insufficient_history():
    with pytest.raises(ValueError):
        ewma_quantiles(np.zeros(250))