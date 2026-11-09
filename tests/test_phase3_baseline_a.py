"""Phase 3 baseline smoke tests; do not touch Phase 1/2 outputs."""
import numpy as np
import pandas as pd
import pytest

from scripts.run_phase3_baseline_a import ALPHA, evaluate, pinball


def test_pinball_left_tail():
    got = pinball([-0.03, 0.02], [-0.02, -0.02])
    np.testing.assert_allclose(got, [0.0095, 0.002])


def test_violation_strict_below_quantile():
    frame = pd.DataFrame({"actual_return": [-0.03, -0.02, 0.0],
                          "quantile_return": [-0.02, -0.02, -0.02]})
    metrics = evaluate(frame)
    assert metrics["violations"] == 1
    assert metrics["n"] == 3
    assert metrics["violation_rate"] == pytest.approx(1 / 3)


def test_var_loss_is_nonnegative():
    metrics = evaluate(pd.DataFrame({"actual_return": [0.01], "quantile_return": [-0.02]}))
    assert metrics["average_var"] == pytest.approx(0.02)
    assert metrics["pinball_loss"] == pytest.approx(0.0015)


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError):
        pinball([0.01, 0.02], [-0.02])
    with pytest.raises(ValueError):
        pinball([float("nan")], [-0.02])
    with pytest.raises(ValueError):
        pinball([0.01], [-0.02], alpha=0)
    with pytest.raises(ValueError):
        evaluate(pd.DataFrame({"actual_return": [], "quantile_return": []}))
