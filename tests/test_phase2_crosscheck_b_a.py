import numpy as np
import pandas as pd
import pytest

from src.phase2.crosscheck_b_a import (
    audited_frame, historical_quantiles, ewma_quantiles_independent,
    verify_target_alignment, verify_quantiles, b_five_expanding_indices,
)


def source_data(n=300):
    dates = pd.bdate_range('2026-01-01', periods=n)
    r = pd.Series(np.sin(np.arange(n) / 12) / 100,
                  index=dates, name='portfolio_simple_return')
    return r


def frame_for(r):
    return pd.DataFrame({
        'model': 'historical_w250', 'fold': 'development',
        'as_of': r.index[249:-1], 'target_date': r.index[250:],
        'quantile_return': historical_quantiles(r.to_numpy()),
        'var_return': np.maximum(0, -historical_quantiles(r.to_numpy())),
        'actual_return': r.to_numpy()[250:],
    })


def test_hs_uses_only_older_250():
    r = source_data()
    q = historical_quantiles(r.to_numpy())
    assert q[0] == pytest.approx(np.quantile(r.iloc[:250], .05))
    changed = r.copy()
    changed.iloc[250] = -0.3
    assert historical_quantiles(changed.to_numpy())[0] == q[0]


def test_ewma_uses_only_earlier_return():
    r = source_data()
    q = ewma_quantiles_independent(r.to_numpy())
    changed = r.copy()
    changed.iloc[250] = -.3
    altered = ewma_quantiles_independent(changed.to_numpy())
    assert q[0] == altered[0]
    assert q[1] != altered[1]


def test_reject_wrong_schema():
    r = source_data()
    frame = frame_for(r).drop(columns=['fold'])
    with pytest.raises(ValueError):
        audited_frame(frame, 'historical_w250')


def test_reject_corrupt_var_sign():
    r = source_data()
    frame = frame_for(r)
    frame.loc[frame.index[0], 'var_return'] = 0
    with pytest.raises(ValueError):
        audited_frame(frame, 'historical_w250')


def test_reject_wrong_model():
    with pytest.raises(ValueError):
        audited_frame(frame_for(source_data()), 'ewma_d094')


def test_accept_valid_alignment():
    r = source_data()
    audited = audited_frame(frame_for(r), 'historical_w250')
    verify_target_alignment(audited, r)


def test_reject_target_return_drift():
    r = source_data()
    frame = audited_frame(frame_for(r), 'historical_w250')
    frame.loc[frame.index[0], 'actual_return'] += .01
    with pytest.raises(ValueError):
        verify_target_alignment(frame, r)


def test_reject_shifted_forecast_date():
    r = source_data()
    frame = audited_frame(frame_for(r), 'historical_w250')
    frame.loc[frame.index[0], 'as_of'] = frame['target_date'].iat[0] - pd.Timedelta(days=2)
    with pytest.raises(ValueError):
        verify_target_alignment(frame, r)


def test_reject_changed_quantiles():
    f = frame_for(source_data())
    q = f['quantile_return'].to_numpy().copy()
    assert verify_quantiles(f, q)['n'] == len(f)
    q[0] += .005
    with pytest.raises(ValueError):
        verify_quantiles(f, q)


def test_five_expanding_indices_match_b_counts():
    splits = b_five_expanding_indices(1645)
    assert [len(x) for x, _ in splits] == [275,549,823,1097,1371]
    assert [len(y) for _, y in splits] == [274]*5
    assert np.concatenate([y for _, y in splits]).size == 1370


def test_reject_duplicate_target_dates():
    f = frame_for(source_data())
    f.loc[f.index[1], 'target_date'] = f['target_date'].iat[0]
    with pytest.raises(ValueError):
        audited_frame(f, 'historical_w250')


def test_ewma_hand_calculated_small_example():
    r = np.array([.10,-.20,.30,-.40], dtype=float)
    got = ewma_quantiles_independent(r, warmup=2)
    from statistics import NormalDist
    z = NormalDist().inv_cdf(.05)
    assert got[0] == pytest.approx(z*np.sqrt(.94*.01+.06*.04))
