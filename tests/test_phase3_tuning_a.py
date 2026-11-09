"""Fast, deterministic tests for Phase 3 nested CV; no market data required."""
import numpy as np
import pandas as pd
import pytest

from src.phase3.tuning_a import (outer_splits,inner_splits,pinball,
                                 candidate_parameters,choose_from_inner,
                                 validate_dev_frame,violation_summary)


def fake_inner(base=.002, gains=None):
    gains=gains or {}
    rows=[]
    for k in candidate_parameters():
        for f in [1,2,3]:
            rows.append(dict(candidate=k, inner_fold=f,
                             pinball_loss=base+(.0001*(f-2)) + gains.get(k,[0,0,0])[f-1]))
    return pd.DataFrame(rows)


def test_outer_chronology_disjoint_600_targets():
    pairs=outer_splits(1179)
    assert [(len(t),len(v)) for t,v in pairs]==[(579,200),(779,200),(979,200)]
    assert np.array_equal(np.concatenate([v for _,v in pairs]),np.arange(579,1179))
    assert all(t[-1]<v[0] for t,v in pairs)


def test_outer_rejects_wrong_n():
    with pytest.raises(ValueError):outer_splits(1200)


def test_inner_chronology_stays_within_outer():
    for n in (579,779,979):
        folds=inner_splits(n)
        assert len(folds)==3
        assert all(t[-1]<v[0] and v[-1]<n for t,v in folds)
        assert len(set(np.concatenate([v for _,v in folds])))==sum(len(v) for _,v in folds)


def test_inner_rejects_small_training():
    with pytest.raises(ValueError): inner_splits(150)


def test_quantile_pinball_hand_calculation():
    np.testing.assert_allclose(pinball([-.04,.02],[-.02,-.02]),[.019,.002],atol=1e-12)


def test_pinball_rejects_nonfinite():
    with pytest.raises(ValueError):pinball([np.nan],[0.0])


def test_candidate_search_is_bounded_and_deterministic():
    p=candidate_parameters()
    assert len(p)==5 and p['C1_frozen']['random_state']==42
    assert all(d['loss']=='quantile' and d['alpha']==.05 for d in p.values())
    assert p['C1_frozen']['n_estimators']==150


def test_choose_default_when_gain_below_threshold():
    scores=fake_inner(gains={'leaf20':[-.000005]*3})
    chosen,meta=choose_from_inner(scores)
    assert chosen=='C1_frozen' and not meta['leaf20']['qualifies']


def test_choose_requires_majority_fold_improvement():
    scores=fake_inner(gains={'subsample080':[-.0001,.00001,.00001]})
    chosen,meta=choose_from_inner(scores)
    assert chosen=='C1_frozen' and meta['subsample080']['fold_wins_vs_c1']==1


def test_choose_sustained_mean_improvement():
    scores=fake_inner(gains={'leaf20':[-.00004,-.00004,-.00004]})
    chosen,meta=choose_from_inner(scores)
    assert chosen=='leaf20' and meta['leaf20']['qualifies']


def test_choose_rejects_missing_fold():
    scores=fake_inner().iloc[:-1]
    with pytest.raises(ValueError):choose_from_inner(scores)


def test_violation_is_strict_less():
    z=violation_summary([-.05,-.02,.0],[-.05,-.01,-.02])
    assert z['violations']==1 and z['violation_rate']==pytest.approx(1/3)


def test_validate_dev_rejects_lookahead():
    dates=pd.bdate_range('2020-04-06',periods=1647)
    frame=pd.DataFrame({'target_date':dates,'as_of':dates,'a':np.ones(1647),'actual_return':np.zeros(1647)})
    with pytest.raises(ValueError):validate_dev_frame(frame,('a',))
