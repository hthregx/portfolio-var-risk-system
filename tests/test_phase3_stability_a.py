"""P3-05 A-only causal walk-forward and quantile risk diagnostic unit tests."""
import numpy as np
import pandas as pd
import pytest

from src.phase3.stability_a import (
    walkforward_schedule,verify_predictions,coverage_tests,rolling_quality,
    summarize,period_metrics,block_bootstrap_paired,coerce_bool, POLICIES
)


def sample(n=600):
    dates=pd.date_range('2021-01-01',periods=n,freq='B')
    days=dates.strftime('%Y-%m-%d').to_numpy()
    asof=(dates-pd.Timedelta(days=3)).strftime('%Y-%m-%d').to_numpy()
    actual=np.sin(np.arange(n))*.01
    rows=[]
    for label in POLICIES:
        for i in range(n):
            q=-.016
            rows.append({'policy':label,'outer_fold':1+i//200,'target_date':days[i],
                         'as_of':asof[i],'quantile_return':q,
                         'var_return':.016,'actual_return':actual[i],
                         'violation':bool(actual[i]<q)})
    return pd.DataFrame(rows),days,asof,actual


def test_schedule_expanding_no_future_and_every20():
    s=walkforward_schedule()
    assert len(s)==30
    assert [len(v) for _,v in s]==[20]*30
    assert [len(t) for t,_ in s]==list(range(579,1179,20))
    assert all(t[-1]<v[0] for t,v in s)
    assert s[-1][1][-1]==1178


def test_schedule_rolling504_and_no_gap_leak():
    s=walkforward_schedule(rolling=504)
    assert len(s)==30
    assert all(len(t)==504 for t,v in s)
    assert s[0][0][0]==75 and s[-1][0][-1]==1158
    assert all(t[-1]<v[0] for t,v in s)


def test_schedule_rejects_bad_parameters():
    with pytest.raises(ValueError):walkforward_schedule(rolling=100)
    with pytest.raises(ValueError):walkforward_schedule(first=50)
    with pytest.raises(ValueError):walkforward_schedule(interval=0)


def test_verify_predictions_accepts_matching_dates():
    frame,dates,asof,actual=sample()
    assert len(verify_predictions(frame,dates,asof,actual))==2400


def test_verify_predictions_rejects_corrupt_actual():
    frame,dates,asof,actual=sample()
    frame.loc[0,'actual_return']=9
    with pytest.raises(ValueError,match='Actual return'):verify_predictions(frame,dates,asof,actual)


def test_verify_predictions_rejects_leaky_date():
    frame,dates,asof,actual=sample()
    frame.loc[0,'as_of']=frame.loc[0,'target_date']
    with pytest.raises(ValueError,match='Target/as_of'):verify_predictions(frame,dates,asof,actual)


def test_verify_predictions_rejects_wrong_var_sign():
    frame,dates,asof,actual=sample()
    frame.loc[0,'var_return']=-.016
    with pytest.raises(ValueError,match='VaR sign'):verify_predictions(frame,dates,asof,actual)


def test_verify_predictions_rejects_wrong_violation_flag():
    frame,dates,asof,actual=sample()
    frame.loc[0,'violation']=True
    with pytest.raises(ValueError,match='Violation flag'):verify_predictions(frame,dates,asof,actual)


def test_verify_predictions_rejects_duplicate_days():
    frame,dates,asof,actual=sample()
    frame.loc[1,'target_date']=frame.loc[0,'target_date']
    with pytest.raises(ValueError,match='Duplicate'):verify_predictions(frame,dates,asof,actual)


def test_coverage_boundaries_and_transitions():
    d=coverage_tests([False,False,False,False,False])
    assert d['violations']==0 and 0<=d['kupiec_p_uc']<=1
    assert d['transitions_00']==4 and d['transitions_11']==0
    x=coverage_tests([True,False,True,False,True])
    assert x['transitions_10']==2 and x['transitions_01']==2
    assert 0<=x['conditional_p_cc']<=1


def test_coverage_independence_with_cluster():
    clustered=coverage_tests([False]*15+[True]*10+[False]*15)
    assert clustered['christoffersen_p_ind']<.05


def test_invalid_flags_rejected():
    with pytest.raises(ValueError):coerce_bool(['yes','no'])
    assert coerce_bool(['False','TRUE']).tolist()==[False,True]


def test_rolling_quality_and_summary():
    f,d,a,y=sample()
    v=verify_predictions(f,d,a,y)
    r=rolling_quality(v,window=60)
    assert len(r)==4*541
    assert set(r.window)=={60}
    s=summarize(v)
    assert len(s)==4 and s.relative_loss_improvement_vs_c1_pct.abs().max()<1e-10
    p=period_metrics(v)
    assert len(p)==12 and set(p.n)=={200}


def test_bootstrap_deterministic_and_positive_advantage():
    base=np.linspace(.002,.004,600)
    candidate=base-.0001
    x=block_bootstrap_paired(base,candidate,repetitions=200)
    xx=block_bootstrap_paired(base,candidate,repetitions=200)
    assert x==xx
    assert np.isclose(x['paired_mean_loss_advantage'],.0001)
    assert x['block_ci_95_low']>0 and x['ci_includes_zero'] is False


def test_bootstrap_zero_advantage_and_bad_inputs():
    x=np.ones(600)*.002
    v=block_bootstrap_paired(x,x,repetitions=100)
    assert v['ci_includes_zero'] is True
    with pytest.raises(ValueError):block_bootstrap_paired(x,x[:-1])
    with pytest.raises(ValueError):block_bootstrap_paired(x,x,block_length=300)
