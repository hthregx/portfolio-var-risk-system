"""P3-06 safety and numerical consistency tests: no live data or model fitting."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.phase3.release_a import (
    safe_artifact_path, audit_manifest_artifacts, verify_feature_ranking,
    verify_policy_predictions, decision_gate, standalone_html, pinball, POLICIES
)


def test_pinball_quantile_signed():
    np.testing.assert_allclose(pinball([-0.10,0.10],[-0.03,-0.03]),[0.0665,0.0065])


def test_safe_path_allows_single_relative_csv(tmp_path):
    assert safe_artifact_path(tmp_path,'cv_predictions',bare_csv_keys=True)==(tmp_path/'cv_predictions.csv').resolve()


def test_safe_path_normalizes_windows_separators(tmp_path):
    assert safe_artifact_path(tmp_path,r'figures\chart.png') == (tmp_path/'figures'/'chart.png').resolve()


@pytest.mark.parametrize('key',['../secret','/etc/passwd','C:\\private\\x','folder/../../foo','//server/secret',''])
def test_safe_path_blocks_traversal(tmp_path,key):
    with pytest.raises(ValueError):safe_artifact_path(tmp_path,key)


def test_manifest_audit_accepts_correct_hash(tmp_path):
    path=tmp_path/'a.csv';path.write_text('x\n1\n',encoding='utf-8')
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert audit_manifest_artifacts(tmp_path,{'artifacts':{'a':{'sha256':sha,'rows':1}}},bare_csv_keys=True)['a.csv']==sha


def test_manifest_audit_rejects_tamper(tmp_path):
    path=tmp_path/'a.csv';path.write_text('x\n1\n',encoding='utf-8')
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text('x\n2\n',encoding='utf-8')
    with pytest.raises(ValueError,match='SHA256'):
        audit_manifest_artifacts(tmp_path,{'artifacts':{'a.csv':{'sha256':sha}}})


def _rank():
    fold=pd.DataFrame({'feature_count':[12]*5+[18]*5+[25]*5,
                       'fold':[1,2,3,4,5]*3,
                       'val_pinball':[.002]*5+[.003]*5+[.004]*5})
    rank=pd.DataFrame({'feature_count':[12,18,25], 'cv_pinball':[.002,.003,.004]})
    return fold,rank


def test_cv_ranking_verified():
    folds,ranking=_rank()
    assert verify_feature_ranking(ranking,folds).feature_count.tolist()==[12,18,25]


def test_cv_ranking_rejects_mismatch():
    folds,ranking=_rank();ranking.loc[0,'cv_pinball']=.001
    with pytest.raises(ValueError,match='does not match'):
        verify_feature_ranking(ranking,folds)


def _forecasts(n=6):
    days=pd.date_range('2021-01-07',periods=n,freq='D').strftime('%Y-%m-%d').tolist()
    asof=pd.date_range('2021-01-06',periods=n,freq='D').strftime('%Y-%m-%d').tolist()
    y=[-0.02,0.02,0.01,-0.03,0.03,0.0]
    q=[-0.01]*6
    records=[];metrics=[]
    for label in POLICIES:
        for i in range(n):
            records.append(dict(policy=label,outer_fold=1,target_date=days[i],as_of=asof[i],
                                actual_return=y[i],quantile_return=q[i],var_return=.01,violation=y[i]<q[i]))
        metrics.append(dict(policy=label,n=n,violations=2,violation_rate=2/n,mean_pinball=float(pinball(y,q).mean())))
    return pd.DataFrame(records),pd.DataFrame(metrics)


def test_policy_predictions_pass_same_dates_and_returns():
    p,m=_forecasts()
    assert len(verify_policy_predictions(p,m,expected_n=6))==4


def test_policy_predictions_reject_corrupted_flag():
    p,m=_forecasts();p.loc[0,'violation']=False
    with pytest.raises(ValueError,match='Exception flags'):
        verify_policy_predictions(p,m,expected_n=6)


def test_policy_predictions_reject_unpaired_returns():
    p,m=_forecasts(); p.loc[6,'actual_return']=.333
    with pytest.raises(ValueError):verify_policy_predictions(p,m,expected_n=6)


def _bootstrap(lo=-.001,hi=.001):
    return pd.DataFrame([{'alternative':k,'block_ci_95_low':lo,'block_ci_95_high':hi,
                         'ci_includes_zero':lo<=0<=hi} for k in POLICIES[1:]])


def test_unchanged_gate_when_interval_includes_zero():
    assert decision_gate(_bootstrap())['candidate_promoted'] is False
    assert decision_gate(_bootstrap())['status']=='NO_CONFIRMED_IMPROVEMENT'


def test_gate_never_automatically_promotes_even_when_ci_excludes_zero():
    r=decision_gate(_bootstrap(lo=.001,hi=.002))
    assert r['candidate_promoted'] is False and r['status']=='REVIEW_REQUIRED'


def test_bootstrap_gate_rejects_inconsistent_boolean():
    df=_bootstrap();df.loc[0,'ci_includes_zero']=False
    with pytest.raises(ValueError,match='Inconsistent CI'):decision_gate(df)


def test_offline_html_embeds_png(tmp_path):
    p=tmp_path/'plot.png';p.write_bytes(b'\x89PNG\r\n\x1a\nTEST')
    _,ranking=_rank(); _,metrics=_forecasts()
    s=standalone_html('Audit < Safety',ranking,metrics,decision_gate(_bootstrap()),[p])
    assert 'data:image/png;base64,' in s and 'Audit &lt; Safety' in s
    assert 'EXPLORATORY DEVELOPMENT ONLY' in s
