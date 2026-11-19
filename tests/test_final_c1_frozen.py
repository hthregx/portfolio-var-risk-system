from pathlib import Path
import json
import numpy as np
from scripts.final_c1.run_final_c1 import quantile_loss_each, evaluate_backtest, time_splits


def test_pinball_risk_tail():
    alpha=.05
    y=np.array([-0.06,-0.02,0.02]);q=np.array([-0.04,-0.04,-0.04])
    computed=quantile_loss_each(y,q,alpha)
    assert np.allclose(computed,[.019,.001,.003])


def test_breach_definition_strict_less_than():
    d=evaluate_backtest(np.array([-0.06,-0.03,0.01]),np.array([-0.05,-0.03,-0.01]),.05)
    assert d['breaches']==1
    assert d['n']==3
    assert 0<=d['kupiec_p']<=1


def test_time_splits_contract():
    cfg=json.loads(Path('configs/c1_final_frozen.json').read_text(encoding='utf-8'))
    folds=list(time_splits(1179,cfg))
    assert len(folds)==5
    assert [len(x[1]) for x in folds]==[199,395,591,787,983]
    assert [len(x[2]) for x in folds]==[196]*5
    assert all(tr[-1]<va[0] for _,tr,va in folds)
