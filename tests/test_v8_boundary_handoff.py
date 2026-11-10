from copy import deepcopy
import pytest
from scripts.continue_v8_after_budget import review_status

def evidence():
    return {'phase':'paused','reason':'deadline','runtime_and_checkpoint_unchanged':True}, {
        'selection':{'complete':True},'fit':{'complete':True},'validation':{'complete':True},
        'benchmarks':{'complete':False,'unresolved_errors':2}}

def test_only_verified_budget_pause_can_resume():
    status,summary=evidence();assert review_status(status,summary)
    for key,value in [('reason','STOP'),('phase','review_required'),('runtime_and_checkpoint_unchanged',False)]:
        bad={**status,key:value}
        with pytest.raises(ValueError):review_status(bad,summary)
    for key in ('selection','fit','validation'):
        bad=deepcopy(summary);bad[key]['complete']=False
        with pytest.raises(ValueError):review_status(status,bad)

def test_complete_never_reruns_and_errors_require_review():
    status,summary=evidence();summary['benchmarks']['complete']=True
    assert not review_status(status,summary)
    summary['benchmarks'].update(complete=False,unresolved_errors=3)
    with pytest.raises(ValueError):review_status(status,summary)
