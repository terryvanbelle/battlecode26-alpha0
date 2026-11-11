import pytest
from scripts import collect_general_roots_v11 as S

def test_backend_and_recorded_actions():
    b=S.backend();assert b.C.RUN==S.R and b.source_spec is S.source_spec
    r=S.RecordedActions([2,0]);assert r(None,[0,2],None,1)==2
    with pytest.raises(ValueError):r(None,[2],None,1)
    assert r.index==1 and r(None,[0],None,1)==0
    with pytest.raises(ValueError):r(None,[0],None,1)

def test_source_schedule_and_equivalence_guard():
    cfg=S.read(S.CFG);jobs=S.jobs(cfg)
    assert jobs==S.ORIGINAL.jobs(S.read(S.ORIGINAL.CFG)) and len(jobs)==64
    old,_=S.original(jobs[0]);assert S.verify_replay(old,old)==old
    bad={**old,'terminal_state_sha256':'changed'}
    with pytest.raises(ValueError):S.verify_replay(bad,old)
    bad={**old,'anchor_candidates':{'0':[], '1':old['anchor_candidates']['1']}}
    with pytest.raises(ValueError):S.verify_replay(bad,old)
