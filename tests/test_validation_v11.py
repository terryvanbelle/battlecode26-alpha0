import pytest
from scripts import collect_validation_v11 as V

def test_schedule_storage_and_exclusion_guard():
    cfg=V.read(V.CFG);jobs=V.jobs(cfg);assert len(jobs)==48
    assert len({x['seed'] for x in jobs})==48
    assert all(x['split']=='validation' for x in jobs)
    e=V.exclusions();expected={'config':cfg};job={**jobs[0],'exclusion_map':e}
    assert V.source_spec(job,cfg,expected)['exclusion_map_sha256']==V.C.fingerprint(e)
    with pytest.raises(ValueError):V.source_spec(jobs[0],cfg,expected)
    b=V.backend();assert b.C.RUN==V.R and callable(b.source_game)

def test_union_retains_each_pinned_input():
    e=V.exclusions();assert len(e)>=1077
    assert set(V.C.historical_exclusions())<=set(e)
    for name,field in [('progress/terminal-target-head-v8/roots-training.json','roots'),('progress/terminal-target-head-v8/roots-validation.json','roots'),('progress/terminal-target-head-v8/anchors.json','anchors'),('progress/general-roots-v11/training-input-plan.json','roots'),('progress/general-roots-v11/training-input-plan.json','anchors')]:
        assert {x['input_hash'] for x in V.read(V.C.ROOT/name)[field]}<=set(e)

def test_exact_role_allocation_and_duplicate_guard(tmp_path,monkeypatch):
    from scripts import review_validation_v11 as R
    monkeypatch.setattr(V,'exclusions',lambda:{})
    src=tmp_path/'source';src.write_text('fixture');rows={}
    for job in V.jobs(V.read(V.CFG)):
        name=job['source_id'];candidates={}
        for role in ('BABY_RAT','RAT_KING'):
            for team in (0,1):candidates[f'{role}:{team}']={'input_hash':f'{name}-{role}-{team}','source_id':name,'decision_index':team+(0 if role=='BABY_RAT' else 2),'legal_actions':[0,1]}
        rows[name]={**job,'source_json':str(src),'experiment_identity':'fixture','cache_npz':'fixture','cache_sha256':'fixture','root_candidates':candidates}
    result=R.select(rows);assert result['complete'] and len(result['roots'])==32
    assert all(sum(x['map_index']==m for x in result['roots'])==4 for m in range(8))
    rows['validation-m0-j1']['root_candidates']['BABY_RAT:0']['input_hash']=rows['validation-m1-j1']['root_candidates']['BABY_RAT:0']['input_hash']
    result=R.select(rows);assert not result['complete'] and result['duplicate_inputs']
