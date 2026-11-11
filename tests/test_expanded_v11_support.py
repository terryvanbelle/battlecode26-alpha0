from copy import deepcopy
import json
import pytest
from alpha0 import expanded_v11_support as S

@pytest.fixture
def actual():
    manifest=json.loads((S.C.ROOT/'progress/general-roots-v11/training-input-plan.json').read_text())
    root=manifest['roots'][0];source=json.loads((S.C.ROOT/root['source_json']).read_text())
    return manifest,root,source

def test_actual_root_membership_and_cache(actual):
    manifest,root,source=actual
    assert S.verify_member(root,source,manifest,source['checkpoint_sha256'])
    values=S.cached_position(root);assert values['hidden'].shape==(128,)
    assert S.verify_member({**root,'worker_task_path':'fixture'},source,manifest,source['checkpoint_sha256'])

@pytest.mark.parametrize('field,value',[('source_experiment_identity','foreign'),('input_hash','foreign'),('decision_index',99999),('legal_actions',[0])])
def test_root_tampering_rejected(actual,field,value):
    manifest,root,source=actual
    with pytest.raises(ValueError):S.verify_member({**root,field:value},source,manifest,source['checkpoint_sha256'])

def test_source_identity_and_terminal_proof_rejected(actual):
    manifest,root,source=actual
    for bad in ({**source,'experiment_identity':'foreign'},{**source,'terminal_state_sha256':'foreign'},{**source,'checkpoint_sha256':'foreign'}):
        with pytest.raises(ValueError):S.verify_member(root,bad,manifest,source['checkpoint_sha256'])

def test_incomplete_manifest_rejected(tmp_path):
    p=tmp_path/'incomplete.json';p.write_text('{"complete": false}')
    pin={'path':str(p),'sha256':S.C.sha256(p)}
    with pytest.raises(ValueError):S.frozen_plans({'input_plans':{k:pin for k in ('training','validation','exclusions')}})

def test_isolated_backend_keeps_exact_source_guard(actual,tmp_path,monkeypatch):
    manifest,root,source=actual
    monkeypatch.setattr(S,'frozen_plans',lambda _: (manifest,{'roots':[]},{}))
    identity={'collector_sha256':S.C.sha256(S.C.ROOT/'alpha0/terminal_v8_collect.py'),'config':{'checkpoint_sha256':source['checkpoint_sha256']}}
    module=S.load_root_backend(tmp_path/'run',tmp_path/'progress',lambda _:None,identity)
    assert module.C.RUN==tmp_path/'run'
    assert module.verify_source_membership(root,source,identity)
    with pytest.raises(ValueError):module.verify_source_membership(root,{**source,'experiment_identity':'foreign'},identity)
    with pytest.raises(ValueError):module.verify_source_membership(root,source,{**identity,'changed':True})

def test_balanced_eight_map_batch_schedule_is_deterministic():
    import numpy as np
    from alpha0.terminal_v8_fit import make_batch_schedule,batch_fingerprint
    strata=np.array([(m,r,t) for m in range(8) for r in (0,1) for t in (0,1) for repeat in range(2)])
    games=np.repeat(np.arange(64),16);players=np.tile(np.repeat([0,1],8),64)
    cfg={'updates':4096,'batch_size':32,'anchor_batch_size':32,'seed':12345}
    a=make_batch_schedule(strata,games,players,cfg);b=make_batch_schedule(strata,games,players,cfg)
    assert a['root'].shape==(4096,32) and a['anchor'].shape==(4096,32)
    assert batch_fingerprint(a)==batch_fingerprint(b)
    assert set(a['root'].ravel())==set(range(64)) and set(a['anchor'].ravel())==set(range(1024))

def test_missing_targets_cannot_form_dataset(actual,tmp_path,monkeypatch):
    manifest,_,_=actual
    monkeypatch.setattr(S,'frozen_plans',lambda _: (manifest,{'roots':[]},{}))
    identity={'config':{'selection_seeds':[223003,223013],'selection_cell_ceiling':3200}}
    with pytest.raises(ValueError,match='comparator incomplete'):S.build_complete_dataset(tmp_path,identity)
