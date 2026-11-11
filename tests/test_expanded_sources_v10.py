from copy import deepcopy
import pytest
from scripts.collect_expanded_sources_v10 import jobs,source_spec,backend,read,CFG,R

def test_full_fixed_schedule_and_no_seed_substitution():
    cfg=read(CFG);schedule=jobs(cfg);assert len(schedule)==64
    assert len({j['seed'] for j in schedule})==64 and len({s for j in schedule for s in j['policy_seeds']})==128
    for m in range(8):assert [j['j'] for j in schedule if j['map_index']==m]==list(range(1,9))
    for job in schedule:assert source_spec(job,cfg,{'config':cfg})['map']==job['map']
    for field in ('seed','j','map_index'):
        bad=deepcopy(schedule[-1]);bad[field]+=1
        with pytest.raises(ValueError):source_spec(bad,cfg,{'config':cfg})
    bad={**schedule[0],'exclusion_map':{'hash':True}}
    with pytest.raises(ValueError):source_spec(bad,cfg,{'config':cfg})
def test_storage_adapter_does_not_mutate_v8():
    from alpha0 import terminal_v8_collect as original
    before=original.C.RUN
    adapted=backend()
    assert adapted.C.RUN==R and original.C.RUN==before and adapted.C is not original.C
    assert adapted.source_game is not original.source_game
