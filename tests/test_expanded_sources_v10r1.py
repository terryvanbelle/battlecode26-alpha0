import pytest
from scripts import collect_expanded_sources_v10r1 as new
from scripts import collect_expanded_sources_v10 as old

def test_scientific_schedule_unchanged_and_all_successes_retained():
    cfg=new.read(new.CFG);original=old.read(old.CFG)
    assert new.jobs(cfg)==old.jobs(original)
    assert cfg['response_timeout_seconds']==30 and cfg['workers']==2
    delta={k for k in cfg if cfg[k]!=original.get(k)}
    assert delta=={'experiment_id','response_timeout_seconds','workers'}
    expected=new.identity();measured=new.report(expected)
    assert measured['completed_sources']>=38 and measured['inherited_sources']==38
    inherited=[r for r in measured['sources'] if r['source_json'].startswith('runs/expanded-corpus-v10/')]
    assert len({r['source_id'] for r in inherited})==38
    with pytest.raises(RuntimeError):new.worker(measured['sources'][0]['source_id'],0,0)

def test_missing_sources_only():
    expected=new.identity();pending=[j for j in new.jobs(expected['config']) if new.source_row(j,expected)[0] is None]
    pins=new.read(new.C.ROOT/'configs/expanded-corpus-v10r1-inherited.json')['sources']
    planned={j['source_id'] for j in new.jobs(expected['config'])}-set(pins)
    assert len(planned)==26 and {j['source_id'] for j in pending}<=planned
    assert 'training-m5-j1' in planned
