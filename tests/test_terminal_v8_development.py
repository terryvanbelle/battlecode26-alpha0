"""Synthetic outcomes on the actual frozen 768-cell development schedule; no games."""
import json
from copy import deepcopy
import pytest
from alpha0 import terminal_v8_common as C
from alpha0 import terminal_v8_development as D
from alpha0.terminal_v8_report import wrap_benchmark_result
from tests.test_evaluation_runner import successful

@pytest.fixture(scope='module')
def context():
    cfg=C.load_config();identity=C.identity(cfg)
    specs=D.build_benchmark_specs({a:C.ROOT/cfg['checkpoint'] for a in D.ARMS},cfg,identity)
    panel=json.loads((C.ROOT/cfg['benchmark_manifest']).read_text())
    pins={a:cfg['checkpoint_sha256'] for a in D.ARMS}
    return specs,identity,panel,pins

def summary(context, rows):
    specs,identity,panel,pins=context
    return D.summarize_benchmarks(specs,rows,identity,expected_panel=panel,expected_checkpoints=pins)

def test_full_schedule_and_comparison(context):
    specs,*_=context
    assert len(specs)==768
    rows=[wrap_benchmark_result(s,successful(s['evaluation_spec'],score=1 if s['arm']=='terminal_head' else 0),1) for s in specs]
    r=summary(context,rows)
    assert r['complete'] and r['identity_verified'] and r['development_improvement']
    assert r['comparisons']['terminal_head']['mean_score_gain']==1
    assert not r['comparisons']['puct_head']['passes_development_improvement_rule']
    assert not summary(context,rows[:-1])['development_improvement']

def test_duplicate_and_unfrozen_candidate_rejected(context):
    specs,identity,panel,pins=context
    row=wrap_benchmark_result(specs[0],successful(specs[0]['evaluation_spec']),1)
    duplicate=wrap_benchmark_result(specs[0],successful(specs[0]['evaluation_spec'],attempt=2),2)
    with pytest.raises(ValueError):summary(context,[row,duplicate])
    with pytest.raises(ValueError):D.summarize_benchmarks(specs,[],identity,expected_panel=panel,expected_checkpoints={**pins,'terminal_head':'wrong'})

def test_zdb_regression_veto(context):
    specs,*_=context
    rows=[]
    for s in specs:
        zdb=s['evaluation_spec']['identity']['opponent']=='vibe26b_zdb'
        score=int((s['arm']=='cp2' and zdb) or (s['arm']=='terminal_head' and not zdb))
        rows.append(wrap_benchmark_result(s,successful(s['evaluation_spec'],score=score),1))
    r=summary(context,rows)
    assert r['comparisons']['terminal_head']['mean_score_gain']>.05
    assert not r['development_improvement']
