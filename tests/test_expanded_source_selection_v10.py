from copy import deepcopy
from scripts.review_expanded_sources_v10 import select

def corpus(tmp_path):
    p=tmp_path/'source.json';p.write_text('{}');rows={}
    for m in range(8):
        for j in range(1,9):
            sid=f'training-m{m}-j{j}'
            def candidate(i,team):return {'source_id':sid,'decision_index':i,'input_hash':f'{sid}-{i}','player':team,'legal_actions':[0,37]}
            rows[sid]={'source_id':sid,'source_json':str(p),'cache_npz':'unused','cache_sha256':'unused','map':str(m),'seed':j,
                'root_candidates':{f'{role}:{team}':candidate(i,team) for i,(role,team) in enumerate([('BABY_RAT',0),('BABY_RAT',1),('RAT_KING',0),('RAT_KING',1)])},
                'anchor_candidates':{str(t):[candidate(4+16*t+i,t) for i in range(16)] for t in (0,1)}}
    return rows

def test_exact_counts_and_no_reward_selection(tmp_path):
    rows=corpus(tmp_path);r=select(rows);assert r['complete'] and len(r['roots'])==64 and len(r['anchors'])==1024
    for row in rows.values():row['winner']=1;row['score']=0
    assert select(rows)==r

def test_matching_parity_reserves_cannot_be_reused(tmp_path):
    rows=corpus(tmp_path)
    for j in (1,3):del rows[f'training-m0-j{j}']['root_candidates']['BABY_RAT:0']
    r=select(rows);assert r['complete']
    babies=[x for x in r['roots'] if x['map_index']==0 and x['player']==0 and x['decision_index']==0]
    assert [x['source_id'] for x in babies]==['training-m0-j5','training-m0-j7']
    for j in (5,7):del rows[f'training-m0-j{j}']['root_candidates']['BABY_RAT:0']
    r=select(rows);assert not r['complete'] and len(r['missing_root_slots'])==2
    assert len(r['anchors'])==1024

def test_revision_keeps_allocation_and_adds_source_identity(tmp_path):
    from scripts.review_expanded_sources_v10r1 import select as revised
    rows=corpus(tmp_path)
    for r in rows.values():r['experiment_identity']='frozen-source'
    baseline=select(rows);actual=revised(rows)
    for field in ('roots','anchors'):
        for r in actual[field]:assert r.pop('source_experiment_identity')=='frozen-source'
    assert actual==baseline
