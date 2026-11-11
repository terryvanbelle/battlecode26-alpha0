"""Audit saved original/re-extracted transcripts and exact cached root inputs."""
import json
from pathlib import Path
import numpy as np
from alpha0 import terminal_v8_common as C
from alpha0.features_v2 import Observation

def main():
 p=C.ROOT/'progress/general-roots-v11';old=C.ROOT/'progress/expanded-corpus-v10r2'
 a=json.loads((p/'summary.json').read_text());b=json.loads((old/'summary.json').read_text());assert a['complete'] and b['complete']
 sources=[];positions=0
 for x,y in zip(a['sources'],b['sources'],strict=True):
  assert x['source_id']==y['source_id']
  for row in (x,y):assert C.sha256(C.ROOT/row['source_json'])==row['source_sha256']
  r=json.loads((C.ROOT/x['source_json']).read_text());o=json.loads((C.ROOT/y['source_json']).read_text())
  for k in ('actions','actions_sha256','terminal_state_sha256','decisions','winner','terminal_round'):assert r[k]==o[k]
  assert C.fingerprint(r['actions'])==r['actions_sha256'] and r['status']=='ok' and r['complete']
  assert C.fingerprint({k:v for k,v in r['terminal_state'].items() if k!='snapshot'})==r['terminal_state_sha256']
  assert C.sha256(C.ROOT/r['cache_npz'])==r['cache_sha256']
  with np.load(C.ROOT/r['cache_npz'],allow_pickle=False) as cache:
   for v in [*r['root_candidates'].values(),*r['anchor_candidates']['0'],*r['anchor_candidates']['1']]:
    i=v['feature_index'];digest=C.input_hash(Observation(cache['local'][i],cache['global_board'][i],cache['scalars'][i]),cache['mask'][i]);assert digest==v['input_hash']
    assert C.fingerprint(r['actions'][:v['decision_index']])==v['prefix_sha256'];positions+=1
  for t in ('0','1'):assert [(v['decision_index'],v['input_hash'],v['state_sha256']) for v in r['anchor_candidates'][t]]==[(v['decision_index'],v['input_hash'],v['state_sha256']) for v in o['anchor_candidates'][t]]
  assert len(r['root_candidates'])==4 and all(len(v['legal_actions'])>=2 for v in r['root_candidates'].values())
  sources.append(x['source_id'])
 plan=json.loads((p/'training-input-plan.json').read_text());assert len(plan['roots'])==64 and len(plan['anchors'])==1024 and plan['complete']
 cells=2*sum(len(v['legal_actions']) for v in plan['roots']);assert cells==2974 and cells<=3200
 result={'sources_verified':len(sources),'cached_positions_verified':positions,'roots':64,'anchors':1024,'unique_root_inputs':len({v['input_hash'] for v in plan['roots']}),'planned_selection_cells':cells,'passed':True,'new_selfplay_games':0,'sgd_updates':0,'limitation':'Independent saved-record/cache audit, not another physical engine replay.'}
 out=p/'reviews/saved-replay-audit.json';out.parent.mkdir(exist_ok=True)
 with out.open('x') as f:json.dump(result,f,indent=2)
 print(result)
if __name__=='__main__':main()
