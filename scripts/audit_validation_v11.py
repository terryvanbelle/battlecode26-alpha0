"""Independent saved validation-source and canonical-cache audit."""
import json
import numpy as np
from alpha0 import terminal_v8_common as C
from alpha0.features_v2 import Observation

def main():
 p=C.ROOT/'progress/validation-v11';summary=json.loads((p/'summary.json').read_text());schedule=json.loads((p/'schedule.json').read_text());registry=json.loads((C.ROOT/'configs/validation-v11-exclusions.json').read_text());excluded=set(registry['input_exclusions'])
 assert summary['complete'] and len(summary['sources'])==len(schedule)==48
 proofs={};positions=0;skipped=0
 for spec,entry in zip(schedule,summary['sources'],strict=True):
  path=C.ROOT/entry['source_json'];assert C.sha256(path)==entry['source_sha256'];r=json.loads(path.read_text());assert all(r[k]==spec[k] for k in spec)
  assert r['status']=='ok' and r['complete'] and r['terminal_state']['terminal'] and r['valid_game'] and not r['benchmark_failures']
  assert C.fingerprint(r['actions'])==r['actions_sha256'] and len(r['actions'])==r['decisions']
  assert C.fingerprint({k:v for k,v in r['terminal_state'].items() if k!='snapshot'})==r['terminal_state_sha256']
  assert r['terminal_state']['winner']==r['winner'] and r['terminal_state']['round']==r['terminal_round']
  assert r['exclusion_map_sha256']==C.fingerprint(registry['input_exclusions'])
  assert C.sha256(C.ROOT/r['cache_npz'])==r['cache_sha256']
  with np.load(C.ROOT/r['cache_npz'],allow_pickle=False) as z:cache={k:z[k] for k in ('local','global_board','scalars','mask')}
  for item in r['root_candidates'].values():
   i=item['feature_index'];assert C.input_hash(Observation(cache['local'][i],cache['global_board'][i],cache['scalars'][i]),cache['mask'][i])==item['input_hash']
   assert item['input_hash'] not in excluded and len(item['legal_actions'])>=2
   assert C.fingerprint(r['actions'][:item['decision_index']])==item['prefix_sha256'];positions+=1
  assert all(x['input_hash'] in excluded for x in r['skipped_input_duplicates']);skipped+=len(r['skipped_input_duplicates']);proofs[r['source_id']]=r
 plan=json.loads((p/'validation-input-plan.json').read_text());assert plan['complete'] and len(plan['roots'])==32
 hashes=[x['input_hash'] for x in plan['roots']];assert len(set(hashes))==32 and not set(hashes)&excluded
 for root in plan['roots']:
  candidate=proofs[root['source_id']]['root_candidates'][f"{root['role']}:{root['player']}"]
  assert all(root[k]==v for k,v in candidate.items())
 for m in range(8):
  for role in ('BABY_RAT','RAT_KING'):
   for player in (0,1):assert sum(r['map_index']==m and r['role']==role and r['player']==player for r in plan['roots'])==1
 out=p/'reviews/independent-input-audit.json';out.parent.mkdir(exist_ok=True)
 result={'passed':True,'complete_sources':48,'candidate_cache_positions':positions,'selected_roots':32,'unique_selected_inputs':32,'excluded_input_count':len(excluded),'skipped_excluded_positions':skipped,'gradient_overlap':0,'sgd_updates':0,'limitation':'Saved-record/cache audit, not independent engine replay or candidate evaluation.'}
 with out.open('x') as f:json.dump(result,f,indent=2)
 print(result)
if __name__=='__main__':main()
