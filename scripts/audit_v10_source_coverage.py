"""Independent saved-source proof/coverage audit; no collector/reviewer imports."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fp(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
 p=ROOT/'progress/expanded-corpus-v10r2';summary=json.loads((p/'summary.json').read_text());schedule=json.loads((p/'schedule.json').read_text());rows=[];counts={}
 for job,item in zip(schedule,summary['sources'],strict=True):
  path=ROOT/item['source_json'];assert digest(path)==item['source_sha256'];r=json.loads(path.read_text());assert r['source_id']==job['source_id']
  assert all(r[k]==job[k] for k in ['map','seed','policy_seeds'])
  raw=r['terminal_state'];assert r['status']=='ok' and r['complete'] and raw['terminal'] and r['valid_game'] and not r['benchmark_failures']
  assert fp({k:v for k,v in raw.items() if k!='snapshot'})==r['terminal_state_sha256'];assert raw['winner']==r['winner'] and raw['round']==r['terminal_round']
  assert fp(r['actions'])==r['actions_sha256'] and len(r['actions'])==r['decisions'];assert digest(ROOT/r['cache_npz'])==r['cache_sha256']
  for key,c in r['root_candidates'].items():assert 37 in c['legal_actions'] and fp(r['actions'][:c['decision_index']])==c['prefix_sha256']
  counts.setdefault(r['map'],[0]*4)
  for i,key in enumerate(['BABY_RAT:0','BABY_RAT:1','RAT_KING:0','RAT_KING:1']):counts[r['map']][i]+=int(key in r['root_candidates'])
  rows.append({'source':r['source_id'],'sha256':digest(path),'root_roles':sorted(r['root_candidates'])})
 assert len(rows)==64
 plan=json.loads((p/'training-input-plan.json').read_text());assert len(plan['roots'])==59 and len(plan['anchors'])==1024 and len(plan['missing_root_slots'])==5
 result={'complete_source_proofs':64,'source_hashes':rows,'role_availability_by_map':counts,'training_roots':59,'required_training_roots':64,'anchors':1024,'coverage_pass':False,'sgd_updates':0,'target_continuations':0}
 out=p/'reviews/independent-source-audit.json';out.parent.mkdir(exist_ok=True)
 with out.open('x') as f:json.dump(result,f,indent=2)
 fig,ax=plt.subplots(figsize=(9,5));data=list(counts.values());ax.imshow(data,vmin=0,vmax=8,cmap='YlGn');ax.set_yticks(range(8),list(counts));ax.set_xticks(range(4),['Baby A','Baby B','King A','King B'])
 for y,row in enumerate(data):
  for x,n in enumerate(row):ax.text(x,y,f'{n}/8',ha='center',va='center')
 ax.set_title('V10 source availability under frozen action-37 rule\n64/64 sources complete; training coverage failed (59/64 roots)');fig.tight_layout();fig.savefig(p/'source-role-coverage.png');plt.close(fig)
if __name__=='__main__':main()
