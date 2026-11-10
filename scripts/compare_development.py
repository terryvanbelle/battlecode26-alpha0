"""Compare complete new-suite panels; bootstrap maps, never individual games."""
import json,sys,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def compare(baseline,candidate,config,manifest):
 bots=[b['id'] for b in manifest['bots']];maps=[m['name'] for m in config['maps']]
 expected={(b,m,s,t) for b in bots for m in maps for s in config['seeds'] for t in config['sides']}
 def indexed(rows):
  out={}
  for r in rows:
   key=(r['opponent'],r['map'],r['seed'],r['side'])
   if key in out:raise ValueError('duplicate game')
   if r['status']!='ok' or r['score'] not in [0,.5,1]:raise ValueError('invalid game')
   if r['benchmark_suite_id']!=manifest['suite_id']:raise ValueError('wrong panel')
   if r['map_sha256']!={m['name']:m['sha256'] for m in config['maps']}[r['map']]:raise ValueError('changed map')
   out[key]=r
  if set(out)!=expected:raise ValueError('incomplete or extra schedule')
  if len({r['checkpoint_sha256'] for r in rows})!=1:raise ValueError('mixed checkpoints')
  return out
 b,c=indexed(baseline),indexed(candidate)
 if {r['checkpoint_sha256'] for r in baseline}!={config['baseline_sha256']}:raise ValueError('wrong baseline')
 for key in expected:
  for field in ['evaluation_config_sha256','panel_fingerprint_sha256','opponent_java_sha256']:
   if b[key][field]!=c[key][field]:raise ValueError('incompatible runtime or opponent')
 delta=np.array([np.mean([c[k]['score']-b[k]['score'] for k in expected if k[1]==m]) for m in maps])
 boot=np.random.default_rng(config['bootstrap_seed']).choice(delta,(config['bootstrap_replicates'],len(delta))).mean(1)
 lo,hi=np.quantile(boot,[.025,.975]);gain=float(delta.mean())
 per={bot:float(np.mean([c[k]['score']-b[k]['score'] for k in expected if k[0]==bot])) for bot in bots}
 return {'complete':True,'games_per_checkpoint':len(expected),'mean_score_gain':gain,'paired_map_bootstrap_95_interval':[float(lo),float(hi)],'per_opponent_score_gain':per,'passes_development_improvement_rule':bool(gain>=config['minimum_absolute_score_gain'] and lo>0 and per['vibe26b_zdb']>=0),'automatic_promotion':False,'all_map_victory':False,'baseline_wins':sum(r['score']==1 for r in baseline),'candidate_wins':sum(r['score']==1 for r in candidate)}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--config',type=Path,default=ROOT/'configs/improvement-v3.json');a=p.parse_args()
 cfg=json.loads(a.config.read_text());m=json.loads((ROOT/cfg['manifest']).read_text())
 read=lambda p:[json.loads(l) for l in p.read_text().splitlines()]
 print(json.dumps(compare(read(a.baseline),read(a.candidate),cfg,m),indent=2))
