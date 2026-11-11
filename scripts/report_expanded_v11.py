"""Derived V11 stage coverage; preserves immutable evidence and benchmark semantics."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from alpha0.expanded_v11_run import C,R,P,read
from alpha0.expanded_v11_support import frozen_plans
from alpha0.terminal_v8_collect import cell_spec,successful_cell

def main():
 expected=read(P/'identity.json');training,validation,_=frozen_plans(expected);cfg=expected['config'];rows=[];scores={}
 for stage,roots in [('selection',training['roots']),('validation',validation['roots'])]:
  actions=read(P/'validation-actions.json')['actions'] if stage=='validation' and (P/'validation-actions.json').exists() else None
  if stage=='validation' and actions is None:continue
  for root in roots:
   for action in (sorted(root['legal_actions']) if stage=='selection' else sorted(set(actions[root['root_id']].values()))):
    for seed in cfg[stage+'_seeds']:
     spec=cell_spec(root,stage,action,seed,cfg,expected);directory=P/'cells'/f"{stage}-{root['root_id']}-a{action:02d}-s{seed}"
     attempts=[read(p) for p in sorted(directory.glob('attempt-[0-9][0-9][0-9].json'))];result=successful_cell(attempts,spec)
     if result is not None:scores[stage,root['root_id'],action,seed]=result['score']
     rows.append({'stage':stage,'complete':result is not None,'attempts':len(attempts)})
 counts={s:{'complete':sum(r['complete'] for r in rows if r['stage']==s),'expected':sum(r['stage']==s for r in rows)} for s in ('selection','validation')}
 mechanism=None
 if counts['validation']['expected'] and counts['validation']['complete']==counts['validation']['expected']:
  actions=read(P/'validation-actions.json')['actions'];per_root=[]
  for root in validation['roots']:
   means={arm:sum(scores['validation',root['root_id'],action,seed] for seed in cfg['validation_seeds'])/len(cfg['validation_seeds']) for arm,action in actions[root['root_id']].items()}
   per_root.append({'root_id':root['root_id'],'scores':means,'terminal_minus_cp2':means['terminal_head']-means['cp2'],'terminal_minus_puct':means['terminal_head']-means['puct_head']})
  mechanism={'complete':True,'root_count':32,'per_root':per_root,'mean_terminal_minus_cp2':sum(r['terminal_minus_cp2'] for r in per_root)/32,'mean_terminal_minus_puct':sum(r['terminal_minus_puct'] for r in per_root)/32,'diagnostic_only':True}
 benchmark=read(P/'benchmark-summary.json') if (P/'benchmark-summary.json').exists() else None
 C.atomic_json(P/'derived-progress.json',{'counts':counts,'mechanism':mechanism,'benchmark':benchmark,'updated_at':C.now(),'playing_strength_claim':False})
 fig,ax=plt.subplots(figsize=(7,4));labels=list(counts);ax.bar(labels,[counts[x]['expected'] for x in labels],color='lightgray',label='scheduled');ax.bar(labels,[counts[x]['complete'] for x in labels],label='valid terminal continuations');ax.set_title('V11 collection coverage — not benchmark win rate');ax.legend();fig.tight_layout();fig.savefig(P/'coverage.png');plt.close(fig)
if __name__=='__main__':main()
