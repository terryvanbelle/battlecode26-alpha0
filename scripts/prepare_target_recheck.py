"""Read-only planning inventory; no new games, training or final preregistration."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
    progress=ROOT/'progress/terminal-target-head-v8'
    manifest=json.loads((progress/'roots-training.json').read_text())
    data=ROOT/'runs/terminal-target-head-v8/dataset/training.npz'
    meta=json.loads(data.with_suffix('.json').read_text())
    assert meta['complete'] and meta['npz_sha256']==sha(data)
    d=np.load(data,allow_pickle=False)
    roots=manifest['roots'];assert len(roots)==32
    pairs=[]
    for i,r in enumerate(roots):
        assert r['root_ordinal']==i
        a=int(d['terminal_actions'][i]);b=int(np.where(d['root_mask'][i],d['root_prior'][i],-np.inf).argmax())
        scores={}
        for action in sorted({a,b}):
            values=[]
            for seed in (205003,205013):
                paths=list((progress/'cells'/f"selection-{r['root_id']}-a{action:02d}-s{seed}").glob('attempt-[0-9][0-9][0-9].json'))
                valid=[json.loads(p.read_text()) for p in paths if json.loads(p.read_text())['status']=='ok']
                assert len(valid)==1
                row=valid[0];assert row['root_input_hash']==r['input_hash'] and row['action']==action and row['seed']==seed
                values.append(row['score'])
            scores[str(action)]=sum(values)/2
        pairs.append({'root_id':r['root_id'],'input_hash':r['input_hash'],'source_json':r['source_json'],
            'source_sha256':r['source_sha256'],'map':r['map'],'player':r['player'],'role':r['role'],
            'selected_action':a,'cp2_action':b,'distinct_actions':len({a,b}),
            'selection_seed_advantage':scores[str(a)]-scores[str(b)]})
    return {'status':'unlaunched_planning_only','no_new_games_or_gradients':True,
        'training_npz_sha256':sha(data),'roots_manifest_sha256':sha(progress/'roots-training.json'),
        'roots':pairs,'root_count':32,'changed_actions':sum(p['selected_action']!=p['cp2_action'] for p in pairs),
        'distinct_cells_for_four_new_seeds':sum(p['distinct_actions'] for p in pairs)*4,
        'selection_seed_mean_advantage':sum(p['selection_seed_advantage'] for p in pairs)/32,
        'warning':'Selection-seed advantage is biased by maximization; it is not independent evidence or playing strength.',
        'script_sha256':sha(Path(__file__))}
if __name__=='__main__':
    result=prepare();out=ROOT/'diagnostics/target-recheck-planning';out.mkdir(parents=True,exist_ok=True)
    with (out/'action-inventory.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='roots'},indent=2))
