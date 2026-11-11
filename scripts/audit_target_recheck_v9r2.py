"""Independent read-only terminal/schedule audit; no collector or reporter imports."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'progress/target-recheck-v9r2'
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def audit():
    identity=read(P/'identity.json');cfg=read(ROOT/'configs/target-recheck-v9r2.json');assert cfg==identity['config']
    for name,h in identity['files'].items():assert sha(ROOT/name)==h
    pairs=read(P/'actions.json')['roots'];assert len(pairs)==32 and len({p['root_id'] for p in pairs})==32
    inherited=read(ROOT/'configs/target-recheck-v9r2-inherited.json');assert len(inherited)==8
    evidence={};views=[];failed=[];inherited_used=0
    planned=set()
    for pair in pairs:
        values={}
        for action in sorted({pair['selected_action'],pair['cp2_action']}):
            for seed in cfg['seeds']:
                name=f"{pair['root_id']}-a{action}-s{seed}";planned.add(name)
                files=sorted((P/'cells'/name).glob('attempt-[0-9][0-9][0-9].json'))
                imported=False
                if name in inherited:
                    assert not files,'inherited successful cell was rerun'
                    pin=inherited[name];path=ROOT/pin['path'];assert sha(path)==pin['sha256']
                    files=[path];imported=True
                successes=[]
                for path in files:
                    r=read(path);evidence[str(path.relative_to(ROOT))]=sha(path)
                    assert r['experiment_identity']==(inherited[name]['experiment_identity'] if imported else digest(identity))
                    assert r['root_id']==pair['root_id'] and r['root_player']==pair['player'] and r['input_hash']==pair['input_hash']
                    assert r['action']==action and r['seed']==seed and r['temperature']==.35
                    if r['status']!='ok':failed.append(str(path.relative_to(ROOT)));continue
                    raw=r['terminal_state'];winner=raw['winner'];actions=r['actions']
                    assert r['terminal'] is True and r['valid_game'] is True and r['benchmark_failures']==[]
                    assert raw['terminal'] is True and raw.get('valid_game',True) is True and raw.get('benchmark_failures',[])==[]
                    assert type(winner) is int and winner in (-1,0,1) and winner==r['winner']
                    assert type(raw['round']) is int and raw['round']>0 and raw['round']==r['terminal_round']
                    assert digest(raw)==r['terminal_state_sha256'] and digest(actions)==r['actions_sha256']
                    assert actions and actions[0]==action and all(type(a) is int and 0<=a<40 for a in actions)
                    if imported:assert 'decisions' not in r;inherited_used+=1
                    else:assert type(r['decisions']) is int and r['decisions']==len(actions)
                    assert r['score']==(.5 if winner==-1 else float(winner==pair['player']))
                    successes.append(r['score'])
                assert len(successes)<=1
                if successes:values[action,seed]=successes[0]
        complete=len(values)==len({pair['selected_action'],pair['cp2_action']})*4
        delta=sum(values[pair['selected_action'],s]-values[pair['cp2_action'],s] for s in cfg['seeds'])/4 if complete else None
        views.append({'root_id':pair['root_id'],'source':pair['source_json'],'complete':complete,'paired_advantage':delta,'completed_cells':len(values)})
    assert len(planned)==184
    for path in (P/'cells').glob('*/attempt-[0-9][0-9][0-9].json'):assert path.parent.name in planned
    complete=all(v['complete'] for v in views) and not failed
    result={'complete':complete,'verified_cells':sum(v['completed_cells'] for v in views),'expected_cells':184,
            'inherited_cells_verified':inherited_used,'failed_attempts':failed,'roots':views,'journal_hashes':evidence,
            'no_sgd':True,'benchmark_improvement':False,'auditor_sha256':sha(Path(__file__))}
    if complete:
        sources=sorted({v['source'] for v in views});groups=[[v['paired_advantage'] for v in views if v['source']==s] for s in sources]
        rng=np.random.default_rng(219101);boot=[]
        for _ in range(10000):
            sampled=[x for idx in rng.integers(len(groups),size=len(groups)) for x in groups[idx]];boot.append(sum(sampled)/len(sampled))
        mean=sum(v['paired_advantage'] for v in views)/32;interval=np.quantile(boot,[.025,.975]).tolist()
        result.update(mean_advantage=mean,source_cluster_bootstrap_95=interval,review_for_expansion=mean>=.1 and interval[0]>0)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit()
    with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('roots','journal_hashes')},indent=2))
