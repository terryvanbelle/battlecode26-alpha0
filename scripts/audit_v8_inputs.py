"""Independent read-only reconstruction of V8 cached-input and exclusion hashes.

This audit does not import the collector, learner or their hashing helpers and
never launches an engine. It checks the frozen inputs, not playing strength.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def reconstruct(cache,index):
    names=('local','global_board','scalars','mask');arrays=[];fields=[]
    shapes=((33,17,17),(33,60,60),(64,),(40,))
    for i,(name,shape) in enumerate(zip(names,shapes)):
        a=np.array(cache[name][index],dtype='u1' if i==3 else '<f4',order='C',copy=True)
        assert a.shape==shape and np.isfinite(a).all()
        if i==3:assert np.isin(a,[0,1]).all() and a.any()
        else:a[a==0]=0
        arrays.append(a);fields.append(dict(name='legal_mask' if i==3 else name,shape=list(shape),dtype='|u1' if i==3 else '<f4'))
    header=canonical(dict(schema='gradient-input-v1',feature_version=2,action_version='macro_v2',fields=fields))
    h=hashlib.sha256(len(header).to_bytes(8,'little')+header)
    for a in arrays:h.update(a.tobytes())
    return h.hexdigest(),set(np.flatnonzero(arrays[-1]).tolist())


def audit(root=ROOT):
    p=root/'progress/terminal-target-head-v8';read=lambda path:json.loads(path.read_text())
    identity=read(p/'identity.json');identity_hash=digest(identity)
    t=read(p/'roots-training.json');v=read(p/'roots-validation.json');a=read(p/'anchors.json');registry=read(p/'validation-exclusions.json')
    assert len(t['roots'])==32 and len(v['roots'])==16 and len(a['anchors'])==512
    for obj in (t,v,a):assert obj['complete'] and obj['identity_sha256']==identity_hash
    assert registry['identity_sha256']==identity_hash and not t['unavailable_slots'] and not v['unavailable_slots']
    records=t['roots']+a['anchors']+v['roots'];caches={};sources={};files={}
    for r in records:
        for key,hkey in [('cache_npz','cache_sha256'),('source_json','source_sha256')]:
            name=r[key];path=root/name
            assert path.resolve().is_relative_to((root/'runs/terminal-target-head-v8/sources').resolve())
            if name not in files:files[name]=file_sha(path)
            assert files[name]==r[hkey]
        name=r['cache_npz']
        if name not in caches:
            with np.load(root/name,allow_pickle=False) as c:caches[name]={k:c[k] for k in ('local','global_board','scalars','mask')}
        actual,legal=reconstruct(caches[name],r['feature_index'])
        assert actual==r['input_hash'] and legal==set(r['legal_actions'])
        src=r['source_json']
        if src not in sources:
            s=read(root/src);sources[src]=s
            assert s['status']=='ok' and s['terminal'] and s['valid_game'] and s['complete']
            assert not s['benchmark_failures'] and s['experiment_identity']==identity_hash
            assert s['checkpoint_sha256']==identity['checkpoint_sha256'] and s['source_temperature']==1
            m,j=s['map_index'],s['j'];train=s['split']=='training'
            assert s['seed']==(20263200 if train else 20264200)+100*m+j
            assert s['policy_seeds']==[2*((330000 if train else 340000)+100*m+j)+team for team in (0,1)]
        s=sources[src];assert s['map']==r['map'] and s['seed']==r['seed']
    excluded={'Stash','Hike','heart','ratsareants'}
    assert not excluded.intersection(r['map'] for r in records)
    for roots,count in [(t['roots'],2),(v['roots'],1)]:
        groups=Counter((r['map'],r['role'],r['player']) for r in roots)
        assert len(groups)==16 and set(groups.values())=={count}
        for r in roots:
            parity=r['slot_group_j']%2
            assert r['player']==(1-parity if r['role']=='BABY_RAT' else parity)
            assert sources[r['source_json']]['j']%2==parity
    gradient={r['input_hash'] for r in t['roots']+a['anchors']}
    oldt=read(root/'runs/terminal-target-head-v6/dataset/training.json')
    oldv=read(root/'progress/terminal-target-head-v6/roots-validation.json')
    historical=set(oldt['root_input_hashes']+oldt['anchor_input_hashes']+[r['input_hash'] for r in oldv['roots']])
    for name,sha in identity['historical_input_pins'].items():assert file_sha(root/name)==sha
    validation={r['input_hash'] for r in v['roots']}
    assert len(historical)==560 and len(validation)==16 and not validation.intersection(gradient|historical)
    for key,values in [('gradient_input_hashes',gradient),('historical_input_hashes',historical),('validation_exclusion_hashes',gradient|historical)]:
        assert registry[key]==sorted(values) and registry[key+'_sha256']==digest(sorted(values))
    return dict(verified=True,scope='Frozen cached model inputs and exclusions only; not completed training, benchmark improvement, or recovery verification',identity_sha256=identity_hash,
                training_roots=32,anchors=512,validation_roots=16,canonical_inputs_reconstructed=len(records),
                unique_gradient_inputs=len(gradient),historical_inputs=len(historical),exclusion_union=len(gradient|historical),
                validation_intersections=0,referenced_source_games=len(sources),selection_cells=sum(len(r['legal_actions']) for r in t['roots'])*2,
                verified_file_hashes=files,auditor_sha256=file_sha(Path(__file__)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    result=audit();args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='verified_file_hashes'},indent=2))
