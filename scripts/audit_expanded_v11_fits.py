"""Read-only audit of saved V11 heads, optimizer/batch state and training actions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bits(t):return t.detach().cpu().contiguous().numpy().tobytes()
def actions(state,hidden,mask):
    logits=torch.from_numpy(hidden)@state['policy.weight'].T+state['policy.bias']
    return logits.masked_fill(~torch.from_numpy(mask.astype(bool)),-torch.inf).argmax(1).numpy()

def verify_batches(d,batches):
    assert set(batches)=={'root','anchor'} and all(x.shape==(4096,32) for x in batches.values())
    rng=np.random.default_rng(223101)
    groups=[np.flatnonzero((d['root_strata']==group).all(axis=1)) for group in np.unique(d['root_strata'],axis=0)]
    games=sorted(set(d['anchor_game'].tolist()))
    ag={(g,t):np.flatnonzero((d['anchor_game']==g)&(d['anchor_player']==t)) for g in games for t in (0,1)}
    for step in range(4096):
        for col in range(32):
            group=groups[int(rng.integers(len(groups)))];assert batches['root'][step,col]==group[int(rng.integers(len(group)))]
        for col in range(32):
            g=games[int(rng.integers(len(games)))];group=ag[g,int(rng.integers(2))]
            assert batches['anchor'][step,col]==group[int(rng.integers(len(group)))]
    return rng

def target_ambiguity(input_hashes, labels):
    """Equal-slot exact-label ceiling; diagnostic only, never deduplicates data."""
    from collections import Counter, defaultdict
    if len(input_hashes)!=len(labels) or not input_hashes:
        raise ValueError('complete equally sized slot/label lists required')
    groups=defaultdict(list)
    for index,(key,label) in enumerate(zip(input_hashes,labels)):
        groups[key].append((index,int(label)))
    conflicts=[];maximum=0
    for key,rows in groups.items():
        counts=Counter(label for _,label in rows);maximum+=max(counts.values())
        if len(counts)>1:
            conflicts.append({'input_hash':key,'slot_indices':[i for i,_ in rows],
                              'label_counts':dict(sorted(counts.items()))})
    return {'slots':len(labels),'unique_inputs':len(groups),
            'maximum_consistent_label_matches':maximum,
            'conflicting_input_groups':conflicts,
            'scope':'Equal-slot training-label ceiling for a deterministic policy; not playing strength; all original slots retained'}

def audit():
    torch.set_num_threads(1)
    run=ROOT/'runs/expanded-terminal-v11';baseline=ROOT/'runs/warmstart-v3/checkpoint-0002.pt'
    expected='20d9e813349010630940e1cdfbaee5206490c7d323997688d26f47508bbeee7c'
    assert sha(baseline)==expected
    base=torch.load(baseline,map_location='cpu',weights_only=True)['model_state_dict']
    meta=json.loads((run/'dataset/training.json').read_text())
    assert meta['complete'] and meta['root_count']==64 and meta['anchor_count']==1024
    assert meta['benchmark_data_used'] is False and meta['validation_used_for_gradients'] is False
    fit=json.loads((run/'fit/identity.json').read_text())
    manifest=json.loads((run/'fit/manifest.json').read_text())
    assert manifest['complete'] and set(manifest['checkpoints'])=={'terminal_head','puct_head'}
    assert meta['selection_cells']==2974 and manifest['fixed_candidate']=='terminal_head'
    assert fit['dataset_sha256']==manifest['dataset_sha256']==sha(run/'dataset/training.npz')
    assert fit['metadata_sha256']==sha(run/'dataset/training.json')
    fit_hash=hashlib.sha256(json.dumps(fit,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert fit_hash==manifest['fit_identity_sha256']
    identity=json.loads((ROOT/'progress/expanded-terminal-v11/identity.json').read_text())
    identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert fit['experiment_identity']==manifest['identity_sha256']==meta['identity_sha256']==identity_hash
    assert fit['fit']==identity['config']['fit'] and fit['base_sha256']==expected
    assert meta['input_plan_pins']==identity['input_plans']
    assert fit['fitter_sha256']==sha(ROOT/'alpha0/expanded_v11_fit.py')
    with np.load(run/'dataset/training.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    with np.load(run/'fit/batches.npz',allow_pickle=False) as f:batches={k:f[k] for k in f.files}
    assert set(batches)=={'root','anchor'} and all(x.shape==(4096,32) for x in batches.values())
    training_pin=identity['input_plans']['training']
    assert sha(ROOT/training_pin['path'])==training_pin['sha256']
    training=json.loads((ROOT/training_pin['path']).read_text())
    assert len(training['roots'])==64
    ambiguity={arm:target_ambiguity([r['input_hash'] for r in training['roots']],d[key])
               for arm,key in [('terminal_head','terminal_actions'),('puct_head','puct_actions')]}
    rng=verify_batches(d,batches)
    models={};counts={}
    for arm,target in [('terminal_head','terminal_actions'),('puct_head','puct_actions')]:
        path=run/'fit'/f'{arm}.pt';p=torch.load(path,map_location='cpu',weights_only=True);state=p['model_state_dict']
        assert p['updates']==4096 and p['trainable_parameters']==5160 and p['policy_head_only']
        assert p['experiment_id']=='expanded-terminal-v11' and p['parent_checkpoint_sha256']==expected
        assert p['benchmark_data_used'] is False and p['validation_data_used'] is False
        assert p['dataset_sha256']==sha(run/'dataset/training.npz') and p['fit_identity_sha256']==fit_hash
        h=hashlib.sha256()
        for key in ('root','anchor'):
            values=np.ascontiguousarray(batches[key],dtype='<i8')
            h.update(json.dumps([key,list(values.shape)],separators=(',',':')).encode());h.update(values.tobytes())
        rng_hash=hashlib.sha256(json.dumps(p['numpy_rng'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        h.update(rng_hash.encode());assert p['batch_sha256']==h.hexdigest()
        assert sha(path)==manifest['checkpoints'][arm]['sha256']
        snapshots=sorted((run/'fit'/f'states-{arm}').glob('step-*.json'))
        assert [json.loads(q.read_text())['step'] for q in snapshots]==list(range(128,4097,128))
        for q in snapshots:
            record=json.loads(q.read_text());assert record['fit_identity']==fit_hash and record['sha256']==sha(q.with_suffix('.pt'))
        final=torch.load(snapshots[-1].with_suffix('.pt'),map_location='cpu',weights_only=True)
        assert final['step']==4096 and final['arm']==arm
        assert all(bits(final['policy_state_dict'][k])==bits(state['policy.'+k]) for k in ('weight','bias'))
        assert torch.equal(final['torch_rng'],p['torch_rng']) and final['numpy_rng']==p['numpy_rng']
        assert final['batch_sha256']==p['batch_sha256']==fit['batch_sha256']
        assert set(state)==set(base)
        assert all(torch.isfinite(t).all() for t in state.values())
        changed=[k for k in base if bits(base[k])!=bits(state[k])]
        assert set(changed)=={'policy.weight','policy.bias'}
        assert sum(state[k].numel() for k in changed)==5160
        assert len(p['optimizer_state_dict']['state'])==2
        assert all(float(x['step'])==4096 for x in p['optimizer_state_dict']['state'].values())
        assert len(p['fit_history'])==4096 and p['numpy_rng']==rng.bit_generator.state
        for key in ('root','anchor'):assert np.array_equal(p['batch_indices'][key].numpy(),batches[key])
        pred=actions(state,d['root_hidden'],d['root_mask']);anchor=actions(state,d['anchor_hidden'],d['anchor_mask'])
        counts[arm]={'checkpoint_sha256':sha(path),'updates':4096,'only_policy_head_changed':True,
                     'training_matches':int((pred==d[target]).sum()),'training_roots':64,
                     'changed_training_actions':int((pred!=actions(base,d['root_hidden'],d['root_mask'])).sum()),
                     'changed_anchor_actions':int((anchor!=actions(base,d['anchor_hidden'],d['anchor_mask'])).sum()),'anchors':1024}
        assert counts[arm]['training_matches']==p['fit_metrics']['root_selected_action_matches']
        assert counts[arm]['changed_anchor_actions']==p['fit_metrics']['anchor_action_changes']
        models[arm]=p
    assert torch.equal(models['terminal_head']['torch_rng'],models['puct_head']['torch_rng'])
    return {'verified':True,'scope':'Saved fitting state and training predictions only; not playing-strength improvement or final recovery verification',
            'baseline_sha256':expected,'data_sha256':sha(run/'dataset/training.npz'),'shared_batches_sha256':sha(run/'fit/batches.npz'),
            'arms':counts,'target_ambiguity':ambiguity,'auditor_sha256':sha(Path(__file__))}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    result=audit();a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
