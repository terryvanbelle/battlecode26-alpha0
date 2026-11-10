#!/usr/bin/env python3
"""Independent, read-only completed v6 head-fit audit; never trains or runs Java.

This file deliberately lives outside the experiment's frozen runtime. Run only
once both fitted candidates have been atomically published. It verifies tensor,
objective, batch, optimizer and recovery evidence, not the outcome provenance
of the terminal selector (covered by the separate experiment evidence audit).
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

ARMS = ('terminal_head', 'puct_head')
POLICY = {'policy.weight', 'policy.bias'}
EXPECTED_FIT = {'updates':128, 'batch_size':32, 'anchor_batch_size':32,
    'learning_rate':.0003, 'weight_decay':.0001, 'betas':[.9,.999], 'eps':1e-8,
    'seed':203101, 'target_weight':.75, 'anchor_kl_weight':1}
EXPECTED_BASE = '20d9e813349010630940e1cdfbaee5206490c7d323997688d26f47508bbeee7c'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def require(condition, description):
    if not condition:
        raise ValueError(description)

def tensor_bytes(value):
    return value.detach().cpu().contiguous().numpy().tobytes()

def same_tree(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and tensor_bytes(a) == tensor_bytes(b)
    if isinstance(a, dict):
        return isinstance(b,dict) and set(a)==set(b) and all(same_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return isinstance(b,(list,tuple)) and len(a)==len(b) and all(same_tree(x,y) for x,y in zip(a,b))
    return a == b

def frozen_digest(state):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        if key in POLICY:
            continue
        header = json.dumps([key,str(tensor.dtype),list(tensor.shape)],separators=(',',':')).encode()
        h.update(len(header).to_bytes(8,'little')); h.update(header); h.update(tensor_bytes(tensor))
    return h.hexdigest()

def normalize(prior, mask):
    require(prior.shape == mask.shape and prior.shape[-1] == 40, 'prior shape')
    require(np.isfinite(prior).all() and (prior >= 0).all(), 'invalid prior')
    require(np.isin(mask,(0,1)).all() and mask.any(-1).all(), 'invalid mask')
    x = np.where(mask, prior, 0).astype(np.float64)
    require((x.sum(-1) > 0).all(), 'zero legal prior mass')
    return (x / x.sum(-1,keepdims=True)).astype(np.float32)

def schedule(data):
    strata, games, teams = data['root_strata'], data['anchor_game'], data['anchor_player']
    groups = [np.flatnonzero((strata == (m,r,t)).all(1)) for m in range(4) for r in range(2) for t in range(2)]
    require(all(len(group)==2 for group in groups), 'root strata must contain two rows each')
    ids = sorted(set(games.tolist()))
    anchor = {g:[np.flatnonzero((games==g)&(teams==t)) for t in range(2)] for g in ids}
    require(all(len(x)>0 for groups in anchor.values() for x in groups), 'missing anchor game/team stratum')
    rng = np.random.default_rng(203101)
    result = {key:np.empty((128,32),dtype=np.int64) for key in ('root','anchor')}
    for step in range(128):
        for i in range(32):
            group = groups[int(rng.integers(16))]
            result['root'][step,i] = group[int(rng.integers(len(group)))]
        for i in range(32):
            game = ids[int(rng.integers(len(ids)))]
            group = anchor[game][int(rng.integers(2))]
            result['anchor'][step,i] = group[int(rng.integers(len(group)))]
    result['rng_state'] = copy.deepcopy(rng.bit_generator.state)
    h = hashlib.sha256()
    for key in ('root','anchor'):
        x = np.ascontiguousarray(result[key],dtype='<i8')
        h.update(json.dumps([key,list(x.shape)],separators=(',',':')).encode()); h.update(x.tobytes())
    h.update(fingerprint(result['rng_state']).encode())
    return result,h.hexdigest()

def log_prob(state, hidden, mask):
    raw = F.linear(torch.as_tensor(hidden),state['policy.weight'],state['policy.bias'])
    require(torch.isfinite(raw).all().item(), 'nonfinite policy logits')
    return raw.masked_fill(~torch.as_tensor(mask,dtype=torch.bool),torch.finfo(torch.float32).min).log_softmax(-1)

def metrics(state, data, actions, roots=None, anchors=None):
    roots = np.arange(32) if roots is None else roots
    anchors = np.arange(512) if anchors is None else anchors
    rp = normalize(data['root_prior'][roots],data['root_mask'][roots])
    ap = normalize(data['anchor_prior'][anchors],data['anchor_mask'][anchors])
    rlog = log_prob(state,data['root_hidden'][roots],data['root_mask'][roots])
    alog = log_prob(state,data['anchor_hidden'][anchors],data['anchor_mask'][anchors])
    selected = actions[roots]
    require(data['root_mask'][roots,selected].all(), 'selected action is illegal')
    target = torch.as_tensor(rp)*.25 + F.one_hot(torch.as_tensor(selected),40).float()*.75
    ce = -(target*rlog).sum(-1).mean()
    anchor_p = torch.as_tensor(ap)
    kl = (torch.xlogy(anchor_p,anchor_p)-anchor_p*alog).sum(-1).mean()
    choices, anchor_choices = rlog.argmax(-1).numpy(),alog.argmax(-1).numpy()
    result = {'loss':float(ce+kl),'target_ce':float(ce),'anchor_kl':float(kl),
        'root_target_agreement':float(np.mean(choices==selected)),
        'root_action_changed_fraction':float(np.mean(choices!=rp.argmax(-1))),
        'anchor_action_changed_fraction':float(np.mean(anchor_choices!=ap.argmax(-1)))}
    require(all(np.isfinite(v) for v in result.values()), 'nonfinite audit objective')
    return result,choices,anchor_choices

def value_outputs(state, hidden):
    return F.linear(F.silu(F.linear(hidden,state['value.0.weight'],state['value.0.bias'])),
                    state['value.2.weight'],state['value.2.bias'])

def checked_metrics(recomputed, saved, keys=None):
    keys = recomputed if keys is None else keys
    for key in keys:
        require(key in saved and abs(recomputed[key]-saved[key]) <= 2e-6, 'objective metric mismatch: '+key)

def audit(root):
    run = root/'runs/terminal-target-head-v6'
    fit = run/'fit'
    manifest_path = fit/'manifest.json'
    require(manifest_path.is_file(), 'No completed fit manifest: do not run this audit before both candidates are published')
    manifest = read(manifest_path)
    require(manifest.get('complete') is True and set(manifest.get('checkpoints',{}))==set(ARMS),'fit pair incomplete')
    require(manifest.get('experiment_id')=='terminal-target-head-v6','wrong experiment')
    require(manifest['fit_config']==EXPECTED_FIT,'fit protocol differs')
    expected_files = {arm+'.pt' for arm in ARMS}|{'batches.npz','batch-rng.json','parent-metadata.json','identity.json'}
    require(set(manifest['files'])==expected_files,'fit artifact inventory differs')
    for name, sha in manifest['files'].items():
        require(digest(fit/name)==sha,'artifact hash mismatch: '+name)
    identity = read(fit/'identity.json')
    require(identity==manifest['fit_identity'] and fingerprint(identity)==manifest['fit_identity_sha256'],'fit identity differs')
    require(identity['fit_config']==EXPECTED_FIT and identity['checkpoint_sha256']==EXPECTED_BASE,'fit ancestor/settings differ')
    require(identity['identity_sha256']==manifest['identity_sha256'],'experiment identity mismatch')
    experiment_identity_path=root/'progress/terminal-target-head-v6/identity.json'
    experiment_identity=read(experiment_identity_path)
    require(fingerprint(experiment_identity)==identity['identity_sha256'],'frozen experiment identity differs')
    require(experiment_identity['checkpoint_sha256']==EXPECTED_BASE
        and experiment_identity['config']['fit']==EXPECTED_FIT,'frozen experiment fit protocol differs')
    for name,key in (('alpha0/terminal_v6_fit.py','fitter_sha256'),('alpha0/terminal_v6_common.py','common_sha256')):
        require(identity[key]==experiment_identity['sources'][name]==digest(root/name)
            ==digest(root/'progress/terminal-target-head-v6/source'/name),'fitter/common source identity differs')
    dataset = run/'dataset/training.npz'
    meta = read(dataset.with_suffix('.json'))
    require(digest(dataset)==identity['dataset_sha256']==manifest['dataset']['sha256']==meta['npz_sha256'],'dataset hash mismatch')
    require(digest(dataset.with_suffix('.json'))==identity['dataset_metadata_sha256']==manifest['dataset']['metadata_sha256'],'dataset metadata hash mismatch')
    require(meta['complete'] is True and meta['source']=='own_selfplay' and meta['benchmark_data_used'] is False
            and meta['validation_data_used'] is False,'dataset provenance markers invalid')
    require(meta['identity_sha256']==manifest['identity_sha256'],'dataset experiment identity differs')
    gradient = set(meta['root_input_hashes'])|set(meta['anchor_input_hashes'])
    require(len(meta['root_input_hashes'])==32 and len(meta['anchor_input_hashes'])==512
            and len(meta['validation_input_hashes'])==16,'input guard counts differ')
    require(gradient==set(meta['gradient_input_hashes']) and not gradient.intersection(meta['validation_input_hashes']),'input overlap guard failed')
    guard = {'checked':True,'gradient_input_count':len(gradient),'validation_input_count':16,
             'intersection_count':0,'set_sha256':fingerprint(sorted(gradient))}
    require(guard==manifest['gradient_input_overlap'],'input guard report differs')

    # Import numerical libraries only after the complete-pair preflight above.
    global np,torch,F
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[name]='1'
    import numpy as np
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(identity['torch_version']==str(torch.__version__) and identity['numpy_version']==np.__version__,
        'numerical library version differs from fitting')
    with np.load(dataset,allow_pickle=False) as archive:
        data = {key:archive[key] for key in archive.files}
    shapes = {'root_hidden':(32,128),'root_prior':(32,40),'root_mask':(32,40),'terminal_actions':(32,),
        'puct_actions':(32,),'root_strata':(32,3),'anchor_hidden':(512,128),'anchor_prior':(512,40),
        'anchor_mask':(512,40),'anchor_game':(512,),'anchor_player':(512,)}
    require(set(data)==set(shapes),'dataset array inventory differs')
    for key,shape in shapes.items():
        dtype = np.uint8 if key.endswith('mask') else np.float32 if key.endswith(('hidden','prior')) else np.int64
        require(data[key].shape==shape and data[key].dtype==dtype,'dataset shape/dtype: '+key)
        require(np.isfinite(data[key]).all(),'nonfinite dataset: '+key)
    # Match the production loader's first float64-to-float32 normalization;
    # each objective calculation deliberately normalizes this cached p0 again.
    for prefix in ('root','anchor'):
        data[prefix+'_prior']=normalize(data[prefix+'_prior'],data[prefix+'_mask'])
    batches,batch_sha = schedule(data)
    require(identity['batch_sha256']==batch_sha,'fit identity batch digest differs')
    batch_record=manifest['batch_schedule']
    require((root/batch_record['path']).resolve()==(fit/'batches.npz').resolve()
        and batch_record['sha256']==manifest['files']['batches.npz']
        and batch_record['root_shape']==[128,32] and batch_record['anchor_shape']==[128,32]
        and batch_record['seed']==203101,'batch schedule manifest differs')
    require(read(fit/'batch-rng.json')==batches['rng_state'],'batch RNG sidecar mismatch')
    with np.load(fit/'batches.npz',allow_pickle=False) as recorded:
        require(set(recorded.files)=={'root','anchor'} and all(recorded[k].dtype==np.int64
            and np.array_equal(recorded[k],batches[k]) for k in ('root','anchor')),'stored sampler differs')
    base_path = root/'runs/warmstart-v3/checkpoint-0002.pt'
    require(digest(base_path)==EXPECTED_BASE,'cp2 baseline changed')
    base_payload = torch.load(base_path,map_location='cpu',weights_only=True)
    base = base_payload['model_state_dict']
    ancestor = {k:v for k,v in base_payload.items() if k not in ('model_state_dict','optimizer_state_dict','torch_rng','numpy_rng')}
    require(read(fit/'parent-metadata.json')==ancestor,'parent metadata sidecar differs')
    frozen_sha = frozen_digest(base)
    require(frozen_sha==manifest['frozen_state_sha256'],'baseline frozen-state hash differs')
    for prefix in ('root','anchor'):
        p = log_prob(base,data[prefix+'_hidden'],data[prefix+'_mask']).exp().numpy()
        require(np.allclose(p,normalize(data[prefix+'_prior'],data[prefix+'_mask']),atol=1e-6,rtol=1e-5),'cached cp2 priors differ: '+prefix)
    hidden = torch.as_tensor(np.concatenate((data['root_hidden'],data['anchor_hidden'])))
    baseline_values = value_outputs(base,hidden)
    seeded = torch.Generator().manual_seed(203101)
    baseline_torch_rng = seeded.get_state()
    seeded_draws = torch.rand(16,generator=seeded)
    rng = np.random.default_rng(); rng.bit_generator.state = copy.deepcopy(batches['rng_state'])
    reference_numpy_draws = rng.integers(2**31,size=16)
    report = {'format_version':1,'experiment_id':'terminal-target-head-v6','audit_kind':'independent-post-fit-head-tensors-and-recovery',
        'fit_manifest_sha256':digest(manifest_path),'fit_identity_sha256':manifest['fit_identity_sha256'],
        'experiment_identity_sha256':fingerprint(experiment_identity),'experiment_identity_file_sha256':digest(experiment_identity_path),
        'numerical_versions':{'torch':str(torch.__version__),'numpy':np.__version__},
        'dataset_sha256':digest(dataset),'dataset_metadata_sha256':digest(dataset.with_suffix('.json')),
        'parent_checkpoint_sha256':EXPECTED_BASE,'batch_sha256':batch_sha,'frozen_state_sha256':frozen_sha,
        'no_engine_runs':True,'no_gradient_updates':True,'no_benchmark_data_used':True,
        'gradient_input_overlap':guard,'arms':{}}
    choices_by_arm = {}; anchors_by_arm = {}; candidate_states = {}
    with torch.inference_mode():
        for arm in ARMS:
            path = fit/(arm+'.pt'); record = manifest['checkpoints'][arm]
            require((root/record['path']).resolve()==path.resolve() and digest(path)==record['sha256'],'candidate path/hash differs: '+arm)
            payload = torch.load(path,map_location='cpu',weights_only=True); state = payload['model_state_dict']
            require(payload['arm']==arm and payload['updates']==payload['gradient_updates']==record['updates']==128,'candidate update/arm mismatch')
            require(payload['trainable_parameters']==record['trainable_parameters']==5160 and payload['policy_head_only'] is True,'head parameter declaration differs')
            require(payload['fit_identity_sha256']==manifest['fit_identity_sha256'] and payload['parent_checkpoint_sha256']==EXPECTED_BASE,'candidate lineage identity differs')
            require(payload['dataset_sha256']==identity['dataset_sha256'] and payload['dataset_metadata_sha256']==identity['dataset_metadata_sha256'],'candidate dataset identity differs')
            require(payload['parent_checkpoint_metadata']==ancestor and payload['gradient_input_overlap']==guard,'candidate parent/guard differs')
            require(payload['benchmark_data_used'] is False and payload['validation_data_used'] is False,'candidate provenance differs')
            require(payload['config']['initialization']['artifact_sha256']==EXPECTED_BASE,'candidate initialization differs')
            require(payload['config']['fit']==EXPECTED_FIT,'candidate fit settings differ')
            require(set(state)==set(base),'model tensor inventory differs')
            changed = []
            for key in base:
                require(state[key].shape==base[key].shape and state[key].dtype==base[key].dtype,'tensor shape/dtype differs: '+key)
                require(torch.isfinite(state[key]).all().item(),'nonfinite tensor: '+key)
                if tensor_bytes(state[key])!=tensor_bytes(base[key]): changed.append(key)
            require(set(changed)<=POLICY and sum(base[k].numel() for k in POLICY)==5160,'updated non-policy tensors or wrong head size')
            require(frozen_digest(state)==frozen_sha==payload['frozen_state_sha256'],'candidate frozen bytes differ')
            require(tensor_bytes(value_outputs(state,hidden))==tensor_bytes(baseline_values),'frozen critic output differs')
            require(payload['batch_sha256']==batch_sha,'candidate batch digest differs')
            require(all(payload['batch_indices'][k].dtype==torch.int64 and np.array_equal(payload['batch_indices'][k].numpy(),batches[k]) for k in ('root','anchor')),'candidate batch indices differ')
            require(same_tree(payload['numpy_rng'],batches['rng_state']) and same_tree(payload['torch_rng'],baseline_torch_rng),'candidate RNG state differs')
            restored = torch.Generator(); restored.set_state(payload['torch_rng'])
            require(torch.equal(torch.rand(16,generator=restored),seeded_draws),'Torch RNG recovery differs')
            restored_np = np.random.default_rng(); restored_np.bit_generator.state=copy.deepcopy(payload['numpy_rng'])
            require(np.array_equal(restored_np.integers(2**31,size=16),reference_numpy_draws),'NumPy RNG recovery differs')
            saved_opt = payload['optimizer_state_dict']; groups=saved_opt['param_groups']; states=saved_opt['state']
            require(len(groups)==1 and groups[0]['params']==[0,1] and set(states)=={0,1},'optimizer parameter inventory differs')
            group=groups[0]
            require(group['lr']==.0003 and group['weight_decay']==.0001 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8,'optimizer settings differ')
            for index,key in enumerate(('policy.weight','policy.bias')):
                item=states[index]
                require(set(item)=={'step','exp_avg','exp_avg_sq'},'optimizer state key inventory differs')
                require(item['step'].dtype==torch.float32 and item['step'].shape==torch.Size([])
                    and float(item['step'])==128,'optimizer step count/dtype differs')
                for moment in ('exp_avg','exp_avg_sq'):
                    require(item[moment].dtype==state[key].dtype and item[moment].shape==state[key].shape
                        and torch.isfinite(item[moment]).all().item(),'invalid optimizer moment')
                require((item['exp_avg_sq']>=0).all().item(),'negative second moment')
            # Loading state is checked without taking an optimizer step.
            parameters=[torch.nn.Parameter(state[k].clone()) for k in ('policy.weight','policy.bias')]
            opt=torch.optim.AdamW(parameters,lr=.0003,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
            require(same_tree(opt.state_dict()['param_groups'],groups),'optimizer flags/defaults differ')
            opt.load_state_dict(copy.deepcopy(saved_opt))
            require(same_tree(opt.state_dict(),saved_opt),'optimizer round-trip differs')
            actions=data['terminal_actions' if arm=='terminal_head' else 'puct_actions']
            objective,chosen,anchor_chosen=metrics(state,data,actions)
            checked_metrics(objective,payload['fit_metrics']); checked_metrics(objective,record['metrics'])
            history=payload['fit_history']
            require(len(history)==128 and [x['step'] for x in history]==list(range(1,129)),'fit history update sequence differs')
            require(all(all(np.isfinite(row[k]) for k in ('loss','target_ce','anchor_kl')) for row in history),'nonfinite fit history')
            first,_,_=metrics(base,data,actions,batches['root'][0],batches['anchor'][0])
            checked_metrics(first,history[0],('loss','target_ce','anchor_kl'))
            report['arms'][arm]={'checkpoint_sha256':digest(path),'changed_tensor_keys':changed,
                'allowed_policy_parameters':5160,'changed_parameter_values':sum(int((state[k]!=base[k]).sum()) for k in POLICY),
                'policy_delta_l2':float(torch.cat([(state[k]-base[k]).flatten() for k in sorted(POLICY)]).norm()),
                'frozen_tensor_bytes_equal':True,'frozen_critic_outputs_equal_544_inputs':True,
                'shared_batches_reconstructed':True,'optimizer_updates':128,'optimizer_round_trip_exact':True,
                'rng_recovery_exact':True,'initial_batch_objective_matches':True,'full_cached_metrics':objective,
                'root_choices':chosen.tolist(),'anchor_choices_sha256':hashlib.sha256(anchor_chosen.astype('<i8').tobytes()).hexdigest()}
            choices_by_arm[arm]=chosen; anchors_by_arm[arm]=anchor_chosen; candidate_states[arm]=state
    terminal,puct=data['terminal_actions'],data['puct_actions']
    root_rows=[]
    for i in range(32):
        root_rows.append({'root_id':meta['root_ids'][i],'stratum':data['root_strata'][i].tolist(),
            'terminal_target':int(terminal[i]),'puct_target':int(puct[i]),'targets_equal':bool(terminal[i]==puct[i]),
            'cp2_greedy':int(normalize(data['root_prior'][i],data['root_mask'][i]).argmax()),
            **{arm+'_greedy':int(choices_by_arm[arm][i]) for arm in ARMS}})
    report['target_comparison']={'root_count':32,'same_selected_targets':int((terminal==puct).sum()),
        'different_selected_targets':int((terminal!=puct).sum()),'rows':root_rows,
        'candidate_root_greedy_agreement':float(np.mean(choices_by_arm[ARMS[0]]==choices_by_arm[ARMS[1]])),
        'candidate_anchor_greedy_agreement':float(np.mean(anchors_by_arm[ARMS[0]]==anchors_by_arm[ARMS[1]])),
        'head_pair_l2_difference':float(torch.cat([(candidate_states[ARMS[0]][k]-candidate_states[ARMS[1]][k]).flatten() for k in sorted(POLICY)]).norm())}
    # Ensure none of the audited files changed while being read.
    require(digest(manifest_path)==report['fit_manifest_sha256'] and digest(dataset)==report['dataset_sha256']
        and digest(dataset.with_suffix('.json'))==report['dataset_metadata_sha256'] and digest(base_path)==EXPECTED_BASE,'audit inputs changed')
    require(all(digest(fit/name)==sha for name,sha in manifest['files'].items()),'fit artifact changed during audit')
    report['passed']=True
    report['scope_limitations']=[
        'Reads completed tensors/cached features only; does not independently regenerate feature inputs from game prefixes.',
        'Checks metadata input-overlap sets; the separate evidence audit verifies their reconstruction and chronology.',
        'Does not independently replay all 128 gradient updates; verifies saved update counts, optimizer state, batch reconstruction and objective values.',
        'Training-root/anchor metrics are in-sample; no generalization or benchmark-strength claim follows.']
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/workspace/battlecode26-alpha0'))
    parser.add_argument('--output',type=Path,default=Path('/tmp/v6-fit-independent-audit.json'))
    args=parser.parse_args()
    result=audit(args.root.resolve())
    result['audit_source_sha256']=digest(__file__)
    payload=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        require(args.output.read_bytes()==payload,'refusing to overwrite differing audit report')
    else:
        fd,name=tempfile.mkstemp(prefix='.fit-audit-',dir=args.output.parent)
        try:
            with os.fdopen(fd,'wb') as stream: stream.write(payload); stream.flush(); os.fsync(stream.fileno())
            os.link(name,args.output)
        finally:
            os.unlink(name)
    print(json.dumps({'passed':True,'report':str(args.output),'sha256':digest(args.output),
        'targets_equal':result['target_comparison']['same_selected_targets'],
        'metrics':{k:v['full_cached_metrics'] for k,v in result['arms'].items()}},sort_keys=True))

if __name__=='__main__':
    main()
