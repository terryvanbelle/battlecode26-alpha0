"""Draft V11 matched-head fitting/publication; not a standalone launcher."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from alpha0 import terminal_v8_common as C
from alpha0 import terminal_v8_fit as F
from alpha0.learn_v2 import load_checkpoint
from alpha0.expanded_v11_support import frozen_plans,build_complete_dataset,cached_position

FIT={'updates':4096,'batch_size':32,'anchor_batch_size':32,'learning_rate':.0003,'weight_decay':.0001,'betas':[.9,.999],'eps':1e-8,'seed':223101,'target_weight':.75,'anchor_kl_weight':1}
ARMS=('terminal_head','puct_head')

def validate_config(config):
    if config.get('fit')!=FIT or config.get('torch_threads')!=1 or config.get('experiment_id')!='expanded-terminal-v11':raise ValueError('fit protocol changed')
    if config.get('selection_seeds')!=[223003,223013]:raise ValueError('target seeds changed')

def verify_complete(manifest,identity,base_model):
    if not manifest['complete'] or manifest['identity_sha256']!=C.fingerprint(identity) or set(manifest['checkpoints'])!=set(ARMS):raise ValueError('incomplete or foreign fit pair')
    for arm,row in manifest['checkpoints'].items():
        path=C.ROOT/row['path']
        if C.sha256(path)!=row['sha256']:raise ValueError('candidate changed')
        model,payload=load_checkpoint(path)
        if F.frozen_state_sha256(model)!=F.frozen_state_sha256(base_model) or payload['updates']!=4096 or payload['arm']!=arm or payload['dataset_sha256']!=manifest['dataset_sha256']:raise ValueError('candidate frozen tensors/provenance changed')
        if payload['fit_identity_sha256']!=manifest['fit_identity_sha256']:raise ValueError('candidate fit identity mismatch')
    return manifest

def fit_pair(run_dir,progress_dir,identity,check_stop):
    config=identity['config'];validate_config(config);torch.set_num_threads(1)
    training,validation,_=frozen_plans(identity)
    checkpoint=C.ROOT/config['checkpoint']
    if C.sha256(checkpoint)!=config['checkpoint_sha256']:raise ValueError('ancestor changed')
    base,payload=load_checkpoint(checkpoint)
    arrays,metadata=build_complete_dataset(progress_dir,identity)
    F._verify_cached_priors(base,arrays)
    run_dir=Path(run_dir);progress_dir=Path(progress_dir);directory=run_dir/'fit';directory.mkdir(parents=True,exist_ok=True)
    dataset=run_dir/'dataset/training.npz';C.immutable_npz(dataset,arrays);C.immutable_json(dataset.with_suffix('.json'),metadata)
    batches=F.make_batch_schedule(arrays['root_strata'],arrays['anchor_game'],arrays['anchor_player'],FIT)
    C.immutable_npz(directory/'batches.npz',{k:batches[k] for k in ('root','anchor')});C.immutable_json(directory/'batch-rng.json',batches['rng_state'])
    fit_identity={'experiment_identity':C.fingerprint(identity),'dataset_sha256':C.sha256(dataset),'metadata_sha256':C.sha256(dataset.with_suffix('.json')),'batch_sha256':F.batch_fingerprint(batches),'fit':FIT,'fitter_sha256':C.sha256(Path(__file__)),'base_sha256':config['checkpoint_sha256']}
    C.immutable_json(directory/'identity.json',fit_identity);fit_hash=C.fingerprint(fit_identity)
    manifest_path=directory/'manifest.json'
    if manifest_path.exists():return verify_complete(json.loads(manifest_path.read_text()),identity,base)
    results={};parent={k:deepcopy(v) for k,v in payload.items() if k not in ('model_state_dict','optimizer_state_dict','torch_rng','numpy_rng')}
    for arm in ARMS:
        check_stop();generations=directory/f'states-{arm}';generations.mkdir(exist_ok=True);resume=None
        for file in sorted(generations.glob('step-*.json')):
            record=json.loads(file.read_text());state_path=file.with_suffix('.pt')
            if record['fit_identity']!=fit_hash or C.sha256(state_path)!=record['sha256']:raise ValueError('fit resume changed')
            state=torch.load(state_path,map_location='cpu',weights_only=True)
            if state['step']!=record['step']:raise ValueError('resume step changed')
            resume=state
        def persist(state):
            if state['step']%128==0 or state['step']==4096:
                path=generations/f"step-{state['step']:05d}.pt";digest=F._immutable_torch(path,state)
                C.immutable_json(path.with_suffix('.json'),{'step':state['step'],'fit_identity':fit_hash,'sha256':digest})
            check_stop()
        model,optimizer,state,metrics=F.fit_cached_head(base,arrays,batches,FIT,arm,resume=resume,on_update=persist)
        ancestry=deepcopy(config);ancestry['initialization']={'artifact':config['checkpoint'],'artifact_sha256':config['checkpoint_sha256'],'method':'frozen cp2 weights-only ancestor'}
        maps=sorted(set(config['maps'])|set(payload.get('training_maps',payload.get('config',{}).get('maps',[]))))
        if set(maps)&{'Stash','Hike','heart','ratsareants'}:raise ValueError('excluded map in checkpoint ancestry')
        games={r['source_id']:{'map':r['map'],'seed':r['seed']} for r in training['anchors']}
        candidate=model.checkpoint(experiment_id=config['experiment_id'],arm=arm,fit_version='expanded-v11',config=ancestry,parent_checkpoint_sha256=config['checkpoint_sha256'],parent_checkpoint_metadata=parent,training_maps=maps,training_seeds=[g['seed'] for g in games.values()],games_total=64,positions_total=1088,iteration=0,updates=4096,gradient_updates=4096,policy_head_only=True,trainable_parameters=5160,benchmark_data_used=False,validation_data_used=False,dataset_sha256=fit_identity['dataset_sha256'],fit_identity_sha256=fit_hash,batch_sha256=F.batch_fingerprint(batches),batch_indices={k:torch.from_numpy(batches[k].copy()) for k in ('root','anchor')},optimizer_state_dict=optimizer.state_dict(),torch_rng=state['torch_rng'],numpy_rng=state['numpy_rng'],fit_history=state['history'],fit_metrics=metrics,frozen_state_sha256=F.frozen_state_sha256(base),terminal_target_continuations=metadata['selection_cells'],input_plan_pins=identity['input_plans'])
        path=directory/f'{arm}.pt';digest=F._immutable_torch(path,candidate)
        results[arm]={'path':str(path.relative_to(C.ROOT)),'sha256':digest,'metrics':metrics}
    manifest={'complete':True,'identity_sha256':C.fingerprint(identity),'fit_identity_sha256':fit_hash,'dataset_sha256':fit_identity['dataset_sha256'],'checkpoints':results,'fixed_candidate':'terminal_head','updates_per_arm':4096}
    verify_complete(manifest,identity,base);C.immutable_json(manifest_path,manifest);C.immutable_json(progress_dir/'fit-summary.json',manifest)
    return manifest

def freeze_validation_actions(progress_dir,identity,fit):
    _,validation,_=frozen_plans(identity);config=identity['config'];base,_=load_checkpoint(C.ROOT/config['checkpoint']);verify_complete(fit,identity,base)
    checkpoints={'cp2':{'path':config['checkpoint'],'sha256':config['checkpoint_sha256']},**fit['checkpoints']};models={}
    for arm,row in checkpoints.items():
        path=C.ROOT/row['path']
        if C.sha256(path)!=row['sha256']:raise ValueError('candidate changed before validation')
        models[arm]=load_checkpoint(path)[0].eval()
    actions={}
    with torch.inference_mode():
        for root in validation['roots']:
            arrays=cached_position(root);hidden=torch.from_numpy(arrays['hidden'])[None];mask=torch.from_numpy(arrays['mask'].astype(bool))[None]
            actions[root['root_id']]={arm:int(model.policy(hidden).masked_fill(~mask,-torch.inf).argmax(-1).item()) for arm,model in models.items()}
    result={'identity_sha256':C.fingerprint(identity),'checkpoints':checkpoints,'actions':actions,'validation_manifest_sha256':identity['input_plans']['validation']['sha256']}
    C.immutable_json(Path(progress_dir)/'validation-actions.json',result)
    return result
