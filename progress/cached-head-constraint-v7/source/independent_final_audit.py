#!/usr/bin/env python3
"""Independent v7 final audit. Run with .venv/bin/python after quiescence.
No experiment-module imports, optimizer creation/steps, Java, or held-out data.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import argparse, fcntl, hashlib, json, math, os, platform, re, tempfile
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
import numpy as np

ROOT=Path('/workspace/battlecode26-alpha0')
IH='ae384626094084e15417051b066cecdb9d480436cac9c5a016ac4ba3f4c96eb2'
NAME='cached-head-constraint-v7'
ARMS={'kl_1':1.,'kl_0':0.}
STEPS=(128,512,2048)
POLICY={'policy.weight','policy.bias'}
LOCKS=('runs/campaign/campaign.lock','runs/warmstart-v3/experiment.lock','runs/terminal-target-head-v5/experiment.lock','runs/terminal-target-head-v6/experiment.lock',f'runs/{NAME}/experiment.lock')
STOPS=tuple(str(Path(p).with_name('STOP')) for p in LOCKS)

def need(ok,message):
    if not ok:raise ValueError(message)
def fp(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def stamp(value):
    parsed=datetime.fromisoformat(value);need(parsed.tzinfo is not None,'timestamp lacks timezone');return parsed.timestamp()
def number(v):return type(v) in (int,float) and math.isfinite(v)
def actors():
    found=[];names={'alpha0.constraint_v7','alpha0.terminal_v5','alpha0.terminal_v6','alpha0.root_ranking_v4','alpha0.campaign','alpha0.learn_v2','alpha0.train','alpha0.evaluate','alpha0.victory','alpha0.EngineBridge'}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:argv=[x for x in p.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0') if x]
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:found.append({'pid':int(p.name),'reason':'unreadable command'});continue
        if not any(x in names or Path(x).name in ('run_warmstart_experiment.py','run_campaign.py') for x in argv):continue
        try:
            cwd=Path(os.readlink(p/'cwd')).resolve()
            if cwd!=ROOT:continue
            reason='matching actor in repository'
        except OSError as e:
            if not p.exists():continue
            reason='matching actor with unreadable cwd: '+type(e).__name__
        found.append({'pid':int(p.name),'command':argv,'reason':reason})
    return found

def tensorbytes(t):return t.detach().cpu().contiguous().numpy().tobytes()
def same(a,b):
    if torch.is_tensor(a) or torch.is_tensor(b):return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and tensorbytes(a)==tensorbytes(b)
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b

def normal(prior,mask):
    p=np.where(mask,prior,0).astype(np.float64);need(np.isfinite(p).all() and (p>=0).all() and (p.sum(-1)>0).all(),'invalid prior');return (p/p.sum(-1,keepdims=True)).astype(np.float32)

def classify(counts):
    need(set(counts)==set(ARMS),'missing arm');first={}
    for arm in ARMS:
        need(set(counts[arm])==set(map(str,STEPS)),'missing/extra fixed snapshot')
        need(all(type(n)is int and 0<=n<=32 for n in counts[arm].values()),'invalid agreement count')
        full=[step for step in STEPS if counts[arm][str(step)]==32];first[arm]=min(full) if full else None
    need(first['kl_1']!=128,'invalid control: v6 baseline does not match32')
    budget=first['kl_1'] in (512,2048);anchor=first['kl_1'] is None and first['kl_0'] is not None
    return {'complete':True,'classification':'budget' if budget else 'anchoring' if anchor else 'neither','first_full_match_step':first,'budget_evidence':budget,'anchoring_evidence':anchor,'automatic_promotion':False,'training_data_only':True}

class Audit:
    def __init__(self):
        self.run=ROOT/'runs'/NAME;self.progress=ROOT/'progress'/NAME;self.inputs={};self.observations={}
        self.output=self.progress/'independent-final-audit.json';self.saved={}
        if self.output.exists():self.saved=json.loads(self.output.read_text())['chronology']['original_file_observations']
    def path(self,p):return Path(p) if Path(p).is_absolute() else ROOT/p
    def track(self,p):
        p=self.path(p);h=digest(p);self.inputs[str(p.relative_to(ROOT))]=h;return h
    def read(self,p):p=self.path(p);self.track(p);return json.loads(p.read_text())
    def load(self,p):p=self.path(p);self.track(p);return torch.load(p,map_location='cpu',weights_only=True)
    def time(self,p):
        p=self.path(p);name=str(p.relative_to(ROOT));h=self.track(p)
        if name in self.saved:r=self.saved[name];need(r['sha256']==h,'saved chronology hash differs')
        else:
            need(not self.output.exists(),'restored chronology lacks original observation');r={'sha256':h,'mtime_ns':p.stat().st_mtime_ns,'observed_at':datetime.now(timezone.utc).isoformat()}
        self.observations[name]=r;return r['mtime_ns']/1e9
    def identity(self):
        self.ident=self.read(self.progress/'identity.json');need(fp(self.ident)==IH,'frozen experiment identity differs')
        self.config=self.read(self.progress/'config.json');c=self.config;i=self.ident
        need(c==i['config'] and c['experiment_id']==NAME and self.track('configs/'+NAME+'.json')==i['config_file_sha256'],'config differs')
        for p,h in i['inputs'].items():need(self.track(p)==h,'pinned dependency changed: '+p)
        for p,h in i['sources'].items():need(self.track(p)==h and self.track(self.progress/'source'/p)==h,'executed source changed: '+p)
        need(i['runtime']=={'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__),'torch_threads':1,'torch_interop_threads':1,'device':'cpu'},'numerical runtime differs')
        need(i['training_only'] is True and i['validation_data_read'] is False and i['benchmark_data_read'] is False and i['engine_games']==0 and i['promotion_allowed'] is False,'identity scope differs')
        need(c['updates']==2048 and c['snapshots']==list(STEPS) and c['checkpoint_interval']==16 and c['total_wall_budget_seconds']==900 and c['workers']==c['torch_threads']==1,'fixed schedule/budget differs')
        self.settings={**c['fit'],'updates':2048,'anchor_kl_weight':1}
        need(self.settings=={'updates':2048,'batch_size':32,'anchor_batch_size':32,'learning_rate':.0003,'weight_decay':.0001,'betas':[.9,.999],'eps':1e-8,'seed':203101,'target_weight':.75,'anchor_kl_weight':1},'optimizer/target protocol differs')
        self.pre=self.read(self.progress/'prelaunch-observation.json');self.post=self.read(self.progress/'quiescence-observation.json')
        need(self.pre['identity_sha256']==IH and self.pre['actors']==[] and self.pre['locks_exclusively_acquired']==list(LOCKS) and self.pre['stops']==dict.fromkeys(STOPS,False) and self.pre['sgd_launched'] is False,'launch observation differs')
        need(self.post['actors']==[] and self.post['all_five_locks_acquired'] is True and self.post['stops']==dict.fromkeys(STOPS,False),'quiescence observation differs')
    def dataset(self):
        c=self.config;self.meta=self.read(c['dataset_metadata']['path']);m=self.meta;self.track(c['dataset']['path'])
        need(m['complete'] is True and m['source']=='own_selfplay' and m['root_count']==32 and m['anchor_count']==512 and m['benchmark_data_used'] is False and m['validation_data_used'] is False,'training dataset provenance differs')
        with np.load(self.path(c['dataset']['path']),allow_pickle=False) as z:self.data={k:z[k] for k in z.files}
        d=self.data;shapes={'root_hidden':(32,128),'root_prior':(32,40),'root_mask':(32,40),'root_strata':(32,3),'terminal_actions':(32,),'puct_actions':(32,),'anchor_hidden':(512,128),'anchor_prior':(512,40),'anchor_mask':(512,40),'anchor_game':(512,),'anchor_player':(512,)}
        need(set(d)==set(shapes),'unexpected dataset tensors')
        for k,shape in shapes.items():
            dtype=np.float32 if k.endswith(('hidden','prior')) else np.uint8 if k.endswith('mask') else np.int64
            need(d[k].shape==shape and d[k].dtype==dtype and np.isfinite(d[k]).all(),'dataset dtype/shape/finite differs: '+k)
        for prefix in ('root','anchor'):
            mask=d[prefix+'_mask'];need(np.isin(mask,[0,1]).all() and mask.any(-1).all(),'invalid legal mask');d[prefix+'_prior']=normal(d[prefix+'_prior'],mask)
        need(d['root_mask'][np.arange(32),d['terminal_actions']].all(),'illegal terminal target')
        need(np.array_equal(d['puct_actions'],d['root_prior'].argmax(-1)) and (d['terminal_actions']!=d['root_prior'].argmax(-1)).sum()==9,'changed9/retained23 target definition differs')
        self.gradient=sorted(set(m['root_input_hashes']+m['anchor_input_hashes']));need(self.gradient==m['gradient_input_hashes'] and not set(self.gradient)&set(m['validation_input_hashes']),'opaque input-overlap guard differs')
        self.guard={'checked':True,'gradient_input_count':len(self.gradient),'validation_input_count':16,'intersection_count':0,'set_sha256':fp(self.gradient)}
        groups=[np.flatnonzero((d['root_strata']==(a,b,t)).all(1)) for a in range(4) for b in (0,1) for t in (0,1)];need(all(len(g)==2 for g in groups),'root balance differs')
        games=sorted(set(d['anchor_game'].tolist()));need(games==list(range(len(m['training_games']))),'anchor source-game coverage differs')
        anchors={g:[np.flatnonzero((d['anchor_game']==g)&(d['anchor_player']==t)) for t in (0,1)] for g in games};need(all(len(v) for pair in anchors.values() for v in pair),'missing anchor game/team')
        self.anchor_weights=np.empty(512,dtype=np.float32)
        for g,pair in anchors.items():
            for v in pair:self.anchor_weights[v]=1/(len(games)*2*len(v))
        rng=np.random.default_rng(203101);self.batches={k:np.empty((2048,32),dtype=np.int64) for k in ('root','anchor')}
        for step in range(2048):
            for col in range(32):
                group=groups[int(rng.integers(16))];self.batches['root'][step,col]=group[int(rng.integers(len(group)))]
            for col in range(32):
                group=anchors[games[int(rng.integers(len(games)))]][int(rng.integers(2))];self.batches['anchor'][step,col]=group[int(rng.integers(len(group)))]
            if step==127:self.prefix_rng=json.loads(json.dumps(rng.bit_generator.state))
        self.rng=rng.bit_generator.state;h=hashlib.sha256()
        for k,v in self.batches.items():h.update(json.dumps([k,list(v.shape)],separators=(',',':')).encode());h.update(v.astype('<i8').tobytes())
        h.update(fp(self.rng).encode());self.batchsha=h.hexdigest()
        self.track(c['reference_batches']['path'])
        with np.load(self.path(c['reference_batches']['path']),allow_pickle=False) as saved:
            need(set(saved.files)=={'root','anchor'} and all(saved[k].dtype==np.int64 and np.array_equal(saved[k],self.batches[k][:128]) for k in saved.files),'extended stream changed v6 prefix')
        self.base=self.load(c['checkpoint']['path']);self.reference=self.load(c['reference_terminal_checkpoint']['path']);self.base_state=self.base['model_state_dict']
        self.parent={k:v for k,v in self.base.items() if k not in ('model_state_dict','optimizer_state_dict','torch_rng','numpy_rng')}
        h=hashlib.sha256()
        for k,v in sorted(self.base_state.items()):
            if k in POLICY:continue
            header=json.dumps([k,str(v.dtype),list(v.shape)],separators=(',',':')).encode();h.update(len(header).to_bytes(8,'little'));h.update(header);h.update(tensorbytes(v))
        self.frozen=h.hexdigest();self.torch_rng=torch.Generator().manual_seed(203101).get_state()
        need(sum(self.base_state[k].numel() for k in POLICY)==5160,'policy parameter count differs')
        self.fit_identity={'identity_sha256':IH,'input_hashes':{k:c[k]['sha256'] for k in ('checkpoint','dataset','dataset_metadata','reference_terminal_checkpoint','reference_batches','v6_identity')},'batch_sha256':self.batchsha,'settings':self.settings,'fitter_sha256':self.track('alpha0/constraint_v7_fit.py'),'common_sha256':self.track('alpha0/constraint_v7_common.py'),'v6_fitter_sha256':self.track('alpha0/terminal_v6_fit.py'),'torch_version':str(torch.__version__),'numpy_version':np.__version__}
    def state(self,s,arm,step):
        need(s['arm']==arm and type(s['step'])is int and s['step']==step and s['coefficient']==ARMS[arm],'generation arm/step differs')
        need(s['batch_sha256']==self.batchsha and s['frozen_state_sha256']==self.frozen and s['fit_config_sha256']==fp(self.settings),'generation provenance differs')
        need(same(s['torch_rng'],self.torch_rng) and same(s['numpy_rng'],self.rng),'generation RNG differs')
        p=s['policy_state_dict'];need(set(p)=={'weight','bias'},'policy state inventory differs')
        for k,v in p.items():need(v.dtype==torch.float32 and v.shape==self.base_state['policy.'+k].shape and bool(torch.isfinite(v).all()),'policy tensor invalid')
        hist=s['history'];need(len(hist)==step and [r['step'] for r in hist]==list(range(1,step+1)),'generation history steps differ')
        for r in hist:need(all(number(r[k]) for k in ('loss','target_ce','anchor_kl')) and abs(r['loss']-(r['target_ce']+ARMS[arm]*r['anchor_kl']))<2e-6,'history objective differs')
        opt=s['optimizer_state_dict'];need(same(opt['param_groups'],self.reference['optimizer_state_dict']['param_groups']) and set(opt['state'])=={0,1},'AdamW settings/parameter order differs')
        for index,key in enumerate(('weight','bias')):
            row=opt['state'][index];need(set(row)=={'step','exp_avg','exp_avg_sq'} and row['step'].dtype==torch.float32 and row['step'].shape==torch.Size([]) and float(row['step'])==step,'optimizer step differs')
            for k in ('exp_avg','exp_avg_sq'):need(row[k].dtype==torch.float32 and row[k].shape==p[key].shape and bool(torch.isfinite(row[k]).all()),'optimizer moment invalid')
            need(bool((row['exp_avg_sq']>=0).all()),'negative optimizer second moment')
    def metrics(self,state,coefficient):
        d=self.data;rh=torch.as_tensor(d['root_hidden']);ah=torch.as_tensor(d['anchor_hidden']);rm=torch.as_tensor(d['root_mask'],dtype=torch.bool);am=torch.as_tensor(d['anchor_mask'],dtype=torch.bool)
        w=state['policy.weight'].detach().clone().requires_grad_(True);b=state['policy.bias'].detach().clone().requires_grad_(True)
        rp=torch.as_tensor(normal(d['root_prior'],d['root_mask']));ap=torch.as_tensor(normal(d['anchor_prior'],d['anchor_mask']));q=.25*rp+.75*TF.one_hot(torch.as_tensor(d['terminal_actions']),40).float()
        rlog=TF.log_softmax(TF.linear(rh,w,b).masked_fill(~rm,torch.finfo(torch.float32).min),-1);alog=TF.log_softmax(TF.linear(ah,w,b).masked_fill(~am,torch.finfo(torch.float32).min),-1)
        ce=-(q*rlog).sum(-1).mean();perkl=(torch.xlogy(ap,ap)-ap*alog).sum(-1);plainkl=perkl.mean();weightedkl=(perkl*torch.as_tensor(self.anchor_weights)).sum()
        cg=torch.cat([x.reshape(-1) for x in torch.autograd.grad(ce,(w,b),retain_graph=True)]);kg=torch.cat([x.reshape(-1) for x in torch.autograd.grad(weightedkl,(w,b))]);denom=float(cg.norm()*kg.norm())
        chosen=rlog.detach().argmax(-1).numpy();anchorchosen=alog.detach().argmax(-1).numpy();targets=d['terminal_actions'];changed=targets!=d['root_prior'].argmax(-1)
        delta=torch.cat([(state['policy.weight']-self.base_state['policy.weight']).reshape(-1),(state['policy.bias']-self.base_state['policy.bias']).reshape(-1)])
        return {'loss':float((ce+coefficient*plainkl).detach()),'target_ce':float(ce.detach()),'anchor_kl':float(plainkl.detach()),'root_count':32,'root_selected_action_matches':int((chosen==targets).sum()),'root_target_agreement':float(np.mean(chosen==targets)),'discordant_target_count':9,'discordant_target_matches':int(((chosen==targets)&changed).sum()),'retained_target_count':23,'retained_target_matches':int(((chosen==targets)&~changed).sum()),'root_action_changed_fraction':float(np.mean(chosen!=d['root_prior'].argmax(-1))),'anchor_action_changes':int((anchorchosen!=d['anchor_prior'].argmax(-1)).sum()),'anchor_action_changed_fraction':float(np.mean(anchorchosen!=d['anchor_prior'].argmax(-1))),'expected_anchor_kl':float(weightedkl.detach()),'expected_objective':float(ce.detach())+coefficient*float(weightedkl.detach()),'root_actions':chosen.tolist(),'parameter_displacement_l2':float(delta.norm()),'target_ce_gradient_norm':float(cg.norm()),'balanced_anchor_kl_gradient_norm':float(kg.norm()),'expected_total_gradient_norm':float((cg+coefficient*kg).norm()),'ce_kl_gradient_cosine':float(torch.dot(cg,kg))/denom if denom else None}
    def artifacts(self):
        directory=self.run/'fit';self.manifest=self.read(directory/'manifest.json');m=self.manifest;fitsha=fp(self.fit_identity)
        need(m['fit_identity']==self.fit_identity and m['fit_identity_sha256']==fitsha and m['identity_sha256']==IH and m['complete'] is True,'fit manifest identity differs')
        top={'schema_version':1,'experiment_id':NAME,'updates_per_arm':2048,'snapshots':list(STEPS),'trainable_parameters':5160,'training_data_only':True,'validation_data_used':False,'benchmark_data_used':False,'automatic_promotion':False,'gradient_input_overlap':self.guard,'frozen_state_sha256':self.frozen}
        need(all(same(m[k],v) for k,v in top.items()),'manifest protocol/scope differs')
        required={'identity.json','batches.npz','batch-rng.json','parent-metadata.json','reproduction-128.json'}
        for arm in ARMS:
            for step in range(16,2049,16):required.update(f'{arm}/states/step-{step:04d}.{ext}' for ext in ('pt','json'))
            for step in STEPS:required.update(f'{arm}/checkpoint-{step:04d}.{ext}' for ext in ('pt','json'))
        actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and not p.name.startswith('.') and p.name!='manifest.json'}
        need(len(required)==529 and actual==required==set(m['files']),'529-file inventory differs')
        for name,h in m['files'].items():need(self.track(directory/name)==h,'recovery/snapshot file hash differs: '+name)
        need(self.read(directory/'identity.json')==self.fit_identity and self.read(directory/'batch-rng.json')==self.rng and same(self.read(directory/'parent-metadata.json'),self.parent),'fit sidecar identity/RNG/ancestor differs')
        with np.load(directory/'batches.npz',allow_pickle=False) as z:need(set(z.files)=={'root','anchor'} and all(z[k].dtype==np.int64 and np.array_equal(z[k],self.batches[k]) for k in z.files),'saved shared2048 batches differ')
        need(m['batch_schedule']=={'path':str((directory/'batches.npz').relative_to(ROOT)),'sha256':m['files']['batches.npz'],'batch_sha256':self.batchsha,'root_shape':[2048,32],'anchor_shape':[2048,32],'seed':203101},'batch schedule declaration differs')
        need(set(m['checkpoints'])==set(ARMS),'checkpoint arm inventory differs')
        self.results={};self.generations={};self.payloads={};self.history={};self.generation_times={}
        hidden=torch.as_tensor(np.concatenate((self.data['root_hidden'],self.data['anchor_hidden'])))
        def critic(s):return TF.linear(TF.silu(TF.linear(hidden,s['value.0.weight'],s['value.0.bias'])),s['value.2.weight'],s['value.2.bias'])
        with torch.no_grad():basevalue=critic(self.base_state)
        for arm,coef in ARMS.items():
            previous=[];saved={};times=[]
            for step in range(16,2049,16):
                p=directory/arm/'states'/f'step-{step:04d}.pt';row=self.read(p.with_suffix('.json'));s=self.load(p)
                need(row=={'fit_identity_sha256':fitsha,'step':step,'sha256':self.track(p)},'generation sidecar differs');self.state(s,arm,step)
                need(same(s['history'][:len(previous)],previous),'recovery history prefix changed');previous=s['history'];times.append(self.time(p))
                if step in STEPS:saved[step]=s
            need(all(a<=b+.002 for a,b in zip(times,times[1:])),'generation times reverse update order')
            self.generation_times[arm]=times;self.generations[arm]=saved;self.history[arm]=previous;self.results[arm]={};self.payloads[arm]={}
            need(set(m['checkpoints'][arm])==set(map(str,STEPS)),'fixed snapshot inventory differs')
            for step in STEPS:
                p=directory/arm/f'checkpoint-{step:04d}.pt';row=m['checkpoints'][arm][str(step)];payload=self.load(p);state=payload['model_state_dict'];generation=saved[step]
                need(row==self.read(p.with_suffix('.json')) and row['path']==str(p.relative_to(ROOT)) and row['sha256']==self.track(p),'snapshot path/sidecar/hash differs')
                for k,v in {'fit_identity_sha256':fitsha,'arm':arm,'updates':step,'anchor_kl_weight':coef,'trainable_parameters':5160,'frozen_tensors_equal':True,'value_logits_equal':True}.items():need(same(row.get(k),v),'snapshot declaration differs: '+k)
                for k,v in {'experiment_id':NAME,'arm':arm,'anchor_kl_weight':coef,'updates':step,'gradient_updates':step,'trainable_parameters':5160,'policy_head_only':True,'benchmark_data_used':False,'validation_data_used':False,'automatic_promotion':False,'identity_sha256':IH,'fit_identity_sha256':fitsha,'frozen_state_sha256':self.frozen,'batch_sha256':self.batchsha,'gradient_input_overlap':self.guard,'positions_total':544}.items():need(same(payload.get(k),v),'candidate declaration differs: '+k)
                need(payload['parent_checkpoint_sha256']==self.config['checkpoint']['sha256'] and payload['dataset_sha256']==self.config['dataset']['sha256'] and payload['dataset_metadata_sha256']==self.config['dataset_metadata']['sha256'] and payload['reference_v6_checkpoint_sha256']==self.config['reference_terminal_checkpoint']['sha256'],'candidate lineage differs')
                lineage={**self.config,'initialization':{'artifact':self.config['checkpoint']['path'],'artifact_sha256':self.config['checkpoint']['sha256'],'method':'fresh cp2 frozen features; training-only KL/budget diagnostic'}}
                need(same(payload['config'],lineage) and same(payload['parent_checkpoint_metadata'],self.parent) and payload['training_seeds']==self.meta['training_seeds'] and payload['games_total']==len(self.meta['training_seeds']),'candidate ancestry/config differs')
                need(payload['training_maps']==sorted(set(self.meta['training_maps'])|set(self.base.get('training_maps',[]))),'candidate map ancestry differs')
                need(payload['network_config']==self.base['network_config'] and payload['model_version']==self.base['model_version'] and payload['feature_version']==self.base['feature_version'] and payload['action_version']==self.base['action_version'],'candidate model schema differs')
                need(set(state)==set(self.base_state),'candidate tensor inventory differs')
                changed=[]
                for k,v in self.base_state.items():
                    need(state[k].dtype==v.dtype and state[k].shape==v.shape and bool(torch.isfinite(state[k]).all()),'candidate tensor invalid')
                    if not same(state[k],v):changed.append(k)
                need(set(changed)<=POLICY,'non-policy tensor changed')
                need(same(state['policy.weight'],generation['policy_state_dict']['weight']) and same(state['policy.bias'],generation['policy_state_dict']['bias']),'snapshot differs from recovery generation')
                for pkey,gkey in (('optimizer_state_dict','optimizer_state_dict'),('torch_rng','torch_rng'),('numpy_rng','numpy_rng'),('fit_history','history')):need(same(payload[pkey],generation[gkey]),'snapshot recovery state differs: '+pkey)
                need(set(payload['batch_indices'])=={'root','anchor'} and all(payload['batch_indices'][k].dtype==torch.int64 and np.array_equal(payload['batch_indices'][k].numpy(),self.batches[k]) for k in self.batches),'candidate shared schedule differs')
                with torch.no_grad():need(same(critic(state),basevalue),'frozen critic outputs changed')
                measured=self.metrics(state,coef);need(set(measured)==set(payload['fit_metrics']) and same(payload['fit_metrics'],row['metrics']),'metric inventory/sidecar differs')
                for k,v in measured.items():
                    actual=payload['fit_metrics'][k]
                    if type(v)is float:need(number(actual) and abs(actual-v)<=2e-6,'recomputed metric differs: '+k)
                    else:need(same(actual,v),'recomputed discrete metric differs: '+k)
                self.results[arm][str(step)]={'checkpoint_sha256':self.track(p),'changed_tensor_keys':changed,'frozen_tensor_bytes_equal':True,'critic_544_outputs_equal':True,'recovery_state_equal':True,'metrics':measured};self.payloads[arm][step]=payload;self.time(p)
        control=self.payloads['kl_1'][128];ref=self.reference
        for key in ('model_state_dict','optimizer_state_dict','torch_rng','fit_history'):need(same(control[key],ref[key]),'mandatory KL1@128 v6 byte reproduction differs: '+key)
        need(same(self.prefix_rng,ref['numpy_rng']),'v6 NumPy prefix state differs')
        for k in self.batches:need(np.array_equal(control['batch_indices'][k][:128].numpy(),ref['batch_indices'][k].numpy()),'v6 saved batch prefix differs')
        for k in ('target_ce','anchor_kl'):need(self.history['kl_0'][0][k]==ref['fit_history'][0][k],'KL0 first pre-update objective differs from fresh cp2 control')
        proof={'passed':True,'updates':128,'model_bytes_equal':True,'optimizer_state_equal':True,'torch_rng_equal':True,'prefix_numpy_rng_equal':True,'batch_prefix_equal':True,'loss_history_equal':True,'extended_schedule_rng_is_distinct_metadata':True,'reference_checkpoint_sha256':self.config['reference_terminal_checkpoint']['sha256'],'fit_identity_sha256':fitsha}
        need(self.read(directory/'reproduction-128.json')==proof==m['reproduction'],'reproduction proof declaration differs');self.proof=proof;self.time(directory/'reproduction-128.json');self.time(directory/'manifest.json')
        counts={arm:{step:r['metrics']['root_selected_action_matches'] for step,r in rows.items()} for arm,rows in self.results.items()};self.classification=classify(counts)
        need(m['classification']==self.classification and self.read(self.progress/'fit-summary.json')==m,'published fit classification/summary differs')
        verified=self.read(self.progress/'final-verification.json');need(verified=={'identity_sha256':IH,'complete':True,'integrity_verified':True,'training_only':True,'promotion_allowed':False,'fit_report_sha256':fp(m)},'supervisor final verification differs')
        summary=self.read(self.progress/'summary.json');decision=summary['decision']
        need(decision['classification']==self.classification['classification'] and decision['budget_evidence']==self.classification['budget_evidence'] and decision['anchoring_evidence']==self.classification['anchoring_evidence'] and decision['first_full_match_step']==self.classification['first_full_match_step'],'public decision differs')
        need(decision['complete'] is True and decision['integrity_verified'] is True and decision['reproduction_verified'] is True and decision['all_map_victory'] is False and decision['automatic_promotion'] is False and decision['training_data_only'] is True and decision['snapshots_observed']==decision['snapshots_required']==6,'public decision scope/completeness differs')
        need(summary['identity_sha256']==IH and summary['fit_identity_sha256']==fitsha and summary['reproduction']==proof and summary['root_baseline_actions']==self.data['root_prior'].argmax(-1).tolist() and summary['root_targets']==self.data['terminal_actions'].tolist(),'public provenance/root labels differ')
        need(summary['training_only'] is True and summary['validation_data_read'] is False and summary['benchmark_data_read'] is False and summary['engine_games']==0 and summary['automatic_promotion'] is False,'public report scope differs')
        for arm,coef in ARMS.items():
            public=summary['arms'][arm];need(public['coefficient']==coef and public['completed_updates']==2048,'public arm declaration differs')
            expected_history=[{**r,'weighted_anchor_kl':coef*r['anchor_kl']} for r in self.history[arm]]
            need(public['history']==expected_history,'public loss curves differ from saved training history')
            for step in STEPS:
                row=public['snapshots'][str(step)];original=self.payloads[arm][step]['fit_metrics'];metrics={**original,'weighted_anchor_kl_contribution':coef*original['expected_anchor_kl']}
                need(row=={'step':step,'checkpoint_sha256':self.results[arm][str(step)]['checkpoint_sha256'],'metrics':metrics},'public snapshot graph data differs')

    def lifecycle(self):
        directory=self.progress/'invocations';files=list(directory.glob('*.json'));starts={};ends={}
        for p in files:
            need(re.fullmatch(r'\d{3,}(?:-start)?\.json',p.name) is not None,'unknown invocation file');r=self.read(p);n=int(p.name.split('-')[0].split('.')[0]);isstart=p.name.endswith('-start.json');bucket=starts if isstart else ends
            need(n not in bucket and r['invocation']==n and type(r['invocation'])is int and r['identity_sha256']==IH and r['experiment_id']==NAME,'invocation identity differs');bucket[n]=r
        need(starts and set(ends)<=set(starts),'orphan final invocation');charged=0.;intervals=[];records=[]
        for n,start in sorted(starts.items()):
            reserved=start['reserved_wall_seconds'];need(number(reserved) and 0<reserved<=900-charged+1e-6 and abs(start['wall_budget_previously_consumed']-charged)<1e-6,'budget reservation reset/overflow')
            descriptor=self.read(self.run/'workers'/f'invocation-{n:03d}.json');need(descriptor['identity_sha256']==IH and descriptor['parent_pid']==start['pid'] and descriptor['invocation']==n and descriptor['started_at']==start['started_at'],'worker descriptor lineage differs')
            if n not in ends:charged+=reserved;records.append({'invocation':n,'unclosed':True,'charged_wall_seconds':reserved});continue
            end=ends[n];need(all(end[k]==v for k,v in start.items()),'invocation start/end differs');duration=stamp(end['ended_at'])-stamp(start['started_at']);cost=end['charged_wall_seconds']
            need(number(cost) and 0<=cost<=reserved+10 and duration>=0 and abs(duration-cost)<=2,'invocation wall charge differs');need(end['worker_pid'] is None and end['training_only'] is True and end['engine_games']==0 and end['promotion_allowed'] is False,'invocation scope/quiescence differs')
            need(number(end['worker_tree_cpu_seconds']) and end['worker_tree_cpu_seconds']>=0 and number(end['supervisor_cpu_seconds']) and end['supervisor_cpu_seconds']>=0,'CPU accounting invalid')
            charged+=cost;intervals.append((stamp(start['started_at']),stamp(end['ended_at'])));records.append(end)
        need(charged<=900,'total charged wall budget exceeded900');need(all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])),'supervisor invocations overlapped')
        self.status=self.read(self.run/'status.json');need(self.status==ends[max(ends)] and self.status['phase']=='complete_review_required' and self.status['integrity_verified'] is True and self.status['reason'] is None,'latest lifecycle not complete and verified')
        need(set(starts)==set(ends),'unclosed invocation remains in completed run')
        earliest=min(x[0] for x in intervals);latest=max(x[1] for x in intervals)
        need(stamp(self.pre['observed_at'])<=earliest and stamp(self.post['observed_at'])>=latest,'operator observations do not bracket run')
        for name,row in self.observations.items():
            moment=row['mtime_ns']/1e9;need(any(a-.002<=moment<=b+.002 for a,b in intervals),'fit artifact outside supervised budget interval: '+name)
        need(self.generation_times['kl_1'][-1]<=self.generation_times['kl_0'][0]+.002,'one-worker arm order differs')
        proof_time=self.observations[str((self.run/'fit/reproduction-128.json').relative_to(ROOT))]['mtime_ns']/1e9
        need(self.generation_times['kl_1'][7]<=proof_time+.002 and proof_time<=self.generation_times['kl_1'][8]+.002,'first later KL1 generation precedes mandatory128 reproduction')
        control_time=self.observations[str((self.run/'fit/kl_1/checkpoint-0128.pt').relative_to(ROOT))]['mtime_ns']/1e9
        need(proof_time<=control_time+.002,'control checkpoint published before reproduction proof')
        self.lifecycle_report={'invocations':records,'total_charged_wall_seconds':charged,'remaining_wall_seconds':900-charged,'worker_tree_cpu_seconds':sum(r['worker_tree_cpu_seconds'] for r in ends.values()),'supervisor_cpu_seconds':sum(r['supervisor_cpu_seconds'] for r in ends.values()),'prelaunch_observation_sha256':self.track(self.progress/'prelaunch-observation.json'),'quiescence_observation_sha256':self.track(self.progress/'quiescence-observation.json'),'original_file_observations':self.observations,'limitations':'Observed original local mtimes are hash-bound provenance evidence, not a trusted-clock attestation; restored audits reuse these observations. CPU totals are the supervisor-reported reaped-worker accounting, not independent historical process sampling.'}
    def run_all(self):
        self.identity();self.dataset();self.artifacts();self.lifecycle()
        return {'schema_version':1,'experiment_id':NAME,'created_at':datetime.now(timezone.utc).isoformat(),'audit_kind':'independent-training-only-state-metrics-recovery-and-budget-audit','identity_sha256':IH,'integrity_verified':True,'complete':True,'new_optimizer_steps':0,'engine_games_run_by_auditor':0,'validation_examples_actions_or_outcomes_parsed':False,'benchmark_examples_or_outcomes_parsed':False,'fixed_training_roots':32,'fixed_training_anchors':512,'trainable_parameters':5160,'verified_recovery_generations':256,'verified_artifact_files':529,'verified_snapshots':6,'same_2048_batches_for_both_arms':True,'batch_sha256':self.batchsha,'reproduction_128':self.proof,'snapshots':self.results,'classification':self.classification,'chronology':self.lifecycle_report,'all_map_victory':False,'automatic_promotion':False,'limitations':['This audit recomputes diagnostic values/derivatives and checks saved state, ancestry and history; it takes no optimizer steps. Exact reproduction is demonstrated against the frozen v6 result at128. Later2048-update execution relies on the pinned source, consistent recovery generations and supervised journals rather than an additional SGD replay.','These are training-set mechanism findings, not evidence of playing strength or a guarantee that the selected terminal labels are correct.'],'input_sha256':dict(sorted(self.inputs.items()))}

def publish(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);fd,name=tempfile.mkstemp(prefix='.'+path.name+'-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.link(name,path)
        descriptor=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:Path(name).unlink(missing_ok=True)

def main():
    global ROOT,torch,TF
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--final',action='store_true');parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args();ROOT=args.root.resolve();source=Path(__file__).read_bytes()
    with ExitStack() as stack:
        held=[]
        for name in LOCKS:
            p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);f=stack.enter_context(p.open('a+'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);held.append({'path':name,'inode':p.stat().st_ino,'device':p.stat().st_dev})
        need(not actors(),'active actor/orphan blocks final audit')
        import torch
        from torch.nn import functional as TF
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        audit=Audit();need(not args.final or not audit.output.exists(),'immutable final audit already exists');report=audit.run_all()
        for path,h in audit.inputs.items():need(digest(ROOT/path)==h,'input changed during audit: '+path)
        need(not actors(),'actor appeared during audit');need(Path(__file__).read_bytes()==source,'auditor source changed')
        for row in held:need((ROOT/row['path']).stat().st_ino==row['inode'] and (ROOT/row['path']).stat().st_dev==row['device'],'held lock replaced')
        report['auditor_source_sha256']=hashlib.sha256(source).hexdigest();report['quiescence']={'locks_held_through_publication':held,'matching_actors':[],'unreadable_matching_cwd_blocks':True,'pid':os.getpid(),'observed_stop_states':{p:(ROOT/p).exists() for p in STOPS}}
        if args.final:
            archive=audit.progress/'source/independent_final_audit.py'
            if archive.exists():need(archive.read_bytes()==source,'archived auditor differs')
            else:publish(archive,source)
            report['auditor_source_path']=str(archive.relative_to(ROOT));publish(audit.output,(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
        print(json.dumps({'published':args.final,'path':str(audit.output),'sha256':digest(audit.output) if args.final else None,'source_sha256':report['auditor_source_sha256'],'classification':report['classification'],'snapshots':{arm:{step:row['metrics']['root_selected_action_matches'] for step,row in rows.items()} for arm,rows in report['snapshots'].items()},'charged_wall_seconds':report['chronology']['total_charged_wall_seconds'],'new_optimizer_steps':0},indent=2))

if __name__=='__main__':main()
