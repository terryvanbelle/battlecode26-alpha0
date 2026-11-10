#!/usr/bin/env python3
"""Independent v6 evidence audit: no engine or experiment-module imports.

Use .venv/bin/python with this script. Run --final only after the supervising
agent confirms quiescence. All four
exclusion locks remain held through verification and exclusive publication.
Without --final, verify a quiescent existing run without publishing a decision.
"""
from contextlib import ExitStack
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from importlib import metadata as packages
import argparse, fcntl, hashlib, json, math, os, re, tempfile
from pathlib import Path
import numpy as np

ROOT=Path('/workspace/battlecode26-alpha0')
RUN_NAME='runs/terminal-target-head-v6'
PROGRESS_NAME='progress/terminal-target-head-v6'
LOCKS=('runs/campaign/campaign.lock','runs/warmstart-v3/experiment.lock','runs/terminal-target-head-v5/experiment.lock',RUN_NAME+'/experiment.lock')
ARMS=('cp2','terminal_head','puct_head')
CP2='20d9e813349010630940e1cdfbaee5206490c7d323997688d26f47508bbeee7c'
CHRONOLOGY_OBSERVATION_SHA='b89fe4c82d48c6c2ee38240559bece5045dbf101c061642e4de34c0cca4e0254'
POLICY_KEYS={'policy.weight','policy.bias'}
FIELDS={'hidden':(128,),'prior':(40,),'mask':(40,),'local':(33,17,17),'global_board':(33,60,60),'scalars':(64,)}

def require(ok,message):
    if not ok: raise ValueError(message)

def fp(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def stamp(value):
    parsed=datetime.fromisoformat(value)
    require(parsed.tzinfo is not None,'timestamp lacks timezone')
    return parsed.timestamp()

def number(value):
    return type(value) in (int,float) and math.isfinite(value)

def check_hash(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None

def input_hash(values):
    fields=[]; chunks=[]
    for name,value in zip(('local','global_board','scalars','legal_mask'),values):
        array=np.array(value,dtype='u1' if name=='legal_mask' else '<f4',order='C',copy=True)
        require(np.isfinite(array).all(),'nonfinite model input')
        if name!='legal_mask': array[array==0]=0
        fields.append({'name':name,'shape':list(array.shape),'dtype':'|u1' if name=='legal_mask' else '<f4'})
        chunks.append(array.tobytes(order='C'))
    header=json.dumps({'schema':'gradient-input-v1','feature_version':2,'action_version':'macro_v2','fields':fields},sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(len(header).to_bytes(8,'little')+header+b''.join(chunks)).hexdigest()

def normalized(prior,mask):
    result=np.where(mask,prior,0).astype(np.float64)
    require(np.isfinite(result).all() and (result>=0).all() and (result.sum(-1)>0).all(),'invalid prior')
    return np.ascontiguousarray(result/result.sum(-1,keepdims=True),dtype=np.float32)

def actor_state():
    found=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name)==os.getpid():continue
        try:args=path.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')[:-1]
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        matching=any(a in {'alpha0.terminal_v5','alpha0.terminal_v6','alpha0.EngineBridge','alpha0.campaign','alpha0.learn_v2','alpha0.root_ranking_v4','alpha0.evaluate','alpha0.victory'} or Path(a).name=='run_warmstart_experiment.py' for a in args)
        if not matching:continue
        try:
            cwd=Path(os.readlink(path.joinpath('cwd'))).resolve()
            if cwd!=ROOT:continue
            reason='matching actor in this repository'
        except (PermissionError,OSError) as error:
            cwd=None;reason='matching actor with unreadable cwd: '+type(error).__name__
        found.append({'pid':int(path.name),'command':args,'cwd':str(cwd) if cwd is not None else None,'reason':reason})
    return found

class Audit:
    def __init__(self):
        self.run=ROOT/RUN_NAME;self.progress=ROOT/PROGRESS_NAME
        self.output=self.progress/'independent-final-audit.json'
        self.execution_root=ROOT
        self.tracked={};self.observations={};self.saved={}
        observation_path=self.progress/'chronology-observation.json'
        require(self.track(observation_path)==CHRONOLOGY_OBSERVATION_SHA,'early chronology observation differs')
        self.early_observation=self.read(observation_path);self.saved=dict(self.early_observation['observations']);self.observations=dict(self.saved)
        if self.output.exists():
            archived=json.loads(self.output.read_text());saved=archived.get('chronology',{}).get('observations',{})
            require(all(saved.get(k)==v for k,v in self.saved.items()),'final chronology differs from early observed evidence');self.saved=saved;self.execution_root=Path(archived.get('execution_root',str(ROOT)))
        self.sources={};self.cache_meta={};self.cached={};self.source_rows=[]
        self.cells={};self.cell_rows=[];self.starts=[];self.interrupted_cell_attempts=[];self.failed_attempts=[]
        self.rootrows={};self.selected={};self.fit=None;self.validation=None;self.benchmarks=None

    def path(self,value):
        path=Path(value);return path if path.is_absolute() else ROOT/path

    def track(self,path):
        path=self.path(path);digest=sha(path)
        name=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        self.tracked[name]=digest;return digest

    def read(self,path):
        path=self.path(path);self.track(path);return json.loads(path.read_text())

    def original_time(self,path):
        path=self.path(path);name=str(path.relative_to(ROOT));digest=self.track(path)
        if name in self.saved:
            row=self.saved[name];require(row['sha256']==digest,'archived chronology hash differs: '+name)
        else:
            require(not self.output.exists(),'restored audit lacks original chronology observation: '+name)
            row={'sha256':digest,'mtime_ns':path.stat().st_mtime_ns,'observed_at':datetime.now(timezone.utc).isoformat()}
        self.observations[name]=row;return row['mtime_ns']/1e9

    def assert_identity(self,allow_active=False):
        self.identity=self.read(self.progress/'identity.json');self.config=self.read(self.progress/'config.json')
        cfg=self.config;i=self.identity;self.ih=fp(i)
        require(cfg==i['config'] and cfg['experiment_id']=='terminal-target-head-v6','wrong configuration')
        require(self.early_observation['experiment_identity_sha256']==self.ih and self.track(self.progress/'source/observe_chronology.py')==self.early_observation['auditor_source_sha256'],'early observation identity/source differs')
        for name,row in self.early_observation['observations'].items():require(self.track(name)==row['sha256'],'early-observed artifact changed: '+name)
        exact={'checkpoint_sha256':CP2,'maps':['tiny','DefaultSmall','closeup','knifefight'],'source_temperature':1.0,'temperature':.35,'training_roots':32,'validation_roots':16,'anchor_count':512,'anchor_seed':203151,'selection_seeds':[203003,203013],'validation_seeds':[203021,203043,203067,203091],'selection_cell_ceiling':1600,'workers':3,'torch_threads':1,'wall_budget_seconds':10800,'benchmark_data_used_for_training':False}
        require(all(cfg[k]==v for k,v in exact.items()),'preregistered protocol changed')
        require(cfg['fit']=={'updates':128,'batch_size':32,'anchor_batch_size':32,'learning_rate':.0003,'weight_decay':.0001,'betas':[.9,.999],'eps':1e-8,'seed':203101,'target_weight':.75,'anchor_kl_weight':1},'fit protocol changed')
        require(cfg['search']=={'simulations':32,'max_depth':12,'cpuct':1.5,'dirichlet_alpha':.3,'exploration_fraction':.25,'root_noise':True,'root_seed_base':203201},'PUCT protocol changed')
        require(self.track(cfg['checkpoint'])==CP2==i['checkpoint_sha256'],'cp2 bytes changed')
        require(self.track('configs/terminal-target-head-v6.json')==i['config_file_sha256'],'config bytes changed')
        for name,h in i['sources'].items():
            require(self.track(name)==h,'runtime source changed: '+name)
            require(self.track(self.progress/'source'/name)==h,'executed source snapshot changed: '+name)
        for name,h in i['runtime']['files'].items(): require(self.track(name)==h,'training runtime changed: '+name)
        classes={str(p.relative_to(ROOT)):self.track(p) for p in sorted((ROOT/'build/java').rglob('*.class'))}
        require(classes==i['runtime']['compiled_classes']==i['evaluation_runtime']['compiled_classes'],'compiled runtime changed')
        require(self.track(ROOT/'vendor/jdk/release')==i['runtime']['jdk_release_sha256']==i['evaluation_runtime']['jdk_release_sha256'],'JDK changed')
        for package in ('numpy','torch'): require(packages.version(package)==i['runtime'][package+'_version']==i['evaluation_runtime'][package+'_version'],'package changed: '+package)
        for name,h in i['evaluation_runtime']['python_sources'].items(): require(self.track(name)==h,'evaluation Python runtime changed')
        require(fp(i['evaluation_runtime']['python_sources'])==i['evaluation_runtime']['python_adapter_sha256'],'evaluation adapter fingerprint differs')
        require(fp(classes)==i['evaluation_runtime']['bridge_sha256'],'bridge fingerprint differs')
        maps=[{'name':p.stem,'sha256':self.track(p)} for p in sorted((ROOT/'maps').glob('*.map26'))]
        require(maps==i['maps'],'frozen map inventory changed')
        for name,key in [('configs/benchmarks.json','panel_sha256'),('configs/victory_gate.json','gate_sha256'),('progress/root-ranking-v4/independent-final-audit.json','v4_audit_sha256'),('progress/terminal-target-head-v5/independent-coverage-audit.json','v5_coverage_audit_sha256')]: require(self.track(name)==i[key],'prerequisite changed: '+name)
        before=self.read('progress/terminal-target-head-v5/config.json')
        require(all(cfg[k]==v for k,v in before.items() if k!='experiment_id'),'v6 changed more than source temperature')
        self.panel=self.read('configs/benchmarks.json')
        self.status=self.read(self.run/'status.json')
        require(self.status['identity_sha256']==self.ih,'supervisor identity differs')
        if not allow_active:
            require(self.status['active_jobs']==[] and self.status['runtime_and_checkpoint_unchanged'] is True,'final supervisor integrity differs')
            require(self.status['phase'] in ('complete_review_required','review_required','paused'),'not a finalized valid invocation')

    def terminal(self,row,player=None):
        require(row.get('status')=='ok' and row.get('terminal') is True and row.get('valid_game') is True and row.get('benchmark_failures')==[],'invalid completed terminal flags')
        raw=row['terminal_state'];winner=row['winner'];actions=row['actions']
        require(type(winner)is int and winner in (-1,0,1) and raw['terminal'] is True and raw['winner']==winner,'winner mismatch')
        require(type(row['terminal_round'])is int and row['terminal_round']>0 and raw['round']==row['terminal_round'],'round mismatch')
        require(raw.get('valid_game',True) is True and raw.get('benchmark_failures',[])==[],'failed terminal world')
        require(fp(raw)==row['terminal_state_sha256'],'terminal-world hash differs')
        require(row['terminal_summary']=={k:raw.get(k) for k in ('terminal','round','winner','win_reason','bank','kings','transferred','cat_damage')},'terminal summary differs')
        require(len(actions)==row['decisions']>0 and all(type(a)is int and 0<=a<40 for a in actions) and fp(actions)==row['actions_sha256'],'action transcript differs')
        require(stamp(row['ended_at'])>=stamp(row['started_at']),'terminal chronology reversed')
        require(number(row['elapsed_seconds']) and row['elapsed_seconds']>=0 and abs(stamp(row['ended_at'])-stamp(row['started_at'])-row['elapsed_seconds'])<2,'elapsed time differs')
        if player is not None:
            expected=Fraction(1,2) if winner==-1 else Fraction(int(winner==player))
            require(type(row['score']) in (int,float) and row['score'] in (0,.5,1) and Fraction(row['score'])==expected,'root-relative outcome differs')
        return raw

    def assert_sources(self):
        for path in sorted((self.run/'sources').glob('*.json')):
            r=self.read(path);sid=path.stem;split=r['split'];m,j=r['map_index'],r['j']
            require(split in ('training','validation') and type(m)is int and 0<=m<4 and type(j)is int and 1<=j<=(8 if split=='training' else 6),'source outside fixed inventory')
            engine_base,policy_base=(20261200,310000) if split=='training' else (20262200,320000)
            expected={'source_id':f'{split}-m{m}-j{j}','split':split,'map_index':m,'j':j,'map':self.config['maps'][m],'seed':engine_base+100*m+j,'policy_seeds':[2*(policy_base+100*m+j)+t for t in (0,1)],'experiment_identity':self.ih,'checkpoint_sha256':CP2,'temperature':1.0,'source_temperature':1.0,'anchor_seed':203151,'exclusion_map_sha256':r['exclusion_map_sha256']}
            require(sid==expected['source_id'] and all(r[k]==v for k,v in expected.items()) and r['identity']==fp(expected),'source identity differs')
            require(r.get('complete') is True and r['source']=='selfplay' and r['score'] is None,'incomplete or foreign source')
            raw=self.terminal(r);require(raw['map']==r['map'] and raw['seed']==r['seed'],'source terminal map/seed differs')
            require(self.path(r['source_json']).resolve()==path.resolve(),'source points to another version')
            cache=self.path(r['cache_npz']);require(cache.resolve()==path.with_suffix('.npz').resolve() and self.track(cache)==r['cache_sha256'],'source cache changed/reused')
            with np.load(cache,allow_pickle=False) as bundle:
                require(set(bundle.files)==set(FIELDS),'cache fields differ')
                arrays={k:bundle[k] for k in bundle.files};count=len(r['cache_positions'])
                for k,shape in FIELDS.items():require(arrays[k].shape==(count,*shape) and arrays[k].dtype==('uint8' if k=='mask' else 'float32') and np.isfinite(arrays[k]).all(),'cache shape/dtype/finite mismatch')
                for index,p in enumerate(r['cache_positions']):
                    require(p['feature_index']==index and p['source_id']==sid and 0<=p['decision_index']<r['decisions'],'cache position reference differs')
                    require(p['prefix_sha256']==fp(r['actions'][:p['decision_index']]),'root hidden-state prefix differs')
                    legal=p['legal_actions'];require(len(legal)==len(set(legal)) and legal and all(type(a)is int and 0<=a<40 for a in legal),'bad cached legality')
                    mask=np.zeros(40,dtype=np.uint8);mask[legal]=1
                    prior=arrays['prior'][index]
                    require(np.array_equal(mask,arrays['mask'][index]) and (prior>=0).all() and (prior[mask==0]==0).all() and abs(prior.sum(dtype=np.float64)-1)<1e-6,'cached mask/prior differs')
                    require(input_hash([arrays[k][index] for k in ('local','global_board','scalars','mask')])==p['input_hash'],'canonical model-input hash differs')
                    key=(sid,index);self.cache_meta[key]=p
                    self.cached[key]={k:arrays[k][index].copy() for k in ('hidden','prior','mask')}
                for role,p in r['root_candidates'].items():require(role==f"{p['role']}:{p['player']}" and 37 in p['legal_actions'] and p==r['cache_positions'][p['feature_index']],'candidate cache/role mismatch')
                for team in (0,1):
                    anchors=r['anchor_candidates'][str(team)];expected_count=min(16,r['actor_positions_seen'][str(team)]) if split=='training' else 0
                    require(len(anchors)==expected_count and len({a['decision_index'] for a in anchors})==len(anchors) and all(a['player']==team and a==r['cache_positions'][a['feature_index']] for a in anchors),'reservoir reference mismatch')
                    require(r['anchor_reservoir']['rng_seeds'][str(team)]==int(fp({'anchor_seed':203151,'source_id':sid,'player':team})[:16],16),'reservoir RNG changed')
            self.sources[sid]=r
            self.source_rows.append({'source_id':sid,'map':r['map'],'split':split,'decisions':r['decisions'],'round':r['terminal_round'],'winner':r['winner'],'root_candidates':sorted(r['root_candidates']),'cache_positions':len(r['cache_positions'])})
        # Preserve and verify all source attempts, including starts without an end.
        self.source_pending=[];self.interrupted_source_attempts=[];self.source_errors=0
        for directory in sorted((self.run/'sources').glob('*.attempts')):
            starts={};ends={}
            for path in sorted(directory.glob('attempt-*.json')):
                r=self.read(path);n=r['attempt'];isstart=path.name.endswith('-start.json');bucket=starts if isstart else ends
                require(type(n)is int and n>=1 and r['experiment_identity']==self.ih and path.name==f'attempt-{n:03d}'+('-start.json' if isstart else '.json') and n not in bucket and directory.name==r['source_id']+'.attempts','bad/duplicate source attempt')
                require(r['status'] in (('incomplete',) if isstart else ('ok','error','incomplete')),'unknown source attempt status')
                bucket[n]=r
            successful=[]
            for n,r in ends.items():
                require(n in starts and starts[n]['identity']==r['identity'] and starts[n]['started_at']==r['started_at'],'source attempt start differs')
                if r['status']=='ok':self.terminal(r);successful.append(n);require(self.sources.get(r['source_id'])==r,'source success not final immutable source')
                else:require(r.get('score') is None and r.get('terminal') is not True,'failed source has a score');self.source_errors+=1
            require(len(successful)<=1 and (not successful or not any(n>successful[0] for n in starts)),'successful source replay scheduled')
            errors=sum(row['status']=='error' for row in ends.values());require(errors<=2 and (not successful or errors<2),'source exceeded bounded error attempts')
            for n,r in starts.items():
                if n not in ends:
                    orphan={'source_id':r['source_id'],'attempt':n,'resolved_by_later_success':bool(successful and successful[0]>n)};self.interrupted_source_attempts.append(orphan)
                    if not orphan['resolved_by_later_success']:self.source_pending.append(orphan)

    def assert_roots(self,split):
        path=self.progress/f'roots-{split}.json'
        if not path.exists():return None
        manifest=self.read(path);require(manifest['identity_sha256']==self.ih and manifest['split']==split,'root manifest identity differs')
        count,maxj=(4,8) if split=='training' else (2,6)
        chosen=[];assignments=[];unavailable=[];usedsources=set()
        for m in range(4):
            used=set()
            for j in range(1,count+1):
                roles=[('BABY_RAT',0 if j%2 else 1),('RAT_KING',1 if j%2 else 0)];missing=list(roles);accepted={}
                for n in [j]+[n for n in range(count+1,maxj+1) if n%2==j%2]:
                    if n!=j and n in used:continue
                    if n!=j:used.add(n)
                    sid=f'{split}-m{m}-j{n}';require(sid in self.sources,'published manifest lacks required complete discovery source')
                    usedsources.add(sid);filled=[]
                    for role,team in list(missing):
                        candidate=self.sources[sid]['root_candidates'].get(f'{role}:{team}')
                        if candidate is not None:accepted[(role,team)]=(sid,candidate);missing.remove((role,team));filled.append(f'{role}:{team}')
                    assignments.append({'map_index':m,'slot_group_j':j,'source_j':n,'source_id':sid,'filled':filled})
                    if not missing:break
                for role,team in roles:
                    if (role,team) in accepted:chosen.append((m,j,*accepted[(role,team)]))
                    else:unavailable.append({'map_index':m,'j':j,'role':role,'player':team})
        require(assignments==manifest['assignments'] and unavailable==manifest['unavailable_slots'],'reserve assignment differs')
        require(usedsources==set(manifest['sources'])=={sid for sid,r in self.sources.items() if r['split']==split},'completed discovery source omitted/added')
        require(len(chosen)==len(manifest['roots']) and manifest['complete']==(not unavailable and len(chosen)==(32 if split=='training' else 16)),'root completeness differs')
        seen=set()
        for ordinal,((m,j,sid,p),r) in enumerate(zip(chosen,manifest['roots'])):
            require(all(r[k]==v for k,v in p.items()) and r['root_id']==f'{split}-{ordinal:03d}' and r['root_ordinal']==ordinal and r['map_index']==m and r['slot_group_j']==j and r['split']==split,'selected root differs from allocation')
            require(r['map']==self.sources[sid]['map'] and r['seed']==self.sources[sid]['seed'] and self.track(r['source_json'])==r['source_sha256'] and r['cache_sha256']==self.sources[sid]['cache_sha256'],'selected root provenance differs')
            key=(sid,r['decision_index']);require(key not in seen,'physical root reused');seen.add(key);self.rootrows[r['root_id']]=r
        require(manifest['distinct_root_source_games']==len({r['source_id'] for r in manifest['roots']}),'represented source-game count differs')
        self.original_time(path)
        return manifest

    def assert_anchors(self,training,validation):
        path=self.progress/'anchors.json'
        self.anchors=None;self.gradient=[]
        if not path.exists():return
        require(training is not None and training['complete'],'anchors precede complete training coverage')
        a=self.read(path);require(a['identity_sha256']==self.ih and a['complete'] is True and a['count']==len(a['anchors'])==512,'anchor completeness differs')
        rng=np.random.default_rng(203151);strata=[(sid,t) for sid in sorted(training['sources']) for t in (0,1)]
        quotient,remainder=divmod(512,len(strata));expected=[];quotas=[]
        for ordinal,(sid,team) in enumerate(strata):
            candidates=sorted(self.sources[sid]['anchor_candidates'][str(team)],key=lambda p:p['decision_index'])
            quota=quotient+(ordinal<remainder);require(len(candidates)>=quota,'anchor stratum under quota')
            indexes=sorted(rng.choice(len(candidates),size=quota,replace=False).tolist())
            expected.extend((sid,candidates[index]) for index in indexes)
            quotas.append({'source_id':sid,'player':team,'quota':quota})
        require(a['quotas']==quotas,'anchor integer quotas differ')
        for ordinal,((sid,p),actual) in enumerate(zip(expected,a['anchors'])):
            require(all(actual[k]==v for k,v in p.items()) and actual['anchor_id']==f'anchor-{ordinal:03d}','anchor subsampling differs')
            require(self.track(actual['source_json'])==actual['source_sha256'] and actual['cache_sha256']==self.sources[sid]['cache_sha256'],'anchor provenance differs')
        exclusions={}
        for kind,rows in (('root',training['roots']),('anchor',a['anchors'])):
            for row in rows:exclusions.setdefault(row['input_hash'],[]).append({'kind':kind,'id':row['root_id'] if kind=='root' else row['anchor_id']})
        gradient=sorted(exclusions)
        require(a['exclusion_map']==exclusions and a['gradient_input_hashes']==gradient and a['gradient_input_hashes_sha256']==fp(gradient),'gradient-input union differs')
        for sid,source in self.sources.items():
            if source['split']=='training':require(source['exclusion_map_sha256']==fp({}),'training input filtered')
            else:
                require(source['exclusion_map_sha256']==fp(exclusions),'validation used another gradient-input exclusion set')
                for skipped in source['skipped_input_duplicates']:
                    require(skipped['input_hash'] in exclusions and skipped['matches']==exclusions[skipped['input_hash']] and skipped['source_id']==sid,'invalid duplicate skip')
                require(all(p['input_hash'] not in exclusions for p in source['root_candidates'].values()),'validation candidate overlaps gradient inputs')
        if validation is not None:
            require(validation['gradient_input_hashes_sha256']==fp(gradient) and all(r['input_hash'] not in exclusions for r in validation['roots']),'validation manifest overlaps G')
        self.anchors=a;self.gradient=gradient;self.original_time(path)

    def assert_budget_comparators(self,training,validation):
        self.budget=None;self.comparators={}
        path=self.progress/'target-budget.json'
        if path.exists():
            require(training is not None and training['complete'],'budget precedes training coverage')
            self.budget=self.read(path);count=2*sum(len(r['legal_actions']) for r in training['roots'])
            require(self.budget=={'identity_sha256':self.ih,'selection_cells':count,'ceiling':1600,'within_ceiling':count<=1600},'pre-outcome ceiling differs')
            self.original_time(path)
        for path in sorted((self.progress/'comparators').glob('*.json')):
            row=self.read(path);root=self.rootrows.get(row['root_id'])
            require(root is not None and root['split']=='training','comparator outside training roots')
            spec={'experiment_identity':self.ih,'root_id':root['root_id'],'root_state_sha256':root['state_sha256'],'root_input_hash':root['input_hash'],'source_sha256':root['source_sha256'],'search':self.config['search'],'search_seed':203201+root['root_ordinal']}
            require(row['identity']==fp(spec) and all(row[k]==v for k,v in spec.items()),'PUCT comparator identity differs')
            visits=row['visits'];legal=root['legal_actions']
            require(set(visits)=={str(a) for a in legal} and all(type(v)is int and v>=0 for v in visits.values()) and sum(visits.values())==32,'PUCT budget/legal visits differ')
            expected=max(legal,key=lambda a:visits[str(a)])
            require(row['selected_action']==expected and 0<=row['max_depth']<=12,'PUCT action/depth differs')
            policy=np.zeros(40,dtype=np.float32)
            for a in legal:policy[a]=visits[str(a)]/32
            require(np.array_equal(policy,np.asarray(row['visit_policy'],dtype=np.float32)),'PUCT visit distribution differs')
            cached=self.cached[(root['source_id'],root['feature_index'])]
            require(np.array_equal(np.asarray(row['policy_prior'],dtype=np.float32),cached['prior']),'PUCT recorded prior differs from frozen cp2 cache')
            require(number(row['network_value']) and -1<=row['network_value']<=1,'invalid recorded PUCT value')
            self.comparators[root['root_id']]=row;self.original_time(path)
        if self.comparators:require(validation is not None and validation['complete'] and self.anchors is not None and self.budget['within_ceiling'],'comparators precede frozen allocation/G/budget')

    def cell_spec(self,root,stage,action,seed):
        return {'experiment_identity':self.ih,'root_id':root['root_id'],'root_player':root['player'],'root_state_sha256':root['state_sha256'],'root_prefix_sha256':root['prefix_sha256'],'root_input_hash':root['input_hash'],'source_sha256':root['source_sha256'],'stage':stage,'action':action,'seed':seed,'temperature':.35}

    def assert_cells(self,training,validation):
        self.actions=None
        path=self.progress/'validation-actions.json'
        if path.exists():self.actions=self.read(path);self.original_time(path)
        for directory in sorted((self.progress/'cells').glob('*')):
            if not directory.is_dir():continue
            starts={};ends={}
            for path in sorted(directory.glob('attempt-*.json')):
                row=self.read(path);rid=row['root_id'];root=self.rootrows.get(rid);stage=row['stage'];action,seed=row['action'],row['seed'];n=row['attempt']
                require(root is not None and stage in ('selection','validation') and root['split']==('training' if stage=='selection' else 'validation'),'cell outside split/roots')
                require(type(n)is int and n>=1 and type(action)is int and action in root['legal_actions'] and seed in self.config[stage+'_seeds'],'cell outside legal/seed inventory')
                require(directory.name==f'{stage}-{rid}-a{action:02d}-s{seed}','cell directory differs')
                if stage=='validation':require(self.actions is not None and action in self.actions['actions'][rid].values(),'validation action not frozen')
                spec=self.cell_spec(root,stage,action,seed)
                require(row['identity']==fp(spec) and all(row[k]==v for k,v in spec.items()),'cell identity differs')
                isstart=path.name.endswith('-start.json');bucket=starts if isstart else ends
                require(path.name==f'attempt-{n:03d}'+('-start.json' if isstart else '.json') and n not in bucket,'bad/duplicate cell attempt filename')
                require(row['status'] in (('incomplete',) if isstart else ('ok','error','incomplete')),'unknown cell attempt status')
                bucket[n]=row
            successes=[];key=None
            for n,row in ends.items():
                key=(row['stage'],row['root_id'],row['action'],row['seed'])
                require(n in starts and row['started_at']==starts[n]['started_at'],'cell start/result differs')
                if row['status']=='ok':
                    raw=self.terminal(row,row['root_player']);root=self.rootrows[row['root_id']]
                    require(row['actions'][0]==row['action'] and raw['map']==root['map'] and raw['seed']==root['seed'],'forced action or physical source differs')
                    require(key not in self.cells,'duplicate successful physical cell')
                    self.cells[key]=row;successes.append(n);self.cell_rows.append(row)
                else:
                    require(row.get('score') is None and row.get('terminal') is not True,'partial cell has terminal score')
                    self.failed_attempts.append(row)
            require(len(successes)<=1 and (not successes or not any(n>successes[0] for n in starts)),'successful cell replay was scheduled')
            errors=sum(row['status']=='error' for row in ends.values());require(errors<=2 and (not successes or errors<2),'cell exceeded bounded error attempts')
            for n,r in starts.items():
                if n not in ends:
                    orphan={'stage':r['stage'],'root_id':r['root_id'],'action':r['action'],'seed':r['seed'],'attempt':n,'resolved_by_later_success':bool(successes and successes[0]>n)};self.interrupted_cell_attempts.append(orphan)
                    if not orphan['resolved_by_later_success']:self.starts.append(orphan)
        if self.cell_rows or self.failed_attempts or self.starts:
            require(training is not None and training['complete'] and validation is not None and validation['complete'] and len(self.comparators)==32 and self.budget['within_ceiling'],'terminal scoring began before frozen complete prerequisites')
        self.selection_expected=[]
        if training is not None:
            for root in training['roots']:
                rid=root['root_id'];means={}
                for action in root['legal_actions']:
                    keys=[('selection',rid,action,seed) for seed in self.config['selection_seeds']]
                    self.selection_expected.extend(keys)
                    if all(k in self.cells for k in keys):means[action]=sum(Fraction(self.cells[k]['score']) for k in keys)/2
                if len(means)==len(root['legal_actions']):
                    prior=self.cached[(root['source_id'],root['feature_index'])]['prior']
                    self.selected[rid]=max(root['legal_actions'],key=lambda a:(means[a],float(prior[a]),-a))
        self.selection_complete=training is not None and training['complete'] and len(self.selected)==32

    def assert_dataset_fit(self,training,validation,replay=True):
        self.dataset=None;self.fit=None;self.fit_audit=None;self.payloads={};self.pending_fit=[]
        for path in sorted((self.run/'fit.pending').rglob('*')):
            if path.is_file():self.pending_fit.append({'path':str(path.relative_to(ROOT)),'sha256':self.track(path)})
        dataset=self.run/'dataset/training.npz';metadata=dataset.with_suffix('.json');fitdir=self.run/'fit'
        if not metadata.exists():
            require(not fitdir.exists() and self.actions is None,'fit/actions without completed dataset')
            return
        require(self.selection_complete and not any(r['stage']=='selection' for r in self.starts) and self.anchors is not None,'dataset before complete target evidence')
        meta=self.read(metadata);self.original_time(metadata);dsha=self.track(dataset)
        games=sorted(training['sources']);roots=training['roots'];anchors=self.anchors['anchors']
        expected={'format_version':1,'identity_sha256':self.ih,'npz_sha256':dsha,'complete':True,'source':'own_selfplay','benchmark_data_used':False,'validation_data_used':False,'root_count':32,'anchor_count':512,'training_maps':self.config['maps'],
            'training_seeds':sorted(self.sources[g]['seed'] for g in games),'selection_counts':{'expected':len(self.selection_expected),'completed':len(self.selection_expected),'errors':0,'attempt_errors':sum(r.get('status')=='error' and r['stage']=='selection' for r in self.failed_attempts)},
            'training_games':[{'game_id':g,'map':self.sources[g]['map'],'seed':self.sources[g]['seed']} for g in games],
            'root_ids':[r['root_id'] for r in roots],'anchor_refs':anchors,'root_input_hashes':[r['input_hash'] for r in roots],'anchor_input_hashes':[r['input_hash'] for r in anchors],'validation_input_hashes':[r['input_hash'] for r in validation['roots']],'gradient_input_hashes':self.gradient,
            'provenance':{'training_roots_sha256':fp(training),'validation_roots_sha256':fp(validation),'anchors_sha256':fp(self.anchors),'checkpoint_sha256':CP2,'comparators':{r['root_id']:self.track(self.progress/'comparators'/f"{r['root_id']}.json") for r in roots}}}
        require(meta==expected,'dataset metadata differs from independently verified evidence')
        want={f'{prefix}_{k}':np.stack([self.cached[(r['source_id'],r['feature_index'])][k] for r in rows]) for prefix,rows in (('root',roots),('anchor',anchors)) for k in ('hidden','prior','mask')}
        want.update(terminal_actions=np.asarray([self.selected[r['root_id']] for r in roots],dtype=np.int64),puct_actions=np.asarray([self.comparators[r['root_id']]['selected_action'] for r in roots],dtype=np.int64),root_strata=np.asarray([[r['map_index'],int(r['role']=='RAT_KING'),r['player']] for r in roots],dtype=np.int64),anchor_game=np.asarray([games.index(r['source_id']) for r in anchors],dtype=np.int64),anchor_player=np.asarray([r['player'] for r in anchors],dtype=np.int64))
        with np.load(dataset,allow_pickle=False) as archive:
            require(set(archive.files)==set(want),'dataset tensor inventory differs')
            for k,v in want.items():require(archive[k].dtype==v.dtype and np.array_equal(archive[k],v),'dataset cached tensor/selected target differs: '+k)
        self.dataset={'sha256':dsha,'metadata_sha256':self.track(metadata),'terminal_actions':want['terminal_actions'].tolist(),'puct_actions':want['puct_actions'].tolist(),'target_disagreements':int(np.sum(want['terminal_actions']!=want['puct_actions']))}
        if not fitdir.exists():require(self.actions is None,'validation actions before atomic completed fit');return
        self.fit=self.read(fitdir/'manifest.json');self.original_time(fitdir/'manifest.json')
        fit=self.fit;cfg=self.config['fit'];arrays=want
        for prefix in ('root','anchor'):arrays[prefix+'_prior']=normalized(arrays[prefix+'_prior'],arrays[prefix+'_mask'])
        strata=arrays['root_strata'];groups=[np.flatnonzero(np.all(strata==(m,r,t),axis=1)) for m in range(4) for r in (0,1) for t in (0,1)]
        require(all(len(g)==2 for g in groups),'two-per-stratum balance differs')
        anchor_groups={g:[np.flatnonzero((arrays['anchor_game']==g)&(arrays['anchor_player']==t)) for t in (0,1)] for g in range(len(games))}
        rng=np.random.default_rng(203101);batches={k:np.empty((128,32),dtype=np.int64) for k in ('root','anchor')}
        for update in range(128):
            for column in range(32):
                group=groups[int(rng.integers(16))];batches['root'][update,column]=group[int(rng.integers(len(group)))]
            for column in range(32):
                group=anchor_groups[int(rng.integers(len(games)))][int(rng.integers(2))];batches['anchor'][update,column]=group[int(rng.integers(len(group)))]
        rngstate=rng.bit_generator.state;h=hashlib.sha256()
        for k in ('root','anchor'):h.update(json.dumps([k,list(batches[k].shape)],separators=(',',':')).encode());h.update(batches[k].astype('<i8').tobytes())
        h.update(fp(rngstate).encode());bsha=h.hexdigest()
        guard={'checked':True,'gradient_input_count':len(self.gradient),'validation_input_count':16,'intersection_count':0,'set_sha256':fp(self.gradient)}
        identity={'identity_sha256':self.ih,'dataset_sha256':dsha,'dataset_metadata_sha256':self.track(metadata),'checkpoint_sha256':CP2,'fit_config':cfg,'batch_sha256':bsha,'fitter_sha256':self.track('alpha0/terminal_v6_fit.py'),'common_sha256':self.track('alpha0/terminal_v6_common.py'),'torch_version':packages.version('torch'),'numpy_version':np.__version__}
        require(fit['complete'] is True and fit['experiment_id']=='terminal-target-head-v6' and fit['identity_sha256']==self.ih and fit['fit_identity']==identity and fit['fit_identity_sha256']==fp(identity) and fit['fit_config']==cfg and fit['gradient_input_overlap']==guard,'fit identity/guard differs')
        files={'terminal_head.pt','puct_head.pt','batches.npz','batch-rng.json','parent-metadata.json','identity.json'}
        require(set(fit['files'])==files and all(self.track(fitdir/k)==v for k,v in fit['files'].items()),'atomic fit artifact inventory differs')
        require(self.read(fitdir/'identity.json')==identity and self.read(fitdir/'batch-rng.json')==rngstate,'fit sidecar identity/RNG differs')
        with np.load(fitdir/'batches.npz',allow_pickle=False) as archive:require(set(archive.files)=={'root','anchor'} and all(archive[k].dtype==np.int64 and np.array_equal(archive[k],batches[k]) for k in batches),'shared batches differ')
        require(fit['dataset']=={'path':str(dataset.relative_to(ROOT)),'sha256':dsha,'metadata_sha256':self.track(metadata),'root_count':32,'anchor_count':512},'fit dataset declaration differs')
        require(fit['batch_schedule']=={'path':str((fitdir/'batches.npz').relative_to(ROOT)),'sha256':fit['files']['batches.npz'],'root_shape':[128,32],'anchor_shape':[128,32],'seed':203101},'fit batch declaration differs')
        import torch
        from torch.nn import functional as F
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        base=torch.load(self.path(self.config['checkpoint']),map_location='cpu',weights_only=True);self.payloads['cp2']=base;state0=base['model_state_dict']
        ancestor={k:v for k,v in base.items() if k not in ('model_state_dict','optimizer_state_dict','torch_rng','numpy_rng')}
        require(self.read(fitdir/'parent-metadata.json')==ancestor,'cp2 ancestry snapshot differs')
        def tensorbytes(x):return x.detach().cpu().contiguous().numpy().tobytes()
        def same(a,b):
            if torch.is_tensor(a) or torch.is_tensor(b):return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and tensorbytes(a)==tensorbytes(b)
            if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(same(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
            return type(a)==type(b) and a==b
        frozen=hashlib.sha256()
        for key,value in sorted(state0.items()):
            if key in POLICY_KEYS:continue
            header=json.dumps([key,str(value.dtype),list(value.shape)],separators=(',',':')).encode();frozen.update(len(header).to_bytes(8,'little'));frozen.update(header);frozen.update(tensorbytes(value))
        frozen=frozen.hexdigest();require(fit['frozen_state_sha256']==frozen,'fit frozen state hash differs')
        allhidden=torch.as_tensor(np.concatenate((arrays['root_hidden'],arrays['anchor_hidden'])))
        def value_outputs(state):return F.linear(F.silu(F.linear(allhidden,state['value.0.weight'],state['value.0.bias'])),state['value.2.weight'],state['value.2.bias'])
        with torch.inference_mode():value0=value_outputs(state0)
        for prefix in ('root','anchor'):
            with torch.inference_mode():
                logits=F.linear(torch.as_tensor(arrays[prefix+'_hidden']),state0['policy.weight'],state0['policy.bias']);actual=logits.masked_fill(~torch.as_tensor(arrays[prefix+'_mask'],dtype=torch.bool),torch.finfo(torch.float32).min).softmax(-1).numpy()
            require(np.allclose(actual,arrays[prefix+'_prior'],atol=1e-6,rtol=1e-5),'cached baseline priors differ from frozen hidden/head')
        def logp(head,hidden,mask):return F.log_softmax(head(torch.as_tensor(hidden)).masked_fill(~torch.as_tensor(mask,dtype=torch.bool),torch.finfo(torch.float32).min),dim=-1)
        def loss(head,ri,ai,actions):
            rp=torch.as_tensor(normalized(arrays['root_prior'][ri],arrays['root_mask'][ri]));ap=torch.as_tensor(normalized(arrays['anchor_prior'][ai],arrays['anchor_mask'][ai]))
            q=.25*rp+.75*F.one_hot(torch.as_tensor(actions[ri]),40).float()
            ce=-(q*logp(head,arrays['root_hidden'][ri],arrays['root_mask'][ri])).sum(-1).mean()
            kl=(torch.xlogy(ap,ap)-ap*logp(head,arrays['anchor_hidden'][ai],arrays['anchor_mask'][ai])).sum(-1).mean()
            return ce+kl,{'loss':float((ce+kl).detach()),'target_ce':float(ce.detach()),'anchor_kl':float(kl.detach())}
        report={'fit_identity_sha256':fp(identity),'shared_batches_sha256':bsha,'frozen_state_sha256':frozen,'gradient_input_overlap':guard,'replay_performed':bool(replay),'arms':{}}
        for arm in ARMS[1:]:
            path=fitdir/(arm+'.pt');record=fit['checkpoints'][arm];payload=torch.load(path,map_location='cpu',weights_only=True);state=payload['model_state_dict'];self.payloads[arm]=payload
            require(record['path']==str(path.relative_to(ROOT)) and record['sha256']==self.track(path) and record['fit_identity_sha256']==fp(identity) and record['updates']==128 and record['trainable_parameters']==5160 and record['frozen_tensors_equal'] is True and record['value_logits_equal'] is True,'candidate manifest path/hash/proofs differ')
            require(payload['arm']==arm and payload['fit_version']=='policy-head-v6-1' and payload['updates']==payload['gradient_updates']==128 and payload['trainable_parameters']==5160 and payload['policy_head_only'] is True and payload['benchmark_data_used'] is False and payload['validation_data_used'] is False,'candidate protocol differs')
            require(payload['parent_checkpoint_sha256']==CP2 and payload['parent_checkpoint_metadata']==ancestor and payload['fit_identity_sha256']==fp(identity) and payload['dataset_sha256']==dsha and payload['dataset_metadata_sha256']==self.track(metadata) and payload['batch_sha256']==bsha and payload['frozen_state_sha256']==frozen and payload['gradient_input_overlap']==guard,'candidate lineage differs')
            require(set(state)==set(state0) and all(state[k].shape==state0[k].shape and state[k].dtype==state0[k].dtype and torch.isfinite(state[k]).all() for k in state),'candidate tensor inventory/finite differs')
            require(all(same(state[k],state0[k]) for k in state if k not in POLICY_KEYS) and sum(state[k].numel() for k in POLICY_KEYS)==5160,'candidate changed frozen tensors')
            with torch.inference_mode():require(same(value_outputs(state),value0),'candidate cached critic outputs differ')
            require(all(payload['batch_indices'][k].dtype==torch.int64 and np.array_equal(payload['batch_indices'][k].numpy(),batches[k]) for k in batches) and same(payload['numpy_rng'],rngstate),'candidate batches/RNG differ')
            seeded=torch.Generator().manual_seed(203101).get_state();require(same(payload['torch_rng'],seeded),'candidate Torch RNG differs')
            require(len(payload['fit_history'])==128,'fit history length differs')
            head=torch.nn.Linear(128,40);head.load_state_dict({k.split('.')[1]:state0[k] for k in POLICY_KEYS});optimizer=torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
            actions=arrays['terminal_actions' if arm=='terminal_head' else 'puct_actions']
            if replay:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(203101)
                    for update in range(128):
                        optimizer.zero_grad(set_to_none=True);objective,metrics=loss(head,batches['root'][update],batches['anchor'][update],actions);objective.backward();optimizer.step()
                        history=payload['fit_history'][update]
                        require(history['step']==update+1 and all(abs(history[k]-v)<=2e-6 for k,v in metrics.items()),'independent replay loss history differs')
                    require(same(head.weight,state['policy.weight']) and same(head.bias,state['policy.bias']),'independent 128-update head replay differs')
                    require(same(optimizer.state_dict(),payload['optimizer_state_dict']),'independent AdamW moments/step/settings differ')
                    require(same(torch.get_rng_state(),payload['torch_rng']),'independent replay RNG differs')
            else:head.load_state_dict({k.split('.')[1]:state[k] for k in POLICY_KEYS})
            with torch.inference_mode():
                _,metrics=loss(head,np.arange(32),np.arange(512),actions)
                rootchoices=logp(head,arrays['root_hidden'],arrays['root_mask']).argmax(-1).numpy();anchorchoices=logp(head,arrays['anchor_hidden'],arrays['anchor_mask']).argmax(-1).numpy()
                metrics.update(root_target_agreement=float(np.mean(rootchoices==actions)),root_action_changed_fraction=float(np.mean(rootchoices!=arrays['root_prior'].argmax(-1))),anchor_action_changed_fraction=float(np.mean(anchorchoices!=arrays['anchor_prior'].argmax(-1))))
            require(payload['fit_metrics']==record['metrics'] and all(abs(record['metrics'][k]-v)<=2e-6 for k,v in metrics.items()),'fit full-data objective differs')
            report['arms'][arm]={'sha256':self.track(path),'updates':128,'trainable_parameters':5160,'frozen_tensor_bytes_equal':True,'head_bytes_replayed_exactly':bool(replay),'optimizer_replayed_exactly':bool(replay),'metrics':metrics}
        self.fit_audit=report
        summary=self.progress/'fit-summary.json'
        if summary.exists():require(self.read(summary)==fit,'public fit summary differs')

    def assert_validation(self,validation):
        self.validation={'complete':False,'roots':[],'expected_cells':0,'completed_cells':0,'research_rule_pass':False}
        if self.actions is None:return
        require(self.fit is not None and self.selection_complete and validation['complete'],'validation action freeze before complete fit')
        import torch
        from torch.nn import functional as F
        require(self.actions['identity_sha256']==self.ih and self.actions['roots_sha256']==fp(validation),'validation action identity differs')
        hashes={'cp2':CP2,**{arm:self.fit['checkpoints'][arm]['sha256'] for arm in ARMS[1:]}}
        require(self.actions['checkpoint_sha256_by_arm']==hashes and set(self.actions['actions'])=={r['root_id'] for r in validation['roots']},'validation checkpoint/root inventory differs')
        rows=[];expected=set()
        for root in validation['roots']:
            rid=root['root_id'];cached=self.cached[(root['source_id'],root['feature_index'])];chosen={}
            for arm in ARMS:
                state=self.payloads[arm]['model_state_dict']
                with torch.inference_mode():
                    logits=F.linear(torch.as_tensor(cached['hidden']).unsqueeze(0),state['policy.weight'],state['policy.bias'])[0]
                    chosen[arm]=int(logits.masked_fill(~torch.as_tensor(cached['mask'],dtype=torch.bool),-torch.inf).argmax())
            require(self.actions['actions'][rid]==chosen,'validation greedy action differs')
            for action in set(chosen.values()):expected.update(('validation',rid,action,seed) for seed in self.config['validation_seeds'])
            if all(('validation',rid,a,s) in self.cells for a in set(chosen.values()) for s in self.config['validation_seeds']):
                means={arm:sum(Fraction(self.cells[('validation',rid,chosen[arm],seed)]['score']) for seed in self.config['validation_seeds'])/4 for arm in ARMS}
                rows.append({'root_id':rid,'actions':chosen,'score_by_arm':{k:float(v) for k,v in means.items()},'delta_terminal_minus_puct':float(means['terminal_head']-means['puct_head']),'delta_terminal_minus_cp2':float(means['terminal_head']-means['cp2'])})
        complete=len(rows)==16;result={'complete':complete,'roots':rows,'expected_cells':len(expected),'completed_cells':len(expected&set(self.cells)),'research_rule_pass':False}
        if complete:
            ds=[Fraction(r['delta_terminal_minus_puct']) for r in rows];base=[Fraction(r['delta_terminal_minus_cp2']) for r in rows];mean=sum(ds)/16;meanbase=sum(base)/16
            result.update(mean_delta=float(mean),mean_delta_exact=str(mean),improved_roots=sum(d>0 for d in ds),worst_root_delta=float(min(ds)),mean_vs_cp2=float(meanbase),mean_vs_cp2_exact=str(meanbase),research_rule_pass=mean>=Fraction(1,10) and sum(d>0 for d in ds)>=2 and min(ds)>=Fraction(-1,4) and meanbase>=0)
        self.validation=result

    def assert_benchmarks(self):
        self.benchmarks={'complete':False,'expected_cells':72,'completed_cells':0,'opponents':[],'each_opponent_nonregression':False};self.benchmark_rows=[];self.benchmark_attempts=[];self.benchmark_specs={}
        path=self.progress/'benchmarks/specs.json'
        if not path.exists():
            require(not list((self.progress/'benchmarks/cells').glob('*/attempt-*.json')),'benchmark evidence lacks frozen specs')
            return
        require(self.validation['complete'] and self.fit is not None,'benchmarks before complete held validation')
        saved=self.read(path);self.original_time(path);require(saved['identity_sha256']==self.ih,'benchmark spec experiment differs')
        bots=self.panel['bots'];require(len(bots)==12 and len({b['id'] for b in bots})==12,'panel is not all twelve')
        panel=fp(sorted(({k:b[k] for k in ('id','source_commit','java_tree_sha256')} for b in bots),key=lambda r:r['id']))
        runtime=self.identity['evaluation_runtime'];expected=[];mapsha=next(m['sha256'] for m in self.identity['maps'] if m['name']=='tiny')
        for bot in bots:
            for source in bot['files']:require(self.track(source['path'])==source['sha256'],'benchmark source changed')
            require({str(p.relative_to(ROOT)) for p in self.path(bot['path']).rglob('*.java')}=={f['path'] for f in bot['files'] if f['path'].endswith('.java')},'extra/missing benchmark Java sources')
        for arm in ARMS:
            payload=self.payloads[arm];cp=self.path(self.config['checkpoint']) if arm=='cp2' else self.run/'fit'/(arm+'.pt');cpsha=self.track(cp)
            schema={'model_version':'dual_scale_v2','feature_version':2,'action_version':'macro_v2','action_dim':40,'network_config':payload['network_config']}
            require(payload['model_version']=='dual_scale_v2' and payload['network_config']['action_dim']==40 and payload.get('feature_version',2)==2 and payload.get('action_version','macro_v2')=='macro_v2','benchmark model schema differs')
            config={'policy_schema':schema,'runtime':runtime,'search_mode':'greedy','search_config':{},'timeout_seconds':1800.0,'torch_threads':1,'tie_break':'first-legal-action','policy_random_seed':0}
            for bot in bots:
                for side in ('A','B'):
                    identity={'artifact_version':2,'checkpoint_sha256':cpsha,'evaluation_config_sha256':fp(config),'evaluation_config':config,'benchmark_suite_id':self.panel['suite_id'],'panel_fingerprint_sha256':panel,'opponent':bot['id'],'opponent_java_sha256':bot['java_tree_sha256'],'opponent_source_commit':bot['source_commit'],'map':'tiny','map_sha256':mapsha,'seed':0,'side':side,'stage':'smoke'}
                    evaluation={'evaluation_cell_id':fp(identity),'identity':identity,'checkpoint_path':str(self.execution_root/cp.relative_to(ROOT)),'iteration':int(payload.get('iteration',0)),'generation':int(payload.get('iteration',0)),'suite':'smoke','stage':'smoke','maps_dir':str(self.execution_root/'maps'),'bot':bot}
                    key={'experiment_identity':self.ih,'arm':arm,'evaluation_cell_id':fp(identity)};expected.append({'stage':'benchmark',**key,'cell_id':fp(key),'evaluation_spec':evaluation})
        require(saved['specs']==expected,'benchmark 72-cell specification differs')
        self.benchmark_specs={s['cell_id']:s for s in expected};success={}
        for directory in sorted((self.progress/'benchmarks/cells').glob('*')):
            require(directory.is_dir() and directory.name in self.benchmark_specs,'unexpected benchmark cell directory')
            spec=self.benchmark_specs[directory.name];evspec=spec['evaluation_spec'];identity=evspec['identity'];seen=[];ok=[]
            for path in sorted(directory.glob('attempt-*.json')):
                row=self.read(path);n=row['attempt'];ev=row['evaluation'];seen.append(n)
                require(type(n)is int and 1<=n<=2 and path.name==f'attempt-{n:03d}.json','benchmark attempt filename/index differs')
                require(row['schema_version']==1 and row['stage']=='benchmark' and row['experiment_identity']==self.ih and row['cell_id']==spec['cell_id'] and row['arm']==spec['arm'] and row['status']==ev['status'] and ev['attempt']==n,'benchmark wrapper differs')
                require(ev['evaluation_cell_id']==evspec['evaluation_cell_id'] and ev['identity']==identity and ev['checkpoint_path']==evspec['checkpoint_path'] and ev['suite']==ev['stage']=='smoke','benchmark evaluation identity differs')
                require(all(ev[k]==identity[k] for k in ('checkpoint_sha256','evaluation_config_sha256','benchmark_suite_id','panel_fingerprint_sha256','opponent','opponent_java_sha256','opponent_source_commit','map','map_sha256','seed','side')),'benchmark copied identity fields differ')
                require(all(ev[k]==runtime[k] for k in ('engine_version','engine_jar_sha256','engine_patch_sha256','bridge_sha256')) and ev['action_schema']=='macro_v2' and ev['observation_schema']==2 and ev['policy']=='greedy-network-no-search' and ev['search_mode']=='greedy','benchmark copied runtime fields differ')
                if row['status']=='ok':
                    raw=ev['terminal'];winner=raw['winner'];score=Fraction(1,2) if winner==-1 else Fraction(int(winner==(0 if identity['side']=='A' else 1)))
                    require(type(winner)is int and winner in (-1,0,1) and ev['winner']==winner and type(ev['rounds'])is int and ev['rounds']>0 and raw['round']==ev['rounds'],'benchmark terminal winner/round differs')
                    require(raw['terminal'] is True and raw['valid_game'] is True and raw['benchmark_failures']==[] and raw['map']=='tiny' and raw['seed']==0 and raw['action_version']=='macro_v2','benchmark terminal proof differs')
                    require(type(ev['score']) in (int,float) and ev['score'] in (0,.5,1) and Fraction(ev['score'])==score and type(ev['decisions'])is int and ev['decisions']>0,'benchmark learned-seat score differs')
                    require(number(ev['seconds']) and ev['seconds']>=0 and stamp(row['recorded_at'])>=stamp(ev['started_at']) and abs(stamp(row['recorded_at'])-stamp(ev['started_at'])-ev['seconds'])<5,'benchmark wall clock differs')
                    bot=evspec['bot'];buildid=fp({'source':bot['java_tree_sha256'],'engine':runtime['engine_jar_sha256'],'jdk':runtime['jdk_release_sha256']});build=ROOT/'build/evaluation-benchmarks'/bot['id']/buildid
                    classes={p.relative_to(build).as_posix():self.track(p) for p in sorted(build.rglob('*.class'))}
                    require(classes and classes==self.read(build/'.class-hashes.json') and ev['opponent_classes_sha256']==fp(classes),'benchmark compiled opponent proof differs')
                    require(spec['cell_id'] not in success,'duplicate successful benchmark execution');success[spec['cell_id']]=row;ok.append(n);self.benchmark_rows.append(row)
                else:require(ev.get('score') is None,'failed benchmark has score')
                self.benchmark_attempts.append(row)
            require(len(seen)==len(set(seen)) and (not ok or max(seen)==ok[0]),'benchmark success replayed/duplicate attempt')
        opponents=[]
        for bot in bots:
            seats={};complete=True
            for arm in ARMS:
                seats[arm]={}
                for side in ('A','B'):
                    spec=next(s for s in expected if s['arm']==arm and s['evaluation_spec']['identity']['opponent']==bot['id'] and s['evaluation_spec']['identity']['side']==side)
                    if spec['cell_id'] in success:seats[arm][side]=success[spec['cell_id']]['evaluation']['score']
                    else:complete=False
            row={'opponent':bot['id'],'complete':complete,'scores_by_arm_and_seat':seats}
            if complete:
                row['mean_by_arm']={arm:float(sum(map(Fraction,seats[arm].values()))/2) for arm in ARMS};row['terminal_minus_cp2']=row['mean_by_arm']['terminal_head']-row['mean_by_arm']['cp2'];row['nonregression']=row['terminal_minus_cp2']>=0
            opponents.append(row)
        self.benchmarks={'complete':len(success)==72,'expected_cells':72,'completed_cells':len(success),'attempt_errors':sum(r['status']!='ok' for r in self.benchmark_attempts),'opponents':opponents,'each_opponent_nonregression':len(success)==72 and all(r['nonregression'] for r in opponents)}

    def assert_chronology(self,training,validation):
        self.jobs=[];self.job_pending=[];self.interrupted_jobs=[];intervals=[];bykind={};inferred_intervals={}
        for directory in sorted((self.run/'jobs').glob('*')):
            require(directory.is_dir(),'unexpected job artifact')
            descriptor=self.read(directory/'job.json');task=dict(descriptor['job']);taskpath=task.pop('worker_task_path')
            require(descriptor['identity_sha256']==self.ih and directory.name==fp({'identity_sha256':self.ih,'job':task}) and taskpath==str((directory/'task.json').relative_to(ROOT)),'supervisor job identity differs')
            kind,payload=task['kind'],task['payload'];require(kind in ('source','comparator','selection','fit','validation','benchmark'),'unknown supervised operation')
            if kind in ('comparator','selection'):require(payload==self.rootrows[payload['root_id']] and payload['split']=='training','supervised training root differs')
            elif kind=='validation':require(payload=={**self.rootrows[payload['root_id']],'actions':self.actions['actions'][payload['root_id']]},'supervised validation actions differ')
            elif kind=='fit':require(payload=={'dataset':str((self.run/'dataset/training.npz').relative_to(ROOT))},'supervised fit dataset differs')
            elif kind=='benchmark':require(payload==self.benchmark_specs[payload['cell_id']],'supervised benchmark spec differs')
            else:
                split,m,j=payload['split'],payload['map_index'],payload['j'];engine,policy=(20261200,310000) if split=='training' else (20262200,320000)
                require(split in ('training','validation') and type(m)is int and 0<=m<4 and type(j)is int and 1<=j<=(8 if split=='training' else 6),'supervised source outside inventory')
                expected_source={'source_id':f'{split}-m{m}-j{j}','split':split,'map_index':m,'j':j,'map':self.config['maps'][m],'seed':engine+100*m+j,'policy_seeds':[2*(policy+100*m+j)+t for t in (0,1)]}
                if split=='validation':expected_source['exclusion_map']=self.anchors['exclusion_map']
                require(payload==expected_source,'supervised source/G differs')
            starts={};ends={}
            for path in sorted(directory.glob('attempt-*.json')):
                row=self.read(path);n=row['attempt'];isstart=path.name.endswith('-start.json')
                require(type(n)is int and n>0 and path.name==f'attempt-{n:03d}'+('-start.json' if isstart else '.json'),'supervisor attempt filename differs')
                require(row['identity_sha256']==self.ih and row['job_hash']==directory.name and row['job']==task,'supervisor attempt differs')
                bucket=starts if isstart else ends;require(n not in bucket,'duplicate supervisor attempt');bucket[n]=row
            for n,row in ends.items():
                require(n in starts and all(row[k]==v for k,v in starts[n].items()),'supervisor start/result mismatch')
                start,end=stamp(row['started_at']),stamp(row['ended_at']);require(end>=start,'supervisor timestamps reversed')
                if 'elapsed_seconds' in row:require(number(row['elapsed_seconds']) and abs(end-start-row['elapsed_seconds'])<2,'supervisor elapsed mismatch')
                require(type(row['returncode'])is int or (row['returncode'] is None and bool(row.get('reason'))),'supervisor lacks exit code/reason');intervals.append((start,1,kind));intervals.append((end,-1,kind));self.jobs.append(row);bykind.setdefault(kind,[]).append((start,end,row))
            for n,row in starts.items():
                if n not in ends:
                    later_success=[index for index,endrow in ends.items() if index>n and endrow['returncode']==0 and endrow.get('reason') is None]
                    orphan={'job_hash':directory.name,'kind':kind,'attempt':n,'started_at':row['started_at'],'resolved_by_later_success':bool(later_success)};self.interrupted_jobs.append(orphan)
                    if not later_success:self.job_pending.append(orphan)
                    else:
                        later_starts=[stamp(startrow['started_at']) for index,startrow in starts.items() if index>n]
                        inferred_intervals.setdefault(kind,[]).append((stamp(row['started_at']),min(later_starts),row))
        active=peak=0;benchmark_active=benchmark_peak=0
        for _,change,kind in sorted(intervals):
            active+=change;peak=max(peak,active)
            if kind=='benchmark':benchmark_active+=change;benchmark_peak=max(benchmark_peak,benchmark_active)
        require(peak<=3 and benchmark_peak<=2,'supervisor parallelism exceeded preregistration')
        checks=[]
        def initial_successful_ends(kind):
            byroot={}
            for start,end,row in bykind.get(kind,[]):
                if row['returncode']==0 and row.get('reason') is None:
                    key=row['job']['payload'].get('root_id','one');byroot[key]=min(end,byroot.get(key,end))
            return byroot
        def before(name,left,right):
            require(left<=right+.002,'chronology violated: '+name);checks.append({'barrier':name,'before_utc':datetime.fromtimestamp(left,timezone.utc).isoformat(),'after_utc':datetime.fromtimestamp(right,timezone.utc).isoformat()})
        if training is not None:
            before('complete training sources before training-root freeze',max(stamp(r['ended_at']) for r in self.sources.values() if r['split']=='training'),self.original_time(self.progress/'roots-training.json'))
        if self.anchors is not None:
            before('training-root freeze before anchor freeze',self.original_time(self.progress/'roots-training.json'),self.original_time(self.progress/'anchors.json'))
            for source in self.sources.values():
                if source['split']=='validation':before('G frozen before validation source '+source['source_id'],self.original_time(self.progress/'anchors.json'),stamp(source['started_at']))
        if validation is not None:
            before('complete validation sources before validation-root freeze',max(stamp(r['ended_at']) for r in self.sources.values() if r['split']=='validation'),self.original_time(self.progress/'roots-validation.json'))
        if bykind.get('comparator'):
            before('validation roots before comparator jobs',self.original_time(self.progress/'roots-validation.json'),min(x[0] for x in bykind['comparator']))
        if bykind.get('selection'):
            first=min(x[0] for x in bykind['selection']);require(len(self.comparators)==32,'scoring before all comparators')
            before('target ceiling before any selection job',self.original_time(self.progress/'target-budget.json'),first)
            before('all comparators before any selection job',max(self.original_time(self.progress/'comparators'/f'{rid}.json') for rid in self.comparators),first)
            first_comparators=initial_successful_ends('comparator');require(set(first_comparators)==set(self.comparators),'comparator lacks successful supervised completion')
            before('first successful comparator jobs exited before selection',max(first_comparators.values()),first)
        selection=[r for r in self.cell_rows if r['stage']=='selection'];validationcells=[r for r in self.cell_rows if r['stage']=='validation']
        if self.dataset is not None:before('all terminal targets before dataset freeze',max(stamp(r['ended_at']) for r in selection),self.original_time(self.run/'dataset/training.json'))
        if bykind.get('fit'):before('dataset freeze before fit worker',self.original_time(self.run/'dataset/training.json'),min(x[0] for x in bykind['fit']))
        if self.actions is not None:
            before('complete two-head fit before frozen validation actions',self.original_time(self.run/'fit/manifest.json'),self.original_time(self.progress/'validation-actions.json'))
            require(bykind.get('fit'),'actions without supervised fit');before('first successful fit worker exited before actions',min(initial_successful_ends('fit').values()),self.original_time(self.progress/'validation-actions.json'))
        if bykind.get('validation'):before('all three frozen actions before validation jobs',self.original_time(self.progress/'validation-actions.json'),min(x[0] for x in bykind['validation']))
        if bykind.get('benchmark'):
            require(self.validation['complete'],'benchmarks precede complete held validation')
            before('all held validation terminal results before benchmark schedule',max(stamp(r['ended_at']) for r in validationcells),self.original_time(self.progress/'benchmarks/specs.json'))
            first_validation=initial_successful_ends('validation');require(set(first_validation)=={r['root_id'] for r in validation['roots']},'validation lacks successful supervised completion')
            before('first successful validation workers exited before benchmark schedule',max(first_validation.values()),self.original_time(self.progress/'benchmarks/specs.json'))
            before('frozen benchmark schedule before benchmark workers',self.original_time(self.progress/'benchmarks/specs.json'),min(x[0] for x in bykind['benchmark']))
        # Attempt timestamps must lie within a matching supervised job interval.
        for row in self.cell_rows:
            intervals_for_root=[(start,end) for start,end,job in bykind.get(row['stage'],[])+inferred_intervals.get(row['stage'],[]) if job['job']['payload']['root_id']==row['root_id']]
            require(any(start<=stamp(row['started_at'])+.002 and stamp(row['ended_at'])<=end+.002 for start,end in intervals_for_root),'cell ran outside supervised root job')
        for row in self.benchmark_rows:
            intervals_for_cell=[(start,end) for start,end,job in bykind.get('benchmark',[])+inferred_intervals.get('benchmark',[]) if job['job']['payload']['cell_id']==row['cell_id']]
            require(any(start<=stamp(row['evaluation']['started_at'])+.002 and stamp(row['recorded_at'])<=end+.002 for start,end in intervals_for_cell),'benchmark ran outside supervised cell job')
        invocations=[self.read(p) for p in sorted((self.progress/'invocations').glob('*.json'))]
        require(invocations and invocations[-1]==self.status,'final invocation/status differ')
        for invocation in invocations:require(invocation['identity_sha256']==self.ih and 0<invocation['budget_seconds']<=10800 and invocation['active_jobs']==[] and invocation['runtime_and_checkpoint_unchanged'] is True and stamp(invocation['ended_at'])>=stamp(invocation['started_at']),'invalid finalized invocation')
        self.chronology={'early_observation_path':str((self.progress/'chronology-observation.json').relative_to(ROOT)),'early_observation_sha256':CHRONOLOGY_OBSERVATION_SHA,'checks':checks,'observations':self.observations,'timestamp_provenance':'Original-run file mtimes were observed with their exact artifact SHA; restored audits reuse these saved observations, never checkout mtimes. Embedded start/end journals independently establish execution order. These are provenance consistency checks, not a trusted-clock attestation.','peak_observed_completed_worker_jobs':peak,'peak_observed_completed_benchmark_jobs':benchmark_peak,'interrupted_job_attempts':self.interrupted_jobs,'recovery_timing_note':'Start-only job intervals resolved by a later successful attempt retain their raw journal; their endpoint is bounded by the next attempt start for containment only. Peak concurrency excludes these unobserved durations.','completed_job_attempts':len(self.jobs),'pending_job_attempts':self.job_pending,'invocations':invocations}

    def run_all(self,replay=True):
        self.assert_identity();self.assert_sources();training=self.assert_roots('training');validation=self.assert_roots('validation');self.assert_anchors(training,validation);self.assert_budget_comparators(training,validation);self.assert_cells(training,validation)
        self.assert_dataset_fit(training,validation,replay=replay);self.assert_validation(validation);self.assert_benchmarks();self.assert_chronology(training,validation)
        complete=bool(training and training['complete'] and validation and validation['complete'] and self.anchors is not None and self.selection_complete and self.fit is not None and self.validation['complete'] and self.benchmarks['complete'] and not self.starts and not self.source_pending and not self.job_pending)
        passes=bool(complete and self.validation['research_rule_pass'] and self.benchmarks['each_opponent_nonregression'])
        public=self.read(self.progress/'summary.json');decision=public['decision']
        require(public['identity_sha256']==self.ih and decision['all_map_victory'] is False and decision['automatic_promotion'] is False,'public identity/scope differs')
        require(decision['complete']==complete and decision['criteria_pass']==passes and decision['review_for_expansion']==passes and decision['integrity_verified'] is True,'public research decision differs from independent audit')
        require(public['selection']['completed_cells']==sum(r['stage']=='selection' for r in self.cell_rows) and public['selection']['completed_roots']==len(self.selected),'public selection completion differs')
        report={'schema_version':1,'experiment_id':'terminal-target-head-v6','audit_kind':'independent-end-to-end-evidence-and-exact-policy-head-replay','created_at':datetime.now(timezone.utc).isoformat(),'experiment_identity_sha256':self.ih,'execution_root':str(self.execution_root),'baseline_checkpoint_sha256':CP2,'evidence_integrity_verified':True,
            'discovery':{'source_games':len(self.sources),'source_terminal_proofs':self.source_rows,'cached_float32_positions':len(self.cached),'training_roots':len(training['roots']) if training else 0,'validation_roots':len(validation['roots']) if validation else 0,'anchors':len(self.anchors['anchors']) if self.anchors else 0,'gradient_unique_inputs':len(self.gradient),'gradient_input_set_sha256':fp(self.gradient),'validation_gradient_intersection':0 if self.anchors and validation else None,'source_attempt_errors':self.source_errors,'source_pending':self.source_pending,'interrupted_source_attempts':self.interrupted_source_attempts},
            'selection':{'expected_cells':len(self.selection_expected),'completed_cells':sum(r['stage']=='selection' for r in self.cell_rows),'completed_roots':len(self.selected),'complete':self.selection_complete,'selected_actions':self.selected,'puct_comparators':len(self.comparators),'failed_attempts':sum(r['stage']=='selection' for r in self.failed_attempts),'pending_attempts':[r for r in self.starts if r['stage']=='selection'],'interrupted_attempt_history':self.interrupted_cell_attempts,'collection_ceiling':1600},
            'dataset':self.dataset,'fit':self.fit_audit,'fit_recovery_artifacts':self.pending_fit,'validation':self.validation,'benchmarks':self.benchmarks,'chronology':self.chronology,
            'decision':{'complete':complete,'criteria_pass':passes,'review_for_expansion':passes,'all_map_victory':False,'automatic_promotion':False,'scope':'Preregistered small research experiment only; neither 74-map qualification nor model promotion.'},
            'limitations':['No new Java games or engine replay were run by this auditor. Recorded terminal world, action transcript, frozen source, root prefix and model-input hashes are cross-checked; engine authenticity and first-eligible-turn extraction and the underlying streaming reservoir sampling remain dependent on the audited frozen engine/collector implementation.','Thirty-two target roots and sixteen held roots form a small paired mechanism experiment. The fresh benchmark panel contains one map and seed, two seats per opponent; it cannot establish all-map superiority.','The frozen v6 actor scanner can skip matching processes when /proc cwd is unreadable across sandbox namespaces. This standalone final audit instead blocks matching commands conservatively on cwd read failure, holds all four exclusion locks, and never relies on the frozen scanner for final quiescence.','Independent SGD replay operates only on in-memory copies of cached features and frozen baseline heads; it produces no saved training state or promoted candidate.'],
            'input_sha256':dict(sorted(self.tracked.items()))}
        return report


def exclusive_write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.link(tmp,path)
        directory=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--final',action='store_true');parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args()
    ROOT=args.root.resolve();executing_source=Path(__file__).read_bytes()
    with ExitStack() as stack:
        locks=[]
        for name in LOCKS:
            path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True);handle=stack.enter_context(path.open('a+'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append({'path':name,'inode':path.stat().st_ino,'device':path.stat().st_dev})
        require(not actor_state(),'active supervisor/worker/EngineBridge exists')
        audit=Audit();require(not args.final or not audit.output.exists(),'immutable independent final audit already exists')
        report=audit.run_all(replay=True)
        require(not actor_state(),'actor appeared during audit')
        for name,digest in audit.tracked.items():require(sha(audit.path(name))==digest,'input changed during audit: '+name)
        for row in locks:require((ROOT/row['path']).stat().st_ino==row['inode'] and (ROOT/row['path']).stat().st_dev==row['device'],'lock file replaced while held')
        source=Path(__file__).read_bytes();require(source==executing_source,'auditor source changed during verification');report['auditor_source_sha256']=hashlib.sha256(source).hexdigest();report['quiescence']={'verified':True,'locks_held':locks,'pid':os.getpid(),'no_matching_actors_or_orphan_engine_processes':True,'unreadable_cwd_policy':'Matching actor or EngineBridge commands block conservatively when /proc cwd cannot be read.'}
        if args.final:
            destination=audit.progress/'source/independent_final_audit.py'
            if destination.exists():require(destination.read_bytes()==source,'existing auditor archive differs')
            else:exclusive_write(destination,source)
            report['auditor_source_path']=str(destination.relative_to(ROOT));exclusive_write(audit.output,(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
        print(json.dumps({'published':args.final,'audit_path':str(audit.output) if args.final else None,'identity':audit.ih,'decision':report['decision'],'selection':{k:v for k,v in report['selection'].items() if k not in ('selected_actions','pending_attempts')},'validation':{k:v for k,v in report['validation'].items() if k!='roots'},'benchmarks':{k:v for k,v in report['benchmarks'].items() if k!='opponents'},'source_sha256':report['auditor_source_sha256']},sort_keys=True,indent=2))


if __name__=='__main__':main()
