"""Independent read-only coverage audit; no experiment module imports or games."""
from pathlib import Path
from datetime import datetime, timezone
from contextlib import ExitStack
from collections import Counter
import hashlib,json,fcntl,os,tempfile
import numpy as np
ROOT=Path('/workspace/battlecode26-alpha0')
RUN=ROOT/'runs/terminal-target-head-v5'; OUT=ROOT/'progress/terminal-target-head-v5'
tracked={}
def req(condition,message):
 if not condition: raise AssertionError(message)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def track(p):
 p=Path(p); tracked[str(p.relative_to(ROOT))]=sha(p); return tracked[str(p.relative_to(ROOT))]
def read(p): track(p); return json.loads(Path(p).read_text())
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def input_hash(arrays):
 fields=[]; payload=[]
 for name,a in zip(('local','global_board','scalars','legal_mask'),arrays):
  a=np.array(a,dtype='u1' if name=='legal_mask' else '<f4',order='C',copy=True)
  req(np.isfinite(a).all(),'nonfinite input')
  if name!='legal_mask': a[a==0]=0
  fields.append({'name':name,'shape':list(a.shape),'dtype':'|u1' if name=='legal_mask' else '<f4'})
  payload.append(a.tobytes(order='C'))
 header=json.dumps({'schema':'gradient-input-v1','feature_version':2,'action_version':'macro_v2','fields':fields},sort_keys=True,separators=(',',':')).encode()
 return hashlib.sha256(len(header).to_bytes(8,'little')+header+b''.join(payload)).hexdigest()
def actors():
 found=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid(): continue
  try:
   args=p.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')[:-1]
   if p.joinpath('cwd').resolve()!=ROOT: continue
   if any(a in {'alpha0.terminal_v5','alpha0.EngineBridge','alpha0.campaign','alpha0.learn_v2','alpha0.root_ranking_v4','alpha0.evaluate','alpha0.victory'} or Path(a).name=='run_warmstart_experiment.py' for a in args): found.append({'pid':int(p.name),'command':args})
  except (OSError,ProcessLookupError): pass
 return found
with ExitStack() as stack:
 lockpaths=['runs/campaign/campaign.lock','runs/warmstart-v3/experiment.lock','runs/terminal-target-head-v5/experiment.lock']
 for name in lockpaths:
  stream=stack.enter_context((ROOT/name).open('a+')); fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
 req(not actors(),'actors still present')
 identity=read(OUT/'identity.json'); config=read(OUT/'config.json'); roots=read(OUT/'roots-training.json'); summary=read(OUT/'summary.json'); status=read(RUN/'status.json')
 ih=digest(identity)
 req(identity['config']==config and status['identity_sha256']==ih==roots['identity_sha256'],'experiment identity differs')
 req(status['phase']=='review_required' and status['reason']=='training root coverage incomplete; no SGD','unexpected outcome')
 req(status['runtime_and_checkpoint_unchanged'] is True and status['active_jobs']==[],'bad final integrity/status')
 req(sha(ROOT/'configs/terminal-target-head-v5.json')==identity['config_file_sha256'],'config changed')
 req(track(ROOT/config['checkpoint'])==identity['checkpoint_sha256']==config['checkpoint_sha256'],'cp2 changed')
 for name,h in identity['sources'].items(): req(track(ROOT/name)==h,f'source changed {name}')
 for name,h in identity['runtime']['files'].items(): req(track(ROOT/name)==h,f'runtime changed {name}')
 compiled={str(p.relative_to(ROOT)):track(p) for p in sorted((ROOT/'build/java').rglob('*.class'))}
 req(compiled==identity['runtime']['compiled_classes'],'compiled classes changed')
 req(track(ROOT/'vendor/jdk/release')==identity['runtime']['jdk_release_sha256'],'jdk changed')
 for m in identity['maps']: req(track(ROOT/'maps'/f"{m['name']}.map26")==m['sha256'],'map changed')
 for name,k in [('configs/victory_gate.json','gate_sha256'),('configs/benchmarks.json','panel_sha256'),('progress/root-ranking-v4/independent-final-audit.json','v4_audit_sha256')]: req(track(ROOT/name)==identity[k],name+' changed')
 import importlib.metadata
 req(importlib.metadata.version('numpy')==identity['runtime']['numpy_version'],'numpy changed')
 req(importlib.metadata.version('torch')==identity['runtime']['torch_version'],'torch changed')
 expectedids={f'training-m{m}-j{j}' for m in range(4) for j in range(1,5)}|{f'training-m2-j{j}' for j in range(5,9)}
 paths=sorted((RUN/'sources').glob('*.json')); req({p.stem for p in paths}==expectedids,'wrong source inventory')
 sources={}; observations=[]; caches=0
 for path in paths:
  r=read(path); sid=path.stem; sources[sid]=r; m,j=r['map_index'],r['j']
  req(r['source_id']==sid==f'training-m{m}-j{j}' and r['map']==config['maps'][m] and r['seed']==20261200+100*m+j,'source schedule differs')
  req(r['policy_seeds']==[2*(310000+100*m+j)+t for t in (0,1)],'source policy streams differ')
  req(r['status']=='ok' and r['complete'] is True and r['source']=='selfplay' and r['split']=='training','source incomplete/not own selfplay')
  req(r['experiment_identity']==ih and r['checkpoint_sha256']==config['checkpoint_sha256'] and r['temperature']==.35,'source identity/temp differs')
  spec={k:r[k] for k in ('experiment_identity','source_id','split','map_index','j','map','seed','policy_seeds','checkpoint_sha256','temperature','exclusion_map_sha256','anchor_seed')}
  req(digest(spec)==r['identity'] and r['exclusion_map_sha256']==digest({}),'source specification hash differs')
  req(r['terminal'] is True and r['valid_game'] is True and r['benchmark_failures']==[] and r['score'] is None,'source terminal flags differ')
  world=r['terminal_state']; req(world['terminal'] is True and world['winner']==r['winner'] and type(r['winner']) is int and r['winner'] in (-1,0,1),'terminal winner differs')
  req(world['round']==r['terminal_round']>0 and world['seed']==r['seed'] and world['map']==r['map'],'terminal round/map/seed differs')
  req(world.get('valid_game',True) is True and world.get('benchmark_failures',[])==[] and digest(world)==r['terminal_state_sha256'],'terminal proof hash/validity differs')
  req(r['terminal_summary']=={k:world.get(k) for k in ('terminal','round','winner','win_reason','bank','kings','transferred','cat_damage')},'terminal summary differs')
  actions=r['actions']; req(len(actions)==r['decisions']>0 and all(type(a)is int and 0<=a<40 for a in actions) and digest(actions)==r['actions_sha256'],'transcript differs')
  start=datetime.fromisoformat(r['started_at']); end=datetime.fromisoformat(r['ended_at']); req(end>=start,'source end before start')
  attempts=list(path.parent.joinpath(sid+'.attempts').glob('attempt-[0-9][0-9][0-9].json')); req(len(attempts)==1 and read(attempts[0])==r,'source attempt/final differ')
  starts=list(path.parent.joinpath(sid+'.attempts').glob('*-start.json')); req(len(starts)==1,'source start count differs'); sr=read(starts[0]); req(sr['started_at']==r['started_at'] and sr['identity']==r['identity'] and sr['attempt']==r['attempt']==1,'start identity differs')
  cache=ROOT/r['cache_npz']; req(track(cache)==r['cache_sha256'],'cache bytes differ')
  with np.load(cache,allow_pickle=False) as z:
   req(set(z.files)=={'hidden','prior','mask','local','global_board','scalars'},'unexpected cache fields')
   count=len(r['cache_positions']); shapes={'hidden':(128,),'prior':(40,),'mask':(40,),'local':(33,17,17),'global_board':(33,60,60),'scalars':(64,)}
   arrays={k:z[k] for k in z.files}
   for k,shape in shapes.items(): req(arrays[k].shape==(count,*shape) and arrays[k].dtype==('uint8' if k=='mask' else 'float32') and np.isfinite(arrays[k]).all(),f'bad cache array {k}')
   for index,p in enumerate(r['cache_positions']):
    req(p['feature_index']==index and p['source_id']==sid and 0<=p['decision_index']<len(actions),'bad cache reference')
    req(p['prefix_sha256']==digest(actions[:p['decision_index']]),'cached prefix differs')
    mask=np.zeros(40,dtype=np.uint8);mask[p['legal_actions']]=1
    req(np.array_equal(mask,arrays['mask'][index]),'cached legality differs')
    prior=arrays['prior'][index];req(np.all(prior>=0) and np.all(prior[mask==0]==0) and abs(float(prior.sum(dtype=np.float64))-1)<1e-6,'cached prior invalid')
    req(input_hash([arrays[k][index] for k in ('local','global_board','scalars','mask')])==p['input_hash'],'canonical float32 input hash differs')
   for role,p in r['root_candidates'].items():
    req(role==f"{p['role']}:{p['player']}" and 37 in p['legal_actions'] and p==r['cache_positions'][p['feature_index']],'candidate/cache mismatch')
   for team in (0,1):
    anchors=r['anchor_candidates'][str(team)]
    req(len(anchors)==16 and len({a['decision_index'] for a in anchors})==16 and all(a['player']==team and a==r['cache_positions'][a['feature_index']] for a in anchors),'source anchor reservoir mismatch')
    req(r['anchor_reservoir']['rng_seeds'][str(team)]==int(digest({'anchor_seed':203151,'source_id':sid,'player':team})[:16],16),'reservoir seed differs')
   caches+=count
  c=Counter(actions)
  observations.append({'source_id':sid,'map':r['map'],'seed':r['seed'],'terminal_round':r['terminal_round'],'winner':r['winner'],'terminal_reason':world['win_reason'],'decisions':len(actions),'macro21_count':c[21],'candidate_role_teams':sorted(r['root_candidates']),'source_sha256':tracked[str(path.relative_to(ROOT))],'cache_sha256':r['cache_sha256']})
 # Independently replay only the deterministic allocation logic over recorded candidates.
 expectedroots=[]; assignments=[]; unavailable=[]; usedsources=set()
 for m in range(4):
  reserves_used=set()
  for j in range(1,5):
   roles=[('BABY_RAT',0 if j%2 else 1),('RAT_KING',1 if j%2 else 0)]; missing=list(roles); chosen={}
   for n in [j]+[n for n in range(5,9) if n%2==j%2]:
    if n!=j and n in reserves_used: continue
    if n!=j:reserves_used.add(n)
    sid=f'training-m{m}-j{n}'; req(sid in sources,'required source missing'); usedsources.add(sid); filled=[]
    for role,team in list(missing):
     p=sources[sid]['root_candidates'].get(f'{role}:{team}')
     if p is not None: chosen[(role,team)]=(sid,p);missing.remove((role,team));filled.append(f'{role}:{team}')
    assignments.append({'map_index':m,'slot_group_j':j,'source_j':n,'source_id':sid,'filled':filled})
    if not missing:break
   for role,team in roles:
    if (role,team) in chosen: expectedroots.append((m,j,*chosen[(role,team)]))
    else:unavailable.append({'map_index':m,'j':j,'role':role,'player':team})
 req(assignments==roots['assignments'] and unavailable==roots['unavailable_slots'],'reserve allocation differs')
 req(len(expectedroots)==len(roots['roots'])==24 and len(unavailable)==8 and roots['complete'] is False,'coverage differs')
 req(usedsources==expectedids==set(roots['sources']),'discovery source inventory differs')
 for ordinal,((m,j,sid,p),r) in enumerate(zip(expectedroots,roots['roots'])):
  req(all(r[k]==v for k,v in p.items()) and r['root_id']==f'training-{ordinal:03d}' and r['root_ordinal']==ordinal and r['map_index']==m and r['slot_group_j']==j,'selected root differs')
  req(r['source_json']==sources[sid]['source_json'] and r['source_sha256']==tracked[r['source_json']] and r['cache_sha256']==sources[sid]['cache_sha256'],'root provenance differs')
 req({r['map']:sum(x['map']==r['map'] for x in roots['roots']) for r in roots['roots']}=={'tiny':8,'DefaultSmall':8,'knifefight':8},'map counts differ')
 forbidden=['anchors.json','roots-validation.json','validation-actions.json','target-budget.json','fit-summary.json']
 req(not any((OUT/name).exists() for name in forbidden),'post-coverage artifact exists')
 req(not any(p.is_file() for base in (OUT/'cells',OUT/'comparators',OUT/'benchmarks',RUN/'dataset',RUN/'fit',RUN/'fit.pending') for p in base.rglob('*')),'target/scoring/fitting artifacts exist')
 req(summary['collection']['training']['completed_source_games']==20 and summary['selection']['frozen_roots']==24,'summary source/root counts differ')
 req(summary['selection']['completed_cells']==summary['validation']['completed_unique_cells']==summary['benchmarks']['completed']==0 and summary['anchors']['completed']==0 and summary['fit']['complete'] is False,'summary claims later work')
 req(summary['decision']['complete'] is False and summary['decision']['review_for_expansion'] is False and summary['decision']['automatic_promotion'] is False and summary['decision']['all_map_victory'] is False,'incorrect decision')
 req(not actors(),'actor appeared during audit')
 for name,h in tracked.items():req(sha(ROOT/name)==h,'file changed during audit: '+name)
 report={'audit_id':'terminal-target-head-v5-independent-coverage-v1','audited_at':datetime.now(timezone.utc).isoformat(),'experiment_identity_sha256':ih,'quiescent':True,'exclusion_locks_held':lockpaths,'sources_completed':20,'source_attempt_errors':0,'source_terminal_proofs_verified':20,'cached_float32_positions_verified':caches,'training_roots_required':32,'training_roots_frozen':24,'missing_slots':unavailable,'root_source_games':len({r['source_id'] for r in roots['roots']}),'terminal_target_cells':0,'validation_cells':0,'benchmark_games':0,'gradient_updates':0,'cp2_unchanged':True,'frozen_runtime_unchanged':True,'conclusion':'Valid source collection; fixed coverage rule failed entirely on the eight assigned closeup slots. No anchors, terminal targets, validation, benchmarks or SGD occurred. A separately versioned research amendment is required to change collection.','sources':observations,'input_sha256':tracked,'limitations':['No engine replay or neural inference was performed. Recorded terminal-world/transcript proofs, immutable source attempts, lossless cached model inputs and all source/runtime hashes were independently checked.','First qualifying-turn and reservoir-uniformity claims depend on the frozen collector implementation; this audit verifies recorded candidate/prefix/feature consistency and independently recomputes reserve assignment.'],'auditor_source_sha256':sha(__file__)}
 destination=OUT/'independent-coverage-audit.json';req(not destination.exists(),'immutable audit already exists')
 fd,temp=tempfile.mkstemp(dir=OUT,prefix='.independent-coverage-')
 try:
  with os.fdopen(fd,'w') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.link(temp,destination)
 finally:Path(temp).unlink(missing_ok=True)
 print(json.dumps({'audit_path':str(destination.relative_to(ROOT)),'audit_sha256':sha(destination),'sources':20,'positions':caches,'roots':24,'updates':0},indent=2))
