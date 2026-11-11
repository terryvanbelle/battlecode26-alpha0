"""Fresh V11 validation discovery, no targets/SGD/benchmark games."""
import argparse,ctypes,fcntl,importlib.util,json,os,signal,subprocess,sys,time,types
from pathlib import Path
from contextlib import ExitStack
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alpha0 import terminal_v8_common as C
from alpha0.terminal_v8 import active_actors
R=C.ROOT/'runs/validation-v11';P=C.ROOT/'progress/validation-v11';CFG=C.ROOT/'configs/validation-v11.json'
from scripts import collect_general_roots_v11 as ORIGINAL
LOCKS=(*ORIGINAL.LOCKS,R/'experiment.lock')
STOPS=(*ORIGINAL.STOPS,R/'STOP')
def read(p):return json.loads(p.read_text())
def check(deadline):
    if any(p.exists() for p in STOPS) or time.monotonic()>=deadline:raise C.Interrupted('STOP or deadline')
def jobs(cfg):
    return [{'source_id':f'validation-m{m}-j{j}','split':'validation','map_index':m,'j':j,'map':name,
             'seed':20267500+100*m+j,'policy_seeds':[2*(470000+100*m+j)+t for t in (0,1)]}
            for m,name in enumerate(cfg['maps']) for j in range(1,7)]
def source_spec(job,cfg,identity):
    expected=next((x for x in jobs(cfg) if x['source_id']==job['source_id']),None)
    if expected is None or any(job.get(k)!=v for k,v in expected.items()) or job.get('exclusion_map',{})!=exclusions():raise ValueError('source outside frozen schedule')
    if identity['config']!=cfg:raise ValueError('config changed')
    return {'experiment_identity':C.fingerprint(identity),**expected,'checkpoint_sha256':cfg['checkpoint_sha256'],
        'temperature':1.0,'source_temperature':1.0,'exclusion_map_sha256':C.fingerprint(exclusions()),'anchor_seed':cfg['anchor_seed']}
def exclusions():
    record=read(C.ROOT/'configs/validation-v11-exclusions.json')
    for name,digest in record['pins'].items():assert C.sha256(C.ROOT/name)==digest
    return record['input_exclusions']
def identity():
    cfg=read(CFG);files=(Path(__file__).resolve(),CFG,C.ROOT/'docs/validation-v11.md',C.ROOT/'alpha0/terminal_v8_collect.py',C.ROOT/'configs/validation-v11-exclusions.json',C.ROOT/'diagnostics/validation-v11-planning/seed-inventory.json')
    plan=read(ORIGINAL.P/'training-input-plan.json');assert plan['complete'] and plan['budget_pass']
    exclusions()
    return {'config':cfg,'v8_runtime':C.identity(C.load_config()),'training_plan_sha256':C.sha256(ORIGINAL.P/'training-input-plan.json'),'files':{str(p.relative_to(C.ROOT)):C.sha256(p) for p in files}}
def backend():
    path=C.ROOT/'alpha0/terminal_v8_collect.py';text=path.read_text()
    old='candidate = key not in candidates and 37 in state.legal_actions'
    assert text.count(old)==1
    text=text.replace(old,'candidate = key not in candidates and len(state.legal_actions) >= 2')
    module=types.ModuleType('alpha0._v11_replay_backend');module.__file__=str(path)
    exec(compile(text,str(path), 'exec'),module.__dict__)
    namespace=types.SimpleNamespace(**vars(C));namespace.RUN=R;namespace.PROGRESS=P;namespace.check_stop=check
    module.C=namespace;module.source_spec=source_spec
    return module

def source_row(job,expected):
    path=R/'sources'/f"{job['source_id']}.json"
    job={**job,'exclusion_map':exclusions()}
    if path.exists():return backend()._verify_source(read(path),source_spec(job,expected['config'],expected)),path
    return None,path

def report(expected):
    rows=[]
    for job in jobs(expected['config']):
        row,path=source_row(job,expected)
        if row:
            rows.append({'source_id':job['source_id'],'map':job['map'],'root_roles':sorted(row['root_candidates']),
                'anchor_counts':{k:len(v) for k,v in row['anchor_candidates'].items()},'source_json':str(path.relative_to(C.ROOT)),
                'source_sha256':C.sha256(path),'source_experiment_identity':row['experiment_identity']})
    return {'complete':len(rows)==48,'completed_sources':len(rows),'expected_sources':48,'fresh_validation_source_games':len(rows),'sources':rows,'sgd_updates':0,'target_continuations':0,'benchmark_games':0}
def worker(source_id,deadline,parent):
    import torch
    torch.set_num_threads(1);ctypes.CDLL(None).prctl(1,signal.SIGKILL)
    if os.getppid()!=parent:raise RuntimeError('parent changed')
    def interrupted(*_):raise C.Interrupted('supervisor interrupt')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    expected=read(P/'identity.json');assert identity()==expected
    job=next(x for x in jobs(expected['config']) if x['source_id']==source_id)
    job['worker_task_path']=str(R/'tasks'/f'{source_id}.json')
    job['exclusion_map']=exclusions()
    backend().source_game(job,expected['config'],expected,deadline)
    source_row(job,expected)
    assert identity()==expected

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--seconds',type=float,default=10800);a.add_argument('--worker');a.add_argument('--deadline',type=float);a.add_argument('--parent',type=int);a.add_argument('--preflight',action='store_true');args=a.parse_args()
    if args.worker:return worker(args.worker,args.deadline,args.parent)
    assert 0<args.seconds<=10800
    for p in (R,P,R/'tasks',R/'logs'):p.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for p in LOCKS:
            h=stack.enter_context(p.open('a+'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check(time.monotonic()+1);assert not active_actors()
        expected=identity();assert expected['v8_runtime']==read(C.PROGRESS/'identity.json')
        prior=read(C.ROOT/'progress/target-recheck-v9r2/independent-final-audit.json');assert prior['complete'] and prior['review_for_expansion']
        C.immutable_json(P/'identity.json',expected);C.immutable_json(P/'schedule.json',jobs(expected['config']))
        if args.preflight:
            loaded=backend();assert loaded.C.RUN==R and loaded.C.PROGRESS==P and callable(loaded.source_game);report(expected);print('preflight passed');return
        deadline=time.monotonic()+args.seconds;live={}
        def status(phase,**kw):C.atomic_json(R/'status.json',{'phase':phase,'pid':os.getpid(),'updated_at':C.now(),'identity':C.fingerprint(expected),**kw})
        pending=[]
        for job in jobs(expected['config']):
            row,path=source_row(job,expected)
            if row is not None:continue
            if list((R/'sources'/f"{job['source_id']}.attempts").glob('attempt-*.json')):raise RuntimeError('failed/interrupted source requires review; no silent retry')
            pending.append(job)
        status('collecting_sources')
        try:
            while pending or live:
                check(deadline)
                while pending and len(live)<2:
                    job=pending.pop(0);name=job['source_id'];log=stack.enter_context((R/'logs'/f'{name}.log').open('a'))
                    proc=subprocess.Popen([sys.executable,'-u',__file__,'--worker',name,'--deadline',str(deadline),'--parent',str(os.getpid())],cwd=C.ROOT,start_new_session=True,stdout=log,stderr=log)
                    live[name]=(proc,time.monotonic())
                for name,(proc,started) in list(live.items()):
                    if time.monotonic()>started+1830:raise RuntimeError('source watchdog deadline')
                    if proc.poll() is not None:
                        try:os.killpg(proc.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        del live[name]
                        if proc.returncode:raise RuntimeError(f'{name} worker failed; review required')
                        assert identity()==expected
                        measured=report(expected);C.atomic_json(P/'summary.json',measured);status('collecting_sources',completed_sources=measured['completed_sources'])
                time.sleep(.5)
            measured=report(expected);assert measured['complete'];C.atomic_json(P/'summary.json',measured);status('complete_review_required',completed_sources=48)
        except BaseException as e:status('review_required',reason=f'{type(e).__name__}: {e}');raise
        finally:
            for proc,_ in live.values():
                if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
            for proc,_ in live.values():
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
if __name__=='__main__':main()
