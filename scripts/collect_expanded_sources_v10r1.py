"""V10 phase A: fixed 64-game own-self-play source corpus; no targets or SGD."""
import argparse,ctypes,fcntl,importlib.util,json,os,signal,subprocess,sys,time,types
from pathlib import Path
from contextlib import ExitStack
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alpha0 import terminal_v8_common as C
from alpha0.terminal_v8 import active_actors
R=C.ROOT/'runs/expanded-corpus-v10r1';P=C.ROOT/'progress/expanded-corpus-v10r1';CFG=C.ROOT/'configs/expanded-corpus-v10r1.json'
LOCKS=(*C.LOCKS,C.ROOT/'runs/target-recheck-v9r2/experiment.lock',C.ROOT/'runs/expanded-corpus-v10/experiment.lock',R/'experiment.lock')
STOPS=(*C.STOPS,C.ROOT/'runs/target-recheck-v9r2/STOP',C.ROOT/'runs/expanded-corpus-v10/STOP',R/'STOP')
def read(p):return json.loads(p.read_text())
def check(deadline):
    if any(p.exists() for p in STOPS) or time.monotonic()>=deadline:raise C.Interrupted('STOP or deadline')
def jobs(cfg):
    return [{'source_id':f'training-m{m}-j{j}','split':'training','map_index':m,'j':j,'map':name,
             'seed':20265200+100*m+j,'policy_seeds':[2*(430000+100*m+j)+t for t in (0,1)]}
            for m,name in enumerate(cfg['maps']) for j in range(1,9)]
def source_spec(job,cfg,identity):
    expected=next((x for x in jobs(cfg) if x['source_id']==job['source_id']),None)
    if expected is None or any(job.get(k)!=v for k,v in expected.items()) or job.get('exclusion_map',{}):raise ValueError('source outside frozen schedule')
    if identity['config']!=cfg:raise ValueError('config changed')
    return {'experiment_identity':C.fingerprint(identity),**expected,'checkpoint_sha256':cfg['checkpoint_sha256'],
        'temperature':1.0,'source_temperature':1.0,'exclusion_map_sha256':C.fingerprint({}),'anchor_seed':cfg['anchor_seed']}
def identity():
    cfg=read(CFG);files=(Path(__file__).resolve(),CFG,C.ROOT/'docs/expanded-corpus-v10r1.md',C.ROOT/'alpha0/terminal_v8_collect.py',C.ROOT/'configs/expanded-corpus-v10r1-inherited.json',C.ROOT/'scripts/collect_expanded_sources_v10.py')
    return {'config':cfg,'v8_runtime':C.identity(C.load_config()),'files':{str(p.relative_to(C.ROOT)):C.sha256(p) for p in files}}
def backend():
    spec=importlib.util.spec_from_file_location('alpha0._v10_source_backend',C.ROOT/'alpha0/terminal_v8_collect.py',C.ROOT/'configs/expanded-corpus-v10r1-inherited.json',C.ROOT/'scripts/collect_expanded_sources_v10.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    namespace=types.SimpleNamespace(**vars(C));namespace.RUN=R;namespace.PROGRESS=P;namespace.check_stop=check
    module.C=namespace;module.source_spec=source_spec
    return module

def source_row(job,expected):
    inherited=read(C.ROOT/'configs/expanded-corpus-v10r1-inherited.json')
    pin=inherited['sources'].get(job['source_id'])
    path=R/'sources'/f"{job['source_id']}.json"
    if pin:
        assert not path.exists(),'completed source rerun'
        from scripts import collect_expanded_sources_v10 as old
        origin_path=C.ROOT/inherited['original_identity_path'];assert C.sha256(origin_path)==inherited['original_identity_sha256'];origin=read(origin_path)
        path=C.ROOT/pin['path'];assert C.sha256(path)==pin['sha256']
        row=old.backend()._verify_source(read(path),old.source_spec(job,origin['config'],origin))
        return row,path
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
    return {'complete':len(rows)==64,'completed_sources':len(rows),'expected_sources':64,'inherited_sources':38,'sources':rows,'sgd_updates':0,'target_continuations':0,'benchmark_games':0}
def worker(source_id,deadline,parent):
    if source_id in read(C.ROOT/'configs/expanded-corpus-v10r1-inherited.json')['sources']:raise RuntimeError('completed inherited source cannot rerun')
    import torch
    torch.set_num_threads(1);ctypes.CDLL(None).prctl(1,signal.SIGKILL)
    if os.getppid()!=parent:raise RuntimeError('parent changed')
    def interrupted(*_):raise C.Interrupted('supervisor interrupt')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    expected=read(P/'identity.json');assert identity()==expected
    job=next(x for x in jobs(expected['config']) if x['source_id']==source_id)
    job['worker_task_path']=str(R/'tasks'/f'{source_id}.json')
    backend().source_game(job,expected['config'],expected,deadline)
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
        if args.preflight:print('preflight passed');return
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
            measured=report(expected);assert measured['complete'];C.atomic_json(P/'summary.json',measured);status('complete_review_required',completed_sources=64)
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
