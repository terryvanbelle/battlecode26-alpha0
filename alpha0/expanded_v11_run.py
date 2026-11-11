"""Bounded V11 research supervisor with immutable stage proofs."""
import argparse,ctypes,fcntl,json,os,signal,subprocess,sys,time
from contextlib import ExitStack
from pathlib import Path
import torch
from scripts import collect_validation_v11 as V
from alpha0 import terminal_v8_common as C
from alpha0.expanded_v11_support import frozen_plans,load_root_backend
from alpha0.expanded_v11_fit import fit_pair,freeze_validation_actions
from alpha0.terminal_v8_development import build_benchmark_specs,summarize_benchmarks
from alpha0.terminal_v8_report import record_benchmark_attempt,valid_benchmark_result
R=C.ROOT/'runs/expanded-terminal-v11';P=C.ROOT/'progress/expanded-terminal-v11';CFG=C.ROOT/'configs/expanded-terminal-v11.json'
LOCKS=(*V.LOCKS,R/'experiment.lock');STOPS=(*V.STOPS,R/'STOP')
def read(p):return json.loads(Path(p).read_text())
def check(deadline):
 if any(p.exists() for p in STOPS) or time.monotonic()>=deadline:raise C.Interrupted('STOP or deadline')
def identity():
 cfg=read(CFG)
 base=C.identity(C.load_config());base['config']=cfg;base['experiment_id']=cfg['experiment_id']
 paths={'training':'progress/general-roots-v11/training-input-plan.json','validation':'progress/validation-v11/validation-input-plan.json','exclusions':'configs/validation-v11-exclusions.json'}
 base['input_plans']={k:{'path':v,'sha256':C.sha256(C.ROOT/v)} for k,v in paths.items()}
 base['collector_sha256']=C.sha256(C.ROOT/'alpha0/terminal_v8_collect.py')
 for p in ('alpha0/expanded_v11_support.py','alpha0/expanded_v11_fit.py','alpha0/expanded_v11_run.py','configs/expanded-terminal-v11.json','docs/expanded-terminal-v11.md','diagnostics/expanded-v11-planning/seed-inventory.json','diagnostics/expanded-v11-planning/root-reconstruction.json','scripts/report_expanded_v11.py','scripts/export_expanded_v11.py'):
  base['sources'][p]=C.sha256(C.ROOT/p)
 frozen_plans(base)
 proof=read(C.ROOT/'diagnostics/expanded-v11-planning/root-reconstruction.json')
 if not proof['passed'] or proof['verified']!=96 or proof['identity']['adapter_sha256']!=C.sha256(C.ROOT/'alpha0/expanded_v11_support.py'):raise ValueError('physical root reconstruction proof changed')
 return base

def worker(job,expected,deadline,parent):
 ctypes.CDLL(None).prctl(1,signal.SIGKILL)
 if os.getppid()!=parent:raise RuntimeError('parent disappeared')
 torch.set_num_threads(1)
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(C.Interrupted('SIGTERM')))
 assert identity()==expected
 kind=job['kind'];cfg=expected['config'];task=R/'tasks'/f"{job['id']}.json"
 if kind in ('comparator','selection','validation'):
  backend=load_root_backend(R,P,check,expected)
  result=backend.root_job({**job['payload'],'worker_task_path':str(task)},kind,cfg,expected,deadline)
  if kind!='comparator' and not result['complete']:raise RuntimeError('root stage incomplete')
 elif kind=='fit':
  C.set_task(task,'fit',min(900,max(.01,deadline-time.monotonic())))
  fit_pair(R,P,expected,lambda:check(deadline));C.clear_task(task)
 elif kind=='benchmark':
  spec=job['payload'];directory=P/'benchmarks/cells'/spec['cell_id'];old=list(directory.glob('attempt-[0-9][0-9][0-9].json'))
  if old:
   if len(old)!=1 or not valid_benchmark_result(spec,read(old[0])):raise RuntimeError('benchmark error requires explicit review')
  else:
   C.set_task(task,'benchmark',min(cfg['evaluation_timeout_seconds']+30,max(.01,deadline-time.monotonic())))
   result=record_benchmark_attempt(spec,directory,attempt=1,max_attempts=1);C.clear_task(task)
   if not valid_benchmark_result(spec,result):raise RuntimeError('benchmark failure requires review')
 else:raise ValueError('unknown stage')
 assert identity()==expected

def job_outputs(job):
 kind,payload=job['kind'],job['payload']
 if kind=='comparator':files=[P/'comparators'/f"{payload['root_id']}.json"]
 elif kind in ('selection','validation'):
  files=[p for d in (P/'cells').glob(f"{kind}-{payload['root_id']}-a*-s*") for p in d.rglob('*') if p.is_file()]
 elif kind=='fit':files=[p for p in (R/'fit').rglob('*') if p.is_file()]
 elif kind=='benchmark':files=[p for p in (P/'benchmarks/cells'/payload['cell_id']).rglob('*') if p.is_file()]
 else:raise ValueError('unknown output stage')
 if not files or any(not p.exists() for p in files):raise ValueError('completed job lacks output proof')
 return {str(p.relative_to(C.ROOT)):C.sha256(p) for p in sorted(files)}

def run_jobs(kind,payloads,expected,deadline,workers=2):
 pending=[];live={}
 for payload in payloads:
  key=C.fingerprint({'kind':kind,'payload':payload,'identity':C.fingerprint(expected)})
  job={'id':key,'kind':kind,'payload':payload};path=R/'jobs'/f'{key}.json';done=P/'jobs'/f'{key}.json'
  C.immutable_json(path,job)
  if done.exists():
   result=read(done)
   if result['job_sha256']!=C.sha256(path) or result['status']!='ok':raise RuntimeError('saved job failed or changed; explicit review required')
   if result.get('output_hashes')!=job_outputs(job):raise RuntimeError('completed job outputs changed')
   continue
  if (R/'logs'/f'{key}.log').exists():raise RuntimeError('unclosed job requires review')
  pending.append((job,path,done))
 try:
  while pending or live:
   check(deadline)
   while pending and len(live)<workers:
    job,path,done=pending.pop(0);log=(R/'logs'/f"{job['id']}.log").open('x')
    proc=subprocess.Popen([sys.executable,'-u','-m','alpha0.expanded_v11_run','--worker',str(path),'--deadline',str(deadline),'--parent',str(os.getpid())],cwd=C.ROOT,stdout=log,stderr=log,start_new_session=True)
    live[proc.pid]=(proc,log,job,path,done,time.monotonic())
   for pid,(proc,log,job,path,done,started) in list(live.items()):
    task=R/'tasks'/f"{job['id']}.json"
    if task.exists() and time.monotonic()>read(task)['deadline_monotonic']+5:raise RuntimeError('task watchdog expired')
    if time.monotonic()>started+3600:raise RuntimeError('job watchdog expired')
    if proc.poll() is None:continue
    log.close();del live[pid]
    try:os.killpg(pid,signal.SIGKILL)
    except ProcessLookupError:pass
    result={'job_sha256':C.sha256(path),'status':'ok' if proc.returncode==0 else 'error','returncode':proc.returncode,'completed_at':C.now()}
    if proc.returncode==0:result['output_hashes']=job_outputs(job)
    C.immutable_json(done,result)
    if proc.returncode:raise RuntimeError('worker failed; review required')
   C.atomic_json(R/'status.json',{'phase':kind,'pending_jobs':len(pending),'active_jobs':len(live),'updated_at':C.now(),'pid':os.getpid()})
   time.sleep(.5)
 finally:
  for pid,(proc,log,*_) in live.items():
   try:os.killpg(pid,signal.SIGTERM)
   except ProcessLookupError:pass
  for pid,(proc,log,*_) in live.items():
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:
    os.killpg(pid,signal.SIGKILL);proc.wait()
   log.close()

def execute(expected,deadline):
 training,validation,_=frozen_plans(expected);cfg=expected['config']
 run_jobs('comparator',training['roots'],expected,deadline)
 run_jobs('selection',training['roots'],expected,deadline)
 run_jobs('fit',[{}],expected,deadline,workers=1)
 fit=read(R/'fit/manifest.json');frozen=freeze_validation_actions(P,expected,fit)
 run_jobs('validation',[{**r,'actions':frozen['actions'][r['root_id']]} for r in validation['roots']],expected,deadline)
 specs=build_benchmark_specs({a:C.ROOT/v['path'] for a,v in frozen['checkpoints'].items()},cfg,expected)
 C.immutable_json(P/'benchmarks/specs.json',{'identity_sha256':C.fingerprint(expected),'specs':specs})
 run_jobs('benchmark',specs,expected,deadline)
 attempts=[read(p) for p in (P/'benchmarks/cells').glob('*/attempt-[0-9][0-9][0-9].json')]
 result=summarize_benchmarks(specs,attempts,expected,expected_panel=read(C.ROOT/cfg['benchmark_manifest']),expected_checkpoints={a:v['sha256'] for a,v in frozen['checkpoints'].items()})
 C.immutable_json(P/'benchmark-summary.json',result)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=float,default=10800);parser.add_argument('--worker');parser.add_argument('--deadline',type=float);parser.add_argument('--parent',type=int);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
 torch.set_num_threads(1)
 if args.worker:return worker(read(args.worker),read(P/'identity.json'),args.deadline,args.parent)
 if not 0<args.seconds<=10800:raise ValueError('cycle bound')
 for path in (R,P,R/'jobs',R/'logs',R/'tasks',P/'jobs'):path.mkdir(parents=True,exist_ok=True)
 with ExitStack() as stack:
  for path in LOCKS:
   path.parent.mkdir(parents=True,exist_ok=True);h=stack.enter_context(path.open('a+'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
  check(time.monotonic()+1)
  if V.active_actors():raise RuntimeError('another actor present')
  expected=identity()
  if args.preflight:print('provenance preflight passed');return
  if expected['config'].get('launch_status')!='frozen':raise ValueError('full implementation not frozen; launch forbidden')
  C.immutable_json(P/'identity.json',expected)
  try:execute(expected,time.monotonic()+args.seconds);phase='complete_review_required';reason=None
  except BaseException as e:phase='review_required';reason=str(e);raise
  finally:C.atomic_json(R/'status.json',{'phase':phase,'reason':reason,'updated_at':C.now(),'pid':os.getpid()})
if __name__=='__main__':main()
